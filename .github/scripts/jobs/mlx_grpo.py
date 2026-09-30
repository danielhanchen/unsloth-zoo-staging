"""MLX GRPO on Apple Silicon through zoo MLXGRPOTrainer (real rollouts, rewards, KL). Never imports torch.

    python jobs/mlx_grpo.py                   # Qwen2.5-0.5B-Instruct-4bit, 8 steps, runtime CCE
    python jobs/mlx_grpo.py --tiny            # SmolLM-135M-Instruct-4bit
    python jobs/mlx_grpo.py --no-cce          # dense head, for the CCE A/B
    python jobs/mlx_grpo.py --beta 0.04       # KL against the adapter-disabled reference

Records per step: loss, reward, kl, completion length, step time, rollout time; summary: peak
memory, rollout/train split, CCE-vs-dense log-prob and loss parity on one real rollout batch.
Checks: steps, finite loss, adapter changed, rewards varied within a group, kl >= 0 (beta > 0),
CCE parity (when CCE engaged).
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
TINY_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _rows(count=32):
    rows = []
    for i in range(count):
        a, b = 3 + 7 * i % 41, 2 + 5 * i % 23
        rows.append({
            "prompt": [
                {"role": "system", "content": "Answer with just the number."},
                {"role": "user", "content": f"What is {a} plus {b}?"},
            ],
            "answer": str(a + b),
        })
    return rows


def correctness(completions, answer, **kwargs):
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [2.0 if a in t else 0.0 for t, a in zip(texts, answer)]


def brevity(completions, completion_ids, **kwargs):
    return [max(0.0, 1.0 - len(ids) / 32) for ids in completion_ids]


def _peak_gpu_gb():
    import mlx.core as mx
    getter = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    try:
        return round(float(getter()) / 1024**3, 4) if getter else None
    except Exception:
        return None


def _fingerprint(model):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    total, n = 0.0, 0
    for _, v in tree_flatten(model.trainable_parameters()):
        total += float(mx.sum(mx.abs(v.astype(mx.float32))).item())
        n += int(v.size)
    return {"abs_sum": total, "numel": n}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = C.base_parser("mlx_grpo", DEFAULT_MODEL, TINY_MODEL, default_steps=8)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--no-cce", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--rank", type=int, default=16)
    a = C.resolve_args(p, argv)
    if "--lr" not in " ".join(argv):
        a.lr = 2e-5
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with C.JobRecorder(a, backend_hint=None) as rec:
        rec.data["summary"].update(preset="tiny" if a.tiny else "default", use_cce=not a.no_cce,
                                   compile=not a.no_compile, beta=a.beta)
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        import unsloth  # noqa: F401  must precede mlx_lm / transformers
        is_mlx = bool(getattr(unsloth, "_IS_MLX", False))
        rec.set_backend("unsloth-mlx" if is_mlx else "unsloth-no-mlx")
        if not rec.check("mlx_active", is_mlx, f"unsloth._IS_MLX={is_mlx}"):
            return
        rec.backend_check()

        import mlx.core as mx
        from unsloth import FastLanguageModel
        from unsloth_zoo.mlx import grpo
        from unsloth_zoo.mlx.trainer import MLXGRPOConfig, MLXGRPOTrainer

        C.seed_everything(a.seed)
        mx.random.seed(a.seed)
        t0 = time.perf_counter()
        model, tokenizer = FastLanguageModel.from_pretrained(
            a.model, max_seq_length=a.max_seq_length, load_in_4bit=True, text_only=True,
            random_state=a.seed, token=os.environ.get("HF_TOKEN") or None)
        model = FastLanguageModel.get_peft_model(
            model, r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, target_modules=TARGETS,
            use_gradient_checkpointing=False, random_state=a.seed)
        rec.summary(load_s=round(time.perf_counter() - t0, 2))
        before = _fingerprint(model)

        # Time the rollouts separately from the update.
        rollout = {"s": 0.0, "calls": 0, "last": 0.0}
        generate = grpo.generate_batch

        def timed_generate(*args, **kwargs):
            start = time.perf_counter()
            try:
                return generate(*args, **kwargs)
            finally:
                rollout["last"] = time.perf_counter() - start
                rollout["s"] += rollout["last"]
                rollout["calls"] += 1

        grpo.generate_batch = timed_generate
        scores = []

        def correctness_logged(completions, answer, **kwargs):
            values = correctness(completions, answer)
            scores.append(values)
            return values

        correctness_logged.__name__ = "correctness"
        group = a.num_generations
        config = MLXGRPOConfig(
            max_steps=a.max_steps, learning_rate=a.lr, warmup_steps=0, logging_steps=1,
            per_device_train_batch_size=group, gradient_accumulation_steps=1,
            num_generations=group, max_completion_length=a.max_completion_length,
            temperature=a.temperature, beta=a.beta, seed=a.seed, max_seq_length=a.max_seq_length,
            use_cce=not a.no_cce, compile=not a.no_compile, gradient_checkpointing=False,
            output_dir=str(Path(a.out).with_suffix("")), save_steps=0, report_to="none",
        )
        trainer = MLXGRPOTrainer(model, tokenizer, _rows(), [correctness_logged, brevity], args=config)
        last = {"elapsed": 0.0}

        def on_step(step, total, loss, lr, tok_s, peak_gb, elapsed, num_tokens, grad_norm=None, *_, **__):
            dt = float(elapsed) - last["elapsed"]
            last["elapsed"] = float(elapsed)
            logs = next((e for e in reversed(trainer.state.log_history)
                         if e.get("step") == step - 0 and "loss" in e), {})
            rec.step(step=int(step), loss=float(loss), grad_norm=grad_norm, learning_rate=float(lr),
                     time_ms=round(dt * 1000, 2), tokens=None, tokens_per_s=None,
                     peak_mem_gb=float(peak_gb) if peak_gb is not None else None,
                     rollout_ms=round(rollout["last"] * 1000, 2),
                     reward=logs.get("reward"), kl=logs.get("kl"),
                     completion_length=logs.get("completions/mean_length"))

        trainer.add_step_callback(on_step)
        t1 = time.perf_counter()
        result = trainer.train() or {}
        train_s = time.perf_counter() - t1
        logged = [e for e in trainer.state.log_history if "loss" in e]
        for record, entry in zip(rec.data["steps"], logged):
            record.update(reward=entry.get("reward"), kl=entry.get("kl"),
                          completion_length=entry.get("completions/mean_length"))
        rec.summary(train_s=round(train_s, 2), rollout_s=round(rollout["s"], 2),
                    rollout_share=round(rollout["s"] / train_s, 3) if train_s else None,
                    step_s_mean=round(train_s / max(len(logged), 1), 3),
                    peak_mem_gb=_peak_gpu_gb(), train_steps=result.get("train_steps"),
                    rewards=[e.get("reward") for e in logged],
                    kls=[e.get("kl") for e in logged],
                    lengths=[e.get("completions/mean_length") for e in logged],
                    cce_engaged=bool(getattr(trainer, "_rollout_cce_compaction", False)))

        losses = [e["loss"] for e in logged]
        rec.check("completed_steps", len(logged) == a.max_steps, f"{len(logged)} of {a.max_steps}")
        rec.check("finite_loss", bool(losses) and all(math.isfinite(x) for x in losses), losses)
        rec.check("rewards_varied", any(len(set(v)) > 1 for v in scores), scores[:3])
        if a.beta:
            kls = [e.get("kl") for e in logged]
            rec.check("kl_nonnegative", all(k is not None and k >= -1e-5 for k in kls), kls)
        rec.check("adapter_changed", *C.adapter_changed(before, _fingerprint(model)))

        # CCE vs dense on one real rollout from the trained policy.
        scorer = grpo.make_grpo_scorer(model, a.temperature)
        if scorer is not None:
            batch, lengths, advantages, rewards, *rest = next(trainer._rollout_batches(10_000))
            mask = grpo._response_mask(batch[:, 1:], lengths)
            indices = grpo.completion_indices(lengths.tolist(), batch.shape[1])
            dense = grpo._token_logps(model, batch, a.temperature) * mask
            compact = grpo._token_logps(model, batch, a.temperature, mask, scorer, indices) * mask
            diff = float(mx.abs(dense - compact).max().item())
            options = dict(beta=0.0, epsilon_low=0.2, epsilon_high=0.2, temperature=a.temperature)
            l_dense = float(grpo.make_grpo_loss_fn(**options)(model, batch, lengths, advantages, rewards)[0].item())
            l_cce = float(grpo.make_grpo_loss_fn(scorer=scorer, **options)(
                model, batch, lengths, advantages, rewards, indices)[0].item())
            rec.summary(cce_logp_max_abs_diff=diff, loss_dense=l_dense, loss_cce=l_cce,
                        completion_tokens=int(mask.sum().item()))
            rec.check("cce_logps_match_dense", diff < 5e-2, diff)
        else:
            rec.summary(cce_available=False)


if __name__ == "__main__":
    main()
