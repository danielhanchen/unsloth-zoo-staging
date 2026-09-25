"""MLX LoRA-on-lm_head CCE probe (unsloth-zoo #1362) on Apple Silicon Metal.

Random-init Llama (2 layers, hidden 1024, untied vocab 65536), LoRA r=16 on q/k/v/o/gate/up/down
+ lm_head, MLXTrainer compile=True, use_cce=True, gradient checkpointing, AdamW, 2 x 1024 tokens.
Each arm runs in its own process (clean peak memory):
  main      LoRA-head CCE disabled (_supports_text_lora_cce -> False), i.e. main's fallback path
  head      the PR as shipped (recompute decided by the working-set budget)
  retain    head, backward keeps the compiled chunk logits
  recompute head, backward reprojects every chunk
--quant repeats with a 4-bit model. Checks: LoRA-head loss traced on head arms and not on main,
per-step loss / grad-norm parity vs main, head peak memory below main, finite loss.

    python jobs/mlx_lora_head_cce.py [--quant] [--max-steps 5]
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

ARMS = ("main", "head", "retain", "recompute")
VOCAB, HIDDEN, SEQ, BATCH = 65536, 1024, 1024, 2


def _arm(arm, quant, steps, seed, out):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import llama
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from unsloth_zoo.mlx import utils as mu
    from unsloth_zoo.mlx.cce import runtime_cce as rc
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    traced, base_traced = [], []
    make_base = mu.make_baseline_loss_fn

    def counting_base(*a, **k):
        fn = make_base(*a, **k)

        def wrapped(model, *b, **kw):
            base_traced.append(len(b))
            return fn(model, *b, **kw)
        return wrapped
    mu.make_baseline_loss_fn = counting_base
    if arm == "main":
        mu._supports_text_lora_cce = lambda *a, **k: False
    else:
        factory = mu._make_text_lora_cce_loss_fn

        def recording(*a, **k):
            fn = factory(*a, **k)

            def wrapped(model, *b):
                traced.append(len(b))
                return fn(model, *b)
            return wrapped
        mu._make_text_lora_cce_loss_fn = recording
        if arm == "retain":
            rc._RECOMPUTE_LOGITS_BYTES = 1 << 50
        elif arm == "recompute":
            rc._RECOMPUTE_LOGITS_BYTES = 1

    mx.random.seed(seed)
    args = llama.ModelArgs(model_type="llama", hidden_size=HIDDEN, num_hidden_layers=2,
                           intermediate_size=2816, num_attention_heads=16, num_key_value_heads=4,
                           rms_norm_eps=1e-5, vocab_size=VOCAB, head_dim=64,
                           tie_word_embeddings=False)
    model = llama.Model(args)
    model.set_dtype(mx.bfloat16)
    if quant:
        nn.quantize(model, group_size=64, bits=4)
    model.freeze()
    linear_to_lora_layers(model, 2, {"rank": 16, "scale": 2.0, "dropout": 0.0, "keys": [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]})
    model.lm_head = LoRALinear.from_base(model.lm_head, r=16, dropout=0.0, scale=2.0)
    # Non-zero B so the head adapter moves the logits from step 1.
    model.lm_head.lora_b = mx.random.normal(model.lm_head.lora_b.shape) * 0.01
    mx.eval(model.parameters())

    import numpy as np
    rng = np.random.default_rng(seed)
    rows = [mu._FiniteTextRow(tuple(int(x) for x in rng.integers(0, VOCAB, SEQ + 1)), offset=0)
            for _ in range(steps * BATCH)]
    schedule = [tuple(range(i * BATCH, (i + 1) * BATCH)) for i in range(steps)]
    cfg = MLXTrainingConfig(max_steps=steps, per_device_train_batch_size=BATCH,
                            gradient_accumulation_steps=1, learning_rate=1e-4, logging_steps=1,
                            save_steps=0, output_dir=str(Path(out).with_suffix("")), compile=True,
                            use_cce=True, gradient_checkpointing=True, report_to="none",
                            optim="adamw", max_grad_norm=1.0, report_grad_norm=True, seed=seed)
    trainer = MLXTrainer(model=model, tokenizer=None, train_dataset=[], args=cfg)
    trainer._batches = mu.FiniteTextBatchPlan(rows, schedule, max_seq_length=SEQ + 1, pad_id=0)
    trainer.save_model = lambda output_dir=None: None
    steps_log, last = [], {"t": 0.0}

    def on_step(step, total, loss, lr, tok_s, peak_gb, elapsed, num_tokens, grad_norm=None, *_, **__):
        dt = float(elapsed) - last["t"]
        last["t"] = float(elapsed)
        steps_log.append({"step": int(step), "loss": float(loss),
                          "grad_norm": None if grad_norm is None else float(grad_norm),
                          "time_ms": round(dt * 1000, 2), "tokens_per_s": float(tok_s or 0),
                          "peak_mem_gb": float(peak_gb or 0)})

    trainer.add_step_callback(on_step)
    mx.clear_cache()
    mx.reset_peak_memory()
    trainer.train()
    peak = mx.get_peak_memory() / 1e6
    budget = rc._RECOMPUTE_LOGITS_BYTES
    json.dump({"arm": arm, "quant": quant, "steps": steps_log, "peak_mb": round(peak, 1),
               "traced": traced, "base_traced": base_traced, "recompute_budget": budget,
               "recompute": BATCH * SEQ * VOCAB * 2 > budget if budget else None,
               "working_set": mx.device_info().get("max_recommended_working_set_size")},
              open(out, "w"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--child":
        _, arm, quant, steps, seed, out = argv
        return _arm(arm, quant == "1", int(steps), int(seed), out)
    p = C.base_parser("mlx_lora_head_cce", "random llama 1024x128256", "same", default_steps=5)
    p.add_argument("--quant", action="store_true", help="4-bit model instead of bf16")
    a = C.resolve_args(p, argv)
    a.model = f"random-llama-h{HIDDEN}-v{VOCAB}" + ("-4bit" if a.quant else "-bf16")
    with C.JobRecorder(a, backend_hint=None) as rec:
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"Metal needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        rec.set_backend("unsloth-zoo-mlx")
        outdir = Path(a.out).with_suffix("")
        outdir.mkdir(parents=True, exist_ok=True)
        res = {}
        for arm in ARMS:
            out = outdir / f"{arm}.json"
            t = time.perf_counter()
            r = subprocess.run([sys.executable, __file__, "--child", arm, "1" if a.quant else "0",
                                str(a.max_steps), str(a.seed), str(out)], capture_output=True, text=True)
            print(f"== arm {arm} rc={r.returncode} {time.perf_counter() - t:.1f}s", flush=True)
            print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
            if not rec.check(f"{arm}_ran", r.returncode == 0 and out.exists(), r.stderr[-800:]):
                continue
            res[arm] = json.load(open(out))
        for arm, d in res.items():
            for s in d["steps"]:
                rec.step(arm=arm, **s)
        rec.summary(arms={k: {kk: v[kk] for kk in ("peak_mb", "traced", "base_traced", "recompute", "recompute_budget",
                                                   "working_set")} | {
                    "losses": [round(s["loss"], 4) for s in v["steps"]],
                    "grad_norms": [None if s["grad_norm"] is None else round(s["grad_norm"], 4)
                                   for s in v["steps"]],
                    "steady_step_ms": sorted(s["time_ms"] for s in v["steps"][1:])[len(v["steps"][1:]) // 2]
                    if len(v["steps"]) > 1 else None} for k, v in res.items()})
        if "main" not in res:
            return
        base = res["main"]
        rec.check("main_never_traces_lora_cce", base["traced"] == [] and len(base["base_traced"]) > 0,
                  f"lora {base['traced']} baseline {base['base_traced']}")
        for arm in ("head", "retain", "recompute"):
            if arm not in res:
                continue
            d = res[arm]
            # Same number of compiled-step traces as main's baseline loss, never the baseline itself.
            rec.check(f"{arm}_lora_cce_traced_like_main",
                      len(d["traced"]) == len(base["base_traced"]) and d["base_traced"] == [],
                      f"lora {d['traced']} baseline {d['base_traced']} main {base['base_traced']}")
            ls = [s["loss"] for s in d["steps"]]
            bl = [s["loss"] for s in base["steps"]]
            rec.check(f"{arm}_finite_loss", len(ls) == a.max_steps and all(math.isfinite(x) for x in ls), ls)
            rec.check(f"{arm}_loss_parity", len(ls) == len(bl) and all(
                abs(x - y) <= 1e-2 + 1e-2 * abs(y) for x, y in zip(ls, bl)), f"{ls} vs main {bl}")
            gn = [s["grad_norm"] for s in d["steps"]]
            bg = [s["grad_norm"] for s in base["steps"]]
            rec.check(f"{arm}_grad_norm_parity", None not in gn + bg and all(
                abs(x - y) <= 1e-2 + 0.05 * abs(y) for x, y in zip(gn, bg)), f"{gn} vs main {bg}")
            rec.check(f"{arm}_peak_below_main", d["peak_mb"] < base["peak_mb"],
                      f"{d['peak_mb']} MB vs main {base['peak_mb']} MB")


if __name__ == "__main__":
    main()
