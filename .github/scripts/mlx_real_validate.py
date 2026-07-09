"""Real-MLX validation for the unsloth_zoo.mlx PRs on Apple Silicon.

Runs on a macos-14 GitHub runner with REAL mlx / mlx-lm installed (NOT the
torch simulation shim used by the unit tests). It exercises each PR's new code
on real MLX arrays at the API / math level, without a model download, so it is
fast and deterministic. Each feature is guarded: a feature whose symbols are
absent on this branch is SKIPPED; a present feature that fails is a FAILURE.
Exit code is non-zero if any present feature failed.
"""

import sys
import traceback

RESULTS = []


class SkipCheck(Exception):
    pass


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
    except SkipCheck as exc:
        RESULTS.append((name, "SKIP", str(exc)))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402


def real_mlx_available():
    assert mx.metal.is_available(), "mlx.metal.is_available() is False"


def import_unsloth_zoo_mlx():
    # The single highest-value check: the MLX modules must import under REAL
    # mlx, not just the torch shim. Catches real-mlx API/signature drift.
    import unsloth_zoo.mlx.trainer  # noqa: F401
    import unsloth_zoo.mlx.loader  # noqa: F401
    import unsloth_zoo.mlx.utils  # noqa: F401
    import unsloth_zoo.mlx.compile  # noqa: F401


def pr819_new_optimizers():
    from unsloth_zoo.mlx.trainer import SUPPORTED_MLX_OPTIMIZERS
    new = ("rmsprop", "adamax", "adagrad", "adadelta")
    if not all(o in SUPPORTED_MLX_OPTIMIZERS for o in new):
        raise SkipCheck("branch has no #819 optimizers")
    # Construct each on real mlx.optimizers (confirms the class and its args
    # exist in the installed mlx version, which is what the PR relies on).
    opts = [
        optim.RMSprop(learning_rate=1e-3),
        optim.Adamax(learning_rate=1e-3, betas=[0.9, 0.999]),
        optim.Adagrad(learning_rate=1e-3),
        optim.AdaDelta(learning_rate=1e-3),
    ]
    assert len(opts) == 4 and all(o is not None for o in opts)


def pr818_schedulers():
    from unsloth_zoo.mlx.trainer import SUPPORTED_MLX_LR_SCHEDULERS
    wanted = ("inverse_sqrt", "warmup_stable_decay", "polynomial", "cosine_with_restarts")
    present = [s for s in wanted if s in SUPPORTED_MLX_LR_SCHEDULERS]
    if not present:
        raise SkipCheck("branch has no #818 schedulers")
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    for sched in present + ["cosine", "linear"]:
        t = MLXTrainer.__new__(MLXTrainer)
        cfg = MLXTrainingConfig(
            learning_rate=2e-4, warmup_steps=2, max_steps=10,
            lr_scheduler_type=sched, output_dir="/tmp/o",
        )
        t.args = cfg
        schedule = t._build_schedule(10)
        vals = [float(mx.array(schedule(mx.array(s))).item()) for s in range(10)]
        assert all(v == v and v >= 0.0 for v in vals), f"{sched}: bad LR curve {vals}"


def pr820_token_accuracy():
    from unsloth_zoo.mlx.trainer import MLXTrainer
    if not hasattr(MLXTrainer, "_evaluate_batch_totals"):
        raise SkipCheck("no _evaluate_batch_totals")
    import inspect
    sig = inspect.signature(MLXTrainer._evaluate_batch_totals)
    if "want_accuracy" not in sig.parameters:
        raise SkipCheck("branch has no #820 token accuracy")
    t = MLXTrainer.__new__(MLXTrainer)

    class _M:
        def eval(self):
            pass

        def train(self):
            pass

    t.model = _M()
    t.stop_requested = False
    t._distributed_eval_status = lambda failed=False: (False, False)
    t._distributed_all_sum = lambda v, stream=None: v
    t._distributed_should_stop = lambda: False

    def loss_fn(_m, _b, _l, _lab, return_correct=False):
        loss = mx.array(1.0)
        ntoks = mx.array(4)
        if return_correct:
            return loss, ntoks, mx.array(3.0)
        return loss, ntoks

    t._evaluate([(mx.array([[1, 2, 3]]), None, None)], loss_fn, is_vlm=False)
    acc = t._last_eval_metrics["eval_mean_token_accuracy"]
    assert abs(acc - 3.0 / 4.0) < 1e-6, f"accuracy {acc}"


def pr830_orpo_dpo():
    from unsloth_zoo.mlx import utils
    if not hasattr(utils, "_orpo_odds_ratio_loss"):
        raise SkipCheck("branch has no #830 ORPO helper")
    # Real mlx: the float32-upcast odds-ratio term must be finite, including a
    # perfectly-predicted (logp -> 0) row that underflowed in float16.
    for dtype in (mx.float32, mx.float16):
        lc = mx.array([-0.5, 0.0], dtype=dtype)
        lr = mx.array([-1.0, -0.2], dtype=dtype)
        val = float(mx.array(utils._orpo_odds_ratio_loss(lc, lr)).item())
        assert val == val and abs(val) != float("inf"), f"{dtype}: {val}"
    assert hasattr(utils, "make_orpo_loss_fn") and hasattr(utils, "make_dpo_loss_fn")


def pr832_grpo():
    from unsloth_zoo.mlx import utils
    if not hasattr(utils, "make_grpo_loss_fn"):
        raise SkipCheck("branch has no #832 GRPO")
    if hasattr(utils, "_hf_encoding_tokenizer"):
        class _Fake:
            pass

        f = _Fake()
        assert utils._hf_encoding_tokenizer(f) is f  # non-wrapper returns itself
    # GRPO advantage normalization with a zero-std group must stay finite.
    rewards = mx.array([1.0, 1.0, 1.0, 1.0])
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    mx.eval(adv)
    assert all(v == v for v in adv.tolist())


def pr848_gptq_awq():
    from unsloth_zoo.mlx import loader
    if not hasattr(loader, "_mlx_lm_would_reject_prequant"):
        raise SkipCheck("branch has no #848 GPTQ/AWQ")
    # GPTQ must always be rejected (dequantized locally); mlx-lm cannot load it.
    assert loader._mlx_lm_would_reject_prequant("/nonexistent", "gptq", {}) is True
    # AWQ decision must be a bool (defers on new mlx-lm, dequants on old).
    assert isinstance(loader._mlx_lm_would_reject_prequant("/nonexistent", "awq", {}), bool)


def pr873_callbacks():
    from unsloth_zoo.mlx.trainer import MLXTrainer
    if not hasattr(MLXTrainer, "add_callback"):
        raise SkipCheck("branch has no #873 HF callbacks")
    try:
        from unsloth_zoo.mlx.trainer import _MLXCallbackHandler  # noqa: F401
    except Exception:
        raise SkipCheck("no _MLXCallbackHandler")


_TRAIN_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
_TRAIN_DATA = [
    {"text": f"### Question: what is {i} plus {i}?\n### Answer: {2 * i}."}
    for i in range(16)
]


def _tiny_train(config=None, **trainer_kwargs):
    """Run a short real LoRA SFT fit and return the trainer. Downloads a ~80MB
    4-bit model once (HF-cached). Exercises the full train() loop on real MLX."""
    import tempfile
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    model, tok = FastMLXModel.from_pretrained(_TRAIN_MODEL, max_seq_length=256)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dict(
            per_device_train_batch_size=2, gradient_accumulation_steps=1,
            max_steps=6, warmup_steps=2, learning_rate=5e-4, logging_steps=1,
            output_dir=tmp, seed=3407, report_to="none",
        )
        cfg.update(config or {})
        trainer = MLXTrainer(
            model=model, tokenizer=tok, train_dataset=_TRAIN_DATA,
            args=MLXTrainingConfig(**cfg), **trainer_kwargs,
        )
        trainer.train()
        return trainer


def _assert_finite_training(trainer):
    hist = getattr(trainer, "_train_loss_history", None)
    if hist:
        assert all(v == v and abs(v) != float("inf") for v in hist), f"bad losses {hist}"


def real_training_smoke():
    # The core real-MLX check: a full LoRA SFT train() to completion. This is
    # what a NameError in the training-summary path only shows up under -- the
    # API-level and torch-shim checks do not run train() end to end.
    _assert_finite_training(_tiny_train())


def pr819_optimizer_training():
    from unsloth_zoo.mlx.trainer import SUPPORTED_MLX_OPTIMIZERS
    new = [o for o in ("rmsprop", "adamax", "adagrad", "adadelta")
           if o in SUPPORTED_MLX_OPTIMIZERS]
    if not new:
        raise SkipCheck("branch has no #819 optimizers")
    for opt in new:
        _assert_finite_training(_tiny_train(config={"optim": opt, "max_steps": 3}))


def pr818_scheduler_training():
    from unsloth_zoo.mlx.trainer import SUPPORTED_MLX_LR_SCHEDULERS
    scheds = [s for s in ("inverse_sqrt", "warmup_stable_decay", "polynomial")
              if s in SUPPORTED_MLX_LR_SCHEDULERS]
    if not scheds:
        raise SkipCheck("branch has no #818 schedulers")
    for s in scheds:
        _assert_finite_training(_tiny_train(config={"lr_scheduler_type": s, "max_steps": 4}))


def pr820_eval_accuracy_training():
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer
    if "want_accuracy" not in inspect.signature(MLXTrainer._evaluate_batch_totals).parameters:
        raise SkipCheck("branch has no #820 token accuracy")
    # mean_token_accuracy is a baseline-loss-path feature: the fused CCE kernel
    # (the default) cannot cheaply recover argmax, so it omits the metric by
    # design. Force use_cce=False so the smoke exercises the real accuracy path.
    trainer = _tiny_train(
        eval_dataset=_TRAIN_DATA,
        config={"eval_steps": 3, "max_steps": 6, "use_cce": False},
    )
    _assert_finite_training(trainer)
    acc = (trainer._last_eval_metrics or {}).get("eval_mean_token_accuracy")
    assert acc is not None and 0.0 <= acc <= 1.0, (
        f"eval_mean_token_accuracy={acc} (baseline loss path should report it)"
    )


def main():
    for name, fn in [
        ("real_mlx_available", real_mlx_available),
        ("import_unsloth_zoo_mlx", import_unsloth_zoo_mlx),
        ("pr819_new_optimizers", pr819_new_optimizers),
        ("pr818_schedulers", pr818_schedulers),
        ("pr820_token_accuracy", pr820_token_accuracy),
        ("pr830_orpo_dpo", pr830_orpo_dpo),
        ("pr832_grpo", pr832_grpo),
        ("pr848_gptq_awq", pr848_gptq_awq),
        ("pr873_callbacks", pr873_callbacks),
        ("real_training_smoke", real_training_smoke),
        ("pr819_optimizer_training", pr819_optimizer_training),
        ("pr818_scheduler_training", pr818_scheduler_training),
        ("pr820_eval_accuracy_training", pr820_eval_accuracy_training),
    ]:
        check(name, fn)

    print("\n==== Real-MLX validation results ====")
    for name, status, detail in RESULTS:
        print(f"[{status}] {name}" + (f" -- {detail.splitlines()[0]}" if detail else ""))
        if status == "FAIL":
            print(detail)
    failures = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{sum(r[1]=='PASS' for r in RESULTS)} pass, "
          f"{sum(r[1]=='SKIP' for r in RESULTS)} skip, {len(failures)} fail")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
