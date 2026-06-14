"""Metal benchmark: global-norm vs per-leaf-norm vs value gradient clipping.

Measures, on real Apple Silicon, the three things that decide whether the
per-leaf default is the right call:

  1. Peak memory  - the stated rationale for not using global max_grad_norm.
  2. Step time    - the cross-tree reduction / sync-barrier cost.
  3. Training divergence - pre-clip global grad norm (does clipping bind?)
     and the resulting loss trajectory, so we can see whether per-leaf at 1.0
     trains like global at 1.0 or like a much weaker clip.

One (model, mode) per process: peak memory is process-global and sticky.
Usage: python bench_grad_clip_metal.py --model <repo> --mode <global_norm|leaf_norm|value> [--steps N] [--lr F]
"""

import argparse
import json
import time

import mlx.core as mx

from unsloth_zoo.mlx.loader import FastMLXModel
from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig


def _dataset(n=64, repeat=40):
    # Long-ish rows with structure so gradients are non-trivial and clipping
    # actually has something to bind on. Distinct rows so it is not pure memo.
    body = "unsloth gradient clipping parity benchmark on apple silicon metal"
    return [{"text": f"<<ROW {i}>> " + " ".join([body] * (repeat + i % 5))}
            for i in range(n)]


def run(model_name, mode, steps, lr):
    model, tok = FastMLXModel.from_pretrained(model_name, max_seq_length=256)
    model = FastMLXModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0)

    cfg = dict(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=steps,
        warmup_steps=0,
        learning_rate=lr,
        lr_scheduler_type="constant",
        optim="adamw",
        weight_decay=0.0,
        logging_steps=1,
        output_dir=f"/tmp/bench_{mode}",
        seed=3407,
        report_to="none",
    )
    # _resolve_mlx_grad_clipping priority: value > leaf_norm > global_norm.
    # Leave the other two at their None/0 defaults so the intended mode wins.
    if mode == "global_norm":
        cfg["max_grad_norm"] = 1.0
    elif mode == "leaf_norm":
        cfg["max_grad_leaf_norm"] = 1.0
    elif mode == "value":
        cfg["max_grad_value"] = 1.0
    else:
        raise SystemExit(f"unknown mode {mode}")

    trainer = MLXTrainer(
        model=model, tokenizer=tok, train_dataset=_dataset(),
        args=MLXTrainingConfig(**cfg),
    )
    # Reset peak memory after a 2-step warmup so the reported peak is the
    # steady-state training peak, not one-time compile/alloc.
    trainer._benchmark_reset_peak_after_step = 2

    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    peaks = trainer._peak_memory_history
    steady_peak = max(peaks[2:]) if len(peaks) > 2 else (max(peaks) if peaks else 0.0)
    losses = [round(float(x), 4) for x in trainer._train_loss_history]
    gnorms = [round(float(x), 4) for x in trainer._grad_norm_history]

    return {
        "model": model_name,
        "mode": mode,
        "steps": steps,
        "lr": lr,
        "steady_peak_gb": round(steady_peak, 3),
        "wall_s": round(wall, 2),
        "sec_per_step": round(wall / steps, 3),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "losses": losses,
        # Pre-clip GLOBAL grad norm per step is only computed in global_norm
        # mode (the true binding signal). Empty for leaf/value.
        "preclip_global_grad_norms": gnorms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["global_norm", "leaf_norm", "value"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    result = run(args.model, args.mode, args.steps, args.lr)
    print("BENCH_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
