"""Real-model MLX training comparison for unsloth-zoo PR 962, on Apple Silicon.

Trains the same small causal LM twice from the same seed and the same data order,
once with stock mlx AdamW and once with the PR's QuantizedMomentAdamW, and reports
per-step loss and pre-clip gradient norm plus optimizer-state bytes.

The PR's own numeric evidence is 8 steps on a toy MSE model converging toward 1e-9,
where any two optimizers agree. This runs long enough on real transformer gradients
for a difference to be visible.

Run: python ci_probes/pr962_metal_e2e.py --steps 200
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from unsloth_zoo.mlx.optimizers_quantized import QuantizedMomentAdamW

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"


def build_batches(tokenizer, n_batches, batch_size, seq_len, seed):
    from datasets import load_dataset

    rows = load_dataset("unsloth/alpaca-cleaned", split=f"train[:{n_batches * batch_size * 2}]")
    texts = [
        f"### Instruction:\n{r['instruction']}\n\n### Input:\n{r['input']}\n\n### Response:\n{r['output']}"
        for r in rows
    ]
    ids = []
    for t in texts:
        enc = tokenizer.encode(t)
        if len(enc) < 16:
            continue
        enc = (enc + [tokenizer.eos_token_id or 0] * seq_len)[:seq_len]
        ids.append(enc)
        if len(ids) >= n_batches * batch_size:
            break
    while len(ids) < n_batches * batch_size:
        ids.append(ids[len(ids) % max(1, len(ids))])
    arr = mx.array(ids[: n_batches * batch_size])
    return arr.reshape(n_batches, batch_size, seq_len)


def loss_fn(model, batch):
    logits = model(batch[:, :-1]).astype(mx.float32)
    targets = batch[:, 1:]
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def grad_norm(grads):
    total = mx.zeros((), dtype=mx.float32)
    for _, g in tree_flatten(grads):
        total = total + mx.sum(g.astype(mx.float32) ** 2)
    return mx.sqrt(total)


def state_bytes(state):
    return sum(v.nbytes for _, v in tree_flatten(state) if isinstance(v, mx.array))


def make_model(model_name, rank, seed):
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    mx.random.seed(seed)
    model, tokenizer = load(model_name)
    model.freeze()
    n_layers = len(model.layers) if hasattr(model, "layers") else 8
    linear_to_lora_layers(model, min(n_layers, 8), {"rank": rank, "scale": 20.0, "dropout": 0.0})
    model.train()
    mx.eval(model.parameters())
    return model, tokenizer


def run_arm(label, make_opt, args, batches):
    model, _ = make_model(args.model, args.rank, args.seed)
    schedule = optim.cosine_decay(args.lr, args.steps)
    opt = make_opt(args.lr)
    gf = nn.value_and_grad(model, loss_fn)

    losses, norms = [], []
    t0 = time.time()
    for step in range(args.steps):
        opt.learning_rate = schedule(mx.array(step))
        loss, grads = gf(model, batches[step % batches.shape[0]])
        gn = grad_norm(grads)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss, gn)
        losses.append(float(loss))
        norms.append(float(gn))
        if step % 20 == 0 or step == args.steps - 1:
            print(f"  [{label}] step {step:4d}  loss {losses[-1]:.5f}  |g| {norms[-1]:.4f}",
                  flush=True)
    elapsed = time.time() - t0
    return {
        "label": label,
        "losses": losses,
        "grad_norms": norms,
        "state_bytes": state_bytes(opt.state),
        "seconds": elapsed,
        "tokens_per_s": args.steps * batches.shape[1] * batches.shape[2] / elapsed,
    }


def mape(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if y != 0 and math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return float("nan")
    return sum(abs(x - y) / abs(y) for x, y in pairs) / len(pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--n-batches", type=int, default=64)
    args = p.parse_args()

    print(f"mlx default device: {mx.default_device()}", flush=True)
    print(f"model {args.model}  steps {args.steps}  batch {args.batch_size}  seq {args.seq_len}",
          flush=True)

    from mlx_lm import load

    _, tokenizer = load(args.model)
    batches = build_batches(tokenizer, args.n_batches, args.batch_size, args.seq_len, args.seed)
    mx.eval(batches)

    arms = [
        ("adamw", lambda lr: optim.AdamW(learning_rate=lr, weight_decay=0.0, bias_correction=True)),
        ("adamw_8bit", lambda lr: QuantizedMomentAdamW(lr, weight_decay=0.0, bias_correction=True)),
    ]
    results = [run_arm(label, make, args, batches) for label, make in arms]

    ref, got = results
    tail = max(1, args.steps // 4)
    loss_mape = mape(got["losses"][-tail:], ref["losses"][-tail:])
    norm_mape = mape(got["grad_norms"][-tail:], ref["grad_norms"][-tail:])

    print("\n| Method     | Tokens/s | State bytes | Losses [1,2,3,n-1,n] | Grad-norms [1,2,3,n-1,n] |")
    print("|------------|----------|-------------|----------------------|--------------------------|")
    for r in results:
        pick = lambda xs: [round(xs[i], 3) for i in (0, 1, 2, -2, -1)]
        print(f"| {r['label']:<10} | {r['tokens_per_s']:8.1f} | {r['state_bytes']:11,} | "
              f"{pick(r['losses'])} | {pick(r['grad_norms'])} |")

    print(f"\nloss MAPE over last {tail} steps: {loss_mape:.4%}")
    print(f"grad-norm MAPE over last {tail} steps: {norm_mape:.4%}")
    print(f"final loss {ref['losses'][-1]:.5f} -> {got['losses'][-1]:.5f} "
          f"({(got['losses'][-1] / ref['losses'][-1] - 1):+.2%})")
    print(f"optimizer state {ref['state_bytes']:,} -> {got['state_bytes']:,} "
          f"({1 - got['state_bytes'] / ref['state_bytes']:+.2%})")

    with open("pr962_e2e_results.json", "w") as f:
        json.dump({"args": vars(args), "results": results,
                   "loss_mape": loss_mape, "norm_mape": norm_mape}, f, indent=2)

    bad = [r for r in results if any(not math.isfinite(x) for x in r["losses"])]
    if bad:
        print(f"RESULT: FAIL non-finite loss in {[r['label'] for r in bad]}")
        return 1
    print("RESULT: PASS both arms trained to completion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
