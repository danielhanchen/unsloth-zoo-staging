"""Disposable staging batching sanity check for upstream PR #956 (Job F).

Hosted macos-14 is a 3-core M1: these numbers are NOT a reproduction of the
PR's own benchmark table, which was measured on the author's machine with much
larger models. This only answers "does batching help at all, and what does it
cost in memory" on the smallest supported model.
"""

import sys
import time

import mlx.core as mx
from mlx_lm import load

from unsloth_zoo.mlx.generate import (
    GenerationDefaults,
    GenerationRequest,
    generate_batch,
)

MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
MAX_TOKENS = 64
PROMPT = "Write a short paragraph about the sea, in plain language:"


def run(model, tokenizer, batch):
    requests = [
        GenerationRequest(prompt=PROMPT, max_tokens=MAX_TOKENS) for _ in range(batch)
    ]
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    results = generate_batch(
        model, tokenizer, requests, defaults=GenerationDefaults(max_tokens=MAX_TOKENS)
    )
    elapsed = time.perf_counter() - start
    tokens = sum(len(r.token_ids) for r in results)
    peak_gb = mx.get_peak_memory() / 1024**3
    return tokens / elapsed if elapsed else 0.0, peak_gb, tokens


def main():
    model, tokenizer = load(MODEL)
    # warm up so the first measured point does not pay compilation
    run(model, tokenizer, 1)

    print(f"model={MODEL} max_tokens={MAX_TOKENS} (hosted macos-14, 3-core M1)")
    print(f"{'batch':>6} {'tok/s':>10} {'peak GB':>10} {'tokens':>8} {'speedup':>9}")
    baseline = None
    for batch in (1, 4, 8):
        rate, peak, tokens = run(model, tokenizer, batch)
        if baseline is None:
            baseline = rate
        speedup = rate / baseline if baseline else 0.0
        print(f"{batch:>6} {rate:>10.2f} {peak:>10.2f} {tokens:>8} {speedup:>8.2f}x",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
