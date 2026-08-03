"""Disposable staging contract checks for upstream PR #956 (Job D).

Runs on real Apple Silicon against a real model. Each check prints PASS/FAIL and
its evidence; the script exits non-zero if any check fails, so the job is a gate.

Checks:
  1. per-request max_tokens above and below GenerationDefaults.max_tokens
     (the Copilot finding the author rejected)
  2. TokenizerWrapper.detokenizer independence
     (the Codex finding the author rejected twice)
  3. batched-vs-sequential greedy equivalence, ids + logprobs + text + reason
  4. input-order preservation under a reversed request list
  5. fast_generate(kv_bits=...) -- exposed parameter that validation always rejects
"""

import math
import sys
import traceback

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from unsloth_zoo.mlx.generate import (
    GenerationDefaults,
    GenerationRequest,
    fast_generate,
    generate_batch,
)

MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
FAILURES = []


def check(name, fn):
    try:
        detail = fn()
        print(f"PASS  {name}\n      {detail}\n", flush=True)
    except Exception as exc:
        FAILURES.append(name)
        print(f"FAIL  {name}\n      {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        print(flush=True)


def sequential_reference(model, tokenizer, prompt, max_tokens):
    """Upstream's own sequential decoding, normalised to the PR's contract."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    events = list(
        stream_generate(
            model,
            tokenizer,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
        )
    )
    body = [e for e in events if e.finish_reason != "stop"]
    ids = [int(e.token) for e in body]
    logprobs = [float(e.logprobs[int(e.token)].item()) for e in body]
    text = "".join(e.text for e in events)
    return ids, logprobs, events[-1].finish_reason, text


def main():
    model, tokenizer = load(MODEL)

    # 1. per-request max_tokens vs the defaults value ------------------------
    def per_request_max_tokens():
        defaults = GenerationDefaults(max_tokens=2)
        requests = [
            GenerationRequest(prompt="Count: one two three four five six", max_tokens=1),
            GenerationRequest(prompt="Count: one two three four five six", max_tokens=2),
            GenerationRequest(prompt="Count: one two three four five six", max_tokens=12),
        ]
        results = generate_batch(model, tokenizer, requests, defaults=defaults)
        counts = [len(r.token_ids) for r in results]
        reasons = [r.finish_reason for r in results]
        # A request asking for 12 under a default of 2 must not be capped at 2.
        assert counts[0] <= 1, f"request 0 emitted {counts[0]} tokens for max_tokens=1"
        assert counts[1] <= 2, f"request 1 emitted {counts[1]} tokens for max_tokens=2"
        assert counts[2] > 2 or reasons[2] == "stop", (
            f"request 2 asked for 12 tokens under a default of 2 and got "
            f"{counts[2]} with finish_reason={reasons[2]!r} -- the constructor "
            f"default appears to cap per-request budgets"
        )
        return f"counts={counts} reasons={reasons} (defaults.max_tokens=2)"

    check("per-request max_tokens exceeds GenerationDefaults.max_tokens", per_request_max_tokens)

    # 2. detokenizer independence --------------------------------------------
    def detokenizer_independence():
        first = getattr(tokenizer, "detokenizer", None)
        second = getattr(tokenizer, "detokenizer", None)
        assert first is not None, "tokenizer exposes no detokenizer"
        assert first is not second, (
            "tokenizer.detokenizer returned the SAME object twice; the text "
            "adapter builds one per sequence without require_independent=True, "
            "so concurrent sequences would share a buffer"
        )
        return f"{type(first).__name__}: two accesses are distinct objects"

    check("TokenizerWrapper.detokenizer is fresh per access", detokenizer_independence)

    # 3. batched vs sequential greedy equivalence ----------------------------
    def equivalence():
        prompts = [
            "The capital of France is",
            "Two plus two equals",
            "Water freezes at a temperature of",
        ]
        budgets = [8, 5, 10]
        requests = [
            GenerationRequest(prompt=p, max_tokens=n)
            for p, n in zip(prompts, budgets)
        ]
        batched = generate_batch(
            model,
            tokenizer,
            requests,
            defaults=GenerationDefaults(prefill_batch_size=2, completion_batch_size=3),
        )
        lines = []
        for index, (result, prompt, budget) in enumerate(zip(batched, prompts, budgets)):
            ids, logprobs, reason, text = sequential_reference(
                model, tokenizer, prompt, budget
            )
            assert result.token_ids == ids, (
                f"[{index}] token ids differ\n        batched={result.token_ids}\n"
                f"        sequential={ids}"
            )
            assert len(result.logprobs) == len(result.token_ids), (
                f"[{index}] {len(result.logprobs)} logprobs for "
                f"{len(result.token_ids)} tokens"
            )
            # Batching changes the matmul shapes, so a 4-bit quantised model
            # gives logprobs that agree to about one step of its own
            # arithmetic rather than bit-for-bit. Report the deviation; only
            # a difference large enough to imply a different distribution is
            # a failure.
            deviation = max(
                (abs(a - b) for a, b in zip(result.logprobs, logprobs)), default=0.0
            )
            assert deviation <= 0.06, (
                f"[{index}] logprobs deviate by {deviation}\n"
                f"        batched={result.logprobs}\n"
                f"        sequential={logprobs}"
            )
            assert result.finish_reason == reason, (
                f"[{index}] finish_reason {result.finish_reason!r} != {reason!r}"
            )
            assert result.text == text, (
                f"[{index}] text differs\n        {result.text!r}\n        {text!r}"
            )
            lines.append(
                f"[{index}] {len(ids)} ids match exactly, "
                f"max logprob deviation {deviation:.5f}, reason={reason}"
            )
        return "; ".join(lines)

    check("batched greedy == upstream sequential (ids, logprobs, text, reason)", equivalence)

    # 4. input-order preservation --------------------------------------------
    def ordering():
        prompts = ["Alpha beta", "Gamma delta epsilon", "Zeta"]
        requests = [GenerationRequest(prompt=p, max_tokens=6) for p in prompts]
        forward = generate_batch(model, tokenizer, requests)
        backward = generate_batch(model, tokenizer, list(reversed(requests)))
        assert [r.token_ids for r in backward] == [
            r.token_ids for r in reversed(forward)
        ], "reversing the request list did not reverse the results"
        return f"{len(forward)} results, order preserved under reversal"

    check("results follow input order", ordering)

    # 5. kv_bits is exposed by fast_generate but always rejected --------------
    def kv_bits_trap():
        model._tokenizer = tokenizer
        try:
            fast_generate(model, ["hello"], max_tokens=4, kv_bits=4)
        except ValueError as exc:
            return (
                f"fast_generate(kv_bits=4) raised ValueError as validation "
                f"requires: {exc}"
            )
        raise AssertionError(
            "fast_generate(kv_bits=4) did NOT raise; validation was expected to "
            "refuse every KV-quant control"
        )

    check("fast_generate exposes kv_bits but validation always refuses it", kv_bits_trap)

    print(f"\n=== {len(FAILURES)} failed ===", flush=True)
    for name in FAILURES:
        print(f"  - {name}", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
