# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""PR 1191 on a real mlx-community checkpoint: same tokens, kernel actually used.

`fused_decode_conv_silu` has no production call site on that branch, so this
enters the scope explicitly. It asserts the Metal kernel really fires during
single-token decode (and never during the prefill of an ineligible model), that
greedy text is unchanged, and that every instance class is restored on exit.
"""

import argparse
import json
import sys

from mlx_lm import load
from mlx_lm.generate import generate

from unsloth_zoo.mlx import inference

PROMPT = "List three prime numbers."
MAX_TOKENS = 24


def _greedy(model, tokenizer):
    return generate(model, tokenizer, prompt = PROMPT, max_tokens = MAX_TOKENS, verbose = False)


def _count_kernel(fn):
    """Run `fn` with the fused conv+SiLU kernel entry point counted."""
    calls = [0]
    original = inference._decode_conv_silu

    def counting(x, weight):
        calls[0] += 1
        return original(x, weight)

    inference._decode_conv_silu = counting
    try:
        result = fn()
    finally:
        inference._decode_conv_silu = original
    return result, calls[0]


def _patched_count(model, classes):
    return sum(1 for _, module in model.named_modules() if type(module) in classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True)
    parser.add_argument("--revision", default = None)
    parser.add_argument("--expect-recurrent", action = "store_true",
                        help = "fail unless at least one layer is eligible and the kernel fires")
    args = parser.parse_args()

    kwargs = {"revision": args.revision} if args.revision else {}
    model, tokenizer = load(args.model, **kwargs)

    native_text, native_calls = _count_kernel(lambda: _greedy(model, tokenizer))

    with inference.fused_decode_conv_silu(model) as scoped:
        # The scope builds one subclass per eligible base class; identify them by name.
        patched_types = {type(m) for _, m in scoped.named_modules()
                         if type(m).__name__.startswith("_FusedDecodeConvSiLU")}
        eligible = _patched_count(scoped, patched_types)
        fused_text, fused_calls = _count_kernel(lambda: _greedy(scoped, tokenizer))
    leftover = _patched_count(model, patched_types)

    report = {
        "model": args.model,
        "eligible_layers": eligible,
        "still_patched_after_exit": leftover,
        "kernel_calls_native": native_calls,
        "kernel_calls_fused": fused_calls,
        "text_identical": native_text == fused_text,
        "native_text": native_text,
        "fused_text": fused_text,
    }
    print(json.dumps(report, indent = 2))

    failures = []
    if not report["text_identical"]:
        failures.append("greedy output changed under the fusion scope")
    if leftover:
        failures.append(f"{leftover} modules left patched after the scope exited")
    if native_calls:
        failures.append(f"kernel fired {native_calls} times outside the scope")
    if args.expect_recurrent:
        if eligible == 0:
            failures.append("no recurrent layer was eligible; the fusion never ran")
        elif fused_calls == 0:
            failures.append("layers were patched but the kernel never fired during decode")
    else:
        if eligible:
            failures.append(f"control model unexpectedly patched {eligible} layers")
        if fused_calls:
            failures.append(f"control model fired the kernel {fused_calls} times")

    for failure in failures:
        print(f"::error::{failure}")
    print("RESULT_1191:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
