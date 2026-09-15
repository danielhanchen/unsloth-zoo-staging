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

"""PR 1191 against a real mlx-community checkpoint's own recurrent layers.

The PR's own tests build `Qwen3_5GatedDeltaNet` from a hand-written config. This
runs the same prefill/decode contract on the 18 layers of a downloaded 4-bit
Qwen3.5 checkpoint, with their real quantized weights, on real Metal.

Load through mlx-vlm, not mlx-lm. This checkpoint carries a `vision_config`, so
`unsloth_zoo.mlx.loader._is_vlm` is True and unsloth builds it with mlx-vlm,
whose `Qwen3_5GatedDeltaNet` exposes the `_causal_conv1d_decode` fast path the
fusion matches. mlx-lm's own `GatedDeltaNet` has no such method, so loading the
same files with `mlx_lm.load` yields zero eligible layers and measures nothing.
"""

import argparse
import json
import sys

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_vlm import load
from mlx_vlm.models.qwen3_5 import language as native

from unsloth_zoo.mlx import inference

PREFILL, DECODE = 3, 1


def _bits(array):
    return np.array(array.view(mx.uint8))


def _hidden_size(model, path):
    for source in (getattr(getattr(model, "config", None), "text_config", None),
                   getattr(model, "config", None)):
        size = getattr(source, "hidden_size", None)
        if size:
            return int(size)
    config = json.load(open(f"{path}/config.json", encoding = "utf-8"))
    return int(config.get("text_config", config).get("hidden_size"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True)
    parser.add_argument("--revision", default = None)
    parser.add_argument("--expect-recurrent", action = "store_true")
    args = parser.parse_args()

    path = snapshot_download(args.model, revision = args.revision)
    model, _ = load(path)
    hidden = _hidden_size(model, path)

    recurrent = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, native.Qwen3_5GatedDeltaNet)]
    print(f"{len(recurrent)} recurrent layers, hidden_size={hidden}, "
          f"kernel_compiled={inference._decode_conv_silu_kernel() is not None}")

    mx.random.seed(3407)
    samples = [mx.random.normal((1, n, hidden)).astype(mx.bfloat16)
               for n in (PREFILL, DECODE, DECODE)]

    # Native reference first, while nothing is patched.
    expected = {}
    for name, module in recurrent:
        cache = native.ArraysCache(size = 2)
        steps = []
        for x in samples:
            out = module(x, cache = cache)
            mx.eval(out, cache.state)
            steps.append((out, list(cache.state)))
        expected[name] = steps

    calls = []
    original = inference._decode_conv_silu

    def counting(*call_args):
        calls.append(None)
        return original(*call_args)

    inference._decode_conv_silu = counting
    failures, patched_count, prefill_calls, decode_calls = [], 0, 0, 0
    try:
        with inference.fused_decode_conv_silu(model) as scoped:
            patched_count = sum(
                1 for _, module in scoped.named_modules()
                if type(module).__name__.startswith("_FusedDecodeConvSiLU"))
            for name, module in recurrent:
                cache = native.ArraysCache(size = 2)
                for x, (out, states) in zip(samples, expected[name]):
                    calls.clear()
                    actual = module(x, cache = cache)
                    mx.eval(actual, cache.state)
                    if not np.array_equal(_bits(actual), _bits(out)):
                        failures.append(f"{name}: output differs at T={x.shape[1]}")
                    for index, (state, reference) in enumerate(zip(cache.state, states)):
                        if not np.array_equal(_bits(state), _bits(reference)):
                            failures.append(f"{name}: cache[{index}] differs at T={x.shape[1]}")
                    if x.shape[1] == DECODE:
                        decode_calls += len(calls)
                    else:
                        prefill_calls += len(calls)
    finally:
        inference._decode_conv_silu = original

    leftover = sum(1 for _, module in model.named_modules()
                   if type(module).__name__.startswith("_FusedDecodeConvSiLU"))
    report = {
        "model": args.model,
        "recurrent_layers": len(recurrent),
        "patched_in_scope": patched_count,
        "still_patched_after_exit": leftover,
        "kernel_calls_during_decode": decode_calls,
        "kernel_calls_during_prefill": prefill_calls,
        "bitwise_mismatches": len(failures),
    }
    print(json.dumps(report, indent = 2))

    if leftover:
        failures.append(f"{leftover} modules left patched after the scope exited")
    if prefill_calls:
        failures.append(f"the kernel fired {prefill_calls} times during prefill")
    if args.expect_recurrent:
        if patched_count != len(recurrent) or not recurrent:
            failures.append(f"patched {patched_count} of {len(recurrent)} recurrent layers")
        # Two single-token decode steps per layer, one fused call each.
        if decode_calls != 2 * len(recurrent):
            failures.append(f"expected {2 * len(recurrent)} fused decode calls, got {decode_calls}")

    for failure in failures[:12]:
        print(f"::error::{failure}")
    print("RESULT_1191:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
