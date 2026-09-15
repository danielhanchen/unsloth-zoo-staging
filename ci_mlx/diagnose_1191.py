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

"""Why did no layer of a real qwen3_5 checkpoint become eligible for the fusion?

Prints, for the loaded model: whether the Metal kernel compiled at all, every
distinct module class, and the contract verdict per class, so a zero-eligibility
result names its own cause instead of being a bare number.
"""

import argparse
import sys

import mlx.core as mx
from mlx_lm import load

from unsloth_zoo.mlx import inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True)
    parser.add_argument("--revision", default = None)
    args = parser.parse_args()

    print("metal available:", mx.metal.is_available())
    kernel = inference._decode_conv_silu_kernel()
    print("kernel compiled:", kernel is not None)
    if kernel is None:
        # The helper swallows AttributeError/TypeError from mx.fast.metal_kernel,
        # so show the raw failure rather than the None it turns into.
        try:
            mx.fast.metal_kernel(
                name = "probe", input_names = ["x"], output_names = ["out"],
                ensure_row_contiguous = False,
                compile_options = {"math_mode": "safe"},
                source = "uint i = thread_position_in_grid.x; out[i] = x[i];",
            )
            print("raw metal_kernel with compile_options: accepted")
        except Exception as error:
            print(f"raw metal_kernel with compile_options rejected: "
                  f"{type(error).__name__}: {error}")

    kwargs = {"revision": args.revision} if args.revision else {}
    model, _ = load(args.model, **kwargs)

    classes = {}
    for name, module in model.named_modules():
        classes.setdefault(type(module), []).append(name)
    print(f"{len(classes)} distinct module classes")
    for cls, names in sorted(classes.items(), key = lambda kv: -len(kv[1])):
        if not (hasattr(cls, "_causal_conv1d_decode") or "Delta" in cls.__name__
                or "Recurrent" in cls.__name__ or "Linear" in cls.__name__ and False):
            continue
        sample = model
        for part in names[0].split("."):
            sample = sample[int(part)] if part.isdigit() else getattr(sample, part)
        print(f"  {cls.__module__}.{cls.__name__}: {len(names)} instances, e.g. {names[0]}")
        print(f"    training={getattr(sample, 'training', None)} "
              f"has _causal_conv1d_decode key={'_causal_conv1d_decode' in sample} "
              f"has __call__ key={'__call__' in sample}")
        print(f"    contract: {inference._decode_conv_silu_contract(cls) is not None}")

    delta = [cls for cls in classes if hasattr(cls, "_causal_conv1d_decode")]
    print("classes exposing _causal_conv1d_decode:", [c.__name__ for c in delta] or "NONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
