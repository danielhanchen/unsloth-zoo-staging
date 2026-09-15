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

"""Hard gate: this runner really executes MLX on Metal, including a custom kernel.

Exits non-zero rather than skipping. A staging job whose MLX tests all skip is
green and vacuous, which is the failure mode this file exists to prevent.
"""

import platform
import sys

import mlx.core as mx


def main():
    print("machine:", platform.machine(), "python:", sys.version.split()[0])
    print("mlx:", mx.__version__ if hasattr(mx, "__version__") else "?")
    assert platform.machine() == "arm64", "not Apple Silicon"
    assert mx.metal.is_available(), "Metal unavailable"

    mx.set_default_device(mx.gpu)
    a = mx.ones((64, 64), dtype = mx.float32)
    b = a @ a
    mx.eval(b)
    mx.synchronize()
    assert mx.all(b == 64).item(), "matmul on gpu produced wrong values"

    kernel = mx.fast.metal_kernel(
        name = "unsloth_ci_probe",
        input_names = ["x"],
        output_names = ["y"],
        source = "uint i = thread_position_in_grid.x; y[i] = x[i] + 1.0f;",
    )
    x = mx.arange(32, dtype = mx.float32)
    (y,) = kernel(
        inputs = [x],
        grid = (32, 1, 1),
        threadgroup = (32, 1, 1),
        output_shapes = [x.shape],
        output_dtypes = [mx.float32],
    )
    mx.eval(y)
    mx.synchronize()
    assert mx.array_equal(y, x + 1).item(), "custom metal kernel produced wrong values"

    try:
        print("device_info:", mx.metal.device_info())
    except Exception as error:
        print("device_info unavailable:", error)
    print("METAL_EXECUTION_PASS")


if __name__ == "__main__":
    main()
