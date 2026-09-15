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

"""Fail when a pytest run passed nothing.

Every MLX suite here gates on `mx.metal.is_available()`, so a runner without
Metal skips all of them and the job reports success having proven nothing.
Usage: python assert_ran.py <pytest-output-file> <minimum-passed>
"""

import re
import sys


def main(path, minimum):
    text = open(path, encoding = "utf-8", errors = "replace").read()
    passed = sum(int(n) for n in re.findall(r"(\d+) passed", text))
    skipped = sum(int(n) for n in re.findall(r"(\d+) skipped", text))
    failed = sum(int(n) for n in re.findall(r"(\d+) (?:failed|error)", text))
    print(f"ASSERT_RAN passed={passed} skipped={skipped} failed={failed} minimum={minimum}")
    if passed < minimum:
        print(f"::error::only {passed} tests passed (minimum {minimum}); "
              "a skipped MLX suite is not evidence")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2])))
