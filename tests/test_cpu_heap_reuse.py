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

"""CPU training on Linux glibc: large buffers freed in one step are reused by the next, not faulted in again."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_MALLOC_ENV = ("MALLOC_MMAP_MAX_", "MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "MALLOC_TOP_PAD_",
               "GLIBC_TUNABLES", "LD_PRELOAD", "UNSLOTH_CPU_HEAP_REUSE")
_glibc_linux = pytest.mark.skipif(not sys.platform.startswith("linux") or platform.libc_ver()[0] != "glibc",
                                  reason = "glibc mallopt is Linux glibc only")

# A training step's large buffers in miniature: allocate, write and free 64 MB six times, after the CPU training
# defaults are installed. Prints [fitted state, page faults of each round after the first, pages per buffer].
_PROBE = """
import json, resource, torch
from unsloth_zoo.cpu import runtime
from unsloth_zoo.cpu.trainer import install_cpu_training_defaults
install_cpu_training_defaults()
def faults():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_minflt + r.ru_majflt
per_round = []
for _ in range(6):
    f = faults()
    x = torch.empty(16 * 2**20).fill_(1.0)
    del x
    per_round.append(faults() - f)
# A freed top of the heap above 2 GiB stays mapped too (no trimming at any size). Untouched, so no pages fault in.
import ctypes
libc = ctypes.CDLL("libc.so.6")
libc.sbrk.restype, libc.sbrk.argtypes = ctypes.c_void_p, [ctypes.c_long]
libc.malloc.restype, libc.malloc.argtypes = ctypes.c_void_p, [ctypes.c_size_t]
libc.free.argtypes = [ctypes.c_void_p]
top = libc.sbrk(0)
block = libc.malloc(3 * 2**30)
grew = (libc.sbrk(0) - top) / 2**30
libc.free(block)
kept = (libc.sbrk(0) - top) / 2**30
print(json.dumps([list(getattr(runtime, "_ALLOCATOR_FITTED", None) or ()), per_round[1:], 64 * 2**20 // resource.getpagesize(),
                  round(grew, 2), round(kept, 2)]))
"""


def _probe(extra):
    env = {k: v for k, v in os.environ.items() if k not in _MALLOC_ENV}
    env.update(extra)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    out = subprocess.run([sys.executable, "-c", _PROBE], env = env, capture_output = True, text = True, timeout = 600)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@_glibc_linux
def test_freed_buffers_are_reused_without_page_faults():
    fitted, faults, pages, grew, kept = _probe({})
    # Without reuse every round faults in the whole buffer again (~16K pages); with it only a few stray pages.
    assert max(faults) < pages // 8, (faults, pages)
    assert fitted == [True, "glibc heap reuse"], fitted
    # A 3 GiB block came from the heap (not mmap: the break moved by more than 2 GiB) and freeing it did not trim.
    assert grew > 2 and kept >= grew, (grew, kept)


@_glibc_linux
@pytest.mark.parametrize("extra, reason", [
    ({"UNSLOTH_CPU_HEAP_REUSE": "0"}, "UNSLOTH_CPU_HEAP_REUSE=0"),
    ({"MALLOC_MMAP_THRESHOLD_": "131072"}, "glibc malloc tuned by the environment"),
    ({"GLIBC_TUNABLES": "glibc.malloc.trim_threshold=1048576"}, "glibc malloc tuned by the environment"),
])
def test_kill_switch_and_user_tuning_keep_glibc_defaults(extra, reason):
    fitted, faults, pages, grew, kept = _probe(extra)
    assert fitted == [False, reason], fitted
    # glibc's own behaviour: a 64 MB buffer is above its mmap threshold, so each round faults it in again.
    assert min(faults) > pages // 2, (faults, pages)


def test_other_platforms_and_preloaded_allocators_are_left_alone(monkeypatch):
    from unsloth_zoo.cpu import runtime
    for platform_name, env, reason in (("darwin", {}, "not Linux"), ("win32", {}, "not Linux"),
                                        ("linux", {"LD_PRELOAD": "/usr/lib/libjemalloc.so.2"},
                                         "another allocator is preloaded")):
        monkeypatch.setattr(runtime, "_ALLOCATOR_FITTED", None)
        monkeypatch.setattr(runtime.sys, "platform", platform_name)
        for k in _MALLOC_ENV:
            monkeypatch.delenv(k, raising = False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert runtime.fit_cpu_allocator() == (False, reason)
        # Idempotent: the first answer is kept.
        assert runtime.fit_cpu_allocator() == (False, reason)
