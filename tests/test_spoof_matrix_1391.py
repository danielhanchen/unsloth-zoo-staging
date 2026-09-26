# SPDX-License-Identifier: AGPL-3.0-only
"""Spoofed-accelerator detection matrix for unsloth-zoo #1391 (Ascend NPU support).

Every cell is ONE child process (vendor spoofs patch torch process-wide). The child loads the
real `unsloth_zoo.device_type` (and, when detection succeeds, the touched helper modules
gradient_checkpointing / loss_utils / vllm_utils) from a given package root through a stub
parent package, so `unsloth_zoo/__init__.py` never runs and the SAME cell can be run against

  HEAD: the unsloth_zoo package under SPOOF_HEAD_ROOT (default: this repo checkout), and
  BASE: a temp copy of that package with the PR-touched files replaced by the merge-base copies
        vendored in tests/_spoof_base_1391/ (regenerate: temp/spoof_matrix/vendor_base_1391.sh).

Each cell asserts the head verdict, then the differential: EXPECT "same" cells must produce a
byte-identical record on base (Gate B2: detection only widens), "changed" cells must differ.

Cell id: `<OS>-patched|<vendor>|<state>`. OS is always a platform.system()/machine() patch: the
only OS-dependent branch in the touched code is `is_mlx_available()` (Darwin + arm64 + mlx), so no
cell needs real-OS behaviour; the real runner OS is recorded in each record for the log.
"""

import concurrent.futures as cf
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_HEAD_ROOT = Path(os.environ.get("SPOOF_HEAD_ROOT") or _HERE.parent).resolve()
_BASE_FILES = _HERE / "_spoof_base_1391"

_CHILD = r'''
import json, os, platform, sys, tempfile, textwrap, traceback, types, importlib.machinery
spec = json.loads(sys.argv[1]); pkg_root = sys.argv[2]
OS, vendor, state = spec["os"], spec["vendor"], spec["state"]
_real_os = platform.system()
_real_system, _real_machine = platform.system, platform.machine

import torch
calls = []
def rec(name, ret=None):
    def f(*a, **k):
        calls.append(name); return ret
    return f

def reset():
    torch.cuda.is_available = lambda: False
    torch.cuda.empty_cache = rec("cuda.empty_cache")  # real one is a silent no-op on CPU torch
    torch.version.hip = None
    if hasattr(torch, "xpu"):
        torch.xpu.is_available = lambda: False
    if hasattr(torch, "npu"):
        del torch.npu
    if hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False
    set_accel(None)

def set_accel(acc):
    if hasattr(torch, "accelerator"):
        torch.accelerator.is_available = lambda: acc is not None
        torch.accelerator.current_accelerator = lambda: acc

def spoof_cuda(n=2):
    c = torch.cuda
    c.is_available = lambda: True
    c.device_count = lambda: n
    c.synchronize = rec("cuda.synchronize"); c.empty_cache = rec("cuda.empty_cache")
    c.is_bf16_supported = rec("cuda.is_bf16_supported", True)
    c.mem_get_info = rec("cuda.mem_get_info", (1, 2))
    c.get_device_capability = lambda *a: (8, 0)

def spoof_xpu(n=1):
    x = torch.xpu
    x.is_available = lambda: True
    x.device_count = lambda: n
    x.synchronize = rec("xpu.synchronize"); x.empty_cache = rec("xpu.empty_cache")
    x.is_bf16_supported = rec("xpu.is_bf16_supported", True)
    x.mem_get_info = rec("xpu.mem_get_info", (1, 2))

def make_npu(available=True, n=8, raises=False):
    m = types.ModuleType("torch.npu")
    if raises:
        def _boom(): raise RuntimeError("npu driver probe failed")
        m.is_available = _boom
    else:
        m.is_available = lambda: available
    m.device_count = lambda: n
    m.synchronize = rec("npu.synchronize"); m.empty_cache = rec("npu.empty_cache")
    m.is_bf16_supported = rec("npu.is_bf16_supported", True)
    m.mem_get_info = rec("npu.mem_get_info", (1, 2))
    m.stream = lambda s: s
    m.device = type("npu_device", (), {})
    torch.npu = m
    return m

def torch_npu_pkg(body):
    d = tempfile.mkdtemp(prefix="spoof_torch_npu_")
    os.makedirs(os.path.join(d, "torch_npu"))
    with open(os.path.join(d, "torch_npu", "__init__.py"), "w") as fh:
        fh.write(textwrap.dedent(body))
    sys.path.insert(0, d)

hooks = types.ModuleType("_spoof_hooks"); hooks.make_npu = make_npu; hooks.tried = []; sys.modules["_spoof_hooks"] = hooks

reset()
if vendor == "nvidia":
    spoof_cuda(); set_accel("cuda")
elif vendor == "amd_rocm":
    spoof_cuda(1); torch.version.hip = "7.2.1"; set_accel("cuda")
elif vendor == "intel_xpu":
    spoof_xpu(); set_accel("xpu")
elif vendor == "apple_mps":
    torch.backends.mps.is_available = lambda: True; set_accel("mps")
elif vendor == "apple_mlx":
    mlx = types.ModuleType("mlx"); mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", None)
    sys.modules["mlx"] = mlx
elif vendor == "ascend_npu":
    if state == "healthy":
        make_npu(); set_accel("npu")
    elif state == "npu_lazy_import":  # torch.npu appears only once torch_npu is imported
        torch_npu_pkg("import torch, _spoof_hooks\n_spoof_hooks.tried.append(1)\n_spoof_hooks.make_npu()\n")
        set_accel(None)
    elif state == "npu_is_available_raises":
        make_npu(raises=True); set_accel(None)
    elif state == "npu_probe_raises_accel_npu":
        make_npu(raises=True); set_accel("npu")
    elif state == "npu_unavailable":
        make_npu(available=False); set_accel(None)
    elif state == "npu_zero_devices":
        make_npu(n=0); set_accel("npu")
if state == "broken_torch_npu":  # torch_npu installed, import dies (no CANN driver)
    torch_npu_pkg("import _spoof_hooks\n_spoof_hooks.tried.append(1)\n"
                  "raise RuntimeError('libascendcl.so: cannot open shared object file')\n")
if state == "plus_npu":  # a second backend present; the original one must keep winning
    make_npu()

stub = types.ModuleType("unsloth_zoo"); stub.__path__ = [os.path.join(pkg_root, "unsloth_zoo")]
sys.modules["unsloth_zoo"] = stub

def ns_of(obj, attr):
    for ns in ("cuda", "xpu", "npu"):
        mod = getattr(torch, ns, None)
        if mod is not None and getattr(mod, attr, object()) is obj:
            return ns
    return repr(obj)

out = {"real_os": _real_os, "torch_npu_imported": None}
try:
    # OS patched only while device_type imports: is_mlx_available() is cached there, and torch /
    # transformers / triton must keep loading their real native libraries.
    platform.system = lambda: OS
    platform.machine = lambda: {"Darwin": "arm64", "Windows": "AMD64"}.get(OS, "x86_64")
    try:
        import unsloth_zoo.device_type as dt
    finally:
        platform.system, platform.machine = _real_system, _real_machine
    out["detect"] = "OK " + dt.DEVICE_TYPE
    out["device_type_torch"] = dt.DEVICE_TYPE_TORCH
    out["device_count"] = dt.DEVICE_COUNT
except BaseException as e:
    out["detect"] = "RAISE " + type(e).__name__
    out["detect_msg"] = str(e).strip().splitlines()[0][:90] if str(e).strip() else ""
out["torch_npu_imported"] = "torch_npu" in sys.modules
out["torch_npu_import_attempted"] = bool(hooks.tried)
if out["detect"].startswith("OK ") and dt.DEVICE_TYPE != "mlx":
    del calls[:]
    dt.device_synchronize(); dt.device_empty_cache()
    out["helper_calls"] = list(calls)
    out["bf16"] = dt.device_is_bf16_supported()
    stub.DEVICE_TYPE, stub.DEVICE_TYPE_TORCH = dt.DEVICE_TYPE, dt.DEVICE_TYPE_TORCH
    try:
        import unsloth_zoo.gradient_checkpointing as gc
        out["gc_amp_device"] = getattr(gc.torch_amp_custom_fwd, "keywords", {}).get("device_type")
        out["gc_stream_ns"] = ns_of(getattr(gc, "torch_gpu_stream", None), "stream")
    except BaseException as e:
        out["gc_amp_device"] = out["gc_stream_ns"] = "unavailable: " + type(e).__name__ + ": " + str(e)[:80]
    # loss_utils / vllm_utils / training_utils import triton + transformers (and triton probes a
    # real CUDA device at import), so their device-selection code is lifted by AST from the file
    # under test and exec'd against the same spoofed torch + detected DEVICE_TYPE.
    import ast
    def lift(fname, keep):
        path = os.path.join(pkg_root, "unsloth_zoo", fname)
        tree = ast.parse(open(path, encoding="utf-8").read())
        ns = {"torch": torch, "DEVICE_TYPE": dt.DEVICE_TYPE, "__name__": "lifted"}
        exec(compile(ast.Module([n for n in tree.body if keep(n)], []), path, "exec"), ns)
        return ns, tree
    def assigns(n, name):
        return any(isinstance(t, ast.Name) and t.id == name
                   for a in ast.walk(n) if isinstance(a, ast.Assign) for t in a.targets)
    ns, _ = lift("loss_utils.py", lambda n: isinstance(n, (ast.Assign, ast.If)) and assigns(n, "current_device"))
    out["loss_current_device_ns"] = ns_of(ns.get("current_device"), "device")
    ns, _ = lift("vllm_utils.py", lambda n: isinstance(n, ast.FunctionDef) and n.name in ("get_mem_info", "_device_empty_cache"))
    del calls[:]; ns["get_mem_info"]()
    out["vllm_mem_info_calls"] = list(calls)
    del calls[:]
    if "_device_empty_cache" in ns:
        ns["_device_empty_cache"]()
    else:  # merge base: the call sites the helper replaced were inline torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    out["vllm_empty_cache_calls"] = list(calls)
    tree = ast.parse(open(os.path.join(pkg_root, "unsloth_zoo", "training_utils.py"), encoding="utf-8").read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "unsloth_train")
    amp = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and assigns(n, "_amp_device")]
    # merge base: GradScaler / autocast / zeros / .to() were handed the literal "cuda"
    out["train_amp_device"] = eval(compile(ast.Expression(amp[0].value), "t", "eval"), {"DEVICE_TYPE": dt.DEVICE_TYPE}) if amp else "cuda"
print("SPOOF_RESULT " + json.dumps(out, sort_keys=True))
'''

_OSES = ("Linux", "Windows", "Darwin")
_VENDORS = ("nvidia", "amd_rocm", "intel_xpu", "ascend_npu", "apple_mps", "cpu_only")
_NS = {"cuda": "cuda", "hip": "cuda", "xpu": "xpu", "npu": "npu"}
_RAISE = "RAISE NotImplementedError"
_HEALTHY = {"nvidia": "OK cuda", "amd_rocm": "OK hip", "intel_xpu": "OK xpu", "ascend_npu": "OK npu",
            "apple_mps": _RAISE, "cpu_only": _RAISE, "apple_mlx": "OK mlx"}


def _cells():
    """(id, spec, expected head detect, expect) with expect 'same' or 'changed:<why>'."""
    cells = []
    for osn in _OSES:
        vendors = _VENDORS + (("apple_mlx",) if osn == "Darwin" else ())
        for v in vendors:
            states = ["healthy", "broken_torch_npu"]
            if v in ("nvidia", "intel_xpu"):
                states.append("plus_npu")
            if v == "ascend_npu":
                states += ["npu_lazy_import", "npu_is_available_raises", "npu_probe_raises_accel_npu",
                           "npu_unavailable", "npu_zero_devices"]
            for s in states:
                head, expect = _HEALTHY[v], "same"
                if v == "ascend_npu":
                    head = {"healthy": "OK npu", "npu_lazy_import": "OK npu", "npu_zero_devices": "OK npu",
                            "npu_probe_raises_accel_npu": "RAISE RuntimeError"}.get(s, _RAISE)
                    if s in ("healthy", "npu_lazy_import", "npu_zero_devices"):
                        expect = "changed:Ascend host now selects npu instead of raising"
                    elif s == "npu_probe_raises_accel_npu":
                        expect = "changed:broken NPU torch now says 'reinstall torch' (RuntimeError)"
                spec = {"os": osn, "vendor": v, "state": s}
                cells.append((f"{osn}-patched|{v}|{s}", spec, head, expect))
    return cells


CELLS = _cells()
_IDS = [c[0] for c in CELLS]


def _run(spec, pkg_root, workdir):
    env = {k: v for k, v in os.environ.items()
           if k not in ("UNSLOTH_ZOO_DISABLE_GPU_INIT", "UNSLOTH_ALLOW_CPU", "PYTHONPATH")}
    env["CUDA_VISIBLE_DEVICES"] = ""  # a real host GPU must never answer for a spoofed cell
    env["UNSLOTH_COMPILE_DISABLE"] = "1"
    if spec["vendor"] == "apple_mlx":
        env.pop("UNSLOTH_FORCE_GPU_PATH", None)
    else:
        env["UNSLOTH_FORCE_GPU_PATH"] = "1"  # a real mlx install on macOS runners must not win
    p = subprocess.run([sys.executable, str(workdir / "child.py"), json.dumps(spec), str(pkg_root)],
                       capture_output=True, text=True, env=env, timeout=600, cwd=str(workdir))
    for line in p.stdout.splitlines():
        if line.startswith("SPOOF_RESULT "):
            return json.loads(line[len("SPOOF_RESULT "):])
    return {"detect": "HARNESS_ERROR rc=%s" % p.returncode, "stderr": p.stderr[-1500:]}


@pytest.fixture(scope="module")
def results():
    try:
        import torch  # noqa: F401
    except Exception as e:  # never skip: a skipped matrix is green and proves nothing
        pytest.fail(f"spoof matrix needs torch on this runner (staging: --extra-deps torch): {e!r}")
    if not (_HEAD_ROOT / "unsloth_zoo" / "device_type.py").is_file():
        pytest.fail(f"no unsloth_zoo package under SPOOF_HEAD_ROOT={_HEAD_ROOT}")
    work = Path(tempfile.mkdtemp(prefix="spoof1391_"))
    try:
        (work / "child.py").write_text(_CHILD, encoding="utf-8")
        base_root = work / "base"
        shutil.copytree(_HEAD_ROOT / "unsloth_zoo", base_root / "unsloth_zoo",
                        ignore=shutil.ignore_patterns("__pycache__"))
        vendored = sorted((_BASE_FILES / "unsloth_zoo").glob("*.py"))
        assert vendored, f"no vendored base files under {_BASE_FILES}"
        for f in vendored:
            shutil.copy2(f, base_root / "unsloth_zoo" / f.name)
        jobs = {}
        with cf.ThreadPoolExecutor(max_workers=max(2, min(8, os.cpu_count() or 2))) as ex:
            for cid, spec, _, _ in CELLS:
                jobs[(cid, "head")] = ex.submit(_run, spec, _HEAD_ROOT, work)
                jobs[(cid, "base")] = ex.submit(_run, spec, base_root, work)
            got = {k: f.result() for k, f in jobs.items()}
        summary = {"cells": len(CELLS), "same": 0, "changed": 0}
        for cid, _, _, expect in CELLS:
            summary["same" if expect == "same" else "changed"] += 1
        print("SPOOF_MATRIX_1391", json.dumps(summary), "head_root", _HEAD_ROOT, "runner", platform.system())
        return got
    finally:
        shutil.rmtree(work, ignore_errors=True)


@pytest.mark.parametrize("cid,spec,head,expect", CELLS, ids=_IDS)
def test_head_verdict(results, cid, spec, head, expect):
    r = results[(cid, "head")]
    assert r["detect"] == head, r
    if spec["state"] in ("broken_torch_npu", "npu_lazy_import"):
        # The new torch_npu import is reached only when nothing earlier answered, and a broken
        # torch_npu fails OPEN (falls through), never crashes detection.
        reached = spec["vendor"] in ("ascend_npu", "apple_mps", "cpu_only")
        assert r["torch_npu_import_attempted"] is reached, r
    if not head.startswith("OK ") or head == "OK mlx":
        return
    dev = head[3:]
    ns = _NS[dev]
    # Helpers must reach the selected backend, never a CUDA no-op (the NPU half of the PR).
    assert r["helper_calls"] == [f"{ns}.synchronize", f"{ns}.empty_cache"], r["helper_calls"]
    assert r["bf16"] is True, r
    if spec["state"] == "npu_zero_devices":
        assert r["device_count"] == 0, r
    # gradient_checkpointing needs only torch, so it is imported for real on every runner.
    assert r["gc_amp_device"] == ("npu" if dev == "npu" else "cuda"), r
    assert r["gc_stream_ns"] == ns, r
    assert r["loss_current_device_ns"] == ns, r
    assert r["vllm_mem_info_calls"] == [f"{ns}.mem_get_info"], r
    empty = "npu" if dev == "npu" else "cuda"  # vLLM paths are CUDA/ROCm/NPU only
    assert r["vllm_empty_cache_calls"] == [f"{empty}.empty_cache"], r
    assert r["train_amp_device"] == ("npu" if dev == "npu" else "cuda"), r


@pytest.mark.parametrize("cid,spec,head,expect", CELLS, ids=_IDS)
def test_base_vs_head(results, cid, spec, head, expect):
    h, b = results[(cid, "head")], results[(cid, "base")]
    assert not b["detect"].startswith("HARNESS_ERROR"), b
    h = {k: v for k, v in h.items() if k != "torch_npu_import_attempted"}  # head-only probe side effect
    b = {k: v for k, v in b.items() if k != "torch_npu_import_attempted"}
    if expect == "same":
        assert b == h, f"cell moved on a path the PR does not intend to change:\nbase={b}\nhead={h}"
    else:
        assert b["detect"] != h["detect"], f"{expect!r} but base already answers {b['detect']}"
