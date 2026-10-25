"""Which CPU is this runner, what does torch think it supports, and which matmuls run without crashing.

    python jobs/cpu_isa_probe.py

Each matmul runs in its own subprocess (an illegal instruction kills only that probe) under a few library
caps, so a crash is attributed to oneDNN, MKL or neither. Writes outputs/jobs/cpu_isa_probe.json.
"""

import argparse
import json
import os
import platform
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--tiny", action="store_true", help="accepted for the staging harness; ignored")
p.add_argument("--out", default="outputs/jobs/cpu_isa_probe.json")
a = p.parse_args()

INFO = r"""
import json, torch
from torch import cpu
flags = {}
for name in ("_is_avx2_supported", "_is_avx512_supported", "_is_avx512_vnni_supported", "_is_avx512_bf16_supported",
             "_is_amx_tile_supported", "_is_amx_fp16_supported", "_is_arm_sve_supported"):
    fn = getattr(cpu, name, None)
    flags[name.strip("_")] = fn() if fn else None
print(json.dumps({"torch": torch.__version__, "capability": torch.backends.cpu.get_cpu_capability(),
                  "mkldnn_bf16": torch.ops.mkldnn._is_mkldnn_bf16_supported(), "flags": flags}))
"""

MM = r"""
import sys, torch
dtype = getattr(torch, sys.argv[1]); m, k, n = map(int, sys.argv[2:5])
torch.manual_seed(0)
a = torch.randn(m, k).to(dtype); b = torch.randn(n, k).to(dtype)
out = a @ b.t()
ref = a.double() @ b.double().t()
print(float((out.double() - ref).abs().max()))
"""


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if r.stdout.strip():
            return r.stdout.strip()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def run(code, args=(), env=None):
    r = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True,
                       env=dict(os.environ, **(env or {})), timeout=300)
    return {"rc": r.returncode, "out": r.stdout.strip()[-400:], "err": r.stderr.strip()[-600:]}


result = {"job": "cpu_isa_probe", "cpu": cpu_name(), "platform": platform.platform(), "machine": platform.machine(),
          "cpu_count": os.cpu_count()}
info = run(INFO)
result["torch"] = json.loads(info["out"]) if info["rc"] == 0 else info
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py-cpuinfo"], capture_output=True)
try:
    import cpuinfo  # py-cpuinfo
    fl = set(cpuinfo.get_cpu_info().get("flags", []))
    result["cpuid_flags"] = sorted(f for f in fl if f.startswith(("avx512", "amx", "avx_vnni", "avxvnni")) or f in ("avx2", "fma"))
except Exception as e:  # noqa: BLE001
    result["cpuid_flags"] = repr(e)

shapes = {"small": ("8", "64", "64"), "large": ("512", "1024", "2048")}
caps = {"default": {}, "onednn_avx512_core": {"ONEDNN_MAX_CPU_ISA": "AVX512_CORE"},
        "onednn_avx2": {"ONEDNN_MAX_CPU_ISA": "AVX2"}, "mkl_avx2": {"MKL_ENABLE_INSTRUCTIONS": "AVX2"}}
result["matmul"] = {}
for dtype in ("float32", "bfloat16", "float16"):
    for shape, dims in shapes.items():
        for cap, env in caps.items():
            if dtype == "float32" and cap != "default":
                continue
            r = run(MM, (dtype, *dims), env)
            result["matmul"][f"{dtype}/{shape}/{cap}"] = "ok" if r["rc"] == 0 else f"rc={r['rc']} {r['err'][-200:]}"
verbose = run(MM, ("bfloat16", *shapes["small"]), {"ONEDNN_VERBOSE": "1"})
result["onednn_verbose_bf16_small"] = [line for line in (verbose["out"] + "\n" + verbose["err"]).splitlines()
                                       if "exec" in line or "info" in line][:12]
result["passed"] = all(v == "ok" for k, v in result["matmul"].items() if k.endswith("/default"))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(result, open(a.out, "w"), indent=2)
print("JOB_RESULT " + json.dumps(result))
sys.exit(0)
