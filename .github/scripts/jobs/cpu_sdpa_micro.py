"""CPU SDPA forward + backward at Qwen3-0.6B attention shapes: bf16 flash (what bf16 autocast runs) vs fp32 flash on
bf16-valued inputs (cast in, cast the output back) vs bf16 math. Round 17: on Linux aarch64 (Cobalt 100) the bf16
flash backward was 18-24% of a bf16 SFT / DPO / full FT step.

    python jobs/cpu_sdpa_micro.py --out sdpa.json [--threads 4] [--seqs 128,512,1024]

Errors are max |out - ref| / max |ref| (and the same for dQ / dK / dV) against an fp64 math reference computed on the
same bf16-valued inputs.
"""
import argparse
import json
import platform
import statistics
import sys
import time

import torch
import torch.nn.functional as F


def cpu_id():
    out = {"machine": platform.machine(), "os": sys.platform, "torch": torch.__version__,
           "capability": torch.backends.cpu.get_cpu_capability() if hasattr(torch.backends, "cpu") else ""}
    try:
        if sys.platform.startswith("linux"):
            info = open("/proc/cpuinfo").read()
            names = [ln.split(":", 1)[1].strip() for ln in info.splitlines()
                     if ln.lower().startswith(("model name", "cpu part", "cpu implementer"))]
            out["brand"] = names[:3]
            flags = next((ln for ln in info.splitlines() if ln.lower().startswith(("flags", "features"))), "").split()
            out["flags"] = [x for x in ("avx2", "avx512f", "avx512_bf16", "amx_bf16", "bf16", "sve", "sve2", "i8mm")
                            if x in flags]
        elif sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            out["brand"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        elif sys.platform == "darwin":
            import subprocess
            out["brand"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                          text=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        out["brand_error"] = repr(e)
    return out


def run(fn, q, k, v, g, bwd):
    q, k, v = (t.detach().requires_grad_(bwd) for t in (q, k, v))
    t0 = time.perf_counter()
    out = fn(q, k, v)
    if bwd:
        out.backward(g)
    dt = time.perf_counter() - t0
    grads = (q.grad, k.grad, v.grad) if bwd else ()
    return dt, out.detach(), grads


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seqs", default="128,512,1024")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--tiny", action="store_true")  # staging CI passes it
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    H, KV, D = 16, 8, 128  # Qwen3-0.6B: 16 query heads, 8 KV heads (repeated, as transformers does), head dim 128

    def bf16_flash(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal = True)

    def fp32_flash(q, k, v):
        return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal = True).to(q.dtype)

    def bf16_math(q, k, v):
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, is_causal = True)

    methods = {"bf16_flash": bf16_flash, "fp32_flash": fp32_flash, "bf16_math": bf16_math}
    rec = {"cpu": cpu_id(), "threads": torch.get_num_threads(), "batch": a.batch, "res": {}}
    gen = torch.Generator().manual_seed(0)
    for S in [int(s) for s in a.seqs.split(",") if s]:
        q = torch.randn(a.batch, H, S, D, generator = gen).bfloat16()
        k = torch.randn(a.batch, KV, S, D, generator = gen).bfloat16().repeat_interleave(H // KV, 1)
        v = torch.randn(a.batch, KV, S, D, generator = gen).bfloat16().repeat_interleave(H // KV, 1)
        g = torch.randn(a.batch, H, S, D, generator = gen).bfloat16()
        qd, kd, vd = (t.double().requires_grad_(True) for t in (q, k, v))
        with sdpa_kernel([SDPBackend.MATH]):
            ref = F.scaled_dot_product_attention(qd, kd, vd, is_causal = True)
        ref.backward(g.double())
        refs = (ref.detach(), qd.grad, kd.grad, vd.grad)
        res = {}
        for name, fn in methods.items():
            row = {}
            for bwd in (False, True):
                times = []
                for i in range(a.reps + 1):
                    dt, out, grads = run(fn, q, k, v, g, bwd)
                    if i:
                        times.append(dt * 1e3)
                row["fwdbwd_ms" if bwd else "fwd_ms"] = round(statistics.median(times), 3)
                if bwd:
                    errs = [float((x.double() - r).abs().max() / r.abs().max()) for x, r in zip((out, *grads), refs)]
                    row["rel_err"] = {n: f"{e:.2e}" for n, e in zip(("out", "dq", "dk", "dv"), errs)}
            res[name] = row
        rec["res"][S] = res
        print(f"S={S}: " + "; ".join(f"{n} fwd {r['fwd_ms']} fwd+bwd {r['fwdbwd_ms']} err {r['rel_err']}"
                                     for n, r in res.items()), flush = True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent = 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
