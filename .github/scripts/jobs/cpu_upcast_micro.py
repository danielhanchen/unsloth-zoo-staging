"""Frozen bf16 weight x fp32 activation linear on CPU: per-call upcast (main) vs alternatives, Qwen3-0.6B shapes.

One "pass" = the 196 projections of Qwen3-0.6B in layer order (0.88 GB bf16, larger than any cache), so each weight
comes from DRAM as in a real decode step. Methods (forward y = x @ W.T unless noted):
  main     F.linear(x, W.to(fp32))                       (what _Bf16StoredLinearFunction does)
  fp32     F.linear(x, W32)                              (fp32 storage, 2x the bytes)
  scratch  W copied into one reused fp32 buffer, then F.linear
  tileN<k> rows of W upcast k KiB at a time into a reused buffer, y.T[rows] = W32[rows] @ x.T (out= slices)
  bf16     F.linear(x.bf16(), W).float()                  (bf16 GEMM, activations rounded: numerics differ)
Backward-shaped dX = dY @ W:  main / tileK<k> (row tiles of W, accumulated with addmm).
    python jobs/cpu_upcast_micro.py --out micro.json [--threads 2]
"""
import argparse
import json
import platform
import statistics
import sys
import time

import torch
import torch.nn.functional as F

H, I, Q, KV, LAYERS = 1024, 3072, 2048, 1024, 28


def cpu_id():
    out = {"processor": platform.processor(), "machine": platform.machine(), "os": sys.platform}
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            out["brand"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        elif sys.platform == "darwin":
            import subprocess
            out["brand"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        else:
            info = open("/proc/cpuinfo").read()
            out["brand"] = next((ln.split(":", 1)[1].strip() for ln in info.splitlines() if ln.startswith("model name")), "")
            flags = next((ln for ln in info.splitlines() if ln.startswith("flags")), "").split()
            out["flags"] = [x for x in ("avx2", "avx512f", "avx512bw", "avx512_bf16", "amx_bf16") if x in flags]
            out["cache"] = {ln.split(":")[0].strip(): ln.split(":", 1)[1].strip() for ln in info.splitlines()
                            if ln.startswith("cache size")}
    except Exception as e:  # noqa: BLE001
        out["brand_error"] = repr(e)
    out["cpu_capability"] = torch.backends.cpu.get_cpu_capability() if hasattr(torch.backends, "cpu") else ""
    out["torch"] = torch.__version__
    return out


def shapes():
    per = [(Q, H), (KV, H), (KV, H), (H, Q), (I, H), (I, H), (H, I)]  # q k v o gate up down (out, in)
    return per * LAYERS


def make_methods(ws, w32, tile_kib):
    big = max(w.numel() for w in ws)
    buf = torch.empty(big)
    tbufs = {k: torch.empty(k * 256) for k in tile_kib}

    def main(x, w, i):
        return F.linear(x, w.to(torch.float32))

    def fp32(x, w, i):
        return F.linear(x, w32[i])

    def scratch(x, w, i):
        return F.linear(x, buf[:w.numel()].view(w.shape).copy_(w))

    def bf16(x, w, i):
        return F.linear(x.bfloat16(), w).float()

    def tile(k):
        tb = tbufs[k]

        def f(x, w, i):
            n, kk = w.shape
            rows = max(1, tb.numel() // kk)
            out_t = torch.empty(n, x.shape[0])
            xt = x.t()
            for r in range(0, n, rows):
                rr = min(rows, n - r)
                b = tb[:rr * kk].view(rr, kk).copy_(w[r:r + rr])
                torch.mm(b, xt, out = out_t[r:r + rr])
            return out_t.t()
        return f

    m = {"main": main, "fp32": fp32, "scratch": scratch, "bf16": bf16}
    for k in tile_kib:
        m[f"tileN{k}"] = tile(k)
    return m


def make_bwd(tile_kib):
    tbufs = {k: torch.empty(k * 256) for k in tile_kib}

    def main(g, w, i):
        return g @ w.to(torch.float32)

    def tile(k):
        tb = tbufs[k]

        def f(g, w, i):
            n, kk = w.shape
            rows = max(1, tb.numel() // kk)
            out = torch.zeros(g.shape[0], kk)
            for r in range(0, n, rows):
                rr = min(rows, n - r)
                b = tb[:rr * kk].view(rr, kk).copy_(w[r:r + rr])
                out.addmm_(g[:, r:r + rr], b)
            return out
        return f

    m = {"main": main}
    for k in tile_kib:
        m[f"tileK{k}"] = tile(k)
    return m


def run_pass(fn, ws, xs, key = 1):
    t = time.perf_counter()
    for i, w in enumerate(ws):
        fn(xs[w.shape[key]], w, i)
    return time.perf_counter() - t


def bench(methods, ws, M, reps, in_dim_key = 1):
    xs = {d: torch.randn(M, d) for d in {w.shape[in_dim_key] for w in ws}}
    # numerics on 7 distinct weights (one layer)
    err = {}
    for name, fn in methods.items():
        worst, exact = 0.0, True
        for i in range(7):
            ref = list(methods.values())[0](xs[ws[i].shape[in_dim_key]], ws[i], i)
            y = fn(xs[ws[i].shape[in_dim_key]], ws[i], i)
            exact &= bool(torch.equal(ref, y))
            worst = max(worst, ((y - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item())
        err[name] = {"bitwise": exact, "max_rel_err": worst}
    times = {n: [] for n in methods}
    for name, fn in methods.items():
        run_pass(fn, ws, xs, in_dim_key)  # warm
    for _ in range(reps):
        for name, fn in methods.items():
            times[name].append(run_pass(fn, ws, xs, in_dim_key))
    return {n: {"median_ms": round(1e3 * statistics.median(v), 2), "min_ms": round(1e3 * min(v), 2),
                "spread_pct": round(100 * (max(v) - min(v)) / statistics.median(v), 1), **err[n]}
            for n, v in times.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="cpu_upcast_micro.json")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--tiles", default="128,512,2048")
    p.add_argument("--decode-m", default="2,8")
    p.add_argument("--prefill-m", default="256")
    p.add_argument("--tiny", action="store_true")
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    tiles = [int(t) for t in a.tiles.split(",") if t]
    ws = [(torch.randn(s) * 0.02).bfloat16() for s in shapes()]
    w32 = [w.float() for w in ws]
    res = {"cpu": cpu_id(), "threads": torch.get_num_threads(), "fwd": {}, "bwd": {}}
    print(json.dumps(res["cpu"]), flush = True)
    fwd = make_methods(ws, w32, tiles)
    for M in [int(m) for m in (a.decode_m + "," + a.prefill_m).split(",") if m]:
        res["fwd"][M] = bench(fwd, ws, M, a.reps)
        print("fwd M", M, json.dumps({k: v["median_ms"] for k, v in res["fwd"][M].items()}), flush = True)
    bwd = make_bwd(tiles)
    for M in [int(m) for m in a.prefill_m.split(",") if m]:
        res["bwd"][M] = bench(bwd, ws, M, a.reps, in_dim_key = 0)
        print("bwd M", M, json.dumps({k: v["median_ms"] for k, v in res["bwd"][M].items()}), flush = True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent = 1)


if __name__ == "__main__":
    main()
