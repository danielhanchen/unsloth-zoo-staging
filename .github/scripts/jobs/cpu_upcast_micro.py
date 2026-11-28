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
    if FP16W[0]:  # round 16: FBGEMM fp16-packed weights, fp32 activations + accumulation
        try:
            t0 = time.perf_counter()
            packs = [torch.ops.quantized.linear_prepack_fp16(w.float(), None) for w in ws]
            FP16W.append({"pack_s": round(time.perf_counter() - t0, 3)})
            m["fp16w"] = lambda x, w, i: torch.ops.quantized.linear_dynamic_fp16(x, packs[i])
        except Exception as e:  # noqa: BLE001
            FP16W.append(repr(e)[:200])
    if INDUCTOR[0]:
        try:  # round 15: Inductor C++ fused bf16 load + fp32 FMA reduction (no fp32 copy written)
            red = torch.compile(lambda x, w: (x[:, None, :] * w.float()[None, :, :]).sum(-1), dynamic = False)
            red(torch.randn(2, H), ws[0])
            m["inductor_red"] = lambda x, w, i: red(x, w)
        except Exception as e:  # noqa: BLE001
            INDUCTOR.append(repr(e)[:200])
    return m


INDUCTOR = [False]
FP16W = [False]


def lmhead_methods(wb, w32, tile_kib):
    """Round 15: tied LM head [V, H] per decoded token. fp32 = today on fp32-path x86; bf16 copy + row-chunked upcast
    into a reused buffer (exact values; GEMM split by vocab rows)."""
    tbufs = {k: torch.empty(k * 256) for k in tile_kib}

    def fp32(x, w, i):
        return F.linear(x, w32)

    def fp32_t(x, w, i):
        return torch.mm(w32, x.t()).t()

    def tile(k):
        tb = tbufs[k]

        def f(x, w, i):
            n, kk = wb.shape
            rows = max(1, tb.numel() // kk)
            out_t = torch.empty(n, x.shape[0])
            xt = x.t()
            for r in range(0, n, rows):
                rr = min(rows, n - r)
                b = tb[:rr * kk].view(rr, kk).copy_(wb[r:r + rr])
                torch.mm(b, xt, out = out_t[r:r + rr])
            return out_t.t()
        return f

    def tile_lin(k):  # same chunking, x @ W.T orientation per chunk (out column slices)
        tb = tbufs[k]

        def f(x, w, i):
            n, kk = wb.shape
            rows = max(1, tb.numel() // kk)
            out = torch.empty(x.shape[0], n)
            for r in range(0, n, rows):
                rr = min(rows, n - r)
                b = tb[:rr * kk].view(rr, kk).copy_(wb[r:r + rr])
                out[:, r:r + rr] = F.linear(x, b)
            return out
        return f

    def bf16(x, w, i):
        return F.linear(x.bfloat16(), wb).float()

    m = {"fp32": fp32, "fp32_t": fp32_t}
    for k in tile_kib:
        m[f"tileN{k}"] = tile(k)
        m[f"tileL{k}"] = tile_lin(k)
    m["bf16"] = bf16
    if FP16W[0]:
        try:
            pk = torch.ops.quantized.linear_prepack_fp16(w32, None)
            m["fp16w"] = lambda x, w, i: torch.ops.quantized.linear_dynamic_fp16(x, pk)
        except Exception as e:  # noqa: BLE001
            FP16W.append(repr(e)[:200])
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


def argmax_bench(rows, vocab, reps):
    """Round 16: last-dim argmax of [rows, vocab] logits (DPO token accuracy) by several equal-result routes."""
    out = {}
    for dt in (torch.float32, torch.bfloat16):
        x = torch.randn(rows, vocab).to(dt)
        ref = x.argmax(-1)

        def chunked(x, c = 16384):
            vals, idx = [], []
            for s0 in range(0, x.shape[-1], c):
                v, i = x[:, s0:s0 + c].max(-1)
                vals.append(v); idx.append(i + s0)
            v = torch.stack(vals, -1); i = torch.stack(idx, -1)
            return i.gather(-1, v.argmax(-1, keepdim = True)).squeeze(-1)
        fns = {"argmax": lambda x: x.argmax(-1), "max_indices": lambda x: x.max(-1).indices,
               "amax_eq": lambda x: (x == x.amax(-1, keepdim = True)).to(torch.uint8).argmax(-1),
               "topk1": lambda x: x.topk(1, -1).indices.squeeze(-1), "chunked16k": chunked,
               "float_argmax": lambda x: x.float().argmax(-1)}
        r = {}
        for name, fn in fns.items():
            y = fn(x)
            ts = []
            for _ in range(max(5, reps)):
                t = time.perf_counter(); fn(x); ts.append(time.perf_counter() - t)
            r[name] = {"median_ms": round(1e3 * statistics.median(ts), 2), "equal": bool(torch.equal(y, ref))}
        out[str(dt).replace("torch.", "")] = r
    return out


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
        for i in range(min(7, len(ws))):
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
    p.add_argument("--thread-list", default="")
    p.add_argument("--lmhead-tiles", default="")
    p.add_argument("--inductor", action="store_true")
    p.add_argument("--no-bwd", action="store_true")
    p.add_argument("--fp16w", action="store_true")
    p.add_argument("--argmax", default="", help="round 16: rows,vocab of a DPO token-accuracy argmax (e.g. 80,151936)")
    a = p.parse_args()
    INDUCTOR[0] = a.inductor
    FP16W[0] = a.fp16w
    if a.thread_list:
        res_all = {}
        for t in [int(x) for x in a.thread_list.split(",") if x]:
            torch.set_num_threads(t)
            res_all[t] = one(a)
        with open(a.out, "w") as f:
            json.dump({"by_threads": res_all, "cpu": cpu_id()}, f, indent = 1)
        return
    if a.threads:
        torch.set_num_threads(a.threads)
    with open(a.out, "w") as f:
        json.dump(one(a), f, indent = 1)


def one(a):
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
    res["inductor"] = INDUCTOR[1:] or bool(INDUCTOR[0])
    res["fp16w"] = FP16W[1:] or bool(FP16W[0])
    res["cpu"]["qengines"] = list(torch.backends.quantized.supported_engines)
    if a.lmhead_tiles or a.fp16w:
        lt = [int(t) for t in a.lmhead_tiles.split(",") if t]
        wb = (torch.randn(151936, H) * 0.02).bfloat16()
        lm = lmhead_methods(wb, wb.float(), lt)
        res["lmhead"] = {}
        for M in [int(m) for m in a.decode_m.split(",") if m]:
            res["lmhead"][M] = bench(lm, [wb], M, a.reps)
            print("lmhead M", M, json.dumps({k: v["median_ms"] for k, v in res["lmhead"][M].items()}), flush = True)
        del wb, lm
    if a.argmax:
        res["argmax"] = argmax_bench(*[int(v) for v in a.argmax.split(",")], a.reps)
        print("argmax", json.dumps(res["argmax"]), flush = True)
    if not a.no_bwd:
        bwd = make_bwd(tiles)
        for M in [int(m) for m in a.prefill_m.split(",") if m]:
            res["bwd"][M] = bench(bwd, ws, M, a.reps, in_dim_key = 0)
            print("bwd M", M, json.dumps({k: v["median_ms"] for k, v in res["bwd"][M].items()}), flush = True)
    return res


if __name__ == "__main__":
    main()
