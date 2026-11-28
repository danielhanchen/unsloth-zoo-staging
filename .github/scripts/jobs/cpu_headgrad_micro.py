"""Round 18: the chunked CE's LM-head weight gradient for bf16 weights (bf16 full finetuning) on Qwen3-0.6B shapes.

Current: an fp32 GEMM of the fp32 logit gradient [rows, V] with the upcast hidden [rows, H] into an fp32 [V, H]
buffer, scaled, then cast to bf16. Candidate: a bf16 GEMM (fp32 accumulation inside) straight to bf16, as autograd
forms every other bf16 layer's weight gradient. Times each part (warm heap: the same buffers reused) and reports the
relative error of both against an fp64 reference.

    python jobs/cpu_headgrad_micro.py --out headgrad.json
"""
import argparse, json, platform, statistics, sys, time
import torch

V, H = 151936, 1024


def med(fn, reps):
    fn()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return round(statistics.median(out) * 1e3, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default = "48,110,256")
    ap.add_argument("--reps", type = int, default = 5)
    ap.add_argument("--out", default = "cpu_headgrad_micro.json")
    ap.add_argument("--tiny", action = "store_true", help = "ignored (staging passes it to every job)")
    a = ap.parse_args()
    if sys.platform.startswith("linux"):  # warm-heap timing on every arm: page faults are measured elsewhere
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-1, 2**31 - 1), libc.mallopt(-4, 0)
    rep = {"torch": torch.__version__, "platform": platform.platform(), "threads": torch.get_num_threads(), "rows": {}}
    try:
        rep["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    except Exception:  # noqa: BLE001
        pass
    gen = torch.Generator().manual_seed(0)
    for rows in [int(x) for x in a.rows.split(",")]:
        # Logit gradient: softmax minus one-hot, so a few large negatives and many tiny positives.
        z = torch.randn(rows, V, generator = gen) * 2
        g = torch.softmax(z, -1)
        g[torch.arange(rows), torch.randint(0, V, (rows,), generator = gen)] -= 1
        h = (torch.randn(rows, H, generator = gen)).to(torch.bfloat16)
        ref = (g.double().t() @ h.double())
        r = {}
        r["fp32_gemm_ms"] = med(lambda: g.t() @ h.float(), a.reps)
        w32 = g.t() @ h.float()
        r["cast_to_bf16_ms"] = med(lambda: w32.to(torch.bfloat16), a.reps)
        r["bf16_gemm_ms"] = med(lambda: g.to(torch.bfloat16).t() @ h, a.reps)
        cur = w32.to(torch.bfloat16)
        new = g.to(torch.bfloat16).t() @ h
        for name, x in (("current", cur), ("bf16_gemm", new)):
            d = x.double() - ref
            r[f"{name}_rel_fro"] = float(d.norm() / ref.norm())
            r[f"{name}_max_abs"] = float(d.abs().max())
        r["current_ms"] = round(r["fp32_gemm_ms"] + r["cast_to_bf16_ms"], 1)
        rep["rows"][rows] = r
        print(rows, r, flush = True)
        del z, g, h, ref, w32, cur, new
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent = 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
