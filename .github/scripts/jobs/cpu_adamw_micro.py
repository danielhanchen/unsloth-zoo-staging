"""Round 18: is CPU fused AdamW single-threaded, call-bound, or bandwidth-bound on Qwen3-0.6B-shaped fp32 params?

Times optimizer.step() for fused / foreach / per-tensor AdamW at default, half and 1 thread, one flat buffer (no
per-tensor overhead), and an all-bf16 copy (the byte floor a bf16-state kernel could reach), next to a STREAM-like
add / copy on the same element count, so each arm reads as effective GB/s against the runner's own bandwidth.

    python jobs/cpu_adamw_micro.py --layers 14 --out adamw.json
"""
import argparse, json, os, platform, statistics, sys, time
import torch

H, I, Q, KV, V = 1024, 3072, 2048, 1024, 151936


def shapes(layers):
    return [(V, H)] + [(Q, H), (KV, H), (KV, H), (H, Q), (I, H), (I, H), (H, I), (H,), (H,), (128,), (128,)] * layers + [(H,)]


def cpu_model():
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name") or line.startswith("Model"):
                    return line.split(":", 1)[1].strip()
            return platform.processor()
        if sys.platform == "win32":
            import subprocess
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output = True, text = True, timeout = 60).stdout.strip()
            return out or platform.processor()
        import subprocess
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output = True, text = True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return repr(e)[:100]


def make(shp, dtype, gen):
    ps = [torch.nn.Parameter(torch.randn(s, generator = gen, dtype = torch.float32).to(dtype) * 0.02) for s in shp]
    for p in ps:
        p.grad = (torch.randn(p.shape, generator = gen) * 1e-3).to(dtype)
    return ps


def time_step(opt, reps):
    for _ in range(2):
        opt.step()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        opt.step()
        out.append(time.perf_counter() - t)
    return round(statistics.median(out) * 1e3, 1), round((max(out) - min(out)) * 1e3, 1)


def bw(fn, nbytes, reps):
    fn()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    m = statistics.median(out)
    return round(m * 1e3, 1), round(nbytes / m / 1e9, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type = int, default = 14)
    ap.add_argument("--reps", type = int, default = 5)
    ap.add_argument("--out", default = "cpu_adamw_micro.json")
    ap.add_argument("--tiny", action = "store_true", help = "ignored (staging passes it to every job)")
    a = ap.parse_args()
    nthr = torch.get_num_threads()
    rep = {"torch": torch.__version__, "platform": platform.platform(), "machine": platform.machine(),
           "cpu_count": os.cpu_count(), "threads_default": nthr, "layers": a.layers, "arms": {}}
    rep["cpu_model"] = cpu_model()
    try:
        rep["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    except Exception:  # noqa: BLE001
        pass
    gen = torch.Generator().manual_seed(0)
    shp = shapes(a.layers)
    ps = make(shp, torch.float32, gen)
    n = sum(p.numel() for p in ps)
    rep["params"] = n
    rep["tensors"] = len(ps)
    step_bytes = 28 * n  # reads p, g, m, v (16 B) and writes p, m, v (12 B) per fp32 element
    rep["adamw_bytes_GB"] = round(step_bytes / 1e9, 2)

    def run(name, params, threads, **kw):
        torch.set_num_threads(threads)
        opt = torch.optim.AdamW(params, lr = 1e-5, weight_decay = 0.01, **kw)
        ms, spread = time_step(opt, a.reps)
        nb = sum(p.numel() * (16 + 12) // (1 if p.dtype == torch.float32 else 2) for p in params)
        rep["arms"][name] = {"ms": ms, "spread_ms": spread, "threads": threads, "GBps": round(nb / (ms / 1e3) / 1e9, 1)}
        print(name, rep["arms"][name], flush = True)
        del opt
        torch.set_num_threads(nthr)

    half = max(1, nthr // 2)
    run("fused", ps, nthr, fused = True)
    run("fused_half_threads", ps, half, fused = True)
    run("fused_1_thread", ps, 1, fused = True)
    run("foreach", ps, nthr, foreach = True)
    run("per_tensor", ps, nthr, foreach = False)
    # Bandwidth references on the same element count (one flat buffer each).
    x = torch.randn(n, generator = gen)
    y = torch.randn(n, generator = gen)
    z = torch.empty(n)
    rep["add_ms"], rep["add_GBps"] = bw(lambda: torch.add(x, y, out = z), 12 * n, a.reps)
    rep["copy_ms"], rep["copy_GBps"] = bw(lambda: z.copy_(x), 8 * n, a.reps)
    torch.set_num_threads(1)
    rep["add_1thr_ms"], rep["add_1thr_GBps"] = bw(lambda: torch.add(x, y, out = z), 12 * n, a.reps)
    torch.set_num_threads(nthr)
    del x, y, z
    # One flat buffer: the same bytes with no per-tensor dispatch.
    flat = [torch.nn.Parameter(torch.cat([p.detach().reshape(-1) for p in ps]))]
    flat[0].grad = torch.cat([p.grad.reshape(-1) for p in ps])
    run("fused_flat", flat, nthr, fused = True)
    del flat
    run("fused_again", ps, nthr, fused = True)  # order check: the per-tensor arm once more, after the flat one
    del ps
    # All-bf16 copy: half the bytes, the floor for any bf16-state variant (which still moves fp32 p and g: 20 B).
    psb = make(shp, torch.bfloat16, gen)
    run("fused_all_bf16", psb, nthr, fused = True)
    del psb
    f = rep["arms"]["fused"]
    rep["verdict"] = {
        "threading_speedup_default_vs_1": round(rep["arms"]["fused_1_thread"]["ms"] / f["ms"], 2),
        "fused_vs_add_bandwidth": round(f["GBps"] / rep["add_GBps"], 2),
        "per_tensor_overhead_ms": round(min(f["ms"], rep["arms"]["fused_again"]["ms"]) - rep["arms"]["fused_flat"]["ms"], 1),
        "bf16_state_floor_ms": round(f["ms"] * 20 / 28, 1),
    }
    print("verdict", rep["verdict"], flush = True)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent = 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
