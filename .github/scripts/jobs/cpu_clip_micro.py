"""Round 17: full-FT grad clipping cost split on Qwen3-0.6B-shaped fp32 gradients (311 tensors, 2.4 GB): total norm
(zoo row-wise vs torch's own vs one flat buffer) and the clip multiply (per tensor vs foreach vs one flat buffer).

    python jobs/cpu_clip_micro.py --out clip.json
"""
# Full-FT grad clipping on Qwen3-0.6B-shaped fp32 gradients (311 tensors, tied embedding): what clip_grad_norm_ costs
# and where it goes (total norm vs the clip multiply), zoo's row-wise norm vs torch's own, and a one-buffer bound.
import argparse, json, platform, statistics, sys, time
import torch
from torch.nn.utils import clip_grad
import unsloth_zoo.cpu.ops as ops
ops.patch_cpu_grad_total_norm()
H, I, Q, KV, L, V = 1024, 3072, 2048, 1024, 28, 151936
shapes = [(V, H)] + [(Q, H), (KV, H), (KV, H), (H, Q), (I, H), (I, H), (H, I), (H,), (H,), (128,), (128,)] * L + [(H,)]
gen = torch.Generator().manual_seed(0)
grads = [torch.randn(s, generator = gen) * 1e-2 for s in shapes]
flat = torch.cat([g.reshape(-1) for g in grads])
zoo = clip_grad._get_total_norm; stock = zoo._unsloth_original
def t(fn, reps = 7):
    out = []
    for i in range(reps + 1):
        t0 = time.perf_counter(); fn(); dt = time.perf_counter() - t0
        if i: out.append(dt * 1e3)
    return round(statistics.median(out), 2)
rec = {"threads": torch.get_num_threads(), "numel": flat.numel(), "tensors": len(grads)}
rec["norm_zoo_ms"] = t(lambda: zoo(grads, 2.0))
rec["norm_stock_ms"] = t(lambda: stock(grads, 2.0))
rec["norm_flat_ms"] = t(lambda: flat.norm())
rec["mul_per_tensor_ms"] = t(lambda: [g.mul_(1.0) for g in grads])
rec["mul_foreach_ms"] = t(lambda: torch._foreach_mul_(grads, 1.0))
rec["mul_flat_ms"] = t(lambda: flat.mul_(1.0))
params = [torch.nn.Parameter(torch.empty_like(g)) for g in grads]
for p, g in zip(params, grads): p.grad = g
rec["clip_grad_norm_ms"] = t(lambda: torch.nn.utils.clip_grad_norm_(params, 1.0))
ref = torch.sqrt(sum((g.double() ** 2).sum() for g in grads)).item()
rec["rel_err"] = {k: abs(float(f(grads, 2.0)) - ref) / ref for k, f in (("zoo", zoo), ("stock", stock))}
rec["gb_read"] = round(flat.numel() * 4 / 1e9, 2)
rec["machine"] = platform.machine(); rec["os"] = sys.platform
try:
    rec["brand"] = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "")
except OSError:
    rec["brand"] = platform.processor()
print("CLIP " + json.dumps(rec))
p = argparse.ArgumentParser(); p.add_argument("--out"); p.add_argument("--tiny", action = "store_true")
a, _ = p.parse_known_args()
if a.out:
    json.dump(rec, open(a.out, "w"), indent = 1)
