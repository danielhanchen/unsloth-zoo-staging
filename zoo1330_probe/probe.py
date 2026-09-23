"""Metal differential probe for unsloth-zoo#1330. Run once per arm in a fresh process.

python probe.py ARM OUT_JSON   (ARM is a label only; the arm's runtime_cce.py is already in place)
"""
import hashlib
import json
import math
import sys
import time

import numpy as np
import mlx.core as mx

from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss
from unsloth_zoo.mlx.cce import runtime_cce as rt

ARM, OUT = sys.argv[1], sys.argv[2]
res = {"arm": ARM, "mlx": mx.__version__, "metal": bool(mx.metal.is_available())}
try:
    res["device"] = mx.metal.device_info().get("architecture")
except Exception as e:
    res["device"] = repr(e)
res["has_helper"] = hasattr(rt, "_SOFTCAP_HEADER")


# ---------------------------------------------------------------- tanh table (H1, H2, H3)
VALS = [-math.inf, -3.0e38, -1.0e30, -1.0e4, -100.0, -45.0, -44.0, -30.0, -20.5, -20.0, -10.0, -9.0,
        -1.0, 0.0, 1.0, 9.0, 10.0, 20.0, 20.5, 30.0, 44.0, 45.0, 89.0, 100.0, 1.0e4, 1.0e30, 3.0e38,
        math.inf, math.nan]
src = """
    uint i = thread_position_in_grid.x;
    float x = inp[i];
    fast_out[i] = fast::tanh(x);
    precise_out[i] = precise::tanh(x);
    helper_out[i] = x > 20.0f ? 1.0f : fast::tanh(x);
"""
k = mx.fast.metal_kernel(name="probe_tanh_table", input_names=["inp"],
                         output_names=["fast_out", "precise_out", "helper_out"], source=src)
xin = mx.array(VALS, dtype=mx.float32)
fo, po, ho = k(inputs=[xin], grid=(len(VALS), 1, 1), threadgroup=(len(VALS), 1, 1),
               output_shapes=[(len(VALS),)] * 3, output_dtypes=[mx.float32] * 3)
mg = mx.tanh(xin, stream=mx.gpu)
mc = mx.tanh(xin, stream=mx.cpu)
mx.eval(fo, po, ho, mg, mc)
res["tanh_table"] = [
    {"x": repr(v), "fast": repr(float(a)), "precise": repr(float(b)), "helper_as_in_pr": repr(float(c)),
     "mx_tanh_gpu": repr(float(d)), "mx_tanh_cpu": repr(float(e)), "np": repr(float(np.tanh(np.float64(v))))}
    for v, a, b, c, d, e in zip(VALS, fo.tolist(), po.tolist(), ho.tolist(), mg.tolist(), mc.tolist())
]


# ---------------------------------------------------------------- helpers
def run(hidden, weight, targets, *, softcap, quantized=False, frozen=False, precompute=False,
        label_smoothing=0.0, chunk=2048):
    args = (hidden, weight, targets)
    deq = weight
    if quantized:
        packed, scales, biases = mx.quantize(weight, group_size=64, bits=4)
        args = (hidden, packed, scales, biases, targets)
        deq = mx.dequantize(packed, scales, biases, group_size=64, bits=4).astype(weight.dtype)
    kw = dict(ignore_index=-100, logit_softcap=softcap, chunk_size=chunk, quantized=quantized,
              group_size=64 if quantized else None, bits=4 if quantized else None,
              label_smoothing=label_smoothing)
    if frozen:
        kw["weight_is_frozen"] = True
    if precompute:
        kw["precompute_hidden_gradient"] = True
    runtime, used = make_chunked_cross_entropy_loss(**kw)
    losses, grad = mx.value_and_grad(lambda h: runtime(h, *args[1:]).sum())(hidden)
    per_tok = runtime(hidden, *args[1:])
    mx.eval(losses, grad, per_tok, deq)
    return per_tok, grad, deq, bool(used)


def reference(hidden, deq, targets, softcap, label_smoothing=0.0):
    # logits in the same dtype the kernels read (matmul output dtype), then float64 true tanh.
    logits = np.array((hidden @ deq.T).astype(mx.float32), dtype=np.float64)
    with np.errstate(all="ignore"):
        t = np.tanh(logits / softcap) if softcap > 0 else None
        capped = softcap * t if softcap > 0 else logits
        m = capped.max(axis=1, keepdims=True)
        lse = (m + np.log(np.exp(capped - m).sum(axis=1, keepdims=True)))[:, 0]
        tg = np.array(targets.tolist())
        rows = np.arange(len(tg))
        eps = label_smoothing
        V = capped.shape[1]
        loss = lse - (1 - eps) * capped[rows, tg] - eps * capped.mean(axis=1)
        p = np.exp(capped - lse[:, None])
        onehot = np.zeros_like(p)
        onehot[rows, tg] = 1.0
        d = p - (1 - eps) * onehot - eps / V
        if softcap > 0:
            d = d * (1 - t * t)
        grad = d @ np.array(deq.astype(mx.float32), dtype=np.float64)
    return loss, grad


def summarize(per_tok, grad, ref_loss, ref_grad):
    L = np.array(per_tok.astype(mx.float32), dtype=np.float64)
    G = np.array(grad.astype(mx.float32), dtype=np.float64)
    out = {
        "loss_sum": repr(float(L.sum())),
        "loss_finite": bool(np.isfinite(L).all()),
        "loss_nan_rows": int(np.isnan(L).sum()),
        "grad_finite": bool(np.isfinite(G).all()),
        "grad_nan": int(np.isnan(G).sum()),
        "ref_loss_sum": repr(float(ref_loss.sum())),
        "ref_finite": bool(np.isfinite(ref_loss).all() and np.isfinite(ref_grad).all()),
    }
    if out["loss_finite"] and np.isfinite(ref_loss).all():
        out["loss_max_abs_err"] = float(np.abs(L - ref_loss).max())
    if out["grad_finite"] and np.isfinite(ref_grad).all():
        scale = max(1e-30, float(np.abs(ref_grad).max()))
        out["grad_max_abs_err"] = float(np.abs(G - ref_grad).max())
        out["grad_rel_err"] = float(np.abs(G - ref_grad).max() / scale)
    return out


def h(a):
    return hashlib.sha256(np.array(a.astype(mx.float32)).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- saturation scenarios (H1, H3, H6, H7, H10)
N, H, V = 64, 128, 8192
targets = (mx.arange(N) * 61).astype(mx.int32)
alt = mx.where((mx.arange(V) % 2 == 0)[:, None], 1.0, -1.0)  # alternate row sign


def scen(name, dtype):
    if name == "pos_overflow":      # PR's own case: fp16 logits -> +inf
        return mx.full((N, H), 300, dtype=dtype), mx.full((V, H), 300, dtype=dtype)
    if name == "neg_overflow":      # every logit -> -inf (fp16) / very negative
        return mx.full((N, H), 300, dtype=dtype), mx.full((V, H), -300, dtype=dtype)
    if name == "mixed_overflow":    # half the vocab +huge, half -huge
        return mx.full((N, H), 300, dtype=dtype), (300 * alt * mx.ones((V, H))).astype(dtype)
    if name == "ratio_30":          # finite logits, ratio ~ +-30 (inside (20, 44])
        return mx.ones((N, H), dtype=dtype), ((270.0 / H) * alt * mx.ones((V, H))).astype(dtype)
    if name == "ratio_60":          # finite logits, ratio ~ +-60 (exp(2x) overflows fp32 above 44.36)
        return mx.ones((N, H), dtype=dtype), ((540.0 / H) * alt * mx.ones((V, H))).astype(dtype)
    if name == "ratio_1e4_fp32":    # finite fp32 logits, ratio ~ +-1e4
        return mx.full((N, H), 30, dtype=dtype), ((3000.0) * alt * mx.ones((V, H))).astype(dtype)
    if name == "nan_hidden":        # one NaN row must stay NaN (not masked to finite)
        hid = mx.random.normal((N, H), key=mx.random.key(1)).astype(dtype)
        hid[3, 5] = float("nan")
        return hid, (mx.random.normal((V, H), key=mx.random.key(2)) * 0.05).astype(dtype)
    raise KeyError(name)


res["scenarios"] = []
for name in ["pos_overflow", "neg_overflow", "mixed_overflow", "ratio_30", "ratio_60", "ratio_1e4_fp32", "nan_hidden"]:
    for dtype in [mx.float16, mx.bfloat16, mx.float32]:
        if name == "ratio_1e4_fp32" and dtype != mx.float32:
            continue
        for path in ["dense", "quant", "quant_frozen_precompute", "dense_label_smoothing"]:
            hid, w = scen(name, dtype)
            kw = dict(softcap=9.0)
            if path.startswith("quant"):
                kw["quantized"] = True
            if path == "quant_frozen_precompute":
                kw.update(frozen=True, precompute=True)
            if path == "dense_label_smoothing":
                kw["label_smoothing"] = 0.1
            rec = {"scenario": name, "dtype": str(dtype), "path": path}
            try:
                per_tok, grad, deq, used = run(hid, w, targets, **kw)
                rl, rg = reference(hid, deq, targets, 9.0, kw.get("label_smoothing", 0.0))
                rec.update(summarize(per_tok, grad, rl, rg))
                rec["metal_kernels_used"] = used
                rec["loss_hash"], rec["grad_hash"] = h(per_tok), h(grad)
            except Exception as e:  # record, keep going
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            res["scenarios"].append(rec)


# ---------------------------------------------------------------- bit parity on ordinary inputs (step 3)
res["parity"] = []
for seed in range(4):
    for dtype in [mx.float16, mx.bfloat16, mx.float32]:
        for softcap in [0.0, 9.0, 30.0]:
            for path in ["dense", "quant", "quant_frozen_precompute"]:
                n, hh, v = 96, 256, 10000
                scale = [0.02, 0.2, 1.0, 3.0][seed]  # last seeds push ratios past 20 for softcap 9
                hid = (mx.random.normal((n, hh), key=mx.random.key(10 + seed))).astype(dtype)
                w = (mx.random.normal((v, hh), key=mx.random.key(20 + seed)) * scale).astype(dtype)
                tg = mx.random.randint(0, v, (n,), key=mx.random.key(30 + seed)).astype(mx.int32)
                tg[0] = -100
                kw = dict(softcap=softcap, chunk=4096)
                if path.startswith("quant"):
                    kw["quantized"] = True
                if path == "quant_frozen_precompute":
                    kw.update(frozen=True, precompute=True)
                rec = {"seed": seed, "dtype": str(dtype), "softcap": softcap, "path": path}
                try:
                    per_tok, grad, deq, used = run(hid, w, tg, **kw)
                    logits = np.array((hid @ deq.T).astype(mx.float32))
                    rec["max_abs_ratio"] = float(np.abs(logits).max() / softcap) if softcap else None
                    rec["loss_hash"], rec["grad_hash"] = h(per_tok), h(grad)
                    rec["finite"] = bool(mx.all(mx.isfinite(per_tok[1:])).item() and mx.all(mx.isfinite(grad)).item())
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                res["parity"].append(rec)


# ---------------------------------------------------------------- cost (H8)
def bench(softcap, dtype, reps=15):
    n, hh, v = 4096, 2048, 65536
    hid = (mx.random.normal((n, hh), key=mx.random.key(5)) * 0.5).astype(dtype)
    w = (mx.random.normal((v, hh), key=mx.random.key(6)) * 0.05).astype(dtype)
    tg = mx.random.randint(0, v, (n,), key=mx.random.key(7)).astype(mx.int32)
    runtime, _ = make_chunked_cross_entropy_loss(ignore_index=-100, logit_softcap=softcap, chunk_size=0)
    f = mx.value_and_grad(lambda a, b: runtime(a, b, tg).sum(), argnums=(0, 1))
    for _ in range(3):
        l, g = f(hid, w)
        mx.eval(l, g)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        l, g = f(hid, w)
        mx.eval(l, g)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"softcap": softcap, "dtype": str(dtype), "median_ms": ts[len(ts) // 2], "min_ms": ts[0], "max_ms": ts[-1]}


if "--no-bench" not in sys.argv:
    res["bench"] = [bench(sc, dt) for dt in [mx.bfloat16, mx.float16] for sc in [0.0, 30.0]]

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
print("wrote", OUT)
