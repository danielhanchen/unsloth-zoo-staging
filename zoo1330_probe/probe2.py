"""Second Metal probe for unsloth-zoo#1330: why do sub-threshold softcap cells move, and by how much.

python probe2.py OUT_JSON    (run from a checkout; variants are patched in-process via _SOFTCAP_HEADER)
"""
import json
import sys
import time

import numpy as np
import mlx.core as mx

from unsloth_zoo.mlx.cce import runtime_cce as rt
from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

OUT = sys.argv[1]
ONLY = sys.argv[2]
res = {"mlx": mx.__version__}

VARIANTS = {
    # plain fast::tanh through the helper (== base / revert arm)
    "fast": "return fast::tanh(ratio);",
    # the PR as submitted
    "pr_ternary": "return ratio > 20.0f ? 1.0f : fast::tanh(ratio);",
    # clamp the argument instead of branching on the result
    "clamp_arg": "return fast::tanh(metal::min(ratio, 20.0f));",
    # IEEE-strict tanh
    "precise": "return precise::tanh(ratio);",
}

# ------------------------------------------------ elementwise: helper(x) vs fast::tanh(x), dense grid
grid = np.concatenate([np.linspace(-50, 50, 2_000_001, dtype=np.float32),
                       np.array([np.inf, -np.inf, 3e38, -3e38, 1e30, 44.0, 44.5, 45.0], np.float32)])
xin = mx.array(grid)
res["elementwise"] = {}
for name, body in VARIANTS.items():
    src = "uint i = thread_position_in_grid.x; out[i] = helper(inp[i]);"
    k = mx.fast.metal_kernel(name="p2_elem_" + name, input_names=["inp"], output_names=["out"], source=src,
                             header="inline float helper(float ratio) { " + body + " }")
    (o,) = k(inputs=[xin], grid=(grid.size, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[grid.shape], output_dtypes=[mx.float32])
    o = np.array(o)
    ref = np.tanh(grid.astype(np.float64))
    fin = np.isfinite(grid)
    res["elementwise"][name] = {
        "nonfinite_outputs_on_finite_inputs": int((~np.isfinite(o[fin])).sum()),
        "max_abs_err_vs_f64": float(np.nanmax(np.abs(o[fin] - ref[fin]))),
        "values_at_edges": {repr(float(x)): repr(float(y)) for x, y in zip(grid[-8:], o[-8:])},
    }
    res["elementwise"][name]["_raw"] = o
base_o = res["elementwise"]["fast"]["_raw"]
for name in VARIANTS:
    o = res["elementwise"][name].pop("_raw")
    m = np.abs(grid) <= 20
    res["elementwise"][name]["bitdiff_vs_fast_on_|x|<=20"] = int((o[m].view(np.uint32) != base_o[m].view(np.uint32)).sum())

# ------------------------------------------------ end-to-end: per-cell deltas vs base and vs float64 reference
def run(hidden, weight, targets, softcap, quantized, frozen):
    args = (hidden, weight, targets)
    deq = weight
    if quantized:
        packed, scales, biases = mx.quantize(weight, group_size=64, bits=4)
        args = (hidden, packed, scales, biases, targets)
        deq = mx.dequantize(packed, scales, biases, group_size=64, bits=4).astype(weight.dtype)
    kw = dict(ignore_index=-100, logit_softcap=softcap, chunk_size=4096, quantized=quantized,
              group_size=64 if quantized else None, bits=4 if quantized else None)
    if frozen:
        kw.update(weight_is_frozen=True, precompute_hidden_gradient=True)
    runtime, _ = make_chunked_cross_entropy_loss(**kw)
    per_tok = runtime(hidden, *args[1:])
    _, grad = mx.value_and_grad(lambda h: runtime(h, *args[1:]).sum())(hidden)
    mx.eval(per_tok, grad, deq)
    return np.array(per_tok.astype(mx.float32), np.float64), np.array(grad.astype(mx.float32), np.float64), deq


def ref(hidden, deq, targets, softcap):
    logits = np.array((hidden @ deq.T).astype(mx.float32), np.float64)
    t = np.tanh(logits / softcap)
    c = softcap * t
    m = c.max(1, keepdims=True)
    lse = (m + np.log(np.exp(c - m).sum(1, keepdims=True)))[:, 0]
    tg = np.array(targets.tolist())
    rows = np.arange(len(tg))
    valid = tg != -100
    loss = np.where(valid, lse - c[rows, np.where(valid, tg, 0)], 0.0)
    p = np.exp(c - lse[:, None])
    p[rows[valid], tg[valid]] -= 1.0
    p[~valid] = 0.0
    return loss, (p * (1 - t * t)) @ np.array(deq.astype(mx.float32), np.float64)


cells = []
for seed in range(4):
    for dtype in [mx.float16, mx.bfloat16, mx.float32]:
        for softcap in [9.0, 30.0]:
            for quantized, frozen in [(False, False), (True, False), (True, True)]:
                scale = [0.02, 0.2, 1.0, 3.0][seed]
                n, hh, v = 96, 256, 10000
                hid = mx.random.normal((n, hh), key=mx.random.key(10 + seed)).astype(dtype)
                w = (mx.random.normal((v, hh), key=mx.random.key(20 + seed)) * scale).astype(dtype)
                tg = mx.random.randint(0, v, (n,), key=mx.random.key(30 + seed)).astype(mx.int32)
                tg[0] = -100
                cells.append((seed, dtype, softcap, quantized, frozen, hid, w, tg))

outs = {}
for name, body in [(ONLY, VARIANTS[ONLY])]:
    rt._SOFTCAP_HEADER = "\ninline float cce_softcap_tanh(float ratio) {\n    " + body + "\n}\n"
    outs[name] = []
    for (seed, dtype, softcap, quantized, frozen, hid, w, tg) in cells:
        L, G, deq = run(hid, w, tg, softcap, quantized, frozen)
        outs[name].append((L, G))
        if name == "fast":
            pass
    # cost of this variant, softcap on, realistic size
    n, hh, v = 4096, 2048, 65536
    hid = (mx.random.normal((n, hh), key=mx.random.key(5)) * 0.5).astype(mx.bfloat16)
    w = (mx.random.normal((v, hh), key=mx.random.key(6)) * 0.05).astype(mx.bfloat16)
    tg = mx.random.randint(0, v, (n,), key=mx.random.key(7)).astype(mx.int32)
    runtime, _ = make_chunked_cross_entropy_loss(ignore_index=-100, logit_softcap=30.0, chunk_size=0)
    f = mx.value_and_grad(lambda a, b: runtime(a, b, tg).sum(), argnums=(0, 1))
    for _ in range(3):
        mx.eval(*f(hid, w))
    ts = []
    for _ in range(15):
        t0 = time.perf_counter()
        mx.eval(*f(hid, w))
        ts.append((time.perf_counter() - t0) * 1e3)
    res.setdefault("cost_ms_softcap30_bf16", {})[name] = sorted(ts)[7]

refs = []
for (seed, dtype, softcap, quantized, frozen, hid, w, tg) in cells:
    deq = w
    if quantized:
        p, s_, b = mx.quantize(w, group_size=64, bits=4)
        deq = mx.dequantize(p, s_, b, group_size=64, bits=4).astype(w.dtype)
    refs.append(ref(hid, deq, tg, softcap))
arrs = {}
for i, (L, G) in enumerate(outs[ONLY]):
    arrs[f"L{i}"], arrs[f"G{i}"] = L, G
    arrs[f"RL{i}"], arrs[f"RG{i}"] = refs[i]
np.savez_compressed(OUT.replace(".json", ".npz"), **arrs)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1)[:3000])
