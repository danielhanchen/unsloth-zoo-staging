"""Throwaway: which operation arms the mlx 0.32.1 teardown segfault.

Run as `python seg_probe.py <mode>`; the exit code is the signal (0 clean, 139 crash).
Staging only, never merged upstream.
"""
import sys
import mlx.core as mx

MODE = sys.argv[1]


def build():
    import mlx_vlm.models.qwen3_5.language as qlang
    sys.path.insert(0, "tests")
    from test_qwen35_vjp_metal import _tiny_text_config
    cfg = _tiny_text_config(full_attention_interval=1)
    model = qlang.Qwen3_5Model(cfg)
    mx.eval(model.parameters())
    rotaries = [l.self_attn.rotary_emb for l in model.layers if not l.is_linear]
    return cfg, model, rotaries


def inputs(cfg):
    B, H, L = 1, cfg.num_attention_heads, 4
    q = mx.random.normal((B, H, L, cfg.head_dim)).astype(mx.bfloat16)
    k = mx.random.normal((B, 1, L, cfg.head_dim)).astype(mx.bfloat16)
    pos = mx.tile(mx.expand_dims(mx.arange(L), 0)[None], (3, 1, 1))
    return q, k, pos


print("mode:", MODE, "mlx:", mx.__version__)

if MODE == "K":                       # build the model only
    build()

elif MODE == "L":                     # + one fused forward through the Metal kernel
    cfg, model, rots = build()
    q, k, pos = inputs(cfg)
    a, b = rots[0].apply_rotary(q, k, pos, unsqueeze_dim=1)
    mx.eval(a, b)

elif MODE == "M":                     # + a VJP through the fused kernel
    cfg, model, rots = build()
    q, k, pos = inputs(cfg)
    rot = rots[0]
    def loss(q_, k_):
        oq, ok = rot.apply_rotary(q_, k_, pos, unsqueeze_dim=1)
        return oq.astype(mx.float32).sum() + ok.astype(mx.float32).sum()
    try:
        val, _ = mx.value_and_grad(loss, argnums=(0, 1))(q, k)
        mx.eval(val)
    except ValueError as e:
        print("vjp raised (expected on some releases):", str(e)[:80])

elif MODE == "N":                     # M, then drop references and collect
    import gc
    cfg, model, rots = build()
    q, k, pos = inputs(cfg)
    rot = rots[0]
    def loss(q_, k_):
        oq, ok = rot.apply_rotary(q_, k_, pos, unsqueeze_dim=1)
        return oq.astype(mx.float32).sum() + ok.astype(mx.float32).sum()
    try:
        val, _ = mx.value_and_grad(loss, argnums=(0, 1))(q, k)
        mx.eval(val)
    except ValueError as e:
        print("vjp raised:", str(e)[:80])
    del loss, rot, rots, model, q, k, pos, val
    gc.collect()
    mx.synchronize()
    print("released and collected")

elif MODE == "O":                     # M, then gc.collect registered at exit
    import atexit, gc
    atexit.register(gc.collect)
    cfg, model, rots = build()
    q, k, pos = inputs(cfg)
    rot = rots[0]
    def loss(q_, k_):
        oq, ok = rot.apply_rotary(q_, k_, pos, unsqueeze_dim=1)
        return oq.astype(mx.float32).sum() + ok.astype(mx.float32).sum()
    try:
        val, _ = mx.value_and_grad(loss, argnums=(0, 1))(q, k)
        mx.eval(val)
    except ValueError as e:
        print("vjp raised:", str(e)[:80])

print("reached end of", MODE)
