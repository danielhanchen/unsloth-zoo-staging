import sys
sys.path.insert(0, "tests")
import test_mlx_decode_metal as t
import mlx.core as mx
import unsloth_zoo
from unsloth_zoo.mlx import attention as A, inference as D
print("unsloth_zoo from", unsloth_zoo.__file__)
scores, complete = mx.random.normal((1, 4, 2, 64)), mx.full((1, 2), 64, dtype = mx.int32)
try:
    out = A.select_blocks(scores, complete, 0, 128)
    mx.eval(out)
    print("DIRECT: kernel launched with topk 0, output shape", out.shape)
except ValueError as e:
    print("DIRECT: refused before launch:", e)
att = t._qsa_layer()
att.indexer.block_topk = 0
launches, orig = [], A.select_blocks
A.select_blocks = lambda *a: launches.append(a[2]) or orig(*a)
D._QSA_VERDICTS.clear()
try:
    with D.fused_qsa_attention(att):
        outs = t._qsa_run(att)
    mx.eval(outs)
    print("E2E: decode finished")
except Exception as e:
    print("E2E: decode raised", type(e).__name__, str(e)[:300])
print("E2E: select kernel launches with topk 0:", len(launches))
