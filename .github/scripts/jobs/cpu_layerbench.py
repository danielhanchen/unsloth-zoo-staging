"""Per-component CPU forward+backward timing for one Qwen3.5 decoder of each kind, at real sizes, random
weights (config only, no checkpoint), plus the ops inside the gated delta rule. Pinpoints which part of a
Qwen3.5 step is slow on a given CPU without the torch.profiler (whose post-processing hangs on small
macOS runners).

    python jobs/cpu_layerbench.py --model unsloth/Qwen3.5-0.8B --seq 128
"""

import argparse
import json
import os
import platform
import statistics
import time

import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
p.add_argument("--seq", type=int, default=128)
p.add_argument("--reps", type=int, default=5)
p.add_argument("--tiny", action="store_true", help="accepted for the staging harness; ignored")
p.add_argument("--out", default="outputs/jobs/cpu_layerbench.json")
a = p.parse_args()
torch.manual_seed(0)


def bench(fn, reps=a.reps):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return round(statistics.median(ts) * 1e3, 2)


def fwd_bwd(module, *inputs, **kw):
    def run():
        xs = [x.detach().requires_grad_(x.is_floating_point()) for x in inputs]
        out = module(*xs, **kw)
        out = out[0] if isinstance(out, tuple) else out
        out.float().sum().backward()
    return run


from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as m  # noqa: E402

cfg = AutoConfig.from_pretrained(a.model).get_text_config()
cfg._attn_implementation = "sdpa"
H, S = cfg.hidden_size, a.seq
x = torch.randn(1, S, H) * 0.1
pos = torch.arange(S)[None, None].expand(3, 1, S)  # multimodal RoPE: (3, batch, seq)
rotary = m.Qwen3_5TextRotaryEmbedding(cfg)
pe = rotary(x, pos)
results = {"host": {"system": platform.system(), "machine": platform.machine(), "threads": torch.get_num_threads(),
                    "torch": torch.__version__, "capability": torch.backends.cpu.get_cpu_capability()},
           "seq": S, "ms": {}}
r = results["ms"]

lin = m.Qwen3_5GatedDeltaNet(cfg, layer_idx=0)
r["gated_deltanet_layer"] = bench(fwd_bwd(lin, x))
attn = m.Qwen3_5Attention(cfg, layer_idx=3)
r["full_attention_layer"] = bench(fwd_bwd(attn, x, position_embeddings=pe, attention_mask=None))
mlp = m.Qwen3_5MLP(cfg, cfg.intermediate_size)
r["mlp"] = bench(fwd_bwd(mlp, x))

# Inside the delta rule: shapes as the chunked kernel sees them.
nh, dk, dv, C = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim, 64
nchunks = max(1, S // C)
ut = torch.randn(1, nh, nchunks, C, C).tril(-1) * 0.1 + torch.eye(C)
rhs = torch.randn(1, nh, nchunks, C, dv)
r["solve_triangular_fwd_bwd"] = bench(fwd_bwd(lambda A, B: torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True), ut, rhs))
r["solve_triangular_as_inverse_matmul"] = bench(fwd_bwd(lambda A, B: torch.linalg.inv_ex(A)[0] @ B, ut, rhs))
conv_dim = cfg.linear_num_key_heads * dk * 2 + nh * dv
conv = torch.nn.Conv1d(conv_dim, conv_dim, cfg.linear_conv_kernel_dim, groups=conv_dim, padding=cfg.linear_conv_kernel_dim - 1, bias=False)
r["depthwise_conv1d"] = bench(fwd_bwd(lambda t: conv(t)[..., :S], torch.randn(1, conv_dim, S)))
q = torch.randn(1, S, nh, dk)
k = torch.randn(1, S, nh, dk)
v = torch.randn(1, S, nh, dv)
g = -torch.rand(1, S, nh)
beta = torch.rand(1, S, nh)
r["chunk_gated_delta_rule"] = bench(fwd_bwd(lambda *t: m.torch_chunk_gated_delta_rule(*t, use_qk_l2norm_in_kernel=True)[0], q, k, v, g, beta))
w = torch.randn(cfg.vocab_size, H) * 0.02
r["lm_head_matmul_fwd"] = bench(lambda: (x[0] @ w.t()).sum())
r["sgemm_1024x1024x3584"] = bench(lambda: torch.randn(1024, 1024) @ torch.randn(1024, 3584))

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(results, open(a.out, "w"), indent=2)
print("JOB_RESULT " + json.dumps(results))
