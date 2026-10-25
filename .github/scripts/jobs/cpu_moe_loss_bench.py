"""CPU: (1) transformers MoE experts backends eager / batched_mm / grouped_mm, forward+backward, parity and
time; (2) can the CPU chunked CE and fused-LM-head loss paths be torch.compiled (graph breaks, parity, time).

    CUDA_VISIBLE_DEVICES= python jobs/cpu_moe_loss_bench.py --threads 8 --out outputs/cpu_moe_loss_bench.json
"""

import argparse
import copy
import json
import os
import platform
import statistics
import sys
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--threads", type=int, default=0)
p.add_argument("--reps", type=int, default=5)
p.add_argument("--tokens", type=int, default=256)
p.add_argument("--skip-moe", action="store_true")
p.add_argument("--skip-loss", action="store_true")
p.add_argument("--tiny", action="store_true", help="accepted for the staging harness; ignored")
p.add_argument("--out", default="outputs/jobs/cpu_moe_loss_bench.json")
a = p.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
torch.manual_seed(0)


def bench(fn, reps=a.reps):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return {"median_ms": round(statistics.median(ts) * 1e3, 2), "min_ms": round(min(ts) * 1e3, 2),
            "max_ms": round(max(ts) * 1e3, 2)}


def rel(x, y):
    return float((x.double() - y.double()).abs().max() / y.double().abs().max().clamp_min(1e-30))


res = {"host": {"system": platform.system(), "machine": platform.machine(), "torch": torch.__version__,
                "threads": torch.get_num_threads(), "capability": torch.backends.cpu.get_cpu_capability(),
                "grouped_mm": hasattr(torch, "_grouped_mm"), "F.grouped_mm": hasattr(torch.nn.functional, "grouped_mm")},
       "moe": {}, "loss": {}}

if not a.skip_moe:
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe import modeling_qwen3_moe as qm
    shapes = {
        # Qwen3-30B-A3B layer: 128 experts, top-8.
        "qwen3_30b_a3b": dict(hidden_size=2048, moe_intermediate_size=768, num_experts=128, num_experts_per_tok=8),
        # Small MoE (OLMoE-ish but narrower): 64 experts, top-8.
        "small_64e": dict(hidden_size=1024, moe_intermediate_size=512, num_experts=64, num_experts_per_tok=8),
    }
    for name, kw in shapes.items():
        for dtype, frozen in ((torch.float32, True), (torch.bfloat16, True), (torch.float32, False), (torch.bfloat16, False)):
            if not frozen and name != "small_64e":
                continue  # trainable experts (expert LoRA / full finetuning): small shape only, for runner RAM
            out_ref = grad_ref = None
            x0 = (torch.randn(1, a.tokens, kw["hidden_size"]) * 0.5).to(dtype)
            for impl in ("eager", "batched_mm", "grouped_mm"):
                cfg = Qwen3MoeConfig(**kw, norm_topk_prob=True)
                cfg._experts_implementation = impl
                block = qm.Qwen3MoeSparseMoeBlock(cfg)
                gen = torch.Generator().manual_seed(1)  # same weights for every backend, no reference copy
                with torch.no_grad():
                    for prm in block.parameters():
                        prm.copy_(torch.randn(prm.shape, generator=gen) * 0.02)
                block = block.to(dtype)
                for prm in block.parameters():
                    prm.requires_grad_(not frozen)  # frozen = LoRA on attention only: backward needs dX alone

                def run():
                    x = x0.detach().requires_grad_(True)
                    out = block(x)
                    out = out[0] if isinstance(out, tuple) else out
                    out.float().square().mean().backward()
                    return out.detach(), x.grad

                key = f"{name}/{str(dtype).split('.')[-1]}/{'frozen' if frozen else 'trainable'}/{impl}"
                # batched_mm gathers one weight copy per routed token: tokens * top_k * (gate_up + down) weights.
                gathered = a.tokens * kw["num_experts_per_tok"] * 3 * kw["hidden_size"] * kw["moe_intermediate_size"]
                gathered_gb = gathered * torch.finfo(dtype).bits / 8 / 2**30
                if impl == "batched_mm" and gathered_gb > 2:
                    res["moe"][key] = {"skipped": f"would gather {gathered_gb:.1f} GB of expert weights"}
                    print(key, res["moe"][key], flush=True)
                    del block
                    continue
                try:
                    out, grad = run()
                    entry = bench(run)
                    if out_ref is None:
                        out_ref, grad_ref = out, grad
                    else:
                        entry["out_rel_vs_eager"] = rel(out, out_ref)
                        entry["dx_rel_vs_eager"] = rel(grad, grad_ref)
                except Exception as e:  # noqa: BLE001
                    entry = {"error": repr(e)[:300]}
                res["moe"][key] = entry
                print(key, entry, flush=True)
                del block

if not a.skip_loss:
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy, cpu_chunked_log_probs
    import torch._dynamo
    V, H, N = 151936, 1024, 512
    for dtype in (torch.float32, torch.bfloat16):
        w = (torch.randn(V, H) * 0.02).to(dtype)
        h0 = torch.randn(1, N, H).to(dtype)
        labels = torch.randint(0, V, (1, N))
        labels[:, : N // 4] = -100

        def ce(fn):
            def run():
                h = h0.detach().requires_grad_(True)
                loss = fn(h, w, labels)
                loss.backward()
                return loss.detach(), h.grad
            return run

        def lp(fn):
            def run():
                h = h0[0].detach().requires_grad_(True)
                out = fn(h, w, None, labels[0].clamp_min(0), 1.0, 128)
                lpv = out[0] if isinstance(out, (tuple, list)) else out
                lpv.sum().backward()
                return lpv.detach(), h.grad
            return run

        for name, maker, fn in (("chunked_ce", ce, chunked_linear_cross_entropy),
                                ("fused_head_logprobs", lp, cpu_chunked_log_probs)):
            dt = str(dtype).split(".")[-1]
            eager = maker(fn)
            out_e, g_e = eager()
            entry = {"eager": bench(eager)}
            torch._dynamo.reset()
            try:
                explain = torch._dynamo.explain(maker(fn))()
                entry["graph_breaks"] = explain.graph_break_count
                entry["graphs"] = explain.graph_count
                entry["break_reasons"] = [str(b.reason)[:200] for b in explain.break_reasons][:4]
            except Exception as e:  # noqa: BLE001
                entry["explain_error"] = repr(e)[:300]
            torch._dynamo.reset()
            try:
                cfn = torch.compile(fn, dynamic=False)
                comp = maker(cfn)
                t0 = time.perf_counter()
                out_c, g_c = comp()
                entry["compile_first_call_s"] = round(time.perf_counter() - t0, 2)
                entry["compiled"] = bench(comp)
                entry["out_rel"] = rel(out_c, out_e)
                entry["grad_rel"] = rel(g_c, g_e)
            except Exception as e:  # noqa: BLE001
                entry["compile_error"] = repr(e)[:400]
            res["loss"][f"{name}/{dt}"] = entry
            print(name, dt, entry, flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2, default=str)
print("JOB_RESULT " + json.dumps({"job": "cpu_moe_loss_bench", "passed": True}))
sys.exit(0)
