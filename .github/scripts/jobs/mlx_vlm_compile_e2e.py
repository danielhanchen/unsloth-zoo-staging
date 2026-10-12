"""Apple Silicon job: real-tower mx.compile training parity for glm4v / glm4v_moe / glm5_next /
qwen3_omni_moe (zoo#1705, #1706). Tiny random weights from the real configs.
"""

import json, os, sys, copy
from types import SimpleNamespace as NS

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
ARCH = os.environ.get("E2E_ARCH", "")
GRIDS = ((1, 4, 4), (1, 4, 6))
GLM5_NEXT_CFG = json.loads('{"architectures": ["Glm5NextForConditionalGeneration"], "dtype": "bfloat16", "image_end_token_id": 154831, "image_start_token_id": 154830, "image_token_id": 154854, "model_type": "glm5_next", "text_config": {"attention_bias": false, "attention_dropout": 0.0, "bos_token_id": null, "dtype": "float32", "eos_token_id": null, "hc_eps": 1e-06, "hc_mult": 4, "hc_sinkhorn_iters": 20, "head_dim": 0, "hidden_act": "silu", "hidden_size": 64, "id2label": {"0": "LABEL_0"}, "index_head_dim": 16, "index_kpool": 16, "index_kpool_always_select_tail": true, "index_n_heads": 4, "index_topk": 2048, "indexer_types": ["full", "full"], "initializer_range": 0.02, "intermediate_size": 128, "kv_lora_rank": 32, "layer_types": ["linear_attention", "deepseek_sparse_attention"], "linear_conv_kernel_dim": 4, "linear_head_dim": 16, "linear_lower_bound": -5.0, "linear_num_heads": 4, "max_position_embeddings": 1048576, "mlp_layer_types": ["dense", "sparse"], "model_type": "glm5_next_text", "moe_intermediate_size": 64, "n_group": 1, "n_routed_experts": 4, "n_shared_experts": 1, "norm_topk_prob": true, "num_attention_heads": 4, "num_experts_per_tok": 2, "num_hidden_layers": 2, "num_key_value_heads": 4, "output_router_logits": false, "pad_token_id": 154820, "q_lora_rank": 32, "qk_head_dim": 16, "qk_nope_head_dim": 16, "qk_rope_head_dim": 0, "rms_norm_eps": 1e-05, "routed_scaling_factor": 2.5, "router_aux_loss_coef": 0.001, "swiglu_limit": 10.0, "tie_word_embeddings": false, "topk_group": 1, "use_cache": true, "v_head_dim": 16, "vocab_size": 154880}, "tie_word_embeddings": false, "transformers_version": "5.17.0", "video_end_token_id": 154833, "video_start_token_id": 154832, "video_token_id": 154855, "vision_config": {"attention_bias": true, "attention_dropout": 0.0, "depth": 2, "dtype": "float32", "hidden_act": "silu", "hidden_size": 64, "image_size": 336, "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 128, "model_type": "glm5_next_vision", "num_heads": 4, "out_hidden_size": 64, "patch_size": 14, "projection_intermediate_size": 640, "rms_norm_eps": 1e-05, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"}, "spatial_merge_size": 2, "swiglu_limit": 10.0, "temporal_patch_size": 2}}')
REPOS = {"glm4v": "zai-org/GLM-4.1V-9B-Thinking", "glm4v_moe": "zai-org/GLM-4.5V", "qwen3_omni_moe": "Qwen/Qwen3-Omni-30B-A3B-Instruct"}


def shrink_text(t, moe=False):
    t.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
             num_key_value_heads=2, max_position_embeddings=4096)
    if moe:
        t.update(moe_intermediate_size=32, n_routed_experts=4, num_experts=4, num_experts_per_tok=2,
                 first_k_dense_replace=1, n_group=1, topk_group=1, n_shared_experts=1)
    return t


def shrink_vision(v, out):
    v.update(depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=out)
    return v


def build_config():
    if ARCH == "glm5_next":
        return copy.deepcopy(GLM5_NEXT_CFG)
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(REPOS[ARCH], "config.json")))
    if ARCH == "glm4v":
        shrink_text(cfg["text_config"])
        cfg["text_config"].update(head_dim=32, rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]})
        cfg["text_config"].pop("rope_parameters", None)
        shrink_vision(cfg["vision_config"], 64)
    elif ARCH == "glm4v_moe":
        t = shrink_text(cfg["text_config"], moe=True)
        t.update(head_dim=32)
        rp = t.get("rope_parameters") or t.get("rope_scaling") or {}
        rp = dict(rp, mrope_section=[2, 3, 3])
        if "rope_parameters" in t:
            t["rope_parameters"] = rp
        t["rope_scaling"] = rp
        shrink_vision(cfg["vision_config"], 64)
    elif ARCH == "qwen3_omni_moe":
        cfg["enable_audio_output"] = False
        th = cfg["thinker_config"]
        t = shrink_text(th["text_config"], moe=True)
        t.update(head_dim=16, decoder_sparse_step=1, mlp_only_layers=[])
        rs = dict(t.get("rope_scaling") or {}, mrope_section=[2, 3, 3])
        t["rope_scaling"] = rs
        v = th["vision_config"]
        v.update(depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64,
                 deepstack_visual_indexes=[0, 1])
        a = th["audio_config"]
        a.update(d_model=64, encoder_layers=1, encoder_attention_heads=4, encoder_ffn_dim=128,
                 output_dim=64, downsample_hidden_size=32, num_hidden_layers=1)
    return cfg


def main():
    from mlx_vlm.utils import get_model_and_args
    cfg = build_config()
    model_module, _ = get_model_and_args(cfg)
    config = model_module.ModelConfig.from_dict(cfg)
    for key in ("text_config", "vision_config", "thinker_config"):
        sub = getattr(config, key, None)
        if isinstance(sub, dict):
            cls = {"text_config": "TextConfig", "vision_config": "VisionConfig",
                   "thinker_config": "ThinkerConfig"}[key]
            if hasattr(model_module, cls):
                setattr(config, key, getattr(model_module, cls).from_dict(sub))
    mx.random.seed(0)
    model = model_module.Model(config)
    # Small random weights so logits stay finite and bf16-free (fp32 everywhere).
    params = tree_flatten(model.parameters())
    mx.random.seed(1)
    model.update(__import__("mlx.utils", fromlist=["tree_unflatten"]).tree_unflatten(
        [(k, (mx.random.normal(v.shape) * 0.02).astype(mx.float32)) for k, v in params]))
    mx.eval(model.parameters())
    model.train()
    # LoRA-like trainable set: linear weights in attention / vision / mlp, never routers.
    model.freeze()
    for path, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(x in path for x in ("gate.", "router", "shared_expert_gate")) \
                and not path.endswith(".gate") and any(x in path for x in ("self_attn", "attn", "vision", "visual", "mlp")):
            mod.unfreeze(keys=["weight"])

    mc = cfg.get("thinker_config", cfg)
    vc = mc["vision_config"]
    merge = vc.get("spatial_merge_size", 2)
    patch = vc["patch_size"]; tps = vc.get("temporal_patch_size", 2); ch = vc.get("in_channels", 3)
    image_id = mc["image_token_id"]
    start_id = mc.get("vision_start_token_id") or mc.get("image_start_token_id")
    end_id = mc.get("vision_end_token_id") or mc.get("image_end_token_id")
    n_patches = sum(t * h * w for t, h, w in GRIDS)
    rng = np.random.default_rng(0)
    pixel_values = mx.array(rng.standard_normal((n_patches, ch * tps * patch * patch)).astype(np.float32))
    row = [11, 12]
    for t, h, w in GRIDS:
        row += [start_id] + [image_id] * (t * h * w // merge ** 2) + [end_id, 13]
    row += [14, 15, 16]
    input_ids = np.array([row], np.int32)
    attention_mask = np.ones_like(input_ids)

    def ce(logits, ids):
        logits = logits[:, :-1].astype(mx.float32)
        tgt = ids[:, 1:]
        return nn.losses.cross_entropy(logits, tgt, reduction="mean")

    ids_mx = mx.array(input_ids)
    out = {"arch": ARCH, "grids": GRIDS, "seq_len": int(input_ids.shape[1])}

    from unsloth_zoo.mlx import utils as _mu0
    _mu0.acquire_mlx_training_patches()  # generic: detach integer index grads (not part of the PR)
    # Arm 1: upstream (no patches).
    def up_loss(m):
        o = m(ids_mx, pixel_values=pixel_values, mask=None, image_grid_thw=mx.array(GRIDS, dtype=mx.int32))
        logits = o.logits if hasattr(o, "logits") else o
        return ce(logits, ids_mx)
    try:
        l_up, g_up = nn.value_and_grad(model, up_loss)(model)
        mx.eval(l_up, g_up)
        out["upstream_loss"] = float(l_up)
    except Exception as e:
        out["upstream_error"] = f"{type(e).__name__}: {e}"[:300]
        g_up = None

    # Inference parity: eval-mode upstream logits, compared after the patches are installed.
    def eval_logits(m):
        m.eval()
        if hasattr(m, "language_model"):
            m.language_model._position_ids = None
            m.language_model._rope_deltas = None
        o = m(ids_mx, pixel_values=pixel_values, mask=None, image_grid_thw=mx.array(GRIDS, dtype=mx.int32))
        lg = o.logits if hasattr(o, "logits") else o
        mx.eval(lg); m.train()
        return lg
    try:
        lg_up = eval_logits(model)
    except Exception as e:
        lg_up = None; out["eval_upstream_error"] = f"{type(e).__name__}: {e}"[:300]

    # Arm 2/3: zoo path.
    import unsloth_zoo.mlx.compile as mcmod
    from unsloth_zoo.mlx import utils as mu
    out["zoo_file"] = mcmod.__file__
    mcmod.install_mlx_compile_patches()
    qual = mcmod.build_compile_qualifications().get(ARCH)
    out["training_compile_qualified"] = bool(getattr(qual, "training_compile", False))
    if lg_up is not None:
        try:
            out["eval_logits_max_abs_diff"] = float(mx.max(mx.abs(eval_logits(model) - lg_up)))
        except Exception as e:
            out["eval_patched_error"] = f"{type(e).__name__}: {e}"[:300]
    model._unsloth_compile_decision = NS(enabled=True)
    batch = {"input_ids": mx.array(input_ids), "attention_mask": mx.array(attention_mask),
             "pixel_values": pixel_values, "image_grid_thw": np.array(GRIDS, np.int64)}
    batch = mu._prepare_vlm_batch_for_compile(batch, cfg)
    batch = {k: (mx.array(v) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
    out["batch_keys"] = sorted(batch)

    def zoo_loss(m, b):
        return ce(mu._vlm_forward_logits(m, b), b["input_ids"])

    vg = nn.value_and_grad(model, zoo_loss)
    try:
        l_pe, g_pe = vg(model, batch)
        mx.eval(l_pe, g_pe)
        out["patched_eager_loss"] = float(l_pe)
    except Exception as e:
        out["patched_eager_error"] = f"{type(e).__name__}: {e}"[:400]
        print("E2E_RESULT " + json.dumps(out)); return

    state = [model.state]
    def step(b):
        return vg(model, b)
    try:
        cstep = mx.compile(step, inputs=state, outputs=state)
        losses = []
        for _ in range(2):
            l_c, g_c = cstep(batch)
            mx.eval(l_c, g_c)
            losses.append(float(l_c))
        out["compiled_losses"] = losses
    except Exception as e:
        out["compiled_error"] = f"{type(e).__name__}: {e}"[:400]
        g_c = None

    def gdiff(a, b):
        fa, fb = dict(tree_flatten(a)), dict(tree_flatten(b))
        keys = set(fa) & set(fb)
        worst = max((float(mx.max(mx.abs(fa[k] - fb[k]))) for k in keys), default=None)
        nz = sum(1 for k in keys if float(mx.max(mx.abs(fb[k]))) > 0)
        return {"max_abs": worst, "n": len(keys), "nonzero_in_second": nz}

    if g_up is not None:
        out["grad_upstream_vs_patched_eager"] = gdiff(g_up, g_pe)
    if g_c is not None:
        out["grad_patched_eager_vs_compiled"] = gdiff(g_pe, g_c)
        out["compiled_bitwise_eager_loss"] = out["compiled_losses"][0] == out["patched_eager_loss"]
    vis = {k: v for k, v in tree_flatten(g_pe) if "vision" in k}
    out["vision_grad_nonzero"] = sum(1 for v in vis.values() if float(mx.max(mx.abs(v))) > 0)
    out["vision_grad_total"] = len(vis)
    print("E2E_RESULT " + json.dumps(out))


def driver():
    import argparse, subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="glm4v,glm4v_moe,glm5_next,qwen3_omni_moe")
    ap.add_argument("--out", default="mlx_vlm_compile_e2e.json")
    a = ap.parse_args()
    results, ok = {}, True
    for arch in a.archs.split(","):
        p = subprocess.run([sys.executable, os.path.abspath(__file__)], env={**os.environ, "E2E_ARCH": arch},
                           capture_output=True, text=True, timeout=1800)
        line = [l for l in p.stdout.splitlines() if l.startswith("E2E_RESULT ")]
        r = json.loads(line[-1][11:]) if line else {"error": (p.stderr or p.stdout)[-1500:]}
        r["device"] = str(mx.default_device())
        r["ok"] = bool(line) and "compiled_error" not in r and "patched_eager_error" not in r \
            and r.get("compiled_bitwise_eager_loss") is True \
            and r.get("grad_patched_eager_vs_compiled", {}).get("max_abs") == 0.0 \
            and r.get("training_compile_qualified") is True \
            and r.get("eval_logits_max_abs_diff") == 0.0
        ok &= r["ok"]
        results[arch] = r
        print(arch, "OK" if r["ok"] else "FAIL", json.dumps(r)[:1500], flush=True)
    json.dump({"job": "mlx_vlm_compile_e2e", "passed": ok, "results": results}, open(a.out, "w"), indent=1)
    print("JOB_RESULT " + json.dumps({"passed": ok, "archs": {k: v["ok"] for k, v in results.items()}}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main() if os.environ.get("E2E_ARCH") else driver()
