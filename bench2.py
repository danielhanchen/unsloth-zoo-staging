# In-process interleaved A/B: base vs head fused scope (and native) on one loaded model.
import sys, json, time, importlib.util, contextlib, statistics as st
model_id, out = sys.argv[1], sys.argv[2]
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
def mod(name, zoo):
    spec = importlib.util.spec_from_file_location(name, zoo + "/unsloth_zoo/mlx/inference.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
arms = {"base": mod("zbase", "base"), "head": mod("zhead", "head")}
model, processor = load(model_id)
prompt = apply_chat_template(processor, model.config, "Explain in detail how a transformer language model generates text, step by step.", num_images=0)
def scope(arm):
    return contextlib.nullcontext() if arm == "native" else arms[arm].fused_decode_conv_silu(model)
res = {"model": model_id, "gen": {a: [] for a in ("base", "head", "native")}, "layer_us": {a: [] for a in ("base", "head", "native")}}
texts = {}
for a in ("base", "head", "native"):
    with scope(a):
        texts[a] = generate(model, processor, prompt, max_tokens=64, temperature=0.0, verbose=False).text
order = ["base", "head", "native", "native", "head", "base"]
for rep in range(5):
    for a in order:
        with scope(a):
            r = generate(model, processor, prompt, max_tokens=128, temperature=0.0, verbose=False)
        res["gen"][a].append(r.generation_tps)
# Layer microbench: one real GatedDeltaNet layer, decode step with cache, timed per arm in interleaved blocks.
from mlx_vlm.models.qwen3_5 import language as native
layer = next(m for _, m in model.named_modules() if isinstance(m, native.Qwen3_5GatedDeltaNet))
hidden = layer.in_proj_qkv.weight.shape[-1] if not hasattr(layer.in_proj_qkv, "scales") else layer.in_proj_z.weight.shape[-1] * (32 // layer.in_proj_z.bits if hasattr(layer.in_proj_z, "bits") else 1)
x = mx.random.normal((1, 1, model.config.text_config.hidden_size if hasattr(model.config, "text_config") else hidden)).astype(mx.bfloat16)
for blk in range(8):
    for a in (["base", "head", "native"] if blk % 2 == 0 else ["native", "head", "base"]):
        with scope(a):
            cache = native.ArraysCache(size=2)
            for _ in range(20):
                o = layer(x, cache=cache); mx.eval(o, cache.state)
            t = time.perf_counter()
            for _ in range(300):
                o = layer(x, cache=cache); mx.eval(o, cache.state)
            res["layer_us"][a].append((time.perf_counter() - t) / 300 * 1e6)
res["text_eq_native"] = {a: texts[a] == texts["native"] for a in texts}
summ = {k: {a: (round(st.median(v), 2), round(min(v), 2), round(max(v), 2)) for a, v in res[k].items()} for k in ("gen", "layer_us")}
print(json.dumps({"model": model_id, "summary": summ, "text_eq_native": res["text_eq_native"]}))
open(out, "a").write(json.dumps(res) + "\n")
