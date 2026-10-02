# One measurement: python bench.py ZOO_DIR ARM MODEL OUT_JSONL  (ARM in native|fused)
import sys, json, time, importlib.util, contextlib
zoo, arm, model_id, out = sys.argv[1:5]
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
spec = importlib.util.spec_from_file_location("zinf", zoo + "/unsloth_zoo/mlx/inference.py")
I = importlib.util.module_from_spec(spec); spec.loader.exec_module(I)
kname = "_decode_conv" if hasattr(I, "_decode_conv") else "_decode_conv_silu"
calls = [0]; real = getattr(I, kname)
def counted(*a):
    calls[0] += 1
    return real(*a)
setattr(I, kname, counted)
model, processor = load(model_id)
prompt = apply_chat_template(processor, model.config, "Explain in detail how a transformer language model generates text, step by step.", num_images=0)
ctx = I.fused_decode_conv_silu(model) if arm == "fused" else contextlib.nullcontext()
res = {}
with ctx:
    patched = sorted({type(m).__name__ for _, m in model.named_modules() if "Fused" in type(m).__name__})
    generate(model, processor, prompt, max_tokens=16, temperature=0.0, verbose=False)  # warmup
    calls[0] = 0
    r = generate(model, processor, prompt, max_tokens=200, temperature=0.0, verbose=False)
res = dict(zoo=zoo, arm=arm, model=model_id, kernel=kname, calls=calls[0], patched=patched,
           gen_tps=r.generation_tps, prompt_tps=r.prompt_tps, peak_mem=r.peak_memory,
           text=r.text, metal=mx.metal.is_available())
print(json.dumps({k: v for k, v in res.items() if k != "text"}))
open(out, "a").write(json.dumps(res) + "\n")
