"""unsloth-zoo#2230 on Apple Silicon: tiny LFM3-Audio / LFM3-Omni load + 2 audio LoRA steps, base vs head zoo."""
import argparse, json, os, subprocess, sys, textwrap, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--head", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
W = Path.home() / "pr2230"
W.mkdir(exist_ok=True)
res = {"job": "mlx_pr2230_lfm3", "cells": {}, "checks": {}}

BUILD = r'''
import json, sys
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
kind, d = sys.argv[1], Path(sys.argv[2]); cfg = json.loads((d / "config.json").read_text())
def shrink(c, layers):
    c.update(hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1, head_dim=32, block_multiple_of=32)
    if "num_hidden_layers" in c: c["num_hidden_layers"] = layers
    if isinstance(c.get("layer_types"), list): c["layer_types"] = c["layer_types"][:layers]
if kind == "audio":
    shrink(cfg["lfm"], 2); shrink(cfg["encoder"]["lfm"], 1); shrink(cfg["depthformer"], 1)
    cfg["encoder"].update(dim_conv=32, dim_out=64)
    from mlx_vlm.models.lfm3_audio import Model, ModelConfig
else:
    shrink(cfg["text_config"], 2)
    vc = cfg["vision_config"]
    vc.update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, layer_types=vc["layer_types"][:1])
    cfg["projector_hidden_size"] = 64
    ae = cfg["audio_encoder_config"]
    if "text_config" in ae: shrink(ae["text_config"], 1)
    ae.update({k: v for k, v in dict(conv_channels=32, hidden_size=64).items() if k in ae})
    shrink(cfg["depth_config"], 1)
    from mlx_vlm.models.lfm3_omni import Model, ModelConfig
mx.random.seed(0)
model = Model(ModelConfig.from_dict(cfg))
w = {k: (v * 0.02 if v.ndim > 1 else v) for k, v in tree_flatten(model.parameters())}
mx.save_safetensors(str(d / "model.safetensors"), w)
(d / "config.json").write_text(json.dumps(cfg))
'''

PROBE = r'''
import json, os, sys, traceback
import numpy as np
ckpt, out = sys.argv[1], sys.argv[2]
r = {}
try:
    import mlx.core as mx, unsloth_zoo
    r["zoo"], r["device"] = unsloth_zoo.__file__, str(mx.default_device())
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    model, proc = FastMLXModel.from_pretrained(ckpt, max_seq_length=256)
    r["processor"] = type(proc).__name__
    model = FastMLXModel.get_peft_model(model, r=4, lora_alpha=8, lora_dropout=0)
    rows = []
    for i, hz in enumerate((220, 440, 880, 1760)):
        t = np.arange(int(16000 * (0.5 + 0.25 * i))) / 16000
        rows.append({"messages": [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Transcribe."}]},
                                  {"role": "assistant", "content": [{"type": "text", "text": f"tone {hz}"}]}],
                     "audio": [{"array": np.sin(2 * np.pi * hz * t).astype(np.float32), "sampling_rate": 16000}]})
    tr = MLXTrainer(model=model, tokenizer=proc, train_dataset=rows, args=MLXTrainingConfig(
        per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=2, learning_rate=1e-3,
        logging_steps=1, output_dir=os.path.dirname(out), seed=3407, report_to="none"))
    tr.train()
    r["losses"] = [float(x) for x in tr._train_loss_history]
    r["ok"] = len(r["losses"]) == 2 and all(np.isfinite(r["losses"]))
except Exception as e:
    r["ok"], r["error"], r["tb"] = False, f"{type(e).__name__}: {e}"[:600], traceback.format_exc()[-1500:]
json.dump(r, open(out, "w"))
'''

try:
    from huggingface_hub import snapshot_download
    (W / "build.py").write_text(BUILD)
    (W / "probe.py").write_text(PROBE)
    subprocess.run(f"{sys.executable} -m pip install -q -U 'mlx-vlm==0.7.11'", shell=True, check=True)
    for kind, repo in (("audio", "LiquidAI/LFM3-Audio-2B"), ("omni", "LiquidAI/LFM3-Omni-2B")):
        d = W / f"tiny_{kind}"
        snapshot_download(repo, local_dir=str(d), allow_patterns=["*.json", "*.jinja", "tokenizer*"])
        subprocess.run([sys.executable, str(W / "build.py"), kind, str(d)], check=True)
    for arm, sha in (("base", a.base), ("head", a.head)):
        src = W / f"zoo_{arm}"
        subprocess.run(f"rm -rf {src} && git init -q {src} && git -C {src} fetch -q --depth 1 https://github.com/unslothai/unsloth-zoo {sha} && git -C {src} checkout -q FETCH_HEAD", shell=True, check=True)
        for kind in ("audio", "omni"):
            o = W / f"{arm}_{kind}.json"
            env = dict(os.environ, PYTHONPATH=str(src))
            env.pop("UNSLOTH_IS_PRESENT", None)
            t = time.time()
            subprocess.run([sys.executable, str(W / "probe.py"), str(W / f"tiny_{kind}"), str(o)], env=env, timeout=1200)
            cell = json.loads(o.read_text()) if o.exists() else {"ok": False, "error": "no output"}
            cell["s"] = round(time.time() - t, 1)
            res["cells"][f"{arm}_{kind}"] = {k: v for k, v in cell.items() if k != "tb"} | ({"tb": cell["tb"]} if not cell.get("ok") and "tb" in cell else {})
    c = res["cells"]
    res["checks"] = {
        "head_audio_trains": {"ok": c["head_audio"].get("ok", False)},
        "head_omni_trains": {"ok": c["head_omni"].get("ok", False)},
        "base_audio_fails": {"ok": not c["base_audio"].get("ok", True)},
        "base_omni_fails": {"ok": not c["base_omni"].get("ok", True)},
    }
    res["passed"] = all(v["ok"] for v in res["checks"].values())
except Exception as e:
    res["passed"], res["error"] = False, f"{type(e).__name__}: {e}"[:2000]
Path(a.out).write_text(json.dumps(res, indent=1, default=str))
print("JOB_RESULT " + json.dumps({k: res.get(k) for k in ("passed", "checks", "error")}))
sys.exit(0 if res.get("passed") else 1)
