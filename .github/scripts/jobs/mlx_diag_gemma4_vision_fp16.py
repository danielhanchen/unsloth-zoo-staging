"""Gemma 4 26B-A4B vision tower in float16 on Apple Silicon (unsloth#15473).

    python jobs/mlx_diag_gemma4_vision_fp16.py [--model M] [--side 384,768] [--out j.json]

Downloads only the vision shard of the 4-bit MLX checkpoint (~4.7 GB, the tower is unquantized
bf16), then runs vision_tower + embed_vision on the issue's image (white canvas, red bar, blue
disc) through mlx-vlm's own VisionModel.__call__ in float32, stock float16, and float16 with
unsloth-zoo's `_fix_gemma4_vision_pooler_fp32`. Checks: the fixed float16 soft tokens are finite
and match float32 (cosine >= 0.999). Stock float16 overflowing is recorded, not checked (it
depends on the image). Prints `JOB_RESULT {...}`; exit 0 pass, 1 fail, 3 not Apple Silicon.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import sys
import time
import traceback


def _ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None


def _image(side):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (side, side), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(side * 0.19), int(side * 0.28), int(side * 0.70)], fill = (228, 52, 38))
    r = int(side * 0.25)
    cx, cy = int(side * 0.88), int(side * 0.44)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill = (0, 0, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "mlx-community/gemma-4-26b-a4b-it-4bit")
    ap.add_argument("--side", default = "384,768")
    ap.add_argument("--out", default = "mlx_diag_gemma4_vision_fp16.json")
    ap.add_argument("--tiny", action = "store_true", help = "ignored (staging_ci)")
    a = ap.parse_args()
    res = {"job": "mlx_diag_gemma4_vision_fp16", "model": a.model, "backend": "mlx", "device": platform.machine(),
           "versions": {p: _ver(p) for p in ("mlx", "mlx-metal", "mlx-vlm", "unsloth_zoo")},
           "runs": [], "checks": {}, "passed": False, "error": None, "skipped": None}

    def finish(code):
        res["passed"] = code == 0
        with open(a.out, "w") as f:
            json.dump(res, f, indent = 1, default = str)
        brief = {k: res[k] for k in ("job", "model", "device", "passed", "skipped")}
        brief["failed_checks"] = [k for k, c in res["checks"].items() if not c["ok"]]
        print("JOB_RESULT " + json.dumps(brief), flush = True)
        sys.exit(code)

    if sys.platform != "darwin" or platform.machine() != "arm64":
        res["skipped"] = "needs Apple Silicon"
        finish(3)
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from huggingface_hub import hf_hub_download
        from mlx.utils import tree_map
        from mlx_vlm.models.gemma4 import vision
        from mlx_vlm.models.gemma4.config import VisionConfig
        from mlx_vlm.models.gemma4.gemma4 import MultimodalEmbedder

        res["device"] = str(mx.device_info().get("device_name", ""))
        if os.path.isdir(a.model):
            hf_hub_download = lambda repo, name: os.path.join(repo, name)  # noqa: E731
        cfg = json.load(open(hf_hub_download(a.model, "config.json")))
        index = json.load(open(hf_hub_download(a.model, "model.safetensors.index.json")))["weight_map"]
        shards = sorted({f for k, f in index.items() if k.startswith(("vision_tower.", "embed_vision."))})
        weights = {}
        for shard in shards:
            path = hf_hub_download(a.model, shard)
            # mx.load is lazy: only the vision tensors kept here are ever read from the shard.
            w = mx.load(path)
            weights.update({k: v for k, v in w.items() if k.startswith(("vision_tower.", "embed_vision."))})
            del w
        vcfg = VisionConfig.from_dict(cfg["vision_config"])
        res["use_clipped_linears"] = getattr(vcfg, "use_clipped_linears", None)
        if not getattr(vcfg, "use_clipped_linears", False):
            weights = {k: v for k, v in weights.items()
                       if not any(s in k for s in ("input_max", "input_min", "output_max", "output_min"))}

        def build(dtype):
            tower = vision.VisionModel(vcfg)
            embed = MultimodalEmbedder(vcfg.hidden_size, cfg["text_config"]["hidden_size"], eps = vcfg.rms_norm_eps)
            q = cfg.get("quantization", {})
            nn.quantize(embed, group_size = q.get("group_size", 64), bits = q.get("bits", 4))
            tower.load_weights([(k[len("vision_tower."):], v) for k, v in weights.items() if k.startswith("vision_tower.")])
            embed.load_weights([(k[len("embed_vision."):], v) for k, v in weights.items() if k.startswith("embed_vision.")])
            for m in (tower, embed):
                m.update(tree_map(lambda v: v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v, m.parameters()))
            return tower, embed

        def run(arm, dtype, side):
            tower, embed = build(dtype)
            img = np.asarray(_image(side), dtype = np.float32) / 255.0
            pv = mx.array(img.transpose(2, 0, 1)[None]).astype(dtype)
            t = time.time()
            out = tower(pv)
            feats = embed(out)
            mx.eval(out, feats)
            f = np.array(feats.astype(mx.float32))
            r = {"arm": arm, "side": side, "tower_out_dtype": str(out.dtype),
                 "tower_out_inf": int(np.isinf(np.array(out.astype(mx.float32))).sum()),
                 "soft_token_nan": int(np.isnan(f).sum()), "soft_tokens": int(f.shape[1]),
                 "s": round(time.time() - t, 2)}
            res["runs"].append(r)
            print("RUN " + json.dumps(r), flush = True)
            return f

        sides = [int(s) for s in a.side.split(",") if s]
        ref = {s: run("float32", mx.float32, s) for s in sides}
        stock = {s: run("float16_stock", mx.float16, s) for s in sides}
        from unsloth_zoo.mlx.loader import _fix_gemma4_vision_pooler_fp32

        res["checks"]["fix_installed"] = {"ok": bool(_fix_gemma4_vision_pooler_fp32()), "detail": ""}
        for s in sides:
            fixed = run("float16_fixed", mx.float16, s)
            x, y = ref[s].ravel(), fixed.ravel()
            finite = bool(np.isfinite(y).all())
            cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) if finite else None
            res["checks"][f"fixed_fp16_finite_{s}"] = {"ok": finite, "detail": f"stock NaN {int(np.isnan(stock[s]).sum())}"}
            res["checks"][f"fixed_fp16_matches_fp32_{s}"] = {"ok": cos is not None and cos >= 0.999, "detail": f"cosine {cos}"}
    except Exception as e:  # noqa: BLE001 - recorded, then a failed job
        res["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
        print(res["error"], file = sys.stderr)
        finish(1)
    finish(0 if res["checks"] and all(c["ok"] for c in res["checks"].values()) else 1)


if __name__ == "__main__":
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    main()
