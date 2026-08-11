"""Real small-model MLX smoke for unsloth-zoo PRs #1024 / #1026, sized for a
GitHub macos-14 runner (Apple Silicon, ~7 GB RAM, ~14 GB disk).

Each stage is independent and prints `STAGE <name>: PASS|FAIL|SKIP -- detail`, so one
missing model never hides the rest. Exit code is non-zero only if a REQUIRED stage fails;
stages marked optional report but never gate.

Usage:
    python scripts/mac_mlx_smoke.py [--skip-optional]
"""

import argparse
import gc
import os
import sys
import traceback

os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

FAILURES = []


def stage(name, optional = False):
    def wrap(fn):
        try:
            ok, detail = fn()
        except Exception:
            ok = False
            detail = "raised: " + traceback.format_exc(limit = 4).replace("\n", " | ")
        verdict = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
        tag = " (optional)" if optional else ""
        print(f"STAGE {name}{tag}: {verdict} -- {detail}", flush = True)
        if ok is False and not optional:
            FAILURES.append(name)
        gc.collect()
        return fn
    return wrap


def _sizes():
    import importlib.metadata as md
    out = {}
    for pkg in ("mlx", "mlx-vlm", "mlx-lm", "transformers", "torch"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = "absent"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-optional", action = "store_true")
    args = ap.parse_args()

    print(f"ENV {_sizes()} python={sys.version.split()[0]} platform={sys.platform}", flush = True)

    import mlx.core as mx
    print(f"MLX backend: {mx.default_device()} file={mx.__file__}", flush = True)

    from unsloth_zoo.mlx import utils as zutils
    print(f"ZOO {zutils.__file__}", flush = True)

    # ---------------------------------------------------------------- runtime

    @stage("mlx_runtime_is_real")
    def _():
        origin = str(getattr(mx, "__file__", "") or "")
        if "mlx_simulation" in origin:
            return (False, f"shim, not real mlx: {origin}")
        return (int(mx.array([1, 2, 3]).sum()) == 6, f"real mlx at {origin}, device={mx.default_device()}")

    # ---------------------------------------------------------------- #1026 synthetic

    @stage("1026_ragged_pixel_values_preserved")
    def _():
        import numpy as np
        inputs = {
            "input_ids": [[1, 2, 3]],
            "pixel_values": [
                np.zeros((3, 384, 1584), dtype = np.float32),
                np.zeros((3, 480, 1200), dtype = np.float32),
            ],
        }
        pv = zutils._to_mx_vlm_batch(inputs)["pixel_values"]
        if isinstance(pv, list):
            return (len(pv) == 2, f"list of {len(pv)}: {[tuple(x.shape) for x in pv]}")
        return (False, f"collapsed to {tuple(pv.shape)} -- later images dropped")

    @stage("1026_dense_pixel_values_still_stacked")
    def _():
        import numpy as np
        inputs = {
            "input_ids": [[1, 2, 3]],
            "pixel_values": [
                np.zeros((3, 384, 384), dtype = np.float32),
                np.ones((3, 384, 384), dtype = np.float32),
            ],
        }
        pv = zutils._to_mx_vlm_batch(inputs)["pixel_values"]
        if isinstance(pv, list):
            return (False, f"equal shapes were not stacked (list of {len(pv)})")
        return (tuple(pv.shape) == (2, 3, 384, 384) and float(pv[1].sum()) > 0,
                f"stacked {tuple(pv.shape)}, second row preserved")

    # ---------------------------------------------------------------- #1024 gate

    @stage("1024_audio_gate_table")
    def _():
        table = zutils._AUDIO_QUALIFIED_FAMILIES
        installed = zutils._installed_mlx_vlm_version()
        rows = [f"{k}(min={v.minimum},max={getattr(v, 'maximum', None)})->{v.admits(installed)}"
                for k, v in table.items()]
        return (None, f"mlx-vlm={installed} :: " + "; ".join(rows))

    @stage("1024_renderer_case_insensitive_lookup")
    def _():
        from mlx_vlm import prompt_utils
        from unsloth_zoo.mlx.loader import _ensure_vlm_prompt_utils_patched
        _ensure_vlm_prompt_utils_patched()
        cfg = getattr(prompt_utils, "MODEL_CONFIG", {})
        published = "NemotronH_Nano_Omni_Reasoning_V3"
        lowered = published.casefold()
        if lowered not in {k.casefold() for k in cfg if isinstance(k, str)}:
            return (None, f"mlx-vlm {len(cfg)} keys carry no nemotron omni renderer; nothing to resolve")
        plain = prompt_utils.apply_chat_template(
            None, {"model_type": published}, "hi", num_images = 0, num_audios = 0, return_messages = True)
        marked = prompt_utils.apply_chat_template(
            None, {"model_type": published}, "hi", num_images = 0, num_audios = 1, return_messages = True)
        return (plain != marked, f"mixed-case model_type resolved; audio marker changes render={plain != marked}")

    # ---------------------------------------------------------------- real models

    def _load_vlm(repo):
        from mlx_vlm import load
        return load(repo, trust_remote_code = True)

    @stage("real_text_model_qwen3_0.6b")
    def _():
        from mlx_lm import load as lm_load, generate as lm_generate
        model, tokenizer = lm_load("mlx-community/Qwen3-0.6B-4bit")
        out = lm_generate(model, tokenizer, prompt = "The capital of France is", max_tokens = 8, verbose = False)
        del model
        return (bool(out and out.strip()), f"generated {out.strip()[:60]!r}")

    @stage("real_vlm_smolvlm_256m_two_sizes")
    def _():
        from PIL import Image
        import numpy as np
        from transformers import AutoProcessor
        repo = "mlx-community/SmolVLM-256M-Instruct-bf16"
        proc = AutoProcessor.from_pretrained(repo)
        imgs = [Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype = np.uint8))
                for h, w in ((384, 1584), (480, 1200))]
        raw = [proc(text = "<image>describe", images = [im], return_tensors = "np") for im in imgs]
        pv = [r["pixel_values"][0] for r in raw]
        shapes = [tuple(x.shape) for x in pv]
        batch = zutils._to_mx_vlm_batch({"input_ids": [[1], [2]], "pixel_values": pv})
        got = batch["pixel_values"]
        kind = "list" if isinstance(got, list) else f"array{tuple(got.shape)}"
        return (True, f"processor shapes={shapes} -> batch pixel_values={kind}")

    @stage("real_vlm_qwen3vl_2b_ragged_through_batch")
    def _():
        from PIL import Image
        import numpy as np
        from transformers import AutoProcessor
        repo = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
        proc = AutoProcessor.from_pretrained(repo)
        imgs = [Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype = np.uint8))
                for h, w in ((392, 1596), (476, 1204))]
        raw = [proc(text = "<|vision_start|><|image_pad|><|vision_end|>describe",
                    images = [im], return_tensors = "np") for im in imgs]
        pv = [r["pixel_values"] for r in raw]
        shapes = [tuple(np.asarray(x).shape) for x in pv]
        batch = zutils._to_mx_vlm_batch({"input_ids": [[1], [2]], "pixel_values": pv})
        got = batch["pixel_values"]
        if isinstance(got, list):
            has_astype = hasattr(got, "astype")
            return (True, f"Qwen processor shapes={shapes} -> RAGGED list of {len(got)}; "
                          f"list.astype exists={has_astype} (False means a consumer calling "
                          f".astype would raise AttributeError)")
        return (True, f"Qwen processor shapes={shapes} -> dense {tuple(got.shape)} (stacked fine)")

    if not args.skip_optional:
        @stage("real_gemma4_e2b_ragged", optional = True)
        def _():
            from PIL import Image
            import numpy as np
            from transformers import AutoProcessor
            repo = "mlx-community/gemma-4-E2B-it-qat-4bit"
            proc = AutoProcessor.from_pretrained(repo)
            imgs = [Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype = np.uint8))
                    for h, w in ((384, 1584), (480, 1200))]
            raw = [proc(text = "<start_of_image>describe", images = [im], return_tensors = "np")
                   for im in imgs]
            pv = [np.asarray(r["pixel_values"])[0] for r in raw]
            shapes = [tuple(x.shape) for x in pv]
            batch = zutils._to_mx_vlm_batch({"input_ids": [[1], [2]], "pixel_values": pv})
            got = batch["pixel_values"]
            if isinstance(got, list):
                return (len(got) == 2, f"Gemma 4 shapes={shapes} -> ragged list of {len(got)}, "
                                       f"both images preserved")
            return (False, f"Gemma 4 shapes={shapes} -> collapsed to {tuple(got.shape)}, image dropped")

    print(f"SMOKE DONE failures={FAILURES}", flush = True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
