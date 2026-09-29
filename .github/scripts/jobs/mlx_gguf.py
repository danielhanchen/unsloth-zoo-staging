"""MLX GGUF export on Apple Silicon: real LoRA merge, real convert_hf_to_gguf, real llama-quantize.

    python jobs/mlx_gguf.py --tiny          # SmolLM-135M-Instruct-4bit
    python jobs/mlx_gguf.py --no-list       # scalar exports only (a tree without list support)

Exports QUANTS once as a list (one merge + convert) and once per type as scalars, then checks each
file's GGUF general.file_type, that the scratch BF16 intermediate is gone, and records both wall
times (list_export_s vs scalar_exports_s). llama.cpp is installed by the export itself if missing.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen3.5-2B-4bit"
TINY_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
QUANTS = ["q4_k_m", "q8_0", "f16"]
# llama.cpp LLAMA_FTYPE values written to general.file_type.
FTYPE = {"f32": 0, "f16": 1, "q8_0": 7, "q4_k_m": 15, "bf16": 32}


def _file_type(path):
    from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR

    gguf_py = os.path.join(LLAMA_CPP_DEFAULT_DIR, "gguf-py")
    if os.path.isdir(gguf_py) and gguf_py not in sys.path:
        sys.path.insert(0, gguf_py)
    from gguf import GGUFReader

    return int(GGUFReader(str(path)).fields["general.file_type"].parts[-1][0])


def _export(model, tokenizer, out, method):
    t0 = time.perf_counter()
    model.save_pretrained_gguf(str(out), tokenizer, quantization_method=method)
    return round(time.perf_counter() - t0, 2), {p.name: p for p in Path(out).glob("*.gguf")}


def main(argv=None):
    p = C.base_parser("mlx_gguf", DEFAULT_MODEL, TINY_MODEL, default_steps=0)
    p.add_argument("--no-list", action="store_true", help="skip the list-form export")
    a = C.resolve_args(p, argv)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with C.JobRecorder(a, backend_hint=None) as rec:
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        import unsloth  # noqa: F401  must precede mlx_lm / transformers
        is_mlx = bool(getattr(unsloth, "_IS_MLX", False))
        rec.set_backend("unsloth-mlx" if is_mlx else "unsloth-no-mlx")
        if not rec.check("mlx_active", is_mlx, f"unsloth._IS_MLX={is_mlx}"):
            return
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            a.model, max_seq_length=a.max_seq_length, load_in_4bit=True, text_only=True,
            random_state=a.seed, token=os.environ.get("HF_TOKEN") or None)
        model = FastLanguageModel.get_peft_model(
            model, r=8, lora_alpha=16, lora_dropout=0.0, random_state=a.seed,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            use_gradient_checkpointing=False)
        root = Path(a.out).resolve().parent / "mlx_gguf_out"
        stem = (getattr(model, "_hf_repo", None) or "model").split("/")[-1]

        scalar_s = 0.0
        for q in QUANTS:
            secs, files = _export(model, tokenizer, root / f"scalar_{q}", q)
            scalar_s += secs
            want = f"{stem}.{q.upper()}.gguf"
            got = {n: _file_type(f) for n, f in files.items()}
            rec.check(f"scalar_{q}", got == {want: FTYPE[q]}, got)
        rec.summary(scalar_exports_s=round(scalar_s, 2))

        if not a.no_list:
            try:
                secs, files = _export(model, tokenizer, root / "list", list(QUANTS))
            except Exception as e:  # noqa: BLE001  base trees raise TypeError here
                rec.check("list_export", False, f"{type(e).__name__}: {e}")
                return
            got = {n: _file_type(f) for n, f in files.items()}
            want = {f"{stem}.{q.upper()}.gguf": FTYPE[q] for q in QUANTS}
            rec.check("list_export", got == want, got)
            rec.check("list_intermediate_removed", f"{stem}.BF16.gguf" not in got, sorted(got))
            rec.summary(list_export_s=secs, list_vs_scalar=round(secs / max(scalar_s, 1e-9), 3))


if __name__ == "__main__":
    main()
