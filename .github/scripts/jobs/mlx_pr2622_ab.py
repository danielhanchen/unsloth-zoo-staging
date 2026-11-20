"""unsloth-zoo#2622 on Apple Silicon: PR test + tiny mlx-lm Gemma 3n clipped bf16 full FT, head tree then the
staged tree's trainer.py swapped for upstream main's (base). Run from the staged zoo checkout (editable install)."""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

TRAINER = Path("unsloth_zoo/mlx/trainer.py")
BASE_URL = "https://raw.githubusercontent.com/unslothai/unsloth-zoo/main/unsloth_zoo/mlx/trainer.py"
TEST = "tests/test_mlx_training_e2e_metal.py"


def repro(clip, compiled, out):
    import tempfile

    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.models import gemma3n

    import unsloth_zoo
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import FiniteTextBatchPlan, _FiniteTextRow

    L = 4
    text = dict(
        model_type="gemma3n_text", hidden_size=64, num_hidden_layers=L, intermediate_size=[128] * L,
        num_attention_heads=2, head_dim=32, rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=1,
        num_kv_shared_layers=0, vocab_size_per_layer_input=128, sliding_window=16, max_position_embeddings=64,
        rope_local_base_freq=10000.0, rope_theta=1000000.0, final_logit_softcapping=30.0,
        layer_types=["sliding_attention"] * 3 + ["full_attention"], activation_sparsity_pattern=[0.0] * L,
        hidden_size_per_layer_input=16, altup_num_inputs=4, altup_coef_clip=120.0, altup_correct_scale=True,
        altup_active_idx=0, laurel_rank=8,
    )
    mx.random.seed(3407)
    model = gemma3n.Model(gemma3n.ModelArgs(model_type="gemma3n", text_config=text))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    before = {k: np.asarray(v.astype(mx.float32)) for k, v in tree_flatten(model.trainable_parameters())}
    rows = [_FiniteTextRow(input_ids=tuple(int(x) for x in np.random.RandomState(i).randint(1, 128, 12)))
            for i in range(3)]
    args = MLXTrainingConfig(
        per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=6, warmup_steps=0,
        learning_rate=1e-5, weight_decay=0.0, optim="adamw", logging_steps=1, eval_steps=0, save_steps=0,
        max_seq_length=16, output_dir=tempfile.mkdtemp(), compile=bool(compiled),
        compile_mode="strict" if compiled else "eager", gradient_checkpointing=False,
        dataset_order="sequential", disable_memory_limits=True, use_cce=False, max_grad_norm=clip,
        lr_scheduler_type="constant",
    )
    trainer = MLXTrainer(model, type("Tok", (), {"pad_token_id": 0, "eos_token_id": 0})(), [], args=args)
    trainer._batches = FiniteTextBatchPlan(rows, [(i % 3,) for i in range(6)], max_seq_length=16, pad_id=0)
    trainer.save_model = lambda *_a, **_k: None
    losses = []
    trainer.add_step_callback(lambda step, total, loss, *r, **k: losses.append(float(loss)))
    result = trainer.train()
    coefs = {}
    for k, v in tree_flatten(model.trainable_parameters()):
        if "coefs" in k:
            d = np.abs(np.asarray(v.astype(mx.float32)) - before[k])
            coefs[k] = {"dtype": str(v.dtype), "max_move": float(d.max()), "frac_moved": float((d > 0).mean())}
    json.dump({"zoo": unsloth_zoo.__file__, "device": str(mx.default_device()), "losses": losses,
               "compile_enabled": result.get("compile_enabled"), "coefs": coefs}, open(out, "w"), indent=1)


def run_repro(clip, compiled, tag):
    out = Path(f"pr2622_{tag}.json").resolve()
    r = subprocess.run([sys.executable, __file__, "--repro", str(clip), str(compiled), str(out)],
                       capture_output=True, text=True, env={**os.environ, "UNSLOTH_IS_PRESENT": "1"})
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    return json.loads(out.read_text()) if r.returncode == 0 and out.exists() else {"error": r.stderr[-1500:]}


def run_test():
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", TEST,
                        "-k", "forward_stores", "-q"], capture_output=True, text=True, env=env)
    print(r.stdout[-4000:], r.stderr[-2000:], flush=True)
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|error)", r.stdout)}
    return counts


def main(argv=None):
    p = C.base_parser("mlx_pr2622_ab", "tiny gemma3n", "tiny gemma3n", default_steps=6)
    a = C.resolve_args(p, argv)
    with C.JobRecorder(a, backend_hint=None) as rec:
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        rec.set_backend("unsloth-mlx")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
        import mlx.core as mx
        rec.check("metal", mx.metal.is_available(), str(mx.default_device()))
        head_src = TRAINER.read_text(encoding="utf-8")
        rec.check("head_has_fix", "_restorable_storage_names" in head_src, str(TRAINER))
        base_src = urllib.request.urlopen(BASE_URL, timeout=60).read().decode("utf-8")
        rec.check("base_lacks_fix", "_restorable_storage_names" not in base_src, BASE_URL)
        res = {}
        try:
            for side, src in (("head", head_src), ("base", base_src)):
                TRAINER.write_text(src, encoding="utf-8")
                res[side] = {"test": run_test()}
                for c in (0, 1):
                    res[side][f"clip_c{c}"] = run_repro(1.0, c, f"{side}_clip_c{c}")
                res[side]["noclip_c0"] = run_repro(0.0, 0, f"{side}_noclip_c0")
        finally:
            TRAINER.write_text(head_src, encoding="utf-8")
        rec.summary(results=res)

        def full(r):
            return sum(v["frac_moved"] == 1.0 for v in r.get("coefs", {}).values()), len(r.get("coefs", {}))

        rec.check("head_test_4_passed", res["head"]["test"] == {"passed": 4}, res["head"]["test"])
        rec.check("base_test_4_failed", res["base"]["test"].get("failed") == 4, res["base"]["test"])
        for c in (0, 1):
            h, b = full(res["head"][f"clip_c{c}"]), full(res["base"][f"clip_c{c}"])
            rec.check(f"head_clip_c{c}_all_coefs_train", h[1] == 8 and h[0] == 8, h)
            rec.check(f"base_clip_c{c}_coefs_stall", b[1] == 8 and b[0] < 8, b)
            rec.check(f"clip_c{c}_compile_flag", res["head"][f"clip_c{c}"].get("compile_enabled") is bool(c),
                      res["head"][f"clip_c{c}"].get("compile_enabled"))
        for side in ("head", "base"):
            n = full(res[side]["noclip_c0"])
            rec.check(f"{side}_noclip_all_coefs_train", n == (8, 8), n)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--repro":
        repro(float(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        main()
