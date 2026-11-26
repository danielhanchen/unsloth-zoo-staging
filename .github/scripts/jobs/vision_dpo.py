# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""VLM DPO (and ORPO / KTO where the installed TRL takes images) on drawn images: FastVisionModel +
PatchDPOTrainer, ref_model=None (the frozen base is the reference), notebook DPO settings (beta 0.1).

    python jobs/vision_dpo.py --tiny --model tiny:Qwen2_5_VLForConditionalGeneration           # LoRA
    python jobs/vision_dpo.py --tiny --model tiny:Gemma3ForConditionalGeneration --mode qlora
    python jobs/vision_dpo.py --tiny --backend hf ...   # plain TRL + PEFT reference (step-1 logps parity)

Checks, beyond the standard training ones:
- same_media_both_branches: every policy forward carries 2 x prompts images and the [chosen; rejected]
  halves see identical pixel tensors (and image grids);
- image_tokens_intact: every row holds the same number of image tokens as its image produced (no truncation);
- vision_side_grads: the projector / vision LoRA (or full) parameters moved;
- preference head engagement: TRL >= 1.15 fused_lm_head forwards (no full logits) or Unsloth's
  chunked concatenated_forward head on older TRL, with its fallback reasons.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402
import _vision as vz  # noqa: E402

p = c.base_parser("vision_dpo", "unsloth/Qwen2.5-VL-3B-Instruct",
                  "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration", default_steps=3)
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--lora-r", type=int, default=16)
p.add_argument("--rows", type=int, default=8)
p.add_argument("--batch", type=int, default=2)
p.add_argument("--image-size", type=int, default=56)
p.add_argument("--image-size2", type=int, default=None)
p.add_argument("--trainer", choices=("dpo", "orpo", "kto"), default="dpo")
vz.add_mode_arg(p)
p.set_defaults(lr=5e-5, max_seq_length=1024)
a = c.resolve_args(p)
if a.trainer != "dpo":
    a._job = f"vision_{a.trainer}"

VISION_PARAM_TAGS = ("visual", "vision", "multi_modal_projector", "mm_projector", "merger", "connector",
                     "embed_vision", "image_newline")


def rows(n, size, size2):
    out = []
    for i, (img, name) in enumerate(vz.drawn_images(n, (size,) if size2 is None else (size, size2))):
        wrong = {"red": "blue", "green": "red", "blue": "yellow", "yellow": "green"}[name]
        out.append({"images": [img],
                    "prompt": [{"role": "user", "content": [{"type": "image"},
                                                            {"type": "text", "text": "What color is the square?"}]}],
                    "chosen": [{"role": "assistant", "content": [{"type": "text", "text": f"It is {name}."}]}],
                    "rejected": [{"role": "assistant", "content": [{"type": "text", "text": f"It is {wrong}."}]}]})
    return out


def kto_rows(pairs):
    out = []
    for r in pairs:
        out.append({"images": r["images"], "prompt": r["prompt"], "completion": r["chosen"], "label": True})
        out.append({"images": r["images"], "prompt": r["prompt"], "completion": r["rejected"], "label": False})
    return out


def vision_snapshot(model):
    import torch
    with torch.no_grad():
        return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()
                if p.requires_grad and any(t in n for t in VISION_PARAM_TAGS)}


with c.JobRecorder(a) as rec:
    c.seed_everything(a.seed)
    if a.backend != "hf":
        import unsloth  # noqa: F401
        from unsloth import FastVisionModel, PatchDPOTrainer
        PatchDPOTrainer()
    import torch
    import _compat as k
    from datasets import Dataset, Features, Image, Sequence, Value  # noqa: F401
    import trl

    if a.trainer == "dpo":
        from trl import DPOConfig as Cfg, DPOTrainer as Trainer
    elif a.trainer == "orpo":
        try:
            from trl import ORPOConfig as Cfg, ORPOTrainer as Trainer
        except ImportError:
            from trl.experimental.orpo import ORPOConfig as Cfg, ORPOTrainer as Trainer
    else:
        try:
            from trl import KTOConfig as Cfg, KTOTrainer as Trainer
        except ImportError:
            from trl.experimental.kto import KTOConfig as Cfg, KTOTrainer as Trainer

    rec.set_backend(k.backend_name(a, c.detect_device()))
    data = rows(a.rows, a.image_size, a.image_size2)
    if a.trainer == "kto":
        data = kto_rows(data)
    ds = Dataset.from_list(data)

    if a.backend != "hf":
        lk = vz.load_kwargs(a.mode)
        # Unsloth's trainer caps max_length at the model's max_seq_length.
        model, tok = FastVisionModel.from_pretrained(a.model, use_gradient_checkpointing="unsloth",
                                                     max_seq_length=a.max_seq_length, **lk)
        if not lk["full_finetuning"]:
            model = FastVisionModel.get_peft_model(model, finetune_vision_layers=True, finetune_language_layers=True,
                                                   finetune_attention_modules=True, finetune_mlp_modules=True,
                                                   r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0, bias="none",
                                                   random_state=a.seed, target_modules="all-linear")
        FastVisionModel.for_training(model)
    else:
        if a.mode != "lora":
            rec.skip(f"--backend hf runs the LoRA reference only (asked --mode {a.mode})")
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoProcessor.from_pretrained(a.model)
        model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=k.cuda_dtype()).to(dev)
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0.0,
                                                 target_modules="all-linear"))
    probes = vz.Probes(model) if a.backend != "hf" else None

    cfg_model = getattr(model, "config", None)
    image_token_id = None
    for src in (cfg_model, getattr(cfg_model, "text_config", None)):
        for attr in ("image_token_id", "image_token_index"):
            v = getattr(src, attr, None) if src is not None else None
            if isinstance(v, int) and image_token_id is None:
                image_token_id = v

    seen = {"forwards": 0, "with_pixels": 0, "same_media": 0, "media_checked": 0, "image_tokens_per_row": [],
            "fused_lm_head": 0, "rows": []}

    def _pre(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        pv = kwargs.get("pixel_values")
        if ids is None or pv is None or not torch.is_tensor(pv) or not module.training:
            return
        seen["forwards"] += 1
        seen["with_pixels"] += 1
        seen["fused_lm_head"] += int(bool(kwargs.get("fused_lm_head", False)))
        n = ids.shape[0]
        seen["rows"].append(n)
        if a.trainer != "kto" and n % 2 == 0:
            grid = kwargs.get("image_grid_thw")
            seen["media_checked"] += 1
            ok = True
            if grid is not None:
                half = grid.shape[0] // 2
                ok = grid.shape[0] % 2 == 0 and torch.equal(grid[:half], grid[half:])
                rows_ = int(grid[:half].prod(-1).sum()) if ok else 0
                ok = ok and pv.shape[0] == 2 * rows_ and torch.equal(pv[:rows_], pv[rows_:])
            elif pv.shape[0] % 2 == 0:
                half = pv.shape[0] // 2
                ok = torch.equal(pv[:half], pv[half:])
            else:
                ok = False
            seen["same_media"] += int(ok)
        if image_token_id is not None:
            seen["image_tokens_per_row"].append((ids == image_token_id).sum(1).tolist())
    model.register_forward_pre_hook(_pre, with_kwargs=True)

    dropped = []
    ta = k.common_train_kwargs(a, os.path.join(os.path.dirname(a.out) or ".", f"{a._job}_run"))
    ta.update(per_device_train_batch_size=a.batch, max_grad_norm=1.0, remove_unused_columns=False,
              dataset_num_proc=1)
    cfg_kw = dict(beta=a.beta, max_length=a.max_seq_length, max_prompt_length=None, **ta)
    if a.trainer == "kto":
        cfg_kw.update(desirable_weight=1.0, undesirable_weight=1.0)
    cfg_kw.update(c.cfg_overrides(a))
    cfg = k.make(Cfg, dropped, **cfg_kw)
    c.check_cfg_overrides(rec, a, dropped)
    kw = {"ref_model": None} if a.trainer in ("dpo", "kto") else {}
    callbacks = [c.metrics_callback(rec)] + ([probes.callback()] if probes else [])
    trainer = Trainer(model=model, args=cfg, train_dataset=ds, callbacks=callbacks, **kw,
                      **k.tokenizer_kwarg(Trainer, tok))
    rec.summary(dropped_config_args=dropped, rows=len(ds), beta=a.beta, mode=a.mode, trainer=a.trainer,
                trl=trl.__version__, image_token_id=image_token_id,
                trainable_params=sum(x.numel() for x in model.parameters() if x.requires_grad))

    before = c.adapter_fingerprint(model)
    vis_before = vision_snapshot(model)
    out = trainer.train()
    after = c.adapter_fingerprint(model)
    vis_after = vision_snapshot(model)
    moved = sorted(n for n in vis_before if not torch.equal(vis_before[n], vis_after[n]))
    rec.summary(train_runtime_s=out.metrics.get("train_runtime"), final_loss=out.training_loss,
                forwards={kk: v for kk, v in seen.items() if kk not in ("image_tokens_per_row", "rows")},
                rows_per_forward=seen["rows"][:8], image_tokens_per_row=seen["image_tokens_per_row"][:4],
                vision_params=len(vis_before), vision_params_moved=len(moved), vision_moved_sample=moved[:6])
    if probes is not None:
        rec.summary(probes=probes.summary(trainer))

    rec.standard_training_checks()
    rec.backend_check()
    rec.check("adapter_changed", *c.adapter_changed(before, after))
    rec.check("images_reached_model", seen["with_pixels"] > 0, seen["with_pixels"])
    if a.trainer != "kto":
        rec.check("same_media_both_branches", seen["media_checked"] > 0 and seen["same_media"] == seen["media_checked"],
                  f"{seen['same_media']}/{seen['media_checked']}")
    per_row = [x for f in seen["image_tokens_per_row"] for x in f]
    rec.check("image_tokens_intact", image_token_id is None or (bool(per_row) and min(per_row) > 0
              and (a.image_size2 is not None or len(set(per_row)) == 1)),
              f"image_token_id {image_token_id}: {sorted(set(per_row))}")
    rec.check("vision_side_grads", len(moved) > 0, f"{len(moved)}/{len(vis_before)} moved")
    steps = rec.data["steps"]
    if a.trainer == "dpo":
        first = steps[0].get("loss") if steps else None
        rec.check("dpo_initial_loss_near_log2", first is not None and abs(first - 0.6931) < 0.05, first)
    if probes is not None and a.trainer == "dpo":
        pr = rec.data["summary"]["probes"]
        fused_head = seen["fused_lm_head"] > 0
        chunked = pr.get("unsloth_dpo_chunked_calls", 0) > 0
        rec.summary(preference_head={"trl_fused_lm_head_forwards": seen["fused_lm_head"],
                                     "unsloth_chunked_calls": pr.get("unsloth_dpo_chunked_calls", 0),
                                     "unsloth_chunked_fallbacks": pr.get("unsloth_dpo_chunked_fallbacks")})
        rec.check("preference_head_no_full_logits", fused_head or chunked,
                  rec.data["summary"]["preference_head"])
