"""TTS LM SFT on Llama / Qwen (Orpheus, Llasa, OuteTTS, Spark-TTS): text tokens in, codec tokens out.

The notebooks train the LM on token ids; the codec (SNAC / xcodec2 / DAC / BiCodec) encodes and decodes outside
Unsloth, so this job feeds deterministic codec-token sequences drawn from the top of the vocabulary after a
tokenized text prompt (uneven lengths, right padded), the notebook's loader (FastLanguageModel or FastModel),
LoRA targets, and full finetuning in float32 for Spark. After training: greedy codec-token generate, eager
(UNSLOTH_DECODE_CUDA_GRAPHS=0) vs default (decode CUDA graphs), tokens must be identical.

    python jobs/tts_sft.py --preset orpheus --max-steps 3
    python jobs/tts_sft.py --preset spark --tiny
    python jobs/tts_sft.py --preset oute --tiny --loader FastLanguageModel   # the FastModel notebook rerouted
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402
import _presets as P  # noqa: E402

PRESET = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--preset"), "orpheus")
pre, _model = P.resolve(P.TTS_SFT, PRESET, tiny="--tiny" in sys.argv)
p = c.base_parser("tts_sft", pre["model"], _model if "--tiny" in sys.argv else pre["tiny"], default_steps=3)
p.add_argument("--preset", default="orpheus", choices=sorted(P.TTS_SFT))
p.add_argument("--loader", default=pre["loader"], choices=("FastLanguageModel", "FastModel"))
p.add_argument("--load-in-4bit", action="store_true")
p.add_argument("--rows", type=int, default=16)
p.add_argument("--codec-tokens", type=int, default=48, help="codec tokens per row (before the uneven trim)")
p.add_argument("--gen-tokens", type=int, default=24)
p.set_defaults(lr=pre["lr"], max_seq_length=min(pre["max_seq_length"], 512))
a = c.resolve_args(p)

TEXTS = ["Hello there.", "The weather is lovely today, isn't it?", "One two three four five.",
         "Unsloth makes fine tuning faster.", "Speak slowly and clearly please."]


def rows(tok, n, codec, vocab, seed):
    """Text ids + codec ids from the top 4096 of the vocabulary (where SNAC / xcodec2 codes live), uneven lengths."""
    import torch
    g = torch.Generator().manual_seed(seed)
    lo = max(vocab - 4096, 1000)
    out = []
    for i in range(n):
        text = tok(TEXTS[i % len(TEXTS)], add_special_tokens=True)["input_ids"]
        k = codec - (i * 7) % (codec // 2)
        ids = list(text) + torch.randint(lo, vocab, (k,), generator=g).tolist()
        out.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)})
    return out


def collate(batch):
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda xs, v: [x + [v] * (n - len(x)) for x in xs]  # noqa: E731
    return {"input_ids": torch.tensor(pad([b["input_ids"] for b in batch], 0)),
            "attention_mask": torch.tensor(pad([b["attention_mask"] for b in batch], 0)),
            "labels": torch.tensor(pad([b["labels"] for b in batch], -100))}


def decode_stats(model):
    """Decode-graph counters of whichever path generated (FastModel generic / FastLanguageModel llama)."""
    from unsloth.models import _decode_graphs as DG, _generic_decode_graphs as GG
    out = {}
    for m in model.modules():
        if GG._ATTR in m.__dict__:
            s = GG.stats(m)
            out["generic"] = {k: s.get(k) for k in ("captures", "replays", "refusals")}
        s = DG.stats(m) if hasattr(DG, "stats") else None
        if s:
            out["llama"] = s
    return out


with c.JobRecorder(a) as rec:
    c.seed_everything(a.seed)
    if a.backend == "hf":
        rec.skip("tts_sft covers the Unsloth loaders only (--backend unsloth, or auto on an accelerator)")
    import unsloth  # noqa: F401  (must precede transformers)
    import torch
    import _compat as k
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments
    from unsloth import FastLanguageModel, FastModel

    rec.set_backend(k.backend_name(a, c.detect_device()))
    Loader = FastLanguageModel if a.loader == "FastLanguageModel" else FastModel
    full = bool(pre.get("full_finetuning"))
    # Spark: float32 + full finetuning, as its notebook.
    dtype = torch.float32 if PRESET == "spark" else None
    model, tok = Loader.from_pretrained(model_name=a.model, max_seq_length=a.max_seq_length, dtype=dtype,
                                        load_in_4bit=a.load_in_4bit, full_finetuning=full)
    tok = getattr(tok, "tokenizer", tok)
    if not full:
        model = Loader.get_peft_model(model, r=pre["lora_r"], lora_alpha=pre["lora_alpha"], lora_dropout=0,
                                      bias="none", target_modules=pre["target_modules"],
                                      use_gradient_checkpointing="unsloth", random_state=a.seed)
    vocab = model.config.get_text_config().vocab_size
    ds = Dataset.from_list(rows(tok, a.rows, a.codec_tokens, vocab, a.seed))
    dropped = []
    ta = k.common_train_kwargs(a, os.path.join(os.path.dirname(a.out) or ".", "tts_run"))
    ta.update(per_device_train_batch_size=pre["batch"], remove_unused_columns=False)
    if dtype is torch.float32:
        ta.update(bf16=False, fp16=False)
    ta.update(c.cfg_overrides(a))
    args = k.make(TrainingArguments, dropped, **ta)
    c.check_cfg_overrides(rec, a, dropped)
    trainer = Trainer(model=model, train_dataset=ds, data_collator=collate, args=args,
                      callbacks=[c.metrics_callback(rec)])
    rec.summary(dropped_config_args=dropped, notebook=pre["notebook"], preset=PRESET, loader=a.loader,
                full_finetuning=full, load_in_4bit=a.load_in_4bit, model_class=type(model).__name__,
                trainable_params=sum(x.numel() for x in model.parameters() if x.requires_grad))
    before = c.adapter_fingerprint(model)
    out = trainer.train()
    after = c.adapter_fingerprint(model)
    rec.summary(train_runtime_s=out.metrics.get("train_runtime"), final_loss=out.training_loss)
    rec.standard_training_checks()
    rec.backend_check()
    rec.check("weights_changed" if full else "adapter_changed", *c.adapter_changed(before, after))

    # Codec-token generate: eager vs decode CUDA graphs (the default), greedy, fixed length.
    Loader.for_inference(model)
    dev = next(model.parameters()).device
    prompt = torch.tensor([tok(TEXTS[0])["input_ids"]], device=dev)
    gen = {}
    for arm, env in (("eager", "0"), ("graphed", None)):
        if env is None:
            os.environ.pop("UNSLOTH_DECODE_CUDA_GRAPHS", None)
        else:
            os.environ["UNSLOTH_DECODE_CUDA_GRAPHS"] = env
        with torch.no_grad():
            seq = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
                                 max_new_tokens=a.gen_tokens, min_new_tokens=a.gen_tokens, pad_token_id=0)
        gen[arm] = seq[0, prompt.shape[1]:].tolist()
    os.environ.pop("UNSLOTH_DECODE_CUDA_GRAPHS", None)
    stats = decode_stats(model)
    rec.summary(generate_tokens=gen, decode_graph_stats=stats)
    rec.check("generate_length", all(len(v) == a.gen_tokens for v in gen.values()), {k_: len(v) for k_, v in gen.items()})
    rec.check("generate_graphed_matches_eager", gen["eager"] == gen["graphed"], stats)
