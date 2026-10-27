"""E2E for unsloth-zoo#2109 on real MLX (CPU backend) + mlx-vlm with a tiny random VLM.

python e2e.py SCENARIO MODEL OUT.json ; arm chosen by PYTHONPATH. Prints RESULT json.
"""
import json, os, sys, tempfile, traceback
os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
import numpy as np
import mlx.core as mx
from PIL import Image

scenario, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"scenario": scenario, "model": MODEL, "ok": False}


def img(i, size=48):
    rng = np.random.default_rng(i)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def lora_snapshot(model):
    from mlx.utils import tree_flatten
    return {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.trainable_parameters())}


def lora_moved(before, model):
    after = lora_snapshot(model)
    return sum(int(not np.array_equal(before[k], after[k])) for k in before), len(before)


def count_vision(model):
    """Count vision tower forwards: proves pixels reached the policy."""
    counter = {"n": 0}
    tower = None
    for name in ("vision_tower", "vision_model", "visual"):
        tower = getattr(model, name, None)
        if tower is not None:
            break
    if tower is None:
        return counter
    cls = type(tower)
    orig = cls.__call__

    def wrapped(self, *a, **k):
        counter["n"] += 1
        return orig(self, *a, **k)
    cls.__call__ = wrapped
    return counter


def cfg(cls, out, **kw):
    base = dict(per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=2,
                warmup_steps=0, learning_rate=1e-3, logging_steps=1, output_dir=out,
                report_to="none", max_seq_length=512, seed=3407)
    base.update(kw)
    return cls(**base)


def losses(trainer):
    hist = getattr(getattr(trainer, "state", None), "log_history", None) or []
    return [round(float(h["loss"]), 6) for h in hist if "loss" in h]


try:
    import unsloth_zoo
    res["zoo"] = unsloth_zoo.__file__
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx import trainer as T
    from datasets import Dataset
    out = tempfile.mkdtemp()
    model, proc = FastMLXModel.from_pretrained(MODEL, max_seq_length=512, load_in_4bit=False, text_only=False)
    vision = count_vision(model)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0, target_modules="all-linear")
    before = lora_snapshot(model)

    if scenario == "grpo_vlm":
        rows = [{"prompt": [{"role": "user", "content": f"Describe image {i}."}], "image": img(i)} for i in range(4)]
        ds = Dataset.from_list(rows)
        seen = {"prompts": None}

        def reward(prompts, completions, **kw):
            seen["prompts"] = prompts[0]
            return [float(len(c[0]["content"]) % 7) for c in completions]
        tr = T.MLXGRPOTrainer(model=model, tokenizer=proc, train_dataset=ds, reward_funcs=[reward],
                              args=cfg(T.MLXGRPOConfig, out, num_generations=2, max_completion_length=8,
                                       per_device_train_batch_size=2, beta=0.04, loss_type="grpo"))
        r = tr.train()
        res["reward_prompt"] = json.loads(json.dumps(seen["prompts"], default=str))
    elif scenario == "grpo_vlm_text":
        rows = [{"prompt": [{"role": "user", "content": f"Say {i}."}]} for i in range(4)]
        tr = T.MLXGRPOTrainer(model=model, tokenizer=proc, train_dataset=rows,
                              reward_funcs=[lambda completions, **kw: [float(len(c[0]["content"]) % 5) for c in completions]],
                              args=cfg(T.MLXGRPOConfig, out, num_generations=2, max_completion_length=8))
        r = tr.train()
    elif scenario == "kto_vlm":
        rows = [{"prompt": [{"role": "user", "content": f"What is in image {i}?"}],
                 "completion": [{"role": "assistant", "content": f"A pattern {i}."}],
                 "label": bool(i % 2), "images": [img(i)]} for i in range(4)]
        tr = T.MLXKTOTrainer(model, proc, Dataset.from_list(rows), args=cfg(T.MLXKTOConfig, out, max_length=256))
        r = tr.train()
    elif scenario == "cpo_vlm_text":
        rows = [{"prompt": f"Question {i}?", "chosen": f" yes {i}.", "rejected": f" no {i + 3}."} for i in range(4)]
        tr = T.MLXCPOTrainer(model=model, tokenizer=proc, train_dataset=rows, args=cfg(T.MLXCPOConfig, out))
        r = tr.train()
    elif scenario == "cpo_vlm_image":
        rows = [{"prompt": [{"role": "user", "content": f"Q {i}?"}], "chosen": [{"role": "assistant", "content": "a"}],
                 "rejected": [{"role": "assistant", "content": "b"}], "images": [img(i)]} for i in range(4)]
        tr = T.MLXCPOTrainer(model=model, tokenizer=proc, train_dataset=Dataset.from_list(rows), args=cfg(T.MLXCPOConfig, out))
        r = tr.train()
    elif scenario == "dpo_vlm_hf":
        rows = [{"prompt": [{"role": "user", "content": f"What is in image {i}?"}],
                 "chosen": [{"role": "assistant", "content": f"Pattern {i}."}],
                 "rejected": [{"role": "assistant", "content": f"Nothing {i + 5}."}], "images": [img(i)]} for i in range(4)]
        tr = T.MLXDPOTrainer(model=model, tokenizer=proc, train_dataset=Dataset.from_list(rows),
                             eval_dataset=Dataset.from_list(rows[:2]),
                             args=cfg(T.MLXDPOConfig, out, beta=0.1, max_steps=2, eval_steps=2,
                                      precompute_ref_log_probs=os.environ.get("PRECOMP") == "1"))
        r = tr.train()
        res["eval"] = {k: v for h in tr.state.log_history for k, v in h.items() if k.startswith("eval_")}
    elif scenario == "gen_stop_ids":
        from unsloth_zoo.mlx.generate import generate_batch, GenerationRequest, GenerationDefaults, SamplingParams
        tok = getattr(proc, "tokenizer", proc)
        crit = getattr(tok, "stopping_criteria", None)
        before_eos = list(getattr(crit, "eos_token_ids", []) or [])
        text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}],
                                        add_generation_prompt=True, tokenize=False)
        free = generate_batch(model, proc, [GenerationRequest(prompt=text, image=img(1), max_tokens=6,
                              sampling=SamplingParams(temperature=0.0))], defaults=GenerationDefaults(max_tokens=6))
        first = free[0].token_ids[0]
        got = generate_batch(model, proc, [GenerationRequest(prompt=text, image=img(1), max_tokens=6,
                             sampling=SamplingParams(temperature=0.0))],
                             defaults=GenerationDefaults(max_tokens=6, stop_token_ids=[first]))
        res["free"] = {"ids": list(free[0].token_ids), "finish": free[0].finish_reason}
        res["stopped"] = {"ids": list(got[0].token_ids), "finish": got[0].finish_reason}
        res["shared_eos_unchanged"] = list(getattr(crit, "eos_token_ids", []) or []) == before_eos
        # prepared prompt path: processor outputs as vision_inputs
        inputs = proc(text=[text], images=[[img(1)]], return_tensors="np")
        ids = inputs["input_ids"][0].tolist()
        vi = {k: mx.array(v) for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        prep = generate_batch(model, proc, [GenerationRequest(prompt_token_ids=ids, vision_inputs=vi, max_tokens=6,
                              sampling=SamplingParams(temperature=0.0))], defaults=GenerationDefaults(max_tokens=6))
        res["prepared"] = {"ids": list(prep[0].token_ids), "finish": prep[0].finish_reason}
        res["prepared_equals_rendered"] = list(prep[0].token_ids) == list(free[0].token_ids)
        r = {}
    else:
        raise SystemExit("unknown scenario")
    if scenario != "gen_stop_ids":
        res["train_steps"] = r.get("train_steps") if isinstance(r, dict) else None
        res["losses"] = losses(tr)
        res["lora_moved"] = lora_moved(before, model)
    res["vision_forwards"] = vision["n"]
    res["ok"] = True
except Exception as e:
    res["error"] = f"{type(e).__name__}: {str(e)[:600]}"
    res["tb"] = traceback.format_exc()[-2500:]
json.dump(res, open(OUT, "w"), indent=1, default=str)
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "tb"}, default=str))
