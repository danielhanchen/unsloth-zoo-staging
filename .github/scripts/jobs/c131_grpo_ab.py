"""CPU GRPO (HF generate rollouts, no vLLM): merge-base zoo vs head zoo, interleaved A/A x N in one job.

Each arm is its own subprocess. The base arm imports a git checkout of unsloth-zoo at --base-ref through PYTHONPATH;
the head arm imports the installed zoo (the staging branch) unless --head-ref is given. Arms run ABBA-interleaved on
the same runner. Records s/step (median of steps 2..N), the generate() share, rollout hashes per step (equal seeds +
equal logits => equal sampled rollouts), a greedy-decode hash before training, and whether the growable KV path
engaged (unsloth_zoo.cpu.kv_cache.ENGAGED).

    python jobs/c131_grpo_ab.py --base-ref <sha> --repeats 3 --steps 10 --out c131.json

macOS Apple Silicon: forces the torch CPU path (UNSLOTH_FORCE_GPU_PATH=1), not MLX. `--tiny` is accepted and
ignored (staging CI passes it); the effect needs the real Qwen3-0.6B shape.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

TAG = "C131_RESULT "


def checkout(ref, repo="unsloth-zoo", cache={}):
    if (repo, ref) not in cache:
        d = tempfile.mkdtemp(prefix="c131_" + repo.replace("-", "_") + "_")
        url = f"https://github.com/unslothai/{repo}"
        subprocess.run(["git", "init", "-q", d], check=True)
        for attempt in range(4):
            if subprocess.run(["git", "-C", d, "fetch", "-q", "--depth", "1", url, ref]).returncode == 0:
                break
            time.sleep(15 * (attempt + 1))
        else:
            subprocess.run(["git", "-C", d, "fetch", "-q", "--depth", "1", url, ref], check=True)
        subprocess.run(["git", "-C", d, "checkout", "-q", "FETCH_HEAD"], check=True)
        sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        cache[(repo, ref)] = (d, sha)
    return cache[(repo, ref)]


def child(a):
    rec = {"step_wall_s": [], "losses": [], "rollout_sha": [], "completion_tokens": []}
    for k in ("UNSLOTH_ZOO_DISABLE_GPU_INIT", "UNSLOTH_ALLOW_CPU", "UNSLOTH_IS_PRESENT", "UNSLOTH_COMPILE_DISABLE"):
        os.environ.pop(k, None)
    if sys.platform == "darwin":
        os.environ.setdefault("UNSLOTH_FORCE_GPU_PATH", "1")
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch
    import unsloth_zoo
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    rec["zoo_file"] = unsloth_zoo.__file__
    rec["unsloth_file"] = unsloth.__file__
    try:
        from unsloth_zoo.cpu import kv_cache
        rec["has_kv_cache_module"] = True
    except Exception:  # noqa: BLE001
        kv_cache = None
        rec["has_kv_cache_module"] = False
    from unsloth_zoo import device_type as dt
    rec["device_type"] = dt.DEVICE_TYPE
    if a.threads:
        torch.set_num_threads(a.threads)
    rec["torch_threads"] = torch.get_num_threads()

    gen = {"n": 0, "s": 0.0}
    from transformers.generation.utils import GenerationMixin
    orig_generate = GenerationMixin.generate

    def timed_generate(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig_generate(self, *args, **kwargs)
        finally:
            gen["n"] += 1
            gen["s"] += time.perf_counter() - t0
    GenerationMixin.generate = timed_generate

    seq = a.prompt + a.completion + 16
    model, tok = FastLanguageModel.from_pretrained(model_name=a.model, max_seq_length=seq, load_in_4bit=False,
                                                   dtype=None)
    model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8, lora_dropout=0, random_state=0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"],
                                             use_gradient_checkpointing="unsloth")
    t = getattr(tok, "tokenizer", tok)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    t.padding_side = "left"
    words = ("the quick brown fox jumps over the lazy dog while a small cat sleeps in the warm sun and "
             "three birds sing on a tall green tree near the old stone bridge").split()

    def sentence(i, n):
        return " ".join(words[(i * 7 + j) % len(words)] for j in range(n))

    # Greedy decode before training: base and head must produce the same tokens (left-padded, ragged batch).
    FastLanguageModel.for_inference(model)
    enc = t([sentence(i, 10 + 9 * i) for i in range(4)], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.greedy_tokens, min_new_tokens=a.greedy_tokens, do_sample=False)
    rec["greedy_sha"] = hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16]
    FastLanguageModel.for_training(model)

    # Prompts of ~a.prompt tokens; distinct per row, several rows per prompt (num_generations).
    ds = Dataset.from_list([{"prompt": sentence(i, max(8, a.prompt * 3 // 4))} for i in range(64)])

    def reward(completions, **kw):
        rec["rollout_sha"].append(hashlib.sha256("\x00".join(map(str, completions)).encode()).hexdigest()[:16])
        return [float(len(set(str(x).split()))) / 50.0 for x in completions]

    names = {f.name for f in __import__("dataclasses").fields(GRPOConfig)}
    kw = dict(output_dir=tempfile.mkdtemp(prefix="c131_grpo_"), max_steps=a.steps,
              per_device_train_batch_size=a.batch, gradient_accumulation_steps=1, learning_rate=1e-5,
              logging_steps=1, save_strategy="no", report_to="none", seed=0, num_generations=a.gens,
              max_completion_length=a.completion, max_prompt_length=a.prompt, use_vllm=False, temperature=1.0,
              beta=0.0)
    if a.fixed_length:
        kw["generation_kwargs"] = {"min_new_tokens": a.completion}
    rec["dropped_cfg"] = sorted(k for k in kw if k not in names)
    c = GRPOConfig(**{k: v for k, v in kw.items() if k in names})
    params = __import__("inspect").signature(GRPOTrainer.__init__).parameters
    tk = {"processing_class": tok} if "processing_class" in params else {"tokenizer": tok}
    trainer = GRPOTrainer(model=model, reward_funcs=[reward], train_dataset=ds, args=c, **tk)

    class CB(TrainerCallback):
        def on_step_begin(self, args, state, control, **k):
            self.t = time.perf_counter()

        def on_step_end(self, args, state, control, **k):
            rec["step_wall_s"].append(round(time.perf_counter() - self.t, 3))

        def on_log(self, args, state, control, logs=None, **k):
            logs = logs or {}
            if "loss" in logs:
                rec["losses"].append(logs["loss"])
            for key in ("completions/mean_length", "completion_length"):
                if key in logs:
                    rec["completion_tokens"].append(logs[key])
                    break
    trainer.add_callback(CB())
    engaged0 = kv_cache.ENGAGED[0] if kv_cache is not None else None
    gen["n"], gen["s"] = 0, 0.0
    t0 = time.perf_counter()
    trainer.train()
    rec["train_s"] = round(time.perf_counter() - t0, 2)
    rec["generate_calls"], rec["generate_s"] = gen["n"], round(gen["s"], 2)
    rec["growable_engaged"] = (kv_cache.ENGAGED[0] - engaged0) if kv_cache is not None else 0
    steady = rec["step_wall_s"][1:] or rec["step_wall_s"]
    rec["s_per_step"] = sorted(steady)[len(steady) // 2] if steady else None
    print(TAG + json.dumps(rec), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ref", required=False, default=os.environ.get("C131_BASE_REF"))
    p.add_argument("--head-ref", default=None, help="default: the installed unsloth_zoo")
    p.add_argument("--model", default="unsloth/Qwen3-0.6B-Base")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--gens", type=int, default=4)
    p.add_argument("--prompt", type=int, default=64)
    p.add_argument("--completion", type=int, default=512)
    p.add_argument("--greedy-tokens", type=int, default=48)
    p.add_argument("--fixed-length", type=int, default=1, help="min_new_tokens = completion (same work per arm)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--out", default="c131_grpo_ab.json")
    p.add_argument("--tiny", action="store_true", help="ignored")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.child:
        return child(a)
    if not a.base_ref:
        raise SystemExit("--base-ref (or C131_BASE_REF) is required")
    report = {"env": {"platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count(),
                      "python": platform.python_version()}, "args": vars(a), "runs": []}
    for pkg in ("torch", "transformers", "trl", "peft", "unsloth"):
        try:
            report["env"][pkg] = __import__("importlib.metadata").metadata.version(pkg)
        except Exception:  # noqa: BLE001
            pass
    arms = {"base": a.base_ref, "head": a.head_ref}
    order = []
    for rep in range(a.repeats):
        order += [(rep, n) for n in (("base", "head") if rep % 2 == 0 else ("head", "base"))]
    child_args = ["--child", "--model", a.model, "--steps", str(a.steps), "--batch", str(a.batch), "--gens",
                  str(a.gens), "--prompt", str(a.prompt), "--completion", str(a.completion), "--greedy-tokens",
                  str(a.greedy_tokens), "--fixed-length", str(a.fixed_length), "--threads", str(a.threads)]
    for rep, arm in order:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        ref = arms[arm]
        if ref:
            d, sha = checkout(ref)
            env["PYTHONPATH"] = d + os.pathsep + env.get("PYTHONPATH", "")
            report.setdefault("refs", {})[arm] = sha
        t0 = time.perf_counter()
        try:
            pr = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), *child_args], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=a.timeout,
                                encoding="utf-8", errors="replace")
            out, rc = pr.stdout, pr.returncode
        except subprocess.TimeoutExpired as e:
            out, rc = str(e.stdout or ""), "timeout"
        run = {"arm": arm, "rep": rep, "rc": rc, "wall_s": round(time.perf_counter() - t0, 1)}
        res = [ln[len(TAG):] for ln in out.splitlines() if ln.startswith(TAG)]
        if res:
            run.update(json.loads(res[-1]))
        else:
            run["tail"] = out.splitlines()[-60:]
        report["runs"].append(run)
        print(f"[{arm}#{rep}] rc={rc} s/step={run.get('s_per_step')} steps={run.get('step_wall_s')} "
              f"gen_s={run.get('generate_s')} engaged={run.get('growable_engaged')} greedy={run.get('greedy_sha')} "
              f"zoo={run.get('zoo_file')} threads={run.get('torch_threads')}", flush=True)
        if "tail" in run:
            print("\n".join("    | " + x for x in run["tail"][-40:]), flush=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)

    def med(v):
        v = sorted(x for x in v if x is not None)
        return v[len(v) // 2] if v else None
    summary = {}
    for arm in ("base", "head"):
        per = [r.get("s_per_step") for r in report["runs"] if r["arm"] == arm]
        good = [x for x in per if x]
        summary[arm] = {"s_per_step": per, "median": med(per),
                        "spread_pct": round(100 * (max(good) - min(good)) / med(good), 2) if good else None,
                        "generate_s": [r.get("generate_s") for r in report["runs"] if r["arm"] == arm],
                        "engaged": [r.get("growable_engaged") for r in report["runs"] if r["arm"] == arm]}
    b, h = summary["base"]["median"], summary["head"]["median"]
    if b and h:
        summary["head_vs_base_pct"] = round(100 * (h - b) / b, 2)
    greedy = {r.get("greedy_sha") for r in report["runs"] if r.get("greedy_sha")}
    summary["greedy_equal"] = len(greedy) == 1
    rollouts = {tuple(r.get("rollout_sha") or ()) for r in report["runs"] if r.get("rollout_sha")}
    summary["rollouts_equal_all_runs"] = len(rollouts) == 1
    summary["zoo_files"] = {f"{r['arm']}#{r['rep']}": r.get("zoo_file") for r in report["runs"]}
    report["summary"] = summary
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("SUMMARY " + json.dumps(summary), flush=True)
    ok = all(r["rc"] == 0 and r.get("s_per_step") for r in report["runs"])
    ok = ok and all(r.get("growable_engaged", 0) > 0 for r in report["runs"] if r["arm"] == "head")
    ok = ok and all(not r.get("has_kv_cache_module") for r in report["runs"] if r["arm"] == "base")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
