"""MTP deferred-priming A/B on Apple Silicon (unsloth-zoo#2907): a persistent controller, one short reply, then long-prompt
replies; per reply the draft rounds, tok/s and whether the output equals plain greedy decoding. Head = the installed
unsloth_zoo; base = the same tree with unsloth_zoo/mlx/speculative.py at --base-sha. Arms alternate B H H B, one process each."""
import argparse, json, os, platform, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

REPO = "Qwen/Qwen3.5-0.8B"  # --model; the MTP head is read from its checkpoint


def worker(a):
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.utils import get_model_path

    import unsloth_zoo.mlx.speculative as spec
    from unsloth_zoo.mlx.generate import SamplingParams
    from unsloth_zoo.mlx.loader import _ensure_exact_verify_sums_promoted, _ensure_verify_qk_norm_matches_decode

    _ensure_exact_verify_sums_promoted()
    _ensure_verify_qk_norm_matches_decode()
    model, processor = load(a.model, lazy = True)
    if a.bits:
        import mlx.nn as nn
        nn.quantize(model.language_model, group_size = 64, bits = a.bits)
    mx.eval(model.parameters())
    tok, lm = processor.tokenizer, model.language_model
    chat = lambda text: tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt = True, tokenize = False),
                                   add_special_tokens = False)
    notes = lambda seed: "".join(f"Note {i}: the service in region {(i * seed) % 7} restarted after its cache filled; the fix was raising "
                                 f"the limit to {i * 13 + seed} MB. " for i in range(a.repeat))
    prompts = [chat("Count from one to forty in words.")] + [
        chat(notes(seed) + "\nList every region mentioned and its final limit, one per line.") for seed in (3, 5)]
    drafter = spec.native_mtp_drafter(get_model_path(a.model), model)
    # Studio's default controller (studio/backend/core/inference/mlx_speculative.py:_draft), kept across replies.
    controller = spec.DraftController(max_depth = int(getattr(drafter, "max_depth", 8)), max_copy = 16, can_copy = True, fixed_depth = False)

    def prefill(prompt, cache, capture = False):
        # Chunked: one long prompt forward trips the Metal watchdog on a paravirtual runner GPU.
        hidden = []
        for start in range(0, len(prompt), a.chunk):
            out = lm(mx.array([prompt[start:start + a.chunk]]), cache = cache, **(spec._capture(drafter) if capture else {}))
            if capture:
                hidden.append(spec._features(drafter, out))
            mx.eval(out.logits, *hidden[-1:])
        return out, mx.concatenate(hidden, axis = 1) if capture else None

    def plain(prompt):
        cache = make_prompt_cache(lm)
        token = int(mx.argmax(prefill(prompt, cache)[0].logits[:, -1], -1).item())
        out = [token]
        for _ in range(1, a.tokens):
            token = int(mx.argmax(lm(mx.array([[token]]), cache = cache).logits[:, -1], -1).item())
            out.append(token)
        return out

    replies = []
    for prompt in prompts:
        engine, cache = spec.SpeculativeEngine(model, controller, drafter), make_prompt_cache(lm)
        out, hidden = prefill(prompt, cache, capture = True)
        first = int(mx.argmax(out.logits[:, -1], -1).item())
        engine.add(spec.EngineRow(cache = cache, pending = first, prompt = prompt, sampling = SamplingParams(), max_tokens = a.tokens,
                                  hidden = hidden, uid = 0))
        plans, toks, original = [], [first], controller.plan
        controller.plan = lambda rows: plans.append(original(rows)) or plans[-1]
        mx.synchronize()
        start = time.perf_counter()
        while engine.rows:
            for step in engine.step():
                toks.extend(step.tokens)
        mx.synchronize()
        seconds = time.perf_counter() - start
        controller.plan = original
        replies.append(dict(prompt_tokens = len(prompt), tokens = len(toks), seconds = round(seconds, 3), tok_s = round(len(toks) / seconds, 2),
                            draft_rounds = sum(p.kind == "round" and any(r.source == "draft" for r in p.rows) for p in plans),
                            plans = len(plans), matches_plain = toks[:a.tokens] == plain(prompt)))
    print("ARM " + json.dumps(dict(speculative = spec.__file__, replies = replies)), flush = True)


def driver(a):
    import unsloth_zoo
    head_pkg = Path(unsloth_zoo.__file__).parent
    base_root = Path(tempfile.mkdtemp(prefix = "zoo_base_"))
    shutil.copytree(head_pkg, base_root / "unsloth_zoo", ignore = shutil.ignore_patterns("__pycache__"))
    url = f"https://raw.githubusercontent.com/unslothai/unsloth-zoo/{a.base_sha}/unsloth_zoo/mlx/speculative.py"
    (base_root / "unsloth_zoo" / "mlx" / "speculative.py").write_bytes(urllib.request.urlopen(url, timeout = 60).read())
    roots = {"base": str(base_root), "head": None}
    record = dict(job = "mlx_mtp_prime", model = a.model, backend = "unsloth-mlx", device = platform.platform(), base_sha = a.base_sha,
                  config = vars(a), arms = [], checks = {}, passed = False, error = None)
    try:
        for arm in a.order.split(","):
            env = dict(os.environ)
            if roots[arm]:
                env["PYTHONPATH"] = roots[arm] + os.pathsep + env.get("PYTHONPATH", "")
            run = subprocess.run([sys.executable, __file__, "--worker", "--tokens", str(a.tokens), "--repeat", str(a.repeat),
                                  "--model", a.model, "--bits", str(a.bits), "--chunk", str(a.chunk)],
                                 env = env, capture_output = True, text = True, timeout = 3600)
            line = next((l for l in run.stdout.splitlines() if l.startswith("ARM ")), None)
            if line is None:
                raise RuntimeError(f"{arm} arm failed rc={run.returncode}: {run.stderr[-3000:]}")
            result = json.loads(line[4:])
            expected = str(base_root) if arm == "base" else str(head_pkg)
            if not result["speculative"].startswith(expected):
                raise RuntimeError(f"{arm} arm imported {result['speculative']}, not {expected}")
            record["arms"].append(dict(arm = arm, **result))
            print(arm, json.dumps(result["replies"]), flush = True)
        replies = [reply for arm in record["arms"] for reply in arm["replies"]]
        record["checks"]["outputs_match_plain"] = all(reply["matches_plain"] for reply in replies)
        record["passed"] = all(record["checks"].values())
    except Exception as exc:
        record["error"] = repr(exc)
    Path(a.out).write_text(json.dumps(record, indent = 2))
    print("JOB_RESULT " + json.dumps(dict(job = record["job"], model = a.model, backend = record["backend"], device = "mlx",
                                          passed = record["passed"], error = record["error"], out = a.out)), flush = True)
    sys.exit(0 if record["passed"] else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action = "store_true")
    p.add_argument("--model", default = REPO)
    p.add_argument("--bits", type = int, default = 0, help = "quantize the target's language model (0 = as stored)")
    p.add_argument("--base-sha", default = "cf4eb467ffdbf30fdc9d1a607f273bfc7b000448")
    p.add_argument("--order", default = "base,head,head,base")
    p.add_argument("--tokens", type = int, default = 128)
    p.add_argument("--chunk", type = int, default = 256)
    p.add_argument("--repeat", type = int, default = 130)
    p.add_argument("--out", default = "outputs/jobs/mlx_mtp_prime.json")
    a = p.parse_args()
    if sys.platform != "darwin" and not a.worker:
        print("JOB_RESULT " + json.dumps(dict(job = "mlx_mtp_prime", skipped = "Apple Silicon only", passed = False)))
        sys.exit(3)
    worker(a) if a.worker else driver(a)
