"""Inference parity matrix for VLM / audio families: eager vs decode CUDA graph vs compiled decode.

    python jobs/inference_matrix.py --models qwen25vl,whisper,csm --out outputs/w12/infer/run.json
    python jobs/inference_matrix.py --models tts_llama@flm      # FastLanguageModel (llama decode graphs)
    python jobs/inference_matrix.py --models dir:/path/to/tiny  # any local model dir

A model is a regression registry key, `dir:<path>` or `tiny:<Arch>` (tiny_models.py); `@flm` loads it
through FastLanguageModel instead of FastModel. Correctness only: no timing is recorded.

Modes (each one sequence of generate calls on the same loaded model, env flags flipped between modes):
  eager          UNSLOTH_DECODE_GENERIC_GRAPHS=0, UNSLOTH_DECODE_CUDA_GRAPHS=0
  graph          defaults (generic StaticCache graphs are on inside plain generate; FastLanguageModel
                 decode graphs are on only under RL, so this mode shows what a user gets)
  graph_forced   UNSLOTH_DECODE_CUDA_GRAPHS=1 (llama-path graphs outside RL too)
  compiled       UNSLOTH_COMPILE_DECODE=1, only where vision._compiles_decode(model) says it applies
Call sequence per mode: A, B, A, C (B = same shapes, other media; C = other image size / audio length).

Checks per family:
  tokens_equal_<mode>     greedy tokens of every call equal the eager mode's
  hold_first_output       the tensor call 1 returned is unchanged after call 2 ran (buffer aliasing)
  media_sensitive         eager prefill logits for A and B differ (otherwise the stale check is VOID)
  no_stale_conditioning   graph B tokens == eager B tokens and the repeated A == first A
  replays / captures      generic + llama decode graph counters; repeated same-shape call adds 0 captures
Refusals: why a mode did not graph (planner counters, refusal reasons, the cache decision inputs).
Writes one JSON with every family, plus outputs/w12/cells/infer_<key>.json rows for the W12 ledger.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _regression():
    # The workspace root holds source clones (vllm/) that break `import unsloth` when it is on sys.path:
    # import the regression package from its directory instead.
    if "regression" not in sys.modules:
        pkg = _load_file("regression", WS / "regression" / "__init__.py")
        pkg.__path__ = [str(WS / "regression")]
    from regression import registry
    from regression.lib import common, media
    return registry, common, media


MODES = ("eager", "graph", "graph_forced", "compiled")
ENV = {
    "eager": {"UNSLOTH_DECODE_GENERIC_GRAPHS": "0", "UNSLOTH_DECODE_CUDA_GRAPHS": "0", "UNSLOTH_COMPILE_DECODE": "0"},
    "graph": {"UNSLOTH_DECODE_GENERIC_GRAPHS": None, "UNSLOTH_DECODE_CUDA_GRAPHS": None, "UNSLOTH_COMPILE_DECODE": "0"},
    "graph_forced": {"UNSLOTH_DECODE_GENERIC_GRAPHS": None, "UNSLOTH_DECODE_CUDA_GRAPHS": "1", "UNSLOTH_COMPILE_DECODE": "0"},
    "compiled": {"UNSLOTH_DECODE_GENERIC_GRAPHS": None, "UNSLOTH_DECODE_CUDA_GRAPHS": None, "UNSLOTH_COMPILE_DECODE": "1"},
}
SEQ = ("A", "B", "A", "C")


def set_env(mode):
    for k, v in ENV[mode].items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def resolve_model(spec):
    registry, common, _ = _regression()
    loader = "fm"
    if spec.endswith("@flm"):
        spec, loader = spec[:-4], "flm"
    if spec.startswith("dir:"):
        return spec[4:], Path(spec[4:]).name, loader, False
    if spec.startswith("tiny:"):
        sys.path.append(str(WS))
        import tiny_models
        path = tiny_models.get(spec[5:])
        return path, spec[5:], loader, False
    m = registry.BY_KEY[spec]
    return common.resolve(m), m.key, loader, m.trust_remote_code


def _stats(model):
    from unsloth.models import _decode_graphs as G
    from unsloth.models import _generic_decode_graphs as GG
    out = {"generic": {}, "llama": {}}
    seen = set()
    for mod in [model] + [m for _, m in model.named_modules()][:4]:
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        s = GG.stats(mod)
        if s:
            out["generic"] = s
            break
    for mod in [model, getattr(model, "model", None), getattr(getattr(model, "base_model", None), "model", None)]:
        if mod is None:
            continue
        try:
            s = G.stats(mod)
        except Exception as e:  # noqa: BLE001
            s = {"error": f"{type(e).__name__}: {e}"[:200]}
        if s:
            out["llama"] = s
            break
    out["plans"] = dict(GG.PLANS)
    return out


def _counts(st):
    g, l = st.get("generic") or {}, st.get("llama") or {}
    caps = int(g.get("captures", 0) or 0) + int(l.get("captures", l.get("graphs_captured", 0)) or 0)
    reps = int(g.get("replays", 0) or 0) + int(l.get("replays", 0) or 0)
    return caps, reps


def _diff_plans(before, after):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)}


def _cache_decision(model, kwargs):
    """Inputs of unsloth_base_fast_generate's static-cache / graph decision, for a refusal reason."""
    out = {}
    try:
        from unsloth.models import vision as V
        out["generate_fn"] = getattr(getattr(type(model), "generate", None), "__qualname__", None)
        out["unsloth_generate"] = getattr(type(model).generate, "__name__", "") in ("unsloth_base_fast_generate",) \
            or "unsloth" in (getattr(type(model).generate, "__module__", "") or "")
        out["supports_static_cache"] = bool(getattr(model, "_supports_static_cache", getattr(model, "_can_compile_fullgraph", True)))
        out["flash_attention"] = bool(V._uses_flash_attention_for_generation(model.config))
        try:
            out["bidirectional_mm_mask"] = bool(V._needs_bidirectional_multimodal_mask(model, dict(kwargs)))
        except Exception as e:  # noqa: BLE001
            out["bidirectional_mm_mask"] = f"error {type(e).__name__}"
        out["compiles_decode"] = bool(V._compiles_decode(model))
        out["compile_decode_models"] = list(V.COMPILE_DECODE_MODELS)
        out["is_encoder_decoder"] = bool(getattr(model.config, "is_encoder_decoder", False))
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def _tokens(out):
    import torch
    if isinstance(out, torch.Tensor):
        return out
    for k in ("sequences", "audio_codes"):
        v = getattr(out, k, None)
        if isinstance(v, torch.Tensor):
            return v
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"generate returned {type(out).__name__}")


def run_family(spec, a):
    import torch
    registry, common, media = _regression()
    path, key, loader, trust = resolve_model(spec)
    import transformers as T
    cfg = T.AutoConfig.from_pretrained(path, trust_remote_code=trust)
    kind = media.kind(cfg)
    rec = {"spec": spec, "key": key, "path": str(path), "loader": loader, "kind": kind, "precision": a.precision,
           "model_type": cfg.model_type, "modes": {}, "checks": {}, "refusals": {}, "notes": []}
    from unsloth import FastLanguageModel, FastModel
    kw = dict(model_name=str(path), max_seq_length=a.max_seq_length, load_in_4bit=a.precision == "4bit",
              dtype=torch.bfloat16, trust_remote_code=trust)
    if kind == "csm":
        kw["auto_model"] = T.CsmForConditionalGeneration
    if kind == "whisper":
        kw["auto_model"] = T.WhisperForConditionalGeneration
    t0 = time.time()
    if loader == "flm":
        model, proc = FastLanguageModel.from_pretrained(**kw)
        FastLanguageModel.for_inference(model)
    else:
        model, proc = FastModel.from_pretrained(**kw)
        FastModel.for_inference(model)
    rec["load_s"] = round(time.time() - t0, 1)
    rec["class"] = type(model).__name__
    dev = "cuda"
    inputs = {v: media.gen_inputs(kind, model, proc, v, dev) for v in "ABC"}
    rec["media_keys"] = media.media_keys(inputs["A"])
    rec["input_shapes"] = {v: {k: list(x.shape) for k, x in d.items() if hasattr(x, "shape")} for v, d in inputs.items()}
    tok = getattr(proc, "tokenizer", proc)
    pad = getattr(tok, "pad_token_id", None)
    if pad is None:
        pad = getattr(tok, "eos_token_id", 0) or 0
    gen_kw = dict(max_new_tokens=a.max_new_tokens, min_new_tokens=a.max_new_tokens, do_sample=False)
    if kind == "csm":
        gen_kw = dict(max_new_tokens=a.max_new_tokens, do_sample=False, depth_decoder_do_sample=False, output_audio=False)
    elif kind != "whisper":
        gen_kw["pad_token_id"] = pad
    rec["decision_A"] = _cache_decision(model, inputs["A"])

    # Media sensitivity: eager prefill logits differ between A and B, so a stale replay is detectable.
    try:
        set_env("eager")
        with torch.no_grad():
            fwd = {}
            for v in "AB":
                d = dict(inputs[v])
                if kind == "whisper":
                    d["decoder_input_ids"] = torch.tensor([[int(model.config.decoder_start_token_id)]], device=dev)
                    d["decoder_input_ids"] = d["decoder_input_ids"].expand(d["input_features"].shape[0], 1)
                o = model(**d)
                lg = o.logits
                fwd[v] = lg[:, -1].float()
            rec["checks"]["media_sensitive"] = {"ok": not torch.equal(fwd["A"], fwd["B"]),
                                                "detail": f"max |dA-dB| {float((fwd['A'] - fwd['B']).abs().max()):.4g}"}
    except Exception as e:  # noqa: BLE001
        rec["checks"]["media_sensitive"] = {"ok": None, "detail": f"forward failed {type(e).__name__}: {e}"[:300]}

    def generate(v):
        with torch.no_grad():
            return model.generate(**inputs[v], **gen_kw)

    modes = list(a.modes)
    for mode in modes:
        if mode == "compiled":
            set_env(mode)   # _compiles_decode reads UNSLOTH_COMPILE_DECODE
            d = _cache_decision(model, inputs["A"])
            if not d.get("compiles_decode"):
                rec["modes"][mode] = {"status": "not applicable",
                                      "reason": f"vision._compiles_decode is False: model_type {cfg.model_type} not in "
                                                f"COMPILE_DECODE_MODELS {d.get('compile_decode_models')}"
                                                + ("" if loader == "fm" else " (FastLanguageModel path has no compiled decode)")}
                continue
        set_env(mode)
        mrec = {"calls": []}
        try:
            held = held_copy = None
            for i, v in enumerate(SEQ):
                before = _stats(model)
                out = generate(v)
                toks = _tokens(out)
                after = _stats(model)
                c0, r0 = _counts(before)
                c1, r1 = _counts(after)
                mrec["calls"].append({"variant": v, "shape": list(toks.shape), "tokens": toks.tolist(),
                                      "captures": c1 - c0, "replays": r1 - r0,
                                      "plans": _diff_plans(before.get("plans", {}), after.get("plans", {}))})
                if i == 0:
                    held, held_copy = toks, toks.clone()
                if i == 1:
                    mrec["hold_first_output"] = bool(torch.equal(held, held_copy))
            st = _stats(model)
            mrec["stats"] = st
            mrec["captures"] = sum(c["captures"] for c in mrec["calls"])
            mrec["replays"] = sum(c["replays"] for c in mrec["calls"])
            mrec["repeat_captures"] = mrec["calls"][2]["captures"]   # call 3 repeats call 1's shapes
            refus = dict((st.get("generic") or {}).get("refusals") or {})
            for k, n in ((st.get("llama") or {}).get("refusals") or {}).items():
                refus[f"llama:{k}"] = n
            mrec["refusals"] = refus
            mrec["status"] = "ran"
        except Exception as e:  # noqa: BLE001
            mrec["status"] = "error"
            mrec["error"] = f"{type(e).__name__}: {e}"[:600]
            mrec["traceback"] = traceback.format_exc()[-3000:]
        rec["modes"][mode] = mrec

    # Checks against eager.
    eager = rec["modes"].get("eager", {})
    for mode in modes:
        m = rec["modes"].get(mode, {})
        if mode == "eager" or m.get("status") != "ran" or eager.get("status") != "ran":
            continue
        same = [c["tokens"] == e["tokens"] for c, e in zip(m["calls"], eager["calls"])]
        rec["checks"][f"tokens_equal_{mode}"] = {"ok": all(same), "detail": f"per call {same}"}
        if m["replays"] > 0:
            rec["checks"][f"no_stale_conditioning_{mode}"] = {
                "ok": same[1] and m["calls"][2]["tokens"] == m["calls"][0]["tokens"],
                "detail": "B vs eager B, repeated A vs first A"
                          + ("" if rec["checks"].get("media_sensitive", {}).get("ok") else " (VOID: media not sensitive)")}
            rec["checks"][f"bounded_captures_{mode}"] = {"ok": m["repeat_captures"] == 0,
                                                       "detail": f"call 3 (repeat of A) captured {m['repeat_captures']}"}
        rec["checks"][f"hold_first_output_{mode}"] = {"ok": m.get("hold_first_output"), "detail": "call 1 tensor after call 2"}
        if m["replays"] == 0:
            plans = {}
            for c in m["calls"]:
                for k, n in c["plans"].items():
                    plans[k] = plans.get(k, 0) + n
            rec["refusals"][mode] = {"plans": plans, "refusals": m.get("refusals"),
                                     "decision": rec["decision_A"]}
    for mode in modes:
        m = rec["modes"].get(mode, {})
        if m.get("status") == "error":
            rec["checks"][f"mode_{mode}_runs"] = {"ok": False, "detail": m.get("error")}
    return rec


def i_status(rec):
    """W12 ledger I status for one family from its record."""
    ch = rec.get("checks", {})
    bad = [k for k, v in ch.items() if v.get("ok") is False and k != "media_sensitive"]
    if bad:
        return "failed", f"failed checks {bad}"
    graphed = [m for m in ("graph", "graph_forced") if rec["modes"].get(m, {}).get("replays", 0) > 0]
    if graphed:
        comp = rec["modes"].get("compiled", {})
        note = f"graphs replay in {graphed}; tokens equal eager"
        if comp.get("status") == "not applicable":
            note += f"; compiled decode n/a ({comp.get('reason', '')[:80]})"
        if not ch.get("media_sensitive", {}).get("ok"):
            note += "; stale-conditioning check VOID (media not sensitive)"
        return "verified", note
    if rec["refusals"]:
        return "supported fallback", "graphs refused: " + json.dumps(rec["refusals"].get("graph", rec["refusals"]), default=str)[:400]
    return "failed", "no mode replayed and no refusal recorded"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma list: registry key | dir:<path> | tiny:<Arch>, optional @flm")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--precision", default="16bit", choices=("16bit", "4bit"))
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--out", default=str(WS / "outputs" / "w12" / "infer" / f"infer_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    ap.add_argument("--cells-dir", default=str(WS / "outputs" / "w12" / "cells"))
    ap.add_argument("--in-process", action="store_true",
                    help="run every model in this process (default: one fresh process per model; loader env such as "
                         "CSM's UNSLOTH_COMPILE_DISABLE=partial and compiled modules otherwise leak into the next model)")
    a = ap.parse_args(argv)
    a.modes = [m for m in a.modes.split(",") if m]
    specs = [s for s in a.models.split(",") if s]
    if len(specs) > 1 and not a.in_process:
        import subprocess
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged, rc = {"families": {}, "argv": sys.argv[1:], "per_model_files": []}, 0
        for i, spec in enumerate(specs):
            part = out.with_name(f"{out.stem}.part{i}{out.suffix}")
            cmd = [sys.executable, "-u", __file__, "--models", spec, "--modes", ",".join(a.modes), "--precision", a.precision,
                   "--max-new-tokens", str(a.max_new_tokens), "--max-seq-length", str(a.max_seq_length),
                   "--out", str(part), "--cells-dir", a.cells_dir]
            r = subprocess.run(cmd)
            rc = rc or r.returncode
            merged["per_model_files"].append(str(part))
            try:
                d = json.loads(part.read_text())
                merged["families"].update(d.get("families", {}))
                merged.setdefault("versions", d.get("versions"))
                merged.setdefault("device", d.get("device"))
            except Exception as e:  # noqa: BLE001
                merged["families"][f"{spec}|{a.precision}"] = {"spec": spec, "error": f"child wrote no JSON: {e}", "I": {
                    "status": "blocked", "note": f"child process rc {r.returncode}, no JSON"}}
            out.write_text(json.dumps(merged, indent=1, default=str))
        print(f"wrote {out}")
        return rc
    import unsloth  # noqa: F401  (before transformers)
    import torch
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"versions": {}, "families": {}, "argv": sys.argv[1:]}
    for mod in ("torch", "transformers", "trl", "peft", "unsloth", "unsloth_zoo"):
        try:
            m = __import__(mod)
            report["versions"][mod] = f"{getattr(m, '__version__', '?')} {Path(m.__file__).resolve().parent}"
        except Exception:
            pass
    report["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    rc = 0
    for spec in [s for s in a.models.split(",") if s]:
        try:
            rec = run_family(spec, a)
        except Exception as e:  # noqa: BLE001
            rec = {"spec": spec, "error": f"{type(e).__name__}: {e}"[:600], "traceback": traceback.format_exc()[-3000:],
                   "modes": {}, "checks": {}, "refusals": {}}
        st, note = ("blocked", "load / setup failed: " + rec["error"][:300]) if "error" in rec else i_status(rec)
        rec["I"] = {"status": st, "note": note}
        if st == "failed":
            rc = 1
        report["families"][f"{spec}|{a.precision}"] = rec
        out.write_text(json.dumps(report, indent=1, default=str))
        cells = Path(a.cells_dir)
        cells.mkdir(parents=True, exist_ok=True)
        safe = spec.replace("/", "_").replace(":", "_").replace("@", "_")
        (cells / f"infer_{safe}_{a.precision}.json").write_text(json.dumps([{
            "match": {"infer_key": rec.get("key", spec), "loader": rec.get("loader", "fm"), "precision": a.precision},
            "prop": "I", "status": st, "evidence": str(out), "note": note}], indent=1))
        print(f"INFER_RESULT {spec} {a.precision}: {st} | {note}", flush=True)
        try:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
    print(f"wrote {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
