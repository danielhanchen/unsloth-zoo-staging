"""GPTQ / AWQ pre-quantized checkpoint accuracy + cost, MLX (Apple Silicon) vs a torch reference.

    python jobs/prequant_accuracy.py            # darwin-arm64: Unsloth MLX arms; else torch reference arms

Same eval on both platforms: wikitext-2 test, first 4 x 512 tokens, orig tokenizer. Per arm: per-chunk NLL,
argmax token per position (cross-platform agreement), KL vs the same platform's 16-bit original on the
first 256 positions of chunk 0, load time, peak memory, decode tok/s (MLX). Dequant parity: every packed
module dequantized by an independent numpy port of AutoGPTQ / AutoAWQ (dequantize_gemm) and hashed
(fp16 bytes); on MLX the PR's own dequant is hashed too and must match bit for bit.
"""

from __future__ import annotations

import gc
import glob
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

ORIG = "Qwen/Qwen2.5-0.5B-Instruct"
GPTQ = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"
AWQ = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
N_CHUNKS, CHUNK, KL_POS = 4, 512, 256
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _eval_ids():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    f = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                        repo_type="dataset")
    text = "".join(pq.read_table(f).column("text").to_pylist()[:2000])
    ids = AutoTokenizer.from_pretrained(ORIG).encode(text)
    assert len(ids) >= N_CHUNKS * CHUNK, len(ids)
    return [ids[i * CHUNK:(i + 1) * CHUNK] for i in range(N_CHUNKS)]


# ---- independent numpy dequant (AutoGPTQ v1 QuantLinear forward; AutoAWQ packing_utils.dequantize_gemm)
def _np_unpack_cols(x):  # [r, c] int32 -> [r, 8c], nibble j of word k -> column 8k+j
    import numpy as np
    u = x.astype(np.int64) & 0xFFFFFFFF
    return ((u[:, :, None] >> (4 * np.arange(8))[None, None, :]) & 0xF).reshape(x.shape[0], -1)


def np_gptq(qweight, qzeros, scales, g_idx):
    import numpy as np
    u = qweight.astype(np.int64) & 0xFFFFFFFF  # [in//8, out], nibble j of row k -> input row 8k+j
    w = ((u[:, None, :] >> (4 * np.arange(8))[None, :, None]) & 0xF).reshape(-1, qweight.shape[1])
    z = (_np_unpack_cols(qzeros) + 1) & 0xF
    s = scales.astype(np.float32)
    return ((w.astype(np.float32) - z[g_idx].astype(np.float32)) * s[g_idx]).T.astype(np.float16)


def np_awq(qweight, qzeros, scales, group_size):
    import numpy as np
    order = np.arange(qweight.shape[1] * 8).reshape(-1, 8)[:, AWQ_REVERSE_ORDER].reshape(-1)
    w = _np_unpack_cols(qweight)[:, order]
    z = _np_unpack_cols(qzeros)[:, order]
    s = scales.astype(np.float32)
    g = np.arange(w.shape[0]) // group_size
    return ((w.astype(np.float32) - z[g].astype(np.float32)) * s[g]).T.astype(np.float16)


def _np_ckpt(repo):
    from huggingface_hub import snapshot_download
    from safetensors.numpy import load_file
    d = snapshot_download(repo)
    t = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        t.update(load_file(f))
    cfg = json.load(open(os.path.join(d, "config.json")))["quantization_config"]
    return d, t, cfg


def ref_dense(repo):
    """{module: fp16 [out, in]} via the independent numpy dequant."""
    import numpy as np
    _, t, cfg = _np_ckpt(repo)
    gs = int(cfg.get("group_size", cfg.get("q_group_size", 128)))
    out = {}
    for k in sorted(k for k in t if k.endswith(".qweight")):
        m = k[: -len(".qweight")]
        if cfg["quant_method"] == "gptq":
            g = t.get(m + ".g_idx")
            if g is None:
                g = np.arange(t[k].shape[0] * 8) // gs
            out[m] = np_gptq(t[k], t[m + ".qzeros"], t[m + ".scales"], g.astype(np.int64))
        else:
            out[m] = np_awq(t[k], t[m + ".qzeros"], t[m + ".scales"], gs)
    return out, t


def _h(a):
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


# ---- per-platform arms
def _logprobs_np(logits):
    import numpy as np
    x = logits.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _score(logits_fn, chunks, ref_lp=None):
    import numpy as np
    nll, argmax, lp0 = [], [], None
    for i, ids in enumerate(chunks):
        lg = logits_fn(ids)  # [T, V] float32 numpy
        lp = _logprobs_np(lg[:-1])
        tgt = np.array(ids[1:])
        nll.append(float(-lp[np.arange(len(tgt)), tgt].mean()))
        argmax.extend(int(x) for x in lg.argmax(-1))
        if i == 0:
            lp0 = lp[:KL_POS]
    res = {"nll": nll, "ppl": math.exp(sum(nll) / len(nll)), "argmax": argmax}
    if ref_lp is not None:
        kl = (np.exp(ref_lp) * (ref_lp - lp0)).sum(-1)
        res["kl_mean"] = float(kl.mean())
        res["kl_p99"] = float(np.percentile(kl, 99))
        res["top1_vs_ref16"] = None
    return res, lp0


def run_mlx(rec, chunks):
    import mlx.core as mx
    import numpy as np
    import unsloth  # noqa: F401
    from mlx.utils import tree_flatten
    from mlx_lm import generate
    from unsloth_zoo.mlx import loader as L

    arms = [("orig16", ORIG, dict(load_in_16bit=True)), ("orig4", ORIG, dict(load_in_4bit=True)),
            ("gptq4", GPTQ, dict(load_in_4bit=True)), ("gptq16", GPTQ, dict(load_in_16bit=True)),
            ("awq4", AWQ, dict(load_in_4bit=True)), ("awq16", AWQ, dict(load_in_16bit=True))]
    ref_lp, ref_argmax, out = None, None, {}
    for name, repo, kw in arms:
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        model, tok = L.FastMLXModel.from_pretrained(repo, max_seq_length=CHUNK, text_only=True, **kw)
        load_s = time.perf_counter() - t0
        peak_load = mx.get_peak_memory() / 1024**3

        def logits_fn(ids, model=model):
            lg = model(mx.array([ids], dtype=mx.int32))
            lg = getattr(lg, "logits", lg)[0].astype(mx.float32)
            mx.eval(lg)
            return np.array(lg)

        r, lp0 = _score(logits_fn, chunks, ref_lp)
        if name == "orig16":
            ref_lp, ref_argmax = lp0, r["argmax"]
        r["top1_vs_ref16"] = float(np.mean(np.array(r["argmax"]) == np.array(ref_argmax)))
        prompt = tok.apply_chat_template([{"role": "user", "content": "Write a short poem about the sea."}],
                                         add_generation_prompt=True, tokenize=False)
        generate(model, tok, prompt=prompt, max_tokens=8)
        t1 = time.perf_counter()
        text = generate(model, tok, prompt=prompt, max_tokens=64)
        n = len(tok.encode(text))
        r.update(load_s=round(load_s, 2), peak_load_gb=round(peak_load, 3),
                 peak_total_gb=round(mx.get_peak_memory() / 1024**3, 3),
                 decode_tok_s=round(n / (time.perf_counter() - t1), 1),
                 weight_gb=round(sum(v.nbytes for _, v in tree_flatten(
                     model.parameters())) / 1024**3, 3),
                 quantized_source=getattr(model, "_unsloth_quantized_source", None), sample=text[:120])
        out[name] = r
        print(name, {k: v for k, v in r.items() if k != "argmax"}, flush=True)
        rec.summary(arms={k: {kk: vv for kk, vv in v.items() if kk != "argmax"} for k, v in out.items()})
        del model, tok
    # PR dequant vs independent numpy dequant, bit for bit.
    parity = {}
    for label, repo in (("gptq", GPTQ), ("awq", AWQ)):
        ref, t = ref_dense(repo)
        cfg = _np_ckpt(repo)[2]
        gs = int(cfg.get("group_size", cfg.get("q_group_size", 128)))
        mism, hashes = [], {}
        for m, want in ref.items():
            if label == "gptq":
                g = t.get(m + ".g_idx")
                g = mx.arange(t[m + ".qweight"].shape[0] * 8) // gs if g is None else mx.array(g)
                got = L._gptq_dequantize_weight(mx.array(t[m + ".qweight"]), mx.array(t[m + ".qzeros"]),
                                                mx.array(t[m + ".scales"]), g)
            else:
                got = L._awq_dequantize_weight(mx.array(t[m + ".qweight"]), mx.array(t[m + ".qzeros"]),
                                               mx.array(t[m + ".scales"]), gs)
            got = np.array(got.astype(mx.float16))
            hashes[m] = _h(want)
            if not np.array_equal(got, want):
                mism.append((m, float(np.abs(got.astype(np.float32) - want.astype(np.float32)).max())))
        parity[label] = {"modules": len(ref), "mismatched": mism[:5], "n_mismatched": len(mism),
                         "hashes": hashes}
    return out, parity


def run_torch(rec, chunks):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref_lp, ref_argmax, out, parity = None, None, {}, {}
    for name, repo in (("orig", ORIG), ("gptq_ref", GPTQ), ("awq_ref", AWQ)):
        model = AutoModelForCausalLM.from_pretrained(ORIG, torch_dtype=torch.float32).to(dev).eval()
        if repo != ORIG:
            dense, t = ref_dense(repo)
            sd = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in t.items()
                  if not k.endswith((".qweight", ".qzeros", ".scales", ".g_idx"))}
            sd.update({m + ".weight": torch.from_numpy(w) for m, w in dense.items()})
            own = model.state_dict()
            extra = sorted(k for k in sd if k not in own)
            nonzero_extra = [k for k in extra if sd[k].abs().sum() != 0]
            missing = sorted(k for k in own if k not in sd and k != "lm_head.weight")
            model.load_state_dict({k: v.float() for k, v in sd.items() if k in own}, strict=False)
            parity[name.split("_")[0]] = {"modules": len(dense), "hashes": {m: _h(w) for m, w in dense.items()},
                                          "extra_keys": len(extra), "nonzero_extra": nonzero_extra[:5],
                                          "missing": missing[:5]}

        @torch.no_grad()
        def logits_fn(ids, model=model):
            return model(torch.tensor([ids], device=dev)).logits[0].float().cpu().numpy()

        r, lp0 = _score(logits_fn, chunks, ref_lp)
        if name == "orig":
            ref_lp, ref_argmax = lp0, r["argmax"]
        r["top1_vs_ref16"] = float(np.mean(np.array(r["argmax"]) == np.array(ref_argmax)))
        out[name] = r
        print(name, {k: v for k, v in r.items() if k != "argmax"}, flush=True)
        del model
        gc.collect()
    return out, parity


def main(argv=None):
    p = C.base_parser("prequant_accuracy", GPTQ, GPTQ, default_steps=0)
    a = C.resolve_args(p, argv)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    is_mac = platform.system() == "Darwin" and platform.machine() == "arm64"
    with C.JobRecorder(a, backend_hint="unsloth-mlx" if is_mac else "torch-reference") as rec:
        chunks = _eval_ids()
        arms, parity = (run_mlx if is_mac else run_torch)(rec, chunks)
        rec.data["arms_full"] = arms
        rec.data["parity"] = parity
        rec.summary(arms={k: {kk: vv for kk, vv in v.items() if kk != "argmax"} for k, v in arms.items()})
        for k, v in arms.items():
            rec.check(f"{k}_finite_ppl", math.isfinite(v["ppl"]) and v["ppl"] < 100, v["ppl"])
        if is_mac:
            for label, pr in parity.items():
                rec.check(f"{label}_dequant_bit_exact", pr["n_mismatched"] == 0,
                          f"{pr['n_mismatched']}/{pr['modules']} {pr['mismatched']}")
        else:
            for label, pr in parity.items():
                rec.check(f"{label}_ref_loads_clean", not pr["missing"] and not pr["nonzero_extra"],
                          f"missing={pr['missing']} nonzero_extra={pr['nonzero_extra']}")


if __name__ == "__main__":
    main()
