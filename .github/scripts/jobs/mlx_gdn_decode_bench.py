"""Fused GatedDeltaNet decode step: TY (simdgroups per value head) A/B on this Apple GPU.

    python jobs/mlx_gdn_decode_bench.py [--rows 8,16] [--rounds 7] [--layers 24] [--e2e-model M] [--out J]

Flips `unsloth_zoo.mlx.inference._DECODE_STEP_ROWS` in one process (the kernel takes TY as a template,
so each value is its own pipeline) and checks, per TY: the fused step is engaged (call count), its
outputs / window / state are bitwise equal across TY and to the native layer, then times a dependent
24-layer decode chain and end-to-end greedy decode in ABBA rounds. Correctness checks gate the exit
code; timings are reported only (a CI VM is not an exclusive GPU).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


def _bytes(a):
    import mlx.core as mx
    import numpy as np
    return np.array(a.view(mx.uint8)).tobytes()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="8,16")
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--e2e-model", default="mlx-community/Qwen3.5-2B-4bit")
    p.add_argument("--e2e-tokens", type=int, default=128)
    p.add_argument("--probe-rows", default="32", help="extra TY values launched once in a subprocess")
    p.add_argument("--out", default="mlx_gdn_decode_bench.json")
    a = p.parse_args(argv)
    res = {"job": "mlx_gdn_decode_bench", "checks": {}, "timing": {}, "passed": False, "error": None}

    def check(name, ok, detail=""):
        res["checks"][name] = {"ok": bool(ok), "detail": str(detail)}
        print(f"CHECK {name}: {'OK' if ok else 'FAIL'} {detail}", flush=True)
        return bool(ok)

    def dump():
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)

    if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        res["skipped"] = "needs Apple Silicon"
        dump()
        print("JOB_RESULT " + json.dumps(res))
        return 3
    try:
        import unsloth  # noqa: F401  must precede mlx_lm / transformers
    except Exception as e:  # zoo-only installs still work for the layer bench
        res["unsloth_import"] = f"{type(e).__name__}: {e}"
    import mlx.core as mx
    import mlx.nn as nn
    import unsloth_zoo
    from mlx_vlm.models.qwen3_5 import language as native
    from unsloth_zoo.mlx import inference as decode

    rows = [int(r) for r in a.rows.split(",")]
    info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
    res["device"] = {k: str(v) for k, v in info.items()}
    res["gpu_generation"] = decode._gpu_generation()
    res["zoo_file"] = unsloth_zoo.__file__
    res["default_rows"] = decode._DECODE_STEP_ROWS
    import importlib.metadata as md
    res["versions"] = {k: md.version(k) for k in ("mlx", "mlx-vlm", "mlx-lm") if _has(md, k)}
    print("DEVICE", json.dumps(res["device"]), "gen", res["gpu_generation"], "default TY", decode._DECODE_STEP_ROWS,
          flush=True)

    class Chain(nn.Module):
        def __init__(self, n, kh, vh):
            super().__init__()
            self.layers = [self._layer(kh, vh) for _ in range(n)]

        @staticmethod
        def _layer(kh, vh):
            m = native.Qwen3_5GatedDeltaNet(SimpleNamespace(
                hidden_size=64, linear_num_value_heads=vh, linear_num_key_heads=kh, linear_key_head_dim=128,
                linear_value_head_dim=128, linear_conv_kernel_dim=4, rms_norm_eps=1e-6))
            m.set_dtype(mx.bfloat16)
            m.A_log = (mx.random.normal((vh,)) * 1.5).astype(mx.float32)
            m.dt_bias = (mx.random.normal((vh,)) * 3).astype(mx.bfloat16)
            m.norm.weight = (1 + 0.3 * mx.random.normal((128,))).astype(mx.bfloat16)
            return m

    def run_chain(chain, caches, x):
        for layer, c in zip(chain.layers, caches):
            x = mx.fast.rms_norm(layer(x, cache=c), None, 1e-6)
        return x

    def fresh(chain, B):
        mx.random.seed(7)
        caches = [native.ArraysCache(size=2) for _ in chain.layers]
        mx.eval(run_chain(chain, caches, (mx.random.normal((B, 5, 64)) * 4).astype(mx.bfloat16)))
        return caches

    calls = [0]
    stock = decode._decode_step

    def counted(*args):
        calls[0] += 1
        return stock(*args)

    decode._decode_step = counted
    ok_all = True
    # (key heads, value heads, batch): Qwen3.5-2B (16 / 16), 4B / 9B / 35B-A3B (16 / 32), batch 2, 27B (16 / 48).
    shapes = [(16, 16, 1), (16, 32, 1), (16, 16, 2), (16, 48, 1)]
    for kh, vh, B in shapes:
        tag = f"kh{kh}_vh{vh}_b{B}"
        mx.random.seed(0)
        chain = Chain(a.layers, kh, vh)
        mx.eval(chain.parameters())
        chain.eval()
        chain.freeze()
        mx.random.seed(1)
        xs = [(mx.random.normal((B, 1, 64)) * 4).astype(mx.bfloat16) for _ in range(4)]
        # Native reference (scope off) for a few decode steps.
        caches = fresh(chain, B)
        ref = []
        for x in xs:
            out = run_chain(chain, caches, x)
            mx.eval(out)
            ref.append((_bytes(out), [_bytes(c[0]) + _bytes(c[1]) for c in caches]))
        decode._DECODE_STEP_VERDICTS.clear()
        per_ty = {}
        with decode.fused_decode_conv_silu(chain):
            for ty in rows:
                decode._DECODE_STEP_ROWS = ty
                caches = fresh(chain, B)
                calls[0] = 0
                got = []
                for x in xs:
                    out = run_chain(chain, caches, x)
                    mx.eval(out)
                    got.append((_bytes(out), [_bytes(c[0]) + _bytes(c[1]) for c in caches]))
                per_ty[ty] = (got, calls[0])
            for ty in rows:
                got, n = per_ty[ty]
                # The first decode step of a layout runs the bitwise check on the native branch; later ones fuse.
                ok_all &= check(f"{tag}_ty{ty}_engaged", n >= a.layers * (len(xs) - 1), f"{n} fused calls")
                ok_all &= check(f"{tag}_ty{ty}_bitwise_vs_native", got == ref, "outputs + window + state")
            ok_all &= check(f"{tag}_verdicts_true", all(decode._DECODE_STEP_VERDICTS.values())
                            and decode._DECODE_STEP_VERDICTS, dict((str(k), v) for k, v in
                                                                   decode._DECODE_STEP_VERDICTS.items()))
            # Timing: ABBA over TY, a fresh warmed state per measurement, dependent chain, eval per token.
            times = {ty: [] for ty in rows}
            for ty in rows:  # warm every pipeline
                decode._DECODE_STEP_ROWS = ty
                caches = fresh(chain, B)
                for x in xs:
                    mx.eval(run_chain(chain, caches, x))
            for r in range(a.rounds):
                for ty in (rows if r % 2 == 0 else rows[::-1]):
                    decode._DECODE_STEP_ROWS = ty
                    caches = fresh(chain, B)
                    x = xs[0]
                    mx.synchronize() if hasattr(mx, "synchronize") else None
                    t0 = time.perf_counter()
                    for s in range(a.steps):
                        x = run_chain(chain, caches, x)
                        mx.eval(x)
                    times[ty].append((time.perf_counter() - t0) / (a.steps * a.layers) * 1e6)
            res["timing"][tag] = {str(ty): {"median_us_per_layer": round(statistics.median(v), 3),
                                            "min": round(min(v), 3), "max": round(max(v), 3),
                                            "all": [round(t, 3) for t in v]} for ty, v in times.items()}
            print("TIMING", tag, json.dumps({k: v["median_us_per_layer"] for k, v in res["timing"][tag].items()}),
                  flush=True)
        dump()

    # The repo's own decode Metal tests at each TY (strict; a missing file is a failure).
    import os
    tests = "tests/test_mlx_decode_metal.py"
    res["pytest"] = {}
    for ty in rows:
        code = ("import sys, pytest\n"
                "class P:\n"
                "    seen = set()\n"
                "    def pytest_runtest_setup(self, item):\n"
                "        import unsloth_zoo.mlx.inference as d\n"
                f"        d._DECODE_STEP_ROWS = {ty}; P.seen.add(d._DECODE_STEP_ROWS)\n"
                f"rc = pytest.main(['-q', '-p', 'no:cacheprovider', {tests!r}], plugins=[P()])\n"
                "print('TY_SEEN', sorted(P.seen)); sys.exit(rc)\n")
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, ["tests", os.environ.get("PYTHONPATH")]))}
        pr = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600, env=env)
        tail = (pr.stdout + pr.stderr)[-1500:]
        res["pytest"][str(ty)] = {"rc": pr.returncode, "tail": tail}
        print(f"PYTEST TY={ty} rc={pr.returncode}\n{tail}", flush=True)
        ok_all &= check(f"pytest_decode_metal_ty{ty}", os.path.isfile(tests) and pr.returncode == 0
                        and " passed" in tail and f"TY_SEEN [{ty}]" in tail, tail.strip().splitlines()[-1] if tail.strip() else "")
    dump()

    # Extra TY values (e.g. 32 = 1024 threads) launched once in a fresh process: does this pipeline take them?
    probes = {}
    for ty in [int(t) for t in a.probe_rows.split(",") if t]:
        code = (
            "import mlx.core as mx\nfrom unsloth_zoo.mlx import inference as d\n"
            f"d._DECODE_STEP_ROWS = {ty}\n"
            "B,Hk,Hv,K,D=1,16,16,4,128\nC=2*Hk*D+Hv*D\n"
            "r=lambda *s,dt=mx.bfloat16: mx.random.normal(s).astype(dt)\n"
            "out=d._decode_step(r(B,K-1,C), r(B,1,C), r(K,C,dt=mx.float32), r(B,1,Hv), r(B,1,Hv),"
            " r(Hv,dt=mx.float32), r(Hv), r(B,1,Hv,D), r(D), 1e-6, r(B,Hv,D,D,dt=mx.float32), Hk)\n"
            "mx.eval(out); print('LAUNCH_OK', [o.shape for o in out])\n")
        pr = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        probes[ty] = {"rc": pr.returncode, "tail": (pr.stdout + pr.stderr)[-600:]}
        print("PROBE TY", ty, pr.returncode, (pr.stdout + pr.stderr)[-300:], flush=True)
    res["probes"] = probes

    # End to end: greedy decode tok/s through Unsloth's generation_mode scopes, ABBA over TY.
    try:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(a.e2e_model, max_seq_length=2048, text_only=True,
                                                       load_in_4bit=True)
        prompt = "Write a long story about a lighthouse keeper. " * 20
        e2e, texts = {ty: [] for ty in rows}, {}
        ids = mx.array([list(tok.encode(prompt))], dtype=mx.int32)
        for r in range(a.rounds + 1):
            for ty in (rows if r % 2 == 0 else rows[::-1]):
                decode._DECODE_STEP_ROWS = ty
                calls[0] = 0
                t0 = time.perf_counter()
                out = model.generate(input_ids=ids, max_new_tokens=a.e2e_tokens, do_sample=False)
                dt = time.perf_counter() - t0
                text = out if isinstance(out, str) else repr(getattr(out, "tolist", lambda: out)())
                if r == 0:
                    texts[ty] = text
                    ok_all &= check(f"e2e_ty{ty}_engaged", calls[0] > 0, f"{calls[0]} fused calls")
                    continue  # warmup round
                e2e[ty].append(a.e2e_tokens / dt)
        res["timing"]["e2e"] = {str(ty): {"median_tok_s": round(statistics.median(v), 3),
                                          "all": [round(t, 3) for t in v]} for ty, v in e2e.items()}
        ok_all &= check("e2e_greedy_identical", len(set(texts.values())) == 1, {k: v[:60] for k, v in texts.items()})
        print("E2E", json.dumps(res["timing"]["e2e"]), flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        res["e2e_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        print("E2E ERROR", res["e2e_error"], flush=True)
    res["passed"] = bool(ok_all)
    dump()
    print("JOB_RESULT " + json.dumps({k: res[k] for k in ("passed", "gpu_generation", "timing")}))
    return 0 if ok_all else 1


def _has(md, k):
    try:
        md.version(k)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    sys.exit(main())
