"""Qwen4-Exp hyper-connection reads, fused (zoo#1946) vs native, at the release widths with random weights.

ABBA rounds in one process (kill switches toggled per round), bitwise equality and engagement checked.
Shared CI VM: ratios against the arm's own native rounds only, never absolute numbers.
Exit 0 all checks pass, 1 a check failed, 3 no Metal / no qwen4_exp.
"""
import argparse, json, os, statistics, sys, time
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--prefill-rows", type=int, default=4096)
    ap.add_argument("--out", default="job_mlx_qwen4_hc_bench.json")
    a = ap.parse_args()
    result = {"job": "mlx_qwen4_hc_bench", "checks": {}, "summary": {}, "passed": False, "error": None,
              "skipped": False, "finished": False}

    def dump():
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)

    try:
        import mlx.core as mx, mlx.nn as nn
        if not mx.metal.is_available():
            raise ImportError("no Metal")
        import mlx_vlm.models.qwen4_exp.language as qwen4
        from unsloth_zoo.mlx import inference as zoo
    except ImportError as e:
        result.update(skipped=True, error=str(e), finished=True)
        dump()
        print("JOB_RESULT", json.dumps(result))
        sys.exit(3)
    import importlib.metadata as md
    result["versions"] = {p: md.version(p) for p in ("mlx", "mlx-vlm", "mlx-lm")}
    result["device"] = mx.metal.device_info().get("device_name") if hasattr(mx.metal, "device_info") else None

    mx.random.seed(0)
    config = SimpleNamespace(hidden_size=2560, hc_count=4, hc_lowrank=320, rms_norm_eps=1e-6)

    def make(quantize):
        m = qwen4.Qwen4ExpGatedResidual(config, use_combine=True)
        m.update(nn.utils.tree_map(lambda p: mx.random.normal(p.shape) * 0.05, m.parameters()))
        m.set_dtype(mx.bfloat16)
        if quantize:
            nn.quantize(m, group_size=64, bits=4)
        m.eval()
        mx.eval(m.parameters())
        return m

    layers = [make(True) for _ in range(a.layers)]
    model = nn.Module()
    model.layers = layers
    prefill_mod = make(False)
    forward = qwen4._QWEN4_BATCH_INVARIANT_FORWARD
    xs = [(mx.random.normal((1, 1, 4 * 2560))).astype(mx.bfloat16) for _ in range(4)]
    xp = mx.random.normal((1, a.prefill_rows, 4 * 2560)).astype(mx.bfloat16)
    mx.eval(xs, xp)

    def decode_step():
        outs = [forward._hyper_connection(m, x) for m in layers for x in xs]
        mx.eval(outs)
        return outs

    def prefill_step():
        out = prefill_mod(xp)
        mx.eval(out)
        return out

    def timed(scope, env, fn, target):
        os.environ[env] = "1" if scope else "0"
        with getattr(zoo, "fused_qwen4_decode" if "DECODE" in env else "fused_qwen4_prefill")(target):
            hooked = ("_hyper_connection" in vars(forward)) if "DECODE" in env else type(target) is not qwen4.Qwen4ExpGatedResidual
            fn()  # warm (first use verifies)
            t = time.perf_counter()
            for _ in range(5):
                out = fn()
            return (time.perf_counter() - t) / 5, out, hooked

    def flat(o):
        return [v for _, v in nn.utils.tree_flatten(o)]

    for name, env, fn, target in (("decode", "UNSLOTH_MLX_QWEN4_DECODE", decode_step, model),
                                  ("prefill", "UNSLOTH_MLX_QWEN4_PREFILL", prefill_step, prefill_mod)):
        ratios, fused_t, native_t, equal, engaged = [], [], [], True, True
        for r in range(a.rounds):
            order = (True, False) if r % 2 == 0 else (False, True)
            got = {}
            for scope in order:
                got[scope] = timed(scope, env, fn, target)
            fused_t.append(got[True][0]); native_t.append(got[False][0])
            ratios.append(got[False][0] / got[True][0])
            engaged &= got[True][2] and not got[False][2]
            for u, v in zip(flat(got[True][1]), flat(got[False][1])):
                equal &= u.dtype == v.dtype and u.shape == v.shape and bool(mx.array_equal(u.view(mx.uint8), v.view(mx.uint8)).item())
        result["summary"][name] = {
            "speedup_median": round(statistics.median(ratios), 4),
            "speedup_min": round(min(ratios), 4), "speedup_max": round(max(ratios), 4),
            "fused_ms_median": round(1e3 * statistics.median(fused_t), 3),
            "native_ms_median": round(1e3 * statistics.median(native_t), 3),
            "reads_per_step": len(layers) * len(xs) if name == "decode" else 1}
        result["checks"][f"{name}_bitwise_equal"] = {"ok": equal, "detail": "fused vs native outputs, every round"}
        result["checks"][f"{name}_engaged"] = {"ok": engaged, "detail": "fused route installed only with the switch on"}
        print(name, json.dumps(result["summary"][name]), flush=True)
    result["passed"] = all(c["ok"] for c in result["checks"].values())
    result["finished"] = True
    dump()
    print("JOB_RESULT", json.dumps(result))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
