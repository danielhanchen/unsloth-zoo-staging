"""Metal-backend probes for unsloth-zoo PR 962 (8-bit first-moment MLX optimizer).

Staging-only review harness. Four probes, each printed as a PROBE/RESULT block and
each returning a nonzero contribution to the exit code when it fails:

  1. zero_gradient_invariant  -- a coordinate whose gradient is identically zero must
     not move. Plain Adam guarantees this; affine per-group int8 does not.
  2. dead_zone                -- how much of the stored first moment survives when a
     64-wide group spans several decades.
  3. state_bytes              -- measured optimizer-state reduction for fp32 vs bf16.
  4. peak_memory              -- mx.get_peak_memory() around a training step, which is
     0 on the CPU backend and therefore only measurable here.

Run: python ci_probes/pr962_metal_probe.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from unsloth_zoo.mlx.optimizers_quantized import QuantizedMomentAdam, QuantizedMomentAdamW

FAILURES = []


def _header(name):
    print(f"\n{'=' * 78}\nPROBE: {name}\n{'=' * 78}", flush=True)


def _fail(name, msg):
    FAILURES.append(name)
    print(f"RESULT: FAIL [{name}] {msg}", flush=True)


def _ok(name, msg):
    print(f"RESULT: PASS [{name}] {msg}", flush=True)


def _drive(opt, grad, steps, shape):
    """Externally driven optimizer, no model. Returns the final parameter."""
    param = mx.zeros(shape)
    opt.init({"w": param})
    for _ in range(steps):
        param = opt.apply_gradients({"w": grad}, {"w": param})["w"]
        mx.eval(param, opt.state)
    return param


# 1 ---------------------------------------------------------------------------
def probe_zero_gradient_invariant(steps=500, lr=1e-3):
    name = "zero_gradient_invariant"
    _header(name)
    values = [1.0, 0.0, 1e-2, 1e-3, 1e-4, 0.0]
    grad = mx.array([values + [0.0] * (64 - len(values))])

    ref = _drive(optim.Adam(learning_rate=lr, bias_correction=True), grad, steps, (1, 64))
    got = _drive(QuantizedMomentAdam(lr, bias_correction=True), grad, steps, (1, 64))

    print(f"{'grad':>10} {'plain Adam':>14} {'quantized-m':>14}")
    for i, v in enumerate(values):
        print(f"{v:>10.0e} {float(ref[0, i]):>14.6f} {float(got[0, i]):>14.6f}")

    zero_cols = [i for i, v in enumerate(values) if v == 0.0] + list(range(len(values), 64))
    ref_zero = max(abs(float(ref[0, i])) for i in zero_cols)
    got_zero = max(abs(float(got[0, i])) for i in zero_cols)
    budget = lr * steps
    print(f"\nmax |displacement| of zero-gradient coordinates:")
    print(f"  plain Adam     {ref_zero:.6e}")
    print(f"  quantized-m    {got_zero:.6e}   (lr*steps budget = {budget:.3f})")

    if got_zero > 1e-6:
        _fail(name, f"zero-gradient coordinate moved {got_zero:.6f} ({got_zero / budget:.1f}x "
                    f"the lr*steps budget); plain Adam moved {ref_zero:.3e}")
    else:
        _ok(name, f"zero-gradient coordinates stayed put ({got_zero:.3e})")

    # Small-gradient attenuation, same run.
    print("\nsmall-gradient update ratio (quantized / plain):")
    worst = 1.0
    for i, v in enumerate(values):
        if v == 0.0:
            continue
        r = float(got[0, i]) / float(ref[0, i])
        worst = min(worst, r)
        print(f"  grad {v:.0e}  ratio {r:.4f}")
    if worst < 0.8:
        _fail(name + "/attenuation",
              f"a nonzero-gradient coordinate received only {worst:.1%} of its plain-Adam update")
    else:
        _ok(name + "/attenuation", f"worst ratio {worst:.4f}")


# 2 ---------------------------------------------------------------------------
def probe_dead_zone(group_sizes=(32, 64, 128)):
    name = "dead_zone"
    _header(name)
    mx.random.seed(0)
    # 64 columns per group, magnitudes spanning 4 decades within a group.
    rows, cols = 256, 512
    scale = mx.power(10.0, mx.random.uniform(-2, 2, (rows, cols)))
    m = mx.random.normal((rows, cols)) * scale
    mx.eval(m)

    print(f"{'group':>7} {'rel L2 err':>12} {'dead frac':>12} {'sign-flip':>12}")
    worst_dead = 0.0
    for g in group_sizes:
        d = mx.dequantize(*mx.quantize(m, group_size=g, bits=8), group_size=g, bits=8)
        err = float(mx.linalg.norm(d - m) / mx.linalg.norm(m))
        dead = float(mx.mean((d == 0) & (m != 0)))
        flip = float(mx.mean(mx.sign(d) != mx.sign(m)))
        worst_dead = max(worst_dead, dead)
        print(f"{g:>7} {err:>12.4%} {dead:>12.4%} {flip:>12.4%}")

    if worst_dead > 0.02:
        _fail(name, f"{worst_dead:.2%} of coordinates store as exactly zero (>2% threshold); "
                    "their momentum history is discarded every step")
    else:
        _ok(name, f"worst dead fraction {worst_dead:.4%}")


# 3 ---------------------------------------------------------------------------
def _state_bytes(state):
    return sum(v.nbytes for _, v in tree_flatten(state) if isinstance(v, mx.array))


def probe_state_bytes():
    name = "state_bytes"
    _header(name)
    print(f"{'dtype':>10} {'plain AdamW':>14} {'quantized':>14} {'reduction':>11}")
    results = {}
    for dtype, label in ((mx.float32, "float32"), (mx.bfloat16, "bfloat16")):
        rows = []
        for make in (lambda: optim.AdamW(learning_rate=1e-3, bias_correction=True),
                     lambda: QuantizedMomentAdamW(1e-3, bias_correction=True)):
            opt = make()
            w = mx.zeros((512, 512), dtype=dtype)
            g = mx.ones((512, 512), dtype=dtype)
            opt.init({"w": w})
            opt.apply_gradients({"w": g}, {"w": w})
            mx.eval(opt.state)
            rows.append(_state_bytes(opt.state))
        red = 1.0 - rows[1] / rows[0]
        results[label] = red
        print(f"{label:>10} {rows[0]:>14,} {rows[1]:>14,} {red:>11.2%}")

    print("\nNote: MLX moments inherit the parameter dtype (mx.zeros_like), so a bf16 run")
    print("cannot reach the fp32 figure. The PR abstract quotes the fp32 number.")
    _ok(name, f"fp32 {results['float32']:.2%}, bf16 {results['bfloat16']:.2%}")


# 4 ---------------------------------------------------------------------------
def probe_peak_memory(steps=20):
    name = "peak_memory"
    _header(name)
    if not hasattr(mx, "get_peak_memory"):
        _ok(name, "mx.get_peak_memory unavailable; skipped")
        return

    def measure(make_opt):
        mx.random.seed(0)
        model = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 2048))
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())
        x = mx.random.normal((16, 2048)).astype(mx.bfloat16)
        y = mx.random.normal((16, 2048)).astype(mx.bfloat16)
        opt = make_opt()
        gf = nn.value_and_grad(model, lambda m, x, y: ((m(x) - y) ** 2).mean())
        for _ in range(3):  # warmup
            _, g = gf(model, x, y)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
        mx.reset_peak_memory()
        for _ in range(steps):
            _, g = gf(model, x, y)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
        return mx.get_peak_memory(), _state_bytes(opt.state)

    ref_peak, ref_state = measure(lambda: optim.AdamW(learning_rate=1e-3, bias_correction=True))
    got_peak, got_state = measure(lambda: QuantizedMomentAdamW(1e-3, bias_correction=True))

    print(f"{'arm':>14} {'peak bytes':>16} {'state bytes':>14}")
    print(f"{'adamw':>14} {ref_peak:>16,} {ref_state:>14,}")
    print(f"{'adamw_8bit':>14} {got_peak:>16,} {got_state:>14,}")
    peak_change = got_peak / ref_peak - 1.0
    state_change = got_state / ref_state - 1.0
    print(f"\npeak memory  {peak_change:+.2%}  (positive means adamw_8bit uses MORE)")
    print(f"state bytes  {state_change:+.2%}")
    print("apply_single dequantizes m to full width every step, so a saving in persistent")
    print("state does not imply a saving in peak allocation.")
    if peak_change > 0.02:
        _fail(name, f"peak memory rose {peak_change:+.2%} while state fell {state_change:+.2%}")
    else:
        _ok(name, f"peak {peak_change:+.2%}, state {state_change:+.2%}")


def main():
    print(f"mlx default device: {mx.default_device()}", flush=True)
    probe_zero_gradient_invariant()
    probe_dead_zone()
    probe_state_bytes()
    probe_peak_memory()

    print(f"\n{'=' * 78}")
    if FAILURES:
        print(f"FAILED PROBES: {', '.join(FAILURES)}")
        return 1
    print("ALL PROBES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
