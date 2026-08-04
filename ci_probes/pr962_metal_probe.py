"""Metal-backend verification for unsloth-zoo PR 962 after the fixes.

Checks the two things the fixes claim: a zero-gradient coordinate no longer
moves, and a checkpoint written by plain AdamW is re-packed on resume. Also
reports the state-byte saving and the peak-allocation cost per dtype.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from unsloth_zoo.mlx.optimizers_quantized import QuantizedMomentAdam, QuantizedMomentAdamW
from unsloth_zoo.mlx.utils import load_optimizer_state, save_optimizer_state

FAILED = []


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'} [{name}] {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def _drive(opt, grad, steps, shape):
    param = mx.zeros(shape)
    opt.init({"w": param})
    for _ in range(steps):
        param = opt.apply_gradients({"w": grad}, {"w": param})["w"]
        mx.eval(param, opt.state)
    return param


def zero_gradient_invariant(steps=500, lr=1e-3):
    print("\n== zero-gradient invariant ==")
    for outlier in (1.0, 10.0, 100.0):
        grad = mx.array([[outlier] + [0.0] * 63])
        ref = _drive(optim.Adam(learning_rate=lr, bias_correction=True), grad, steps, (1, 64))
        got = _drive(QuantizedMomentAdam(lr, bias_correction=True), grad, steps, (1, 64))
        moved = max(abs(float(got[0, i])) for i in range(1, 64))
        live = abs(float(got[0, 0]) - float(ref[0, 0]))
        check(f"zero_grad/outlier={outlier:.0f}", moved == 0.0 and live < 1e-6,
              f"zero-grad drift {moved:.3e} (was 1.933 on Metal before the fix), "
              f"live coordinate matches stock Adam to {live:.3e}")


def plain_checkpoint_migrates():
    print("\n== plain AdamW checkpoint resumed as 8-bit ==")
    mx.random.seed(0)
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
    mx.eval(model.parameters())
    mx.random.seed(1)
    x, y = mx.random.normal((32, 256)), mx.random.normal((32, 256))
    gf = nn.value_and_grad(model, lambda m, x, y: ((m(x) - y) ** 2).mean())

    plain = optim.Adam(learning_rate=1e-3, bias_correction=True)
    for _ in range(4):
        _, g = gf(model, x, y)
        plain.update(model, g)
        mx.eval(model.parameters(), plain.state)
    ckpt = tempfile.mkdtemp()
    save_optimizer_state(plain, ckpt)

    packed = QuantizedMomentAdam(1e-3, bias_correction=True)
    _, g = gf(model, x, y)
    packed.update(model, g)
    load_optimizer_state(packed, ckpt)
    _, g = gf(model, x, y)
    packed.update(model, g)
    mx.eval(model.parameters(), packed.state)
    moment = packed.state["layers"][0]["weight"]["m"]
    check("plain_checkpoint_migrates", isinstance(moment, (tuple, list)),
          f"first moment after resume is {type(moment).__name__} (was array before the fix)")


def state_and_peak():
    print("\n== state bytes and peak allocation ==")
    def bytes_of(state):
        return sum(v.nbytes for _, v in tree_flatten(state) if isinstance(v, mx.array))

    for dtype, label in ((mx.float32, "float32"), (mx.bfloat16, "bfloat16")):
        sizes = []
        for make in (lambda: optim.AdamW(learning_rate=1e-3, bias_correction=True),
                     lambda: QuantizedMomentAdamW(1e-3, bias_correction=True)):
            opt = make()
            w = mx.zeros((512, 512), dtype=dtype)
            opt.init({"w": w})
            opt.apply_gradients({"w": mx.ones((512, 512), dtype=dtype)}, {"w": w})
            mx.eval(opt.state)
            sizes.append(bytes_of(opt.state))
        print(f"  {label}: {sizes[0]:,} -> {sizes[1]:,} ({1 - sizes[1] / sizes[0]:.2%} smaller)")

    if not hasattr(mx, "get_peak_memory"):
        return

    def peak(make_opt, steps=20):
        mx.random.seed(0)
        model = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 2048))
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())
        x = mx.random.normal((16, 2048)).astype(mx.bfloat16)
        y = mx.random.normal((16, 2048)).astype(mx.bfloat16)
        opt = make_opt()
        gf = nn.value_and_grad(model, lambda m, x, y: ((m(x) - y) ** 2).mean())
        for _ in range(3):
            _, g = gf(model, x, y)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
        mx.reset_peak_memory()
        for _ in range(steps):
            _, g = gf(model, x, y)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
        return mx.get_peak_memory()

    a = peak(lambda: optim.AdamW(learning_rate=1e-3, bias_correction=True))
    b = peak(lambda: QuantizedMomentAdamW(1e-3, bias_correction=True))
    print(f"  peak allocation {a:,} -> {b:,} ({b / a - 1:+.2%}); documented, not a regression")


def main():
    print(f"device {mx.default_device()}", flush=True)
    zero_gradient_invariant()
    plain_checkpoint_migrates()
    state_and_peak()
    print()
    if FAILED:
        print(f"FAILED: {', '.join(FAILED)}")
        return 1
    print("all Metal checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
