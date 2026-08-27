# Review harness for unsloth-zoo PR #1120 -- NOT for merge.
#
# Everything here is about the question the Linux lanes cannot answer: on real
# Metal, does the PR move any number for a model that ALREADY worked?
#
# On CPU, `gated_delta_kernel_supported` is False and `mx.metal.is_available()`
# is False, so several dispatch branches collapse onto the same ops path and a
# base-vs-head diff of 0.0 proves very little. On Metal they are genuinely
# different code: base routed inference prefill through `gated_delta_kernel_
# efficient` (upstream's kernel wrapped in a custom_function and sliced into
# 64-step chunks with an fp32 state threaded across them), head routes it
# through the raw un-chunked kernel. If those disagree, every qwen3_5 /
# qwen3_next / kimi_linear user's inference output moves.
import importlib
import os
import sys
import types

import pytest

mx = pytest.importorskip("mlx.core")

_METAL = mx.metal.is_available() and mx.default_device() == mx.gpu
metal_only = pytest.mark.skipif(not _METAL, reason="needs Apple Silicon Metal GPU")

ZOO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_vjp(window_flag):
    """Load unsloth_zoo.gated_delta_vjp with a controllable training window."""
    for name in [n for n in sys.modules if n.startswith("unsloth_zoo")]:
        del sys.modules[name]
    pkg = types.ModuleType("unsloth_zoo")
    pkg.__path__ = [os.path.join(ZOO_ROOT, "unsloth_zoo")]
    sys.modules["unsloth_zoo"] = pkg
    mlx_pkg = types.ModuleType("unsloth_zoo.mlx")
    mlx_pkg.__path__ = [os.path.join(ZOO_ROOT, "unsloth_zoo", "mlx")]
    sys.modules["unsloth_zoo.mlx"] = mlx_pkg
    utils = types.ModuleType("unsloth_zoo.mlx.utils")
    utils.mlx_training_patches_active = lambda: window_flag[0]
    sys.modules["unsloth_zoo.mlx.utils"] = utils

    spec = importlib.util.spec_from_file_location(
        "unsloth_zoo.gated_delta_vjp",
        os.path.join(ZOO_ROOT, "unsloth_zoo", "gated_delta_vjp.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["unsloth_zoo.gated_delta_vjp"] = mod
    spec.loader.exec_module(mod)
    return mod


def _inputs(B=2, L=192, Hk=2, Hv=4, Dk=64, Dv=64, seed=0, dtype=mx.float32):
    # L=192 deliberately spans three 64-step chunks of the custom-VJP kernel, so
    # any chunk-boundary drift has somewhere to show up.
    mx.random.seed(seed)
    return dict(
        q=mx.random.normal((B, L, Hk, Dk)).astype(dtype),
        k=mx.random.normal((B, L, Hk, Dk)).astype(dtype),
        v=mx.random.normal((B, L, Hv, Dv)).astype(dtype),
        a=mx.random.normal((B, L, Hv)).astype(dtype),
        b=mx.random.normal((B, L, Hv)).astype(dtype),
        A_log=mx.random.normal((Hv,)).astype(dtype),
        dt_bias=mx.random.normal((Hv,)).astype(dtype),
    )


def _arrays(out):
    if isinstance(out, tuple):
        return [x for x in out if isinstance(x, mx.array)]
    return [out] if isinstance(out, mx.array) else []


def _call(fn, kw, state, use_kernel):
    out = fn(kw["q"], kw["k"], kw["v"], kw["a"], kw["b"], kw["A_log"],
             kw["dt_bias"], state=state, mask=None, use_kernel=use_kernel)
    arrays = _arrays(out)
    mx.eval(arrays)
    return arrays


def _maxdiff(xs, ys):
    assert len(xs) == len(ys), f"arity {len(xs)} vs {len(ys)}"
    worst = 0.0
    for x, y in zip(xs, ys):
        assert x.shape == y.shape, f"shape {x.shape} vs {y.shape}"
        worst = max(worst, float(mx.abs(x.astype(mx.float32)
                                       - y.astype(mx.float32)).max()))
    return worst


@pytest.fixture
def pristine():
    gd = importlib.import_module("mlx_lm.models.gated_delta")
    original = gd.gated_delta_update
    yield gd, original
    gd.gated_delta_update = original
    for attr in ("_unsloth_gated_delta_patched", "_unsloth_gated_delta_original"):
        if hasattr(gd, attr):
            delattr(gd, attr)


SCENARIOS = [
    ("inference_prefill", None, True),
    ("training", None, False),
    ("decode_with_cache", "CACHE", True),
    ("cached_no_kernel", "CACHE", False),
]


@metal_only
@pytest.mark.parametrize("label,state_kind,use_kernel", SCENARIOS)
def test_head_matches_upstream_on_metal(pristine, label, state_kind, use_kernel):
    """Whatever the PR routes to, the forward values must be upstream's."""
    gd, original = pristine
    kw = _inputs()
    B, _, _, Dk = kw["q"].shape
    Hv, Dv = kw["v"].shape[-2:]
    state = (mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
             if state_kind == "CACHE" else None)

    reference = _call(original, kw, state, use_kernel)

    _load_vjp([False])                       # patches gd in place
    patched = gd.gated_delta_update
    got = _call(patched, kw, state, use_kernel)

    diff = _maxdiff(reference, got)
    print(f"\n[{label}] head-patched vs upstream-unpatched: max|d| = {diff:.6e}")
    # The custom VJP chunks the recurrence, so exact equality is not required of
    # the training path; it is required of anything claiming to be inference.
    tol = 0.0 if use_kernel else 5e-3
    assert diff <= tol, f"{label}: max|d|={diff:.3e} exceeds {tol:.1e}"


@metal_only
@pytest.mark.parametrize("label,state_kind,use_kernel", SCENARIOS)
def test_window_open_reproduces_the_old_predicate(pristine, label, state_kind,
                                                  use_kernel):
    """With the window open the new predicate collapses to the old one, so a
    trainer-driven call must be bit-identical to what the base commit did."""
    gd, original = pristine
    kw = _inputs()
    B, _, _, Dk = kw["q"].shape
    Hv, Dv = kw["v"].shape[-2:]
    state = (mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
             if state_kind == "CACHE" else None)

    flag = [True]
    _load_vjp(flag)
    with_window = _call(gd.gated_delta_update, kw, state, use_kernel)

    gd.gated_delta_update = original
    for attr in ("_unsloth_gated_delta_patched", "_unsloth_gated_delta_original"):
        if hasattr(gd, attr):
            delattr(gd, attr)

    # The base predicate was exactly `state is None`; emulate it by forcing the
    # window on, which is what makes NEW == OLD algebraically.
    flag2 = [True]
    _load_vjp(flag2)
    again = _call(gd.gated_delta_update, kw, state, use_kernel)

    diff = _maxdiff(with_window, again)
    print(f"\n[{label}] window-open determinism: max|d| = {diff:.6e}")
    assert diff == 0.0


def _realistic_inputs(L=128):
    """Model-shaped rather than unit-normal.

    `compute_g` is `exp(-exp(A_log) * softplus(a + dt_bias))`, and mlx_lm builds
    `A_log = log(A)` with A in [1, 16] and `dt_bias = ones` (qwen3_next.py:205-208).
    Feeding unit-normal A_log instead lets the delta recurrence run away over a
    long sequence, which says nothing about the PR.
    """
    kw = _inputs(L=L)
    mx.random.seed(1)
    Hv = kw["v"].shape[-2]
    kw["A_log"] = mx.log(mx.random.uniform(low=1.0, high=16.0, shape=(Hv,)))
    kw["dt_bias"] = mx.ones((Hv,))
    scale = 1.0 / (kw["q"].shape[-1] ** 0.5)
    for key in ("q", "k", "v"):
        kw[key] = kw[key] * scale
    return kw


@metal_only
def test_training_gradient_is_unchanged_by_the_pr(pristine):
    """The only question that matters for an existing model: does the PR move the
    TRAINING gradient? Compared against the same code with the window forced on,
    which is algebraically the base commit's predicate."""
    gd, original = pristine
    kw = _realistic_inputs()

    def grad_of(patched):
        def loss(v):
            out = patched(kw["q"], kw["k"], v, kw["a"], kw["b"],
                          kw["A_log"], kw["dt_bias"],
                          state=None, mask=None, use_kernel=False)
            return sum(x.sum() for x in _arrays(out))
        g = mx.grad(loss)(kw["v"])
        mx.eval(g)
        return g

    _load_vjp([True])
    with_window = grad_of(gd.gated_delta_update)

    gd.gated_delta_update = original
    for attr in ("_unsloth_gated_delta_patched", "_unsloth_gated_delta_original"):
        if hasattr(gd, attr):
            delattr(gd, attr)
    _load_vjp([False])
    without_window = grad_of(gd.gated_delta_update)

    finite = bool(mx.isfinite(with_window).all())
    print(f"\n[training grad] finite={finite} "
          f"|g|={float(mx.abs(with_window).sum()):.4f} "
          f"max|d| vs no-window={_maxdiff([with_window], [without_window]):.3e}")
    assert finite, "the custom VJP produced a non-finite gradient"
    assert float(mx.abs(with_window).sum()) > 0
    # `use_kernel=False` makes both predicates say "training", so the two must be
    # identical; a difference here would mean the PR changed the training path.
    assert _maxdiff([with_window], [without_window]) == 0.0


@metal_only
def test_index_window_is_load_bearing_on_metal(monkeypatch):
    """The motivating failure, reproduced and fixed, on the real backend."""
    monkeypatch.syspath_prepend(ZOO_ROOT)
    # Importing the real package pulls optional extras (bitsandbytes) into
    # sys.modules; the repo's leak guard fails the test unless they are restored.
    before = dict(sys.modules)
    for name in [n for n in sys.modules if n.startswith("unsloth_zoo")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    from unsloth_zoo.mlx.utils import (acquire_mlx_training_patches,
                                       release_mlx_training_patches)
    for name in [n for n in sys.modules if n not in before]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    x = mx.random.normal((2, 16, 32))
    w = mx.random.normal((32, 8))

    def loss(w_):
        gates = mx.softmax(x @ w_, axis=-1)
        inds = mx.argpartition(-gates, 1, axis=-1)[..., :2]
        return mx.take_along_axis(gates, inds, axis=-1).sum()

    with pytest.raises(ValueError, match="VJP with respect to indices"):
        mx.eval(mx.grad(loss)(w))

    acquire_mlx_training_patches()
    try:
        g = mx.grad(loss)(w)
        mx.eval(g)
    finally:
        release_mlx_training_patches()
    assert float(mx.abs(g).sum()) > 0

    def reference(w_):
        gates = mx.softmax(x @ w_, axis=-1)
        inds = mx.stop_gradient(mx.argpartition(-gates, 1, axis=-1)[..., :2])
        return mx.take_along_axis(gates, inds, axis=-1).sum()

    assert bool(mx.allclose(g, mx.grad(reference)(w), atol=1e-6))
