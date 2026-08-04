"""Two Unsloth fixes collide on Gemma3N vision; the run should not die of it.

One fix wraps the gradient-checkpointing `requires_grad` hooks in
`torch.compiler.disable`, because Dynamo cannot trace
`Tensor.requires_grad_()`. Another compiles
`Gemma3nMultimodalEmbedder_forward` with `fullgraph = True`. The disabled
hook is then invoked from inside that fullgraph region and Dynamo refuses:

    Unsupported: Skip calling `torch.compiler.disable()`d function
      Explanation: Skip calling function
        `requires_grad_for_gradient_checkpointing.<locals>.requires_grad_pre_hook`
        since it was wrapped with `torch.compiler.disable`
      Hint: Remove the `torch.compiler.disable` call

Observed on Gemma3N_(4B)-Vision, which reached cell 15 and then stopped.
Neither fix is wrong on its own, and a user can do nothing about the
combination, so it should cost speed rather than the run.

The hazard being guarded against is over-catching. The existing fallback is
deliberately narrow -- it takes cache-exhaustion errors only, so that a
genuine graph break under `fullgraph` still raises and stays visible. This
adds exactly one more case, matched on the disable signature, and every test
below that matters is about the cases that must STILL raise.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth_zoo.temporary_patches import utils as U  # noqa: E402

DISABLE_MSG = (
    "Skip calling `torch.compiler.disable()`d function\n"
    "  Explanation: Skip calling function "
    "`<function requires_grad_for_gradient_checkpointing.<locals>."
    "requires_grad_pre_hook at 0x7f00>` since it was wrapped with "
    "`torch.compiler.disable` (reason: None)\n"
    "  Hint: Remove the `torch.compiler.disable` call"
)


# ---- the matcher ----------------------------------------------------------

def test_it_recognises_our_own_disabled_hook():
    assert U._is_our_own_disabled_hook(RuntimeError(DISABLE_MSG))


def test_an_ordinary_graph_break_is_not_matched():
    """The whole point. These must keep raising."""
    for text in (
        "Unsupported: call_function BuiltinVariable(print)",
        "Unsupported Tensor.requires_grad_() call",
        "Dynamo failed to trace a data-dependent branch",
        "graph break in user code",
    ):
        assert not U._is_our_own_disabled_hook(RuntimeError(text)), text


def test_a_mention_of_disable_alone_is_not_enough():
    """Both halves of the signature are required, so a message that merely
    talks about torch.compiler.disable does not qualify."""
    assert not U._is_our_own_disabled_hook(
        RuntimeError("consider using torch.compiler.disable here"))


def test_an_unstringifiable_exception_does_not_crash():
    class Bad(Exception):
        def __str__(self):
            raise ValueError("nope")
    assert U._is_our_own_disabled_hook(Bad()) is False


# ---- the fallback ---------------------------------------------------------

def _wrap(compiled, eager=None):
    eager = eager or (lambda *a, **k: "eager")
    return U._fall_back_to_eager_on_recompile_limit(compiled, eager, "TestMod")


def _unsupported(msg):
    import torch._dynamo.exc as exc
    cls = getattr(exc, "Unsupported", None)
    if cls is None:
        pytest.skip("this torch has no torch._dynamo.exc.Unsupported")
    try:
        return cls(msg)
    except Exception:
        pytest.skip("Unsupported cannot be constructed on this torch")


def test_our_disabled_hook_falls_back_to_eager():
    def compiled(*a, **k):
        raise _unsupported(DISABLE_MSG)
    assert _wrap(compiled)() == "eager"


def test_the_fallback_does_not_latch():
    """It used to latch, and that latch broke activation checkpointing.

    The flag was shared by every call site of the wrapped function, while
    checkpointing pairs each individual forward with its own recompute. An
    early layer packed while compiled, a later layer flipped the flag, and the
    early layer's recompute then ran eager -- saving a different set of
    intermediates, which torch detects and aborts on. Gemma4_(E2B)-Vision died
    exactly that way at the first backward.

    So the compiler must be re-entered per call. It costs a cheap raise once
    the cache is exhausted, and only in the already-degraded case.
    """
    calls = {"n": 0}

    def compiled(*a, **k):
        calls["n"] += 1
        raise _unsupported(DISABLE_MSG)

    w = _wrap(compiled)
    w(); w(); w()
    assert calls["n"] == 3, "each call must get its own attempt"


def test_a_call_site_that_still_compiles_keeps_compiling():
    """The property the checkpoint actually depends on.

    One failing call must not push every other call site onto the eager path,
    because those calls already packed their activations under the compiled
    graph and have to recompute the same way.
    """
    outcomes = iter(["ok", "fail", "ok", "ok"])
    seen = []

    def compiled(*a, **k):
        which = next(outcomes)
        seen.append(which)
        if which == "fail":
            raise _unsupported(DISABLE_MSG)
        return "compiled"

    w = _wrap(compiled)
    assert w() == "compiled"
    assert w() == "eager"      # the one that exhausted the cache
    assert w() == "compiled"   # must NOT have been latched onto eager
    assert w() == "compiled"


def test_it_warns_once_and_not_per_call(caplog):
    """The condition repeats every call; the log must not."""
    import logging

    def compiled(*a, **k):
        raise _unsupported(DISABLE_MSG)

    w = _wrap(compiled)
    with caplog.at_level(logging.WARNING):
        w(); w(); w(); w()
    warnings = [r for r in caplog.records if "eagerly" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]


def test_a_real_graph_break_still_raises():
    """The property the narrow match exists to protect."""
    def compiled(*a, **k):
        raise _unsupported("Unsupported: call_function on a data-dependent value")
    with pytest.raises(Exception) as ei:
        _wrap(compiled)()
    assert "data-dependent" in str(ei.value)


def test_an_unrelated_exception_still_raises():
    def compiled(*a, **k):
        raise ValueError("something else entirely")
    with pytest.raises(ValueError):
        _wrap(compiled)()


def test_a_successful_compile_is_untouched():
    assert _wrap(lambda *a, **k: "compiled")() == "compiled"


def test_arguments_reach_the_eager_function():
    def compiled(*a, **k):
        raise _unsupported(DISABLE_MSG)
    w = _wrap(compiled, eager=lambda x, y=0: x + y)
    assert w(3, y=4) == 7


# ---- what must be preserved ----------------------------------------------

def test_the_recompile_limit_fallback_still_works():
    errs = U._recompile_limit_errors()
    if not errs:
        pytest.skip("no recompile-limit exceptions on this torch")

    def compiled(*a, **k):
        raise errs[0]("recompile_limit reached with fullgraph=True")
    assert _wrap(compiled)() == "eager"


def test_the_compiled_callable_stays_reachable():
    """Anything that unwraps the wrapper must still find it."""
    def compiled(*a, **k):
        return "compiled"
    compiled.get_compiler_config = lambda: {}
    w = _wrap(compiled)
    assert w._unsloth_compiled_func is compiled
    assert hasattr(w, "get_compiler_config")


def test_it_degrades_to_the_compiled_function_when_torch_offers_neither():
    """On a torch with no such exceptions at all, wrapping buys nothing and
    must not add a layer."""
    import unittest.mock as mock
    with mock.patch.object(U, "_recompile_limit_errors", lambda: ()), \
         mock.patch.object(U, "_disabled_hook_graph_break_error", lambda: ()):
        sentinel = object()
        assert U._fall_back_to_eager_on_recompile_limit(
            sentinel, lambda: None, "x") is sentinel


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
