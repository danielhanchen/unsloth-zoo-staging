# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A call site about to exhaust its recompile budget goes eager BETWEEN steps.

Deferring the switch to the next step boundary is right when the budget runs
out during a forward. It cannot help when it runs out during a checkpoint
RECOMPUTE: that step's forward has already packed compiled activations, and the
recompute that ran out of budget is the one those activations are compared
against, so torch aborts the backward with

    AssertionError: Something went unexpectedly wrong in activation checkpoint

inside the very step the deferral promised to fix at its end. Measured on a
Kaggle T4 with gemma-4-E2B-it, whose 504 RMSNorms share one code object.

The step in which the budget runs out is therefore already lost when the
exhaustion becomes observable, so the decision has to be made before the step:
Dynamo's cache occupancy and the previous step's consumption of it are both
readable at a boundary, and they say whether the step ahead fits.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unsloth_zoo.temporary_patches import utils as U  # noqa: E402


class Boom(Exception):
    pass


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    """Every test gets its own registry, label sets and headroom history.

    All of them are module-global and permanent by design, so without this one
    test's saturated call site answers for the next one's healthy one.
    """
    monkeypatch.setattr(U, "_EAGER_FALLBACK_WRAPPERS", [])
    monkeypatch.setattr(U, "_LATCHED_EAGER_LABELS", set())
    monkeypatch.setattr(U, "_PENDING_EAGER_LABELS", set())
    monkeypatch.setattr(U, "_RECENT_EAGER_LABELS", set())
    monkeypatch.setattr(U, "_COMPILED_OK_LABELS", set())
    monkeypatch.setattr(U, "_RECOMPILE_HEADROOM", {})
    monkeypatch.setattr(U, "_PREEMPTIVE_EAGER_LABELS", set())
    monkeypatch.setattr(U, "_EAGER_FALLBACK_PRUNE_AT", 64)


def _wrapper(label = "M.forward", fail_after = 10 ** 9):
    """A wrapped call site, plus the tally of how each call was served."""
    calls = {"compiled": 0, "eager": 0}

    def compiled(x):
        calls["compiled"] += 1
        if calls["compiled"] > fail_after:
            raise Boom("recompile_limit reached with fullgraph=True")
        return "compiled"

    def eager(x):
        calls["eager"] += 1
        return "eager"

    wrapper = U._fall_back_to_eager_on_recompile_limit(compiled, eager, label)
    return wrapper, calls


def _fake_cache(monkeypatch, readings, budgets = (8, 256)):
    """Stand in for Dynamo, handing out one `(total, per frame)` per boundary."""
    remaining = list(readings)
    monkeypatch.setattr(U, "_recompile_budgets", lambda: budgets)
    monkeypatch.setattr(
        U, "_recompile_cache_occupancy",
        lambda code: remaining.pop(0) if remaining else readings[-1],
    )


# ---- the decision --------------------------------------------------------

def test_a_call_site_that_ate_its_budget_last_step_is_eager_before_the_next():
    """The whole point. Nothing has failed yet; the cache says it is about to."""
    wrapper, calls = _wrapper()
    with pytest.MonkeyPatch.context() as patch:
        # Empty at the first boundary, then 6 of 8 gone after one step: two
        # left, and the last step wanted six.
        _fake_cache(patch, [(0, 0), (6, 6)])
        assert U.apply_pending_eager_fallbacks() == 0
        assert wrapper(1) == "compiled"
        assert U.apply_pending_eager_fallbacks() == 1
    assert wrapper(1) == "eager"
    assert "M.forward" in U._LATCHED_EAGER_LABELS
    assert "M.forward" in U._PREEMPTIVE_EAGER_LABELS
    assert calls["compiled"] == 1


def test_the_first_boundary_only_measures():
    """One reading is an occupancy, not a rate. A model may compile several
    variants during warmup and then never need another, and taking that as
    evidence of an exhaustion to come would un-compile it for nothing."""
    wrapper, _ = _wrapper()
    with pytest.MonkeyPatch.context() as patch:
        _fake_cache(patch, [(7, 7)])
        assert U.apply_pending_eager_fallbacks() == 0
    assert wrapper(1) == "compiled"
    assert U._LATCHED_EAGER_LABELS == set()


def test_a_settled_call_site_keeps_its_compilation():
    """The healthy path, and the one this must not cost anything. A cache that
    filled during warmup and has not moved since consumes nothing per step, and
    nothing consumed can never predict an exhaustion however full it looks."""
    wrapper, calls = _wrapper()
    with pytest.MonkeyPatch.context() as patch:
        _fake_cache(patch, [(7, 7), (7, 7), (7, 7), (7, 7)])
        for _ in range(4):
            assert U.apply_pending_eager_fallbacks() == 0
            assert wrapper(1) == "compiled"
    assert calls["eager"] == 0
    assert U._LATCHED_EAGER_LABELS == set()


def test_a_full_cache_latches_even_with_nothing_consumed():
    """No room at all is not a prediction: the next new shape raises, and under
    a checkpoint that raise lands in the middle of a packed step."""
    wrapper, _ = _wrapper()
    with pytest.MonkeyPatch.context() as patch:
        _fake_cache(patch, [(8, 8), (8, 8)])
        U.apply_pending_eager_fallbacks()
        assert U.apply_pending_eager_fallbacks() == 1
    assert wrapper(1) == "eager"


def test_headroom_handed_back_is_not_read_as_consumption():
    """A restored bump RAISES the headroom between two boundaries. Left
    unclamped that is a negative consumption, which would let a call site with
    no room left look like it had spare."""
    wrapper, _ = _wrapper()
    with pytest.MonkeyPatch.context() as patch:
        _fake_cache(patch, [(8, 8), (4, 4)])
        U.apply_pending_eager_fallbacks()
        assert U.apply_pending_eager_fallbacks() == 0
    assert wrapper(1) == "compiled"


def test_the_two_budgets_are_both_respected():
    """`recompile_limit` counts what one frame can see, and
    `accumulated_recompile_limit` counts the whole cache. Either one running out
    raises, so the headroom is the smaller of the two."""
    wrapper, _ = _wrapper()
    with pytest.MonkeyPatch.context() as patch:
        # Miles of room per frame, almost none in the accumulated total: 504
        # RMSNorm instances against a 256 accumulated limit is exactly this.
        _fake_cache(patch, [(250, 1), (255, 1)], budgets = (8, 256))
        U.apply_pending_eager_fallbacks()
        assert U.apply_pending_eager_fallbacks() == 1
    assert wrapper(1) == "eager"


# ---- what it must not disturb -------------------------------------------

def test_a_boundary_latch_is_not_recorded_as_a_mid_step_flip():
    """`_RECENT_EAGER_LABELS` answers "did the compile mode change DURING this
    step", which is what tells a caller a checkpoint failure is worth retrying.
    A boundary latch changes it BETWEEN steps, so every pack and recompute in
    the step ahead agree and a failure inside it belongs to somebody else."""
    _held, _ = _wrapper()          # held: the registry is weak
    with pytest.MonkeyPatch.context() as patch:
        _fake_cache(patch, [(0, 0), (6, 6)])
        U.apply_pending_eager_fallbacks()
        U.apply_pending_eager_fallbacks()
    assert U._RECENT_EAGER_LABELS == set()
    assert U._PENDING_EAGER_LABELS == set()


def test_a_rebuilt_wrapper_stays_eager():
    """Labels outlive wrappers on purpose -- GRPO rebuilds `accumulate_chunk`
    inside every backward -- so the decision has to be recorded by label."""
    original, _ = _wrapper()          # held: the registry is weak
    with pytest.MonkeyPatch.context() as patch:
        _fake_cache(patch, [(0, 0), (6, 6)])
        U.apply_pending_eager_fallbacks()
        U.apply_pending_eager_fallbacks()
    rebuilt, calls = _wrapper()
    assert rebuilt(1) == "eager"
    assert calls["compiled"] == 0


def test_nothing_happens_when_torch_cannot_report_its_cache():
    """`_debug_get_cache_entry_list` is private and the guard internals move
    between releases. Unreadable has to mean "no prediction", not an error and
    not a guess."""
    wrapper, _ = _wrapper()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(U, "_recompile_cache_occupancy", lambda code: None)
        for _ in range(3):
            assert U.apply_pending_eager_fallbacks() == 0
    assert wrapper(1) == "compiled"


def test_nothing_happens_when_the_budgets_cannot_be_read():
    wrapper, _ = _wrapper()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(U, "_recompile_budgets", lambda: None)
        for _ in range(3):
            assert U.apply_pending_eager_fallbacks() == 0
    assert wrapper(1) == "compiled"


def test_a_call_site_with_no_code_object_is_skipped():
    """A callable object or a `functools.partial` has no `__code__`, so there is
    no Dynamo cache to read and no prediction to make."""

    class Callable:
        def __call__(self, x): return "eager"

    wrapper = U._fall_back_to_eager_on_recompile_limit(
        lambda x: "compiled", Callable(), "M.callable")
    assert wrapper._unsloth_eager_code is None
    with pytest.MonkeyPatch.context() as patch:
        _fake_cache(patch, [(8, 8), (8, 8)])
        U.apply_pending_eager_fallbacks()
        assert U.apply_pending_eager_fallbacks() == 0
    assert wrapper(1) == "compiled"


# ---- reading the real cache ---------------------------------------------

def test_the_occupancy_counts_what_dynamo_counts():
    """Against a real Dynamo cache, not a stand-in. `backend = "eager"` keeps it
    to tracing, which is where the cache entries are made."""

    def kernel(x, weight):
        return x * weight

    torch._dynamo.reset()
    compiled = torch.compile(kernel, fullgraph = True, dynamic = False,
                             backend = "eager")
    for width in (3, 5, 7):
        compiled(torch.randn(2, width), torch.randn(width))
    occupancy = U._recompile_cache_occupancy(kernel.__code__)
    assert occupancy is not None, "torch stopped exposing its cache entries"
    total, per_frame = occupancy
    assert total == 3
    # No ID_MATCHed objects for a plain tensor kernel, so every entry counts
    # against every frame and the two numbers agree.
    assert per_frame == 3


def test_entries_matched_to_different_objects_do_not_crowd_one_frame():
    """`recompile_limit` counts only the entries whose ID_MATCHed objects are
    the frame's own. Counting the whole cache instead would un-compile a model
    with many module instances on a torch that still ID_MATCHes them."""

    class Entry:
        def __init__(self, matched):
            self.guard_manager = type(
                "GuardManager", (), {"id_matched_objs": matched})()

    kept = [object(), object(), object()]
    entries = [Entry({"self": (lambda o = o: o)}) for o in kept]
    entries.append(Entry({}))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(U, "_dynamo_cache_entries", lambda code: entries)
        total, per_frame = U._recompile_cache_occupancy(object)
    assert total == 4
    # One instance's entry, plus the one that matches everybody.
    assert per_frame == 2


def test_a_collected_id_match_counts_for_every_frame():
    """torch skips a dead weakref when it compares, so such an entry matches
    whatever frame comes next and has to be counted for all of them."""

    class Entry:
        def __init__(self, matched):
            self.guard_manager = type(
                "GuardManager", (), {"id_matched_objs": matched})()

    entries = [Entry({"self": lambda: None}), Entry({"self": lambda: None})]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(U, "_dynamo_cache_entries", lambda code: entries)
        total, per_frame = U._recompile_cache_occupancy(object)
    assert (total, per_frame) == (2, 2)


def test_an_outstanding_bump_is_not_counted_as_budget(monkeypatch):
    """`_bump_recompile_limits` raises the process limits to get one call out of
    trouble and hands them back later. Reading the raised value would tell a
    saturated call site it has room it is about to lose again."""
    import torch._dynamo.config as config

    monkeypatch.setattr(U, "_ORIGINAL_RECOMPILE_LIMITS", {"recompile_limit": 8})
    with config.patch(recompile_limit = 24):
        assert U._recompile_budgets()[0] == 8


# ---- the failure this exists to stop ------------------------------------

def test_the_exhausting_call_never_happens_mid_step():
    """End to end, through real Dynamo, in the shape the T4 run failed in.

    Step one fills the cache. Without the boundary decision, step two's new
    shape raises inside the step -- and on the T4 that raise landed in a
    checkpoint recompute, where the step was already lost. Here the call site is
    eager before step two starts, so the compiler is never consulted again and
    nothing changes mode with a packed forward outstanding.
    """

    def kernel(x, weight):
        return x * weight

    torch._dynamo.reset()
    with torch._dynamo.config.patch(recompile_limit = 4,
                                    accumulated_recompile_limit = 256):
        compiled = torch.compile(kernel, fullgraph = True, dynamic = False,
                                 backend = "eager")
        wrapper = U._fall_back_to_eager_on_recompile_limit(
            compiled, kernel, "K.kernel")

        assert U.apply_pending_eager_fallbacks() == 0          # boundary 1
        for width in (3, 5, 7):                                # step 1
            wrapper(torch.randn(2, width), torch.randn(width))
        assert not wrapper._unsloth_fallback_state["eager"]

        assert U.apply_pending_eager_fallbacks() == 1          # boundary 2
        assert wrapper._unsloth_fallback_state["eager"]

        for width in (11, 13):                                 # step 2
            wrapper(torch.randn(2, width), torch.randn(width))
        # Nothing deferred, so nothing was still compiled when the new shapes
        # arrived. A deferral here is the bug: it means the switch happened
        # DURING the step rather than before it.
        assert not wrapper._unsloth_fallback_state["pending_eager"]
        assert U._PENDING_EAGER_LABELS == set()
