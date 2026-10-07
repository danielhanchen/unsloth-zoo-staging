# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""
The CPU gates overlap through native background steps, and the overlap must not
weaken a gate.

`Repo tests (CPU)` starts the behavioral gate with `background: true` and runs the
other pytest invocations beside it. Three things keep that equivalent to running
them one after another: the background step is joined by a `wait` (GitHub rejects
the whole file if that wait carries `if:`), every gate is still its own pytest
process with the same readiness condition so one failure does not skip the rest,
and only the advisory group is allowed to fail without failing the job.
"""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/consolidated-tests-ci.yml"
READY = "${{ !cancelled() && steps.deps.outcome == 'success' }}"
ADVISORY = "pytest tests/test_mlx_module_exports + zoo-specific CPU tests"


def _steps():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["repo-tests-cpu"]["steps"]


def _gates(steps):
    return [(i, s) for i, s in enumerate(steps) if "-m pytest" in str(s.get("run", ""))]


def test_only_the_behavioral_gate_runs_in_the_background_and_is_joined_last():
    steps = _steps()
    assert [s.get("id") for s in steps if s.get("background")] == ["behavioral-gates"]
    waits = [(i, s) for i, s in enumerate(steps) if "wait" in s or "wait-all" in s]
    assert [s.get("wait") for _, s in waits] == ["behavioral-gates"]
    assert waits[0][0] > max(i for i, _ in _gates(steps))
    assert all("if" not in s for _, s in waits)


def test_every_gate_is_one_pytest_process_gated_only_on_the_install():
    steps = _steps()
    assert any(s.get("id") == "deps" for s in steps)
    gates = _gates(steps)
    assert len(gates) == 8
    for _, step in gates:
        assert step["run"].count("-m pytest") == 1, step["name"]
        assert step.get("if") == READY, step["name"]


def test_only_the_advisory_group_may_fail_without_failing_the_job():
    lenient = [s["name"] for s in _steps() if s.get("continue-on-error")]
    assert lenient == [ADVISORY]
