# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""
The CPU gates overlap through native background steps, and the overlap must not
weaken a gate.

`Repo tests (CPU)` starts the behavioral gate with `background: true` and runs the
other pytest invocations beside it. That stays equivalent to running them one after
another only while the background step is joined by a `wait` after the last gate,
every gate is its own pytest process gated on the install alone (so one failure does
not skip the rest), and the PyPI-reaching advisory group is the only gate allowed
to fail without failing the job.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/consolidated-tests-ci.yml"
READY = "${{ !cancelled() && steps.deps.outcome == 'success' }}"
PYTEST = re.compile(r"^\s*(?:python3? -m )?pytest\s", re.M)


def _steps():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]["repo-tests-cpu"]["steps"]


def _gates(steps):
    return [(i, s) for i, s in enumerate(steps) if PYTEST.search(str(s.get("run", "")))]


def test_only_the_behavioral_gate_runs_in_the_background_and_is_joined_last():
    steps = _steps()
    assert [s.get("id") for s in steps if s.get("background")] == ["behavioral-gates"]
    waits = [(i, s) for i, s in enumerate(steps) if "wait" in s or "wait-all" in s]
    assert [s.get("wait") for _, s in waits] == ["behavioral-gates"]
    assert waits[0][0] > max(i for i, _ in _gates(steps))


def test_every_gate_is_one_pytest_process_gated_only_on_the_install():
    steps = _steps()
    deps = next(i for i, s in enumerate(steps) if s.get("id") == "deps")
    gates = _gates(steps)
    assert deps < min(i for i, _ in gates)
    for _, step in gates:
        assert len(PYTEST.findall(step["run"])) == 1, step["name"]
        assert " ".join(str(step.get("if")).split()) == READY, step["name"]


def test_only_the_advisory_group_may_fail_without_failing_the_job():
    lenient = [s for s in _steps() if s.get("continue-on-error")]
    assert len(lenient) == 1
    assert "tests/test_pypi_version_sync.py" in lenient[0]["run"]
