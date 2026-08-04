"""A torchao built for a newer torch must not be an unreadable crash.

`pyproject.toml` declares `torchao>=0.13.0` with no upper bound, so a
resolver handed a pinned older torch will install a torchao built for a
newer one. That torchao's float8 path imports symbols straight out of
torch -- `from torch.nn.functional import ScalingType` -- and on the
older torch the symbol is simply absent.

The ImportError then surfaces while importing transformers, so it is
caught by the `Unpack` guard in temporary_patches/utils.py, does not
mention Unpack, and falls through to a bare `raise Exception(e)`. The
user sees:

    cannot import name 'ScalingType' from 'torch.nn.functional'

`import unsloth` is dead, and the message names neither torchao nor
torch nor anything to do. Qwen3_5_(4B)_Vision and Qwen3_8B_FP8_GRPO both
died in cell 2 this way; both pin torch==2.8.0 and then install
unsloth_zoo[base].

The guard is narrow on purpose. Widening it to "any missing name" would
swallow the Unpack move it sits next to, which is a different problem
with a different fix.
"""

import ast
import sys
from pathlib import Path

import pytest

UTILS = (Path(__file__).resolve().parents[1] / "unsloth_zoo"
         / "temporary_patches" / "utils.py")


def _load_helpers():
    """Exec just the two helpers.

    Importing the module would run the very import guard under test and,
    on a healthy install, do nothing interesting -- and on a broken one,
    raise before a single assertion ran.
    """
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    wanted = {"_torchao_is_newer_than_torch", "_torchao_torch_mismatch_message"}
    ns: dict = {}
    for node in tree.body:
        keep = False
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            keep = True
        elif isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "_TORCHAO_TORCH_SYMBOLS"
                for t in node.targets):
            keep = True
        if keep:
            exec(compile(ast.Module([node], []), "<utils>", "exec"), ns)
    missing = wanted - set(ns)
    assert not missing, f"helpers not found in utils.py: {missing}"
    return ns


HELPERS = _load_helpers()
looks_like = HELPERS["_torchao_is_newer_than_torch"]
message_of = HELPERS["_torchao_torch_mismatch_message"]

REAL = ("cannot import name 'ScalingType' from 'torch.nn.functional' "
        "(/usr/local/lib/python3.12/dist-packages/torch/nn/functional.py)")


# ---- what it must catch ---------------------------------------------------

def test_the_error_seen_in_the_wild():
    assert looks_like(REAL) is True


@pytest.mark.parametrize("sym", ["ScalingType", "ScalingGranularity",
                                 "Float8Tensor"])
def test_the_other_torchao_symbols(sym):
    assert looks_like(
        f"cannot import name '{sym}' from 'torch.nn.functional'") is True


# ---- what it must NOT catch ----------------------------------------------

def test_the_unpack_move_is_left_alone():
    """It sits directly beside the Unpack branch; swallowing that would
    hide a different problem with a different fix."""
    assert looks_like(
        "cannot import name 'Unpack' from 'transformers.processing_utils'"
    ) is False


def test_the_same_symbol_from_a_non_torch_module():
    assert looks_like("cannot import name 'ScalingType' from 'somelib.x'") is False


def test_a_non_import_error():
    assert looks_like("torchvision::nms does not exist") is False


def test_an_unrelated_missing_name_from_torch():
    assert looks_like(
        "cannot import name 'some_new_api' from 'torch.nn.functional'") is False


# ---- the guard must never be what raises ---------------------------------

@pytest.mark.parametrize("bad", [None, 123, object()])
def test_non_string_input_does_not_raise(bad):
    assert looks_like(bad) in (True, False)


def test_the_message_names_both_versions():
    m = message_of(REAL)
    assert "torchao" in m and "torch " in m
    assert REAL.split(" (")[0] in m, "the original error must survive"
    assert "torchao<0.18" in m, "the user needs something to run"


def test_the_message_survives_missing_metadata(monkeypatch):
    """Version lookup is best-effort; it must not turn a diagnostic into a
    second exception."""
    import importlib.metadata as md
    monkeypatch.setattr(md, "version",
                        lambda *_a, **_k: (_ for _ in ()).throw(Exception("no")))
    m = message_of(REAL)
    assert "unknown" in m


# ---- the call site --------------------------------------------------------

def test_the_branch_runs_before_the_generic_reraise():
    """Placed after `raise Exception(e)` it would never be reached."""
    src = UTILS.read_text(encoding="utf-8")
    guard = src.index("_torchao_is_newer_than_torch(e)")
    generic = src.index('elif "Unpack" not in e:')
    assert guard < generic, (
        "the torchao branch must precede the generic re-raise")


def test_the_branch_raises_runtimeerror_not_a_bare_exception():
    src = UTILS.read_text(encoding="utf-8")
    i = src.index("_torchao_is_newer_than_torch(e)")
    window = src[i:i + 200]
    assert "RuntimeError(_torchao_torch_mismatch_message(e))" in window


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
