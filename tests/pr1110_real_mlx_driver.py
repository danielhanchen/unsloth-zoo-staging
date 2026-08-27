"""Drive unsloth_zoo.mlx.preference against REAL mlx (no torch simulation shim).

Works against both the PR head API (length_policy=) and the pre-PR base API
(max_seq_length=), so the same numbers can be diffed across the two trees.

Run with the tree to test first on sys.path (PYTHONPATH) or installed.
"""
import inspect
import json
import os
import sys
import warnings

os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")

import numpy as np

# Fail loudly if anything shimmed mlx: this driver exists to reach the real one.
import mlx.core as mx
import mlx.nn as nn

origin = str(getattr(mx, "__file__", "") or "")
assert "mlx_simulation" not in origin, f"shim, not real mlx: {origin}"
assert int(mx.array([1, 2, 3]).sum()) == 6, "mlx cannot add"

import unsloth_zoo
from unsloth_zoo.mlx import preference as P

OUT = {
    "tree": os.path.dirname(os.path.dirname(os.path.dirname(P.__file__))),
    "preference_file": P.__file__,
    "zoo_version": unsloth_zoo.__version__,
    "mlx_file": origin,
    "python": sys.version.split()[0],
}
try:
    from importlib.metadata import version as _v
    OUT["mlx_version"] = _v("mlx")
    try:
        OUT["mlx_cpu_version"] = _v("mlx-cpu")
    except Exception:
        OUT["mlx_cpu_version"] = None
except Exception:
    pass
OUT["default_device"] = str(mx.default_device())
OUT["has_length_policy_api"] = "PreferenceLengthPolicy" in dir(P)


# ---------------------------------------------------------------- fixtures
class Tokenizer:
    """Same deterministic tokenizer the repo's own test file uses."""
    bos_token = None
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [3 + (ord(character) % 43) for character in text]

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False,
        continue_final_message=False, **kwargs,
    ):
        rendered = "".join(
            f"<{m['role']}>{m['content']}" for m in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def rows(count=6):
    return [
        {
            "prompt": f"question {i}: ",
            "chosen": "yes it is" if i % 2 else "yes",
            "rejected": "no" if i % 3 else "definitely not at all",
        }
        for i in range(count)
    ]


VOCAB = 64
DIM = 8


class TinyModel(nn.Module):
    """Embedding -> Linear logits. Deterministic weights, no randomness."""

    def __init__(self):
        super().__init__()
        rng = np.random.RandomState(0)
        self.embed = nn.Embedding(VOCAB, DIM)
        self.out = nn.Linear(DIM, VOCAB, bias=False)
        self.embed.weight = mx.array(
            rng.standard_normal((VOCAB, DIM)).astype(np.float32) * 0.1
        )
        self.out.weight = mx.array(
            rng.standard_normal((VOCAB, DIM)).astype(np.float32) * 0.1
        )

    def __call__(self, x):
        return self.out(self.embed(x))


def make_policy(kind, max_seq_length=64, **kw):
    """Head: a PreferenceLengthPolicy. Base: the bare int it used instead."""
    if not OUT["has_length_policy_api"]:
        return max_seq_length
    options = dict(
        kind=kind, max_length=max_seq_length, max_prompt_length=None,
        max_completion_length=None, truncation_mode="keep_end",
        max_seq_length=max_seq_length,
    )
    options.update(kw)
    return P.PreferenceLengthPolicy(**options)


def budget_kwargs(policy):
    """Bridge the max_seq_length= -> length_policy= rename."""
    if OUT["has_length_policy_api"]:
        return {"length_policy": policy}
    return {"max_seq_length": policy}


def f(value):
    return round(float(value), 8)


# ---------------------------------------------------------------- 1. tokenize
def step_tokenize():
    results = {}
    for kind in ("dpo", "orpo"):
        policy = make_policy(kind)
        per_row = []
        for row in rows():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                tok = P.tokenize_preference_row(
                    Tokenizer(), row, **budget_kwargs(policy)
                )
            per_row.append({
                "chosen_prompt_ids": list(tok.chosen_prompt_ids),
                "chosen_ids": list(tok.chosen_ids),
                "rejected_prompt_ids": list(tok.rejected_prompt_ids),
                "rejected_ids": list(tok.rejected_ids),
                "warnings": [str(w.message) for w in caught],
            })
        results[kind] = per_row

    # An implicit-prompt row: only the PR head is meant to recover the prompt.
    implicit = {
        "chosen": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "yes"},
        ],
        "rejected": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "no"},
        ],
    }
    try:
        tok = P.tokenize_preference_row(
            Tokenizer(), implicit, **budget_kwargs(make_policy("dpo"))
        )
        results["implicit_prompt"] = {
            "ok": True,
            "chosen_prompt_ids": list(tok.chosen_prompt_ids),
            "chosen_ids": list(tok.chosen_ids),
            "rejected_prompt_ids": list(tok.rejected_prompt_ids),
            "rejected_ids": list(tok.rejected_ids),
        }
    except Exception as error:
        results["implicit_prompt"] = {
            "ok": False, "error": f"{type(error).__name__}: {error}",
        }
    return results


# ---------------------------------------------------------------- 2. plan
def build_plan(kind="dpo", **kw):
    options = dict(batch_size=2, dataset_order="sequential", grad_accum=2,
                   num_epochs=1)
    options.update(kw)
    options.update(budget_kwargs(make_policy(kind)))
    return P.create_preference_batch_plan(rows(), Tokenizer(), **options)


def step_plan():
    out = {}
    for kind in ("dpo", "orpo"):
        plan = build_plan(kind)
        out[kind] = {
            "len": len(plan),
            "schedule": [list(b) for b in plan.schedule],
            "widths": list(plan.widths),
            "cycle_length": plan.cycle_length,
            "max_seq_length": plan.max_seq_length,
            "pad_id": plan.pad_id,
            "normalizers": [list(n) for n in plan._normalizers],
            "families": [plan.batch_family(i) for i in range(len(plan))],
        }
    return out


# ---------------------------------------------------------------- 3. materialize
def step_materialize():
    out = {}
    for kind in ("dpo", "orpo"):
        plan = build_plan(kind)
        batches = []
        for i in range(len(plan)):
            batch, lengths, norms = plan.materialize(i)
            mx.eval(batch, lengths, norms)
            for name, arr in (("batch", batch), ("lengths", lengths),
                              ("normalizers", norms)):
                assert isinstance(arr, mx.array), (name, type(arr))
            batches.append({
                "dtypes": [str(batch.dtype), str(lengths.dtype),
                           str(norms.dtype)],
                "shapes": [list(batch.shape), list(lengths.shape),
                           list(norms.shape)],
                "batch": np.array(batch).tolist(),
                "lengths": np.array(lengths).tolist(),
                "normalizers": np.array(norms).tolist(),
            })
        out[kind] = batches
    return out


# ---------------------------------------------------------------- 4. losses
class HalfReference:
    """Stand-in reference policy: a real second forward, gradient-stopped."""

    def forward(self, model, batch, lengths):
        return mx.stop_gradient(P._response_logps(model, batch, lengths) * 0.5)


def step_losses():
    out = {}
    model = TinyModel()
    mx.eval(model.parameters())

    cases = [
        ("orpo_beta0.1", "orpo", P.make_orpo_loss_fn(beta=0.1)),
        ("orpo_beta0.5", "orpo", P.make_orpo_loss_fn(beta=0.5)),
        ("dpo_reference_free", "dpo",
         P.make_dpo_loss_fn(beta=0.1, reference_free=True)),
        ("dpo_smoothed", "dpo",
         P.make_dpo_loss_fn(beta=0.1, label_smoothing=0.1,
                            reference_free=True)),
        ("dpo_with_reference", "dpo",
         P.make_dpo_loss_fn(beta=0.1, reference_policy=HalfReference(),
                            reference_free=False)),
    ]
    for name, kind, loss_fn in cases:
        plan = build_plan(kind)
        per_batch = []
        for i in range(len(plan)):
            batch, lengths, norms = plan.materialize(i)
            loss, count = loss_fn(model, batch, lengths, norms)
            mx.eval(loss, count)
            supervised = loss_fn._unsloth_supervised_tokens(
                (batch, lengths, norms)
            )
            mx.eval(supervised)

            # Real MLX autodiff through the whole objective.
            def scalar(params, batch=batch, lengths=lengths, norms=norms,
                       loss_fn=loss_fn):
                model.update(params)
                return loss_fn(model, batch, lengths, norms)[0]

            grads = mx.grad(scalar)(model.parameters())
            mx.eval(grads)
            flat = [np.array(v) for _, v in
                    __import__("mlx.utils", fromlist=["tree_flatten"])
                    .tree_flatten(grads)]
            gnorm = float(np.sqrt(sum(float((a ** 2).sum()) for a in flat)))
            per_batch.append({
                "loss": f(loss.item()),
                "count": int(count.item()),
                "supervised_tokens": int(supervised.item()),
                "grad_norm": round(gnorm, 6),
                "grad_finite": bool(all(np.isfinite(a).all() for a in flat)),
            })
        out[name] = per_batch
    return out


# ---------------------------------------------------------------- 5. context
def step_run_context():
    model = TinyModel()
    ctx = P.PreferenceRunContext(model, enabled=True)
    ctx.restore()
    return {"ok": True}


def main():
    steps = [
        ("tokenize", step_tokenize),
        ("plan", step_plan),
        ("materialize", step_materialize),
        ("losses", step_losses),
        ("run_context", step_run_context),
    ]
    for name, fn in steps:
        try:
            OUT[name] = fn()
        except Exception as error:
            import traceback
            OUT[name] = {
                "ERROR": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc().splitlines()[-6:],
            }
    print(json.dumps(OUT, indent=1, sort_keys=True, default=str))
    # No step here has an expected failure, so every ERROR and every ok=False is
    # a real one. Walk the whole tree rather than the top level: the per-row
    # results nest, and a step that caught its own error would otherwise report
    # a green job while the section it was measuring never ran.
    failures = []

    def walk(node, path):
        if isinstance(node, dict):
            if "ERROR" in node:
                failures.append(f"{path or '<root>'}: {node['ERROR']}")
            if node.get("ok") is False:
                failures.append(f"{path or '<root>'}: {node.get('error', 'ok=False')}")
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(OUT, "")
    if failures:
        print(f"\nFAILED: {len(failures)} real-MLX failure(s)")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nOK: real MLX driver clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
