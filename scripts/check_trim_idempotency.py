#!/usr/bin/env python3
"""Prove a comment-only change did not move what compiler.py's rewriters emit.

This repo rewrites Python source TEXT at runtime, so a comment is not guaranteed
to be inert. Several finders in compiler.py anchor on CONSECUTIVE lines and are
not comment-tolerant: a comment sitting between the anchored lines suppresses
that rewrite today, and deleting it turns the rewrite on. An AST diff cannot see
this, because the source really did only change in its comments.

Two properties are checked.

I1  Rewriter decisions are unchanged from the base ref. For every rewritable
    module, run each rewriter over the base text and the head text and compare
    what it DID: whether it fired, how many lines it inserted, and a hash of
    those lines. Comment text is excluded from the hash, so this reports a
    decision change rather than the trim itself.

I3  Each rewriter is a fixpoint: f(f(src)) == f(src). This is the property the
    idempotency sentinels exist to guarantee (_AITER_MARKER is emitted into
    every rewritten block and tested before rewriting, so a second pass is a
    no-op). If a sentinel's text drifts, already-rewritten source stops matching
    and the block is wrapped twice; that shows up here as a non-fixpoint.

Usage:
    python scripts/check_trim_idempotency.py --base origin/main

Exit 0 = both properties hold, 1 = a rewriter changed its mind or is not
idempotent, 2 = the harness could not load the rewriters (never a silent pass).
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REWRITER_NAMES = (
    "higher_precision_softmax",
    "higher_precision_layernorms",
    "fix_attention_dtype_consistency",
    "patch_residual_stream",
    # Carries _AITER_MARKER, the clearest idempotency sentinel in the file: the
    # marker is emitted into every rewritten block and tested before rewriting, so
    # the two SDPA fallbacks it emits are themselves valid matches and a second
    # pass would triple the block if the emitted and tested text ever diverged.
    "replace_sdpa_with_amd_aiter",
)

TARGET_DIRS = ("unsloth_zoo/temporary_patches", "unsloth_zoo")
SKIP_PARTS = ("_vendored", "diffusion_studio", "stubs", "__pycache__")


def load_rewriters(compiler_src: str) -> dict:
    """Recover the pure source->source rewriters without importing the package.

    `import unsloth_zoo` pulls in torch and dies on any version skew, which would
    make this check unrunnable on exactly the runners that need it. The rewriters
    need only `re`, so exec the module body with the import statements dropped and
    keep whatever survives. Anything missing is reported and exits non-zero: a
    check that quietly verifies nothing is worse than no check.
    """
    import re as _re

    tree = ast.parse(compiler_src)
    kept = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    # The rewriters need `re` and `ast` (the aiter one selects on the parsed AST so
    # that comments and string literals can never be rewritten). Supply both.
    ns: dict = {"re": _re, "ast": ast, "os": os, "sys": sys, "__name__": "compiler_shim"}
    for node in kept:
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<compiler>", "exec"), ns)
        except Exception:
            # Statements that need torch simply do not land. Fine, as long as the
            # rewriters themselves did, which is asserted by the caller.
            continue

    out, missing = {}, []
    for name in REWRITER_NAMES:
        fn = ns.get(name)
        (out.__setitem__(name, fn) if callable(fn) else missing.append(name))
    if missing:
        print(f"FATAL: could not recover rewriters: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(2)
    return out


def added_lines(before: str, after: str) -> list[str]:
    """Lines the rewriter introduced, stripped of indentation.

    Indentation is dropped on purpose: a rewriter copies the indent of the line it
    matched, so re-indented but otherwise identical output is not a decision change.
    """
    return [
        ln[2:].strip()
        for ln in difflib.ndiff(before.splitlines(), after.splitlines())
        if ln.startswith("+ ")
    ]


def decision(fn, src: str) -> tuple[str, int, bool]:
    try:
        result = fn(src)
    except Exception as exc:
        # A refusal is itself a stable observation. Record its type so a trim that
        # turns refusal into acceptance surfaces rather than passing silently.
        return (f"EXC:{type(exc).__name__}", 0, False)
    if not isinstance(result, str):
        result = repr(result)
    ins = added_lines(src, result)
    digest = hashlib.sha256("\n".join(ins).encode("utf-8")).hexdigest()[:16]
    return (digest, len(ins), result != src)


# A source the aiter rewriter will actually rewrite: a plain causal SDPA call on
# its own line with q/k/v as bare names, which is what guard 2 matches on the AST.
AITER_FIXTURE = '''
import torch

def forward(self, q, k, v):
    attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return attn
'''


def _stub_device_type():
    """Make the aiter rewriter believe it is on ROCm with amd-aiter present.

    Without this the function returns at its first line on any non-AMD runner, so
    a fixpoint check over it would pass while executing nothing. The sentinel this
    exercises (_AITER_MARKER) is the whole reason the rewriter is idempotent, so a
    vacuous check here is worse than none.
    """
    import types

    pkg = sys.modules.get("unsloth_zoo")
    if pkg is None:
        pkg = types.ModuleType("unsloth_zoo")
        pkg.__path__ = []
        sys.modules["unsloth_zoo"] = pkg
    dt = types.ModuleType("unsloth_zoo.device_type")
    dt.get_amd_attention_implementation = lambda: "amd_aiter"
    dt.get_amd_flash_attn_func = lambda: None
    sys.modules["unsloth_zoo.device_type"] = dt
    pkg.device_type = dt


def check_aiter_fixpoint(fns: dict) -> int:
    """Exercise the aiter rewriter on a fixture and assert it is a fixed point."""
    fn = fns.get("replace_sdpa_with_amd_aiter")
    if fn is None:
        print("[I3 FAIL] replace_sdpa_with_amd_aiter not recovered")
        return 1
    _stub_device_type()
    once = fn(AITER_FIXTURE)
    if once == AITER_FIXTURE:
        # Anti-vacuity: if it did not fire, the fixpoint result below is meaningless.
        print("[I3 FAIL] aiter fixture did not trigger a rewrite; this check is vacuous")
        return 1
    twice = fn(once)
    if once != twice:
        extra = len(added_lines(once, twice))
        print(f"[I3 FAIL] replace_sdpa_with_amd_aiter re-wrapped its own output (+{extra} lines)")
        print("          the emitted marker no longer matches the guard that tests for it")
        return 1
    print(f"[ ok ] aiter fixture rewritten (+{len(added_lines(AITER_FIXTURE, once))} lines) and stable on a 2nd pass")
    return 0


def git_show(rev: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def iter_targets(root: Path):
    seen = set()
    for d in TARGET_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if any(x in p.parts for x in SKIP_PARTS):
                continue
            rel = str(p.relative_to(root))
            if rel not in seen:
                seen.add(rel)
                yield rel, p.read_text(encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    head_compiler = (root / "unsloth_zoo" / "compiler.py").read_text(encoding="utf-8")
    base_compiler = git_show(args.base, "unsloth_zoo/compiler.py")
    if base_compiler is None:
        print(f"FATAL: cannot read compiler.py at {args.base}", file=sys.stderr)
        return 2

    head_fns = load_rewriters(head_compiler)
    base_fns = load_rewriters(base_compiler)

    i1 = i3 = 0
    checked = fired = 0

    # Exercised on a fixture because this rewriter is gated on AMD hardware and so
    # never fires on the repo's own files.
    i3 += check_aiter_fixpoint(head_fns)

    for rel, head_src in iter_targets(root):
        base_src = git_show(args.base, rel)
        for name in REWRITER_NAMES:
            checked += 1

            # I3: fixpoint on the head tree.
            try:
                once = head_fns[name](head_src)
                twice = head_fns[name](once)
                if once != twice:
                    extra = len(added_lines(once, twice))
                    print(f"[I3 FAIL] {rel}: {name} is not a fixpoint (+{extra} lines on 2nd pass)")
                    i3 += 1
            except Exception:
                pass  # refusal is covered by I1's EXC record

            # I1: same decision as base.
            if base_src is None:
                continue
            b = decision(base_fns[name], base_src)
            h = decision(head_fns[name], head_src)
            if h[2]:
                fired += 1
            if b != h:
                print(f"[I1 FAIL] {rel}: {name} changed decision")
                print(f"          base fired={b[2]} inserts={b[1]} hash={b[0]}")
                print(f"          head fired={h[2]} inserts={h[1]} hash={h[0]}")
                i1 += 1

    print(
        f"\nrewriter checks: {checked}   firings on head: {fired}\n"
        f"I1 decision changes vs {args.base}: {i1}\n"
        f"I3 non-fixpoint rewriters: {i3}"
    )
    return 1 if (i1 or i3) else 0


if __name__ == "__main__":
    sys.exit(main())
