"""Is an identical MLX forward reproducible on Metal? No unsloth involved.

On macos-14, running the same batch through the same model twice gave different
losses, including a `nan` on one first call, while Linux CPU MLX was exact to
the last bit. Every run that showed this went through `FastMLXModel` and
`make_vlm_baseline_loss_fn`, so it cannot yet be attributed.

This removes unsloth entirely. `mlx_vlm.load`, one fixed token array, the same
forward five times, compare logits. Whatever it reports belongs to mlx-vlm or
to mlx, and nothing should be filed upstream before it is known which.

Deliberately no audio, no images, no training loop, no optimizer: the earlier
control showed text-only drifts further than audio, so the smallest thing that
still reproduces it is a bare forward on token ids.

Run: python tests/mlx_only_repeat_forward.py [--model REPO] [--rounds N]
Exit 0 when every round agrees bit for bit.
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-4-e2b-it-4bit")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_vlm
    from mlx_vlm.utils import get_model_path, load_model

    print(f"=== mlx-only repeat forward: mlx {mx.__version__}, "
          f"mlx-vlm {getattr(mlx_vlm, '__version__', '?')} ===", flush=True)

    path = get_model_path(args.model)
    if isinstance(path, tuple):
        path = path[0]
    model = load_model(path)

    # A fixed, boring prompt. The ids never change between rounds, so any
    # difference in the output is the runtime's, not the input's.
    ids = mx.array([[2, 106, 1645, 108, 3689, 236743, 236764, 1902, 236764]])

    checksums = []
    for round_index in range(args.rounds):
        out = model.language_model(ids)
        logits = out[0] if isinstance(out, tuple) else out
        logits = getattr(logits, "logits", logits)
        mx.eval(logits)
        total = float(logits.astype(mx.float32).sum())
        finite = bool(mx.isfinite(logits).all())
        checksums.append(total)
        print(f"  round {round_index}: sum={total!r} all_finite={finite}",
              flush=True)

    unique = sorted(set(checksums))
    print(f"distinct results across {args.rounds} identical forwards: "
          f"{len(unique)}", flush=True)
    if len(unique) > 1:
        print(f"spread: {max(unique) - min(unique)!r}", flush=True)
    print("VERDICT " + ("reproducible" if len(unique) == 1
                        else "NOT REPRODUCIBLE"), flush=True)
    sys.exit(0 if len(unique) == 1 else 1)


if __name__ == "__main__":
    main()
