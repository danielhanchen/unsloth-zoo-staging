# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Which layer does the Metal drift start at?

Two identical uncached forwards of Gemma 4 E2B on Apple Silicon give different
logits; on Linux CPU they are bit-exact. `mlx_only_repeat_forward.py` shows
that at the output. It cannot say where it begins, and "where" is the whole
question -- the standing guess was that KV-shared layers read a slot an
uncached forward never populates, so the drift should appear no earlier than
the first KV-shared layer.

mlx-vlm already carries the instrument: `capture_layer_ids` fills a
`hidden_sink` with each decoder layer's output. So run the same input twice,
capture every layer, and report the first index whose two captures are not
bit-identical.

Reading the answer, for Gemma 4 E2B (35 layers, last 20 KV-shared, so the
boundary is 15):

  * first differing layer >= 15  -- consistent with the KV-sharing story
  * first differing layer < 15   -- the KV-shared layers are downstream of a
                                   drift that started before them, and the
                                   story is wrong

No unsloth in the picture on purpose: this is mlx-vlm and mlx only, so a
finding here is reportable upstream without a zoo-shaped caveat.

Run: python tests/mlx_layerwise_drift.py [--model REPO] [--rounds N]
"""

import argparse
import sys

DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"

# "The capital of France is" in Gemma tokens, short enough to stay cheap.
PROMPT_IDS = [[2, 106, 1645, 108, 3689, 236743, 236764, 1902, 236764]]


def capture(model, ids, n_layers):
    """One uncached forward, returning every decoder layer's output."""
    import mlx.core as mx

    out = model.language_model(ids, capture_layer_ids=list(range(n_layers)))
    hidden = out.hidden_states
    if not hidden:
        raise SystemExit(
            "this model's language_model did not fill hidden_sink, so it does "
            "not support capture_layer_ids; nothing to bisect")
    mx.eval(*hidden, out.logits)
    to_np = lambda a: __import__("numpy").asarray(a.astype(mx.float32))
    return [to_np(h) for h in hidden], to_np(out.logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_vlm
    import numpy as np
    from mlx_vlm import load

    print(f"=== layerwise drift: mlx {mx.__version__}, "
          f"mlx-vlm {getattr(mlx_vlm, '__version__', '?')}, {args.model} ===",
          flush=True)

    model, _ = load(args.model)
    lm = model.language_model
    inner = getattr(lm, "model", lm)
    n_layers = len(inner.layers)

    first_shared = getattr(inner, "first_kv_shared_layer_idx", None)
    print(f"layers={n_layers} first_kv_shared_layer_idx={first_shared}",
          flush=True)

    ids = mx.array(PROMPT_IDS)
    runs = []
    for r in range(args.rounds):
        hidden, logits = capture(model, ids, n_layers)
        runs.append(hidden)
        total = float(logits.sum())
        print(f"  round {r}: logit sum={total} finite={np.isfinite(logits).all()}",
              flush=True)

    first_diff = None
    print("layer | max |a-b| vs round 0 | first-nonfinite", flush=True)
    for i in range(n_layers):
        worst = 0.0
        nonfinite = False
        for r in range(1, args.rounds):
            a, b = runs[0][i], runs[r][i]
            nonfinite = nonfinite or not np.isfinite(a).all() \
                or not np.isfinite(b).all()
            d = np.abs(a - b)
            worst = max(worst, float(d.max()) if d.size else 0.0)
        if worst > 0.0 and first_diff is None:
            first_diff = i
        marker = "  <-- first difference" if first_diff == i else ""
        if worst > 0.0 or nonfinite or i < 3 or i == n_layers - 1:
            print(f"{i:5d} | {worst:.6g}{' NONFINITE' if nonfinite else ''}"
                  f"{marker}", flush=True)

    if first_diff is None:
        print("VERDICT bit-identical at every layer; no drift to localize",
              flush=True)
        return 0

    print(f"VERDICT first differing layer: {first_diff}", flush=True)
    if first_shared is None:
        print("this model reports no KV-sharing boundary, so the result "
              "neither supports nor refutes the KV-sharing story", flush=True)
    elif first_diff >= first_shared:
        print(f"consistent with KV sharing: drift starts at or after the "
              f"boundary {first_shared}", flush=True)
    else:
        print(f"NOT the KV-shared layers: drift starts at {first_diff}, "
              f"before the boundary {first_shared}, so the shared layers "
              f"inherit it rather than cause it", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
