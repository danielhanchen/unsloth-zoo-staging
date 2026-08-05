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


"""Does Gemma 3n audio training work on a checkpoint people actually download?

`_AUDIO_QUALIFIED_FAMILIES` admits gemma3n over mlx-vlm 0.4.4 to 0.6.4, widened
there on the strength of its processor sources being byte-identical across that
span. That is an argument about the library, and this family has now shown twice
that the library is not the only variable: the collation bug PR 985 fixed made
gemma3n audio unusable at any realistic `max_seq_length`, and it was found only
because something tried to run it rather than read it.

So this runs it. Stages, in order, each reporting on its own:

  1. zoo loads it
  2. the model has an audio tower
  3. a clip collates with features behind its placeholders -- for gemma3n this
     IS the alignment check, because collation asserts one placeholder run per
     clip and a feature entry behind every surviving run
  4. audio reaches the loss -- distinct audio must give distinct losses, or the
     model is training on text alone

Run: python tests/gemma3n_audio_probe.py [--model REPO]
Exit code 0 only if every stage passed.
"""

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

DEFAULT_MODEL = "mlx-community/gemma-3n-E2B-it-4bit"
RATE = 16000

results = {}


def stage(name):
    def wrap(fn):
        def run(*a, **k):
            try:
                detail = fn(*a, **k)
                results[name] = {"ok": True, "detail": detail}
                print(f"[PASS] {name}: {detail}", flush=True)
                return True
            except Exception as exc:
                results[name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:],
                }
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
                print(traceback.format_exc()[-1500:], flush=True)
                return False
        return run
    return wrap


def tone(seconds, hertz):
    import numpy as np
    t = np.arange(int(RATE * seconds), dtype=np.float32) / RATE
    return (0.5 * np.sin(2.0 * np.pi * hertz * t)).astype(np.float32)


def audio_row(hertz, seconds=1.0):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "audio",
             "audio": {"array": tone(seconds, hertz), "sampling_rate": RATE}},
            {"type": "text", "text": "Transcribe."}]},
        {"role": "assistant", "content": "ok"},
    ]}


@stage("1_zoo_loads_it")
def load_via_zoo(repo):
    from unsloth_zoo.mlx.loader import FastMLXModel
    global _MODEL, _PROCESSOR
    _MODEL, _PROCESSOR = FastMLXModel.from_pretrained(
        model_name=repo, max_seq_length=512,
    )
    return f"{type(_MODEL).__name__} + {type(_PROCESSOR).__name__}"


@stage("2_model_has_audio_tower")
def check_tower(_repo):
    tower = getattr(_MODEL, "audio_tower", None)
    if tower is None:
        raise AssertionError("no audio_tower on this checkpoint")
    return type(tower).__name__


@stage("3_features_land_behind_the_placeholders")
def check_collation(_repo):
    """The bug PR 985 fixed, asserted rather than assumed.

    A ragged batch on purpose: the failure that mattered showed up only when
    rows tokenized to different lengths, and a one-row probe missed it.
    """
    import numpy as np

    from unsloth_zoo.mlx.utils import _collate_vlm_batch, _finalize_vlm_batch

    rows = [audio_row(440.0), audio_row(880.0)]
    rows[0]["messages"][0]["content"][1]["text"] = "Transcribe. " * 400
    batch = _finalize_vlm_batch(
        _collate_vlm_batch(rows, _PROCESSOR, 512, None))
    ids = np.asarray(batch["input_ids"])
    feats = np.asarray(batch["input_features"])
    if feats.shape[0] != len(rows) or feats.shape[1] == 0:
        raise AssertionError(
            f"features {feats.shape} cannot sit behind {len(rows)} clips")
    if ids.shape[1] > 512:
        raise AssertionError(f"row of {ids.shape[1]} tokens exceeds the cap")
    return f"input_ids={tuple(ids.shape)} input_features={tuple(feats.shape)}"


@stage("4_audio_reaches_the_loss")
def check_loss(_repo):
    import numpy as np

    from unsloth_zoo.mlx.utils import (
        _collate_vlm_batch, _finalize_vlm_batch, make_vlm_baseline_loss_fn,
    )

    losses = []
    for hertz in (440.0, 1760.0):
        batch = _finalize_vlm_batch(
            _collate_vlm_batch([audio_row(hertz)], _PROCESSOR, 512, None))
        loss_fn = make_vlm_baseline_loss_fn(_MODEL, ignore_token_ids=[])
        out = loss_fn(_MODEL, batch)
        loss = float(out[0] if isinstance(out, tuple) else out)
        if not np.isfinite(loss):
            raise AssertionError(f"loss is not finite at {hertz} Hz: {loss}")
        losses.append(loss)
    if abs(losses[0] - losses[1]) < 1e-6:
        raise AssertionError(
            f"two different tones gave the same loss ({losses[0]}), so the "
            f"audio is not reaching the objective")
    return f"440Hz={losses[0]:.4f} 1760Hz={losses[1]:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    import mlx_vlm
    import transformers
    version = getattr(mlx_vlm, "__version__", "unknown")
    print(f"=== gemma3n audio probe: mlx-vlm {version}, "
          f"transformers {transformers.__version__}, {args.model} ===",
          flush=True)

    ok = load_via_zoo(args.model)
    for name, fn in (("2_model_has_audio_tower", check_tower),
                     ("3_features_land_behind_the_placeholders",
                      check_collation),
                     ("4_audio_reaches_the_loss", check_loss)):
        if not ok:
            results[name] = {"ok": False, "skipped": True,
                             "error": "prerequisite stage failed"}
            print(f"[SKIP] {name}: prerequisite stage failed", flush=True)
            continue
        fn(args.model)

    print("PROBE_RESULT " + json.dumps(
        {"mlx_vlm": version, "transformers": transformers.__version__,
         "model": args.model, "stages": results}), flush=True)
    sys.exit(0 if all(r["ok"] for r in results.values()) else 1)


if __name__ == "__main__":
    main()
