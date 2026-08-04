"""Does a second Gemma 4 audio forward see state from the first?

On Metal the probe's two losses were not reproducible: mlx-vlm 0.6.3 gave the
same loss for the first clip across two runs (30.5491) and a different one for
the second (26.5851, then 27.8627). A first call that repeats and a second call
that does not is the signature of state carried between forwards, and Gemma 4
E2B reuses K/V across its last 20 layers, which is exactly the kind of state
that could survive.

This asks the question directly: run the identical clip four times and once
with a different clip in the middle. Every 440 Hz loss must be the same number.
If they drift, something from call n is reaching call n+1, and in training that
would be one batch contaminating the next.

Run: python tests/g4_repeat_forward_check.py --model PATH_OR_REPO
"""

import argparse
import os

os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

RATE = 16000


def tone(seconds, hertz):
    import numpy as np
    t = np.arange(int(RATE * seconds), dtype=np.float32) / RATE
    return (0.5 * np.sin(2.0 * np.pi * hertz * t)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    import mlx_vlm
    print(f"=== repeat-forward check: mlx-vlm "
          f"{getattr(mlx_vlm, '__version__', '?')} ===", flush=True)

    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx import utils as U

    version = getattr(mlx_vlm, "__version__", "0")
    U._AUDIO_QUALIFIED_FAMILIES = dict(
        U._AUDIO_QUALIFIED_FAMILIES,
        gemma4=U._AudioVersions(version, version),
    )
    U._AUDIO_MIN_TRANSFORMERS = {}

    model, processor = FastMLXModel.from_pretrained(
        model_name=args.model, max_seq_length=512,
    )

    audio_token_id = model.config.audio_token_id
    held = U.install_audio_merge_patch(model, audio_token_id)
    try:
        def loss_for(hz):
            clip = {"array": tone(1.0, hz), "sampling_rate": RATE}
            messages = [
                {"role": "user", "content": [
                    {"type": "audio", "audio": clip},
                    {"type": "text", "text": "Transcribe."}]},
                {"role": "assistant", "content": "ok"},
            ]
            staged = U._collate_vlm_batch(
                [{"messages": messages}], processor, 512, None)
            batch = U._finalize_vlm_batch(staged)
            loss_fn = U.make_vlm_baseline_loss_fn(model, ignore_token_ids=[])
            out = loss_fn(model, batch)
            return float(out[0] if isinstance(out, tuple) else out)

        # 440 four times, with a different clip interleaved, so a leak from the
        # previous call shows up as the repeats disagreeing.
        seq = [440.0, 440.0, 1760.0, 440.0, 880.0, 440.0]
        seen = []
        for hz in seq:
            value = loss_for(hz)
            seen.append((hz, value))
            print(f"  {hz:>7.1f} Hz -> {value:.6f}", flush=True)

        repeats = [v for hz, v in seen if hz == 440.0]
        spread = max(repeats) - min(repeats)
        print(f"440 Hz repeats: {repeats}", flush=True)
        print(f"spread across identical inputs: {spread:.9f}", flush=True)
        print("VERDICT " + ("stateless" if spread == 0.0 else "STATE LEAKS"),
              flush=True)
    finally:
        if held:
            U.remove_audio_merge_patch(model)


if __name__ == "__main__":
    main()
