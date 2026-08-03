"""Disposable staging probe for upstream PR #956 (Job C).

Reports, for whatever mlx-lm is installed, whether the batched-generation text
path's capability probe accepts it -- and if it does, whether a real two-prompt
generate_batch actually runs. The PR leaves the declared floor at
mlx-lm>=0.28.3; this says what that floor really buys.

Exit code is always 0: the matrix records outcomes, it does not gate on them.
"""

import importlib
import json
import os
import sys
import traceback


def _version(package):
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return "unknown"


def main():
    report = {
        "mlx": _version("mlx"),
        "mlx_lm": _version("mlx-lm"),
        "mlx_vlm": _version("mlx-vlm"),
        "transformers": _version("transformers"),
    }

    from unsloth_zoo.mlx.generate import (
        GenerationDefaults,
        GenerationRequest,
        _probe_sampler_api,
        _probe_text_api,
        generate_batch,
    )

    generate_module = importlib.import_module("mlx_lm.generate")
    sample_utils = importlib.import_module("mlx_lm.sample_utils")

    # --- shape inventory, independent of the probe's verdict -----------------
    batch_generator = getattr(generate_module, "BatchGenerator", None)
    generation_batch = getattr(generate_module, "GenerationBatch", None)
    report["has_BatchGenerator"] = batch_generator is not None
    report["has_GenerationBatch"] = generation_batch is not None
    report["has_GenerationBatch_Response"] = (
        getattr(generation_batch, "Response", None) is not None
    )
    report["has_BatchGenerator_Response"] = (
        getattr(batch_generator, "Response", None) is not None
    )
    for name in ("insert", "next", "next_generated", "remove", "close"):
        report[f"has_{name}"] = callable(getattr(batch_generator, name, None))
    if batch_generator is not None:
        import inspect

        try:
            report["ctor_params"] = sorted(
                inspect.signature(batch_generator).parameters
            )
        except Exception as exc:
            report["ctor_params"] = f"unavailable: {exc}"
        try:
            report["insert_params"] = sorted(
                inspect.signature(batch_generator.insert).parameters
            )
        except Exception as exc:
            report["insert_params"] = f"unavailable: {exc}"

    # --- the probes the PR actually runs -------------------------------------
    try:
        _probe_text_api(generate_module)
        report["probe_text_api"] = "PASS"
    except Exception as exc:
        report["probe_text_api"] = f"FAIL: {type(exc).__name__}: {exc}"

    try:
        _probe_sampler_api(sample_utils)
        report["probe_sampler_api"] = "PASS"
    except Exception as exc:
        report["probe_sampler_api"] = f"FAIL: {type(exc).__name__}: {exc}"

    # --- end to end on a real model ------------------------------------------
    if os.environ.get("PR956_SKIP_MODEL") == "1":
        report["generate_batch"] = "SKIPPED (PR956_SKIP_MODEL=1)"
    else:
        try:
            from mlx_lm import load

            model, tokenizer = load(os.environ.get(
                "PR956_TEXT_MODEL", "mlx-community/SmolLM-135M-Instruct-4bit"
            ))
            results = generate_batch(
                model,
                tokenizer,
                [
                    GenerationRequest(prompt="The capital of France is", max_tokens=8),
                    GenerationRequest(prompt="Two plus two equals", max_tokens=5),
                ],
                defaults=GenerationDefaults(max_tokens=16),
            )
            report["generate_batch"] = "PASS"
            report["results"] = [
                {
                    "token_ids": r.token_ids,
                    "n_logprobs": len(r.logprobs),
                    "aligned": len(r.logprobs) == len(r.token_ids),
                    "finish_reason": r.finish_reason,
                    "text": r.text,
                }
                for r in results
            ]
        except Exception as exc:
            report["generate_batch"] = f"FAIL: {type(exc).__name__}: {exc}"
            report["generate_batch_traceback"] = traceback.format_exc()

    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
