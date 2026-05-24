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

"""Combined coverage for unslothai/unsloth-zoo PR #692 + PR #679.

PR #692 (fix-mlx-export-adapters) makes ``save_lora_adapters`` keep only
tensors whose flattened name contains ``lora_``. PR #679
(fix/mlx-lora-adapter-metadata) makes the same save path persist live
``rank`` / ``scale`` / ``dropout`` plus ``peft_type=LORA`` into
``adapter_config.json``.

Neither PR shipped a test that exercises the combined surface; this
file closes that gap. Uses the ``mlx_simulation`` shim so it runs on
non-Apple CI (Linux/Windows) as well as macOS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


# ---------------------------------------------------------------------------
# Minimal mock LoRA module: exposes the attributes _enrich_mlx_adapter_config
# inspects (``lora_a``, ``lora_b``, ``scale``, ``dropout``) and the iteration
# helpers ``save_lora_adapters`` walks (``parameters``, ``named_modules``).
# ---------------------------------------------------------------------------
class _MockDropout:
    """Mirrors MLX's nn.Dropout: stores keep-probability as ``_p_1``.

    _get_mlx_dropout_probability reads ``.p`` first, then falls back to
    ``1.0 - _p_1``; this fixture exercises the fallback branch (PR #679
    edge case).
    """

    def __init__(self, p: float):
        self._p_1 = 1.0 - p


class _MockLoRALinear:
    """LoRA-wrapped Linear. Shape convention matches mlx-lm's LoRALinear:
    ``lora_a`` is ``(in_features, rank)`` and ``lora_b`` is
    ``(rank, out_features)`` so the matmul partners pair up as
    ``lora_a_shape[-1] == lora_b_shape[0] == rank`` -- which is what
    PR #679's ``_infer_mlx_lora_rank`` reads.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, scale: float, dropout: float):
        self.weight = torch.zeros(out_features, in_features)
        self.lora_a = torch.zeros(in_features, rank)
        self.lora_b = torch.zeros(rank, out_features)
        self.scale = scale
        self.dropout = _MockDropout(dropout)


class _MockPlainLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.zeros(out_features, in_features)


class _MockModel:
    """A tiny model with one LoRA-wrapped attention proj + one plain MLP proj.

    Module attribute names are picked to **not** contain ``lora_`` so the
    substring filter in ``save_lora_adapters`` is exercised against the
    realistic case where only the leaf parameter (``...lora_a`` /
    ``...lora_b``) carries the prefix -- mirroring real MLX models where
    LoRA is attached on e.g. ``model.layers.N.self_attn.q_proj.lora_*``.

    ``parameters()`` returns the flat name→tensor map ``save_lora_adapters``
    consumes via ``mlx.utils.tree_flatten``. ``named_modules()`` is the
    iteration ``_enrich_mlx_adapter_config`` walks looking for objects with
    ``lora_a`` + ``lora_b`` to record rank/scale/dropout.
    """

    def __init__(self):
        self.q_proj = _MockLoRALinear(
            in_features=8, out_features=16, rank=4, scale=2.5, dropout=0.25,
        )
        self.up_proj = _MockPlainLinear(in_features=16, out_features=32)
        # _enrich_mlx_adapter_config probes these; supplying None keeps the
        # config helper on the cheap fast path.
        self._hf_repo = "unsloth/tiny-test-model"
        self._config = None
        self._unsloth_quantization_config = None
        self._unsloth_quantization_policy = None
        self._unsloth_quantized_source = None
        self._unsloth_base_revision = None
        self._unsloth_base_commit_hash = None
        self._src_path = None

    def parameters(self):
        return {
            "q_proj.weight": self.q_proj.weight,
            "q_proj.lora_a": self.q_proj.lora_a,
            "q_proj.lora_b": self.q_proj.lora_b,
            "up_proj.weight": self.up_proj.weight,
        }

    def trainable_parameters(self):
        return self.parameters()

    def named_modules(self):
        yield "", self
        yield "q_proj", self.q_proj
        yield "up_proj", self.up_proj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_save_lora_adapters_filters_to_lora_only(tmp_path: Path):
    """PR #692: adapter export keeps only ``lora_*`` keys.

    Both ``lora_layer.weight`` (base weight inside the LoRA module) and
    ``base_layer.weight`` (plain Linear) must be excluded — only the two
    ``lora_a`` / ``lora_b`` tensors survive.
    """
    from unsloth_zoo.mlx.utils import save_lora_adapters

    model = _MockModel()
    out_dir = tmp_path / "adapter_lora_only"
    save_lora_adapters(model, out_dir)

    safe_path = out_dir / "adapters.safetensors"
    assert safe_path.is_file(), "adapters.safetensors must be written"

    from safetensors.torch import load_file
    saved = load_file(str(safe_path))
    keys = set(saved.keys())

    assert keys == {"q_proj.lora_a", "q_proj.lora_b"}, (
        f"adapter export leaked non-LoRA tensors: {sorted(keys)}"
    )


def test_save_lora_adapters_writes_pr679_metadata(tmp_path: Path):
    """PR #679: adapter_config.json carries live rank/scale/dropout +
    peft_type=LORA + the keep-probability dropout fallback.
    """
    from unsloth_zoo.mlx.utils import save_lora_adapters

    model = _MockModel()
    out_dir = tmp_path / "adapter_metadata"
    save_lora_adapters(model, out_dir)

    cfg_path = out_dir / "adapter_config.json"
    assert cfg_path.is_file(), "adapter_config.json must be written"
    cfg = json.loads(cfg_path.read_text())

    assert cfg.get("peft_type") == "LORA", cfg
    assert cfg.get("rank") == 4, cfg
    assert cfg.get("scale") == pytest.approx(2.5), cfg
    # PR #679 fix: read dropout via the ``_p_1`` keep-prob fallback when
    # ``.p`` is absent (real MLX nn.Dropout stores it that way).
    assert cfg.get("dropout") == pytest.approx(0.25), cfg

    params = cfg.get("lora_parameters") or {}
    assert params.get("rank") == 4, params
    assert params.get("scale") == pytest.approx(2.5), params
    assert params.get("dropout") == pytest.approx(0.25), params

    # PR #679 also records the LoRA topology so reload reconstructs the
    # same attachment surface.
    assert "q_proj" in (cfg.get("unsloth_mlx_lora_module_paths") or []), cfg


def test_save_lora_adapters_raises_when_no_lora_tensors_present(tmp_path: Path):
    """PR #692: explicit ValueError when nothing matched the ``lora_`` filter.

    Guards against silently saving an empty/garbage adapter export when the
    model has no LoRA layers (e.g. merged adapter state).
    """
    from unsloth_zoo.mlx.utils import save_lora_adapters

    class _NoLoRAModel(_MockModel):
        def parameters(self):
            return {"up_proj.weight": self.up_proj.weight}

        def named_modules(self):
            yield "", self
            yield "up_proj", self.up_proj

    out_dir = tmp_path / "adapter_empty"
    with pytest.raises(ValueError, match="LoRA adapter tensors"):
        save_lora_adapters(_NoLoRAModel(), out_dir)


def test_save_trainable_adapters_keeps_all_trainable(tmp_path: Path):
    """PR #692 separation: ``save_trainable_adapters`` (used by mid-training
    checkpoints) keeps ALL trainable tensors, including base weights — it
    must NOT inherit the LoRA-only filter.
    """
    from unsloth_zoo.mlx.utils import save_trainable_adapters

    model = _MockModel()
    out_dir = tmp_path / "adapter_trainable"
    save_trainable_adapters(model, out_dir)

    safe_path = out_dir / "adapters.safetensors"
    assert safe_path.is_file(), "adapters.safetensors must be written"

    from safetensors.torch import load_file
    saved = load_file(str(safe_path))
    keys = set(saved.keys())

    assert keys == {
        "q_proj.weight",
        "q_proj.lora_a",
        "q_proj.lora_b",
        "up_proj.weight",
    }, (
        "training checkpoint should preserve every trainable tensor, "
        f"got {sorted(keys)}"
    )
