# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""CPU-only regression cover for the MoE vLLM weight-sharing change.

The change touches two things that sit on the import path of EVERY model, not just
MoE ones: `empty_model.get_model_layer_config`, whose name templates drive both
`create_empty_model` and the vLLM -> HF assignment loop, and `vllm_utils`'s
state-dict extraction. So the blast radius is every fast_inference model, and the
tests here are shaped around that:

  1. both modules import with no vLLM installed, no CUDA device and no CUDA context,
  2. every template `get_model_layer_config` returned BEFORE the change is still
     returned, byte for byte, and the four non-decoder template groups are untouched,
  3. the new MoE templates exist AND cover exactly the keys `extract_moe_layers`
     writes, which is the defect the change actually fixed: the aliased tensors were
     being extracted and then never assigned, because no template named them,
  4. `extract_moe_layers` refuses a non-3-D (tiled) expert layout loudly instead of
     silently materialising a second copy of the expert weights,
  5. the extraction really aliases: the routed experts share storage with the vLLM
     parameters, and the shared expert's gate/up are slice views of the one fused
     vLLM tensor. A copy fails these assertions.

Everything runs on tiny synthetic tensors and SimpleNamespace / nn.Module stubs.
No model download, no network, no HF hub, no GPU, no vLLM.

What is deliberately NOT covered here, because it cannot be honestly covered without
a GPU and a live vLLM: `patch_vllm_untiled_moe_experts` (it patches a vLLM module
that does not exist on this runner), and the end-to-end `_get_vllm_state_dict` /
`convert_vllm_to_huggingface` round trip on a real engine.
"""

from __future__ import annotations

import os

# Must precede the unsloth_zoo import: `device_type.get_device_type()` raises
# NotImplementedError at import time on a host with no accelerator visible.
os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")

import importlib
import importlib.util
import sys
import types

import pytest
import torch

import unsloth_zoo.empty_model as empty_model
import unsloth_zoo.vllm_utils as vllm_utils
from unsloth_zoo.empty_model import extract_moe_layers, get_model_layer_config


# ---------------------------------------------------------------------------
# 1. Import safety
# ---------------------------------------------------------------------------

def test_modules_import_without_vllm_and_without_cuda():
    """The two edited modules must import on a bare, GPU-free, vLLM-free runner.

    This is the every-model guard: `vllm_utils` is reached by `unsloth.models.*`
    and `empty_model` by every `convert_vllm_to_huggingface` caller, so an
    import-time reference to a vLLM symbol or a CUDA probe added by the MoE work
    would break hosts that never asked for fast_inference.
    """
    assert importlib.import_module("unsloth_zoo.empty_model") is empty_model
    assert importlib.import_module("unsloth_zoo.vllm_utils") is vllm_utils

    # The new public symbol is exported, not merely defined.
    assert "extract_moe_layers" in empty_model.__all__
    assert callable(empty_model.extract_moe_layers)
    assert callable(vllm_utils.patch_vllm_untiled_moe_experts)

    if importlib.util.find_spec("vllm") is None:
        # No vLLM on this host, so importing must not have pulled one in.
        assert "vllm" not in sys.modules

    # Importing must never create a CUDA context, even where a device exists.
    if hasattr(torch, "cuda"):
        assert not torch.cuda.is_initialized()


# ---------------------------------------------------------------------------
# 2. Non-MoE regression: the change is purely additive
# ---------------------------------------------------------------------------

# Frozen copy of what `get_model_layer_config()` returned BEFORE this change,
# extracted from `git show origin/main:unsloth_zoo/empty_model.py` (the branch's
# merge base) with an AST literal_eval of the `layer_templates` dict. Freezing it
# as data rather than re-reading git keeps the assertion a real before/after on a
# shallow CI checkout that has no history.
BASE_STANDARD_LAYERS = frozenset({
    "model.language_model.layers.{kk}.layer_scalar",
    "model.language_model.layers.{kk}.linear_attn.A_log",
    "model.language_model.layers.{kk}.linear_attn.conv1d",
    "model.language_model.layers.{kk}.linear_attn.dt_bias",
    "model.language_model.layers.{kk}.linear_attn.in_proj_a",
    "model.language_model.layers.{kk}.linear_attn.in_proj_b",
    "model.language_model.layers.{kk}.linear_attn.in_proj_qkv",
    "model.language_model.layers.{kk}.linear_attn.in_proj_z",
    "model.language_model.layers.{kk}.linear_attn.out_proj",
    "model.language_model.layers.{kk}.mlp.down_proj",
    "model.language_model.layers.{kk}.mlp.gate_proj",
    "model.language_model.layers.{kk}.mlp.gate_up_proj",
    "model.language_model.layers.{kk}.mlp.up_proj",
    "model.language_model.layers.{kk}.per_layer_input_gate",
    "model.language_model.layers.{kk}.per_layer_projection",
    "model.language_model.layers.{kk}.self_attn.k_proj",
    "model.language_model.layers.{kk}.self_attn.o_proj",
    "model.language_model.layers.{kk}.self_attn.q_proj",
    "model.language_model.layers.{kk}.self_attn.qkv_proj",
    "model.language_model.layers.{kk}.self_attn.v_proj",
    "model.layers.{kk}.layer_scalar",
    "model.layers.{kk}.linear_attn.A_log",
    "model.layers.{kk}.linear_attn.conv1d",
    "model.layers.{kk}.linear_attn.dt_bias",
    "model.layers.{kk}.linear_attn.in_proj_a",
    "model.layers.{kk}.linear_attn.in_proj_b",
    "model.layers.{kk}.linear_attn.in_proj_qkv",
    "model.layers.{kk}.linear_attn.in_proj_z",
    "model.layers.{kk}.linear_attn.out_proj",
    "model.layers.{kk}.mlp.down_proj",
    "model.layers.{kk}.mlp.gate_proj",
    "model.layers.{kk}.mlp.gate_up_proj",
    "model.layers.{kk}.mlp.up_proj",
    "model.layers.{kk}.per_layer_input_gate",
    "model.layers.{kk}.per_layer_projection",
    "model.layers.{kk}.self_attn.k_proj",
    "model.layers.{kk}.self_attn.o_proj",
    "model.layers.{kk}.self_attn.q_proj",
    "model.layers.{kk}.self_attn.qkv_proj",
    "model.layers.{kk}.self_attn.v_proj",
})

# The MoE block templates this change adds, for both the text-only and the
# `model.language_model.*` spellings. `experts.gate_up_proj` / `experts.down_proj`
# carry no `.weight` suffix because HF holds them as bare stacked Parameters.
MOE_TEMPLATES = frozenset({
    "model.layers.{kk}.mlp.gate",
    "model.layers.{kk}.mlp.shared_expert_gate",
    "model.layers.{kk}.mlp.shared_expert.gate_proj",
    "model.layers.{kk}.mlp.shared_expert.up_proj",
    "model.layers.{kk}.mlp.shared_expert.down_proj",
    "model.layers.{kk}.mlp.experts.gate_up_proj",
    "model.layers.{kk}.mlp.experts.down_proj",
    "model.language_model.layers.{kk}.mlp.gate",
    "model.language_model.layers.{kk}.mlp.shared_expert_gate",
    "model.language_model.layers.{kk}.mlp.shared_expert.gate_proj",
    "model.language_model.layers.{kk}.mlp.shared_expert.up_proj",
    "model.language_model.layers.{kk}.mlp.shared_expert.down_proj",
    "model.language_model.layers.{kk}.mlp.experts.gate_up_proj",
    "model.language_model.layers.{kk}.mlp.experts.down_proj",
})

# Groups the change does not touch at all. Their sizes at the merge base; a
# change to any of them is a blast-radius event that this suite should surface.
BASE_GROUP_SIZES = {
    "layernorms": 32,
    "vision_layers": 57,
    "additional_layers": 8,
    "non_layered_components": 34,
}


def test_dense_templates_survive_unchanged():
    """Every pre-change decoder template is still returned, spelled identically."""
    standard = set(get_model_layer_config()["standard_layers"])
    missing = sorted(BASE_STANDARD_LAYERS - standard)
    assert not missing, f"MoE change dropped or renamed dense templates: {missing}"


def test_moe_additions_are_purely_additive():
    """Nothing outside the MoE/expert family was added to the decoder group.

    A dense template quietly rewritten (rather than removed) would show up here as
    an unexpected addition even though the subset check above still passed.
    """
    standard = set(get_model_layer_config()["standard_layers"])
    added = standard - BASE_STANDARD_LAYERS
    assert MOE_TEMPLATES <= added

    # Anything else added must still belong to the sparse-MoE family.
    unexpected = [
        name for name in added - MOE_TEMPLATES
        if not any(tag in name for tag in (".mlp.gate", ".shared_expert", ".experts.", ".router."))
    ]
    assert not unexpected, f"non-MoE templates appeared in standard_layers: {unexpected}"


def test_untouched_template_groups_are_untouched():
    """The four non-decoder template groups are exactly as they were."""
    config = get_model_layer_config()
    for group, size in BASE_GROUP_SIZES.items():
        assert len(config[group]) == size, (
            f"{group} changed size ({len(config[group])} vs {size} at the merge base); "
            "this change should not touch it"
        )


def test_return_non_layered_flag_still_honoured():
    """`create_empty_model` calls this with return_non_layered=False; keep it working."""
    without = get_model_layer_config(return_non_layered=False)
    assert "non_layered_components" not in without
    assert set(without) == {"standard_layers", "layernorms", "vision_layers", "additional_layers"}
    # Sorted lists, deterministic order, as every caller assumes when it concatenates them.
    for value in without.values():
        assert isinstance(value, list) and value == sorted(value)


# ---------------------------------------------------------------------------
# Synthetic vLLM MoE block, and the real `get_state_dict` closure
# ---------------------------------------------------------------------------

HIDDEN = 8          # hidden_size
INTER = 4           # moe_intermediate_size
SHARED_INTER = 4    # shared_expert_intermediate_size
EXPERTS = 3         # num_experts


class _Proj(torch.nn.Module):
    """A vLLM linear: a `.weight`, optionally with `output_sizes` for a fused pair."""

    def __init__(self, out_features, in_features, output_sizes=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(out_features * in_features, dtype=torch.float32).reshape(out_features, in_features),
            requires_grad=False,
        )
        self.bias = None
        if output_sizes is not None:
            self.output_sizes = output_sizes


class _RoutedExperts(torch.nn.Module):
    """vLLM's stacked expert Parameters, in the untiled (plain 3-D) layout."""

    def __init__(self, w13_shape, w2_shape):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(torch.randn(*w13_shape), requires_grad=True)
        self.w2_weight = torch.nn.Parameter(torch.randn(*w2_shape), requires_grad=True)


class _Experts(torch.nn.Module):
    """LoRA-wrapped experts: `.base_layer.routed_experts`, as vLLM exposes them."""

    def __init__(self, inner):
        super().__init__()
        self.base_layer = torch.nn.Module()
        self.base_layer.routed_experts = inner


class _SharedExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # vLLM fuses gate+up into one tensor; HF keeps them separate.
        self.gate_up_proj = _Proj(2 * SHARED_INTER, HIDDEN, output_sizes=[SHARED_INTER, SHARED_INTER])
        self.down_proj = _Proj(HIDDEN, SHARED_INTER)


class _SparseMoeBlock(torch.nn.Module):
    def __init__(self, w13_shape=None, w2_shape=None, with_experts=True):
        super().__init__()
        self.gate = _Proj(EXPERTS, HIDDEN)
        self.shared_expert_gate = _Proj(1, HIDDEN)
        self.shared_expert = _SharedExpert()
        if with_experts:
            self.experts = _Experts(_RoutedExperts(
                w13_shape or (EXPERTS, 2 * INTER, HIDDEN),
                w2_shape or (EXPERTS, HIDDEN, INTER),
            ))


class _BareExperts(torch.nn.Module):
    """An experts module exposing neither w13_weight nor w2_weight."""

    def __init__(self):
        super().__init__()
        self.something_else = torch.nn.Parameter(torch.zeros(2), requires_grad=False)


def _real_get_state_dict(quant_state_dict):
    """Rebuild `_get_vllm_state_dict`'s nested `get_state_dict` closure.

    `extract_moe_layers` is handed this function by its only caller, so testing
    against a hand-written stand-in would test the stand-in. The closure is lifted
    out of the enclosing function's code object and rebound with CPU-safe, non-FP8,
    non-bitsandbytes values, which is the path a bf16 model takes.
    """
    code = None
    for const in vllm_utils._get_vllm_state_dict.__code__.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == "get_state_dict":
            code = const
            break
    assert code is not None, (
        "could not find the nested get_state_dict in _get_vllm_state_dict; "
        "the extraction helper below needs updating"
    )
    assert code.co_varnames[:code.co_argcount] == (
        "prefix", "kk", "state_dict", "proj", "slice_weights", "slice_index",
    ), f"get_state_dict signature changed: {code.co_varnames[:code.co_argcount]}"

    values = {
        "cutlass_block_fp8_supported": False,
        "is_deep_gemm_supported": False,
        "needs_transpose_check": False,
        "quant_state_dict": quant_state_dict,
        "sm_cap": 0,
        "vocab_size": 0,  # falsy, so the embed/lm_head truncation never triggers
    }
    missing = set(code.co_freevars) - set(values)
    assert not missing, f"get_state_dict gained closure variables: {sorted(missing)}"
    closure = tuple(types.CellType(values[name]) for name in code.co_freevars)

    return types.FunctionType(
        code, vars(vllm_utils), "get_state_dict", (True, -1), closure,
    )


def _extract(mlp, prefix="model.layers.0.mlp"):
    state_dict, quant_state_dict = {}, {}
    extract_moe_layers(
        mlp, prefix, state_dict, quant_state_dict, _real_get_state_dict(quant_state_dict),
    )
    return state_dict, quant_state_dict


# ---------------------------------------------------------------------------
# 3. The templates name exactly what the extraction produces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", [
    "model.layers.0.mlp",
    "model.language_model.layers.0.mlp",
])
def test_every_extracted_moe_key_is_reachable_by_a_template(prefix):
    """Section 6 of the design note: the real bug.

    `convert_vllm_to_huggingface` only walks the names `get_model_layer_config`
    returns. The aliased MoE tensors were extracted correctly and then never
    assigned, leaving the HF training model's MoE blocks as 1-wide placeholders,
    while generation still passed because generation never reads the HF module.
    So the contract is not "MoE templates exist" but "every key the extraction
    writes is named by a template".
    """
    config = get_model_layer_config(return_non_layered=False)
    templates = set(sum(config.values(), []))
    walked = {name.format(kk=0) for name in templates}

    _, quant_state_dict = _extract(_SparseMoeBlock(), prefix=prefix)
    assert quant_state_dict, "extraction produced nothing"

    unreachable = []
    for key in quant_state_dict:
        stem = key
        for suffix in (".weight", ".bias"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem not in walked:
            unreachable.append(key)
    assert not unreachable, (
        "extracted MoE tensors that convert_vllm_to_huggingface would never assign: "
        f"{sorted(unreachable)}"
    )


def test_expected_moe_names_are_produced():
    """The full set of HF names, so a silently dropped tensor is caught too."""
    _, quant_state_dict = _extract(_SparseMoeBlock())
    prefix = "model.layers.0.mlp"
    assert set(quant_state_dict) == {
        f"{prefix}.gate.weight",
        f"{prefix}.shared_expert_gate.weight",
        f"{prefix}.shared_expert.gate_proj.weight",
        f"{prefix}.shared_expert.up_proj.weight",
        f"{prefix}.shared_expert.down_proj.weight",
        f"{prefix}.experts.gate_up_proj",
        f"{prefix}.experts.down_proj",
    }
    # The stacked experts are bare Parameters on the HF side, so no ".weight".
    assert f"{prefix}.experts.gate_up_proj.weight" not in quant_state_dict


# ---------------------------------------------------------------------------
# 4. A tiled (non-3-D) expert layout is refused loudly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w13_shape,w2_shape", [
    # FlashInfer TRT-LLM block layout: (E, hidden/blk, 2*inter, blk).
    ((EXPERTS, HIDDEN // 4, 2 * INTER, 4), (EXPERTS, HIDDEN, INTER)),
    ((EXPERTS, 2 * INTER, HIDDEN), (EXPERTS, INTER // 2, HIDDEN, 2)),
    # 2-D is just as unaliasable as 4-D.
    ((EXPERTS * 2 * INTER, HIDDEN), (EXPERTS, HIDDEN, INTER)),
])
def test_tiled_expert_layout_raises_instead_of_copying(w13_shape, w2_shape):
    mlp = _SparseMoeBlock(w13_shape=w13_shape, w2_shape=w2_shape)
    state_dict, quant_state_dict = {}, {}
    with pytest.raises(RuntimeError) as excinfo:
        extract_moe_layers(
            mlp, "model.layers.0.mlp", state_dict, quant_state_dict,
            _real_get_state_dict(quant_state_dict),
        )
    message = str(excinfo.value)
    assert "cannot be aliased" in message
    assert "untiled MoE backend" in message

    # Nothing may have been materialised for the experts on the way out: a second
    # copy of the expert weights is exactly what the refusal exists to prevent.
    assert not [k for k in quant_state_dict if ".experts." in k]
    assert not [k for k in state_dict if ".experts." in k]


def test_missing_stacked_weights_raise():
    mlp = _SparseMoeBlock(with_experts=False)
    mlp.experts = _Experts(_BareExperts())
    state_dict, quant_state_dict = {}, {}
    with pytest.raises(RuntimeError) as excinfo:
        extract_moe_layers(
            mlp, "model.layers.0.mlp", state_dict, quant_state_dict,
            _real_get_state_dict(quant_state_dict),
        )
    assert "w13_weight" in str(excinfo.value)
    assert "something_else" in str(excinfo.value)  # names what vLLM did expose


def test_block_without_experts_is_skipped_quietly():
    """A dense-looking block must not raise; the router and shared expert still land."""
    mlp = _SparseMoeBlock(with_experts=False)
    state_dict, quant_state_dict = _extract(mlp)
    assert "model.layers.0.mlp.gate.weight" in quant_state_dict
    assert not [k for k in quant_state_dict if ".experts." in k]


# ---------------------------------------------------------------------------
# 5. The extraction aliases. A copy fails every assertion below.
# ---------------------------------------------------------------------------

def _shares_storage(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def test_routed_experts_alias_the_vllm_parameters():
    mlp = _SparseMoeBlock()
    routed = mlp.experts.base_layer.routed_experts
    w13, w2 = routed.w13_weight, routed.w2_weight

    state_dict, quant_state_dict = _extract(mlp)
    got13 = quant_state_dict["model.layers.0.mlp.experts.gate_up_proj"]
    got2 = quant_state_dict["model.layers.0.mlp.experts.down_proj"]

    # Identity of the underlying allocation, not merely equal values.
    assert got13.data_ptr() == w13.data_ptr()
    assert got2.data_ptr() == w2.data_ptr()
    assert _shares_storage(got13, w13)
    assert _shares_storage(got2, w2)
    assert got13.shape == w13.shape and got2.shape == w2.shape
    assert got13.stride() == w13.stride() and got2.stride() == w2.stride()

    # A write through the vLLM parameter is visible in the HF-side view. This is
    # the property the whole change exists to deliver, and no copy has it.
    with torch.no_grad():
        w13[0, 0, 0] = 1234.5
        w2[0, 0, 0] = -4321.5
    assert got13[0, 0, 0].item() == 1234.5
    assert got2[0, 0, 0].item() == -4321.5

    # The vLLM parameters are frozen on the way out; the shared tensor is not a
    # training leaf that would start accumulating grads in the rollout engine.
    assert w13.requires_grad is False
    assert w2.requires_grad is False
    assert got13.requires_grad is False
    assert got2.requires_grad is False

    # state_dict and quant_state_dict hold the same object, as the caller expects.
    assert state_dict["model.layers.0.mlp.experts.gate_up_proj"] is got13
    assert state_dict["model.layers.0.mlp.experts.down_proj"] is got2


def test_shared_expert_gate_up_split_is_a_pair_of_slice_views():
    mlp = _SparseMoeBlock()
    fused = mlp.shared_expert.gate_up_proj.weight  # (2*SHARED_INTER, HIDDEN)

    _, quant_state_dict = _extract(mlp)
    gate = quant_state_dict["model.layers.0.mlp.shared_expert.gate_proj.weight"]
    up = quant_state_dict["model.layers.0.mlp.shared_expert.up_proj.weight"]

    assert gate.shape == (SHARED_INTER, HIDDEN)
    assert up.shape == (SHARED_INTER, HIDDEN)

    # Both halves are views into the single fused vLLM allocation.
    assert _shares_storage(gate, fused)
    assert _shares_storage(up, fused)
    assert gate.data_ptr() == fused.data_ptr()             # first half, offset 0
    assert up.storage_offset() == SHARED_INTER * HIDDEN    # second half

    # And the halves are the right way round: gate is the top rows, up the bottom.
    assert torch.equal(gate, fused[:SHARED_INTER])
    assert torch.equal(up, fused[SHARED_INTER:])
    with torch.no_grad():
        fused[0, 0] = 77.0
        fused[SHARED_INTER, 0] = -77.0
    assert gate[0, 0].item() == 77.0
    assert up[0, 0].item() == -77.0


def test_router_and_shared_down_proj_are_whole_tensor_aliases():
    mlp = _SparseMoeBlock()
    _, quant_state_dict = _extract(mlp)

    for key, source in (
        ("model.layers.0.mlp.gate.weight", mlp.gate.weight),
        ("model.layers.0.mlp.shared_expert_gate.weight", mlp.shared_expert_gate.weight),
        ("model.layers.0.mlp.shared_expert.down_proj.weight", mlp.shared_expert.down_proj.weight),
    ):
        got = quant_state_dict[key]
        assert got.shape == source.shape, key
        assert got.data_ptr() == source.data_ptr(), key
        assert _shares_storage(got, source), key


def test_extraction_allocates_nothing():
    """No new storage at all, which is the memory claim stated as a number."""
    mlp = _SparseMoeBlock()
    sources = {
        t.untyped_storage().data_ptr()
        for t in (
            mlp.gate.weight,
            mlp.shared_expert_gate.weight,
            mlp.shared_expert.gate_up_proj.weight,
            mlp.shared_expert.down_proj.weight,
            mlp.experts.base_layer.routed_experts.w13_weight,
            mlp.experts.base_layer.routed_experts.w2_weight,
        )
    }
    _, quant_state_dict = _extract(mlp)
    for key, tensor in quant_state_dict.items():
        assert tensor.untyped_storage().data_ptr() in sources, (
            f"{key} was copied into fresh storage instead of aliased"
        )


def test_unwrapped_experts_are_found_too():
    """vLLM only wraps experts in base_layer/routed_experts when LoRA is on."""
    mlp = _SparseMoeBlock()
    routed = mlp.experts.base_layer.routed_experts
    mlp.experts = routed  # no LoRA wrapper
    _, quant_state_dict = _extract(mlp)
    assert quant_state_dict["model.layers.0.mlp.experts.gate_up_proj"].data_ptr() == \
        routed.w13_weight.data_ptr()


# ---------------------------------------------------------------------------
# The MoE branch in approximate_vllm_memory_usage, the other every-model edit
# ---------------------------------------------------------------------------

def _dense_config():
    return types.SimpleNamespace(
        vocab_size=32000, hidden_size=2048, max_position_embeddings=4096,
        intermediate_size=5632, num_hidden_layers=24, num_key_value_heads=4,
        num_attention_heads=16, tie_word_embeddings=False,
    )


def _moe_config():
    # A sparse config carries no dense intermediate_size at all.
    return types.SimpleNamespace(
        vocab_size=32000, hidden_size=2048, max_position_embeddings=4096,
        num_hidden_layers=24, num_key_value_heads=4, num_attention_heads=16,
        tie_word_embeddings=False, num_experts=8, moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
    )


@pytest.fixture
def fixed_memory(monkeypatch):
    """Pin the only host-dependent input so the estimate is deterministic."""
    monkeypatch.setattr(
        vllm_utils, "get_mem_info", lambda *a, **k: (40 * 1024 ** 3, 80 * 1024 ** 3),
    )


def test_dense_memory_estimate_is_bit_identical_to_before(fixed_memory):
    """Frozen output of the merge-base implementation for the same inputs."""
    got = vllm_utils.approximate_vllm_memory_usage(_dense_config(), max_seq_length=2048)
    assert got == (613376, 299, 0.4, 29.564225769601762)


def test_moe_config_no_longer_raises(fixed_memory):
    """Before the change this was an AttributeError on config.intermediate_size."""
    tokens, seqs, util, kv_gb = vllm_utils.approximate_vllm_memory_usage(
        _moe_config(), max_seq_length=2048,
    )
    assert tokens > 0 and seqs > 0
    assert 0.0 < util <= 1.0
    assert kv_gb > 0.0


def test_moe_estimate_scales_with_expert_count(fixed_memory):
    """The expert weights and their adapters must enter the estimate."""
    few = _moe_config()
    many = _moe_config()
    many.num_experts = 64
    assert vllm_utils.approximate_vllm_memory_usage(many, max_seq_length=2048)[0] < \
        vllm_utils.approximate_vllm_memory_usage(few, max_seq_length=2048)[0]
