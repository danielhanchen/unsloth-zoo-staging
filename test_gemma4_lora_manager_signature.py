import inspect
import sys
import types

import pytest


def _install_vllm_stubs(monkeypatch):
    vllm = types.ModuleType("vllm")
    model_executor = types.ModuleType("vllm.model_executor")
    models_mod = types.ModuleType("vllm.model_executor.models")
    gemma4_mm = types.ModuleType("vllm.model_executor.models.gemma4_mm")
    interfaces = types.ModuleType("vllm.model_executor.models.interfaces")
    lora_pkg = types.ModuleType("vllm.lora")
    lora_mm = types.ModuleType("vllm.lora.model_manager")
    v1_pkg = types.ModuleType("vllm.v1")
    worker_pkg = types.ModuleType("vllm.v1.worker")
    lora_mixin = types.ModuleType("vllm.v1.worker.lora_model_runner_mixin")

    class Gemma4ForConditionalGeneration:
        supports_lora = False
        embedding_modules = None
    gemma4_mm.Gemma4ForConditionalGeneration = Gemma4ForConditionalGeneration

    def supports_lora(model):
        return False
    interfaces.supports_lora = supports_lora
    lora_mixin.supports_lora = supports_lora

    class LoRAModelManager:
        def __init__(self, model, max_num_seqs, max_num_batched_tokens,
                     vocab_size, lora_config, device, vllm_config=None):
            self.model = model
            self.vllm_config = vllm_config
    lora_mm.LoRAModelManager = LoRAModelManager

    def create_lora_manager(model, max_num_seqs, max_num_batched_tokens,
                            vocab_size, lora_config, vllm_config, device,
                            lora_manager_cls=LoRAModelManager, **kwargs):
        return lora_manager_cls(
            model, max_num_seqs, max_num_batched_tokens,
            vocab_size, lora_config, device, vllm_config=vllm_config,
        )
    lora_mm.create_lora_manager = create_lora_manager

    worker_manager_stub = types.ModuleType("unsloth_zoo.vllm_lora_worker_manager")
    worker_manager_stub.create_lora_manager = lora_mm.create_lora_manager

    for name, mod in [
        ("vllm", vllm),
        ("vllm.model_executor", model_executor),
        ("vllm.model_executor.models", models_mod),
        ("vllm.model_executor.models.gemma4_mm", gemma4_mm),
        ("vllm.model_executor.models.interfaces", interfaces),
        ("vllm.lora", lora_pkg),
        ("vllm.lora.model_manager", lora_mm),
        ("vllm.v1", v1_pkg),
        ("vllm.v1.worker", worker_pkg),
        ("vllm.v1.worker.lora_model_runner_mixin", lora_mixin),
        ("unsloth_zoo.vllm_lora_worker_manager", worker_manager_stub),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    return gemma4_mm, lora_mm, worker_manager_stub


def test_patched_create_lora_manager_signature_preserves_vllm_config(monkeypatch):
    _, lora_mm, _worker = _install_vllm_stubs(monkeypatch)

    import unsloth_zoo.empty_model as em
    em.patch_gemma4_vllm_lora_support()

    sig = inspect.signature(lora_mm.create_lora_manager)
    assert "vllm_config" in sig.parameters


def test_patched_create_lora_manager_signature_matches_original(monkeypatch):
    _, lora_mm, _worker = _install_vllm_stubs(monkeypatch)
    original_params = list(inspect.signature(lora_mm.create_lora_manager).parameters.keys())

    import unsloth_zoo.empty_model as em
    em.patch_gemma4_vllm_lora_support()

    patched_params = list(inspect.signature(lora_mm.create_lora_manager).parameters.keys())
    assert patched_params == original_params


def test_patched_create_lora_manager_retains_marker(monkeypatch):
    _, lora_mm, _worker = _install_vllm_stubs(monkeypatch)

    import unsloth_zoo.empty_model as em
    em.patch_gemma4_vllm_lora_support()

    assert hasattr(lora_mm.create_lora_manager, "_unsloth_gemma4_patch")
    assert lora_mm.create_lora_manager._unsloth_gemma4_patch is True


def test_patched_create_lora_manager_forwards_vllm_config_for_gemma4(monkeypatch):
    gemma4_mm, lora_mm, _worker = _install_vllm_stubs(monkeypatch)

    import unsloth_zoo.empty_model as em
    em.patch_gemma4_vllm_lora_support()

    model = gemma4_mm.Gemma4ForConditionalGeneration()
    mgr = lora_mm.create_lora_manager(
        model,
        vllm_config="vllm_cfg",
        max_num_seqs=32,
        max_num_batched_tokens=512,
        vocab_size=128000,
        lora_config="lora_cfg",
        device="cuda:0",
    )
    assert mgr.model is model
    assert mgr.vllm_config == "vllm_cfg"
