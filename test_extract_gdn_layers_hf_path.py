import torch

from unsloth_zoo.empty_model import extract_gdn_layers


class _Lin:
    def __init__(self, out_features, in_features):
        self.weight = torch.randn(out_features, in_features)


def _stub_get_state_dict(prefix, kk, state_dict, module, slice_weights=True):
    state_dict[f"{prefix}.weight"] = module.weight


class _Conv:
    def __init__(self, shape):
        self.weight = torch.nn.Parameter(torch.randn(*shape))


def _make_hf_style_gdn():
    class HFGDN:
        def __init__(self):
            self.in_proj_qkv = _Lin(6, 4)
            self.in_proj_z = _Lin(2, 4)
            self.in_proj_b = _Lin(2, 4)
            self.in_proj_a = _Lin(2, 4)
            self.conv1d = _Conv((6, 1, 4))
            self.dt_bias = torch.nn.Parameter(torch.randn(2))
            self.A_log = torch.nn.Parameter(torch.randn(2))
            self.out_proj = _Lin(4, 2)
            self.norm = None
    return HFGDN()


def _make_fused_gdn():
    class _Base:
        def __init__(self):
            self.weight = torch.randn(10, 4)
            self.output_sizes = [2, 2, 2, 4]

    class _QKVZ:
        def __init__(self):
            self.base_layer = _Base()

    class FusedGDN:
        def __init__(self):
            self.in_proj_qkvz = _QKVZ()
            self.in_proj_ba = _Lin(4, 4)
            self.conv1d = _Conv((6, 1, 4))
            self.dt_bias = torch.nn.Parameter(torch.randn(2))
            self.A_log = torch.nn.Parameter(torch.randn(2))
            self.out_proj = _Lin(4, 2)
            self.norm = None
    return FusedGDN()


def test_extract_gdn_layers_hf_path_without_in_proj_ba():
    gdn = _make_hf_style_gdn()
    state, quant = {}, {}
    extract_gdn_layers(gdn, "model.layers.0.linear_attn", state, quant, _stub_get_state_dict)

    assert "model.layers.0.linear_attn.in_proj_qkv.weight" in state
    assert "model.layers.0.linear_attn.in_proj_z.weight" in state
    assert "model.layers.0.linear_attn.in_proj_b.weight" in state
    assert "model.layers.0.linear_attn.in_proj_a.weight" in state
    assert "model.layers.0.linear_attn.out_proj.weight" in state
    assert "model.layers.0.linear_attn.conv1d.weight" in state


def test_extract_gdn_layers_fused_path_still_uses_in_proj_ba():
    gdn = _make_fused_gdn()
    state, quant = {}, {}
    extract_gdn_layers(gdn, "model.layers.0.linear_attn", state, quant, _stub_get_state_dict)

    assert "model.layers.0.linear_attn.in_proj_qkv.weight" in state
    assert "model.layers.0.linear_attn.in_proj_z.weight" in state
    assert "model.layers.0.linear_attn.in_proj_b.weight" in state
    assert "model.layers.0.linear_attn.in_proj_a.weight" in state


def test_extract_gdn_layers_hf_path_uses_correct_source_modules():
    gdn = _make_hf_style_gdn()
    state, quant = {}, {}
    extract_gdn_layers(gdn, "pfx", state, quant, _stub_get_state_dict)

    assert torch.equal(state["pfx.in_proj_b.weight"], gdn.in_proj_b.weight)
    assert torch.equal(state["pfx.in_proj_a.weight"], gdn.in_proj_a.weight)
