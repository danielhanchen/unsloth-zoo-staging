from unsloth_zoo.empty_model import get_model_layer_config


def test_merger_linear_fc_entries_are_non_layered():
    cfg = get_model_layer_config()
    non_layered = set(cfg["non_layered_components"])
    assert "model.visual.merger.linear_fc1" in non_layered
    assert "model.visual.merger.linear_fc2" in non_layered


def test_merger_linear_fc_templated_entry_removed():
    cfg = get_model_layer_config()
    for category, entries in cfg.items():
        assert "model.visual.merger.linear_fc{kk}" not in set(entries), category


def test_deepstack_merger_list_remains_templated():
    cfg = get_model_layer_config()
    additional = set(cfg["additional_layers"])
    assert "model.visual.deepstack_merger_list.{kk}.linear_fc1" in additional
    assert "model.visual.deepstack_merger_list.{kk}.linear_fc2" in additional
