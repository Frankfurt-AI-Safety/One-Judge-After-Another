"""`scoring/experiment.py`: the config (strict about its keys) and the eval helpers the direct arm uses."""

from __future__ import annotations

import glob

import pytest
import torch

from scoring.dataset_base import EvalExample
from scoring.demographic_experiment import organize_rewards, texts_and_variants
from scoring.experiment import ExperimentConfig


def test_a_misspelt_key_is_an_error_not_a_default():
    base = {"name": "x", "bias_type": "demographic", "model_path": "m"}
    with pytest.raises(ValueError, match="max_test_example"):
        ExperimentConfig.from_dict({**base, "max_test_example": 50})
    assert ExperimentConfig.from_dict({**base, "output_dir": "legacy"}).name == "x"     # upstream key, ignored
    cfg = ExperimentConfig.from_dict({**base, "probe_records": 7, "extra": {"domain": "cv"}})
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize("path", sorted(glob.glob("configs/demographic_*.yaml")))
def test_every_config_loads(path):
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.probe_records and cfg.extra.get("domain")


def test_texts_and_rewards_round_trip():
    examples = [EvalExample(texts={"a": f"a{i}", "b": f"b{i}"}) for i in range(3)]
    texts, meta = texts_and_variants(examples)
    assert texts == ["a0", "b0", "a1", "b1", "a2", "b2"]
    rewards = torch.tensor([float(t[1]) + (0.5 if t[0] == "a" else 0.0) for t in texts])
    assert organize_rewards(rewards, meta, 3) == {"a": [0.5, 1.5, 2.5], "b": [0.0, 1.0, 2.0]}
