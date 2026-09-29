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


# --------------------------------------------------------------------------- run provenance and overrides
def _built(tmp_path):
    from pairs.manifest import write_manifest

    rows = [{"id": "p0", "varied_axis": "sex", "encoding": "explicit", "template_id": "credit_v1",
             "label_a": "female", "label_b": "male", "text_a": "a", "text_b": "b"}]
    return write_manifest(tmp_path, rows, seed=7, discard_report={}, thresholds={}, domain="credit",
                          attribution="a", cells=[{"id": "c0"}])


def test_data_file_names_the_bytes_its_manifest_describes(tmp_path):
    import hashlib

    from scoring.experiment import data_file

    paths = _built(tmp_path)
    d = data_file(paths["pairs"])
    assert d["sha256"] == hashlib.sha256(paths["pairs"].read_bytes()).hexdigest()
    assert (d["n_rows"], d["generator_seed"], d["path"]) == (1, 7, str(paths["pairs"]))
    assert data_file(paths["cells"])["n_rows"] == 1


def test_data_file_refuses_a_file_from_another_build(tmp_path):
    import json

    from scoring.experiment import data_file

    paths = _built(tmp_path)
    with open(paths["pairs"], "a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="different builds"):
        data_file(paths["pairs"])
    m = json.loads(paths["manifest"].read_text())
    del m["files"]  # a manifest from before 0.4.0
    paths["manifest"].write_text(json.dumps(m))
    with pytest.raises(ValueError, match="regenerate"):
        data_file(paths["cells"])
    paths["manifest"].unlink()
    with pytest.raises(FileNotFoundError):
        data_file(paths["cells"])


def test_run_metadata_carries_the_loaded_commit_and_the_data(tmp_path):
    from scoring.experiment import ExperimentConfig, data_file, run_metadata

    cfg = ExperimentConfig(name="x", bias_type="demographic", model_path="org/rm")
    cfg.model_revision = "abc123"  # what load_model writes
    data = {"pairs.jsonl": data_file(_built(tmp_path)["pairs"])}
    meta = run_metadata(cfg, data, {"n_boot": 10})
    assert meta["config"]["model_revision"] == "abc123" and meta["data"] == data
    assert meta["settings"] == {"n_boot": 10} and set(meta["code"]) == {"git_commit", "git_dirty", "git_dirty_paths"}
    assert meta["created_utc"].endswith("+00:00") and isinstance(meta["argv"], list)


def test_overrides_apply_only_what_was_given():
    import argparse

    from scoring.experiment import ExperimentConfig, add_override_args, apply_overrides

    ap = argparse.ArgumentParser()
    add_override_args(ap)
    cfg = ExperimentConfig(name="x", bias_type="demographic", model_path="org/rm", batch_size=8, probe_records=150)
    cfg = apply_overrides(cfg, ap.parse_args(["--model", "org/big", "--revision", "r1", "--probe-records", "50"]))
    assert (cfg.model_path, cfg.model_revision, cfg.probe_records, cfg.batch_size, cfg.device) == \
        ("org/big", "r1", 50, 8, "auto")
