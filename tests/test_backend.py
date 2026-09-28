"""The model-loading entry point must exist and be what `BiasExperiment.load_model` calls.

Regression: `create_backend` was deleted with the MLX backend while `scoring/experiment.py` kept
importing it, so every runner failed at `load_model()`; no test loaded a model."""

from __future__ import annotations

from types import SimpleNamespace


def test_create_backend_delegates_to_the_transformers_loader(monkeypatch):
    import scoring.backend as backend

    seen = {}
    monkeypatch.setattr(backend, "_load_transformers",
                        lambda cfg: seen.setdefault("cfg", cfg) and ("model", "tokenizer"))
    cfg = SimpleNamespace(model_path="m")
    assert backend.create_backend(cfg) == ("model", "tokenizer")
    assert seen["cfg"] is cfg


def test_load_model_uses_it_and_attaches_the_embedding_cache(monkeypatch, tmp_path):
    import scoring.backend as backend
    from probes import embedding_cache as ec
    from scoring.experiment import ExperimentConfig
    from scoring.demographic_experiment import DemographicBiasExperiment
    from tests.test_embedding_cache import _model, _tokenizer

    monkeypatch.delenv(ec.ENV_VAR, raising=False)
    model, tok = _model(), _tokenizer()
    monkeypatch.setattr(backend, "_load_transformers", lambda cfg: (model, tok))
    cfg = ExperimentConfig(name="x", bias_type="demographic", model_path="m",
                           embedding_cache_dir=str(tmp_path), extra={"domain": "credit"})
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    assert exp.model is model and exp.tokenizer is tok
    assert getattr(model, ec.CACHE_ATTR).directory.parent == tmp_path


def test_pinned_revisions_and_the_config_override():
    from scoring.backend import PINNED_REVISIONS, model_revision
    from scoring.experiment import ExperimentConfig

    rb2 = "allenai/Llama-3.1-70B-Instruct-RM-RB2"
    assert model_revision(ExperimentConfig(name="x", bias_type="demographic", model_path=rb2)) == PINNED_REVISIONS[rb2]
    own = ExperimentConfig(name="x", bias_type="demographic", model_path=rb2, model_revision="abc")
    assert model_revision(own) == "abc" and own.to_dict()["model_revision"] == "abc"
    other = ExperimentConfig(name="x", bias_type="demographic", model_path="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    assert model_revision(other) is None
    assert all(len(sha) == 40 for sha in PINNED_REVISIONS.values())


def test_the_runner_takes_a_revision_on_the_command_line():
    from runners.run_cross_marker import apply_overrides, build_parser
    from scoring.experiment import ExperimentConfig

    cfg = ExperimentConfig.from_yaml("configs/demographic_credit_crossmarker_qwen06.yaml")
    assert apply_overrides(cfg, build_parser().parse_args(["--revision", "deadbeef"])).model_revision == "deadbeef"


def test_offloading_is_refused(caplog):
    import logging

    import pytest

    from scoring.backend import ModelOffloaded, check_placement

    with caplog.at_level(logging.INFO, logger="scoring.backend"):
        check_placement(SimpleNamespace(hf_device_map={"model.layers.0": 0, "model.layers.1": 1, "score": 1}))
    assert "modules per device" in caplog.text
    with pytest.raises(ModelOffloaded, match="2 module"):
        check_placement(SimpleNamespace(hf_device_map={"model.layers.0": 0, "model.layers.79": "cpu", "score": "cpu"}))
    with pytest.raises(ModelOffloaded):
        check_placement(SimpleNamespace(hf_device_map={"model.layers.0": "cpu", "score": "disk"}))
    check_placement(SimpleNamespace(hf_device_map={"": "cpu"}))    # --device cpu: the whole model, not an offload
    check_placement(SimpleNamespace())             # single-device load: nothing to report


def test_load_model_records_the_loaded_commit(monkeypatch, tmp_path):
    """Unpinned models load from the Hub's moving main: the result must name the commit it ran on."""
    import scoring.backend as backend
    from probes import embedding_cache as ec
    from scoring.demographic_experiment import DemographicBiasExperiment
    from scoring.experiment import ExperimentConfig
    from tests.test_embedding_cache import _model, _tokenizer

    monkeypatch.delenv(ec.ENV_VAR, raising=False)
    for commit in ("c" * 40, None):                 # None: a local directory, nothing to record
        model, tok = _model(), _tokenizer()
        model.config._commit_hash = commit
        monkeypatch.setattr(backend, "_load_transformers", lambda cfg: (model, tok))
        cfg = ExperimentConfig(name="x", bias_type="demographic", model_path="m",
                               embedding_cache_dir=str(tmp_path), extra={"domain": "credit"})
        DemographicBiasExperiment(cfg).load_model()
        assert cfg.model_revision == commit and cfg.to_dict()["model_revision"] == commit
        assert backend.model_revision(cfg) == commit


def test_remote_code_is_off_unless_asked_for():
    """No model of the eleven needs its checkpoint's own code; with it on, a repo that gains an auto_map
    would have its code run in place of the transformers class."""
    from pathlib import Path

    import yaml

    from scoring.experiment import ExperimentConfig

    assert ExperimentConfig(name="x", bias_type="demographic", model_path="m").trust_remote_code is False
    for path in sorted(Path("configs").glob("*.yaml")):
        assert not yaml.safe_load(path.read_text()).get("trust_remote_code"), path


def test_prefetch_skips_bin_weights_next_to_safetensors():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("prefetch", Path("cluster/prefetch_models.py"))
    prefetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prefetch)
    assert prefetch.skip_patterns(["model-00001-of-00029.safetensors", "pytorch_model-00001-of-00029.bin"]) == [
        "*.bin", "*.pth"]
    assert prefetch.skip_patterns(["pytorch_model.bin", "config.json"]) == []      # .bin-only: keep them
