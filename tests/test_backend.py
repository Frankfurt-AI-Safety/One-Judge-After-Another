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
