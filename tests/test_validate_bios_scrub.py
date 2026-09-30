"""`runners/validate_bios_scrub.py`: the manifest's pool (refused otherwise), the no-model field rules, the paired
probe − occupation comparison, provenance — end to end on the tiny model and the `hiring` fixture."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
import yaml

from runners import validate_bios_scrub as vbs
from tests.test_run_cross_marker import _model, _tokenizer


class _R:
    def __init__(self, gender, profession, target_role="x"):
        self.gender, self.profession, self.target_role = gender, profession, target_role


def test_field_rule_uses_the_probe_splits_majority_and_falls_back_to_its_overall_majority():
    train = [_R(1, "nurse"), _R(1, "nurse"), _R(0, "nurse"), _R(0, "surgeon"), _R(0, "poet")]
    evalr = [_R(1, "nurse"), _R(0, "nurse"), _R(0, "surgeon"), _R(1, "dj"), _R(0, "dj")]
    assert vbs.field_rule(train, evalr, "profession") == [True, False, True, False, True]   # dj unseen -> 0


def test_field_baselines_share_the_eval_bios():
    train = [_R(k % 2, "nurse" if k % 2 else "surgeon", "a" if k % 3 else "b") for k in range(40)]
    evalr = [_R(k % 2, "nurse" if k % 2 else "surgeon", "a") for k in range(20)]
    b = vbs.field_baselines(train, evalr, n_boot=50, seed=0)
    assert b["occupation_acc"]["estimate"] == 1.0 and b["chance"]["estimate"] == 0.5
    assert b["occupation_above_chance"]["estimate"] == 0.5
    assert b["target_role_above_chance"]["n_clusters"] == 20


def test_the_occupation_is_the_paired_reference():
    from probes.erasure import probe_recoverability

    g = torch.Generator().manual_seed(0)
    y_tr, y_ev = [k % 2 for k in range(60)], [k % 2 for k in range(40)]
    X = lambda y: torch.tensor(y, dtype=torch.float32)[:, None] * 3 + torch.randn(len(y), 4, generator=g)
    ref = [k % 4 != 0 for k in range(40)]                                  # a rule right on 75% of the bios
    r = probe_recoverability(X(y_tr), y_tr, X(y_ev), y_ev, n_boot=50, reference_ok=ref)["intervals"]
    assert r["reference_acc"]["estimate"] == 0.75
    assert r["linear_minus_reference"]["estimate"] == pytest.approx(r["linear_acc"]["estimate"] - 0.75)
    with pytest.raises(ValueError, match="reference flags"):
        probe_recoverability(X(y_tr), y_tr, X(y_ev), y_ev, n_boot=5, reference_ok=ref[:-1])


@pytest.fixture
def run(hiring, tmp_path, monkeypatch):
    pairs, raw = hiring
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    loads = []

    def _run(*extra, domain="cv", raw_path=raw):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(pairs),
            "batch_size": 16, "max_length": 1024, "extra": {"domain": domain, "n_boot": 20}}))

        def load_model(self):
            loads.append(1)
            self.model, self.tokenizer = _model(), _tokenizer()
            self.config.model_revision = "abc123"

        monkeypatch.setattr(vbs.DemographicBiasExperiment, "load_model", load_model)
        monkeypatch.setattr("sys.argv", ["validate_bios_scrub.py", "--config", str(cfg_path), "--raw-path",
                                         str(raw_path), "--probe-items", "40", "--eval-items", "30", *extra])
        vbs.main()
        return json.loads((tmp_path / "artifacts/results/demographic/scrubcheck_bios_Tiny-RM.json").read_text())

    _run.loads = loads
    return _run


def test_end_to_end_on_a_tiny_model(run, hiring):
    pairs, raw = hiring
    result = run()
    meta = result["meta"]
    assert meta["config"]["model_revision"] == "abc123"
    assert meta["data"][raw.name]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert meta["data"]["pairs.jsonl"]["sha256"] == hashlib.sha256(pairs.read_bytes()).hexdigest()
    manifest_ids = {json.loads(line)["source_record_id"] for line in open(pairs.parent / "cells.jsonl")}
    assert set(result["probe_records"]) | set(result["eval_records"]) <= manifest_ids
    assert not set(result["probe_records"]) & set(result["eval_records"])
    assert set(result["results"]) == {"scrubbed", "unscrubbed"}
    iv = result["results"]["scrubbed"]["intervals"]
    assert iv["reference_acc"]["estimate"] == pytest.approx(result["field_baselines"]["occupation_acc"]["estimate"])
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


def test_another_corpus_or_pool_is_refused_before_the_model_loads(run, hiring, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="hiring manifest"):
        run(domain="credit")
    other = tmp_path / "other.parquet"
    other.write_bytes(hiring[1].read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="not the corpus"):
        run(raw_path=other)
    # the same corpus loaded into another pool (e.g. another size) is not the manifest's
    real = vbs.load_factorial_bios
    monkeypatch.setattr(vbs, "load_factorial_bios", lambda *a, **kw: real(*a, **{**kw, "n": 50}))
    with pytest.raises(SystemExit, match="not the manifest's"):
        run()
    assert run.loads == []
