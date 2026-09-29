"""`runners/run_battery.py`, the direct arm: cells from the manifest, result names, provenance, the α-sweep, end
to end on the tiny Llama RM and the credit manifest of `tests/test_run_cross_marker.py`."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from runners import run_battery
from runners.run_battery import default_out, manifest_cells, select_cells
from tests.test_run_cross_marker import _model, _tokenizer

CREDIT_CELLS = [("sex", "explicit"), ("age", "explicit"), ("marital_status", "explicit"), ("intersection", "explicit"),
                ("sex", "proxy"), ("age", "proxy"), ("intersection", "proxy")]


def test_cells_come_from_the_manifest(manifest):
    assert sorted(manifest_cells(manifest)) == sorted(CREDIT_CELLS)  # marital status has no proxy


def test_select_cells_filters_and_rejects_unknown_names():
    assert select_cells(CREDIT_CELLS) == CREDIT_CELLS
    assert select_cells(CREDIT_CELLS, axes=["marital_status"], encodings=["explicit", "proxy"]) == \
        [("marital_status", "explicit")]
    for kw in ({"axes": ["grade_level"]}, {"encodings": ["conclusion"]}):
        with pytest.raises(SystemExit, match="not in the manifest"):
            select_cells(CREDIT_CELLS, **kw)


def test_default_names_separate_models_manifests_and_partial_runs():
    full = default_out("credit", "data/demographic/credit/pairs.jsonl", "Skywork/RM-0.6B", CREDIT_CELLS, CREDIT_CELLS)
    assert full.name == "battery_credit_RM-0.6B.json"
    assert default_out("credit", "data/demographic/credit/pairs.jsonl", "Skywork/RM-8B", CREDIT_CELLS,
                       CREDIT_CELLS).name == "battery_credit_RM-8B.json"
    stage = [("grade_level", "explicit"), ("grade_level", "proxy")]
    assert default_out("education", "data/demographic/education/asap2_stage/pairs.jsonl", "o/RM", stage,
                       stage).name == "battery_education_asap2_stage_RM.json"
    part = default_out("credit", "data/demographic/credit/pairs.jsonl", "o/RM", CREDIT_CELLS[:1], CREDIT_CELLS)
    assert part.name == "battery_credit_RM__sex__explicit.json"


@pytest.fixture
def run(manifest, tmp_path, monkeypatch):
    """Call `main` with a config on the fixture manifest; the tiny model stands in for the loader."""
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(manifest),
        "probe_records": 6, "max_test_examples": 20, "batch_size": 16, "max_length": 1024,
        "extra": {"domain": "credit", "n_boot": 50}}))
    loads = []

    def load_model(self):
        loads.append(1)
        self.model, self.tokenizer = _model(), _tokenizer()
        self.config.model_revision = "abc123"  # what the real loader records

    monkeypatch.setattr(run_battery.DemographicBiasExperiment, "load_model", load_model)

    def _run(*extra):
        monkeypatch.setattr("sys.argv", ["run_battery.py", "--config", str(cfg_path), *extra])
        run_battery.main()

    _run.loads = loads
    return _run


def test_end_to_end_on_a_tiny_model(run, manifest, tmp_path):
    run()
    out = tmp_path / "artifacts/results/demographic" / f"battery_credit_{manifest.parent.name}_Tiny-RM.json"
    result = json.loads(out.read_text())
    assert [(c["axis"], c["encoding"]) for c in result["cells"]] == manifest_cells(manifest)
    meta = result["meta"]
    assert meta["config"]["model_revision"] == "abc123" and meta["config"]["model_path"] == "org/Tiny-RM"
    assert meta["data"]["pairs.jsonl"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert meta["settings"]["cells"] == [list(c) for c in manifest_cells(manifest)]
    sex = next(c for c in result["cells"] if (c["axis"], c["encoding"]) == ("sex", "explicit"))
    assert set(sex["alpha_sweep"]) == {"0.0", "0.25", "0.5", "0.75", "1.0"}
    assert all(set(v) == set(run_battery.SWEEP_METRICS) for v in sex["alpha_sweep"].values())
    assert sex["alpha_sweep"]["1.0"]["mean_gap"] == pytest.approx(sex["nulled"]["mean_gap"])
    assert sex["alpha_sweep"]["0.0"]["mean_gap"] == pytest.approx(sex["baseline"]["mean_gap"])
    assert "alpha_sweep" not in next(c for c in result["cells"] if c["axis"] == "age")

    # an existing result is not replaced without --overwrite, and the model is not loaded for nothing
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


def test_bad_inputs_stop_the_run_before_the_model_loads(run, manifest):
    with pytest.raises(SystemExit, match="not in the manifest"):
        run("--axes", "grade_level")
    with open(manifest, "a") as f:  # pairs.jsonl no longer the file its manifest describes
        f.write("{}\n")
    with pytest.raises(ValueError, match="different builds"):
        run()
    assert run.loads == []
