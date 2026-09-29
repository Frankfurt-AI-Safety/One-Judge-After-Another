"""`runners/run_positioned_maineffect.py` (A2): it scores exactly the positioned manifest's texts, counts a tie ½,
names every non-default setting, and runs end to end on the tiny model."""

from __future__ import annotations

import hashlib
import json

import pytest

from runners import run_positioned_maineffect as rpm
from tests.test_education_stage import _asap2_csv
from tests.test_run_cross_marker import _model, _tokenizer

POSITIONS = ("conclusion", "opening", "middle", "random")


def test_the_runner_scores_the_manifests_texts_at_every_position(tmp_path, monkeypatch):
    # Until 2026-09-29 the runner seeded its blocks differently: 93% of the random position's pairs differed.
    from pairs.positionality import positioned_block, select_standpoint_essays
    from runners import generate_positioned
    from substrates.education_clean import load_education_essays

    raw = _asap2_csv(tmp_path / "asap2.csv", n=12)
    monkeypatch.setattr("sys.argv", ["generate_positioned.py", "--raw-path", str(raw), "--out-dir", str(tmp_path / "m"),
                                     "--positions", ",".join(POSITIONS)])
    generate_positioned.main()
    manifest = {r["id"]: (r["text_a"], r["text_b"]) for r in map(json.loads, open(tmp_path / "m/pairs.jsonl"))}
    essays = select_standpoint_essays(load_education_essays(str(raw), source="asap2", seed=42), "plausible", 42)
    from pairs.positionality import block_id_suffix

    ours = {}
    for axis in rpm.POSITIONED_AXES:
        for position in POSITIONS:
            for rec in essays:
                pairs, _ = positioned_block(rec, axis, position, 42)
                for p in pairs:
                    ours[f"pos-{axis}-{position}-{rec.source_record_id}{block_id_suffix(p)}"] = (p.text_a, p.text_b)
    assert ours == manifest and {k.split("-")[2] for k in ours} >= {"random"}


def test_a_tie_counts_half():
    items = [(0.0, 1.0, 1.0), (0.0, 2.0, 1.0)]          # (neutral, a, b): one tie, one a > b
    assert rpm.POSITIONED_STATS["pref_a_rate"](items) == 0.75
    assert rpm.POSITIONED_STATS["auto_influence"](items) == 0.5
    assert rpm.POSITIONED_STATS["identity_gap"](items) == 0.5


def test_default_names_carry_every_non_default_setting():
    kw = dict(positions=["conclusion"], stance="endorse", paraphrase="off", axes=list(rpm.POSITIONED_AXES),
              n_essays=None)
    assert rpm.default_out("asap2", "plausible", "o/RM", **kw).name == "maineffect_edupos_asap2_plausible_RM.json"
    both = rpm.default_out("asap2", "plausible", "o/RM", **{**kw, "stance": "both", "n_essays": 20})
    assert both.name == "maineffect_edupos_asap2_plausible_RM__stance-both__n20.json"


def test_end_to_end_on_a_tiny_model(tmp_path, monkeypatch):
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    raw = _asap2_csv(tmp_path / "asap2.csv", n=8)
    monkeypatch.chdir(tmp_path)
    loads = []

    def load_model(self):
        loads.append(1)
        self.model, self.tokenizer = _model(), _tokenizer()
        self.config.model_revision = "abc123"

    monkeypatch.setattr(rpm.DemographicBiasExperiment, "load_model", load_model)
    from pathlib import Path
    cfg = Path(__file__).resolve().parents[1] / "configs/demographic_edupos_qwen06.yaml"
    monkeypatch.setattr("sys.argv", ["run_positioned_maineffect.py", "--config", str(cfg), "--raw-path", str(raw),
                                     "--model", "org/Tiny-RM", "--batch-size", "16"])
    monkeypatch.setattr(rpm, "DEFAULT_N_BOOT", 50)
    rpm.main()
    result = json.loads((tmp_path / "artifacts/results/demographic/"
                                    "maineffect_edupos_asap2_plausible_Tiny-RM.json").read_text())
    assert result["meta"]["config"]["model_revision"] == "abc123"
    assert result["meta"]["data"]["asap2.csv"]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert result["n_essays"] == 8 and len(result["essays"]) == 8
    assert [r["axis"] for r in result["results"]] == list(rpm.POSITIONED_AXES)
    sex = result["results"][0]
    assert sex["n_pairs"] == 4 * 8 and "probe_accuracy" not in sex
    assert sex["identity_gap"] == sex["intervals"]["identity_gap"]["estimate"]
    assert sex["main_effect"] == pytest.approx((sex["delta_a"] + sex["delta_b"]) / 2)
    with pytest.raises(SystemExit, match="exists"):
        rpm.main()
    assert len(loads) == 1
