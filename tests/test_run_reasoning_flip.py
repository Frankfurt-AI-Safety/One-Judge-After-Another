"""`runners/run_reasoning_flip.py`, the reasoning 2×2 (hiring only): its records (the manifest's pool minus both
directions' probe records), the refusal of other domains and of another corpus, the intervals and the premise −
control comparison, provenance, and the gate columns of a gated head — end to end on the tiny models and a hiring
manifest built by `generate_bios` from a synthetic parquet."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random

import pytest
import yaml

from runners import run_reasoning_flip as rrf
from scoring.demographic_experiment import compute_reasoning_metrics, reasoning_contrast_intervals, reasoning_intervals
from substrates.domains import get_domain
from tests.test_run_cross_marker import _model, _tokenizer


def test_the_reasoning_prompt_is_the_hiring_decision_prompt():
    # gate_fixed rescoring takes the gate of `unmarked_decision_prompt`, which must be the reasoning item's prompt
    # without its premise
    from pairs.verdicts import DECISION_FRAMES, DECISION_PROMPT

    assert DECISION_FRAMES["cv"].prompt == DECISION_PROMPT


def test_intervals_share_the_point_estimates_and_pair_by_record():
    rng = random.Random(0)
    cells = ("true_reject", "true_advance", "false_advance", "false_reject")
    prem = {c: [rng.gauss(k, 1) for _ in range(30)] for k, c in enumerate(cells)}
    ctrl = {c: [x + 0.5 * (c.startswith("true")) for x in prem[c]] for c in cells}   # correctness +0.5
    ci = reasoning_intervals(prem, ctrl, n_boot=50)
    m = compute_reasoning_metrics(prem)
    assert all(ci["baseline"][k]["estimate"] == pytest.approx(m[k]) for k in m if k != "n")
    assert ci["nulled_minus_baseline"]["correctness_effect"]["estimate"] == pytest.approx(0.5)
    diff = reasoning_contrast_intervals(prem, ctrl, n_boot=50)
    # the same records shifted by a constant: the paired difference has no spread
    assert diff["correctness_effect"]["estimate"] == pytest.approx(-0.5)
    assert diff["correctness_effect"]["ci_low"] == pytest.approx(diff["correctness_effect"]["ci_high"])
    assert "nulled" not in reasoning_intervals(prem, None, n_boot=50)
    with pytest.raises(ValueError, match="same records"):
        reasoning_contrast_intervals(prem, {c: v[:-1] for c, v in ctrl.items()}, n_boot=50)


@pytest.fixture
def hiring(tmp_path, monkeypatch):
    """A hiring manifest from `generate_bios` on a synthetic parquet: (pairs.jsonl, parquet)."""
    from tests.test_bios_pipeline import TestGenerateBiosCLI

    raw, out = TestGenerateBiosCLI()._main(tmp_path, monkeypatch)
    return out / "pairs.jsonl", raw


@pytest.fixture
def run(hiring, tmp_path, monkeypatch):
    """`main` on the hiring manifest; ``model_fn``/``tok_fn`` stand in for the loader."""
    from substrates.bios_clean import DEFAULT_N_BIOS, load_factorial_bios

    pairs, raw = hiring
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rrf, "DEFAULT_BIOS_PATH", str(raw))
    monkeypatch.setattr(rrf, "get_domain", lambda name: dataclasses.replace(
        get_domain(name), load_records=lambda: load_factorial_bios(str(raw), n=DEFAULT_N_BIOS)))
    loads = []

    def _run(*extra, model_fn=_model, tok_fn=_tokenizer, domain="cv"):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(pairs),
            "probe_records": 8, "batch_size": 16, "max_length": 1024,
            "extra": {"domain": domain, "n_boot": 50}}))

        def load_model(self):
            loads.append(1)
            self.model, self.tokenizer = model_fn(), tok_fn()
            self.config.model_revision = "abc123"

        monkeypatch.setattr(rrf.DemographicBiasExperiment, "load_model", load_model)
        monkeypatch.setattr("sys.argv", ["run_reasoning_flip.py", "--config", str(cfg_path), "--n-items", "10",
                                         *extra])
        rrf.main()
        return json.loads((tmp_path / "artifacts/results/demographic/reasoning_cv_Tiny-RM.json").read_text())

    _run.loads = loads
    return _run


def test_end_to_end_on_a_tiny_model(run, hiring):
    pairs, raw = hiring
    result = run()
    meta = result["meta"]
    assert meta["config"]["model_revision"] == "abc123"
    assert meta["data"]["pairs.jsonl"]["sha256"] == hashlib.sha256(pairs.read_bytes()).hexdigest()
    assert meta["data"][raw.name]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    # the items are strong records outside both directions' probe records
    dom = get_domain("cv")
    probe = set().union(*(dom.dataset_cls(str(pairs), axis=a, encoding="explicit", probe_records=8,
                                          split_seed=42).probe_record_ids() for a in ("family_status", "intersection")))
    assert result["selection"]["excluded_probe_records"] > 0
    assert len(result["records"]) == 10 and not set(result["records"]) & probe
    by = {r["premise"]: r for r in result["results"]}
    assert list(by) == list(rrf.PREMISES)
    assert "nulled" not in by["commute"] and "nulled_minus_baseline" not in by["commute"]["intervals"]
    pl = by["parental_leave"]
    assert pl["null_axis"] == "family_status" and pl["n_items"] == 10
    for k in ("correctness_effect", "prefers_correct_over_favorable_rate"):
        assert pl["intervals"]["baseline"][k]["estimate"] == pytest.approx(pl["baseline"][k])
        assert pl["intervals"]["nulled"][k]["estimate"] == pytest.approx(pl["nulled"][k])
    vs = result["versus_control"]["parental_leave"]["correctness_effect"]["estimate"]
    assert vs == pytest.approx(pl["baseline"]["correctness_effect"] - by["commute"]["baseline"]["correctness_effect"])
    assert set(result["versus_control"]) == {"parental_leave", "intersection"}
    assert all("gate_fixed" not in r for r in result["results"])          # a linear head has no gate
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


def test_a_gated_head_also_reports_the_gate_fixed_metrics(run):
    from tests.test_qrm import _model as qrm_model, _tokenizer as qrm_tokenizer

    by = {r["premise"]: r for r in run(model_fn=qrm_model, tok_fn=qrm_tokenizer)["results"]}
    pl = by["parental_leave"]
    assert {"gate_fixed", "nulled_gate_fixed", "gate_fixed_intervals"} <= set(pl)
    assert "gate_fixed" in by["commute"] and "nulled_gate_fixed" not in by["commute"]
    # the gate of the prompt without the premise differs, so the rescored rewards move
    assert pl["gate_fixed"]["mean_true_reject"] != pytest.approx(pl["baseline"]["mean_true_reject"], abs=1e-9)


def test_other_domains_and_another_corpus_are_refused_before_the_model_loads(run, hiring, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="hiring-only"):
        run(domain="credit")
    other = tmp_path / "other.parquet"
    other.write_bytes(hiring[1].read_bytes() + b"\0")
    monkeypatch.setattr(rrf, "DEFAULT_BIOS_PATH", str(other))
    with pytest.raises(SystemExit, match="not the corpus"):
        run()
    assert run.loads == []
