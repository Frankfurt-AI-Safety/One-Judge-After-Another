"""`runners/run_reasoning_flip.py`, the reasoning 2×2 (hiring, credit, education): its records (the manifest's pool
minus both directions' probe records and the ineligible ones), the refusal of an unknown domain and of another corpus,
the intervals and the premise − control comparison, provenance, and the gate columns of a gated head — end to end on
the tiny models and manifests the generators build from synthetic corpora."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random

import pytest
import yaml

from runners import run_reasoning_flip as rrf
from scoring.demographic_experiment import compute_reasoning_metrics, reasoning_contrast_intervals, reasoning_intervals
from tests.conftest import reasoning_domain
from tests.test_run_cross_marker import _model, _tokenizer

# the fixture that builds each domain's manifest and corpus (tests/conftest.py)
CORPUS = {"cv": "hiring", "credit": "credit_corpus", "education": "education_corpus"}


def test_premises_and_result_names():
    assert rrf.premise_axes("cv") == {"parental_leave": "family_status", "intersection": "intersection",
                                      "abroad": None}
    assert rrf.premise_axes("credit") == {"age": "age", "intersection": "intersection", "sabbatical": None}
    assert rrf.premise_axes("education") == {"low_income": "economic_status", "intersection": "intersection",
                                             "out_of_district": None}
    assert rrf.default_out("credit", "org/M").name == "reasoning_credit_M.json"
    assert rrf.default_out("credit", "org/M", "__seed-7").name == "reasoning_credit_M__seed-7.json"


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


def test_the_placebo_contrast_is_the_difference_of_the_two_nulling_changes():
    from scoring.demographic_experiment import reasoning_nulling_contrast_intervals

    rng = random.Random(1)
    cells = ("true_reject", "true_advance", "false_advance", "false_reject")
    n = 30
    k = [rng.gauss(0, 1) for _ in range(n)]                     # each record's own nulling change
    shift = lambda d, ks: {c: [x + kk * c.startswith("true") for x, kk in zip(d[c], ks)] for c in cells}
    prem = {c: [rng.gauss(0, 1) for _ in range(n)] for c in cells}
    ctrl = {c: [rng.gauss(0, 1) for _ in range(n)] for c in cells}
    # nulling moves premise and control of a record alike: no change beyond the control's, on every resample
    did = reasoning_nulling_contrast_intervals(prem, shift(prem, k), ctrl, shift(ctrl, k), n_boot=200)
    c = did["correctness_effect"]
    assert c["estimate"] == pytest.approx(0.0, abs=1e-12) and c["ci_low"] == pytest.approx(c["ci_high"], abs=1e-12)
    # the same changes on the control's records rotated by one: the point estimate is the same, the pairing is lost
    rot = k[1:] + k[:1]
    did = reasoning_nulling_contrast_intervals(prem, shift(prem, k), ctrl, shift(ctrl, rot), n_boot=200)
    c = did["correctness_effect"]
    assert c["estimate"] == pytest.approx(0.0, abs=1e-12) and c["ci_high"] - c["ci_low"] > 0.1
    # constant changes: the premise's correctness effect falls by 0.5, the control's by 0.2
    did = reasoning_nulling_contrast_intervals(prem, shift(prem, [-0.5] * n), ctrl, shift(ctrl, [-0.2] * n),
                                               n_boot=50)
    assert did["correctness_effect"]["estimate"] == pytest.approx(-0.3)
    assert did["conclusion_effect"]["estimate"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="same records"):
        reasoning_nulling_contrast_intervals(prem, prem, {c: v[:-1] for c, v in ctrl.items()}, ctrl, n_boot=50)


@pytest.fixture
def run(tmp_path, monkeypatch, request):
    """`main` on a domain's manifest (its corpus fixture, built on first use); ``model_fn``/``tok_fn`` stand in for
    the loader, ``corpus`` for the corpus file the records are read from."""
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    loads = []

    def _run(*extra, model_fn=_model, tok_fn=_tokenizer, domain="cv", config_domain=None, corpus=None, drop=0,
             name="__n_items-10"):
        pairs, raw = request.getfixturevalue(CORPUS[domain])
        monkeypatch.chdir(tmp_path)

        def spec(name):                       # ``drop``: a pool that lacks the manifest's first records
            dom = reasoning_domain(name, corpus or raw)
            return dataclasses.replace(dom, load_records=lambda: dom.load_records()[drop:])

        monkeypatch.setattr(rrf, "get_domain", spec)
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(pairs),
            "probe_records": 8, "batch_size": 16, "max_length": 1024,
            "extra": {"domain": config_domain or domain, "n_boot": 50}}))

        def load_model(self):
            loads.append(1)
            self.model, self.tokenizer = model_fn(), tok_fn()
            self.config.model_revision = "abc123"

        monkeypatch.setattr(rrf.DemographicBiasExperiment, "load_model", load_model)
        monkeypatch.setattr("sys.argv", ["run_reasoning_flip.py", "--config", str(cfg_path), "--n-items", "10",
                                         *extra])
        rrf.main()
        return json.loads((tmp_path / f"artifacts/results/demographic/reasoning_{domain}_Tiny-RM{name}.json")
                          .read_text())

    _run.loads = loads
    return _run


def test_end_to_end_on_a_tiny_model(run, hiring):
    pairs, raw = hiring
    result = run()
    meta = result["meta"]
    assert meta["config"]["model_revision"] == "abc123"
    assert meta["data"]["pairs.jsonl"]["sha256"] == hashlib.sha256(pairs.read_bytes()).hexdigest()
    assert meta["data"][raw.name]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert meta["data"]["cells.jsonl"]["sha256"] == \
        hashlib.sha256((pairs.parent / "cells.jsonl").read_bytes()).hexdigest()
    # the items are strong records outside both directions' probe records
    dom = reasoning_domain("cv", raw)
    probe = set().union(*(dom.dataset_cls(str(pairs), axis=a, encoding="explicit", probe_records=8,
                                          split_seed=42).probe_record_ids() for a in ("family_status", "intersection")))
    assert result["selection"]["excluded_probe_records"] > 0 and result["selection"]["ineligible"] == 0
    assert result["selection"]["outside_manifest"] == 0
    assert len(result["records"]) == 10 and not set(result["records"]) & probe
    by = {r["premise"]: r for r in result["results"]}
    assert list(by) == ["parental_leave", "intersection", "abroad"]
    assert "nulled" not in by["abroad"] and "nulled_minus_baseline" not in by["abroad"]["intervals"]
    pl = by["parental_leave"]
    assert pl["null_axis"] == "family_status" and pl["n_items"] == 10
    for k in ("correctness_effect", "prefers_correct_over_favorable_rate"):
        assert pl["intervals"]["baseline"][k]["estimate"] == pytest.approx(pl["baseline"][k])
        assert pl["intervals"]["nulled"][k]["estimate"] == pytest.approx(pl["nulled"][k])
    vs = result["versus_control"]["parental_leave"]["correctness_effect"]["estimate"]
    assert vs == pytest.approx(pl["baseline"]["correctness_effect"] - by["abroad"]["baseline"]["correctness_effect"])
    assert set(result["versus_control"]) == {"parental_leave", "intersection"}
    assert meta["settings"]["control"] == "abroad"
    # the placebo: the control nulled with every demographic premise's direction, never with its own
    ctl = by["abroad"]
    assert set(ctl["placebo"]) == {"family_status", "intersection"} and "nulled" not in ctl
    assert set(result["nulling_vs_control"]) == {"parental_leave", "intersection"}
    pl_change = pl["nulled"]["correctness_effect"] - pl["baseline"]["correctness_effect"]
    ctl_change = ctl["placebo"]["family_status"]["nulled"]["correctness_effect"] - ctl["baseline"]["correctness_effect"]
    assert result["nulling_vs_control"]["parental_leave"]["correctness_effect"]["estimate"] == \
        pytest.approx(pl_change - ctl_change)
    assert "gate_fixed_nulling_vs_control" not in result
    assert all("gate_fixed" not in r for r in result["results"])          # a linear head has no gate
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


@pytest.mark.parametrize("domain,primary,axis,control", [("credit", "age", "age", "sabbatical"),
                                                         ("education", "low_income", "economic_status",
                                                          "out_of_district")])
def test_credit_and_education_end_to_end(run, request, domain, primary, axis, control):
    pairs, raw = request.getfixturevalue(CORPUS[domain])
    result = run(domain=domain)
    assert result["domain"] == domain
    assert result["meta"]["data"][raw.name]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    by = {r["premise"]: r for r in result["results"]}
    assert list(by) == [primary, "intersection", control]
    assert by[primary]["null_axis"] == axis and by["intersection"]["null_axis"] == "intersection"
    assert "nulled" in by[primary] and "nulled" not in by[control]
    assert set(result["versus_control"]) == {primary, "intersection"}
    dom = reasoning_domain(domain, raw)
    probe = set().union(*(dom.dataset_cls(str(pairs), axis=a, encoding="explicit", probe_records=8,
                                          split_seed=42).probe_record_ids() for a in (axis, "intersection")))
    # the probe records are records of the pool, so excluding them removed some
    assert probe <= {str(r.source_record_id) for r in dom.load_records()}
    assert result["selection"]["excluded_probe_records"] > 0
    assert result["records"] and not set(result["records"]) & probe
    if domain == "credit":
        # the 12 good-credit records whose employment or job reads unemployed (tests/conftest.py) contradict the
        # sabbatical: never drawn
        frame = rrf.REASONING_FRAMES["credit"]
        ineligible = {str(r.source_record_id) for r in dom.load_records() if r.credit_good and not frame.is_eligible(r)}
        assert len(ineligible) == 12 and result["selection"]["ineligible"] == 12
        assert not ineligible & set(result["records"])
    else:
        assert result["selection"]["ineligible"] == 0


def test_a_gated_head_also_reports_the_gate_fixed_metrics(run):
    from tests.test_qrm import _model as qrm_model, _tokenizer as qrm_tokenizer

    result = run(model_fn=qrm_model, tok_fn=qrm_tokenizer)
    by = {r["premise"]: r for r in result["results"]}
    pl = by["parental_leave"]
    assert {"gate_fixed", "nulled_gate_fixed", "gate_fixed_intervals"} <= set(pl)
    assert "gate_fixed" in by["abroad"] and "nulled_gate_fixed" not in by["abroad"]
    assert {"nulled", "intervals", "nulled_gate_fixed", "gate_fixed_intervals"} <= \
        set(by["abroad"]["placebo"]["family_status"])
    assert set(result["gate_fixed_nulling_vs_control"]) == {"parental_leave", "intersection"}
    # the contrast, on the gated tiny model's non-zero effects: each premise against the control nulled with the
    # premise's own direction, free and fixed gate
    ctl = by["abroad"]
    for premise, axis in (("parental_leave", "family_status"), ("intersection", "intersection")):
        for key, nulled, base in (("nulling_vs_control", "nulled", "baseline"),
                                  ("gate_fixed_nulling_vs_control", "nulled_gate_fixed", "gate_fixed")):
            eff = lambda m: m["correctness_effect"]
            expect = (eff(by[premise][nulled]) - eff(by[premise][base])) \
                - (eff(ctl["placebo"][axis][nulled]) - eff(ctl[base]))
            assert result[key][premise]["correctness_effect"]["estimate"] == pytest.approx(expect)
    assert result["nulling_vs_control"]["parental_leave"]["correctness_effect"]["estimate"] != pytest.approx(0.0)
    # the gate of the prompt without the premise differs, so the rescored rewards move
    assert pl["gate_fixed"]["mean_true_reject"] != pytest.approx(pl["baseline"]["mean_true_reject"], abs=1e-9)


def test_an_unknown_domain_and_another_corpus_are_refused_before_the_model_loads(run, hiring, tmp_path):
    with pytest.raises(SystemExit, match="runs on"):
        run(config_domain="nope")
    other = tmp_path / "other.parquet"
    other.write_bytes(hiring[1].read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="not the corpus"):
        run(corpus=other)
    # a pool that lacks a manifest record (another build of the corpus pool) is not the manifest's
    with pytest.raises(SystemExit, match="not the manifest's pool"):
        run(drop=1)
    assert run.loads == []


def test_the_records_are_the_manifests_pool(credit_corpus):
    # reasoning_records draws only records of the manifest's cells.jsonl, with the manifest's labels
    pairs, raw = credit_corpus
    dom, frame = reasoning_domain("credit", raw), rrf.REASONING_FRAMES["credit"]
    records = dom.load_records()
    pool, data, left_out = rrf.reasoning_records(dom, frame, pairs)
    assert set(data) == {"cells.jsonl", raw.name} and left_out == {"outside_manifest": 0, "ineligible": 12}
    # a corpus record the manifest left out is not drawn
    extra = dataclasses.replace(next(r for r in records if r.credit_good), source_record_id="german-9999")
    more = dataclasses.replace(dom, load_records=lambda: records + [extra])
    pool, _, left_out = rrf.reasoning_records(more, frame, pairs)
    assert left_out["outside_manifest"] == 1 and "german-9999" not in {r.source_record_id for r in pool}
    # a record labelled otherwise than in the manifest, and probe records that are not the manifest's, are refused
    first = records[0].source_record_id
    relabelled = dataclasses.replace(dom, is_strong=lambda r: (not r.credit_good) if r.source_record_id == first
                                     else r.credit_good)
    with pytest.raises(SystemExit, match="1 with another quality label"):
        rrf.reasoning_records(relabelled, frame, pairs)
    with pytest.raises(SystemExit, match="probe records are not records of"):
        rrf.reasoning_records(dom, frame, pairs, probe_ids={"credit-0001"})


def test_a_setting_changed_on_the_cli_names_the_result(run, tmp_path):
    run("--seed", "7", name="__n_items-10__seed-7")
    assert (tmp_path / "artifacts/results/demographic/reasoning_cv_Tiny-RM__n_items-10__seed-7.json").exists()
