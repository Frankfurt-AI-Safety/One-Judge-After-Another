"""`runners/run_reasoning_probe.py`: directions fitted on one split of applicants and paraphrases, measured on
another split in wording they never saw; the paired accuracy (a tie ½, records resampled whole); the refusals;
provenance — end to end on the tiny model and the generators' manifests of every domain."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
import yaml

from runners import run_reasoning_probe as rrp
from substrates.domains import get_domain
from tests.conftest import reasoning_domain
from tests.test_run_cross_marker import _model, _tokenizer
from tests.test_run_reasoning_flip import CORPUS


def test_the_eval_wording_is_unseen_and_items_do_not_depend_on_their_neighbours():
    from tests.test_decision_response import _rec

    dom = get_domain("cv")
    recs = [_rec(f"r{i}") for i in range(12)]
    probe = rrp.reasoning_items(dom, "parental_leave", recs, rrp.FIT_PARAPHRASES, 42)
    evals = rrp.reasoning_items(dom, "parental_leave", recs, rrp.EVAL_PARAPHRASES, 42)
    fitted = {v for it in probe for v in it["cells"].values()}
    assert not fitted & {v for it in evals for v in it["cells"].values()}
    # an item's draws depend on its record, not on its position in the split
    assert rrp.reasoning_items(dom, "parental_leave", recs[5:6], rrp.FIT_PARAPHRASES, 42)[0]["cells"] == \
        probe[5]["cells"]


class _States(rrp.SplitStates):
    def __init__(self, cells):                     # cells: {cell: [n, d] states}
        self.n = len(next(iter(cells.values())))
        self.hidden = torch.cat([cells[c] for c in rrp.REASONING_CELLS])


def test_paired_accuracy_counts_a_tie_half_and_resamples_records_whole():
    zero, one = torch.zeros(2, 2), torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    # correctness pairs: (true_reject, false_reject) wins for both records, (true_advance, false_advance) ties
    states = _States({"true_reject": one, "false_reject": zero, "true_advance": zero, "false_advance": zero})
    acc = rrp.paired_accuracy(states, "correctness", torch.tensor([1.0, 0.0]), n_boot=50, seed=0)
    assert acc["estimate"] == 0.75 and acc["n_clusters"] == 2 and acc["n_items"] == 4
    assert acc["ci_low"] == acc["ci_high"] == 0.75                      # every record scores 0.75
    u = rrp.fit_direction(states, "correctness")
    assert torch.allclose(u, torch.tensor([1.0, 0.0]))


@pytest.fixture
def run(tmp_path, monkeypatch, request):
    """`main` on a domain's manifest (its corpus fixture, built on first use), the tiny model standing in for the
    loader."""
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    loads = []

    def _run(*extra, domain="cv", config_domain=None):
        pairs, raw = request.getfixturevalue(CORPUS[domain])
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(rrp, "get_domain", lambda name: reasoning_domain(name, raw))
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(pairs),
            "batch_size": 16, "max_length": 1024, "extra": {"domain": config_domain or domain, "n_boot": 50}}))

        def load_model(self):
            loads.append(1)
            self.model, self.tokenizer = _model(), _tokenizer()
            self.config.model_revision = "abc123"

        monkeypatch.setattr(rrp.DemographicBiasExperiment, "load_model", load_model)
        monkeypatch.setattr("sys.argv", ["run_reasoning_probe.py", "--config", str(cfg_path), "--probe-items", "6",
                                         "--eval-items", "5", *extra])
        rrp.main()
        return json.loads((tmp_path / f"artifacts/results/demographic/reasoning_probe_{domain}_Tiny-RM"
                                      "__probe_items-6__eval_items-5.json").read_text())

    _run.loads = loads
    return _run


def test_end_to_end_on_a_tiny_model(run, hiring):
    result = run()
    raw = hiring[1]
    assert result["meta"]["config"]["model_revision"] == "abc123"
    assert result["meta"]["data"][raw.name]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert result["meta"]["settings"]["eval_paraphrases"] == [2]
    assert len(result["probe_records"]) == 6 and len(result["eval_records"]) == 5
    assert not set(result["probe_records"]) & set(result["eval_records"])
    assert [r["premise"] for r in result["results"]] == ["parental_leave", "abroad"]
    assert result["meta"]["data"]["cells.jsonl"]["sha256"] == \
        hashlib.sha256((hiring[0].parent / "cells.jsonl").read_bytes()).hexdigest()
    assert [r["demographic"] for r in result["results"]] == [True, False]
    pl = result["results"][0]
    corr = pl["directions"]["correctness"]
    assert corr["paired_acc_heldout"]["n_clusters"] == 5
    assert corr["intervals"]["nulled"]["correctness_effect"]["estimate"] == \
        pytest.approx(corr["nulled"]["correctness_effect"])
    assert corr["intervals"]["baseline"]["correctness_effect"]["estimate"] == \
        pytest.approx(pl["baseline"]["correctness_effect"])
    assert set(result["transfer"]) == {"abroad_on_parental_leave", "parental_leave_on_abroad", "cosines"}
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


@pytest.mark.parametrize("domain,primary,control", [("credit", "age", "sabbatical"),
                                                    ("education", "low_income", "out_of_district")])
def test_credit_and_education_end_to_end(run, request, domain, primary, control):
    raw = request.getfixturevalue(CORPUS[domain])[1]
    result = run(domain=domain)
    assert result["domain"] == domain and result["meta"]["settings"]["premises"] == [primary, control]
    assert result["meta"]["data"][raw.name]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert [r["premise"] for r in result["results"]] == [primary, control]
    assert set(result["transfer"]) == {f"{control}_on_{primary}", f"{primary}_on_{control}", "cosines"}
    assert result["selection"]["ineligible"] == (12 if domain == "credit" else 0)
    frame = rrp.REASONING_FRAMES[domain]
    ineligible = {str(r.source_record_id) for r in reasoning_domain(domain, raw).load_records()
                  if not frame.is_eligible(r)}
    assert not ineligible & set(result["probe_records"] + result["eval_records"])


def test_bad_inputs_are_refused_before_the_model_loads(run):
    with pytest.raises(SystemExit, match="runs on"):
        run(config_domain="nope")
    with pytest.raises(SystemExit, match="fewer than"):
        run("--eval-items", "100000")
    assert run.loads == []
