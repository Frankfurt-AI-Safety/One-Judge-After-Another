"""`runners/run_reasoning_probe.py`: directions fitted on one split of applicants and two paraphrase entries, measured
on another split in the third entry, cross-fitted over the three; the valence concept and the favourable-truth
premise that separates it from correctness; the paired accuracy (a tie ½, records resampled with all their folds); the
refusals; provenance — end to end on the tiny model and the generators' manifests of every domain."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
import yaml

from runners import run_reasoning_probe as rrp
from pairs.verdicts import PARAPHRASE_FOLDS
from substrates.domains import get_domain
from tests.conftest import reasoning_domain
from tests.test_run_cross_marker import _model, _tokenizer
from tests.test_run_reasoning_flip import CORPUS


def test_the_held_out_wording_is_unseen_in_every_fold_and_items_do_not_depend_on_their_neighbours():
    from tests.test_decision_response import _rec

    dom = get_domain("cv")
    recs = [_rec(f"r{i}") for i in range(12)]
    for fit, held in PARAPHRASE_FOLDS:
        probe = rrp.reasoning_items(dom, "parental_leave", recs, fit, 42)
        evals = rrp.reasoning_items(dom, "parental_leave", recs, held, 42)
        fitted = {v for it in probe for v in it["cells"].values()}
        assert not fitted & {v for it in evals for v in it["cells"].values()}
        assert all(". " in v and " so " not in v and " but " not in v for it in evals for v in it["cells"].values())
        # an item's draws depend on its record, not on its position in the split
        assert rrp.reasoning_items(dom, "parental_leave", recs[5:6], fit, 42)[0]["cells"] == probe[5]["cells"]


class _States(rrp.SplitStates):
    def __init__(self, cells, favourable_true=False, ids=None):          # cells: {cell: [n, d] states}
        self.n = len(next(iter(cells.values())))
        self.hidden = torch.cat([cells[c] for c in rrp.REASONING_CELLS])
        self.favourable_true = favourable_true
        self.items = [{"meta": {"record_id": i}} for i in (ids or [f"r{k}" for k in range(self.n)])]


def test_paired_accuracy_pools_the_folds_counts_a_tie_half_and_resamples_records_whole():
    zero, one = torch.zeros(2, 2), torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    u = torch.tensor([1.0, 0.0])
    # fold 0: both correctness pairs won by both records; fold 1: the reject pair won, the advance pair tied
    f0 = _States({"true_reject": one, "false_reject": zero, "true_advance": one, "false_advance": zero})
    f1 = _States({"true_reject": one, "false_reject": zero, "true_advance": zero, "false_advance": zero})
    acc = rrp.paired_accuracy([(f0, u), (f1, u)], "correctness", n_boot=50, seed=0)
    assert acc["estimate"] == 0.875 and acc["n_clusters"] == 2 and acc["n_items"] == 8
    assert acc["ci_low"] == acc["ci_high"] == 0.875                    # every record scores the same
    assert acc["by_fold"] == [1.0, 0.75]
    assert acc["by_pair"]["reject"]["estimate"] == 1.0 and acc["by_pair"]["advance"]["estimate"] == 0.75
    # records that differ: the interval is over records, each with both folds
    split = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    g = _States({"true_reject": split, "false_reject": zero, "true_advance": split, "false_advance": zero})
    acc = rrp.paired_accuracy([(g, u), (g, u)], "correctness", n_boot=200, seed=0)
    assert acc["estimate"] == 0.75 and acc["ci_low"] < acc["ci_high"]
    assert torch.allclose(rrp.fit_direction([f0], "correctness"), u)
    with pytest.raises(ValueError, match="same records"):
        rrp.paired_accuracy([(f0, u), (_States({c: one[:1] for c in rrp.REASONING_CELLS}), u)], "correctness",
                            n_boot=10, seed=0)
    # the same number of records in another order is refused too
    swapped = _States({"true_reject": one, "false_reject": zero, "true_advance": one, "false_advance": zero},
                      ids=["r1", "r0"])
    with pytest.raises(ValueError, match="same records in the same order"):
        rrp.paired_accuracy([(f0, u), (swapped, u)], "correctness", n_boot=10, seed=0)


def test_a_direction_of_valence_reads_below_half_on_the_favourable_truth_premise():
    # every state is the claim's valence (+1 favourable, −1 unfavourable) on one axis. On an unfavourable-truth
    # premise the "correctness" direction fitted is then −valence; on the favourable-truth premise, where truth and
    # valence agree, it ranks every false claim above the true one
    fav, unf = torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]])
    control = _States({"true_reject": unf, "true_advance": unf, "false_advance": fav, "false_reject": fav})
    favourable = _States({"true_reject": fav, "true_advance": fav, "false_advance": unf, "false_reject": unf},
                         favourable_true=True)
    u = rrp.fit_direction([control], "correctness")
    assert rrp.paired_accuracy([(favourable, u)], "correctness", n_boot=10, seed=0)["estimate"] == 0.0
    assert rrp.concept_pairs("valence", False) == [("false_reject", "true_reject"), ("false_advance", "true_advance")]
    assert rrp.concept_pairs("valence", True) == [("true_reject", "false_reject"), ("true_advance", "false_advance")]


def _world(truth, valence):
    """The control and favourable-truth premises' states in a world where a state is ``truth`` · t + ``valence`` · v
    (t on the first axis: +1 true claim; v on the second: +1 favourable claim), two records each."""
    def states(favourable_true):
        def cell(c):
            t = 1.0 if c.startswith("true_") else -1.0
            v = 1.0 if (c.startswith("true_") == favourable_true) else -1.0
            return torch.tensor([[truth * t, valence * v]] * 2)
        return _States({c: cell(c) for c in rrp.REASONING_CELLS}, favourable_true=favourable_true)
    return states(False), states(True)


@pytest.mark.parametrize("truth,valence,expect", [(1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.5)])
def test_the_control_direction_on_the_favourable_premise_separates_truth_from_valence(truth, valence, expect):
    # a pure truth code keeps above ½ where truth and valence come apart, a pure valence code falls to 0, and an
    # equal mix ties at ½ (projections cancel): ½ is not evidence of correctness
    control, favourable = _world(truth, valence)
    u = rrp.fit_direction([control], "correctness")
    assert rrp.paired_accuracy([(favourable, u)], "correctness", n_boot=10, seed=0)["estimate"] == expect


def test_the_crossed_fit_cancels_the_other_factor():
    # pooled over the two truth configurations, the correctness direction is pure truth and the valence direction
    # pure valence, even in the mixed world
    control, favourable = _world(1.0, 1.0)
    (d,) = rrp.crossed_directions([(control, control)], [(favourable, favourable)])
    assert torch.allclose(d["correctness"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(d["valence"], torch.tensor([0.0, 1.0]))


def test_the_valence_verdict_has_an_inconclusive_branch():
    iv = lambda lo, hi: {"ci_low": lo, "ci_high": hi}
    assert rrp.valence_verdict(iv(0.4, 0.7), iv(0.8, 0.9)).startswith("inconclusive")       # no transfer at all
    assert rrp.valence_verdict(iv(0.7, 0.9), iv(0.6, 0.8)).startswith("truth")
    assert rrp.valence_verdict(iv(0.7, 0.9), iv(0.1, 0.3)).startswith("valence")
    assert rrp.valence_verdict(iv(0.7, 0.9), iv(0.4, 0.6)).startswith("inconclusive")


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
    assert result["meta"]["settings"]["paraphrase_folds"] == [[[2, 3, 4, 5], [0, 1]], [[0, 1, 4, 5], [2, 3]],
                                                              [[0, 1, 2, 3], [4, 5]]]
    assert result["meta"]["settings"]["connective"] is False
    assert len(result["probe_records"]) == 6 and len(result["eval_records"]) == 5
    assert not set(result["probe_records"]) & set(result["eval_records"])
    assert [r["premise"] for r in result["results"]] == ["parental_leave", "abroad", "no_notice"]
    assert [r["favourable_true"] for r in result["results"]] == [False, False, True]
    assert result["meta"]["data"]["cells.jsonl"]["sha256"] == \
        hashlib.sha256((hiring[0].parent / "cells.jsonl").read_bytes()).hexdigest()
    assert [r["demographic"] for r in result["results"]] == [True, False, False]
    pl = result["results"][0]
    assert set(pl["directions"]) == {"correctness", "conclusion"}          # valence only across premises
    corr = pl["directions"]["correctness"]
    # five eval records, each with its pairs of all three folds
    assert corr["paired_acc_heldout"]["n_clusters"] == 5 and corr["paired_acc_heldout"]["n_items"] == 5 * 3 * 2
    assert len(corr["paired_acc_heldout"]["by_fold"]) == 3 and set(corr["paired_acc_heldout"]["by_pair"]) == \
        {"reject", "advance"}
    assert len(corr["nulled_minus_baseline_by_fold"]) == 3
    assert corr["intervals"]["baseline"]["correctness_effect"]["n_items"] == 5 * 3
    assert len(pl["cosines"]["correctness_conclusion"]) == 3 and len(pl["cosines"]["folds"]["correctness"]) == 3
    # the crossed directions (control + favourable-truth premise), read on every premise
    assert result["crossed"]["fitted_on"] == ["abroad", "no_notice"]
    assert set(result["crossed"]["on"]) == {"parental_leave", "abroad", "no_notice"}
    assert set(result["crossed"]["on"]["parental_leave"]) == {"correctness", "conclusion", "valence"}
    assert set(result["valence_check"]) == {"abroad", "parental_leave"}
    assert result["valence_check"]["abroad"]["test"] == "abroad_on_no_notice"
    # the favourable-truth premise's statistics are its own: no headline pair, coherence-relative interaction
    nn = result["results"][2]
    assert "prefers_correct_over_favorable_rate" not in nn["baseline"]
    assert "prefers_correct_over_favorable_rate" in result["results"][0]["baseline"]
    assert corr["intervals"]["nulled"]["correctness_effect"]["estimate"] == \
        pytest.approx(corr["nulled"]["correctness_effect"])
    assert corr["intervals"]["baseline"]["correctness_effect"]["estimate"] == \
        pytest.approx(pl["baseline"]["correctness_effect"])
    premises = ("parental_leave", "abroad", "no_notice")
    assert set(result["transfer"]) == {f"{a}_on_{b}" for a in premises for b in premises if a != b} | {"cosines"}
    assert set(result["transfer"]["abroad_on_no_notice"]) == {"correctness", "conclusion"}
    assert set(result["transfer"]["abroad_on_no_notice"]["correctness"]) >= {"paired_acc", "nulled", "intervals"}
    assert len(result["transfer"]["cosines"]["correctness"]["abroad"]["no_notice"]) == 3
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


@pytest.mark.parametrize("domain,premises", [("credit", ["age", "sabbatical", "pay_raise"]),
                                             ("education", ["low_income", "out_of_district", "in_district"])])
def test_credit_and_education_end_to_end(run, request, domain, premises):
    raw = request.getfixturevalue(CORPUS[domain])[1]
    result = run(domain=domain)
    assert result["domain"] == domain and result["meta"]["settings"]["premises"] == premises
    assert result["meta"]["data"][raw.name]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert [r["premise"] for r in result["results"]] == premises
    assert len(result["transfer"]) == 3 * 2 + 1
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


def test_each_fold_is_read_with_its_own_direction():
    # fold k's held-out states with fold k's direction (fitted without that fold's held-out wording), never another's
    splits = [(f"fit{k}", f"eval{k}") for k in range(3)]
    directions = [{"correctness": f"u{k}"} for k in range(3)]
    assert rrp.read_with(splits, directions, "correctness") == [("eval0", "u0"), ("eval1", "u1"), ("eval2", "u2")]
