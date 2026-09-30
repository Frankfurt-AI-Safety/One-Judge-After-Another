"""`runners/run_reasoning_probe.py`: directions fitted on one split of applicants and paraphrases, measured on
another split in wording they never saw; the paired accuracy (a tie ½, records resampled whole); the refusals;
provenance — end to end on the tiny model and the `hiring` fixture."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
import torch
import yaml

from runners import run_reasoning_probe as rrp
from substrates.domains import get_domain
from tests.test_run_cross_marker import _model, _tokenizer


def test_the_eval_wording_is_unseen_and_items_do_not_depend_on_their_neighbours():
    from tests.test_reasoning_flip import _rec

    dom = get_domain("cv")
    recs = [_rec(f"r{i}") for i in range(12)]
    probe = rrp.reasoning_items(dom, "parental_leave", recs, rrp.PROBE_PARAPHRASES, 42)
    evals = rrp.reasoning_items(dom, "parental_leave", recs, rrp.EVAL_PARAPHRASES, 42)
    fitted = {v for it in probe for v in it["cells"].values()}
    assert not fitted & {v for it in evals for v in it["cells"].values()}
    assert all(it["meta"]["claim_type"] == "availability" for it in probe + evals)
    # an item's draws depend on its record, not on its position in the split
    assert rrp.reasoning_items(dom, "parental_leave", recs[5:6], rrp.PROBE_PARAPHRASES, 42)[0]["cells"] == \
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
def run(hiring, tmp_path, monkeypatch):
    from substrates.bios_clean import DEFAULT_N_BIOS, load_factorial_bios

    pairs, raw = hiring
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rrp, "DEFAULT_BIOS_PATH", str(raw))
    monkeypatch.setattr(rrp, "get_domain", lambda name: dataclasses.replace(
        get_domain(name), load_records=lambda: load_factorial_bios(str(raw), n=DEFAULT_N_BIOS)))
    loads = []

    def _run(*extra, domain="cv"):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(pairs),
            "batch_size": 16, "max_length": 1024, "extra": {"domain": domain, "n_boot": 50}}))

        def load_model(self):
            loads.append(1)
            self.model, self.tokenizer = _model(), _tokenizer()
            self.config.model_revision = "abc123"

        monkeypatch.setattr(rrp.DemographicBiasExperiment, "load_model", load_model)
        monkeypatch.setattr("sys.argv", ["run_reasoning_probe.py", "--config", str(cfg_path), "--probe-items", "6",
                                         "--eval-items", "5", *extra])
        rrp.main()
        return json.loads((tmp_path / "artifacts/results/demographic/reasoning_probe_cv_Tiny-RM.json").read_text())

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
    assert [r["premise"] for r in result["results"]] == list(rrp.PREMISES)
    pl = result["results"][0]
    corr = pl["directions"]["correctness"]
    assert corr["paired_acc_heldout"]["n_clusters"] == 5
    assert corr["intervals"]["nulled"]["correctness_effect"]["estimate"] == \
        pytest.approx(corr["nulled"]["correctness_effect"])
    assert corr["intervals"]["baseline"]["correctness_effect"]["estimate"] == \
        pytest.approx(pl["baseline"]["correctness_effect"])
    assert set(result["transfer"]) == {"commute_on_parental_leave", "parental_leave_on_commute", "cosines"}
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


def test_bad_inputs_are_refused_before_the_model_loads(run):
    with pytest.raises(SystemExit, match="hiring-only"):
        run(domain="credit")
    with pytest.raises(SystemExit, match="fewer than"):
        run("--eval-items", "100000")
    assert run.loads == []
