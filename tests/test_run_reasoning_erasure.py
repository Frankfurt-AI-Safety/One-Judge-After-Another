"""`runners/run_reasoning_erasure.py`: the design that keeps surface features from answering (no connective,
held-out wording; pinned on the lexical control itself, in every domain), the labels, the refusals, provenance — end
to end on the tiny model and the generators' manifests of every domain."""

from __future__ import annotations

import hashlib
import json
import random

import pytest
import yaml

from runners import run_reasoning_erasure as rre
from tests.conftest import reasoning_domain
from tests.test_decision_response import _domain_case
from tests.test_run_cross_marker import _model, _tokenizer
from tests.test_run_reasoning_flip import CORPUS


def test_labels_follow_the_cells():
    assert [rre.label(c, "correctness") for c in rre.REASONING_CELLS] == [1, 1, 0, 0]
    assert [rre.label(c, "conclusion") for c in rre.REASONING_CELLS] == [0, 1, 1, 0]


def _lexical_mlp_after_leace(fit, evaluate, connective, domain="cv"):
    """MLP accuracy after LEACE on bag-of-words vectors of the verdicts of the domain's demographic premise
    (correctness, 150 + 150 items)."""
    from pairs.verdicts import REASONING_FRAMES, build_reasoning_item
    from probes.erasure import apply_eraser, bag_of_words, leace_erase, probe_recoverability

    rec, render, tid, _ = _domain_case(domain)
    premise = REASONING_FRAMES[domain].primary

    def split(offset, paraphrases):
        items = [build_reasoning_item(rec, premise, render, random.Random(offset + i), template_id=tid, vary=True,
                                      paraphrases=paraphrases, connective=connective, domain=domain)
                 for i in range(150)]
        return ([it["cells"][c] for it in items for c in rre.REASONING_CELLS],
                [rre.label(c, "correctness") for _ in items for c in rre.REASONING_CELLS])

    (tr, ytr), (ev, yev) = split(0, fit), split(10_000, evaluate)
    Xtr, Xev = bag_of_words(tr, ev)
    eraser = leace_erase(Xtr, ytr)
    return probe_recoverability(apply_eraser(eraser, Xtr), ytr, apply_eraser(eraser, Xev), yev, n_boot=10)["mlp_acc"]


@pytest.mark.parametrize("domain", ["cv", "credit", "education"])
def test_surface_features_alone_recover_correctness_after_leace_only_in_the_old_design(domain):
    from pairs.verdicts import EVAL_PARAPHRASES, FIT_PARAPHRASES

    # shared pools (with or without "so"/"but"): word identity alone gives the "entangled" reading
    assert _lexical_mlp_after_leace(None, None, connective=True, domain=domain) > 0.9
    assert _lexical_mlp_after_leace(None, None, connective=False, domain=domain) > 0.9
    # the runner's design: held-out wording, no connective — the lexical control sits at chance
    assert _lexical_mlp_after_leace(FIT_PARAPHRASES, EVAL_PARAPHRASES, connective=False, domain=domain) < 0.6


@pytest.fixture
def run(tmp_path, monkeypatch, request):
    """`main` on a domain's manifest (its corpus fixture, built on first use), the tiny model standing in for the
    loader."""
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    loads = []

    def _run(*extra, domain="cv", config_domain=None):
        pairs, raw = request.getfixturevalue(CORPUS[domain])
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(rre, "get_domain", lambda name: reasoning_domain(name, raw))
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(pairs),
            "batch_size": 16, "max_length": 1024, "extra": {"domain": config_domain or domain, "n_boot": 20}}))

        def load_model(self):
            loads.append(1)
            self.model, self.tokenizer = _model(), _tokenizer()
            self.config.model_revision = "abc123"

        monkeypatch.setattr(rre.DemographicBiasExperiment, "load_model", load_model)
        monkeypatch.setattr("sys.argv", ["run_reasoning_erasure.py", "--config", str(cfg_path), "--probe-items", "8",
                                         "--eval-items", "6", *extra])
        rre.main()
        return json.loads((tmp_path / f"artifacts/results/demographic/erasure_{domain}_Tiny-RM"
                                      "__probe_items-8__eval_items-6.json").read_text())

    _run.loads = loads
    return _run


def test_end_to_end_on_a_tiny_model(run, hiring):
    result = run()
    raw = hiring[1]
    assert result["meta"]["config"]["model_revision"] == "abc123"
    assert result["meta"]["data"][raw.name]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert result["meta"]["settings"]["connective"] is False
    assert not set(result["probe_records"]) & set(result["eval_records"])
    assert [(r["premise"], r["concept"]) for r in result["results"]] == \
        [(p, c) for p in ("parental_leave", "abroad") for c in rre.CONCEPTS]
    r = result["results"][0]
    assert (r["n_train"], r["n_eval"]) == (32, 24)
    for rows in (r["rows"], r["lexical_control"]):
        assert set(rows) == set(rre.METHODS)
        assert rows["leace"]["intervals"]["mlp_acc"]["n_clusters"] == 6        # applicants, not states
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


@pytest.mark.parametrize("domain,premises", [("credit", ["age", "sabbatical"]),
                                             ("education", ["low_income", "out_of_district"])])
def test_credit_and_education_end_to_end(run, request, domain, premises):
    result = run(domain=domain)
    assert result["domain"] == domain and result["meta"]["settings"]["premises"] == premises
    assert [(r["premise"], r["concept"]) for r in result["results"]] == \
        [(p, c) for p in premises for c in rre.CONCEPTS]
    assert result["selection"]["ineligible"] == (12 if domain == "credit" else 0)
    frame, raw = rre.REASONING_FRAMES[domain], request.getfixturevalue(CORPUS[domain])[1]
    ineligible = {str(r.source_record_id) for r in reasoning_domain(domain, raw).load_records()
                  if not frame.is_eligible(r)}
    assert not ineligible & set(result["probe_records"] + result["eval_records"])


def test_bad_inputs_are_refused_before_the_model_loads(run):
    with pytest.raises(SystemExit, match="runs on"):
        run(config_domain="nope")
    with pytest.raises(SystemExit, match="fewer than"):
        run("--eval-items", "100000")
    assert run.loads == []
