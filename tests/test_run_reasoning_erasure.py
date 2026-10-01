"""`runners/run_reasoning_erasure.py`: the design that keeps surface features from answering (no connective,
held-out wording; pinned on the lexical control itself, in every domain), the labels, the refusals, provenance — end
to end on the tiny model and the generators' manifests of every domain."""

from __future__ import annotations

import hashlib
import json
import random

import pytest
import torch
import yaml

from runners import run_reasoning_erasure as rre
from tests.conftest import reasoning_domain
from tests.test_decision_response import _domain_case
from tests.test_run_cross_marker import _model, _tokenizer
from tests.test_run_reasoning_flip import CORPUS


def test_labels_follow_the_cells():
    # cells: true_reject, true_advance, false_advance, false_reject
    assert [rre.label(c, "correctness") for c in rre.REASONING_CELLS] == [1, 1, 0, 0]
    assert [rre.label(c, "conclusion") for c in rre.REASONING_CELLS] == [0, 1, 1, 0]
    # valence: the favourable claim's cells — the false ones, unless the premise makes the favourable claim true
    assert [rre.label(c, "valence", False) for c in rre.REASONING_CELLS] == [0, 0, 1, 1]
    assert [rre.label(c, "valence", True) for c in rre.REASONING_CELLS] == [1, 1, 0, 0]


def _lexical(fit, evaluate, domain, premise, concept="correctness"):
    """The runner's design on bag-of-words vectors of the verdicts: probes trained on the control's and the
    favourable-truth premise's verdicts pooled (paraphrase entries ``fit``; all entries if None), evaluated on
    ``premise``'s (entries ``evaluate``): linear accuracy without erasure and MLP accuracy after LEACE (150 + 150
    items)."""
    from pairs.verdicts import REASONING_FRAMES, build_reasoning_item
    from probes.erasure import apply_eraser, bag_of_words, leace_erase, probe_recoverability

    rec, render, tid, _ = _domain_case(domain)
    frame = REASONING_FRAMES[domain]

    def split(offset, paraphrases, premises):
        items = [build_reasoning_item(rec, p, render, random.Random(offset + i), template_id=tid, vary=True,
                                      paraphrases=paraphrases, domain=domain)
                 for p in premises for i in range(150)]
        return ([it["cells"][c] for it in items for c in rre.REASONING_CELLS],
                [rre.label(c, concept, it["meta"]["favourable_true"]) for it in items for c in rre.REASONING_CELLS])

    (tr, ytr) = split(0, fit, (frame.control, frame.favourable))
    (ev, yev) = split(10_000, evaluate, (premise,))
    Xtr, Xev = bag_of_words(tr, ev)
    none = probe_recoverability(Xtr, ytr, Xev, yev, n_boot=0)["linear_acc"]
    eraser = leace_erase(Xtr, ytr)
    leace = probe_recoverability(apply_eraser(eraser, Xtr), ytr, apply_eraser(eraser, Xev), yev, n_boot=0)["mlp_acc"]
    return none, leace


@pytest.mark.parametrize("domain", ["cv", "credit", "education"])
def test_surface_features_alone_recover_correctness_only_on_shared_wording(domain):
    from pairs.verdicts import PARAPHRASE_FOLDS, REASONING_FRAMES

    frame = REASONING_FRAMES[domain]
    # shared pools: word identity alone recovers correctness, after LEACE too (the old "entangled" reading)
    assert _lexical(None, None, domain, frame.control)[1] > 0.9
    # the runner's design: held-out wording in every fold, every premise, no connective — the words alone sit at
    # chance both ways, without erasure and after LEACE (no word that separates true from false in the fitted
    # wording recurs in the held-out wording)
    for fit, held in PARAPHRASE_FOLDS:
        for premise in (frame.primary, frame.control, frame.favourable):
            none, leace = _lexical(fit, held, domain, premise)
            assert abs(none - 0.5) < 0.1 and abs(leace - 0.5) < 0.1, (domain, held, premise, none, leace)


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
        [(p, c) for p in ("parental_leave", "abroad", "no_notice") for c in rre.CONCEPTS]
    assert result["meta"]["settings"]["connective"] is False and len(result["meta"]["settings"]["paraphrase_folds"]) == 3
    r = result["results"][0]
    # per fold: trained on the control's and the favourable-truth premise's 8 applicants × 4 cells
    assert (r["n_train"], r["n_eval"], r["folds"]) == (64, 24, 3)
    assert r["fitted_on"] == ["abroad", "no_notice"]
    for rows in (r["rows"], r["lexical_control"]):
        assert set(rows) == set(rre.METHODS)
        iv = rows["leace"]["intervals"]["mlp_acc"]
        assert (iv["n_clusters"], iv["n_items"]) == (6, 24 * 3)        # applicants with their states of all folds
        assert len(rows["leace"]["by_fold"]) == 3
        assert rows["leace"]["mlp_acc"] == pytest.approx(sum(f["mlp_acc"] for f in rows["leace"]["by_fold"]) / 3)
    assert [x["favourable_true"] for x in result["results"]] == [False] * 3 + [False] * 3 + [True] * 3
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


@pytest.mark.parametrize("domain,premises", [("credit", ["age", "sabbatical", "pay_raise"]),
                                             ("education", ["low_income", "out_of_district", "in_district"])])
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


def test_pooling_the_folds_keeps_each_applicant_one_cluster():
    # two folds of two applicants × two states; fold 0's MLP right on applicant 0 only, fold 1's on both
    item = lambda ok, label: (ok, ok, label, False)
    folds = [[item(True, 1), item(True, 0), item(False, 1), item(False, 0)],
             [item(True, 1), item(True, 0), item(True, 1), item(True, 0)]]
    groups = [0, 0, 1, 1]                                   # one fold's states → applicant
    pooled = rre.pool_folds(folds, groups, n_boot=200, seed=0)
    iv = pooled["intervals"]["mlp_acc"]
    assert pooled["mlp_acc"] == 0.75 and (iv["n_clusters"], iv["n_items"]) == (2, 8)
    # applicant 0 scores 1, applicant 1 scores ½: resampling whole applicants gives 0.5–1, never another split
    assert iv["ci_low"] == 0.5 and iv["ci_high"] == 1.0
    assert [f["mlp_acc"] for f in pooled["by_fold"]] == [0.5, 1.0]
    with pytest.raises(ValueError, match="cluster keys"):
        rre.pool_folds(folds, groups[:3], n_boot=10, seed=0)


def test_each_premise_slice_of_the_pooled_eval_is_that_premise(run, monkeypatch):
    # the eval states of all premises are embedded together and sliced per premise: a fake embedding encodes each
    # text's premise, and the valence labels differ for the favourable-truth premise, so a permuted slice shows
    from pairs.verdicts import REASONING_FRAMES

    frame = REASONING_FRAMES["cv"]
    premises = (frame.primary, frame.control, frame.favourable)
    clause = {p: frame.premises[p].clause.strip() for p in premises}
    flat = lambda t: " ".join(t) if isinstance(t, (tuple, list)) else t             # a formatted conversation
    fake = lambda model, tok, texts, *a, **k: torch.tensor(
        [[float(next(i for i, p in enumerate(premises) if clause[p] in flat(t))), float(len(flat(t)) % 7), 1.0]
         for t in texts])
    monkeypatch.setattr(rre, "get_embeddings", fake)
    seen = []
    rows = rre.erasure_rows

    def capture(Xtr, ytr, Xev, yev, groups, seed, erase=None):
        seen.append((Xev, yev))
        return rows(Xtr, ytr, Xev, yev, groups, seed, erase)

    monkeypatch.setattr(rre, "erasure_rows", capture)
    run()
    concepts = list(rre.CONCEPTS)
    for i, (Xev, yev) in enumerate(seen):
        if i % 2:                                   # the lexical control's call
            continue
        size = len(yev) // 3
        for k in range(3):
            assert set(Xev[k * size:(k + 1) * size, 0].tolist()) == {float(k)}
        if concepts[(i // 2) % 3] == "valence":
            n = size // 4                 # item-major (Split): true_reject, true_advance, false_advance, false_reject
            assert yev[:size] == [0, 0, 1, 1] * n                       # the primary: the false claim favours
            assert yev[2 * size:] == [1, 1, 0, 0] * n                   # the favourable-truth premise
