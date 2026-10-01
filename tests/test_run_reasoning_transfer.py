"""`runners/run_reasoning_transfer.py`: reasoning directions fitted on one domain (valence-balanced; or its
non-demographic premises; or the other domains, each weighing the same) and measured on another's held-out items over
three wording folds, beside a bag-of-words lexical control; the transfer gap; the refusals; provenance — end to end on
the tiny model and the generators' manifests of all three domains."""

from __future__ import annotations

import json

import pytest
import torch
import yaml

from runners import run_reasoning_transfer as rrt
from runners import run_reasoning_probe as rrp
from tests.conftest import reasoning_domain
from tests.test_run_cross_marker import _model, _tokenizer
from tests.test_run_reasoning_flip import CORPUS

CELLS = ("true_reject", "true_advance", "false_advance", "false_reject")


def _vectors(cells, favourable_true=False):
    """A split of ``n`` items given its states per cell ({cell: [n, d]})."""
    n = len(next(iter(cells.values())))
    return rrt.Vectors(torch.cat([cells[c] for c in CELLS]), n, favourable_true)


def _correctness_split(diff, favourable_true=False):
    """One item whose correctness pairs both differ by ``diff`` (true cells ``diff``, false cells 0)."""
    zero = torch.zeros(1, len(diff))
    return _vectors({"true_reject": diff[None], "true_advance": diff[None], "false_advance": zero,
                     "false_reject": zero}, favourable_true)


def test_the_domain_direction_weighs_the_two_valences_equally():
    # two unfavourable-truth splits pointing along x, one favourable-truth split along y: valence-balanced, the
    # direction is the diagonal, not two thirds x
    x, y = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    u = rrt.domain_direction([_correctness_split(x), _correctness_split(x), _correctness_split(y, True)],
                             "correctness")
    assert torch.allclose(u, torch.tensor([1.0, 1.0]) / 2 ** 0.5, atol=1e-6)
    assert torch.allclose(rrt.domain_direction([_correctness_split(3 * x)], "correctness"), x)


def test_others_weighs_every_domain_the_same():
    # a domain with large pair differences does not dominate: the unit directions are averaged
    u = rrt.others_direction([torch.tensor([10.0, 0.0]), torch.tensor([0.0, 0.1])])
    assert torch.allclose(u, torch.tensor([1.0, 1.0]) / 2 ** 0.5, atol=1e-6)


def test_the_domain_direction_weighs_groups_not_pairs():
    # three unfavourable-truth items along x against one favourable-truth item along 5y: the groups' means count,
    # not their sizes or the number of pairs
    x, y = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    big = _vectors({"true_reject": x.repeat(3, 1), "true_advance": x.repeat(3, 1), "false_advance": torch.zeros(3, 2),
                    "false_reject": torch.zeros(3, 2)})
    u = rrt.domain_direction([big, _correctness_split(5 * y, True)], "correctness")
    assert torch.allclose(u, torch.tensor([1.0, 5.0]) / 26 ** 0.5, atol=1e-6)


def test_the_sources_fit_on_the_right_premises():
    domains = ["cv", "credit", "education"]
    assert rrt.source_members("credit", "cv", domains) == [("credit", ("age", "sabbatical", "pay_raise"))]
    assert rrt.source_members("credit:nondemographic", "cv", domains) == [("credit", ("sabbatical", "pay_raise"))]
    assert rrt.source_members("others", "credit", domains) == [
        ("cv", ("parental_leave", "abroad", "no_notice")), ("education", ("low_income", "out_of_district",
                                                                          "in_district"))]
    # one member: its direction; several: their unit directions averaged
    x, y = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert torch.allclose(rrt.source_direction([[_correctness_split(3 * x)]], "correctness"), x)
    assert torch.allclose(rrt.source_direction([[_correctness_split(3 * x)], [_correctness_split(0.1 * y)]],
                                               "correctness"), torch.tensor([1.0, 1.0]) / 2 ** 0.5, atol=1e-6)


def test_the_lexical_control_is_the_expected_wording_and_cancels_where_one_claim_is_shared():
    # hiring's three premises share one claim: valence-balanced, the expected bag-of-words of true and false claims
    # cancel exactly, so the words' correctness direction is 0 and every held-out pair ties (½), with no draw noise
    from pairs.verdicts import PARAPHRASE_FOLDS
    from runners.run_reasoning_probe import reasoning_items
    from substrates.domains import get_domain
    from tests.test_decision_response import _rec

    fit_entries, held = PARAPHRASE_FOLDS[0]
    members = rrt.source_members("cv", "credit", ["cv", "credit"])
    evaluated = [reasoning_items(get_domain("cv"), "abroad", [_rec(f"r{i}") for i in range(6)], held, 42)]
    fit, ev = rrt.expected_lexical(members, fit_entries, evaluated)
    assert [len(m) for m in fit] == [3] and all(v.n == 1 for v in fit[0]) and ev[0].n == 6
    u = rrt.domain_direction(fit[0], "correctness")
    assert float(u.norm()) == 0.0
    acc = rrt.paired_accuracy([(ev[0], u)], "correctness", n_boot=20, seed=0)
    assert acc["estimate"] == 0.5
    # the vocabulary is the source's fitted wording: a word only the held-out wording uses adds no column
    assert ev[0].hidden.shape[1] == fit[0][0].hidden.shape[1]
    # conclusion is carried by the decision sentences' words (not cancelled by valence balancing)
    assert float(rrt.domain_direction(fit[0], "conclusion").norm()) > 0


@pytest.fixture
def run(tmp_path, monkeypatch, request):
    """`main` on the domains' manifests, the tiny model standing in for the loader; ``texts`` records every text
    embedded through `run_reasoning_probe.SplitStates`."""
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    loads, texts = [], []
    data = {d: request.getfixturevalue(fixture) for d, fixture in CORPUS.items()}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rrt, "get_domain", lambda name: reasoning_domain(name, data[name][1]))
    embed = rrp.embed_with_gates

    def recording(model, tok, flat, **kw):
        texts.extend((t, kw["max_length"]) for t in flat)
        return embed(model, tok, flat, **kw)

    monkeypatch.setattr(rrp, "embed_with_gates", recording)

    def configs(domains, models=None, devices=None, extras=None):
        paths = []
        for i, d in enumerate(domains):
            path = tmp_path / f"cfg_{i}.yaml"
            path.write_text(yaml.safe_dump({
                "name": "t", "bias_type": "demographic", "model_path": (models or {}).get(d, "org/Tiny-RM"),
                "device": (devices or {}).get(d, "auto"), "dataset_source": str(data[d][0]), "batch_size": 16,
                "max_length": 1024, "extra": {"domain": d, **(extras or {}).get(d, {"n_boot": 20})}}))
            paths.append(str(path))
        return paths

    def load_model(self):
        loads.append(1)
        self.model, self.tokenizer = _model(), _tokenizer()
        self.config.model_revision = "abc123"

    def _run(*extra, domains=("cv", "credit", "education"), models=None, devices=None, extras=None,
             name="__probe_items-4__eval_items-3"):
        monkeypatch.setattr(rrt.DemographicBiasExperiment, "load_model", load_model)
        monkeypatch.setattr("sys.argv", ["run_reasoning_transfer.py", "--configs",
                                         *configs(domains, models, devices, extras),
                                         "--probe-items", "4", "--eval-items", "3", *extra])
        rrt.main()
        return json.loads((tmp_path / f"artifacts/results/demographic/reasoning_transfer_Tiny-RM{name}.json")
                          .read_text())

    _run.loads, _run.data, _run.texts, _run.load_model = loads, data, texts, load_model
    return _run


def test_end_to_end_on_a_tiny_model(run):
    result = run()
    domains = ["cv", "credit", "education"]
    assert result["domains"] == domains and len(run.loads) == 1          # one model for every domain
    meta = result["meta"]
    assert {f"{d}/cells.jsonl" for d in domains} <= set(meta["data"])
    # the loaded commit is every config's record
    assert {c["model_revision"] for c in meta["settings"]["configs"].values()} == {"abc123"}
    assert set(result["baselines"]) == set(domains) and set(result["baselines"]["credit"]) == \
        {"age", "sabbatical", "pay_raise"}
    for d in domains:
        recs = result["records"][d]
        assert len(recs["probe"]) == 4 and len(recs["eval"]) == 3 and not set(recs["probe"]) & set(recs["eval"])
    rows = result["rows"]
    sources = [*domains, *(d + ":nondemographic" for d in domains), "others"]
    assert meta["settings"]["sources"] == sources
    # 7 sources × 3 evaluated domains × 3 premises × 3 concepts
    assert len(rows) == 7 * 3 * 3 * 3
    premises = {"cv": ["parental_leave", "abroad", "no_notice"], "credit": ["age", "sabbatical", "pay_raise"],
                "education": ["low_income", "out_of_district", "in_district"]}
    assert {(r["evaluated"], r["premise"]) for r in rows} == {(d, p) for d in domains for p in premises[d]}
    diag = {(r["evaluated"], r["premise"], r["concept"]): r for r in rows if r["source"] == r["evaluated"]}
    for r in rows:
        assert (r["transfer_gap"] is None) == (r["source"] == r["evaluated"])
        if r["source"] == "others":
            assert r["fit_domains"] == [d for d in domains if d != r["evaluated"]]
        else:
            assert r["fit_domains"] == [r["source"].split(":")[0]]
        if r["source"].endswith(":nondemographic"):
            assert len(r["fit_premises"][r["fit_domains"][0]]) == 2
        acc = r["paired_acc"]
        assert (acc["n_clusters"], acc["n_items"], len(acc["by_fold"])) == (3, 3 * 3 * 2, 3)
        lex = r["lexical_paired_acc"]
        assert (lex["n_clusters"], len(lex["by_fold"])) == (3, 3)
        assert "baseline" not in r["intervals"]                     # once per (domain, premise): result["baselines"]
        if r["transfer_gap"] is not None:
            # own change − this source's change, both pooled over the folds
            own = diag[(r["evaluated"], r["premise"], r["concept"])]["intervals"]["nulled_minus_baseline"]
            mine = r["intervals"]["nulled_minus_baseline"]
            for k in ("correctness_effect", "conclusion_effect"):
                assert r["transfer_gap"][k]["estimate"] == pytest.approx(own[k]["estimate"] - mine[k]["estimate"],
                                                                         abs=1e-12)
    # the favourable-truth premise's statistics are its own (no headline pair)
    assert all(("prefers_correct_over_favorable_rate" in r["nulled"]) != r["favourable_true"] for r in rows)
    cos = result["cosines"]["correctness"]
    assert all(len(cos[a][b]) == 3 for a in domains for b in domains)
    with pytest.raises(SystemExit, match="exists"):
        run()


def test_the_texts_are_the_reasoning_probes(run, tmp_path, monkeypatch):
    # the probe runner with the same settings embeds exactly these texts (with the same max_length), so the
    # embedding cache it filled serves the transfer runner whole
    run()
    transfer = set(run.texts)
    run.texts.clear()
    for i in range(3):
        monkeypatch.setattr(rrp, "get_domain", lambda name: reasoning_domain(name, run.data[name][1]))
        monkeypatch.setattr(rrp.DemographicBiasExperiment, "load_model", run.load_model)
        monkeypatch.setattr("sys.argv", ["run_reasoning_probe.py", "--config", str(tmp_path / f"cfg_{i}.yaml"),
                                         "--probe-items", "4", "--eval-items", "3"])
        rrp.main()
    assert set(run.texts) == transfer and transfer


def test_bad_inputs_are_refused_before_the_model_loads(run):
    with pytest.raises(SystemExit, match="distinct domains"):
        run(domains=("cv", "cv"))
    with pytest.raises(SystemExit, match="distinct domains"):
        run(domains=("cv",))
    with pytest.raises(SystemExit, match="one model"):
        run(models={"credit": "org/Other-RM"})
    with pytest.raises(SystemExit, match="differ in"):
        run(devices={"credit": "cpu"})
    with pytest.raises(SystemExit, match="extra.n_boot"):
        run(extras={"credit": {"n_boot": 30}})
    with pytest.raises(SystemExit, match="one model"):
        run(models={"credit": "org/Tiny-RM@other"})
    with pytest.raises(SystemExit, match="at least 2"):
        run("--probe-items", "1")
    with pytest.raises(SystemExit, match="fewer than"):
        run("--eval-items", "100000")
    assert run.loads == []


def test_two_domains_have_no_others_source_and_name_the_domain_set(run):
    result = run(domains=("cv", "credit"), name="__probe_items-4__eval_items-3__domains-cv-credit")
    assert {r["source"] for r in result["rows"]} == {"cv", "credit", "cv:nondemographic", "credit:nondemographic"}
    assert len(result["rows"]) == 4 * 2 * 3 * 3


def test_the_expected_lexical_vectors_are_the_mean_over_every_fitted_wording():
    # credit's age premise, fold 0: each cell's vector is the mean bag-of-words over its 4 × 4 fitted stem and decision
    # entries, with the source's vocabulary
    from pairs.verdicts import PARAPHRASE_FOLDS, REASONING_FRAMES, reasoning_cell_text, reasoning_stem
    from probes.erasure import bag_of_words

    fit_entries, _ = PARAPHRASE_FOLDS[0]
    members = [("credit", ("age",))]
    fit, _ = rrt.expected_lexical(members, fit_entries, [])
    frame = REASONING_FRAMES["credit"]
    texts = {c: [reasoning_cell_text(reasoning_stem("age", "credit", c.startswith("true_"), e),
                                     (frame.advance_decision if c.endswith("_advance") else frame.reject_decision)[k])
                 for e in fit_entries for k in fit_entries] for c in CELLS}
    X, _ = bag_of_words([t for c in CELLS for t in texts[c]], [])
    expect = X.reshape(4, 16, -1).mean(1)
    assert torch.allclose(fit[0][0].hidden, expect)


def test_a_config_without_n_boot_matches_one_with_the_default(run):
    from scoring.intervals import DEFAULT_N_BOOT

    # both read DEFAULT_N_BOOT: not refused as differing (the run itself is the default's, here cut short)
    with pytest.raises(SystemExit, match="fewer than"):
        run("--eval-items", "100000", extras={"cv": {}, "credit": {"n_boot": DEFAULT_N_BOOT}, "education": {}})
