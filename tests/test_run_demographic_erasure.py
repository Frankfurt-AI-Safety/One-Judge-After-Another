"""`runners/run_demographic_erasure.py`: the labels, the cell erasure, the held-out name folds, the reward gap, planted
worlds whose reading is known (low-complexity, intersectional, non-linear beyond the cells), the states being the
battery's texts, and the runner end to end on the credit fixture manifest with the tiny model."""

from __future__ import annotations

import json
from itertools import product

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("sklearn")
pytest.importorskip("concept_erasure")

from pairs.factorial import CREDIT_DESIGN, EDUCATION_DESIGN, HIRING_DESIGN, ProxyNames  # noqa: E402
from runners import run_demographic_erasure as rde  # noqa: E402

D = CREDIT_DESIGN


# --------------------------------------------------------------------------- labels -------------------
def test_concepts_per_encoding():
    assert rde.concepts(D, "explicit") == [
        ("sex", ("sex",)), ("age", ("age",)), ("marital_status", ("marital_status",)),
        ("sex_x_age", ("sex", "age")), ("sex_x_marital_status", ("sex", "marital_status")),
        ("age_x_marital_status", ("age", "marital_status"))]
    # credit's marital status has no proxy: the proxy concepts are sex, age and their interaction
    assert rde.concepts(D, "proxy") == [("sex", ("sex",)), ("age", ("age",)), ("sex_x_age", ("sex", "age"))]
    assert len(rde.concepts(HIRING_DESIGN, "proxy")) == len(rde.concepts(EDUCATION_DESIGN, "proxy")) == 6


def test_labels_are_pole_a_and_same_pole():
    label = lambda cell, *f: rde.concept_label(D, cell, f)
    assert label(("female", 30, "married"), "sex") == 1 and label(("male", 30, "married"), "sex") == 0
    assert label(("female", 50, "single"), "age") == 0                  # pole A of age is the young one
    # interaction: 1 where both factors sit at the same pole
    assert [label((s, a, "married"), "sex", "age") for s, a in product(("female", "male"), (30, 50))] == [1, 0, 0, 1]
    # balanced and orthogonal to both main effects over the 8 cells (±1 coding)
    pm = lambda col: 2 * np.asarray(col) - 1
    sex, age, inter = (pm([label(c, *f) for c in D.cells]) for f in (("sex",), ("age",), ("sex", "age")))
    assert inter.sum() == 0 and inter @ sex == 0 and inter @ age == 0


def test_cell_columns_span_every_function_of_the_cell():
    cols = np.asarray(rde.cell_columns(D, list(D.cells)), dtype=float)
    assert cols.shape == (7, 8)
    # with the constant they span R^8: erased jointly, no linear function separates any cell from another
    assert np.linalg.matrix_rank(np.vstack([cols, np.ones(8)])) == 8


# --------------------------------------------------------------------------- names --------------------
def test_name_folds_hold_out_every_name_once_and_every_pool_evenly():
    for domain, pools in rde.NAME_POOLS.items():
        folds = rde.name_folds(domain, 5, seed=42)
        assert set(folds) == {n for pool in pools for n in pool}
        for pool in pools:
            assert sorted(sum(folds[n] == f for n in pool) for f in range(5)) == [2] * 5
        assert folds == rde.name_folds(domain, 5, seed=42)
        assert folds != rde.name_folds(domain, 5, seed=7)


def test_cell_name_matches_whole_words_only():
    from pairs.cross_marker import CellBlock

    grid = {("female", "white"): "Emily", ("male", "white"): "Ryan", ("female", "black"): "Tyra",
            ("male", "black"): "Tyrone"}
    names = ProxyNames(female="Emily", male="Ryan", grid=grid)
    clauses = {c: EDUCATION_DESIGN.clause(c, "proxy", names, "student") for c in EDUCATION_DESIGN.cells}
    block = CellBlock("r", "t", "proxy", {}, texts={c: "x" + k for c, k in clauses.items()}, clauses=clauses,
                      unmarked="x", names=frozenset(grid.values()))
    assert {rde.cell_name(block, c) for c in EDUCATION_DESIGN.cells if c[:2] == ("male", "black")} == {"Tyrone"}
    assert {rde.cell_name(block, c) for c in EDUCATION_DESIGN.cells if c[:2] == ("female", "black")} == {"Tyra"}


DOMAINS = {"credit": CREDIT_DESIGN, "cv": HIRING_DESIGN, "education": EDUCATION_DESIGN}


def _states(n_records, prefix, encoding="explicit", seed=0, domain="credit"):
    """Synthetic states of a domain: every cell of ``n_records`` records, one template; proxy names drawn per record
    from the domain's pools (education: one per sex × ethnicity cell)."""
    design, rng = DOMAINS[domain], np.random.default_rng(seed)
    pools = rde.NAME_POOLS[domain]
    st = rde.States()
    for r in range(n_records):
        drawn = [str(rng.choice(pool)) for pool in pools]
        grid = (None if len(pools) == 2 else
                dict(zip((("female", "white"), ("male", "white"), ("female", "black"), ("male", "black")), drawn)))
        names = ProxyNames(female=drawn[0], male=drawn[1], grid=grid)
        subject = "student" if domain == "education" else "applicant"
        for cell in design.cells:
            st.cells.append(cell)
            st.records.append(f"{prefix}{r}")
            st.texts.append(None)
            st.clauses.append(design.clause(cell, encoding, names, subject))
            name = None
            if encoding == "proxy":
                name = grid[(cell[0], cell[1])] if grid else (names.female if cell[0] == "female" else names.male)
            st.names.append(name)
    return st


def test_no_eval_state_is_evaluated_by_a_fit_that_saw_its_name():
    train, evals = _states(60, "p", "proxy"), _states(40, "e", "proxy", seed=1)
    folds = rde.name_folds("credit", 5, seed=42)
    splits = rde.fold_indices(train, evals, folds, 5)
    assert sorted(i for _, ev in splits for i in ev) == list(range(len(evals)))      # every eval state once
    for tr, ev in splits:
        assert not {train.names[i] for i in tr} & {evals.names[i] for i in ev}
    # explicit: a single fit on everything
    assert rde.fold_indices(train, evals, None, 5) == [(list(range(len(train))), list(range(len(evals))))]
    with pytest.raises(ValueError, match="no name pool"):
        rde.fold_indices(train, evals, {"Emily": 0}, 5)


# --------------------------------------------------------------------------- planted worlds -----------
def _pm(cell, factor, design=D):
    return 1.0 if cell[design.axes.index(factor)] == design.factors[factor][0] else -1.0


def _world(states, features, seed, noise=0.3, dim=8, design=D):
    """States whose first dimensions are ``features(f1, f2, f3, content)`` (±1 codes of the cell's factors and of a
    per-record content sign), padded with noise dimensions."""
    rng = np.random.default_rng(seed)
    content = {r: float(rng.choice([-1.0, 1.0])) for r in dict.fromkeys(states.records)}
    rows = [features(*(_pm(c, f, design) for f in design.axes), content[r])
            for c, r in zip(states.cells, states.records)]
    X = np.zeros((len(rows), dim))
    X[:, :len(rows[0])] = rows
    return torch.tensor(X + rng.normal(0, noise, X.shape), dtype=torch.float32)


HEAD = np.array([1.0, 0.5, 0.2, 0, 0, 0, 0, 0])


def _run(features, encoding="explicit", domain="credit", n_train=250, n_eval=80):
    design = DOMAINS[domain]
    train, evals = _states(n_train, "p", encoding, domain=domain), _states(n_eval, "e", encoding, seed=1, domain=domain)
    folds = rde.name_folds(domain, 5, seed=42) if encoding == "proxy" else None
    out = rde.erasure_test(_world(train, features, 0, design=design), _world(evals, features, 1, design=design),
                           train, evals, design, encoding, folds, 5, seed=0, n_boot=200,
                           score=lambda H, idx: H.numpy() @ HEAD)
    return {r["concept"]: r for r in out}


def test_a_linearly_encoded_attribute_reads_low_complexity_and_loses_its_reward_gap():
    r = _run(lambda s, a, m, c: [s])["sex"]
    assert r["verdict"] == "not recoverable after LEACE (low-complexity)" and r["read_row"] == "leace"
    assert r["rows"]["none"]["linear_acc"] > 0.95
    gap = r["reward_gap"]
    assert gap["none"]["estimate"] == pytest.approx(2.0, abs=0.1)          # head weight 1 × (+1 − −1)
    for m in ("diffmean", "leace", "leace_cells"):
        assert abs(gap[m]["estimate"]) < 0.1 and gap[f"{m}_minus_none"]["ci_high"] < -1.5
    # the words "woman"/"man" are the label: LEACE leaves them constant up to rounding, and they are snapped
    assert r["snapped_features"]["lexical"]["leace"] >= 2 and r["snapped_features"]["lexical"]["none"] == 0
    assert r["snapped_features"]["model"]["none"] == 0
    # the explicit marker is one word per pole: the words alone are at chance after LEACE
    lex = r["lexical_control"]
    assert lex["none"]["linear_acc"] == 1.0 and lex["leace"]["intervals"]["mlp_above_chance"]["ci_low"] <= 0
    assert r["lexical_recovers_in_read_row"] is False


def test_an_attribute_carried_by_its_interaction_reads_intersectional():
    # sex also lives in the sex×age product: after LEACE on sex alone an MLP rebuilds it from age and the product;
    # erasing every cell removes that route — a linear interaction term, not a high-complexity code
    out = _run(lambda s, a, m, c: [s, a, s * a])
    r = out["sex"]
    assert r["verdict"] == ("recoverable only through a linear interaction with another factor (intersectional, "
                            "not high-complexity)")
    assert r["rows"]["leace"]["intervals"]["mlp_above_chance"]["ci_low"] > 0.2
    assert r["rows"]["leace"]["intervals"]["linear_above_chance"]["ci_high"] < 0.1
    assert r["rows"]["leace_cells"]["mlp_acc"] < 0.6
    # the product itself is linearly decodable as the interaction concept, and nothing of it is left once every cell
    # is erased; its own LEACE row is not what the verdict reads (an MLP rebuilds the XNOR from the main effects)
    inter = out["sex_x_age"]
    assert inter["rows"]["none"]["linear_acc"] > 0.95 and inter["rows"]["leace"]["mlp_acc"] > 0.9
    assert inter["verdict"] == "not recoverable after LEACE on every cell (low-complexity)"
    assert inter["read_row"] == "leace_cells"


def test_without_a_decodable_interaction_the_cell_erasure_reading_is_unresolved():
    # recovered after LEACE on the axis, gone after LEACE on every cell: "intersectional" only where an interaction of
    # that axis is itself linearly decodable — otherwise the cell erasure may only have cost power
    rows = {"none": {"intervals": {"linear_above_chance": {"ci_low": 0.3}, "mlp_above_chance": {"ci_low": 0.3}}},
            "leace": {"intervals": {"mlp_above_chance": {"ci_low": 0.2}}},
            "leace_cells": {"intervals": {"mlp_above_chance": {"ci_low": -0.01}}}}
    flat = {"none": {"intervals": {"linear_above_chance": {"ci_low": -0.02}, "mlp_above_chance": {"ci_low": -0.02}}}}
    todo = rde.concepts(D, "explicit")
    pooled = {c: (rows if c == "sex" else flat) for c, _ in todo}
    assert rde.concept_verdict("sex", ("sex",), pooled, todo).endswith("unresolved")
    pooled["sex_x_marital_status"] = rows
    assert "intersectional" in rde.concept_verdict("sex", ("sex",), pooled, todo)
    # an interaction unrelated to sex does not corroborate it
    pooled["sex_x_marital_status"], pooled["age_x_marital_status"] = flat, rows
    assert rde.concept_verdict("sex", ("sex",), pooled, todo).endswith("unresolved")


def test_the_words_alone_rebuild_an_interaction_from_its_main_effects():
    # the lexical control of the explicit clauses: no word is the sex × marital interaction, yet an MLP computes it
    # from "woman"/"man" and "married"/"single" after LEACE on the interaction alone — why the verdict reads leace_cells
    out = _run(lambda s, a, m, c: [s])
    lex = out["sex_x_marital_status"]["lexical_control"]
    assert lex["leace"]["mlp_acc"] > 0.95 and lex["leace_cells"]["mlp_acc"] < 0.6
    # digits are words: the explicit age ("30-year-old" vs "50-year-old") is visible to the control
    assert out["age"]["lexical_control"]["none"]["linear_acc"] == 1.0


@pytest.mark.parametrize("domain", ["cv", "education"])
@pytest.mark.parametrize("encoding", ["explicit", "proxy"])
def test_the_words_alone_are_at_chance_where_the_verdicts_read(domain, encoding):
    # the review's reproduction (150 + 60 records): words in every clause ("is", "old") got a label-correlated
    # rounding residue from LEACE, which standardising turned into a separator (sex 1.0 after leace and leace_cells)
    out = _run(lambda s, a, m, c: [s], encoding=encoding, domain=domain, n_train=150, n_eval=60)
    for c, r in out.items():
        lex = r["lexical_control"]
        assert lex["leace_cells"]["intervals"]["mlp_above_chance"]["ci_low"] <= 0, (c, "leace_cells")
        assert lex["leace_cells"]["intervals"]["linear_above_chance"]["ci_low"] <= 0, (c, "leace_cells linear")
        if len(r["factors"]) == 1:
            assert lex["leace"]["intervals"]["mlp_above_chance"]["ci_low"] <= 0, (c, "leace")
        assert r["lexical_recovers_in_read_row"] is False, c


def test_an_attribute_carried_beyond_the_cells_survives_the_cell_erasure():
    # sex also lives in its product with the record's content: no function of the cell removes that
    r = _run(lambda s, a, m, c: [s, c, s * c])["sex"]
    assert r["verdict"] == "non-linearly recoverable beyond the cell structure (entangled, high-complexity)"


def test_proxy_rows_pool_the_name_folds_and_resample_names():
    out = _run(lambda s, a, m, c: [s], encoding="proxy")
    assert [r["concept"] for r in out.values()] == ["sex", "age", "sex_x_age"]
    r = out["sex"]
    assert r["folds"] == 5 and len(r["rows"]["none"]["by_fold"]) == 5
    iv = r["rows"]["none"]["intervals"]["linear_acc"]
    assert iv["n_items"] == r["n_eval"] == 80 * 8 and iv["n_clusters"] == 80 and iv["n_clusters_b"] == 20
    by_name = r["rows"]["none"]["by_name"]
    assert len(by_name) == 20 and sum(v["n"] for v in by_name.values()) == 80 * 8
    # the names the lexical control was not trained on carry nothing: at chance even without erasure
    assert r["lexical_control"]["none"]["intervals"]["linear_above_chance"]["ci_low"] <= 0


def test_the_proxy_reward_gap_compares_states_under_one_erasure():
    # the review's failure: a large common mean and name-specific components; per-fold fits scored a record's female
    # and male cells under different erasures, and the uncentered projection's fold-to-fold differences in d times
    # the mean did not cancel (diffmean gap 0.20 [−0.74, 1.17] where 0 is right)
    rng = np.random.default_rng(3)
    dim = 64
    mean = rng.normal(0, 1, dim)
    mean *= 400 / np.linalg.norm(mean)
    name_vec = {n: rng.normal(0, 1.0, dim) for pool in rde.NAME_POOLS["credit"] for n in pool}
    head = np.zeros(dim)
    head[0] = 0.5

    def states(st, seed):
        r = np.random.default_rng(seed)
        X = np.stack([mean + _pm(c, "sex") * np.eye(dim)[0] + name_vec[n] + r.normal(0, 0.5, dim)
                      for c, n in zip(st.cells, st.names)])
        return torch.tensor(X, dtype=torch.float32)

    train, evals = _states(250, "p", "proxy"), _states(80, "e", "proxy", seed=1)
    out = rde.erasure_test(states(train, 0), states(evals, 1), train, evals, D, "proxy",
                           rde.name_folds("credit", 5, seed=42), 5, seed=0, n_boot=200,
                           score=lambda H, idx: H.double().numpy() @ head)
    sex = {r["concept"]: r["reward_gap"] for r in out}["sex"]
    # 1.0 from the sex dimension, plus the drawn names' own head components (a record's female and male name differ)
    assert sex["none"]["ci_low"] > 0.5
    for m in ("diffmean", "leace"):
        # gone, and as precise as the names allow (per-fold fits gave an interval ~1.9 wide)
        assert abs(sex[m]["estimate"]) < 0.1 and sex[m]["ci_low"] < 0 < sex[m]["ci_high"], (m, sex[m])
        assert sex[m]["ci_high"] - sex[m]["ci_low"] < 0.5, (m, sex[m])


def test_reward_gap_per_record():
    labels = [1, 0, 1, 0, 1, 0]
    records = ["a", "a", "a", "a", "b", "b"]
    rewards = {"none": np.array([3.0, 1.0, 5.0, 1.0, 2.0, 2.0]), "leace": np.array([1.0, 1.0, 1.0, 1.0, 0, 0.5])}
    g = rde.reward_gap(rewards, labels, records, n_boot=50, seed=0)
    # a: (3+5)/2 − 1 = 3; b: 0 → mean 1.5; after LEACE a: 0, b: −0.5
    assert g["none"]["estimate"] == 1.5 and g["leace"]["estimate"] == -0.25
    assert g["leace_minus_none"]["estimate"] == -1.75 and g["none"]["n_clusters"] == 2
    with pytest.raises(ValueError, match="one label only"):
        rde.reward_gap(rewards, [1, 1, 1, 1, 1, 0], records, n_boot=10, seed=0)


# --------------------------------------------------------------------------- the manifest -------------
@pytest.mark.parametrize("encoding", ["explicit", "proxy"])
def test_the_states_are_the_battery_texts(manifest, encoding):
    from pairs.cross_marker import load_cell_blocks
    from runners.run_cross_marker import cells_path
    from substrates.domains import get_domain
    from tests.test_run_cross_marker import _tokenizer

    dom, tok = get_domain("credit"), _tokenizer()
    blocks = load_cell_blocks(cells_path(str(manifest), dom.default_pairs), D)
    ids = rde.probe_split_ids(dom, str(manifest), 6, 42)
    train, _, _ = rde.select_records(blocks, [encoding], ids, 4, seed=42)
    states = rde.States.from_blocks(train, encoding, D, dom.dataset_cls.DEFAULT_PROMPT, tok)
    ds = dom.dataset_cls(str(manifest), axis="sex", encoding=encoding, probe_records=6, split_seed=42)
    pairs = ds.get_probe_pairs(tok)
    flat = lambda t: json.dumps(t)
    assert sorted(map(flat, states.texts)) == sorted(flat(t) for p in pairs for t in (p.positive_text,
                                                                                      p.negative_text))
    # every probe pair's side A is a label-1 state of the sex concept
    label = {flat(t): rde.concept_label(D, c, ("sex",)) for t, c in zip(states.texts, states.cells)}
    assert {label[flat(p.positive_text)] for p in pairs} == {1} and {label[flat(p.negative_text)] for p in pairs} == {0}


def _runner(pairs, domain, tmp_path, monkeypatch):
    """`main` on a fixture manifest with the tiny model standing in for the loader."""
    from scoring.demographic_experiment import DemographicBiasExperiment
    from tests.test_run_cross_marker import _model, _tokenizer

    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(pairs),
        "probe_records": 6, "batch_size": 16, "max_length": 1024, "extra": {"domain": domain, "n_boot": 20}}))
    loads = []

    def load_model(self):
        loads.append(1)
        self.model, self.tokenizer = _model(), _tokenizer()
        self.config.model_revision = "abc123"

    monkeypatch.setattr(DemographicBiasExperiment, "load_model", load_model)

    def go(*argv):
        monkeypatch.setattr("sys.argv", ["run_demographic_erasure.py", "--config", str(cfg_path), *argv])
        rde.main()
        return sorted((tmp_path / rde.RESULTS_DIR).glob("*.json"))

    go.loads = loads
    return go


@pytest.fixture
def run(manifest, tmp_path, monkeypatch):
    """`main` on the credit fixture manifest (16 records)."""
    return _runner(manifest, "credit", tmp_path, monkeypatch)


def test_end_to_end_on_the_fixture_manifest(run):
    (path,) = run("--eval-records", "4", "--name-folds", "2")
    # the manifest's folder is the test's tmp dir, so the name carries it (as the battery's names do)
    assert path.name.startswith("erasure_demographic_credit_")
    assert path.name.endswith("_Tiny-RM__eval_records-4__name_folds-2.json")
    result = json.loads(path.read_text())
    assert result["meta"]["config"]["model_revision"] == "abc123"
    assert set(result["meta"]["data"]) == {"pairs.jsonl", "cells.jsonl"}
    assert len(result["probe_records"]) == 6 and len(result["eval_records"]) == 4
    assert not set(result["probe_records"]) & set(result["eval_records"])
    # the training records are the battery's probe split, per encoding
    assert {e: v["identical"] for e, v in result["selection"]["battery_split"].items()} == \
        {"explicit": True, "proxy": True}
    assert [(r["encoding"], r["concept"]) for r in result["results"]] == \
        [("explicit", c) for c, _ in rde.concepts(D, "explicit")] + [("proxy", c) for c, _ in rde.concepts(D, "proxy")]
    r = result["results"][0]
    assert set(r["rows"]) == set(r["lexical_control"]) == set(rde.METHODS)
    assert set(r["reward_gap"]) == set(rde.METHODS) | {f"{m}_minus_none" for m in rde.METHODS[1:]}
    assert r["n_eval"] == 4 * 2 * 8 and r["n_train_by_fold"] == [6 * 2 * 8] and r["folds"] == 1
    assert "n_clusters_b" not in r["rows"]["none"]["intervals"]["mlp_acc"]          # explicit: records only
    proxy = result["results"][-1]
    assert proxy["folds"] == 2 and sum(proxy["n_train_by_fold"]) == 6 * 2 * 8   # each name trains in one fold of 2
    assert "n_clusters_b" in proxy["rows"]["none"]["intervals"]["mlp_acc"] and proxy["rows"]["none"]["by_name"]
    assert "Emily" in result["name_folds"]
    # never replaced without --overwrite; the encodings' order does not change the name
    with pytest.raises(SystemExit, match="exists"):
        run("--eval-records", "4", "--name-folds", "2", "--encodings", "proxy,explicit")
    assert len(run.loads) == 1


def test_education_end_to_end_with_its_name_grid(education_corpus, tmp_path, monkeypatch):
    go = _runner(education_corpus[0], "education", tmp_path, monkeypatch)
    (path,) = go("--eval-records", "4", "--name-folds", "2", "--encodings", "proxy")
    assert path.name.endswith("_Tiny-RM__encodings-proxy__eval_records-4__name_folds-2.json")
    result = json.loads(path.read_text())
    assert [r["concept"] for r in result["results"]] == [c for c, _ in rde.concepts(EDUCATION_DESIGN, "proxy")]
    names = set(result["results"][0]["rows"]["none"]["by_name"])
    pools = rde.NAME_POOLS["education"]
    assert names <= {n for pool in pools for n in pool}
    assert any(n in pools[2] + pools[3] for n in names)                   # the Black-coded pools are in play


def test_bad_inputs_are_refused_before_the_model_loads(run, monkeypatch):
    with pytest.raises(SystemExit, match="--encodings"):
        run("--encodings", "explicit,implicit")
    with pytest.raises(SystemExit, match="at least 2"):
        run("--name-folds", "1")
    # a stale name list fails before the model loads, not after it
    monkeypatch.setitem(rde.NAME_POOLS, "credit", (("Emily",), tuple(rde.NAME_POOLS["credit"][1])))
    with pytest.raises(SystemExit, match="no name pool"):
        run("--eval-records", "4", "--name-folds", "2")
    assert run.loads == []
