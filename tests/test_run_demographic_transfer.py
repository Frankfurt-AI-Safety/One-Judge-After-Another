"""`runners/run_demographic_transfer.py`: the shared name folds, the relations and row kinds, the contrast units and
hold-out keys on generated manifests (the fit pairs are the battery's, the evaluated texts the cross-marker arm's under
every setting; no row is nulled by a direction that saw its names, checked against the scored texts), planted worlds
whose reading is known (a direction shared by two domains transfers on both targets; orthogonal ones do not; a
name-identity code is removed only by a direction that saw the names), hand recomputations of proxy cells, the reward
cache, a gated head, the paired accuracy and its lexical controls, and `main` end to end on three domains with the
tiny model, against the cross-marker arm's own numbers."""

from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("sklearn")
pytest.importorskip("concept_erasure")

from pairs.factorial import CREDIT_DESIGN, EDUCATION_DESIGN, HIRING_DESIGN  # noqa: E402
from pairs.markers import AGE_YOUNG, FEMALE_NAMES  # noqa: E402
from probes.heads import get_head  # noqa: E402
from probes.transfer_directions import UnitStore, mean_unit, random_units  # noqa: E402
from runners import run_cross_marker as rx  # noqa: E402
from runners import run_demographic_transfer as rdt  # noqa: E402
from scoring.dataset_base import format_conversation  # noqa: E402
from scoring.experiment import ExperimentConfig  # noqa: E402
from scoring.placement_matrix import shared_draws  # noqa: E402
from tests.test_run_cross_marker import _model as _tiny  # noqa: E402
from tests.test_run_cross_marker import _tokenizer  # noqa: E402

DESIGNS = {"credit": CREDIT_DESIGN, "cv": HIRING_DESIGN, "education": EDUCATION_DESIGN}
FOLDS = rdt.shared_name_folds(5, 42)
F = frozenset
name = rdt.source_name


# --------------------------------------------------------------------------- names -------------------
def test_name_folds_are_the_same_in_every_domain_and_even_per_pool():
    assert set(FOLDS) == {n for pools in rdt.NAME_POOLS.values() for pool in pools for n in pool}
    for pools in rdt.NAME_POOLS.values():
        for pool in pools:
            assert sorted(sum(FOLDS[n] == f for n in pool) for f in range(5)) == [2] * 5
    assert FOLDS == rdt.shared_name_folds(5, 42) != rdt.shared_name_folds(5, 43)


def test_fold_set():
    assert rdt.fold_set(("a", None, "b"), {"a": 1, "b": 3}) == F({1, 3})
    assert rdt.fold_set((None,), {}) == F()


# --------------------------------------------------------------------------- relations ---------------
def test_axes_per_encoding():
    assert rdt.encoding_axes(CREDIT_DESIGN, "explicit") == ["sex", "age", "marital_status", "intersection"]
    assert rdt.encoding_axes(CREDIT_DESIGN, "proxy") == ["sex", "age", "intersection"]      # no marital proxy


def test_relations():
    rel = lambda s, t: rdt.relation(s, t, DESIGNS)
    t = ("credit", "explicit", "sex")
    assert rel(t, t) == {"relation": "own"}
    assert rel(("credit", "proxy", "sex"), t) == {"relation": "encoding", "proxy_construct_differs": False}
    assert rel(("cv", "explicit", "sex"), t) == {"relation": "domain"}
    assert rel(("education", "proxy", "sex"), t) == {"relation": "domain+encoding", "proxy_construct_differs": False}
    assert rel(("credit", "explicit", "age"), t) == {"relation": "off_axis"}
    assert rel(("cv", "explicit", "age"), t) == {"relation": "unrelated"}
    # age has the same poles in credit and hiring; education has no age
    assert rel(("cv", "explicit", "age"), ("credit", "explicit", "age")) == {"relation": "domain"}
    assert rel(("education", "explicit", "ethnicity"), ("credit", "explicit", "age")) == {"relation": "unrelated"}
    # a proxy that measures a related but different attribute is flagged, whichever side is the proxy
    assert rel(("cv", "proxy", "family_status"), ("cv", "explicit", "family_status")) == {
        "relation": "encoding", "proxy_construct_differs": True}
    assert rel(("education", "explicit", "economic_status"),
               ("education", "proxy", "economic_status"))["proxy_construct_differs"] is True


def test_an_intersection_is_a_component_of_what_shares_a_factor_with_it():
    rel = lambda s, t: rdt.relation(s, t, DESIGNS)
    corner, sex = ("credit", "explicit", "intersection"), ("credit", "explicit", "sex")
    # the corner flips the axis: neither side is a specificity control of the other
    assert rel(corner, sex) == rel(sex, corner) == {"relation": "component"}
    assert rel(("credit", "proxy", "intersection"), sex) == {"relation": "component+encoding",
                                                             "proxy_construct_differs": False}
    assert rel(("cv", "explicit", "intersection"), sex) == {"relation": "component+domain"}
    assert rel(("cv", "proxy", "intersection"), ("credit", "explicit", "age"))["relation"] == \
        "component+domain+encoding"
    # two domains' corners share sex (credit and hiring also age), never every factor
    assert rel(("cv", "explicit", "intersection"), corner) == {"relation": "component+domain"}
    assert rel(("education", "explicit", "intersection"), corner) == {"relation": "component+domain"}
    assert rel(("education", "explicit", "intersection"), ("credit", "explicit", "age")) == {"relation": "unrelated"}
    assert rel(("credit", "proxy", "intersection"), corner) == {"relation": "encoding",
                                                                "proxy_construct_differs": False}
    # the flag reads the shared factors: a corner with a re-labelled proxy factor, but not through sex alone
    for domain in ("cv", "education"):
        assert rel((domain, "proxy", "intersection"), (domain, "explicit", "intersection")) == {
            "relation": "encoding", "proxy_construct_differs": True}
        assert rel((domain, "proxy", "sex"), (domain, "explicit", "intersection")) == {
            "relation": "component+encoding", "proxy_construct_differs": False}
    assert rel(("cv", "proxy", "family_status"), ("cv", "explicit", "intersection")) == {
        "relation": "component+encoding", "proxy_construct_differs": True}
    # over every pair: the controls share no factor, and every label is a known one
    sources = [(d, e, a) for d, design in DESIGNS.items() for e in ("explicit", "proxy")
               for a in rdt.encoding_axes(design, e)]
    labels = Counter(rel(s, t)["relation"] for s in sources for t in sources)
    scopes = ("", "+encoding", "+domain", "+domain+encoding")
    assert set(labels) == {"own", "encoding", "domain", "domain+encoding", "off_axis", "unrelated",
                           *("component" + x for x in scopes)}
    assert labels["own"] == 23 and sum(labels.values()) == 23 * 23
    for s_ in sources:
        for t_ in sources:
            if rel(s_, t_)["relation"] in ("off_axis", "unrelated"):
                assert not rdt.flips(s_, DESIGNS) & rdt.flips(t_, DESIGNS)
                assert "intersection" not in (s_[2], t_[2]) or s_[0] != t_[0]


def _fake_stores(templates=("t1", "t2")):
    stores = {}
    for d, design in DESIGNS.items():
        for e in ("explicit", "proxy"):
            for a in rdt.encoding_axes(design, e):
                g = torch.Generator().manual_seed(len(stores))
                folds = [F({i % 5}) if e == "proxy" else F() for i in range(10)]
                stores[(d, e, a)] = UnitStore(torch.randn(10, 6, generator=g), folds,
                                              [templates[i % len(templates)] for i in range(10)],
                                              [f"r{i}" for i in range(10)])
    return stores


def test_row_kinds():
    stores = _fake_stores()
    sources = [name(s) for s in stores]
    assert len(sources) == 23
    extra = lambda t: [r for r in rdt.row_kinds(t, stores, DESIGNS, 2) if r not in sources]
    assert list(rdt.row_kinds(("credit", "explicit", "sex"), stores, DESIGNS, 0)) == sources + ["template", "others"]
    # explicit: no own_seen; sex has two other domains, age one (its row is the `domain` row), ethnicity none
    assert extra(("credit", "explicit", "sex")) == ["template", "others", "random0", "random1"]
    assert extra(("credit", "proxy", "age")) == ["own_seen", "template", "random0", "random1"]
    assert extra(("education", "explicit", "ethnicity")) == ["template", "random0", "random1"]
    assert extra(("cv", "proxy", "intersection")) == ["own_seen", "template", "singles_joint", "random0", "random1"]
    # one template in the fit: no held-out-template row
    assert "template" not in rdt.row_kinds(("credit", "explicit", "sex"), _fake_stores(("t1",)), DESIGNS, 0)


def test_row_basis():
    stores = _fake_stores()
    random = random_units(6, 2, 0)
    t, key = ("credit", "proxy", "sex"), (F({0}), "t1")
    basis = lambda kind: rdt.row_basis(kind, key, stores, DESIGNS, random)
    assert torch.equal(basis(("source", t)), stores[t].direction(F({0})))
    assert not torch.equal(basis(("source", t)), stores[t].direction())
    assert torch.equal(basis(("own_seen", t)), stores[t].direction())
    assert torch.equal(basis(("template", t)), stores[t].direction(F({0}), "t1"))
    assert torch.equal(basis(("others", t)), mean_unit([stores[("cv", "proxy", "sex")].direction(F({0})),
                                                         stores[("education", "proxy", "sex")].direction(F({0}))]))
    # the joint covers every factor of the corner: credit's marital status has no proxy, so the explicit direction
    corner = ("credit", "proxy", "intersection")
    members = [("credit", "proxy", "sex"), ("credit", "proxy", "age"), ("credit", "explicit", "marital_status")]
    assert rdt.single_axes(corner, stores, DESIGNS) == members
    assert torch.equal(basis(("singles_joint", corner)), torch.stack([stores[m].direction(F({0})) for m in members]))
    assert rdt.single_axes(("cv", "proxy", "intersection"), stores, DESIGNS) == [
        ("cv", "proxy", a) for a in ("sex", "age", "family_status")]
    assert rdt.single_axes(corner, [m for m in stores if m[1] == "proxy"], DESIGNS) == members[:2]   # none: left out
    assert torch.equal(basis(("random", 1)), random[1])
    assert rdt.row_basis(("random", 1), key, stores, DESIGNS) is None        # the lexical space has no random row
    with pytest.raises(ValueError, match="unknown row kind"):
        basis(("leace", t))


def test_row_fit_units_are_the_pairs_the_rows_direction_rests_on():
    stores = _fake_stores()             # 10 pairs per source; a proxy pair sits in fold i % 5, templates alternate
    t = ("credit", "proxy", "sex")
    keys = [(F({0}), "t1"), (F({0, 1}), "t2"), (F({0}), "t1")]
    n = lambda kind: rdt.row_fit_units(kind, keys, stores, DESIGNS)
    assert n(("source", t)) == {"min": 6, "max": 8}                          # without two folds, without one
    assert n(("source", ("credit", "explicit", "sex"))) == {"min": 10, "max": 10}     # no name: every pair
    assert n(("own_seen", t)) == {"min": 10, "max": 10}
    assert n(("template", t)) == {"min": 3, "max": 4}                        # and without the row's template
    assert n(("others", t)) == {"min": 6, "max": 8}
    assert n(("singles_joint", ("credit", "proxy", "intersection"))) == {"min": 6, "max": 8}   # its smallest fit
    assert n(("random", 0)) is None


def test_min_fit_units():
    units = lambda folds, templates: rdt.UnitIndex([0] * len(folds), [1] * len(folds), folds, templates,
                                                   [str(i) for i in range(len(folds))])
    assert rdt.min_fit_units(units([F()] * 6, ["t1", "t2"] * 3), 5) == 3          # no names: a template left out
    assert rdt.min_fit_units(units([F()] * 4, ["t1"] * 4), 5) == 4                # one template: none is left out
    assert rdt.min_fit_units(units([F({0, 1})] * 4, ["t1"] * 4), 5) == 0          # every pair shares fold 0
    mixed = units([F({0, 1}), F({0, 1}), F({2}), F({2, 3}), F({4}), F({4})], ["t1", "t2"] * 3)
    assert rdt.min_fit_units(mixed, 5) == 1                                       # e.g. without folds 2, 4 and t1


# --------------------------------------------------------------------------- domains on manifests ----
def _cfg(pairs, domain, probe=20, **cross_marker):
    return ExperimentConfig.from_dict({
        "name": domain, "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(pairs),
        "probe_records": probe, "batch_size": 16, "max_length": 1024,
        "extra": {"domain": domain, "cross_marker": {"n_strong": 5, "n_weak": 5, "n_boot": 50, "paraphrases": 1,
                                                      **cross_marker}}})


@pytest.fixture
def credit(credit_corpus):
    return rdt.Domain(_cfg(credit_corpus[0], "credit"), {}, FOLDS, 5)


@pytest.fixture
def cv(hiring):
    return rdt.Domain(_cfg(hiring[0], "cv", probe=30), {}, FOLDS, 5)


@pytest.fixture
def domains(credit, cv, education_corpus):
    return {"credit": credit, "cv": cv,
            "education": rdt.Domain(_cfg(education_corpus[0], "education"), {}, FOLDS, 5)}


CORPORA = (("credit", "credit_corpus", 20), ("cv", "hiring", 30), ("education", "education_corpus", 20))


def _fmt(tok):
    return lambda prompt, response: format_conversation(tok, prompt, response)


def test_units_are_the_axis_pairs_of_every_block(credit):
    table = credit.fit_tables["proxy"]
    units = credit.fit_units[("credit", "proxy", "sex")]
    cells = list(CREDIT_DESIGN.cells)
    assert len(units.a) == 4 * len(table.blocks) > 0
    for a, b, folds, template, record in zip(units.a, units.b, units.folds, units.templates, units.records):
        j = a // 8
        assert b // 8 == j and (cells[a % 8], cells[b % 8]) in CREDIT_DESIGN.axis_pairs("sex", "proxy")
        names = {table.names[j][cells[a % 8]], table.names[j][cells[b % 8]]}
        assert len(names) == 2 and folds == F(FOLDS[n] for n in names)         # the sex pair swaps the name
        assert (template, record) == (table.blocks[j].template_id, table.blocks[j].record_id)
    # an age pair keeps its name; an explicit pair has none; the corner pair is one per block
    assert all(len(f) == 1 for f in credit.fit_units[("credit", "proxy", "age")].folds)
    assert not any(credit.fit_units[("credit", "explicit", "sex")].folds)
    assert len(credit.fit_units[("credit", "proxy", "intersection")].a) == len(table.blocks)
    V = torch.arange(len(table.blocks) * 16, dtype=torch.float32).reshape(-1, 2)
    assert torch.equal(units.diffs(V), V[units.a] - V[units.b])


def test_the_fit_units_are_the_batterys_ordered_probe_pairs(domains):
    """Every source: the same (pole A text, pole B text) pairs as the battery's dataset gives, so the direction and its
    sign are the battery's."""
    tok = _tokenizer()
    for d in domains.values():
        assert all(report["identical"] for report in d.fit_report["battery_split"].values())
        assert set(d.fit_report) == {"records_in_cells", "probe_records", "probe_records_without_blocks", "templates",
                                     "battery_split"}
        for encoding, table in d.fit_tables.items():
            convs = table.direct_convs(d.design, d.prompt, tok)
            for axis in rdt.encoding_axes(d.design, encoding):
                ds = d.dom.dataset_cls(d.source, axis=axis, encoding=encoding, probe_records=d.cfg.probe_records,
                                       split_seed=42)
                assert {b.record_id for b in table.blocks} == ds.probe_record_ids() == d.probe_ids
                battery = Counter((p.positive_text, p.negative_text) for p in ds.get_probe_pairs(tok))
                units = d.fit_units[(d.name, encoding, axis)]
                ours = Counter((convs[a], convs[b]) for a, b in zip(units.a, units.b))
                assert ours == battery and all(a != b for a, b in ours), (d.name, encoding, axis)


def test_the_full_direction_is_the_batterys(credit):
    from probes.probe import build_probe_direction, embed_states

    tok, model = _tokenizer(), _tiny()
    ds = credit.dom.dataset_cls(credit.source, axis="sex", encoding="explicit", probe_records=20, split_seed=42)
    battery, meta = build_probe_direction(model, tok, ds.get_probe_pairs(tok), batch_size=16, max_length=1024)
    H, _ = embed_states(model, tok, credit.fit_tables["explicit"].direct_convs(CREDIT_DESIGN, credit.prompt, tok),
                        batch_size=16, max_length=1024, show_progress=False)
    store = credit.fit_units[("credit", "explicit", "sex")].store(H)
    assert meta["probe_raw_norm"] > 1e-3                 # the tiny tokenizer sees "woman" / "man": not two zero vectors
    assert torch.allclose(store.direction(), battery.float(), atol=1e-4)
    assert store.separation == pytest.approx(meta["probe_raw_norm"], rel=1e-3)


@pytest.mark.parametrize("variant", [
    dict(paraphrases=1), dict(paraphrases=3), dict(paraphrases=3, seed=7),
    dict(paraphrases=2, include_unmarked=False), dict(paraphrases=3, encodings=["explicit"]),
    dict(paraphrases=3, encodings=["proxy"])])
def test_the_evaluated_records_and_texts_are_the_cross_marker_arms(request, variant):
    """Under every setting the arm reads: the same probe exclusion, records, decision conversations (paraphrase index,
    unmarked control) and direct-placement conversations as `run_cross_marker.main` builds."""
    tok = _tokenizer()
    fmt = _fmt(tok)
    for name, fixture, probe in CORPORA:
        d = rdt.Domain(_cfg(request.getfixturevalue(fixture)[0], name, probe=probe, **variant), {}, FOLDS, 5)
        d.select(tok)
        settings = rx.resolve_settings(d.cfg.extra, {})
        arm_probe = set()
        for encoding in settings["encodings"]:
            for axis in d.dom.axes:
                if d.design.axis_pairs(axis, encoding):
                    arm_probe |= d.dom.dataset_cls(d.source, axis=axis, encoding=encoding, probe_records=probe,
                                                   split_seed=42).probe_record_ids()
        templates = settings["templates"] or sorted({b.template_id for b in d.blocks})
        selected, _ = rx.select_records(
            d.blocks, quality_field=d.dom.quality_field, encodings=settings["encodings"], templates=templates,
            exclude=arm_probe, n_strong=settings["n_strong"], n_weak=settings["n_weak"], seed=settings["seed"],
            fits=rx.block_fits(name, settings, fmt, rx.token_counter(tok), d.cfg.max_length))
        assert arm_probe == d.probe_ids and list(selected) == list(d.selected) and selected
        assert not set(selected) & d.probe_ids
        key = lambda r: (r["record_id"], r["encoding"], r["template_id"], str(r["cell"]), r.get("response"),
                         r.get("paraphrase"))
        arm = {key(r): c for r, c in zip(*rx.build_rows(selected, name, d.dom.quality_field, fmt, settings))}
        arm_direct = {key(r): c for r, c in zip(*rx.build_direct_rows(selected, d.design, d.dom.quality_field,
                                                                     d.dom.assessment_prompt, fmt))}
        for encoding in d.encodings:
            table = rdt.BlockTable.of(d.selected, encoding, d.design)
            rows, convs, cells, block_of = rdt.decision_rows(table, name, d.dom.quality_field, d.settings, fmt)
            n_cells = 8 + bool(settings["include_unmarked"])
            assert len(rows) == len(table.blocks) * n_cells * 2 == len(convs) == len(cells) == len(block_of)
            assert all(c == arm[key(r)] for r, c in zip(rows, convs)), (name, encoding)
            assert not any("text" in r or "prompt" in r for r in rows)
            d_rows, d_cells, d_block_of = rdt.direct_rows(table, d.design, d.dom.quality_field)
            ours = table.direct_convs(d.design, d.prompt, tok)
            assert len(d_rows) == len(ours) == len(table.blocks) * 8
            for r, cell, j, conv in zip(d_rows, d_cells, d_block_of, ours):
                assert conv == fmt(d.prompt, table.blocks[j].texts[cell]) == arm_direct[(*key(r)[:4], None, None)]
                assert r["cell"] == list(cell) and r["record_id"] == table.blocks[j].record_id


def test_row_keys_hold_out_the_names_of_the_rows_pair(credit):
    tok = _tokenizer()
    credit.select(tok)
    table = rdt.BlockTable.of(credit.selected, "proxy", CREDIT_DESIGN)
    _, _, cells, block_of = rdt.decision_rows(table, "credit", "credit_good", credit.settings, _fmt(tok))
    fold = lambda j, cell: FOLDS[table.names[j][cell]]
    keys = rdt.row_keys(cells, block_of, table, CREDIT_DESIGN, "proxy", "sex", FOLDS)
    assert any(cell is None for cell in cells)
    for cell, j, (folds, template) in zip(cells, block_of, keys):
        assert template == table.blocks[j].template_id
        if cell is None:
            assert folds == F()                                   # the unmarked control carries no name
        else:
            partner = ("male" if cell[0] == "female" else "female",) + cell[1:]
            assert folds == F({fold(j, cell), fold(j, partner)})
    # the age pair shares one name; the intersection reads the two corners only
    keys = rdt.row_keys(cells, block_of, table, CREDIT_DESIGN, "proxy", "age", FOLDS)
    assert all(folds == F({fold(j, cell)}) for cell, j, (folds, _) in zip(cells, block_of, keys) if cell is not None)
    (a, b), = CREDIT_DESIGN.axis_pairs("intersection", "proxy")
    keys = rdt.row_keys(cells, block_of, table, CREDIT_DESIGN, "proxy", "intersection", FOLDS)
    for cell, j, (folds, _) in zip(cells, block_of, keys):
        if cell in (a, b):
            assert folds == F({fold(j, a), fold(j, b)})
        elif cell is not None:
            assert folds == F({fold(j, cell)})
    # explicit rows carry no name
    table = rdt.BlockTable.of(credit.selected, "explicit", CREDIT_DESIGN)
    _, cells, block_of = rdt.direct_rows(table, CREDIT_DESIGN, "credit_good")
    assert not any(f for f, _ in rdt.row_keys(cells, block_of, table, CREDIT_DESIGN, "explicit", "sex", FOLDS))


def _names_in(text):
    return {n for n in FOLDS if re.search(rf"\b{re.escape(n)}\b", str(text))}


def test_no_row_is_nulled_by_a_fit_that_saw_a_name_of_its_pair(domains):
    """Against the scored texts themselves, for every domain (education's four-name grid too), proxy axis and both
    placements: a row's key is the fold set of the names in its pair's two texts, and no fit unit of any proxy source
    that the key keeps carries one of those names."""
    tok = _tokenizer()
    fit_names = {}
    for d in domains.values():
        table = d.fit_tables["proxy"]
        clauses = table.clauses(d.design)
        for axis in rdt.encoding_axes(d.design, "proxy"):
            units = d.fit_units[(d.name, "proxy", axis)]
            per_unit = [_names_in(clauses[a]) | _names_in(clauses[b]) for a, b in zip(units.a, units.b)]
            assert all(1 <= len(n) <= rdt.MAX_PAIR_NAMES for n in per_unit), (d.name, axis)
            fit_names[(d.name, axis)] = (units, per_unit)
    sizes = {}
    for d in domains.values():
        d.select(tok)
        table = rdt.BlockTable.of(d.selected, "proxy", d.design)
        x_rows, x_convs, x_cells, x_block_of = rdt.decision_rows(table, d.name, d.dom.quality_field, d.settings,
                                                                 _fmt(tok))
        d_rows, d_cells, d_block_of = rdt.direct_rows(table, d.design, d.dom.quality_field)
        d_convs = table.direct_convs(d.design, d.prompt, tok)
        for axis in rdt.encoding_axes(d.design, "proxy"):
            partner = {}
            for a, b in d.design.axis_pairs(axis, "proxy"):
                partner[a], partner[b] = b, a
            for rows, convs, cells, block_of in ((x_rows, x_convs, x_cells, x_block_of),
                                                 (d_rows, d_convs, d_cells, d_block_of)):
                at = {(j, c, r["response"]): i for i, (c, j, r) in enumerate(zip(cells, block_of, rows))}
                keys = rdt.row_keys(cells, block_of, table, d.design, "proxy", axis, FOLDS)
                for i, (cell, j, (folds, _)) in enumerate(zip(cells, block_of, keys)):
                    if cell not in partner:
                        continue
                    other = at[(j, partner[cell], rows[i]["response"])]
                    pair_names = _names_in(convs[i]) | _names_in(convs[other])
                    assert folds == F(FOLDS[n] for n in pair_names) and pair_names
                    sizes.setdefault((d.name, axis), set()).add(len(pair_names))
                    for source, (units, per_unit) in fit_names.items():
                        kept = [names for names, f in zip(per_unit, units.folds) if not (f & folds)]
                        assert kept and not any(names & pair_names for names in kept), (d.name, axis, source)
    assert sizes == {("credit", "sex"): {2}, ("credit", "age"): {1}, ("credit", "intersection"): {2},
                     ("cv", "sex"): {2}, ("cv", "age"): {1}, ("cv", "family_status"): {1},
                     ("cv", "intersection"): {2}, ("education", "sex"): {2}, ("education", "ethnicity"): {2},
                     ("education", "economic_status"): {1}, ("education", "intersection"): {2}}


def test_min_fit_units_bounds_every_fit_a_target_asks_for(domains):
    tok = _tokenizer()
    stores = {s: units.store(torch.randn(len(d.fit_tables[s[1]].blocks) * 8, 4))
              for d in domains.values() for s, units in d.fit_units.items()}
    for d in domains.values():
        d.select(tok)
        for encoding in d.encodings:
            table = rdt.BlockTable.of(d.selected, encoding, d.design)
            _, cells, block_of = rdt.direct_rows(table, d.design, d.dom.quality_field)
            for axis in rdt.encoding_axes(d.design, encoding):
                for folds, template in set(rdt.row_keys(cells, block_of, table, d.design, encoding, axis, FOLDS)):
                    for s, store in stores.items():
                        low = rdt.min_fit_units(domains[s[0]].fit_units[s], 5)
                        assert store.n_fit(folds) >= low > 0
                        if len(store.templates) > 1:
                            assert store.n_fit(folds, template) >= low


def test_too_few_probe_pairs_for_a_held_out_fit_is_refused_before_the_model(credit_corpus):
    with pytest.raises(SystemExit, match="held-out fit"):
        rdt.Domain(_cfg(credit_corpus[0], "credit", probe=1), {}, FOLDS, 5)


# --------------------------------------------------------------------------- planted worlds ----------
def _model():
    return _tiny().float()


def _read_by_the_head(w, k, seed=7):
    """k orthonormal directions (columns), each read by the head alike: w·e = |w|/√k."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.cat([(w / w.norm())[:, None], torch.randn(len(w), k - 1, generator=g)], 1))
    q[:, 0] *= torch.sign(q[:, 0] @ w)
    v = -torch.ones(k) / k ** 0.5
    v[0] += 1
    return q @ (torch.eye(k) - 2 * torch.outer(v, v) / (v @ v))     # Householder: its first row is all 1/√k


def _world(model, domains, encoding, direct_state, decision_state, noise=0.02):
    """Stores fitted on planted probe states, and each domain's two placements with planted states.
    ``direct_state(domain, cell, name)`` and ``decision_state(domain, cell, name, response)`` (cell None: unmarked)
    give a row's state before noise."""
    tok = _tokenizer()
    g = torch.Generator().manual_seed(0)
    noisy = lambda states: torch.stack(states) + noise * torch.randn(len(states), 32, generator=g)
    stores, placements = {}, {}
    for d in domains:
        table = d.fit_tables[encoding]
        H = noisy([direct_state(d.name, c, table.names[j][c]) for j in range(len(table.blocks))
                   for c in d.design.cells])
        for axis in rdt.encoding_axes(d.design, encoding):
            stores[(d.name, encoding, axis)] = d.fit_units[(d.name, encoding, axis)].store(H)
        d.select(tok)
        table = rdt.BlockTable.of(d.selected, encoding, d.design)
        rows, cells, block_of = rdt.direct_rows(table, d.design, d.dom.quality_field)
        H = noisy([direct_state(d.name, c, table.names[j][c]) for c, j in zip(cells, block_of)])
        direct = rdt.build_placement(model, "direct", rows, H, torch.float32, None, cells, block_of, table, d.design,
                                     encoding, FOLDS)
        rows, _, cells, block_of = rdt.decision_rows(table, d.name, d.dom.quality_field, d.settings, _fmt(tok))
        Hx = noisy([decision_state(d.name, c, None if c is None else table.names[j][c], r["response"])
                    for r, c, j in zip(rows, cells, block_of)])
        decision = rdt.build_placement(model, "decision", rows, Hx, torch.float32, None, cells, block_of, table,
                                       d.design, encoding, FOLDS)
        placements[d.name] = {"direct": direct, "decision": decision, "table": table, "H": H}
    return stores, placements, {d.name: d.design for d in domains}


def _table(model, world, target, placement, n_random=2):
    stores, placements, designs = world
    pl = placements[target[0]][placement]
    kinds = rdt.row_kinds(target, stores, designs, n_random)
    draws = shared_draws(len(pl.index.records), 100, 42)
    return rdt.target_tables(model, pl, target, kinds, stores, designs, random_units(32, n_random, 0), draws, {})


AMP = 20.0
CREDIT_SEX, CV_SEX = ("credit", "explicit", "sex"), ("cv", "explicit", "sex")


def _sex_age_world(model, domains, sex_direction):
    """Sex along ``sex_direction(domain)`` in both placements (the decision disparity: +AMP on a female applicant's
    approval), age along e_1 in the direct placement."""
    w = get_head(model).effective_weights().reshape(-1)
    e = _read_by_the_head(w, 3)

    def direct(domain, cell, _):
        return AMP * ((0.5 if cell[0] == "female" else -0.5) * sex_direction(domain, e)
                      + (0.5 if cell[1] == AGE_YOUNG else -0.5) * e[:, 1])

    def decision(domain, cell, _, response):
        on = cell is not None and cell[0] == "female" and response == "approve"
        return AMP * sex_direction(domain, e) if on else torch.zeros(32)

    return _world(model, domains, "explicit", direct, decision), AMP * float(w.norm()) / 3 ** 0.5


def test_a_direction_shared_by_two_domains_transfers_on_both_targets(credit, cv):
    model = _model()
    world, effect = _sex_age_world(model, [credit, cv], lambda domain, e: e[:, 0])
    for target, other in ((CREDIT_SEX, CV_SEX), (CV_SEX, CREDIT_SEX)):
        for placement in ("direct", "decision"):
            tables = _table(model, world, target, placement)
            assert list(tables) == ["own_gates"]                    # a linear head has one gate mode
            table = tables["own_gates"]
            base = table["baseline"]["mean"]
            assert base == pytest.approx(effect, rel=0.05) and table["own_row"] == name(target)
            for row in (name(target), name(other)):
                assert abs(table["rows"][row]["nulled"]["mean"]) < 0.05 * base, (target, placement, row)
                assert abs(table["rows"][row]["shortfall"]["mean"]) < 0.05 * base
            assert table["rows"][name(target)]["gap"]["mean"] == 0.0
            # the same domain's age direction removes nothing of the sex effect: the specificity control
            off = table["rows"][name((target[0], "explicit", "age"))]
            assert abs(off["change"]["mean"]) < 0.05 * base
            assert off["shortfall"]["mean"] == pytest.approx(base, rel=0.1)
            assert all(abs(table["rows"][f"random{k}"]["change"]["mean"]) < 0.5 * base for k in range(2))


def test_orthogonal_directions_do_not_transfer(credit, cv):
    model = _model()
    world, effect = _sex_age_world(model, [credit, cv], lambda domain, e: e[:, 0] if domain == "credit" else e[:, 2])
    for target, other in ((CREDIT_SEX, CV_SEX), (CV_SEX, CREDIT_SEX)):
        for placement in ("direct", "decision"):
            table = _table(model, world, target, placement)["own_gates"]
            base = table["baseline"]["mean"]
            assert base == pytest.approx(effect, rel=0.05)
            assert abs(table["rows"][name(target)]["nulled"]["mean"]) < 0.05 * base
            assert abs(table["rows"][name(other)]["change"]["mean"]) < 0.05 * base
            assert table["rows"][name(other)]["shortfall"]["mean"] == pytest.approx(base, rel=0.1)


def test_the_single_axis_directions_cover_the_corner(credit, cv):
    model = _model()
    world, effect = _sex_age_world(model, [credit, cv], lambda domain, e: e[:, 0])
    target = ("credit", "explicit", "intersection")
    table = _table(model, world, target, "direct")["own_gates"]
    base = table["baseline"]["mean"]
    assert base == pytest.approx(2 * effect, rel=0.05)                       # the corner flips sex and age
    assert abs(table["rows"]["singles_joint"]["nulled"]["mean"]) < 0.05 * base
    assert abs(table["rows"][name(target)]["nulled"]["mean"]) < 0.05 * base
    assert table["rows"][name(CREDIT_SEX)]["nulled"]["mean"] == pytest.approx(effect, rel=0.1)   # one axis: half


def test_a_name_identity_code_is_removed_only_by_a_direction_that_saw_the_names(credit):
    """Every female name has a direction of its own, all read by the head alike. The battery's direction (every
    name) removes the gap; a direction fitted without the pair's names is orthogonal to its name and removes nothing."""
    model = _model()
    w = get_head(model).effective_weights().reshape(-1)
    e = _read_by_the_head(w, len(FEMALE_NAMES))
    at = {n: k for k, n in enumerate(FEMALE_NAMES)}
    direct = lambda domain, cell, n: AMP * e[:, at[n]] if n in at else torch.zeros(32)
    world = _world(model, [credit], "proxy", direct, lambda *a: torch.zeros(32))
    target = ("credit", "proxy", "sex")
    table = _table(model, world, target, "direct")["own_gates"]
    base = table["baseline"]["mean"]
    assert base == pytest.approx(AMP * float(w.norm()) / len(at) ** 0.5, rel=0.05)
    assert abs(table["rows"][name(target)]["change"]["mean"]) < 0.05 * base            # held out: nothing removed
    assert table["rows"]["own_seen"]["nulled"]["mean"] < 0.5 * base                    # seen: most of it
    assert table["rows"]["own_seen"]["shortfall"]["mean"] < -0.5 * base
    # no pair's rows are nulled by a direction fitted on a pair sharing one of its name folds
    stores, placements, _ = world
    for folds, _ in set(placements["credit"]["direct"].keys["sex"]):
        kept = [key for key, k in zip(stores[target].keys, stores[target].kept(folds)) if k]
        assert kept and all(not (f & folds) for f, _ in kept)


def test_paired_accuracy_reads_the_poles_along_the_direction(credit, cv):
    model = _model()
    (stores, placements, _), _ = _sex_age_world(model, [credit, cv], lambda domain, e: e[:, 0])
    table, H = placements["credit"]["table"], placements["credit"]["H"]
    units = rdt.UnitIndex.of(table, CREDIT_DESIGN, "explicit", "sex", FOLDS)
    record_at = {r: i for i, r in enumerate(placements["credit"]["direct"].index.records)}
    acc = lambda direction, **kw: (lambda s, c: (s.sum() / c.sum(), c))(
        *rdt.paired_wins(units.diffs(H), units, record_at, direction, **kw))
    rate, counts = acc(lambda key: stores[CV_SEX].direction(key[0]))
    assert rate == 1.0 and set(counts) == {4 * len(credit.templates)}       # 4 pairs per block, every record
    rate, _ = acc(lambda key: -stores[CV_SEX].direction(key[0]))
    assert rate == 0.0
    rate, _ = acc(lambda key: stores[("credit", "explicit", "age")].direction(key[0]))
    assert 0.2 < rate < 0.8                                                 # the age direction reads noise
    for tol in ({}, {"tol": 1e-9}):
        rate, _ = acc(lambda key: torch.zeros(32), **tol)
        assert rate == 0.5                                                  # a tie counts ½, not a win


def test_the_lexical_controls_tell_shared_words_from_new_ones(domains):
    features = rdt.ClauseFeatures(_tokenizer(), sorted({c for d in domains.values() for t in d.fit_tables.values()
                                                        for c in t.clauses(d.design)}))
    vectors = lambda control, d, e: features.vectors(control, d.fit_tables[e].clauses(d.design))
    lexical = {c: {s: units.store(vectors(c, domains[s[0]], s[1])) for d in domains.values()
                   for s, units in d.fit_units.items()} for c in rdt.CONTROLS}

    def control(source, target, held_out=True, kind="words"):
        d = domains[target[0]]
        units = d.fit_units[target]                 # the probe blocks stand in for evaluated ones
        record_at = {r: i for i, r in enumerate(dict.fromkeys(units.records))}
        sums, counts = rdt.paired_wins(units.diffs(vectors(kind, d, target[1])), units, record_at,
                                       lambda key: lexical[kind][source].direction(key[0] if held_out else F()),
                                       tol=1e-9)
        return sums.sum() / counts.sum()

    # credit and hiring share the explicit words ("woman"/"man", the ages); education says "female"/"male"
    assert control(CV_SEX, CREDIT_SEX) == 1.0
    assert control(("cv", "explicit", "age"), ("credit", "explicit", "age")) == 1.0
    assert control(("education", "explicit", "sex"), CREDIT_SEX) == 0.5
    # the proxy carries sex by the first name: unseen names tie, and an explicit direction knows no name
    proxy = ("credit", "proxy", "sex")
    assert control(("cv", "proxy", "sex"), proxy) == 0.5
    assert control(proxy, proxy) == 0.5
    assert control(("cv", "proxy", "sex"), proxy, held_out=False) > 0.9
    assert control(CREDIT_SEX, proxy) == 0.5
    # the birth year is not a name: the age proxy's words transfer under held-out names too
    assert control(("cv", "proxy", "age"), ("credit", "proxy", "age")) == 1.0
    # the corner's words order the pairs of an axis it flips (a component, not a control)
    assert control(("credit", "explicit", "intersection"), CREDIT_SEX) == 1.0
    # the token control reads the tokenizer's tokens: the test tokenizer knows the explicit words ...
    assert control(CV_SEX, CREDIT_SEX, kind="tokens") == 1.0
    assert control(("credit", "explicit", "age"), CREDIT_SEX, kind="tokens") == 0.5
    # ... and maps every first name to one unknown token, so a name swap is invisible to it
    assert control(("cv", "proxy", "sex"), proxy, held_out=False, kind="tokens") == 0.5


def test_token_vectors_are_the_tokenizers_tokens_of_the_clause():
    class Pieces:           # a tokenizer that splits names into shared pieces, as a sub-word tokenizer does
        ids = {"La": 0, "tonya": 1, "toya": 2, "Emily": 3, "is": 4, "Ty": 5, "rone": 6}

        def __call__(self, texts, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [[self.ids[t] for t in text.split()] for text in texts]}

    features = rdt.ClauseFeatures(Pieces(), ["La tonya is", "Emily is"])
    assert features.vocabulary == {0: 0, 1: 1, 3: 2, 4: 3}                   # the fitted clauses' tokens
    X = features.vectors("tokens", ["La toya is", "Ty rone is is"])
    # the unseen name shares its first piece with a fitted one; unknown pieces have no column; binary
    assert X.tolist() == [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="unknown lexical control"):
        features.vectors("letters", ["Emily is"])


def test_geometry_carries_the_exact_decomposition(credit, cv):
    model = _model()
    (stores, placements, _), effect = _sex_age_world(model, [credit, cv], lambda domain, e: e[:, 0])
    pl = placements["credit"]["direct"]
    units = rdt.UnitIndex.of(placements["credit"]["table"], CREDIT_DESIGN, "explicit", "sex", FOLDS)
    delta = units.diffs(placements["credit"]["H"]).mean(0)
    geo = rdt.geometry(pl, CREDIT_SEX, delta, stores, {s: 1.0 for s in stores})
    assert geo["effect_from_states"] == pytest.approx(effect, rel=0.05)
    own, other, off = (geo["sources"][name(s)] for s in (CREDIT_SEX, CV_SEX, ("credit", "explicit", "age")))
    assert own["cosine_to_own"] == pytest.approx(1.0) and other["cosine_to_own"] > 0.99
    assert own["share"] == pytest.approx(1.0, abs=0.02) and other["share"] == pytest.approx(1.0, abs=0.05)
    assert abs(off["share"]) < 0.05 and abs(off["cosine_to_own"]) < 0.1
    assert own["cosine_ceiling"] == 1.0 and own["w_dot_u"] > 0
    # the decision target: Δ_int of the strong records reproduces the decision disparity
    dpl = placements["credit"]["decision"]
    d_delta = rdt.decision_delta(dpl, CREDIT_DESIGN, "explicit", "sex")
    assert float(dpl.w @ d_delta) == pytest.approx(effect, rel=0.05)
    dpl.index.strong[:] = False
    assert rdt.decision_delta(dpl, CREDIT_DESIGN, "explicit", "sex") is None
    without = rdt.geometry(dpl, CREDIT_SEX, None, stores, {s: 1.0 for s in stores})
    assert math.isnan(without["effect_from_states"]) and math.isnan(without["sources"][name(CV_SEX)]["share"])


def _random_world(model, domains, encoding, gates_of=None, seed=0):
    """Random states for every domain: stores and placements (``gates_of(n, generator)`` for a gated head)."""
    tok = _tokenizer()
    g = torch.Generator().manual_seed(seed)
    stores, placements = {}, {}
    for d in domains.values():
        table = d.fit_tables[encoding]
        H = torch.randn(len(table.blocks) * 8, 32, generator=g)
        for axis in rdt.encoding_axes(d.design, encoding):
            stores[(d.name, encoding, axis)] = d.fit_units[(d.name, encoding, axis)].store(H)
        d.select(tok)
        table = rdt.BlockTable.of(d.selected, encoding, d.design)
        rows, _, cells, block_of = rdt.decision_rows(table, d.name, d.dom.quality_field, d.settings, _fmt(tok))
        gates = None if gates_of is None else gates_of(len(rows), g)
        decision = rdt.build_placement(model, "decision", rows, torch.randn(len(rows), 32, generator=g),
                                       torch.float32, gates, cells, block_of, table, d.design, encoding, FOLDS)
        rows, cells, block_of = rdt.direct_rows(table, d.design, d.dom.quality_field)
        gates = None if gates_of is None else gates_of(len(rows), g)
        direct = rdt.build_placement(model, "direct", rows, torch.randn(len(rows), 32, generator=g), torch.float32,
                                     gates, cells, block_of, table, d.design, encoding, FOLDS)
        placements[d.name] = {"direct": direct, "decision": decision, "table": table}
    return stores, placements, {d.name: d.design for d in domains.values()}


@pytest.mark.parametrize("encoding", ["explicit", "proxy"])
def test_the_reward_cache_never_serves_another_targets_rows(domains, encoding):
    """One cache per placement over all its axes gives the tables a fresh cache per target gives: a proxy row's
    rewards depend on the target axis (its pairs' names), and are never reused across axes."""
    model = _model()
    stores, placements, designs = _random_world(model, domains, encoding)
    random = random_units(32, 2, 0)
    for d in domains.values():
        for placement in ("direct", "decision"):
            pl = placements[d.name][placement]
            draws = shared_draws(len(pl.index.records), 30, 1)
            shared = {}
            for axis in rdt.encoding_axes(d.design, encoding):
                target = (d.name, encoding, axis)
                kinds = rdt.row_kinds(target, stores, designs, 2)
                tables = [rdt.target_tables(model, pl, target, kinds, stores, designs, random, draws, cache)
                          for cache in (shared, {})]
                assert json.dumps(tables[0], sort_keys=True) == json.dumps(tables[1], sort_keys=True), target


def test_proxy_decision_cells_recomputed_by_hand(domains):
    """Per pair, both cells' approve and decline states with the direction fitted without the pair's name folds
    projected out; the D-disparity over the strong records. Own, cross-domain and corner rows, education's name grid
    included."""
    model = _model()
    stores, placements, designs = _random_world(model, domains, "proxy")
    w = get_head(model).effective_weights().reshape(-1)
    for domain, axis, source in (("education", "sex", ("credit", "proxy", "sex")),
                                 ("education", "ethnicity", ("education", "proxy", "ethnicity")),
                                 ("education", "intersection", ("education", "proxy", "intersection")),
                                 ("credit", "age", ("cv", "proxy", "age")),
                                 ("cv", "family_status", ("cv", "proxy", "family_status"))):
        d, pl, table = domains[domain], placements[domain]["decision"], placements[domain]["table"]
        target = (domain, "proxy", axis)
        got = rdt.target_tables(model, pl, target, rdt.row_kinds(target, stores, designs, 0), stores, designs,
                                random_units(32, 0, 0), shared_draws(len(pl.index.records), 30, 1), {})["own_gates"]
        at = {(r["record_id"], r["template_id"], None if r["cell"] == "unmarked" else tuple(r["cell"]),
               r["response"]): i for i, r in enumerate(pl.rows)}
        base, nulled, strong = [], [], []
        for rid, blocks in d.selected.items():
            per_pair = {False: [], True: []}
            for j, block in enumerate(table.blocks):
                if block.record_id != rid:
                    continue
                for a, b in d.design.axis_pairs(axis, "proxy"):
                    u = stores[source].direction(F(FOLDS[table.names[j][c]] for c in (a, b)))
                    for null in (False, True):
                        margin = {}
                        for cell in (a, b):
                            h = [pl.H[at[(rid, block.template_id, cell, response)]] for response in
                                 ("approve", "decline")]
                            if null:
                                h = [x - (x @ u) * u / (u @ u) for x in h]
                            margin[cell] = float(w @ h[0] - w @ h[1])
                        per_pair[null].append(margin[a] - margin[b])
            base.append(np.mean(per_pair[False]))
            nulled.append(np.mean(per_pair[True]))
            strong.append(blocks[0].is_strong(d.dom.quality_field))
        strong = np.array(strong)
        assert 0 < strong.sum() < len(strong) and got["baseline"]["n_units"] == strong.sum()
        assert got["baseline"]["mean"] == pytest.approx(np.array(base)[strong].mean(), abs=1e-4)
        assert got["rows"][name(source)]["nulled"]["mean"] == pytest.approx(np.array(nulled)[strong].mean(), abs=1e-4)


def test_a_gated_heads_tables_score_each_mode_with_its_gates(credit):
    from tests.test_qrm import _model as qrm_model
    from tests.test_qrm import _tokenizer as qrm_tokenizer

    model = qrm_model(qrm_tokenizer()).float()
    head = get_head(model)
    gates_of = lambda n, g: torch.softmax(torch.randn(n, model.num_objectives, generator=g), -1)
    stores, placements, designs = _random_world(model, {"credit": credit}, "explicit", gates_of)
    u = stores[CREDIT_SEX].direction()
    kinds = rdt.row_kinds(CREDIT_SEX, stores, designs, 1)
    for placement in ("direct", "decision"):
        pl = placements["credit"][placement]
        cache = {}
        tables = rdt.target_tables(model, pl, CREDIT_SEX, kinds, stores, designs, random_units(32, 1, 0),
                                   shared_draws(len(pl.index.records), 30, 1), cache)
        assert list(tables) == ["gate_fixed", "own_gates"]
        if placement == "direct":       # one prompt for every row: one table, scored once
            assert tables["gate_fixed"] is tables["own_gates"] and not any(k[1] == "own_gates" for k in cache)
            continue
        nulled = pl.H - (pl.H @ u)[:, None] * u[None, :]
        with torch.no_grad():
            want = {"gate_fixed": (head.score(pl.H, pl.fixed), head.score(nulled, pl.fixed)),
                    "own_gates": (head.score(pl.H, pl.gates), head.score(nulled, pl.gates))}
        for mode, (base, null) in want.items():
            assert np.allclose(cache[("baseline", mode)], base.double().numpy(), atol=1e-5)
            assert np.allclose(cache[(("source", CREDIT_SEX), mode, None)], null.double().numpy(), atol=1e-5)
            assert tables[mode]["baseline"]["mean"] == pytest.approx(
                pl.effect(base.double().numpy(), "sex")[pl.counts > 0].mean(), abs=1e-6)
        assert not np.allclose(want["gate_fixed"][0], want["own_gates"][0], atol=1e-3)
        assert tables["gate_fixed"]["baseline"]["mean"] != tables["own_gates"]["baseline"]["mean"]


def test_score_domain_reads_each_placement_from_its_own_states(credit, monkeypatch):
    """`score_domain` on planted states (a fake embedder that reads the conversation): the paired accuracy is the
    held-out one on the direct states, the decision geometry uses the decision states, the draws are the domain's."""
    from pairs.cross_marker import DECISION_RESPONSES

    model, tok = _model(), _tokenizer()
    w = get_head(model).effective_weights().reshape(-1)
    e = _read_by_the_head(w, len(FEMALE_NAMES) + 1)
    of_name = {n: e[:, k + 1] for k, n in enumerate(FEMALE_NAMES)}
    approvals = {DECISION_RESPONSES["credit"].text("approve", k) for k in range(DECISION_RESPONSES["credit"].size)}

    def state(conv):
        prompt, response = conv
        if prompt == credit.prompt:                         # direct: explicit sex along e_0, each female name its own
            names = [v for n, v in of_name.items() if f", {n}," in response]
            sex = 0.5 if " woman." in response else -0.5 if " man." in response else 0.0
            return AMP * (sex * e[:, 0] + (names[0] if names else 0))
        on = " woman." in prompt and response in approvals  # decision: twice the direct effect, explicit only
        return 2 * AMP * e[:, 0] if on else torch.zeros(32)

    embed = lambda model, tok, convs, **kw: (torch.stack([state(c) for c in convs]), torch.float32, None)
    monkeypatch.setattr(rdt, "embed_with_gates", embed)
    drawn = []
    real_draws = rdt.shared_draws
    monkeypatch.setattr(rdt, "shared_draws", lambda *a: (drawn.append(a), real_draws(*a))[1])
    stores = {}
    for encoding, table in credit.fit_tables.items():
        H = embed(model, tok, table.direct_convs(CREDIT_DESIGN, credit.prompt, tok))[0]
        stores.update({s: units.store(H) for s, units in credit.fit_units.items() if s[1] == encoding})
    features = rdt.ClauseFeatures(tok, sorted({c for t in credit.fit_tables.values() for c in t.clauses(CREDIT_DESIGN)}))
    lexical = {c: {s: units.store(features.vectors(c, credit.fit_tables[s[1]].clauses(CREDIT_DESIGN)))
                   for s, units in credit.fit_units.items()} for c in rdt.CONTROLS}
    credit.select(tok)
    out = rdt.score_domain(model, tok, credit, stores, lexical, features, {"credit": CREDIT_DESIGN},
                           {s: 1.0 for s in stores}, FOLDS, random_units(32, 1, 0), 1)
    assert drawn == [(len(credit.selected), 50, 42)]        # one set of draws: the records, the config's n_boot, seed
    targets = {(t["encoding"], t["axis"], t["placement"]): t for t in out}
    effect = AMP * float(w @ e[:, 0])
    explicit, decision = targets[("explicit", "sex", "direct")], targets[("explicit", "sex", "decision")]
    assert explicit["tables"]["own_gates"]["baseline"]["mean"] == pytest.approx(effect, rel=1e-3)
    assert explicit["paired_acc"]["credit/explicit/sex"]["mean"] == 1.0
    assert explicit["geometry"]["effect_from_states"] == pytest.approx(effect, rel=1e-3)
    assert decision["tables"]["own_gates"]["baseline"]["mean"] == pytest.approx(2 * effect, rel=1e-3)
    assert decision["geometry"]["effect_from_states"] == pytest.approx(2 * effect, rel=1e-3)     # Δ_int, not Δ
    assert decision["paired_acc"] is None and decision["token_paired_acc"] is None
    # proxy: a name code. The held-out own row cannot order the pairs; the seen-name direction orders all of them
    proxy = targets[("proxy", "sex", "direct")]
    assert proxy["paired_acc"]["own_seen"]["mean"] == 1.0
    assert 0.2 < proxy["paired_acc"]["credit/proxy/sex"]["mean"] < 0.8
    assert proxy["lexical_paired_acc"]["credit/proxy/sex"] == 0.5 and proxy["lexical_paired_acc"]["own_seen"] == 1.0
    # the token control is its own space: the test tokenizer maps every name to one unknown token
    assert proxy["token_paired_acc"]["own_seen"] == 0.5 and explicit["token_paired_acc"]["credit/explicit/sex"] == 1.0
    own, seen = proxy["rows"]["credit/proxy/sex"]["n_fit"], proxy["rows"]["own_seen"]["n_fit"]
    assert own["min"] <= own["max"] < seen["min"] == seen["max"] == stores[("credit", "proxy", "sex")].n_units
    assert proxy["rows"]["credit/explicit/sex"]["n_fit"]["min"] == stores[CREDIT_SEX].n_units     # no name left out
    assert targets[("proxy", "intersection", "direct")]["singles_joint_axes"] == [
        "credit/proxy/sex", "credit/proxy/age", "credit/explicit/marital_status"]
    assert "singles_joint_axes" not in proxy


def test_a_gated_head_reads_the_decision_target_with_the_unmarked_gate(credit):
    """`build_placement` on gates: the direct placement's gate is fixed already; the decision rows take the gate of
    their record's unmarked prompt in the same template; without a control there is no gate-fixed mode."""
    tok = _tokenizer()
    credit.select(tok)
    table = rdt.BlockTable.of(credit.selected, "explicit", CREDIT_DESIGN)
    rows, _, cells, block_of = rdt.decision_rows(table, "credit", "credit_good", credit.settings, _fmt(tok))
    gates = torch.arange(len(rows), dtype=torch.float32)[:, None]
    H = torch.zeros(len(rows), 32)
    pl = rdt.build_placement(_model(), "decision", rows, H, torch.float32, gates, cells, block_of, table,
                             CREDIT_DESIGN, "explicit", FOLDS)
    assert [m for m, _ in pl.modes()] == ["gate_fixed", "own_gates"]
    for i, r in enumerate(rows):
        ref = rows[int(pl.fixed[i])]
        assert ref["cell"] == "unmarked" and (ref["record_id"], ref["template_id"]) == (r["record_id"], r["template_id"])
    marked = [i for i, c in enumerate(cells) if c is not None]
    no_control = rdt.build_placement(_model(), "decision", [rows[i] for i in marked], H[marked], torch.float32,
                                     gates[marked], [cells[i] for i in marked], [block_of[i] for i in marked], table,
                                     CREDIT_DESIGN, "explicit", FOLDS)
    assert [m for m, _ in no_control.modes()] == ["own_gates"]
    d_rows, d_cells, d_block_of = rdt.direct_rows(table, CREDIT_DESIGN, "credit_good")
    d_gates = torch.ones(len(d_rows), 1)
    direct = rdt.build_placement(_model(), "direct", d_rows, torch.zeros(len(d_rows), 32), torch.float32, d_gates,
                                 d_cells, d_block_of, table, CREDIT_DESIGN, "explicit", FOLDS)
    assert direct.fixed is d_gates and (direct.counts == 1).all()
    follows = rdt.build_placement(_model(), "direct", d_rows, torch.zeros(len(d_rows), 32), torch.float32, d_gates,
                                  d_cells, d_block_of, table, CREDIT_DESIGN, "explicit", FOLDS, gate_fixed=False)
    assert [m for m, _ in follows.modes()] == ["own_gates"]     # as its decision placement without a control
    assert (pl.counts == pl.index.strong).all() and 0 < pl.counts.sum() < len(pl.counts)   # strong records only


# --------------------------------------------------------------------------- main, end to end --------
@pytest.fixture
def run(credit_corpus, hiring, education_corpus, tmp_path, monkeypatch):
    """`main` on the three fixture manifests with the tiny model standing in for the loader."""
    from scoring.demographic_experiment import DemographicBiasExperiment

    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    paths = {}
    for domain, pairs, probe in (("credit", credit_corpus[0], 20), ("cv", hiring[0], 30),
                                 ("education", education_corpus[0], 20)):
        paths[domain] = tmp_path / f"{domain}.yaml"
        paths[domain].write_text(yaml.safe_dump({
            "name": domain, "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(pairs),
            "probe_records": probe, "batch_size": 16, "max_length": 1024,
            "extra": {"domain": domain, "cross_marker": {"n_strong": 3, "n_weak": 3, "n_folds": 2, "n_boot": 20,
                                                          "paraphrases": 1, "alphas": [0.0, 1.0]}}}))
    loads = []

    def load_model(self):
        loads.append(1)
        self.model, self.tokenizer = _tiny(), _tokenizer()
        self.config.model_revision = "abc123"

    monkeypatch.setattr(DemographicBiasExperiment, "load_model", load_model)

    def go(*argv, domains=("credit", "cv", "education")):
        monkeypatch.setattr("sys.argv", ["run_demographic_transfer.py", "--configs",
                                         *(str(paths[d]) for d in domains), *argv])
        rdt.main()

    go.loads, go.paths, go.tmp = loads, paths, tmp_path
    # the hiring and education fixtures' manifest folders are not the domains' default ones (cv, asap2): a result
    # name carries the folders wherever one differs
    folders = {"credit": "credit", "cv": "out", "education": "education"}
    go.out = lambda variant="", domains=("credit", "cv", "education"): (
        tmp_path / rdt.RESULTS_DIR / ("demographic_transfer_Tiny-RM" + variant + (
            "" if domains == ("credit",) else "__manifests-" + "-".join(folders[d] for d in domains)) + ".json"))
    return go


def test_main_end_to_end(run, capsys):
    run()
    text = run.out().read_text()
    result = json.loads(text)
    assert result["meta"]["config"]["model_revision"] == "abc123"
    assert all(c["model_revision"] == "abc123" for c in result["settings"]["configs"].values())
    assert set(result["meta"]["data"]) == {f"{d}/{f}" for d in ("credit", "cv", "education")
                                           for f in ("pairs.jsonl", "cells.jsonl")}
    assert result["domains"] == ["credit", "cv", "education"] and result["name_folds"] == FOLDS
    sources = result["sources"]
    assert len(sources) == 23 and all(s["min_fit_units"] > 0 for s in sources.values())
    assert sources["credit/proxy/sex"]["named"] and not sources["credit/explicit/sex"]["named"]
    assert sources["credit/proxy/sex"]["min_fit_units"] < sources["credit/explicit/sex"]["min_fit_units"]
    assert sources["cv/explicit/sex"]["templates"] == ["bios_v1"]
    # one entry per (target, placement); every source is a row of every target
    targets = {(t["domain"], t["encoding"], t["axis"], t["placement"]): t for t in result["targets"]}
    assert len(targets) == len(result["targets"]) == 46
    for (domain, encoding, axis, placement), t in targets.items():
        table = t["tables"]["own_gates"]
        assert list(t["tables"]) == ["own_gates"] and t["own_row"] == table["own_row"] == f"{domain}/{encoding}/{axis}"
        assert set(t["rows"]) == set(table["rows"]) and set(sources) <= set(t["rows"])
        assert t["rows"][t["own_row"]]["relation"] == "own" and table["rows"][t["own_row"]]["gap"]["mean"] == 0.0
        assert all(("n_fit" in rel) != row.startswith("random") for row, rel in t["rows"].items())
        extra = {r for r in t["rows"] if r not in sources and not r.startswith("random")}
        assert extra == ({"own_seen"} if encoding == "proxy" else set()) \
            | ({"template"} if domain != "cv" else set()) \
            | ({"others"} if axis == "sex" else set()) | ({"singles_joint"} if axis == "intersection" else set())
        assert sum(r.startswith("random") for r in t["rows"]) == 5
        if placement == "direct":
            assert set(t["paired_acc"]) == set(t["lexical_paired_acc"]) == set(t["token_paired_acc"]) == \
                {r for r in t["rows"] if not r.startswith("random") and r != "singles_joint"}
            assert all(0 <= v <= 1 for c in ("lexical_paired_acc", "token_paired_acc") for v in t[c].values())
            assert table["baseline"]["n_units"] == 6
        else:
            assert t["paired_acc"] is None and t["lexical_paired_acc"] is None and t["token_paired_acc"] is None
            assert table["baseline"]["n_units"] == 3                       # the strong records
        assert ("singles_joint_axes" in t) == (axis == "intersection")
        assert set(t["geometry"]["sources"]) == set(sources)
        if (domain, encoding) == ("credit", "explicit"):    # the tiny tokenizer knows these words only: a marker
            # it cannot see gives a zero direction, whose cosine is undefined
            assert t["geometry"]["sources"][t["own_row"]]["cosine_to_own"] == pytest.approx(1.0, abs=1e-5)
    sex = targets[("credit", "explicit", "sex", "direct")]["rows"]
    assert sex["cv/explicit/sex"]["relation"] == "domain" and sex["others"]["relation"] == "others"
    assert sex["education/proxy/sex"]["relation"] == "domain+encoding"
    assert sex["credit/explicit/intersection"]["relation"] == "component"
    assert sex["cv/proxy/intersection"]["relation"] == "component+domain+encoding"
    # an explicit target leaves no name out: every row rests on all of its source's pairs
    assert all(rel["n_fit"]["min"] == rel["n_fit"]["max"] == sources[row]["n_units"]
               for row, rel in sex.items() if row in sources)
    # a proxy target: the own row rests on fewer pairs than the seen-name fit and than the explicit source's
    proxy = targets[("credit", "proxy", "sex", "direct")]["rows"]
    assert proxy["credit/proxy/sex"]["n_fit"]["max"] < proxy["own_seen"]["n_fit"]["min"]
    assert proxy["credit/proxy/sex"]["n_fit"]["min"] >= sources["credit/proxy/sex"]["min_fit_units"]
    assert proxy["credit/explicit/sex"]["n_fit"]["min"] == sources["credit/explicit/sex"]["n_units"]
    assert targets[("credit", "proxy", "intersection", "decision")]["singles_joint_axes"] == [
        "credit/proxy/sex", "credit/proxy/age", "credit/explicit/marital_status"]
    assert set(result["selection"]["credit"]["fit"]) == {
        "records_in_cells", "probe_records", "probe_records_without_blocks", "templates", "battery_split"}
    assert any("name pools" in c for c in result["caveats"])
    # explicit words shared with hiring, not with education; unseen names tie
    words = targets[("credit", "explicit", "sex", "direct")]["lexical_paired_acc"]
    assert words["cv/explicit/sex"] == 1.0 and words["education/explicit/sex"] == 0.5
    assert targets[("credit", "proxy", "sex", "direct")]["lexical_paired_acc"]["cv/proxy/sex"] == 0.5
    # numbers and ids only
    assert "applicant" not in text and "student" not in text
    saved = torch.load(str(run.out().with_suffix("")) + "_directions.pt")
    assert set(saved["full_data"]) == set(sources) and saved["name_folds"] == FOLDS
    for source, v in saved["full_data"].items():        # unit length, or zero where the tokenizer sees no marker
        assert v.norm().item() == pytest.approx(1.0 if sources[source]["separation"] > 1e-6 else 0.0, abs=1e-4)
    assert sources["credit/explicit/sex"]["separation"] > 1e-3
    assert "DEMOGRAPHIC TRANSFER" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


def test_the_targets_are_the_cross_marker_arms_numbers(run, monkeypatch):
    """The same records; the decision baseline is the arm's D-disparity on the strong records and the own row (explicit;
    for a proxy target ``own_seen``, the fit that left no name out) its ``null_direct`` column; the direct baseline is
    the arm's placement-check direct gap over all records. Exactly: the same states, directions and arithmetic."""
    run()
    ours = json.loads(run.out().read_text())
    monkeypatch.setattr(rx, "credit_reference", lambda dom, selected: None)       # needs the real corpus
    monkeypatch.setattr("sys.argv", ["run_cross_marker.py", "--config", str(run.paths["credit"])])
    rx.main()
    arm = json.loads((run.tmp / rdt.RESULTS_DIR / "crossmarker_credit_Tiny-RM.json").read_text())
    assert arm["records"] == ours["records"]["credit"]
    targets = {(t["domain"], t["encoding"], t["axis"], t["placement"]): t for t in ours["targets"]}
    moved = []
    for encoding in ("explicit", "proxy"):
        for axis in rdt.encoding_axes(CREDIT_DESIGN, encoding):
            table = targets[("credit", encoding, axis, "decision")]["tables"]["own_gates"]
            disparity = lambda column: arm["metrics"][encoding][column]["margins"]["D"]["strong"]["disparity"][axis]["mean"]
            row = f"credit/{encoding}/{axis}" if encoding == "explicit" else "own_seen"
            assert table["baseline"]["mean"] == pytest.approx(disparity("baseline"), abs=1e-9)
            assert table["rows"][row]["nulled"]["mean"] == pytest.approx(disparity(f"null_direct:{axis}"), abs=1e-9)
            moved.append(abs(table["rows"][row]["change"]["mean"]))
            check = arm["placement"][encoding]["baseline"]
            n = {g: check[g]["n_records"] for g in ("strong", "weak")}
            gap = sum(check[g]["axes"][axis]["direct_gap"]["mean"] * n[g] for g in n) / sum(n.values())
            direct = targets[("credit", encoding, axis, "direct")]["tables"]["own_gates"]
            assert direct["baseline"]["mean"] == pytest.approx(gap, abs=1e-9)
    assert max(moved) > 1e-4           # the nulling changed something: the comparison is not of two unmoved numbers


def test_a_variant_gets_its_own_name(run):
    two = ("credit", "cv")
    run("--n-strong", "2", "--name-folds", "4", "--random-draws", "1", domains=two)
    out = run.out("__n_strong-2__name_folds-4__random_draws-1__domains-credit-cv", two)
    result = json.loads(out.read_text())
    assert result["selection"]["credit"]["evaluated"]["n_strong"] == 2 and len(result["sources"]) == 15
    assert not any("others" in t["rows"] for t in result["targets"])       # one other domain: its row is `domain`
    assert max(result["name_folds"].values()) == 3
    # one value per config
    run("--n-strong", "2", "3", "--n-weak", "1", domains=two)
    result = json.loads(run.out("__n_strong-2-3__n_weak-1__domains-credit-cv", two).read_text())
    assert [result["selection"][d]["evaluated"]["n_strong"] for d in two] == [2, 3]


def test_default_out_names_the_model_and_the_variant():
    assert rdt.default_out("org/RM-8B") == rdt.RESULTS_DIR / "demographic_transfer_RM-8B.json"
    assert rdt.default_out("org/RM", "__seed-1").name == "demographic_transfer_RM__seed-1.json"


def test_a_gated_model_runs_end_to_end_with_one_set_of_modes_per_target(run, monkeypatch):
    from scoring.demographic_experiment import DemographicBiasExperiment
    from tests.test_qrm import _model as qrm_model
    from tests.test_qrm import _tokenizer as qrm_tokenizer

    def load_model(self):
        tok = qrm_tokenizer()
        self.model, self.tokenizer = qrm_model(tok), tok
        self.config.model_revision = "abc123"

    monkeypatch.setattr(DemographicBiasExperiment, "load_model", load_model)
    run(domains=("credit",))
    result = json.loads(run.out("__domains-credit", ("credit",)).read_text())
    assert {tuple(t["tables"]) for t in result["targets"]} == {("gate_fixed", "own_gates")}
    # without the unmarked control no gate can be fixed: both placements of a target say so alike
    cfg = yaml.safe_load(run.paths["credit"].read_text())
    cfg["extra"]["cross_marker"]["include_unmarked"] = False
    run.paths["credit"].write_text(yaml.safe_dump(cfg))
    run("--overwrite", domains=("credit",))
    result = json.loads(run.out("__domains-credit", ("credit",)).read_text())
    assert {tuple(t["tables"]) for t in result["targets"]} == {("own_gates",)}


def test_bad_requests_stop_before_the_model_loads(run, tmp_path):
    with pytest.raises(SystemExit, match="distinct domains"):
        run(domains=("credit", "credit"))
    with pytest.raises(SystemExit, match="--name-folds must exceed"):
        run("--name-folds", "2")
    with pytest.raises(SystemExit, match="one value or one per config"):
        run("--n-strong", "1", "2")
    with pytest.raises(SystemExit, match="n_strong must be at least 1"):
        run("--n-strong", "0")
    with pytest.raises(SystemExit, match="held-out fit"):
        run("--probe-records", "1")
    with pytest.raises(SystemExit, match="--random-draws"):
        run("--random-draws", "-1")
    other = tmp_path / "other.yaml"
    other.write_text(run.paths["cv"].read_text().replace("org/Tiny-RM", "org/Other-RM"))
    run.paths["cv"] = other
    with pytest.raises(SystemExit, match="one model"):
        run()
    assert run.loads == []
