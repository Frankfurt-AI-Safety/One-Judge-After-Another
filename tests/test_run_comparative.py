"""
Tests for `runners/run_comparative.py`: settings and the result name, the pairing inputs (the candidate pool,
probe exclusion, the name and length checks), and — on tiny random RMs (CPU) — scoring with every direction
(cross-fitting checked on synthetic states), the gated head's ``gate_fixed``, and `main` end to end on a
generator-style credit manifest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch

from pairs.comparative import KINDS, PAIRINGS, UNMARKED, RecordPair, contrast_axes, draw_pairs
from pairs.cross_marker import load_cell_blocks
from pairs.factorial import CREDIT_DESIGN
from runners.run_comparative import (
    DEFAULTS, build_parser, build_rows, candidates_from, check_blocks, check_settings, compatibility, default_out,
    parse_n_pairs, probe_split_ids, resolve_settings, score_encoding,
)
from runners.run_cross_marker import token_counter
from scoring.comparative_metrics import comparative_metrics
from scoring.dataset_base import format_conversation
from scoring.experiment import ExperimentConfig, variant_suffix
from substrates.domains import get_domain
from tests.test_comparative import _blocks

CONFIGS = ("credit_comparative_qwen06", "cv_comparative_qwen06", "edu_comparative_asap2_qwen06")


# --------------------------------------------------------------------------- settings ----------------
class TestSettings:
    def test_n_pairs_forms(self):
        assert parse_n_pairs(20) == {p: 20 for p in PAIRINGS}
        assert parse_n_pairs("10,5,7") == dict(zip(PAIRINGS, (10, 5, 7)))
        assert parse_n_pairs({"strong_weak": 3}) == {"strong_strong": 0, "strong_weak": 3, "weak_weak": 0}
        with pytest.raises(ValueError):
            parse_n_pairs("1,2")
        with pytest.raises(ValueError, match="unknown pairings"):
            parse_n_pairs({"strong": 3})

    def test_precedence_and_checks(self):
        s = resolve_settings({"comparative": {"n_pairs": 50, "paraphrases": 1}}, {"n_pairs": "4", "seed": None})
        assert s["n_pairs"] == {p: 4 for p in PAIRINGS} and s["paraphrases"] == 1 and s["seed"] == DEFAULTS["seed"]
        with pytest.raises(KeyError, match="n_pair"):
            resolve_settings({"comparative": {"n_pair": 5}}, {})
        with pytest.raises(KeyError, match="include_unmarked"):      # always on now: the yardstick needs them
            resolve_settings({"comparative": {"include_unmarked": False}}, {})
        with pytest.raises(ValueError, match="directions"):
            resolve_settings({}, {"directions": ["prompt"]})

    @pytest.mark.parametrize("override, message", [
        ({"paraphrases": 4}, "paraphrases"), ({"paraphrases": 0}, "paraphrases"), ({"n_folds": 1}, "n_folds"),
        ({"n_boot": 0}, "n_boot"), ({"n_pairs": "0"}, "n_pairs"), ({"n_pairs": "-1,2,2"}, "n_pairs"),
        ({"encodings": []}, "encoding")])
    def test_bad_values_are_named(self, override, message):
        with pytest.raises(SystemExit, match=message):
            check_settings(resolve_settings({}, override), "credit")
        check_settings(resolve_settings({}, {}), "credit")

    def test_config_files(self):
        for name in CONFIGS:
            cfg = ExperimentConfig.from_yaml(f"configs/demographic_{name}.yaml")
            s = resolve_settings(cfg.extra, {})
            check_settings(s, cfg.extra["domain"])
            assert s["directions"] == ["direct", "own"]
            assert set(s["n_pairs"]) == set(PAIRINGS) and cfg.dataset_source.endswith("pairs.jsonl")
            # two essays per prompt: only the education config raises max_length
            assert cfg.max_length == (4096 if name.startswith("edu") else 2048)

    def test_default_out_and_variant(self):
        assert str(default_out("credit", "data/demographic/credit/pairs.jsonl", "org/M")) == \
            "artifacts/results/demographic/comparative_credit_M.json"
        assert default_out("education", "data/demographic/education/asap2/pairs.jsonl", "M").name == \
            "comparative_education_asap2_M.json"
        configured = {"n_pairs": {p: 150 for p in PAIRINGS}, "encodings": ["explicit", "proxy"], "seed": 42}
        used = {"n_pairs": {p: 4 for p in PAIRINGS}, "encodings": ["explicit"], "seed": 42}
        assert variant_suffix(configured, used) == "__n_pairs-4-4-4__encodings-explicit"
        assert variant_suffix(configured, configured) == ""
        assert default_out("credit", "d/credit/p.jsonl", "M", "__seed-1").name == "comparative_credit_M__seed-1.json"
        # a revision with a slash stays one file name
        assert variant_suffix({"revision": None}, {"revision": "refs/pr/2"}) == "__revision-refs_pr_2"


# --------------------------------------------------------------------------- pairing inputs ----------
def _flat(blocks):
    return [b for per in blocks.values() for b in per.values()]


class TestCandidates:
    def test_probe_records_and_incomplete_records_are_out(self):
        blocks = _flat(_blocks("credit", [("s1", True), ("s2", True), ("w1", False)]))
        blocks = [b for b in blocks if not (b.record_id == "s2" and b.template_id == "credit_v2")]
        by_record, complete, cand, rep = candidates_from(blocks, quality_field="credit_good", match_field=None,
                                                         exclude={"w1"})
        assert cand == {"s1": (True, None)} and complete == {"s1": True, "w1": False}
        # the strata (which set the pool split) count the complete records, probe records included
        assert rep == {"records_in_cells": 3, "excluded_probe_records": 1, "incomplete_records": 1,
                       "strata": {"strong": 1, "weak": 1}}
        assert set(by_record["s1"]) == {("explicit", "credit_v1"), ("explicit", "credit_v2")}

    def test_the_pool_uses_every_block_whatever_is_requested(self):
        # a record without its proxy blocks is incomplete even for an explicit-only run
        blocks = _flat(_blocks("credit", [("s1", True)])) + _flat(_blocks("credit", [("s2", True)], "proxy")) \
            + _flat(_blocks("credit", [("s2", True)]))
        _, _, cand, rep = candidates_from(blocks, quality_field="credit_good", match_field=None, exclude=set())
        assert set(cand) == {"s2"} and rep["incomplete_records"] == 1

    def test_probe_ids_must_be_records_of_the_cells(self):
        # another id format would exclude nothing and pair the probe records silently
        blocks = _flat(_blocks("credit", [("s1", True), ("w1", False)]))
        with pytest.raises(SystemExit, match="different record ids"):
            candidates_from(blocks, quality_field="credit_good", match_field=None, exclude={"credit-s1"})

    def test_the_fields_the_pairing_reads_are_checked_before_loading(self):
        blocks = _flat(_blocks("cv", [("a", True)]))
        with pytest.raises(SystemExit, match="target_role"):
            check_blocks(blocks, get_domain("cv"))
        ok = [dataclasses.replace(b, real_fields={**b.real_fields, "target_role": "nurse"}) for b in blocks]
        check_blocks(ok, get_domain("cv"))
        no_role = [dataclasses.replace(b, real_fields={"qualified": True, "target_role": "nurse"}) for b in blocks]
        with pytest.raises(SystemExit, match="role"):
            check_blocks(no_role, get_domain("cv"))
        _, _, cand, _ = candidates_from(ok, quality_field="qualified", match_field="target_role", exclude=set())
        assert cand["a"][1] == "nurse"

    def test_probe_split_ids_are_the_direct_directions_records(self, manifest, monkeypatch):
        from runners.run_cross_marker import direct_directions
        from tests.test_run_cross_marker import _model, _tokenizer

        monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
        dom = get_domain("credit")
        ids = probe_split_ids(dom, str(manifest), 6, 42)
        _, fitted_on, _ = direct_directions(_model(), _tokenizer(), dom, str(manifest), ["explicit", "proxy"],
                                            probe_records=6, split_seed=42, batch_size=16, max_length=1024)
        assert ids == fitted_on and len(ids) == 6


def _tok_and_fmt():
    from tests.test_run_cross_marker import _tokenizer
    tok = _tokenizer()
    return tok, (lambda p, r: format_conversation(tok, p, r))


class TestCompatibility:
    def test_names_per_prompt_and_length(self):
        tok, fmt = _tok_and_fmt()
        blocks = _flat(_blocks("credit", [("s1", True), ("s2", True)], encoding="proxy"))
        by_record, _, _, _ = candidates_from(blocks, quality_field="credit_good", match_field=None, exclude=set())
        settings = resolve_settings({}, {})
        pair = RecordPair("strong_strong", "s1", "s2", 0)
        names = {r: {k: frozenset({f"{r}-{k[1]}"}) for k in per} for r, per in by_record.items()}
        named = lambda r, k_name: {k: dataclasses.replace(b, names=k_name(r, k)) for k, b in by_record[r].items()}
        distinct = {r: named(r, lambda r, k: names[r][k]) for r in by_record}
        assert compatibility(distinct, "credit", settings, fmt, token_counter(tok), 10_000)(pair) is None
        # a name shared across templates never meets in one prompt; within one template it does
        across = {"s1": distinct["s1"], "s2": {k: dataclasses.replace(b, names=names["s1"][
            next(j for j in by_record["s1"] if j != k)]) for k, b in by_record["s2"].items()}}
        assert compatibility(across, "credit", settings, fmt, token_counter(tok), 10_000)(pair) is None
        within = {"s1": distinct["s1"], "s2": named("s2", lambda r, k: names["s1"][k])}
        assert compatibility(within, "credit", settings, fmt, token_counter(tok), 10_000)(pair) == "shared_name"
        # a pair with any text over max_length is refused
        assert compatibility(distinct, "credit", settings, fmt, token_counter(tok), 20)(pair) == "too_long"


# --------------------------------------------------------------------------- scoring -----------------
def _scored(manifest, model, tok, directions=("own",), n_folds=2):
    fmt = lambda p, r: format_conversation(tok, p, r)
    settings = resolve_settings({}, {"n_pairs": 100, "n_folds": n_folds, "directions": list(directions),
                                     "encodings": ["explicit"], "paraphrases": 1})
    blocks = load_cell_blocks(manifest.parent / "cells.jsonl", CREDIT_DESIGN)
    by_record, _, cand, _ = candidates_from(blocks, quality_field="credit_good", match_field=None, exclude=set())
    pairs, _ = draw_pairs(cand, settings["n_pairs"], 42)
    rows, convs = build_rows(pairs, by_record, "credit", "explicit", ["credit_v1", "credit_v2"], settings, fmt)
    axes = contrast_axes(CREDIT_DESIGN, "explicit")
    return pairs, rows, convs, axes, settings


def test_rows_carry_ids_not_text(manifest):
    tok, _ = _tok_and_fmt()
    pairs, rows, convs, axes, _ = _scored(manifest, None, tok)
    assert len(rows) == len(convs) == len(pairs) * 2 * (len(axes) * 2 + 1) * 2 * len(KINDS) * 2
    assert set(rows[0]) == {"pair_id", "pairing", "template_id", "encoding", "axis", "protected", "order", "kind",
                            "chosen", "paraphrase"}
    assert {r["axis"] for r in rows} == set(axes) | {UNMARKED}


def test_score_encoding_with_every_direction(manifest, monkeypatch):
    from runners.run_cross_marker import PhaseTimer
    from tests.test_run_cross_marker import _model, _tokenizer

    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    model, tok = _model(), _tokenizer()
    pairs, rows, convs, axes, settings = _scored(manifest, model, tok)
    torch.manual_seed(1)
    direct = {a: torch.nn.functional.normalize(torch.randn(32), dim=0) for a in axes}
    timer = PhaseTimer()
    columns, geometry, saved = score_encoding(model, tok, rows, convs, direct, axes, settings, batch_size=16,
                                              max_length=1024, direct_reliability={a: 0.9 for a in axes},
                                              show_progress=False, timer=timer)
    assert columns == (["baseline"] + [f"null_direct:{a}" for a in axes] + ["null_direct:joint"]
                       + [f"null_own:{a}" for a in axes])
    assert all(c in r for r in rows for c in columns)
    assert set(timer.seconds) == {"embed", "mechanism"}            # the throughput needs the embed phase alone
    assert set(saved["folds"]) == {p.pair_id for p in pairs} and geometry["own_skipped"] is None
    # the tiny random model's last-token state does not register a marker swap ~600 tokens back (the contrasts
    # are exactly 0 even in float32), so its own directions are degenerate and their cosines NaN; cross-fitting is
    # checked on synthetic states below
    assert all(v["cosine"] != v["cosine"] or -1 <= v["cosine"] <= 1 for v in geometry["own_vs_direct"].values())
    assert set(geometry["reliability"]) == {f"own:{a}" for a in axes} | {f"direct:{a}" for a in axes}
    m = comparative_metrics(rows, "null_own:sex", baseline_key="baseline", n_boot=20)
    assert "marker_effect_change" in m["all"]["sex"]["merit"]


def test_cross_fitting_never_nulls_a_pair_with_its_own_direction(manifest, monkeypatch):
    # synthetic states with a "chosen is protected" signal: each fold's rows are nulled with the unit mean of the
    # other folds' pair contrasts, and nothing else (reviewer C's check, 2026-10-01)
    import probes.probe as pp
    from probes import cross_marker_directions as cmd
    from probes.comparative_directions import pair_contrasts
    from probes.probe import rewards_from_hidden
    from tests.test_run_cross_marker import _model, _tokenizer

    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    model, tok = _model(), _tokenizer()
    pairs, rows, convs, axes, settings = _scored(manifest, model, tok, n_folds=3)
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(32, generator=g)
    H = torch.randn(len(rows), 32, generator=g)
    for i, r in enumerate(rows):
        if r["protected"] is not None and r["protected"] == r["chosen"]:
            H[i] += 0.5 * signal + 0.3 * torch.randn(32, generator=g)
    dtype = next(model.parameters()).dtype
    monkeypatch.setattr(pp, "embed_with_gates", lambda *a, **k: (H.clone(), dtype, None))
    _, _, saved = score_encoding(model, tok, rows, convs, {}, axes, settings, batch_size=16, max_length=1024,
                                 show_progress=False)
    folds = saved["folds"]
    assert len(set(folds.values())) == 3
    for axis in axes:
        ids, C = pair_contrasts(H, rows, axis)
        for f in set(folds.values()):
            u = cmd.unit(C[[k for k, p in enumerate(ids) if folds[p] != f]].mean(0))
            assert torch.allclose(u, saved["cross_fitted"][f"own:{axis}"][f], atol=1e-6)
            idx = [i for i, r in enumerate(rows) if folds[r["pair_id"]] == f]
            _, rr = rewards_from_hidden(model, H[idx], dtype, u)
            assert [rows[i][f"null_own:{axis}"] for i in idx] == pytest.approx(rr.tolist(), abs=1e-5)
        assert any(abs(r[f"null_own:{axis}"] - r["baseline"]) > 1e-4 for r in rows)


def test_own_directions_are_skipped_and_recorded_when_one_fold_holds_every_pair(manifest, monkeypatch):
    from tests.test_run_cross_marker import _model, _tokenizer

    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    model, tok = _model(), _tokenizer()
    pairs, rows, convs, axes, settings = _scored(manifest, model, tok)
    keep = {next(p.pair_id for p in pairs if p.pairing == g) for g in PAIRINGS if any(p.pairing == g for p in pairs)}
    sub = [(r, c) for r, c in zip(rows, convs) if r["pair_id"] in keep]       # one pair per pairing: all in fold 0
    columns, geometry, _ = score_encoding(model, tok, [r for r, _ in sub], [c for _, c in sub], {}, axes, settings,
                                          batch_size=16, max_length=1024, show_progress=False)
    assert columns == ["baseline"] and "one fold" in geometry["own_skipped"]


def test_gate_fixed_takes_the_gate_of_the_unmarked_prompt_in_the_same_order(manifest, monkeypatch):
    # the tiny QRM's gate cannot tell the orders apart (it reads the end of the user turn, the same in both), so
    # every row gets a distinct synthetic gate and the test pins which row's gate each row is rescored with
    import probes.probe as pp
    from probes.heads import get_head
    from probes.probe import embed_with_gates
    from tests.test_qrm import _model, _tokenizer

    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    tok = _tokenizer()
    model = _model(tok)
    pairs, rows, convs, axes, settings = _scored(manifest, model, tok, directions=())
    h, dtype, g = embed_with_gates(model, tok, convs, batch_size=16, max_length=1024, show_progress=False)
    g = torch.softmax(torch.randn(g.shape, generator=torch.Generator().manual_seed(3)), dim=-1).to(g.dtype)
    monkeypatch.setattr(pp, "embed_with_gates", lambda *a, **k: (h, dtype, g))
    columns, geometry, _ = score_encoding(model, tok, rows, convs, {}, axes, settings, batch_size=16,
                                          max_length=1024, show_progress=False)
    assert columns == ["baseline", "gate_fixed"] and geometry["head"].startswith("quantile_gated")
    i = next(k for k, r in enumerate(rows) if r["axis"] != UNMARKED and r["order"] == "YX")
    ref = next(k for k, r in enumerate(rows) if r["axis"] == UNMARKED and r["order"] == "YX"
               and (r["pair_id"], r["template_id"]) == (rows[i]["pair_id"], rows[i]["template_id"]))
    other = next(k for k, r in enumerate(rows) if r["axis"] == UNMARKED and r["order"] == "XY"
                 and (r["pair_id"], r["template_id"]) == (rows[i]["pair_id"], rows[i]["template_id"]))
    with torch.no_grad():
        same = float(get_head(model).score(h[i:i + 1].to(dtype), g[ref:ref + 1])[0])
        wrong = float(get_head(model).score(h[i:i + 1].to(dtype), g[other:other + 1])[0])
    assert rows[i]["gate_fixed"] == pytest.approx(same, abs=1e-3) and abs(same - wrong) > 1e-2
    first_unmarked = {}
    for k, r in enumerate(rows):
        if r["axis"] == UNMARKED:
            first_unmarked.setdefault((r["pair_id"], r["template_id"], r["order"]), k)
    assert all(rows[k]["gate_fixed"] == pytest.approx(rows[k]["baseline"]) for k in first_unmarked.values())


# --------------------------------------------------------------------------- main, end to end ---------
@pytest.fixture
def run_main(manifest, tmp_path, monkeypatch):
    """`main` on the fixture manifest with the tiny Llama RM standing in for the loader."""
    import yaml

    from runners import run_comparative as rc
    from tests.test_run_cross_marker import _model, _tokenizer

    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(manifest),
        "probe_records": 6, "batch_size": 16, "max_length": 1024,
        "extra": {"domain": "credit", "comparative": {"n_pairs": 100, "n_folds": 2, "n_boot": 20,
                                                       "paraphrases": 1}}}))
    loads = []

    def load_model(self):
        loads.append(1)
        self.model, self.tokenizer = _model(), _tokenizer()
        self.config.model_revision = "abc123"

    from scoring.demographic_experiment import DemographicBiasExperiment
    monkeypatch.setattr(DemographicBiasExperiment, "load_model", load_model)

    def _run(*extra):
        monkeypatch.setattr("sys.argv", ["run_comparative.py", "--config", str(cfg_path), *extra])
        rc.main()

    _run.loads = loads
    _run.out = lambda suffix="": (tmp_path / "artifacts/results/demographic"
                                  / f"comparative_credit_{manifest.parent.name}_Tiny-RM{suffix}.json")
    return _run


def test_main_end_to_end(run_main, manifest):
    run_main()
    out = run_main.out()
    summary = json.loads(out.read_text())
    stem = str(out.with_suffix(""))
    assert all(Path(stem + s).exists() for s in ("_rewards.jsonl", "_directions.pt"))
    meta = summary["meta"]
    assert meta["config"]["model_revision"] == "abc123"
    for name in ("pairs.jsonl", "cells.jsonl"):
        assert meta["data"][name]["sha256"] == hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest()
    # the direct directions' probe records are never paired
    probe = set(summary["probe_record_ids"])
    assert len(probe) == 6 and all(m["n_records"] == 6 for m in summary["probe_directions"].values())
    assert summary["selection"]["excluded_probe_records"] == 6
    paired = {r for p in summary["pairs"] for r in (p["x"], p["y"])}
    assert len(paired) == 2 * len(summary["pairs"]) and not paired & probe
    assert sum(summary["pairing"][p]["n"] for p in PAIRINGS) == len(summary["pairs"]) > 0
    sizes = summary["pool_sizes"]
    assert sizes["strong_weak:strong"] == sizes["strong_weak:weak"] and sum(sizes.values()) == 16
    assert set(summary["metrics"]) == {"explicit", "proxy"}
    assert "null_direct:joint" in summary["reward_columns"] and "baseline" in summary["metrics"]["proxy"]
    rows = [json.loads(line) for line in open(stem + "_rewards.jsonl")]
    assert len(rows) == summary["n_texts"] and not any("text" in r or "prompt" in r for r in rows)
    with pytest.raises(SystemExit, match="exists"):
        run_main()
    assert len(run_main.loads) == 1


def test_a_variant_run_gets_its_own_name_and_the_same_pairs(run_main):
    # without the direct directions the probe records are still excluded, so the pairs do not change
    run_main()
    run_main("--directions", "none", "--encodings", "explicit")
    full = json.loads(run_main.out().read_text())
    variant = json.loads(run_main.out("__encodings-explicit__directions-none").read_text())
    assert variant["pairs"] == full["pairs"] and variant["reward_columns"] == ["baseline"]
    assert variant["probe_record_ids"] == full["probe_record_ids"]


def test_bad_requests_stop_before_the_model_loads(run_main, manifest):
    with pytest.raises(SystemExit, match="not in cells.jsonl"):
        run_main("--encodings", "explicit,phonetic")
    with pytest.raises(ValueError, match="n_pairs"):
        run_main("--n-pairs", "1,2")
    with pytest.raises(SystemExit, match="paraphrases"):
        run_main("--paraphrases", "9")
    with pytest.raises(SystemExit, match="n_folds"):
        run_main("--n-folds", "1")
    with open(manifest.parent / "cells.jsonl", "a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="different builds"):
        run_main()
    assert run_main.loads == []


def test_cli_parses():
    args = build_parser().parse_args(["--n-pairs", "4", "--directions", "none", "--model", "M"])
    assert args.n_pairs == "4" and args.directions == "none" and args.model == "M"
