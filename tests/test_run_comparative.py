"""
Tests for `runners/run_comparative.py`: settings, the pairing inputs (candidates, probe exclusion, the name and
length checks), and — on tiny random RMs (CPU) — scoring with every direction, the gated head's ``gate_fixed``,
and `main` end to end on a generator-style credit manifest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from pairs.comparative import KINDS, PAIRINGS, UNMARKED, RecordPair, contrast_axes, draw_pairs
from pairs.cross_marker import load_cell_blocks
from pairs.factorial import CREDIT_DESIGN
from runners.run_comparative import (
    DEFAULTS, build_parser, build_rows, candidates_from, compatibility, default_out, parse_n_pairs,
    resolve_settings, score_encoding,
)
from runners.run_cross_marker import token_counter
from scoring.comparative_metrics import comparative_metrics
from scoring.dataset_base import format_conversation
from scoring.experiment import ExperimentConfig
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
        with pytest.raises(ValueError, match="directions"):
            resolve_settings({}, {"directions": ["prompt"]})

    def test_config_files(self):
        for name in CONFIGS:
            cfg = ExperimentConfig.from_yaml(f"configs/demographic_{name}.yaml")
            s = resolve_settings(cfg.extra, {})
            assert s["directions"] == ["direct", "own"] and s["include_unmarked"] is True
            assert set(s["n_pairs"]) == set(PAIRINGS) and cfg.dataset_source.endswith("pairs.jsonl")
            # two essays per prompt: only the education config raises max_length
            assert cfg.max_length == (4096 if name.startswith("edu") else 2048)

    def test_default_out(self):
        assert str(default_out("credit", "data/demographic/credit/pairs.jsonl", "org/M")) == \
            "artifacts/results/demographic/comparative_credit_M.json"
        assert default_out("education", "data/demographic/education/asap2/pairs.jsonl", "M").name == \
            "comparative_education_asap2_M.json"


# --------------------------------------------------------------------------- pairing inputs ----------
def _flat(blocks):
    return [b for per in blocks.values() for b in per.values()]


class TestCandidates:
    def test_probe_records_and_incomplete_records_are_out(self):
        blocks = _flat(_blocks("credit", [("s1", True), ("s2", True), ("w1", False)]))
        blocks = [b for b in blocks if not (b.record_id == "s2" and b.template_id == "credit_v2")]
        by_record, cand, rep = candidates_from(blocks, quality_field="credit_good", match_field=None,
                                               encodings=["explicit"], templates=["credit_v1", "credit_v2"],
                                               exclude={"w1"})
        assert cand == {"s1": (True, None)}
        assert rep == {"records_in_cells": 3, "excluded_probe_records": 1, "incomplete_records": 1}
        assert set(by_record["s1"]) == {("explicit", "credit_v1"), ("explicit", "credit_v2")}

    def test_the_match_field_must_exist(self):
        blocks = _flat(_blocks("cv", [("a", True)]))
        with pytest.raises(KeyError, match="target_role"):
            candidates_from(blocks, quality_field="qualified", match_field="target_role", encodings=["explicit"],
                            templates=["bios_v1", "bios_v2"], exclude=set())
        blocks = [dataclasses.replace(b, real_fields={**b.real_fields, "target_role": "nurse"}) for b in blocks]
        _, cand, _ = candidates_from(blocks, quality_field="qualified", match_field="target_role",
                                     encodings=["explicit"], templates=["bios_v1", "bios_v2"], exclude=set())
        assert cand["a"][1] == "nurse"


def _tok_and_fmt():
    from tests.test_run_cross_marker import _tokenizer
    tok = _tokenizer()
    return tok, (lambda p, r: format_conversation(tok, p, r))


class TestCompatibility:
    def test_shared_names_and_length(self):
        tok, fmt = _tok_and_fmt()
        blocks = _flat(_blocks("credit", [("s1", True), ("s2", True)], encoding="proxy"))
        by_record, _, _ = candidates_from(blocks, quality_field="credit_good", match_field=None,
                                          encodings=["proxy"], templates=["credit_v1", "credit_v2"], exclude=set())
        settings = resolve_settings({}, {})
        check = compatibility(by_record, "credit", settings, fmt, token_counter(tok), 10_000)
        pair = RecordPair("strong_strong", "s1", "s2", 0)
        names = lambda r: set().union(*(b.names for b in by_record[r].values()))
        assert check(pair) == ("shared_name" if names("s1") & names("s2") else None)
        # force a shared name
        same = {k: dataclasses.replace(b, names=by_record["s1"][k].names) for k, b in by_record["s2"].items()}
        assert compatibility({**by_record, "s2": same}, "credit", settings, fmt, token_counter(tok), 10_000)(pair) \
            == "shared_name"
        # a pair with any text over max_length is refused
        plain = {r: {k: dataclasses.replace(b, names=frozenset()) for k, b in per.items()}
                 for r, per in by_record.items()}
        assert compatibility(plain, "credit", settings, fmt, token_counter(tok), 20)(pair) == "too_long"
        assert compatibility(plain, "credit", settings, fmt, token_counter(tok), 10_000)(pair) is None


# --------------------------------------------------------------------------- scoring -----------------
def _scored(manifest, model, tok, directions=("own",), n_folds=2):
    fmt = lambda p, r: format_conversation(tok, p, r)
    settings = resolve_settings({}, {"n_pairs": 100, "n_folds": n_folds, "directions": list(directions),
                                     "encodings": ["explicit"], "paraphrases": 1})
    blocks = load_cell_blocks(manifest.parent / "cells.jsonl", CREDIT_DESIGN)
    by_record, cand, _ = candidates_from(blocks, quality_field="credit_good", match_field=None,
                                         encodings=["explicit"], templates=["credit_v1", "credit_v2"], exclude=set())
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
    import torch

    from tests.test_run_cross_marker import _model, _tokenizer

    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    model, tok = _model(), _tokenizer()
    pairs, rows, convs, axes, settings = _scored(manifest, model, tok)
    torch.manual_seed(1)
    direct = {a: torch.nn.functional.normalize(torch.randn(32), dim=0) for a in axes}
    columns, geometry, saved = score_encoding(model, tok, rows, convs, direct, axes, settings, batch_size=16,
                                              max_length=1024, direct_reliability={a: 0.9 for a in axes},
                                              show_progress=False)
    assert columns == (["baseline"] + [f"null_direct:{a}" for a in axes] + ["null_direct:joint"]
                       + [f"null_own:{a}" for a in axes])
    assert all(c in r for r in rows for c in columns)
    # cross-fitted: every pair is nulled with the direction of the folds it is not in
    folds = saved["folds"]
    assert set(folds) == {p.pair_id for p in pairs} and len(set(folds.values())) == 2
    for a in axes:
        assert set(saved["cross_fitted"][f"own:{a}"]) == {0, 1}
    assert set(geometry["own_vs_direct"]) == set(axes)
    # the tiny random model's last-token state does not register a marker swap ~600 tokens back (the contrasts
    # are exactly 0 even in float32), so the own directions are degenerate here and their cosines NaN; the
    # contrast itself is checked on synthetic states (tests/test_comparative.py), real shapes on an MPS smoke
    assert all(v["cosine"] != v["cosine"] or -1 <= v["cosine"] <= 1 for v in geometry["own_vs_direct"].values())
    assert set(geometry["reliability"]) == {f"own:{a}" for a in axes} | {f"direct:{a}" for a in axes}
    m = comparative_metrics(rows, "null_own:sex", baseline_key="baseline", n_boot=20)
    assert "marker_effect_change" in m["all"]["sex"]["merit"]


def test_gate_fixed_on_a_gated_model(manifest, monkeypatch):
    from tests.test_qrm import _model, _tokenizer

    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    tok = _tokenizer()
    model = _model(tok)
    pairs, rows, convs, axes, settings = _scored(manifest, model, tok, directions=())
    columns, geometry, _ = score_encoding(model, tok, rows, convs, {}, axes, settings, batch_size=16,
                                          max_length=1024, show_progress=False)
    assert columns == ["baseline", "gate_fixed"] and geometry["head"].startswith("quantile_gated")
    unmarked = [r for r in rows if r["axis"] == UNMARKED]
    assert all(r["gate_fixed"] == pytest.approx(r["baseline"]) for r in unmarked)
    assert any(abs(r["gate_fixed"] - r["baseline"]) > 1e-4 for r in rows if r["axis"] != UNMARKED)


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
    return _run


def test_main_end_to_end(run_main, manifest, tmp_path):
    run_main()
    out = tmp_path / "artifacts/results/demographic" / f"comparative_credit_{manifest.parent.name}_Tiny-RM.json"
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
    assert set(summary["metrics"]) == {"explicit", "proxy"}
    assert "null_direct:joint" in summary["reward_columns"] and "baseline" in summary["metrics"]["proxy"]
    rows = [json.loads(line) for line in open(stem + "_rewards.jsonl")]
    assert len(rows) == summary["n_texts"] and not any("text" in r or "prompt" in r for r in rows)
    with pytest.raises(SystemExit, match="exists"):
        run_main()
    assert len(run_main.loads) == 1


def test_bad_requests_stop_before_the_model_loads(run_main, manifest):
    with pytest.raises(SystemExit, match="not in cells.jsonl"):
        run_main("--encodings", "explicit,phonetic")
    with pytest.raises(ValueError, match="n_pairs"):
        run_main("--n-pairs", "1,2")
    with open(manifest.parent / "cells.jsonl", "a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="different builds"):
        run_main()
    assert run_main.loads == []


def test_cli_parses():
    args = build_parser().parse_args(["--n-pairs", "4", "--directions", "none", "--model", "M"])
    assert args.n_pairs == "4" and args.directions == "none" and args.model == "M"
