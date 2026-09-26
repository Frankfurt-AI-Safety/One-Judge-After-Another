"""Regression pin for the CPU-side speedups of 2026-09-26: every number the cross-marker metrics, the placement
check and the mechanism layer produce on fixed synthetic data, compared with the output of the code before the
speedups (``tests/golden/cross_marker.json``). The metrics must match to floating-point rounding; the mechanism
layer (bf16 model states, float32 contrasts) to a looser tolerance, since summation order may change.

Regenerate only on purpose, after a deliberate change of what is computed:
    REGEN_GOLDEN=1 python -m pytest tests/test_cross_marker_golden.py
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import pytest

from pairs.factorial import CREDIT_DESIGN, EDUCATION_DESIGN
from scoring.cross_marker_metrics import cross_marker_metrics, placement_check
from tests.test_cross_marker_metrics import _rows
from tests.test_run_cross_marker import _model, _tokenizer, manifest  # noqa: F401  (fixture)

GOLDEN = Path(__file__).parent / "golden" / "cross_marker.json"
N_BOOT = 300


def _data(design, encoding, seed):
    """Rows of 14 strong and 9 weak records over two templates, with one weak record scored on t1 only; two
    reward columns (continuous, and rounded to 0.1 so ties occur) and document lengths with ties."""
    rng = random.Random(seed)
    records = {**{f"s{i}": True for i in range(14)}, **{f"w{i}": False for i in range(9)}}
    rows = _rows(lambda *a: 0.0, records, design=design, encoding=encoding, templates=("t1", "t2"))
    rows = [r for r in rows if not (r["record_id"] == "w8" and r["template_id"] == "t2")]
    length = {(rid, t): rng.choice([180, 200, 220, 240, 260, 300]) for rid in records for t in ("t1", "t2")}
    for r in rows:
        bump = 0.4 * (r["response"] == "approve") * (1 if r["strong"] else -1)
        r["baseline"] = rng.gauss(0, 1) + bump
        r["tied"] = round(rng.gauss(0, 0.6) + bump, 1)
        r["doc_tokens"] = length[(r["record_id"], r["template_id"])]
    direct = [{"record_id": rid, "template_id": t, "encoding": encoding, "cell": list(c), "strong": s,
               "baseline": rng.gauss(0, 1), "tied": round(rng.gauss(0, 1), 1)}
              for rid, s in records.items() for t in ("t1", "t2") for c in design.cells]
    return rows, direct


def _metrics():
    out = {}
    for name, design, encoding, seed in (("credit_explicit", CREDIT_DESIGN, "explicit", 1),
                                         ("credit_proxy", CREDIT_DESIGN, "proxy", 2),
                                         ("education_explicit", EDUCATION_DESIGN, "explicit", 3)):
        rows, direct = _data(design, encoding, seed)
        for col in ("baseline", "tied"):
            out[f"{name}/{col}/metrics"] = cross_marker_metrics(rows, design, encoding, reward_key=col,
                                                               n_boot=N_BOOT, seed=42)
            out[f"{name}/{col}/placement"] = placement_check(direct, rows, design, encoding, reward_key=col,
                                                            n_boot=N_BOOT, seed=42)
    return out


def _mechanism(manifest):
    """score_encoding end to end on the tiny model: every reward column, the geometry and the α-sweep."""
    from pairs.cross_marker import load_cell_blocks
    from runners.run_cross_marker import (
        block_fits, build_direct_rows, build_rows, direct_directions, resolve_settings, score_encoding,
        select_records, token_counter,
    )
    from scoring.dataset_base import format_conversation
    from substrates.domains import get_domain

    dom, model, tok = get_domain("credit"), _model(), _tokenizer()
    directions, probe_ids, meta = direct_directions(
        model, tok, dom, str(manifest), ["explicit"], probe_size=8, split_seed=42, batch_size=8,
        device="cpu", max_length=1024, probe_records=4)
    settings = resolve_settings({}, {"n_strong": 6, "n_weak": 6, "n_folds": 3, "alphas": [0.0, 0.5, 1.0]})
    fmt = lambda p, r: format_conversation(tok, p, r)
    blocks = load_cell_blocks(manifest.parent / "cells.jsonl", CREDIT_DESIGN)
    selected, _ = select_records(blocks, quality_field="credit_good", encodings=["explicit"],
                                 templates=list(dom.template_ids), exclude=probe_ids, n_strong=6, n_weak=6,
                                 seed=42, fits=block_fits("credit", settings, fmt, token_counter(tok), 1024))
    rows, convs = build_rows(selected, "credit", "credit_good", fmt, settings)
    drows, dconvs = build_direct_rows(selected, CREDIT_DESIGN, "credit_good", dom.assessment_prompt, fmt)
    columns, geometry, sweep, _ = score_encoding(model, tok, "explicit", CREDIT_DESIGN, rows, convs, drows,
                                                 dconvs, directions["explicit"], settings, batch_size=16,
                                                 max_length=1024, show_progress=False)
    return {"columns": columns,
            "rewards": {c: [r[c] for r in rows] for c in columns},
            "direct_rewards": {c: [r[c] for r in drows] for c in columns},
            "geometry": {k: v for k, v in geometry.items() if k != "head"},
            "sweep": sweep,
            "split_half": {k: v["split_half_cosine"] for k, v in meta.items()}}


def _close(a, b, rel, abs_, path="$"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{path}: keys {list(a)} != {list(b)}"
        for k in a:
            _close(a[k], b[k], rel, abs_, f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, rel, abs_, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), f"{path}: {a} != {b}"
        else:
            assert b == pytest.approx(a, rel=rel, abs=abs_), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _roundtrip(obj):
    return json.loads(json.dumps(obj, default=str))


@pytest.fixture(scope="module")
def golden():
    if os.environ.get("REGEN_GOLDEN") or not GOLDEN.exists():
        pytest.skip("golden file being regenerated")
    return json.loads(GOLDEN.read_text())


def test_regenerate(manifest, monkeypatch):
    if not os.environ.get("REGEN_GOLDEN"):
        pytest.skip("set REGEN_GOLDEN=1 to rewrite tests/golden/cross_marker.json")
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(_roundtrip({"metrics": _metrics(), "mechanism": _mechanism(manifest)})))


def test_metrics_match_the_pinned_output(golden):
    _close(golden["metrics"], _roundtrip(_metrics()), rel=1e-9, abs_=1e-12)


def test_mechanism_matches_the_pinned_output(golden, manifest, monkeypatch):
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    _close(golden["mechanism"], _roundtrip(_mechanism(manifest)), rel=1e-3, abs_=2e-3)
