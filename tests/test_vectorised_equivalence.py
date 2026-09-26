"""The CPU speedups of 2026-09-26 against their reference definitions: each vectorised path must give the
same numbers as the straightforward computation it replaced (bit for bit where the arithmetic allows).
`tests/test_cross_marker_golden.py` pins the whole output; these pin each piece."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pairs.factorial import CREDIT_DESIGN, EDUCATION_DESIGN
from probes.cross_marker_directions import record_contrasts, state_index
from runners.run_cross_marker import record_contrast_matrix, record_contrasts as pair_record_contrasts
from scoring.cross_marker_metrics import (
    RewardIndex, _auc_boot, _auc_rows, _counts, _effects, auc, cross_marker_metrics, factorial_effects,
    sweep_point,
)
from tests.test_cross_marker_directions import RECS, _states
from tests.test_cross_marker_golden import _data


# --------------------------------------------------------------------------- metrics ------------------
@pytest.mark.parametrize("design", [CREDIT_DESIGN, EDUCATION_DESIGN])
@pytest.mark.parametrize("rounded", [False, True])
def test_effects_equal_the_reference_bit_for_bit(design, rounded):
    # rounded values make many interactions exactly zero in exact arithmetic: the sign of the residual
    # would decide share_negative, so the vectorised sum must round exactly like the reference
    rng = np.random.default_rng(0)
    m = rng.normal(size=(60, len(design.cells)))
    if rounded:
        m = np.round(m, 1)
    fast = _effects(m, design)
    for r in range(len(m)):
        ref = factorial_effects(dict(zip(design.cells, m[r].tolist())), design)
        assert {k: float(v[r]) for k, v in fast.items()} == ref


@pytest.mark.parametrize("ties", [False, True])
def test_bootstrap_auc_equals_ranking_each_replicate(ties):
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(0.3, 1, 40), rng.normal(0, 1, 25)
    if ties:
        pos, neg = np.round(pos, 0), np.round(neg, 0)
    bs, bw = rng.integers(0, 40, size=(300, 40)), rng.integers(0, 25, size=(300, 25))
    fast, cover = _auc_boot(pos, neg, _counts(bs, 40), _counts(bw, 25))
    assert np.array_equal(fast, _auc_rows(pos[bs], neg[bw]))
    assert np.all(cover == 1.0)
    # the full sample (every multiplicity 1) is the plain AUC (up to rounding: `auc` adds two means)
    assert _auc_boot(pos, neg, np.ones((1, 40)), np.ones((1, 25)))[0][0] == pytest.approx(auc(pos, neg), abs=1e-15)


def test_stratified_bootstrap_auc_counts_only_same_stratum_pairs():
    rng = np.random.default_rng(2)
    pos, neg = np.round(rng.normal(0.3, 1, 30), 1), np.round(rng.normal(0, 1, 20), 1)
    bin_p, bin_n = rng.integers(0, 3, 30), rng.integers(0, 3, 20)
    bs, bw = rng.integers(0, 30, size=(50, 30)), rng.integers(0, 20, size=(50, 20))
    fast, cover = _auc_boot(pos, neg, _counts(bs, 30), _counts(bw, 20), bin_p[:, None] == bin_n[None, :])
    for b in range(50):          # brute force over the replicate's pairs
        p, q, bp, bq = pos[bs[b]], neg[bw[b]], bin_p[bs[b]], bin_n[bw[b]]
        same = bp[:, None] == bq[None, :]
        diff = p[:, None] - q[None, :]
        pairs = same.sum()
        assert cover[b] == pairs / (30 * 20)
        assert fast[b] == ((same & (diff > 0)).sum() + 0.5 * (same & (diff == 0)).sum()) / pairs


@pytest.mark.parametrize("axis", ["sex", "age", "marital_status", "intersection"])
def test_sweep_point_equals_the_full_metrics(axis):
    rows, _ = _data(CREDIT_DESIGN, "explicit", 5)
    index = RewardIndex(rows, CREDIT_DESIGN, "explicit")
    point = sweep_point(index, index.values(rows, "baseline"), axis)
    full = cross_marker_metrics(rows, CREDIT_DESIGN, "explicit", reward_key="baseline", n_boot=1)
    assert point["disparity"] == full["margins"]["D"]["strong"]["disparity"][axis]["mean"]
    assert point["auc_marked"] == full["accuracy"]["auc_marked"]


def test_index_rejects_other_rows_and_unknown_cells():
    rows, _ = _data(CREDIT_DESIGN, "explicit", 6)
    index = RewardIndex(rows, CREDIT_DESIGN, "explicit")
    with pytest.raises(ValueError, match="index was built on"):
        index.values(rows[:-1], "baseline")
    bad = [dict(rows[0], cell=["female", 99, "married"])] + rows[1:]
    with pytest.raises(ValueError, match="not a cell of the design"):
        RewardIndex(bad, CREDIT_DESIGN, "explicit")


def test_passing_an_index_changes_nothing():
    rows, _ = _data(EDUCATION_DESIGN, "explicit", 7)
    index = RewardIndex(rows, EDUCATION_DESIGN, "explicit")
    for key in ("baseline", "tied"):
        assert (cross_marker_metrics(rows, EDUCATION_DESIGN, "explicit", reward_key=key, n_boot=50, index=index)
                == cross_marker_metrics(rows, EDUCATION_DESIGN, "explicit", reward_key=key, n_boot=50))


# --------------------------------------------------------------------------- directions ---------------
@pytest.mark.parametrize("kind,axis", [("prompt", "sex"), ("interaction", "age"), ("unfair", None)])
def test_record_contrasts_equal_the_per_record_loop(kind, axis):
    rows, states = _states(RECS, noise=0.3)
    index = state_index(rows)
    fast = record_contrasts(states, index, RECS, kind, CREDIT_DESIGN, "explicit", axis)
    from probes.cross_marker_directions import _terms
    responses = sorted({k[3] for k in index})
    for n, rid in enumerate(RECS):          # the reference: the loop the vectorised form replaced
        plus_idx, minus_idx, units = [], [], 0
        for plus, minus in _terms(kind, axis, CREDIT_DESIGN, "explicit", ["t1", "t2"], responses):
            kp, km = [(rid, *k) for k in plus], [(rid, *k) for k in minus]
            if all(k in index for k in kp + km):
                plus_idx += [index[k] for k in kp]
                minus_idx += [index[k] for k in km]
                units += 1
        ref = (states[plus_idx].float().sum(0) - states[minus_idx].float().sum(0)) / units
        assert torch.allclose(fast[n], ref, atol=1e-5, rtol=1e-6)


def test_pair_record_contrasts_equal_the_per_record_mean():
    torch.manual_seed(0)
    owners = [f"r{random.Random(i).randint(0, 6)}" for i in range(40)]
    pairs = [SimpleNamespace(metadata={"source_record_id": o}) for o in owners]
    pos, neg = torch.randn(40, 8), torch.randn(40, 8)
    ids, fast = pair_record_contrasts(pairs, pos, neg)
    assert ids == list(dict.fromkeys(owners))                          # first-seen order
    for n, rid in enumerate(ids):
        idx = [i for i, o in enumerate(owners) if o == rid]
        assert torch.allclose(fast[n], (pos[idx] - neg[idx]).mean(0), atol=1e-6)
    assert torch.equal(record_contrast_matrix(pairs, pos, neg), fast)


# --------------------------------------------------------------------------- prepare ------------------
def test_batched_token_counts_equal_the_single_ones():
    from runners.run_cross_marker import token_counter
    from tests.test_run_cross_marker import _tokenizer

    count = token_counter(_tokenizer())
    texts = ["the applicant is a woman", "loan credit approve decline approved the applicant", "a"]
    assert count.batch(texts) == [count(t) for t in texts]
    pairs = [("the applicant", "is a woman"), ("loan", "credit approve")]
    assert count.batch(pairs) == [count(p) for p in pairs]


def test_rows_from_the_length_guards_cache_equal_fresh_ones(manifest):
    from pairs.cross_marker import load_cell_blocks
    from runners.run_cross_marker import block_fits, build_rows, resolve_settings, select_records, token_counter
    from scoring.dataset_base import format_conversation
    from substrates.domains import get_domain
    from tests.test_run_cross_marker import _tokenizer

    tok, dom, settings = _tokenizer(), get_domain("credit"), resolve_settings({}, {})
    fmt = lambda p, r: format_conversation(tok, p, r)
    blocks = load_cell_blocks(manifest.parent / "cells.jsonl", CREDIT_DESIGN)
    cache: dict = {}
    selected, _ = select_records(blocks, quality_field="credit_good", encodings=["explicit", "proxy"],
                                 templates=list(dom.template_ids), exclude=set(), n_strong=5, n_weak=5, seed=42,
                                 fits=block_fits("credit", settings, fmt, token_counter(tok), 1024, cache=cache))
    assert cache
    assert (build_rows(selected, "credit", "credit_good", fmt, settings, cache=cache)
            == build_rows(selected, "credit", "credit_good", fmt, settings))


from tests.test_run_cross_marker import manifest  # noqa: E402,F401  (fixture)
