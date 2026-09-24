"""
Tests for `probes/cross_marker_directions.py` on synthetic states with planted structure. No model.
"""

from __future__ import annotations

import torch
import pytest

from pairs.cross_marker import RESPONSE_TYPES
from pairs.factorial import CREDIT_DESIGN as D
from probes.cross_marker_directions import (
    cosine, cross_fitted, did_share, fold_assignment, head_alignment, record_contrasts,
    split_half_cosine, state_index, unit,
)

DIM = 16
E_SEX, E_DEC, E_OVERT, E_INT = (torch.eye(DIM)[i] for i in range(4))


def _states(records, noise=0.0, interaction=1.0, seed=0):
    """Rows + states: the sex marker adds E_SEX to every state after a woman's prompt; a woman's
    *decline* additionally carries ``interaction`` * E_INT; decline carries E_DEC, overt E_OVERT."""
    g = torch.Generator().manual_seed(seed)
    rows, states = [], []
    for rid in records:
        content = torch.randn(DIM, generator=g)                      # the record's own content
        for t in ("t1", "t2"):
            for c in list(D.cells) + [None]:
                for r in RESPONSE_TYPES:
                    h = content.clone()
                    female = c is not None and c[0] == "female"
                    h += E_SEX if female else 0
                    h += E_DEC if r in ("decline", "coded", "overt") else 0
                    h += E_OVERT if r == "overt" else 0
                    h += interaction * E_INT if (female and r == "decline") else 0
                    h += noise * torch.randn(DIM, generator=g)
                    rows.append({"record_id": rid, "template_id": t, "cell": "unmarked" if c is None else list(c),
                                 "response": r})
                    states.append(h)
    return rows, torch.stack(states)


RECS = [f"r{i}" for i in range(12)]


def test_directions_recover_the_planted_structure():
    rows, states = _states(RECS)
    idx = state_index(rows)
    prompt = record_contrasts(states, idx, RECS, "prompt", D, "explicit", "sex").mean(0)
    inter = record_contrasts(states, idx, RECS, "interaction", D, "explicit", "sex").mean(0)
    unfair = record_contrasts(states, idx, RECS, "unfair", D, "explicit").mean(0)
    # the interaction contrast is (A,approve)-(A,decline)-(B,approve)+(B,decline) = -E_INT
    assert cosine(inter, -E_INT) == pytest.approx(1.0, abs=1e-5)
    # overt - decline = E_OVERT, minus the interaction a woman's decline carries (4 of the 9 cells)
    assert torch.allclose(unfair, E_OVERT - 4 / 9 * E_INT, atol=1e-5)
    # prompt: E_SEX on every response, plus E_INT on the decline share of the responses
    assert cosine(prompt, E_SEX) > 0.95 and float(prompt @ E_INT) > 0
    assert record_contrasts(states, idx, RECS, "prompt", D, "explicit", "age").abs().max() < 1e-5


def test_did_share_and_head_alignment():
    rows, states = _states(RECS)
    idx = state_index(rows)
    delta = record_contrasts(states, idx, RECS, "interaction", D, "explicit", "sex").mean(0)
    w = 2.0 * E_INT + 1.0 * E_SEX
    assert did_share(w, delta, E_INT) == pytest.approx(1.0)       # the disparity lives on E_INT
    assert did_share(w, delta, E_SEX) == pytest.approx(0.0)       # reward-relevant, but carries none of it
    assert head_alignment(w, E_SEX)["w_dot_u"] == pytest.approx(1.0)
    assert head_alignment(w, E_OVERT)["cos_w_u"] == pytest.approx(0.0)
    assert did_share(w, torch.zeros(DIM), E_INT) != did_share(w, torch.zeros(DIM), E_INT)   # nan


def test_folds_are_stratified_and_cross_fitting_leaves_each_fold_out():
    strong = {r: i % 3 != 0 for i, r in enumerate(RECS)}
    folds = fold_assignment(RECS, strong, 4, seed=1)
    assert set(folds.values()) == {0, 1, 2, 3}
    for f in range(4):
        members = [r for r in RECS if folds[r] == f]
        assert any(strong[r] for r in members) and any(not strong[r] for r in members)
    assert fold_assignment(RECS, strong, 4, seed=1) == folds
    # a direction that is pure noise per record: each fold's direction must not use its own records
    contrasts = torch.eye(len(RECS), DIM)
    directions = cross_fitted(contrasts, RECS, folds)
    for f, u in directions.items():
        for n, r in enumerate(RECS):
            assert (float(u[n]) == 0.0) == (folds[r] == f)


def test_split_half_reliability_separates_signal_from_noise():
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(200, DIM, generator=g) * 0.5 + E_SEX * 3
    noise = torch.randn(200, DIM, generator=g)
    assert split_half_cosine(signal, seed=0) > 0.95
    assert abs(split_half_cosine(noise, seed=0)) < 0.5
    assert split_half_cosine(signal[:3], seed=0) != split_half_cosine(signal[:3], seed=0)  # nan


def test_missing_unmarked_control_is_skipped_unit_by_unit():
    rows, states = _states(RECS[:4])
    keep = [i for i, r in enumerate(rows) if r["cell"] != "unmarked"]
    rows2 = [rows[i] for i in keep]
    idx = state_index(rows2)
    unfair = record_contrasts(states[keep], idx, RECS[:4], "unfair", D, "explicit").mean(0)
    assert torch.allclose(unfair, E_OVERT - 4 / 8 * E_INT, atol=1e-5)       # 4 of the 8 marked cells


def test_duplicate_state_key_raises():
    rows, _ = _states(RECS[:1])
    with pytest.raises(ValueError, match="duplicate"):
        state_index(rows + rows[:1])


def test_unit():
    assert float(unit(torch.tensor([3.0, 4.0])).norm()) == pytest.approx(1.0)


def test_degenerate_direction_has_undefined_geometry():
    zero = torch.zeros(DIM)
    nan = lambda x: x != x
    assert nan(cosine(zero, E_SEX)) and nan(did_share(E_SEX, E_SEX, zero))
    assert nan(head_alignment(E_SEX, zero)["cos_w_u"])
