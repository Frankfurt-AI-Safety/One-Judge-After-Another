"""The real-field arm's matched groups (audit item 4.2): divorced vs married men matched exactly on the
good-credit label, checking account and dependents, with the remaining imbalance reported."""

from __future__ import annotations

import dataclasses

import pytest

from runners.run_realfield import (
    CATEGORICAL_FIELDS, MATCH_KEYS, NUMERIC_FIELDS, balance_table, chance_balance, match_groups,
)
from tests.test_credit_pipeline import _fake_record

_CHECKING = ("no checking account", "less than 0 EUR")


def _rec(rid, *, good=True, checking=_CHECKING[0], dependents="0 to 2", **changes):
    return dataclasses.replace(_fake_record(rid), credit_good=good, checking=checking,
                               dependents=dependents, **changes)


def _groups():
    treated = [_rec(f"d{i}", good=i % 2 == 0, checking=_CHECKING[i % 2]) for i in range(6)]
    pool = [_rec(f"m{i}", good=i % 3 != 0, checking=_CHECKING[i % 2],
                 dependents="3 or more" if i % 5 == 0 else "0 to 2", duration_months=6 + i)
            for i in range(40)]
    return treated, pool


def test_pairs_agree_on_every_match_key_and_no_control_is_reused():
    treated, pool = _groups()
    pairs, unmatched = match_groups(treated, pool, MATCH_KEYS, seed=42)
    assert len(pairs) + len(unmatched) == len(treated)
    for t, c in pairs:
        assert all(getattr(t, k) == getattr(c, k) for k in MATCH_KEYS)
    controls = [c.source_record_id for _, c in pairs]
    assert len(controls) == len(set(controls))


def test_deterministic_and_seeded():
    treated, pool = _groups()
    ids = lambda seed: [(t.source_record_id, c.source_record_id)
                        for t, c in match_groups(treated, pool, MATCH_KEYS, seed)[0]]
    assert ids(42) == ids(42)
    assert ids(42) != ids(7)
    # the input order does not matter, only the seed
    assert ids(42) == [(t.source_record_id, c.source_record_id)
                       for t, c in match_groups(treated[::-1], pool[::-1], MATCH_KEYS, 42)[0]]


def test_a_treated_record_without_a_control_is_counted_not_forced():
    treated = [_rec("d0", good=False, dependents="3 or more"), _rec("d1")]
    pool = [_rec("m0"), _rec("m1")]
    pairs, unmatched = match_groups(treated, pool, MATCH_KEYS, seed=42)
    assert [t.source_record_id for t, _ in pairs] == ["d1"]
    assert [t.source_record_id for t in unmatched] == ["d0"]


def test_controls_run_out_without_replacement():
    treated = [_rec(f"d{i}") for i in range(3)]
    pairs, unmatched = match_groups(treated, [_rec("m0"), _rec("m1")], MATCH_KEYS, seed=42)
    assert len(pairs) == 2 and len(unmatched) == 1


def test_balance_is_zero_on_the_match_keys_after_matching():
    treated, pool = _groups()
    pairs, _ = match_groups(treated, pool, MATCH_KEYS, seed=42)
    table = balance_table([t for t, _ in pairs], [c for _, c in pairs])
    assert all(table["total_variation"][k] == 0 for k in MATCH_KEYS)
    assert set(table["total_variation"]) == set(CATEGORICAL_FIELDS)
    assert set(table["standardised_mean_difference"]) == set(NUMERIC_FIELDS)


def test_balance_table_scales():
    same = [_rec(f"a{i}", duration_months=6 + i) for i in range(5)]
    zero = balance_table(same, same)
    assert all(v == 0 for part in zero.values() for v in part.values())
    disjoint = balance_table([_rec("a", good=True), _rec("b", good=True)],
                             [_rec("c", good=False), _rec("d", good=False)])
    assert disjoint["total_variation"]["credit_good"] == pytest.approx(1.0)
    shifted = balance_table([_rec("a", duration_months=10), _rec("b", duration_months=12)],
                            [_rec("c", duration_months=20), _rec("d", duration_months=22)])
    assert shifted["standardised_mean_difference"]["duration_months"] < -5


def test_chance_balance_is_the_floor_of_one_population():
    _, pool = _groups()
    ref = chance_balance(pool, n=10, seed=42, draws=20)
    assert ref == chance_balance(pool, n=10, seed=42, draws=20)
    assert set(ref["total_variation"]) == set(CATEGORICAL_FIELDS)
    assert all(v >= 0 for part in ref.values() for v in part.values())
    assert ref["total_variation"]["credit_good"] > 0          # two samples of 10 differ by chance
