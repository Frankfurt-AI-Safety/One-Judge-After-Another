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


# --------------------------------------------------------------------------- nulling, end to end
def _states(n=10, d=16, seed=0):
    import torch

    g = torch.Generator().manual_seed(seed)
    shift = torch.randn(d, generator=g)
    return torch.randn(n, d, generator=g) + shift, torch.randn(n, d, generator=g)


def test_in_sample_nulling_zeroes_the_mean_gap_leave_one_out_does_not(monkeypatch):
    import torch

    from runners import run_realfield as rf

    w = torch.randn(16, generator=torch.Generator().manual_seed(1))

    def rewards(model, h, dtype, direction=None, gates=None, **kw):
        base = h.float() @ w
        if direction is None:
            return base, base
        u = direction / direction.norm()
        return base, (h.float() - (h.float() @ u)[:, None] * u) @ w

    monkeypatch.setattr(rf, "rewards_from_hidden", rewards)
    h_a, h_b = _states()
    in_sample = rf.pair_gaps(None, h_a, h_b, torch.float32, direction=(h_a - h_b).mean(0))
    assert abs(sum(in_sample) / len(in_sample)) < 1e-5              # the old "nulled" gap: 0 by construction
    loo = rf.leave_one_out_gaps(None, h_a, h_b, torch.float32)
    diffs = h_a - h_b
    for k in (0, 7):                                                  # pair k nulled with the others' direction
        u = (diffs.sum(0) - diffs[k]) / (diffs.sum(0) - diffs[k]).norm()
        expect = float(((diffs[k] - (diffs[k] @ u) * u)) @ w)
        assert loo[k] == pytest.approx(expect, abs=1e-5)
    assert abs(sum(loo) / len(loo)) > 1e-3


def _german_data(path, n=12):
    """A synthetic german.data: n male records, divorced (A91) and married (A93) in turn, alike on MATCH_KEYS."""
    path.write_text("".join(f"A14 12 A34 A43 {1000 + 37 * i} A61 A73 2 {'A91' if i % 2 == 0 else 'A93'} A101 2 A121 "
                            f"{30 + i} A143 A152 1 A173 1 A192 A201 1\n" for i in range(n)))
    return path


def test_main_end_to_end_on_a_tiny_model(manifest, tmp_path, monkeypatch):
    import hashlib
    import json

    import yaml

    from runners import run_realfield as rf
    from tests.test_run_cross_marker import _model, _tokenizer

    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    raw = _german_data(tmp_path / "german.data")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM",
                                   "dataset_source": str(manifest), "probe_records": 6, "batch_size": 16,
                                   "max_length": 1024, "extra": {"domain": "credit"}}))
    loads = []

    def load_model(self):
        loads.append(1)
        self.model, self.tokenizer = _model(), _tokenizer()
        self.config.model_revision = "abc123"

    monkeypatch.setattr(rf.DemographicBiasExperiment, "load_model", load_model)
    monkeypatch.setattr("sys.argv", ["run_realfield.py", "--config", str(cfg), "--raw", str(raw), "--n-boot", "50"])
    rf.main()
    result = json.loads((tmp_path / "artifacts/results/demographic/realfield_marital_Tiny-RM.json").read_text())
    assert result["n_per_group"] == 6 and result["n_unmatched"] == 0
    assert result["meta"]["config"]["model_revision"] == "abc123"
    assert result["meta"]["data"]["german.data"]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert result["meta"]["data"]["pairs.jsonl"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    gap = result["reward_gap_divorced_minus_married"]
    assert set(gap) == {"baseline", "nulled_heldout", "nulled_synthetic_marital"}
    assert set(result["reliability"]) == {"real_marital", "synthetic_marital", "synthetic_sex"}
    with pytest.raises(SystemExit, match="exists"):
        rf.main()
    assert len(loads) == 1
