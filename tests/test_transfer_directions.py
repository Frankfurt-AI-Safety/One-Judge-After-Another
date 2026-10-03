"""`probes/transfer_directions.py`: a direction is the battery's mean pair difference, and a held-out fit leaves out
exactly the units that share a name fold or the template."""

from __future__ import annotations

import pytest
import torch

from probes.cross_marker_directions import unit
from probes.transfer_directions import NoFitUnits, UnitStore, mean_unit, random_units

F = frozenset


def _store(n=40, dim=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    diffs = torch.randn(n, dim, generator=g)
    folds = [F({i % 5, (i // 5) % 5}) for i in range(n)]          # one or two folds per unit
    templates = ["t1" if i % 2 else "t2" for i in range(n)]
    records = [f"r{i // 4}" for i in range(n)]
    return UnitStore(diffs, folds, templates, records), diffs, folds, templates, records


def test_the_full_direction_is_the_unit_mean_of_the_units():
    store, diffs, *_ = _store()
    assert torch.allclose(store.direction(), unit(diffs.mean(0)), atol=1e-6)
    assert store.n_units == 40 and store.n_records == 10 and store.templates == ["t1", "t2"] and store.named
    assert store.n_fit() == 40 and store.separation == pytest.approx(float(diffs.mean(0).norm()))
    blind = UnitStore(torch.zeros(4, 3), [F()] * 4, ["t"] * 4, list("abcd"))     # a marker the tokenizer cannot see
    assert blind.separation == 0.0 and float(blind.direction().norm()) == 0.0


def test_a_held_out_fit_uses_exactly_the_units_without_the_folds_and_the_template():
    store, diffs, folds, templates, _ = _store()
    for excluded, template in ((F({0}), None), (F({1, 3}), None), (F(), "t1"), (F({2, 4}), "t2")):
        keep = [i for i in range(40) if not (folds[i] & excluded) and templates[i] != template]
        assert 0 < len(keep) < 40
        assert store.n_fit(excluded, template) == len(keep)
        assert torch.allclose(store.direction(excluded, template), unit(diffs[keep].mean(0)), atol=1e-6)
        # no kept group shares a fold with the excluded ones, or has the template
        kept = [key for key, k in zip(store.keys, store.kept(excluded, template)) if k]
        assert all(not (f & excluded) and t != template for f, t in kept)


def test_units_without_names_are_never_excluded_by_folds():
    diffs = torch.randn(8, 4, generator=torch.Generator().manual_seed(1))
    store = UnitStore(diffs, [F()] * 8, ["t"] * 8, [f"r{i}" for i in range(8)])
    assert not store.named
    assert torch.equal(store.direction(F({0, 1, 2, 3, 4})), store.direction())


def test_record_contrasts_are_each_records_mean_unit():
    store, diffs, _, _, records = _store()
    for k, r in enumerate(dict.fromkeys(records)):
        own = diffs[[i for i, x in enumerate(records) if x == r]].mean(0)
        assert torch.allclose(store.record_contrasts[k], own, atol=1e-6)


def test_an_empty_fit_raises():
    store, *_ = _store()
    with pytest.raises(NoFitUnits, match="name folds"):
        store.direction(F({0, 1, 2, 3, 4}))
    with pytest.raises(NoFitUnits):
        UnitStore(torch.zeros(0, 3), [], [], [])
    with pytest.raises(ValueError, match="units"):
        UnitStore(torch.zeros(2, 3), [F()], ["t", "t"], ["a", "b"])


def test_mean_unit_weighs_every_direction_alike():
    a, b = torch.tensor([10.0, 0.0]), torch.tensor([0.0, 0.1])
    assert torch.allclose(mean_unit([a, b]), torch.tensor([0.5, 0.5]) / 0.5 ** 0.5, atol=1e-6)


def test_random_units_are_unit_length_and_seeded():
    r = random_units(16, 3, seed=7)
    assert r.shape == (3, 16) and torch.allclose(r.norm(dim=1), torch.ones(3), atol=1e-6)
    assert torch.equal(r, random_units(16, 3, seed=7)) and not torch.equal(r, random_units(16, 3, seed=8))
    assert random_units(16, 0, seed=7).shape == (0, 16)
