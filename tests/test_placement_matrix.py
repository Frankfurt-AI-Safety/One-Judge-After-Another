"""
Tests for `scoring/placement_matrix.py` (the placement matrix's statistics: ratios of sums over pairs, one shared
bootstrap, change and gap) and `scoring.cross_marker_metrics.record_axis_effect`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pairs.factorial import CREDIT_DESIGN
from scoring.cross_marker_metrics import RewardIndex, factorial_effects, record_axis_effect, sweep_point
from scoring.placement_matrix import cell_table, ratio_summary, shared_draws, unit_sums


class TestUnitSums:
    def test_sums_and_counts_per_unit_in_unit_order(self):
        sums, counts = unit_sums([("b", 1.0), ("a", 2.0), ("b", 3.0)], ["a", "b", "c"])
        assert sums.tolist() == [2.0, 4.0, 0.0] and counts.tolist() == [1, 2, 0]

    def test_refuses_unknown_units_and_non_finite_values(self):
        with pytest.raises(KeyError, match="unknown unit"):
            unit_sums([("z", 1.0)], ["a"])
        with pytest.raises(ValueError, match="non-finite"):
            unit_sums([("a", float("nan"))], ["a"])
        with pytest.raises(ValueError, match="duplicate"):
            unit_sums([], ["a", "a"])


class TestRatioSummary:
    def test_equal_counts_give_the_plain_mean_and_its_bootstrap(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=40)
        draws = shared_draws(40, 500, 1)
        s = ratio_summary(v, np.ones(40, dtype=np.int64), draws)
        boot = v[draws].mean(axis=1)
        assert s["mean"] == pytest.approx(v.mean())
        assert (s["ci_low"], s["ci_high"]) == pytest.approx((np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
        assert s["n"] == s["n_units"] == 40 and s["n_boot_valid"] == 500

    def test_uneven_counts_are_a_ratio_of_sums_not_a_mean_of_unit_means(self):
        # unit 0 holds two records (1, 3), unit 1 one (8): the record mean 4, not the unit mean (2 + 8) / 2
        s = ratio_summary(np.array([4.0, 8.0]), np.array([2, 1]), shared_draws(2, 200, 0))
        assert s["mean"] == pytest.approx(4.0) and s["n"] == 3 and s["n_units"] == 2

    def test_units_without_records_count_for_nothing_and_empty_replicates_drop_out(self):
        sums, counts = np.array([1.0, 0.0, 3.0]), np.array([1, 0, 1])
        draws = np.array([[0, 2, 1], [1, 1, 1], [0, 0, 0]])
        s = ratio_summary(sums, counts, draws)
        assert s["mean"] == pytest.approx(2.0) and s["n_units"] == 2 and s["n_boot_valid"] == 2

    def test_one_counted_unit_has_no_interval(self):
        s = ratio_summary(np.array([2.0, 0.0]), np.array([1, 0]), shared_draws(2, 50, 0))
        assert s["mean"] == 2.0 and s["ci_low"] != s["ci_low"] and s["ci_high"] != s["ci_high"]


class TestCellTable:
    def _cells(self):
        rng = np.random.default_rng(3)
        base = rng.normal(-1.0, 1.0, size=30)
        counts = np.ones(30, dtype=np.int64)
        own = base * 0.1                          # the own direction removes 90%
        other = base * 0.6 + rng.normal(0, 0.05, size=30)
        return base, counts, {"own": own, "other": other}

    def test_change_and_gap(self):
        base, counts, nulled = self._cells()
        draws = shared_draws(30, 300, 0)
        t = cell_table((base, counts), nulled, "own", draws)
        assert t["own_row"] == "own" and t["baseline"]["mean"] == pytest.approx(base.mean())
        for name, sums in nulled.items():
            assert t["rows"][name]["nulled"]["mean"] == pytest.approx(sums.mean())
            assert t["rows"][name]["change"]["mean"] == pytest.approx((sums - base).mean())
        assert t["rows"]["other"]["gap"]["mean"] == pytest.approx((nulled["own"] - nulled["other"]).mean())

    def test_the_diagonal_gap_is_exactly_zero(self):
        base, counts, nulled = self._cells()
        gap = cell_table((base, counts), nulled, "own", shared_draws(30, 300, 0))["rows"]["own"]["gap"]
        assert (gap["mean"], gap["ci_low"], gap["ci_high"]) == (0.0, 0.0, 0.0)

    def test_the_gap_interval_is_paired(self):
        # two rows whose changes differ by the same amount in every unit: the gap's interval is that amount alone,
        # however much the changes themselves vary between units (unpaired, it would be as wide as theirs)
        base, counts, _ = self._cells()
        rng = np.random.default_rng(9)
        own = base + rng.normal(0.5, 2.0, size=30)
        other = own - 0.25
        t = cell_table((base, counts), {"own": own, "other": other}, "own", shared_draws(30, 300, 0))
        gap, change = t["rows"]["other"]["gap"], t["rows"]["other"]["change"]
        assert gap["ci_low"] == pytest.approx(0.25) and gap["ci_high"] == pytest.approx(0.25)
        assert change["ci_high"] - change["ci_low"] > 0.5

    def test_refuses_a_missing_own_row_and_other_units(self):
        base, counts, nulled = self._cells()
        draws = shared_draws(30, 10, 0)
        with pytest.raises(KeyError, match="own row"):
            cell_table((base, counts), {"other": nulled["other"]}, "own", draws)
        with pytest.raises(ValueError, match="units"):
            cell_table((base, counts), {"own": nulled["own"][:29]}, "own", draws)


class TestRecordAxisEffect:
    def _rows(self, responses):
        rng = np.random.default_rng(5)
        rows = []
        for rec, strong in (("r0", True), ("r1", False), ("r2", True)):
            for template in ("t1", "t2"):
                for cell in CREDIT_DESIGN.cells:
                    for resp in responses:
                        rows.append({"record_id": rec, "template_id": template, "encoding": "explicit",
                                     "cell": list(cell), "response": resp, "strong": strong,
                                     "reward": float(rng.normal())})
        return rows

    @pytest.mark.parametrize("axis", ["sex", "age", "marital_status", "intersection"])
    def test_decision_disparity_per_record_matches_the_reference(self, axis):
        rows = self._rows(("approve", "decline"))
        index = RewardIndex(rows, CREDIT_DESIGN, "explicit")
        eff = record_axis_effect(index, index.values(rows, "reward"), axis)
        key = "corner" if axis == "intersection" else f"main:{axis}"
        for k, rec in enumerate(index.records):
            d = {}
            for cell in CREDIT_DESIGN.cells:
                r = {resp: np.mean([x["reward"] for x in rows if x["record_id"] == rec and tuple(x["cell"]) == cell
                                    and x["response"] == resp]) for resp in ("approve", "decline")}
                d[cell] = r["approve"] - r["decline"]
            assert eff[k] == pytest.approx(factorial_effects(d, CREDIT_DESIGN)[key])
        strong = [k for k in range(len(index.records)) if index.strong[k]]
        assert eff[strong].mean() == pytest.approx(
            sweep_point(index, index.values(rows, "reward"), axis)["disparity"])

    def test_without_a_margin_it_is_the_direct_gap_of_the_one_response(self):
        rows = self._rows(("document",))
        index = RewardIndex(rows, CREDIT_DESIGN, "explicit")
        eff = record_axis_effect(index, index.values(rows, "reward"), "sex", margin=None)
        rec = index.records[0]
        v = {cell: np.mean([x["reward"] for x in rows if x["record_id"] == rec and tuple(x["cell"]) == cell])
             for cell in CREDIT_DESIGN.cells}
        assert eff[0] == pytest.approx(factorial_effects(v, CREDIT_DESIGN)["main:sex"])
        with pytest.raises(ValueError, match="one response"):
            two = self._rows(("approve", "decline"))
            i2 = RewardIndex(two, CREDIT_DESIGN, "explicit")
            record_axis_effect(i2, i2.values(two, "reward"), "sex", margin=None)


class TestShortfall:
    def _table(self, sign):
        rng = np.random.default_rng(11)
        base = sign * rng.normal(1.0, 0.2, size=30)
        own, weak = base * 0.1, base * 0.7          # the own row removes 90%, the other row 30%
        return cell_table((base, np.ones(30, dtype=np.int64)), {"own": own, "weak": weak}, "own",
                          shared_draws(30, 300, 0))

    @pytest.mark.parametrize("sign", [-1.0, 1.0])
    def test_positive_whenever_a_row_removes_less_than_the_own_one(self, sign):
        t = self._table(sign)
        weak = t["rows"]["weak"]
        assert t["orientation"] == {"baseline_sign": sign, "baseline_ci_covers_zero": False}
        assert weak["shortfall"]["mean"] == pytest.approx(-sign * weak["gap"]["mean"])
        assert weak["shortfall"]["mean"] == pytest.approx(0.6 * abs(t["baseline"]["mean"]))     # 90% − 30% removed
        assert weak["shortfall"]["ci_low"] > 0 and weak["shortfall"]["ci_low"] <= weak["shortfall"]["ci_high"]
        assert t["rows"]["own"]["shortfall"]["mean"] == 0.0

    def test_no_orientation_without_a_baseline(self):
        base = np.zeros(10)
        t = cell_table((base, np.ones(10, dtype=np.int64)), {"own": base + 0.1}, "own", shared_draws(10, 50, 0))
        assert t["orientation"]["baseline_sign"] == 0.0 and t["orientation"]["baseline_ci_covers_zero"]
        assert t["rows"]["own"]["shortfall"]["mean"] != t["rows"]["own"]["shortfall"]["mean"]
