"""
Unit tests for `scoring/cross_marker_metrics.py` on planted effects, and for the credit common-sense
reference scorer. No model required.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from pairs.cross_marker import RESPONSE_TYPES
from pairs.factorial import CREDIT_DESIGN, DESIGNS, EDUCATION_DESIGN
from scoring.cross_marker_metrics import (
    auc, cross_marker_metrics, factorial_effects, margins, record_table, summarize,
    summarize_balanced,
)

D = CREDIT_DESIGN   # sex x age x marital status; pole A = female, 30, married


def _x(design, cell, axis):
    return 1 if cell[design.axes.index(axis)] == design.factors[axis][0] else -1


def _rows(reward, records, design=D, encoding="explicit", templates=("t1",), unmarked=True,
          responses=RESPONSE_TYPES):
    """Reward-table rows; ``reward(rid, cell, response, template)`` (cell None = unmarked).
    ``records`` maps record id -> strong."""
    cells = list(design.cells) + ([None] if unmarked else [])
    return [{"record_id": rid, "template_id": t, "encoding": encoding,
             "cell": "unmarked" if c is None else list(c),   # as the JSONL writes it
             "response": r, "strong": s, "reward": reward(rid, c, r, t)}
            for rid, s in records.items() for t in templates for c in cells for r in responses]


def _records(n_strong=6, n_weak=6):
    out = {f"s{i}": True for i in range(n_strong)}
    out.update({f"w{i}": False for i in range(n_weak)})
    return out


# --------------------------------------------------------------------------- decomposition -----------
class TestDecomposition:
    def test_additive_model_has_no_interactions(self):
        b = {"sex": 0.3, "age": -0.2, "marital_status": 0.5}
        vals = {c: 1.0 + sum(b[a] * _x(D, c, a) for a in D.axes) for c in D.cells}
        e = factorial_effects(vals, D)
        for a in D.axes:
            assert e[f"main:{a}"] == pytest.approx(2 * b[a])
        for k in ("interaction:sex_x_age", "interaction:sex_x_marital_status",
                  "interaction:age_x_marital_status", "three_way", "additivity_gap"):
            assert e[k] == pytest.approx(0.0, abs=1e-12)
        assert e["corner"] == pytest.approx(sum(e[f"main:{a}"] for a in D.axes))

    def test_two_way_cancels_in_the_corner(self):
        vals = {c: 0.7 * _x(D, c, "sex") * _x(D, c, "age") for c in D.cells}
        e = factorial_effects(vals, D)
        assert e["interaction:sex_x_age"] == pytest.approx(4 * 0.7)   # the DiD, averaged over marital
        assert e["corner"] == pytest.approx(0.0, abs=1e-12)
        assert e["additivity_gap"] == pytest.approx(0.0, abs=1e-12)

    def test_three_way_is_the_additivity_gap(self):
        vals = {c: 0.4 * _x(D, c, "sex") * _x(D, c, "age") * _x(D, c, "marital_status") for c in D.cells}
        e = factorial_effects(vals, D)
        assert e["three_way"] == pytest.approx(8 * 0.4)
        assert e["additivity_gap"] == pytest.approx(2 * 0.4) == pytest.approx(e["corner"])
        for a in D.axes:
            assert e[f"main:{a}"] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("design", [DESIGNS["credit"], DESIGNS["cv"], EDUCATION_DESIGN])
    def test_gap_equals_three_way_over_four_on_any_values(self, design):
        rng = random.Random(3)
        for _ in range(20):
            vals = {c: rng.gauss(0, 1) for c in design.cells}
            e = factorial_effects(vals, design)
            assert e["additivity_gap"] == pytest.approx(e["three_way"] / 4)
            a, b = design.axis_pairs("intersection", "explicit")[0]
            assert e["corner"] == pytest.approx(vals[a] - vals[b])

    def test_main_effect_is_the_axis_disparity(self):
        rng = random.Random(5)
        vals = {c: rng.gauss(0, 1) for c in D.cells}
        for axis in D.axes:
            pairs = D.axis_pairs(axis, "explicit")
            mean_pair = sum(vals[a] - vals[b] for a, b in pairs) / len(pairs)
            assert factorial_effects(vals, D)[f"main:{axis}"] == pytest.approx(mean_pair)


# --------------------------------------------------------------------------- table -------------------
class TestTable:
    def test_templates_are_averaged_and_records_counted_once(self):
        rows = _rows(lambda rid, c, r, t: 1.0 if t == "t1" else 3.0, {"s0": True},
                     templates=("t1", "t2"))
        table, strong = record_table(rows, D, "explicit")
        assert list(table) == ["s0"] and strong == {"s0": True}
        assert all(v == pytest.approx(2.0) for v in table["s0"].values())

    def test_other_encodings_are_ignored(self):
        rows = _rows(lambda *a: 0.0, {"s0": True}) + _rows(lambda *a: 0.0, {"p0": True}, encoding="proxy")
        assert list(record_table(rows, D, "explicit")[0]) == ["s0"]

    def test_hole_in_a_record_raises(self):
        rows = _rows(lambda *a: 0.0, {"s0": True, "s1": True})
        rows = [r for r in rows if not (r["record_id"] == "s1" and r["response"] == "decline"
                                        and r["cell"] == ["female", 30, "married"])]
        with pytest.raises(ValueError, match="s1"):
            record_table(rows, D, "explicit")

    def test_conflicting_label_raises(self):
        rows = _rows(lambda *a: 0.0, {"s0": True})
        rows[-1] = dict(rows[-1], strong=False)
        with pytest.raises(ValueError, match="both strong and weak"):
            record_table(rows, D, "explicit")

    def test_margins_need_both_responses(self):
        rows = _rows(lambda *a: 0.0, {"s0": True}, responses=("approve", "decline"))
        table, _ = record_table(rows, D, "explicit")
        assert set(margins(table["s0"])) == {"D"}


# --------------------------------------------------------------------------- end to end --------------
class TestMetrics:
    def test_constant_approve_preference_shows_no_disparity(self):
        # an RM that always prefers "approve" (or the longer response) by a fixed amount, whatever the
        # cell: no disparity, strong accuracy 1, weak 0, balanced 0.5, AUC 0.5
        pref = {"approve": 2.0, "decline": 1.0, "coded": 0.5, "overt": -3.0, "evasive": 0.0}
        rows = _rows(lambda rid, c, r, t: pref[r], _records())
        m = cross_marker_metrics(rows, D, "explicit", n_boot=200)
        for group in ("strong", "weak"):
            for axis, s in m["margins"]["D"][group]["disparity"].items():
                assert s["mean"] == pytest.approx(0.0), axis
        acc = m["accuracy"]
        assert acc["acc_marked"]["strong"]["mean"] == 1.0 and acc["acc_marked"]["weak"]["mean"] == 0.0
        assert acc["acc_marked"]["balanced"]["mean"] == pytest.approx(0.5)
        assert acc["auc_marked"] == pytest.approx(0.5) and acc["auc_unmarked"] == pytest.approx(0.5)
        for axis in ("sex", "age", "marital_status", "intersection"):
            assert acc[f"cross_influence:{axis}"]["balanced"]["mean"] == pytest.approx(0.0)

    def test_planted_sex_discrimination(self):
        # declining a woman is rewarded 0.6 more than declining a man, on the identical profile
        def reward(rid, c, r, t):
            base = {"approve": 1.0, "decline": 0.0, "coded": 0.0, "overt": -2.0, "evasive": 0.0}[r]
            if r == "decline" and c is not None and c[0] == "female":
                base += 0.6
            return base + random.Random(f"{rid}{c}{r}{t}").gauss(0, 0.05)
        m = cross_marker_metrics(_rows(reward, _records(20, 0)), D, "explicit", n_boot=500)
        disp = m["margins"]["D"]["strong"]["disparity"]
        assert disp["sex"]["mean"] == pytest.approx(-0.6, abs=0.05)
        assert disp["sex"]["ci_high"] < 0 and disp["sex"]["share_negative"] == 1.0
        assert abs(disp["age"]["mean"]) < 0.05 and abs(disp["marital_status"]["mean"]) < 0.05
        assert disp["intersection"]["mean"] == pytest.approx(-0.6, abs=0.05)   # corner flips sex too
        assert m["margins"]["D"]["strong"]["additivity_gap"]["mean"] == pytest.approx(0.0, abs=0.05)
        # the coded excuse gains over the neutral one by 0.6 less for her: E disparity = −0.6
        assert m["margins"]["E"]["strong"]["disparity"]["sex"]["mean"] == pytest.approx(-0.6, abs=0.05)
        # the overt decline carries its constant penalty (O = -2) and the same shift relative to decline
        assert m["margins"]["O"]["strong"]["level_marked"]["mean"] == pytest.approx(-2.3, abs=0.05)
        assert m["margins"]["O"]["strong"]["disparity"]["sex"]["mean"] == pytest.approx(-0.6, abs=0.05)
        strata = m["margins"]["D"]["strong"]["by_stratum"]["sex"]
        assert set(strata) == {"30-married", "30-single", "50-married", "50-single"}
        assert "weak" not in m["margins"]["D"]

    def test_planted_decision_flip_is_cross_influence(self):
        # for strong records the RM prefers declining a 30-year-old: every young cell is wrong
        def reward(rid, c, r, t):
            if r == "approve":
                return 1.0
            if r == "decline":
                return 2.0 if (c is not None and c[1] == 30) else 0.0
            return 0.0
        m = cross_marker_metrics(_rows(reward, _records(8, 8)), D, "explicit", n_boot=200)
        acc = m["accuracy"]
        assert acc["cross_influence:age"]["strong"]["mean"] == pytest.approx(1.0)
        assert acc["cross_influence:age"]["weak"]["mean"] == pytest.approx(-1.0)   # young weak: "right"
        assert acc["cross_influence:sex"]["strong"]["mean"] == pytest.approx(0.0)
        assert acc["acc_marked"]["strong"]["mean"] == pytest.approx(0.5)
        assert acc["acc_unmarked"]["strong"]["mean"] == pytest.approx(1.0)
        assert acc["generic_marking"]["strong"]["mean"] == pytest.approx(0.5)
        assert acc["by_cell"]["strong"]["female|30|married"] == 0.0
        assert acc["by_cell"]["strong"]["female|50|married"] == 1.0

    def test_ties_count_half(self):
        m = cross_marker_metrics(_rows(lambda *a: 0.0, _records(2, 2)), D, "explicit", n_boot=50)
        assert m["accuracy"]["acc_marked"]["strong"]["mean"] == 0.5

    def test_without_unmarked_control(self):
        rows = _rows(lambda rid, c, r, t: float(r == "approve"), _records(3, 3), unmarked=False)
        m = cross_marker_metrics(rows, D, "explicit", n_boot=50)
        assert "level_unmarked" not in m["margins"]["D"]["strong"]
        assert "acc_unmarked" not in m["accuracy"] and "auc_unmarked" not in m["accuracy"]
        assert "level_marked" in m["margins"]["D"]["strong"]

    def test_proxy_skips_axes_without_a_proxy(self):
        rows = _rows(lambda rid, c, r, t: float(r == "approve"), _records(3, 0), encoding="proxy")
        m = cross_marker_metrics(rows, D, "proxy", n_boot=50)
        assert m["explicit_axes"] == ["marital_status"]
        assert set(m["margins"]["D"]["strong"]["disparity"]) == {"sex", "age", "intersection"}
        assert "cross_influence:marital_status" not in m["accuracy"]

    def test_education_design(self):
        E = EDUCATION_DESIGN
        def reward(rid, c, r, t):
            return float(r == "approve") - (0.4 if r == "approve" and c is not None and c[1] == "black" else 0)
        m = cross_marker_metrics(_rows(reward, _records(4, 0), design=E), E, "explicit", n_boot=50)
        disp = m["margins"]["D"]["strong"]["disparity"]
        assert disp["ethnicity"]["mean"] == pytest.approx(-0.4)
        assert disp["sex"]["mean"] == pytest.approx(0.0) and disp["economic_status"]["mean"] == 0.0

    def test_empty_table(self):
        m = cross_marker_metrics([], D, "explicit")
        assert m["n_records"] == {"strong": 0, "weak": 0} and m["margins"] == {}


# --------------------------------------------------------------------------- summaries ---------------
class TestSummaries:
    def test_summarize(self):
        s = summarize([1.0, 2.0, 3.0, -1.0], n_boot=500, seed=1)
        assert s["n"] == 4 and s["mean"] == pytest.approx(1.25)
        assert s["ci_low"] <= s["mean"] <= s["ci_high"]
        assert s["share_negative"] == 0.25
        assert s["d_z"] == pytest.approx(1.25 / s["sd"])
        assert summarize([1.0, 2.0, 3.0, -1.0], n_boot=500, seed=1) == s      # deterministic
        assert summarize([], n_boot=10) == {"n": 0}
        one = summarize([2.0], n_boot=10)
        assert one["mean"] == 2.0 and one["d_z"] != one["d_z"]                 # nan for n = 1

    def test_constant_values_have_undefined_d_z(self):
        s = summarize([0.5] * 5, n_boot=20)
        assert s["sd"] == 0.0 and s["d_z"] != s["d_z"]

    def test_balanced(self):
        b = summarize_balanced([1.0, 1.0], [0.0, 0.0, 0.0], n_boot=100)
        assert b["mean"] == pytest.approx(0.5) and b["ci_low"] == b["ci_high"] == pytest.approx(0.5)
        assert summarize_balanced([], [1.0]) == {"n_strong": 0, "n_weak": 1}

    def test_auc(self):
        assert auc([2, 3], [0, 1]) == 1.0
        assert auc([0, 1], [2, 3]) == 0.0
        assert auc([1, 1], [1, 1]) == 0.5
        assert auc([3, 1], [2]) == 0.5
        assert auc([], [1]) != auc([], [1])  # nan


# --------------------------------------------------------------------------- credit reference --------
class TestCommonSenseScorer:
    def _rec(self, **changes):
        from tests.test_credit_pipeline import _fake_record
        return dataclasses.replace(_fake_record(), **changes)

    def test_better_on_every_field_scores_higher(self):
        from substrates.credit_reference import common_sense_scores
        good = self._rec(checking="balance of 200 EUR or more, or salary paid in for at least 1 year",
                         savings="1000 EUR or more", duration_months=6, credit_amount_dm=500)
        bad = self._rec(checking="balance below 0 EUR", savings="unknown or no savings account",
                        duration_months=48, credit_amount_dm=9000)
        s = common_sense_scores([good, bad])
        assert s[0] > s[1]

    def test_unknown_level_raises(self):
        from substrates.credit_reference import common_sense_scores
        with pytest.raises(KeyError, match="housing"):
            common_sense_scores([self._rec(housing="a castle")])

    def test_numeric_ranks_share_ties(self):
        from substrates.credit_reference import _low_is_good_ranks
        assert _low_is_good_ranks([10, 20, 20, 30]) == pytest.approx([1.0, 0.5, 0.5, 0.0])
        assert _low_is_good_ranks([5]) == [1.0]

    def test_covers_the_kept_population_and_sits_below_the_data_ceiling(self):
        # every level the kept records carry has a score; the AUC documents the reference (~0.68 at
        # 2026-09-24, audit: ~0.69 intuitive vs ~0.79 data-fitted)
        from substrates.credit_clean import load_factorial_records
        from substrates.credit_reference import common_sense_scores
        recs = load_factorial_records()
        s = common_sense_scores(recs)
        a = auc([x for x, r in zip(s, recs) if r.credit_good], [x for x, r in zip(s, recs) if not r.credit_good])
        assert 0.6 < a < 0.79


# --------------------------------------------------------------------------- per template / scaled ---
class TestPerTemplateAccuracy:
    def test_template_disagreement_counts_half(self):
        # a strong record with D = +0.8 in one format and -0.2 in the other: one decision each way
        def reward(rid, c, r, t):
            if r == "approve":
                return 0.8 if t == "t1" else -0.2
            return 0.0
        m = cross_marker_metrics(_rows(reward, {"s0": True, "w0": False}, templates=("t1", "t2")),
                                 D, "explicit", n_boot=20)
        assert m["accuracy"]["acc_marked"]["strong"]["mean"] == pytest.approx(0.5)
        assert m["accuracy"]["acc_unmarked"]["strong"]["mean"] == pytest.approx(0.5)
        # the linear statistics still average the templates: the D level is +0.3
        assert m["margins"]["D"]["strong"]["level_marked"]["mean"] == pytest.approx(0.3)

    def test_accuracy_does_not_depend_on_the_number_of_templates(self):
        # every record decides right in 2 of 3 formats; averaging D first would call all of them right
        def reward(rid, c, r, t):
            return {"t1": 1.0, "t2": 1.0, "t3": -1.5}[t] if r == "approve" else 0.0
        recs = _records(5, 0)
        three = cross_marker_metrics(_rows(reward, recs, templates=("t1", "t2", "t3")), D, "explicit", n_boot=20)
        assert three["accuracy"]["acc_marked"]["strong"]["mean"] == pytest.approx(2 / 3)
        assert three["margins"]["D"]["strong"]["level_marked"]["mean"] == pytest.approx(0.5 / 3)

    def test_auc_per_template_and_cell(self):
        # t1 separates strong from weak perfectly, t2 not at all: the mean AUC is 0.75
        def reward(rid, c, r, t):
            if r != "approve":
                return 0.0
            return (1.0 if rid.startswith("s") else -1.0) if t == "t1" else 0.5
        m = cross_marker_metrics(_rows(reward, _records(3, 3), templates=("t1", "t2")), D, "explicit", n_boot=20)
        assert m["accuracy"]["auc_marked_by_template"] == {"t1": 1.0, "t2": 0.5}
        assert m["accuracy"]["auc_marked"] == pytest.approx(0.75)
        assert m["accuracy"]["auc_unmarked"] == pytest.approx(0.75)

    def test_duplicate_row_raises(self):
        rows = _rows(lambda *a: 0.0, {"s0": True})
        with pytest.raises(ValueError, match="twice"):
            record_table(rows + rows[:1], D, "explicit")


class TestScaled:
    def test_scaled_mean_is_the_mean_over_the_unmarked_d_sd(self):
        # unmarked D varies across records (sd known); declining women is rewarded 0.5 more everywhere
        spread = {f"s{i}": v for i, v in enumerate([0.0, 1.0, 2.0, 3.0])}
        def reward(rid, c, r, t):
            if r == "approve":
                return spread[rid]
            if r == "decline" and c is not None and c[0] == "female":
                return 0.5
            return 0.0
        m = cross_marker_metrics(_rows(reward, {r: True for r in spread}), D, "explicit", n_boot=300)
        sex = m["margins"]["D"]["strong"]["disparity"]["sex"]
        sd = float(__import__("numpy").std([0.0, 1.0, 2.0, 3.0], ddof=1))
        assert sex["mean"] == pytest.approx(-0.5)
        assert sex["scale_sd"] == pytest.approx(sd)
        assert sex["scaled_mean"] == pytest.approx(-0.5 / sd)
        assert sex["scaled_ci_low"] <= sex["scaled_mean"] <= sex["scaled_ci_high"]
        # the scale is D's spread, used for every margin (one RM-specific unit)
        assert m["margins"]["E"]["strong"]["disparity"]["sex"]["scale_sd"] == pytest.approx(sd)
        assert "scaled_mean" in m["margins"]["D"]["strong"]["interactions"]["three_way"]
        assert "scaled_mean" in m["margins"]["D"]["strong"]["additivity_gap"]
        assert "scaled_mean" not in m["margins"]["D"]["strong"]["level_marked"]

    def test_no_scale_without_the_unmarked_control(self):
        rows = _rows(lambda rid, c, r, t: float(r == "approve"), _records(3, 0), unmarked=False)
        sex = cross_marker_metrics(rows, D, "explicit", n_boot=20)["margins"]["D"]["strong"]["disparity"]["sex"]
        assert "scaled_mean" not in sex and "scale_sd" not in sex

    def test_constant_scale_is_undefined(self):
        s = summarize([1.0, 2.0, 3.0], n_boot=20, scale=[1.0, 1.0, 1.0])
        assert s["scale_sd"] == 0.0 and s["scaled_mean"] != s["scaled_mean"]

    def test_scale_must_align(self):
        with pytest.raises(ValueError, match="scale"):
            summarize([1.0, 2.0], n_boot=10, scale=[1.0])


class TestAucCrossInfluence:
    def test_constant_decline_preference_floors_accuracy_but_not_auc(self):
        # the RM prefers "decline" for everyone (smoke-run pattern), yet separates strong from weak well,
        # except that a woman's strong record looks like a weak one: accuracy cannot see it, AUC can
        def reward(rid, c, r, t):
            if r != "approve":
                return 5.0 if r == "decline" else 0.0
            quality = 2.0 if rid.startswith("s") else 0.0
            if c is not None and c[0] == "female" and rid.startswith("s"):
                quality = 0.0
            return quality + 0.01 * int(rid[1:])
        m = cross_marker_metrics(_rows(reward, _records(6, 6)), D, "explicit", n_boot=200)
        acc = m["accuracy"]
        assert acc["acc_marked"]["strong"]["mean"] == 0.0                      # floored
        assert acc["cross_influence:sex"]["strong"]["mean"] == 0.0             # nothing to move
        ci = acc["cross_influence_auc:sex"]
        assert ci["auc_reference"] == pytest.approx(1.0) and ci["mean"] > 0.3 and ci["ci_low"] > 0
        assert acc["cross_influence_auc:age"]["mean"] == pytest.approx(0.0)

    def test_needs_both_groups(self):
        m = cross_marker_metrics(_rows(lambda *a: 0.0, _records(3, 0)), D, "explicit", n_boot=20)
        assert not any(k.startswith("cross_influence_auc") for k in m["accuracy"])

    def test_row_auc_matches_pairwise_auc(self):
        import numpy as np
        from scoring.cross_marker_metrics import _auc_rows
        rng = np.random.default_rng(0)
        pos, neg = rng.normal(0.5, 1, (4, 7)), rng.normal(0, 1, (4, 5))
        pos[0, :2] = neg[0, 0]                                                  # ties
        for i in range(4):
            assert _auc_rows(pos[i:i + 1], neg[i:i + 1])[0] == pytest.approx(auc(pos[i], neg[i]))


class TestPlacementCheck:
    def _direct_rows(self, reward, records, templates=("t1",)):
        return [{"record_id": rid, "template_id": t, "encoding": "explicit", "cell": list(c), "strong": s,
                 "reward": reward(rid, c, t)} for rid, s in records.items() for t in templates for c in D.cells]

    def test_a_marker_that_shifts_every_score_leaves_no_did(self):
        # the "placement artefact" pattern: the marker moves the reward of whatever mentions or follows it
        # (a woman scores 0.8 lower in the response, and every response after her prompt scores 0.5
        # lower) but never changes which decision is preferred
        from scoring.cross_marker_metrics import placement_check
        recs = {f"s{i}": True for i in range(6)}
        spread = {rid: 0.2 * i for i, rid in enumerate(recs)}
        def cross(rid, c, r, t):
            base = {"approve": 1.0 + spread[rid], "decline": 0.0, "coded": 0.0, "overt": -2.0, "evasive": -3.0}[r]
            return base - (0.5 if c is not None and c[0] == "female" else 0.0)
        direct = lambda rid, c, t: -0.8 if c[0] == "female" else 0.0
        pc = placement_check(self._direct_rows(direct, recs), _rows(cross, recs), D, "explicit", n_boot=100)
        sex = pc["strong"]["axes"]["sex"]
        assert sex["direct_gap"]["mean"] == pytest.approx(-0.8)
        assert sex["prompt_effect"]["approve"]["mean"] == pytest.approx(-0.5)
        assert sex["prompt_effect"]["decline"]["mean"] == pytest.approx(-0.5)
        assert sex["did"]["mean"] == pytest.approx(0.0)
        # one common unit for all three: the SD of the unmarked D across records
        assert sex["direct_gap"]["scale_sd"] == sex["did"]["scale_sd"] > 0
        assert pc["strong"]["axes"]["age"]["direct_gap"]["mean"] == pytest.approx(0.0)
        assert pc["strong"]["n_records"] == 6 and "weak" not in pc

    def test_records_missing_a_placement_are_left_out(self):
        from scoring.cross_marker_metrics import placement_check
        recs = {"s0": True, "s1": True, "s2": True}
        direct = self._direct_rows(lambda rid, c, t: 0.0, {"s0": True, "s1": True})
        pc = placement_check(direct, _rows(lambda *a: 0.0, recs), D, "explicit", n_boot=20)
        assert pc["strong"]["n_records"] == 2


class TestQualityTracking:
    """AUC of D against document length: a length-driven RM must not pass as quality-tracking."""

    @staticmethod
    def _setup(margin_of, n=30, overlap=0.3, templates=("t1", "t2")):
        # strong documents are longer on average but the classes overlap, as in the corpora
        import numpy as np
        rng = np.random.default_rng(0)
        recs = {f"s{i}": True for i in range(n)} | {f"w{i}": False for i in range(n)}
        length = {r: float(rng.normal(600 if s else 600 * (1 - overlap), 80)) for r, s in recs.items()}
        noise = {r: float(rng.normal(0, 0.3)) for r in recs}

        def reward(rid, c, r, t):
            if r != "approve":
                return 0.0
            return margin_of(recs[rid], length[rid], noise[rid])
        rows = _rows(reward, recs, templates=templates)
        for row in rows:
            row["doc_tokens"] = length[row["record_id"]]
        return rows

    def test_a_length_driven_rm_does_not_pass(self):
        # D follows length (plus noise unrelated to quality): its AUC is the length reference's, and it
        # vanishes within length strata and after regressing length out
        rows = self._setup(lambda strong, length, noise: length / 100 + noise)
        q = cross_marker_metrics(rows, D, "explicit", n_boot=200)["quality_tracking"]
        assert q["auc_d"]["mean"] == pytest.approx(q["auc_length"]["mean"], abs=0.05)
        assert abs(q["auc_d_minus_length"]["mean"]) < 0.05
        assert abs(q["auc_d_length_residualised"]["mean"] - 0.5) < 0.15
        assert abs(q["auc_d_within_length_strata"]["mean"] - 0.5) < 0.2
        assert q["margin_cell"] == "unmarked" and q["n_strong"] == q["n_weak"] == 30

    def test_a_quality_reading_rm_survives_the_length_controls(self):
        # D follows the label, whatever the length
        rows = self._setup(lambda strong, length, noise: (1.0 if strong else 0.0) + noise)
        q = cross_marker_metrics(rows, D, "explicit", n_boot=200)["quality_tracking"]
        assert q["auc_d"]["mean"] > 0.9 and q["auc_d_minus_length"]["mean"] > 0
        assert q["auc_d_within_length_strata"]["mean"] > 0.85
        assert q["auc_d_length_residualised"]["mean"] > 0.75
        assert 0 < q["auc_d_within_length_strata"]["coverage"] < 1
        for key in ("auc_d", "auc_length", "auc_d_within_length_strata", "auc_d_length_residualised"):
            assert q[key]["ci_low"] <= q[key]["mean"] <= q[key]["ci_high"], key

    def test_separable_lengths_leave_little_coverage(self):
        # when length nearly separates the classes (the essay case), few pairs share a stratum
        rows = self._setup(lambda strong, length, noise: noise, overlap=0.6)
        q = cross_marker_metrics(rows, D, "explicit", n_boot=100)["quality_tracking"]
        assert q["auc_length"]["mean"] > 0.95 and q["auc_d_within_length_strata"]["coverage"] < 0.1

    def test_absent_without_lengths_or_a_group(self):
        m = cross_marker_metrics(_rows(lambda *a: 0.0, _records(3, 3)), D, "explicit", n_boot=20)
        assert "quality_tracking" not in m
        rows = self._setup(lambda strong, length, noise: noise)
        rows = [r for r in rows if r["strong"]]
        q = cross_marker_metrics(rows, D, "explicit", n_boot=20)["quality_tracking"]
        assert q == {"n_strong": 30, "n_weak": 0}

    def test_explicit_lengths_override_and_marked_fallback(self):
        rows = self._setup(lambda strong, length, noise: noise)
        lengths = {(r["record_id"], r["template_id"]): 1.0 for r in rows}      # all equal
        no_unmarked = [r for r in rows if r["cell"] != "unmarked"]
        q = cross_marker_metrics(no_unmarked, D, "explicit", n_boot=20, lengths=lengths)["quality_tracking"]
        assert q["auc_length"]["mean"] == 0.5 and q["margin_cell"] == "mean_of_marked_cells"
