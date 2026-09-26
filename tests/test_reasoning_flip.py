"""
Unit tests for the reasoning-flip 2×2 (correctness × conclusion): the verdict builder + the factorial
metric. No model required.
"""

from __future__ import annotations

import random

import pytest

from substrates.bios_ingest import RealCVRecord
from substrates.bios_render import render_bio
from pairs.verdicts import (
    REASONING_CELLS,
    build_reasoning_item,
    verdict_length_report,
)
from scoring.demographic_experiment import compute_reasoning_metrics


def _rec(rid="bios-0001"):
    # Ported from the synthetic CandidateRecord fixture when the synthetic CV substrate was
    # dropped for Bias-in-Bios. These arms only need a record exposing `.role` (named in the
    # prompt) and `.qualified`, both of which RealCVRecord provides.
    return RealCVRecord(
        source_record_id=rid,
        bio_text=("The applicant is a data professional with twelve years in industry, "
                  "specialising in distributed systems and applied statistics. They have led "
                  "teams at two firms and mentor junior colleagues."),
        profession="project_manager",
        target_role="project_manager",
        role="a project coordinator",
        qualified=True,
        gender=0,
    )


class TestBuilder:
    @pytest.mark.parametrize("premise", ["parental_leave", "intersection", "commute"])
    def test_cells_and_premise_clause(self, premise):
        item = build_reasoning_item(_rec(), premise, render_bio, random.Random(0))
        assert set(item["cells"]) == set(REASONING_CELLS)
        assert item["meta"]["premise_clause"] in item["user_prompt"]
        # claim direction is encoded in the text
        assert "reduce the candidate's near-term availability" in item["cells"]["true_reject"]
        assert "increase the candidate's availability" in item["cells"]["false_advance"]
        # conclusions
        assert "not advance" in item["cells"]["true_reject"]
        assert "recommend advancing" in item["cells"]["false_advance"]

    def test_intersection_premise_is_the_hiring_factorial_corner(self):
        # The premise clause is the protected (pole-A) corner of the hiring factorial, explicit —
        # the same text the factorial arms inject for that cell.
        from pairs.factorial import HIRING_DESIGN
        from pairs.verdicts import REASONING_PREMISES

        corner = HIRING_DESIGN.axis_pairs("intersection", "explicit")[0][0]
        assert REASONING_PREMISES["intersection"]["clause"] == HIRING_DESIGN.clause(corner, "explicit")

    def test_commute_is_non_demographic(self):
        item = build_reasoning_item(_rec(), "commute", render_bio, random.Random(0))
        assert item["meta"]["demographic"] is False
        low = item["user_prompt"].lower()
        for w in ("woman", "man", "parental leave", "30-year-old"):
            assert w not in low

    def test_cells_length_matched(self):
        for premise in ("parental_leave", "intersection", "commute"):
            item = build_reasoning_item(_rec(), premise, render_bio, random.Random(1))
            rep = verdict_length_report(item["cells"], keys=REASONING_CELLS)
            assert rep["max_token_delta"] <= 12, f"{premise}: {rep}"

    def test_deterministic(self):
        a = build_reasoning_item(_rec(), "intersection", render_bio, random.Random(3))
        b = build_reasoning_item(_rec(), "intersection", render_bio, random.Random(3))
        assert a == b

    def test_unknown_premise_raises(self):
        with pytest.raises(ValueError):
            build_reasoning_item(_rec(), "nope", render_bio, random.Random(0))

    def test_vary_false_is_fixed_wording(self):
        # default (vary=False) keeps the reproducible availability wording + claim_type
        item = build_reasoning_item(_rec(), "parental_leave", render_bio, random.Random(0))
        assert item["meta"]["claim_type"] == "availability"
        assert "reduce the candidate's near-term availability" in item["cells"]["true_reject"]


class TestVariedVerdicts:
    @pytest.mark.parametrize("claim_type", ["availability", "experience"])
    def test_paraphrase_preserves_truth_and_conclusion(self, claim_type):
        from pairs.verdicts import CLAIM_TYPES, reasoning_cells_varied

        cells = reasoning_cells_varied("being on parental leave", 10, claim_type, random.Random(0))
        true_stem = cells["true_reject"].split(",")[0]
        false_stem = cells["false_advance"].split(",")[0]
        # the two TRUE cells share the true stem; the two FALSE cells share the false stem
        assert cells["true_advance"].startswith(true_stem)
        assert cells["false_reject"].startswith(false_stem)
        # truth value preserved: true stem ∈ true pool, false stem ∈ false pool, and they differ
        filled = lambda key: {p.format(S="Being on parental leave", Y=10)
                              for p in CLAIM_TYPES[claim_type][key]}
        assert true_stem in filled("true") and true_stem not in filled("false")
        assert false_stem in filled("false")
        # conclusion preserved: each cell's connective comes from the right (advance/reject) pool
        from pairs.verdicts import (
            _ADVANCE_BALANCE, _ADVANCE_CONN, _REJECT_BALANCE, _REJECT_CONN,
        )
        conn = lambda cell, stem: cell[len(stem) + 2:-1]  # strip ", " prefix and trailing "."
        assert conn(cells["true_reject"], true_stem) in _REJECT_CONN
        assert conn(cells["true_advance"], true_stem) in _ADVANCE_BALANCE
        assert conn(cells["false_advance"], false_stem) in _ADVANCE_CONN
        assert conn(cells["false_reject"], false_stem) in _REJECT_BALANCE

    def test_claim_types_have_both_directions(self):
        from pairs.verdicts import CLAIM_TYPES

        assert set(CLAIM_TYPES) >= {"availability", "experience"}
        for spec in CLAIM_TYPES.values():
            assert spec["true"] and spec["false"]

    def test_vary_true_samples_claim_type(self):
        # over several seeds, vary=True should surface more than one claim type
        seen = {build_reasoning_item(_rec(), "commute", render_bio, random.Random(s), vary=True)
                ["meta"]["claim_type"] for s in range(12)}
        assert len(seen) >= 2


class TestMetric:
    def test_factorial_effects(self):
        # TRUE-claim cells high, FALSE-claim cells low → positive correctness effect, zero conclusion effect.
        scores = {
            "true_reject":  [10.0, 10.0],
            "true_advance": [10.0, 10.0],
            "false_advance": [0.0, 0.0],
            "false_reject":  [0.0, 0.0],
        }
        m = compute_reasoning_metrics(scores)
        assert m["n"] == 2
        assert m["correctness_effect"] == pytest.approx(10.0)
        assert m["conclusion_effect"] == pytest.approx(0.0)
        assert m["interaction"] == pytest.approx(0.0)
        assert m["prefers_correct_over_favorable_rate"] == pytest.approx(1.0)  # true_reject(10) > false_advance(0)
        assert m["gap_correct_minus_favorable"] == pytest.approx(10.0)

    def test_conclusion_effect(self):
        # advance cells high, reject low → positive conclusion effect, zero correctness effect.
        m = compute_reasoning_metrics({
            "true_reject":  [0.0, 0.0],
            "true_advance": [8.0, 8.0],
            "false_advance": [8.0, 8.0],
            "false_reject":  [0.0, 0.0],
        })
        assert m["conclusion_effect"] == pytest.approx(8.0)
        assert m["correctness_effect"] == pytest.approx(0.0)
        # headline: true_reject(0) vs false_advance(8) → RM prefers the wrong-favorable verdict
        assert m["prefers_correct_over_favorable_rate"] == pytest.approx(0.0)

    def test_handles_none(self):
        m = compute_reasoning_metrics({
            "true_reject": [5.0, None], "true_advance": [5.0, 5.0],
            "false_advance": [1.0, 1.0], "false_reject": [1.0, 1.0],
        })
        assert m["n"] == 1  # only one aligned true_reject/false_advance pair


class TestProbePairs:
    def test_contrastive_pair_cell_mapping(self):
        from runners.run_reasoning_probe import (
            CORRECTNESS_PAIRS, CONCLUSION_PAIRS, contrastive_pairs,
        )

        items = [build_reasoning_item(_rec(), "parental_leave", render_bio, random.Random(0))]
        fmt = lambda u, v: v  # identity formatter → positive/negative carry the verdict text

        corr = contrastive_pairs(items, fmt, CORRECTNESS_PAIRS)
        # correctness: positive = TRUE-claim (reduces availability), negative = FALSE-claim (increases)
        assert len(corr) == 2
        for p in corr:
            assert "reduce the candidate's near-term availability" in p.positive_text
            assert "increase the candidate's availability" in p.negative_text

        concl = contrastive_pairs(items, fmt, CONCLUSION_PAIRS)
        # conclusion: positive = advance, negative = reject
        assert len(concl) == 2
        for p in concl:
            assert "advancing" in p.positive_text and "not advance" in p.negative_text

    def test_split_is_disjoint(self):
        from runners.run_reasoning_probe import _split

        recs = [_rec(f"cv-{i:04d}") for i in range(40)]
        probe, ev = _split(recs, 15, 20)
        assert len(probe) == 15 and len(ev) == 20
        pids = {r.source_record_id for r in probe}
        eids = {r.source_record_id for r in ev}
        assert pids.isdisjoint(eids)
