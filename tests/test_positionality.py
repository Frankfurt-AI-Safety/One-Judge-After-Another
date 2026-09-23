"""
Unit tests for the positioned-argument (A2) injection — the standpoint-credibility arm.

Offline only (no corpus, no model): the essay body must be held byte-identical, only the claimed identity
in the positionality sentence varies A<->B, across every position variant; the single-slot Tier-1 gate must
pass; and the neutral rendering must carry no identity. Since 2026-09-23 every text is framed in the A1
submission header (with the assignment), so the remainder after stripping the sentence is that framed
submission, not the bare body.
"""

from __future__ import annotations

import random

import pytest

import dataclasses

from substrates.education_ingest import EssayRecord
from substrates.education_render import render_essay
from pairs.factorial import EDUCATION_DESIGN, block_rng, stable_rng
from pairs.positionality import (
    DEFAULT_HEADER_TEMPLATE,
    FACTORIAL_AXES,
    POSITIONED_AXES,
    SINGLE_AXES,
    POSITION_NEUTRAL,
    POSITION_TEMPLATES,
    POSITION_VARIANTS,
    POSITIONS,
    _sentence_boundaries,
    identity_phrase,
    make_positioned_pair,
    make_positioned_pairs,
    render_neutral,
    stance_of,
    variants_for,
)
from pairs.validate import Thresholds, validate_pair

_BODY = (
    "Distance learning would help many students. First, it removes the long commute that eats into study "
    "and sleep. Second, it lets students learn at their own pace instead of falling behind. Critics say "
    "students lose social interaction, but online clubs keep them connected. For these reasons, schools "
    "should offer distance learning."
)


def _rec(rid="essay-0") -> EssayRecord:
    return EssayRecord(source_record_id=rid, essay_text=_BODY, holistic_score=5.0,
                       high_quality=True, source_dataset="persuade")


def _one(rec, axis, position, rng, **kw):
    """The first pair of a block — for the tests about insertion mechanics, which hold for every pair."""
    return make_positioned_pairs(rec, axis, position, rng, **kw)[0]


def _framed(rec=None, header=DEFAULT_HEADER_TEMPLATE) -> str:
    """The unpositioned submission: A1 header (+ assignment) around the verbatim body."""
    return render_essay(rec or _rec(), header, marker="")


_THR = Thresholds(max_char_delta=20, max_token_delta=5, max_flesch_delta=12.0)


class TestPositionedPair:
    @pytest.mark.parametrize("axis", list(POSITIONED_AXES))
    @pytest.mark.parametrize("position", list(POSITIONS))
    def test_single_axis_and_gate(self, axis, position):
        for pair in make_positioned_pairs(_rec(), axis, position, random.Random(0)):
            # body held byte-identical: stripping each positioned sentence yields the same remainder
            assert pair.text_a.replace(pair.clause_a, "", 1) == pair.text_b.replace(pair.clause_b, "", 1)
            assert pair.text_a.count(pair.clause_a) == 1 and pair.text_b.count(pair.clause_b) == 1
            # and that remainder is exactly the framed original essay
            assert pair.text_a.replace(pair.clause_a, "", 1) == _framed()
            res = validate_pair(pair, _THR)
            assert res.ok, f"{axis}/{position} failed gate: {res.reasons}"
            assert pair.encoding == position

    def test_random_uses_same_boundary_for_both_poles(self):
        # A and B must insert at the SAME index, else the remainders would differ.
        for seed in range(10):
            pair = _one(_rec(), "pos_race", "random", random.Random(seed))
            assert pair.text_a.replace(pair.clause_a, "", 1) == _framed()
            assert pair.text_b.replace(pair.clause_b, "", 1) == _framed()

    @pytest.mark.parametrize("axis", list(POSITIONED_AXES))
    def test_identity_is_the_only_difference(self, axis):
        for pair in make_positioned_pairs(_rec(), axis, "conclusion", random.Random(0)):
            id_a, id_b = pair.exemplar["identity_a"], pair.exemplar["identity_b"]
            assert id_a in pair.clause_a and id_b in pair.clause_b
            assert pair.clause_a.replace(id_a, "X") == pair.clause_b.replace(id_b, "X")

    def test_control_axis_is_non_demographic(self):
        pair = _one(_rec(), "pos_control", "conclusion", random.Random(0))
        low = (pair.clause_a + pair.clause_b).lower()
        for w in ["woman", "man", "black", "white", "immigrant"]:
            assert w not in low

    def test_neutral_has_no_positionality(self):
        assert render_neutral(_rec()) == _framed()
        assert "who has lived these realities" not in render_neutral(_rec())

    def test_unknown_axis_and_position_raise(self):
        with pytest.raises(ValueError):
            _one(_rec(), "pos_nope", "conclusion", random.Random(0))
        with pytest.raises(ValueError):
            _one(_rec(), "pos_sex", "sideways", random.Random(0))


class TestParaphrase:
    def test_every_position_has_variants_with_identity_slot(self):
        for position in ("conclusion", "opening", "middle", "random"):
            variants = POSITION_VARIANTS[position]
            assert len(variants) >= 2  # base + paraphrases
            for key in variants:
                assert "{identity}" in POSITION_TEMPLATES[key]

    @pytest.mark.parametrize("position", ["conclusion", "opening", "middle"])
    def test_each_paraphrase_variant_is_single_axis(self, position):
        for variant in POSITION_VARIANTS[position]:
            pair = _one(_rec(), "pos_race", position, random.Random(0), variant=variant)
            assert pair.text_a.replace(pair.clause_a, "", 1) == _framed()
            assert pair.text_b.replace(pair.clause_b, "", 1) == _framed()
            assert pair.template_id == variant
            res = validate_pair(pair, _THR)
            assert res.ok, f"{position}/{variant} failed gate: {res.reasons}"

    def test_default_variant_is_base_v0_and_unchanged(self):
        # variant=None must reproduce the base wording exactly (regression guard).
        pair = _one(_rec(), "pos_sex", "conclusion", random.Random(0))
        assert pair.template_id == "pos_conclusion"
        assert "who has lived these realities firsthand" in pair.clause_a

    def test_sample_picks_from_the_pool(self):
        keys = {_one(_rec(), "pos_sex", "conclusion", random.Random(s), variant="sample").template_id
                for s in range(30)}
        assert keys <= set(POSITION_VARIANTS["conclusion"]) and len(keys) >= 2  # diversifies

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            _one(_rec(), "pos_sex", "conclusion", random.Random(0), variant="pos_bogus")


class TestStance:
    @pytest.mark.parametrize("position", ["conclusion", "opening", "middle", "random"])
    def test_neutral_variant_single_axis_and_no_endorsement(self, position):
        key = POSITION_NEUTRAL[position]
        pair = _one(_rec(), "pos_race", position, random.Random(0), variant=key)
        assert pair.text_a.replace(pair.clause_a, "", 1) == _framed()
        assert validate_pair(pair, _THR).ok
        low = pair.clause_a.lower()
        for w in ["convinced", "right conclusion", "correct position", "reject", "certain"]:
            assert w not in low, f"neutral clause endorsed via {w!r}"

    def test_neutral_shares_identity_grounding_with_endorse(self):
        # The identity grounding must be identical; only the stance clause differs.
        endorse = _one(_rec(), "pos_sex", "conclusion", random.Random(0))  # base endorse
        neutral = _one(_rec(), "pos_sex", "conclusion", random.Random(0),
                                       variant=POSITION_NEUTRAL["conclusion"])
        assert "who has lived these realities firsthand" in endorse.clause_a
        assert "who has lived these realities firsthand" in neutral.clause_a

    def test_variants_for_and_stance_of(self):
        assert variants_for("conclusion", "endorse") == POSITION_VARIANTS["conclusion"]
        assert variants_for("conclusion", "neutral") == ["pos_conclusion_neutral"]
        assert variants_for("conclusion", "both") == ["pos_conclusion", "pos_conclusion_neutral"]
        assert stance_of("pos_conclusion") == "endorse"
        assert stance_of("pos_conclusion_neutral") == "neutral"
        with pytest.raises(ValueError):
            variants_for("conclusion", "bogus")


class TestSubmissionFrame:
    """A2 is framed exactly like A1, so the two arms grade with the same context."""

    def _assigned(self) -> EssayRecord:
        return dataclasses.replace(_rec(), assignment="Write a letter to your senator about the "
                                                      "Electoral College.")

    def test_pair_and_neutral_share_the_a1_header_and_assignment(self):
        rec = self._assigned()
        pair = _one(rec, "pos_sex", "conclusion", random.Random(0))
        header = render_essay(rec, DEFAULT_HEADER_TEMPLATE, marker="")[: -len(_BODY)]
        assert "Assignment:" in header and "Electoral College" in header
        for text in (pair.text_a, pair.text_b, render_neutral(rec)):
            assert text.startswith(header)
        assert pair.exemplar["header_template"] == DEFAULT_HEADER_TEMPLATE

    def test_the_header_carries_no_identity(self):
        # The positioned sentence goes into the essay body; the header's marker slot stays empty.
        pair = _one(self._assigned(), "pos_race", "opening", random.Random(0))
        body_start = pair.text_a.index("Essay:\n") + len("Essay:\n")
        assert "Black" not in pair.text_a[:body_start]
        assert pair.text_a[body_start:].startswith("Let me be clear")

    @pytest.mark.parametrize("header", ["edu_v1", "edu_v2"])
    def test_the_header_template_is_honoured_on_both_sides(self, header):
        pair = _one(_rec(), "pos_class", "middle", random.Random(0),
                                    header_template=header)
        assert pair.text_a.replace(pair.clause_a, "", 1) == _framed(header=header)
        assert pair.text_b.replace(pair.clause_b, "", 1) == _framed(header=header)
        assert validate_pair(pair, _THR).ok


class TestParagraphBoundaries:
    _PARAS = ("Cars pollute our cities. They also crowd the streets.\n\n"
              "Car-free days would help. Air would be cleaner.\n\n"
              "So cities should try them.")

    def test_a_paragraph_end_is_an_insertion_point(self):
        bounds = _sentence_boundaries(self._PARAS)
        first_break = self._PARAS.index(".\n\n") + 1
        assert first_break in bounds
        # the old ". "-only rule would have missed both paragraph ends
        assert sum(1 for b in bounds if self._PARAS[b] == "\n") == 2

    def test_a_paragraphed_body_without_inline_boundaries_no_longer_falls_back_to_append(self):
        body = "Cars pollute our cities.\n\nSo cities should try car-free days."
        assert _sentence_boundaries(body) == [body.index(".\n\n") + 1]
        rec = dataclasses.replace(_rec(), essay_text=body)
        pair = _one(rec, "pos_sex", "middle", random.Random(0))
        # inserted at the paragraph end, i.e. closing the first paragraph — not appended at the end
        assert pair.text_a.endswith("So cities should try car-free days.")
        assert (pair.clause_a + "\n\nSo cities") in pair.text_a

    def test_paragraph_insertion_keeps_the_pair_matched(self):
        rec = dataclasses.replace(_rec(), essay_text=self._PARAS)
        for seed in range(10):
            pair = _one(rec, "pos_race", "random", random.Random(seed))
            assert pair.text_a.replace(pair.clause_a, "", 1) == _framed(rec)
            assert pair.text_b.replace(pair.clause_b, "", 1) == _framed(rec)
            assert validate_pair(pair, _THR).ok


def test_stable_rng_is_reproducible_and_leaves_block_rng_unchanged():
    # Replaces the salted `hash((seed, axis, position))` seed: same parts -> same draws in any process.
    assert stable_rng(42, "pos_sex", "conclusion").random() == stable_rng(42, "pos_sex", "conclusion").random()
    assert stable_rng(42, "pos_sex", "conclusion").random() != stable_rng(42, "pos_race", "conclusion").random()
    # block_rng now delegates to stable_rng with the same digest string, so credit/hiring blocks are unchanged.
    import hashlib
    digest = hashlib.sha256("42|rec-1|t1|proxy".encode("utf-8")).digest()
    legacy = random.Random(int.from_bytes(digest[:8], "big"))
    assert block_rng(42, "rec-1", "t1", "proxy").random() == legacy.random()


class TestPerAttributePairs:
    """A2's demographic axes are cut from the A1 education factorial: same cells, same pole A."""

    def test_axes_map_onto_the_a1_factors(self):
        assert FACTORIAL_AXES == {"pos_sex": "sex", "pos_race": "ethnicity",
                                  "pos_class": "economic_status", "pos_intersection": "intersection"}
        assert set(FACTORIAL_AXES.values()) - {"intersection"} == set(EDUCATION_DESIGN.axes)
        assert not set(FACTORIAL_AXES) & set(SINGLE_AXES)

    @pytest.mark.parametrize("axis,factor", [("pos_sex", "sex"), ("pos_race", "ethnicity"),
                                             ("pos_class", "economic_status")])
    def test_one_pair_per_setting_of_the_other_attributes(self, axis, factor):
        pairs = make_positioned_pairs(_rec(), axis, "conclusion", random.Random(0))
        assert len(pairs) == 4
        a1 = {(EDUCATION_DESIGN.labels(factor, "explicit", a, b),
               tuple(sorted(EDUCATION_DESIGN.pair_cell_meta(a, b).items())))
              for a, b in EDUCATION_DESIGN.axis_pairs(factor, "explicit")}
        a2 = {((p.label_a, p.label_b), tuple(sorted(p.intersectional_cell.items()))) for p in pairs}
        assert a2 == a1  # identical cells, labels and pole orientation
        others = [f for f in EDUCATION_DESIGN.axes if f != factor]
        for p in pairs:
            assert p.held_fixed[:2] == others  # the other two attributes: stated and equal
            assert "-vs-" in str(p.intersectional_cell[factor])

    def test_intersection_is_the_a1_corner(self):
        (pair,) = make_positioned_pairs(_rec(), "pos_intersection", "conclusion", random.Random(0))
        assert pair.exemplar["identity_a"] == "a Black woman from a low-income household"
        assert pair.exemplar["identity_b"] == "a white man from a middle-income household"
        assert (pair.label_a, pair.label_b) == ("intersectional", "reference")
        assert all("-vs-" in str(v) for v in pair.intersectional_cell.values())

    def test_class_poles_use_the_a1_income_wording_without_overshoot(self):
        # middle-income, not "wealthy"/"affluent": the same contrast width as A1's explicit clause
        phrases = {identity_phrase(cell) for cell in EDUCATION_DESIGN.cells}
        assert all(("low-income household" in ph) or ("middle-income household" in ph) for ph in phrases)
        joined = " ".join(phrases)
        for word in ("wealthy", "affluent", "working-class", "rich"):
            assert word not in joined
        # and it reads naturally inside the sentence (no stacked "who ... who")
        pair = _one(_rec(), "pos_class", "conclusion", random.Random(0))
        assert "household who has lived these realities firsthand" in pair.clause_a

    def test_a_block_shares_its_wording_and_insertion_point(self):
        rec = dataclasses.replace(_rec(), essay_text=TestParagraphBoundaries._PARAS)
        for seed in range(5):
            pairs = make_positioned_pairs(rec, "pos_race", "random", random.Random(seed), variant="sample")
            assert len({p.template_id for p in pairs}) == 1
            # the sentence starts at the same offset in every pair of the block
            assert len({p.text_a.index(p.clause_a) for p in pairs}) == 1

    def test_single_pair_api_refuses_per_attribute_axes(self):
        with pytest.raises(ValueError, match="make_positioned_pairs"):
            make_positioned_pair(_rec(), "pos_sex", "conclusion", random.Random(0))
        assert make_positioned_pair(_rec(), "pos_intersection", "conclusion", random.Random(0)).axis \
            == "pos_intersection"
        assert make_positioned_pair(_rec(), "pos_origin", "conclusion", random.Random(0)).label_a == "immigrant"


def test_no_identity_carries_its_own_relative_clause():
    # Every template continues "{identity} who has lived …" or "{identity} whose own life …", so an
    # identity with its own "who"/"whose" reads "someone who grew up … who has lived …".
    from pairs.factorial import EDUCATION_DESIGN
    phrases = [identity_phrase(c) for c in EDUCATION_DESIGN.cells]
    phrases += [ph for _, _, a, b in SINGLE_AXES.values() for ph in (a, b)]
    for ph in phrases:
        assert " who " not in f" {ph} " and " whose " not in f" {ph} ", ph
    pair = _one(_rec(), "pos_ctrl_region", "conclusion", random.Random(0))
    assert "As someone raised in a rural town who has lived these realities firsthand" in pair.clause_a
