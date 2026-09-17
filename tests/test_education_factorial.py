"""
Tests for the education arm's sex × ethnicity × economic-status factorial: the design and its
index-matched name grid (`pairs/factorial.py`), the substrate rules (`substrates/education_clean.py`)
and the domain registry entry. The stage axis and its ladder — education's *other* design — are
covered by `tests/test_education_stage.py`.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from pairs.factorial import EDUCATION_DESIGN, ProxyNames, education_marker, factorial_pairs
from pairs.markers import BLACK_FEMALE_NAMES, BLACK_MALE_NAMES, FEMALE_NAMES, MALE_NAMES
from pairs.validate import Thresholds, validate_pair
from substrates.education_clean import FACTORIAL_RULES, NEUTRAL_PROMPTS, stage_rules
from substrates.education_render import render_essay
from substrates.rules import apply_rules
from tests.test_education_pipeline import _ESSAY_BODY, _fake_record


def _rec(rid="essay-test", *, body=_ESSAY_BODY, prompt="The Face on Mars"):
    return dataclasses.replace(_fake_record(rid), essay_text=body, prompt_id=prompt)


class TestDesign:
    def test_factors_and_poles(self):
        assert EDUCATION_DESIGN.axes == ("sex", "ethnicity", "economic_status")
        assert {a: lv[0] for a, lv in EDUCATION_DESIGN.factors.items()} == {
            "sex": "female", "ethnicity": "black", "economic_status": "low_income"}
        assert len(EDUCATION_DESIGN.cells) == 8

    def test_explicit_clause_uses_three_slots_and_no_sex_noun(self):
        assert EDUCATION_DESIGN.clause(("female", "black", "low_income"), "explicit", subject="student") \
            == " The student is Black, female, and from a low-income household."
        assert EDUCATION_DESIGN.clause(("male", "white", "middle_income"), "explicit", subject="student") \
            == " The student is white, male, and from a middle-income household."

    def test_explicit_clause_has_no_age_dependent_noun(self):
        # "girl"/"young woman" would make the sex wording covary with the writer's age.
        for cell in EDUCATION_DESIGN.cells:
            clause = EDUCATION_DESIGN.clause(cell, "explicit", subject="student")
            assert not any(w in clause for w in (" girl", " boy", " woman", " man"))

    def test_proxy_clause_carries_the_name_and_the_school_poverty_share(self):
        names = ProxyNames.draw_grid(random.Random(0))
        a = EDUCATION_DESIGN.clause(("female", "black", "low_income"), "proxy", names, "student")
        b = EDUCATION_DESIGN.clause(("male", "white", "middle_income"), "proxy", names, "student")
        assert names.grid[("female", "black")] in a and "most students qualify" in a
        assert names.grid[("male", "white")] in b and "few students qualify" in b
        # no negation on the reference pole
        assert "not qualify" not in b

    def test_every_axis_has_a_proxy(self):
        for axis in EDUCATION_DESIGN.axes:
            assert len(EDUCATION_DESIGN.axis_pairs(axis, "proxy")) == 4

    def test_economic_proxy_is_relabelled_as_school_poverty(self):
        # Explicit states household income; the proxy measures the school's poverty share.
        pairs, _, _ = factorial_pairs(_rec(), "edu_v1", "proxy", render_essay, random.Random(0),
                                      content_label="essay_content", subject="student",
                                      design=EDUCATION_DESIGN)
        econ = [p for p in pairs if p.axis == "economic_status"]
        assert {(p.label_a, p.label_b) for p in econ} == {("high_poverty_school", "low_poverty_school")}
        expl = factorial_pairs(_rec(), "edu_v1", "explicit", render_essay, random.Random(0),
                               content_label="essay_content", subject="student",
                               design=EDUCATION_DESIGN)[0]
        econ_e = [p for p in expl if p.axis == "economic_status"]
        assert {(p.label_a, p.label_b) for p in econ_e} == {("low_income", "middle_income")}

    @pytest.mark.parametrize("enc", ["explicit", "proxy"])
    def test_all_pairs_pass_the_gate_without_relaxation(self, enc):
        # Unlike credit/hiring, even the composed intersection clause sits inside the default bounds.
        pairs, texts, _ = factorial_pairs(_rec(), "edu_v1", enc, render_essay, random.Random(0),
                                          content_label="essay_content", subject="student",
                                          design=EDUCATION_DESIGN)
        assert len(texts) == 8 and len(pairs) == 13  # 3 axes x 4 + 1 corner
        for p in pairs:
            res = validate_pair(p, Thresholds())
            assert res.ok, (enc, p.axis, res.reasons)
            assert p.held_fixed[-2:] == ["essay_content", "template"]


class TestNameGrid:
    def test_grid_is_index_matched(self):
        for seed in range(10):
            names = ProxyNames.draw_grid(random.Random(seed))
            i = FEMALE_NAMES.index(names.grid[("female", "white")])
            assert names.grid[("male", "white")] == MALE_NAMES[i]
            assert names.grid[("female", "black")] == BLACK_FEMALE_NAMES[i]
            assert names.grid[("male", "black")] == BLACK_MALE_NAMES[i]

    def test_a_sex_swap_moves_one_step_and_holds_ethnicity(self):
        pairs, _, _ = factorial_pairs(_rec(), "edu_v1", "proxy", render_essay, random.Random(3),
                                      content_label="essay_content", subject="student",
                                      design=EDUCATION_DESIGN)
        sex = [p for p in pairs if p.axis == "sex"]
        assert len(sex) == 4
        for p in sex:
            # Both sides keep the same ethnicity pole and the same income wording.
            assert p.intersectional_cell["ethnicity"] in ("black", "white")
            assert "-vs-" not in str(p.intersectional_cell["economic_status"])

    def test_one_grid_is_drawn_per_block(self):
        pairs, _, exemplar = factorial_pairs(_rec(), "edu_v1", "proxy", render_essay, random.Random(1),
                                             content_label="essay_content", subject="student",
                                             design=EDUCATION_DESIGN)
        used = {n for n in exemplar["names"].values()}
        assert len(used) == 4
        for p in pairs:
            assert sum(n in p.clause_a for n in used) == 1

    def test_exemplar_round_trips(self):
        names = ProxyNames.draw_grid(random.Random(7))
        assert ProxyNames.from_exemplar(names.as_exemplar()).grid == names.grid
        # The two-name designs keep their original exemplar keys.
        pair = ProxyNames.draw(random.Random(7))
        assert set(pair.as_exemplar()) == {"female_name", "male_name"}
        assert ProxyNames.from_exemplar(pair.as_exemplar()) == pair

    def test_unequal_pools_cannot_be_index_matched(self, monkeypatch):
        monkeypatch.setattr("pairs.factorial.BLACK_MALE_NAMES", ["Jamal"])
        with pytest.raises(ValueError, match="equally long"):
            ProxyNames.draw_grid(random.Random(0))


class TestSubstrateRules:
    def test_essays_discussing_their_own_household_money_are_dropped(self):
        body = "My family cannot afford a second car. " + _ESSAY_BODY
        kept, report = apply_rules([_rec(body=body)], FACTORIAL_RULES)
        assert kept == [] and report["dropped_by_rule"]["mentions_own_household_money"] == 1

    def test_the_factorial_keeps_the_school_life_prompts(self):
        # Unlike the stage design: an injected income level contradicts no argumentative essay, so the
        # factorial uses the whole corpus and the two designs run on different populations.
        rec = _rec(prompt="Cell phones at school")
        assert apply_rules([rec], FACTORIAL_RULES)[0] == [rec]
        assert apply_rules([rec], stage_rules())[0] == []
        assert "Cell phones at school" not in NEUTRAL_PROMPTS


class TestDomainWiring:
    def test_education_domain_uses_the_factorial(self):
        from substrates.domains import get_domain

        edu = get_domain("education")
        assert edu.axes == ("sex", "ethnicity", "economic_status", "intersection")
        assert edu.factorial is EDUCATION_DESIGN
        assert edu.make_marker is education_marker

    def test_marker_dispatches_factorial_axes_to_the_composite_clause(self):
        spec = education_marker("ethnicity", "explicit", random.Random(1), subject="student")
        assert spec.clause_a.startswith(" The student is Black,")

    def test_marker_falls_through_to_the_stage_design(self):
        # grade_level and stage_<rung> are education's other design; they keep their own clauses.
        spec = education_marker("grade_level", "proxy", random.Random(1), subject="student")
        assert "enrolled in the sixth grade" in spec.clause_a
        assert education_marker("stage_masters", "explicit", random.Random(1)).clause_b.endswith(
            "23 years old.")
        with pytest.raises(ValueError):
            education_marker("marital_status", "explicit", random.Random(1))
