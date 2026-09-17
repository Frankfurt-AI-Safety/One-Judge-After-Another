"""
Tests for the hiring arm's sex × age × family-status factorial: the plausibility rules
(`substrates/bios_clean.py`), the hiring design (`pairs/factorial.py`) and the domain registry entry.
The credit design is covered by `tests/test_credit_factorial.py`.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from pairs.factorial import HIRING_DESIGN, ProxyNames, factorial_pairs, hiring_marker
from pairs.validate import Thresholds, validate_pair
from substrates.bios_clean import FACTORIAL_RULES
from substrates.bios_render import render_bio
from substrates.rules import apply_rules
from tests.test_bios_pipeline import _fake_record


def _rec(rid="bios-test", **changes):
    return dataclasses.replace(_fake_record(rid), **changes)


_NEUTRAL_BIO = ("The applicant leads a distributed systems team and mentors junior colleagues. "
                "They publish on scheduling and speak at industry meetups.")


class TestPlausibilityRules:
    @pytest.mark.parametrize("bio,rule", [
        ("They earned a doctorate in 2005 and now lead a lab.", "mentions_year_before_2014"),
        ("They bring 25 years of experience to the team.", "mentions_more_than_8_years"),
        ("They have spent two decades in cardiology.", "mentions_many_years_in_words"),
        ("They retired from the bench in a recent term.", "mentions_retirement"),
    ])
    def test_age_incompatible_bios_are_dropped(self, bio, rule):
        kept, report = apply_rules([_rec(bio_text=bio)], FACTORIAL_RULES)
        assert kept == []
        assert report["dropped_by_rule"][rule] == 1

    def test_neutral_bio_is_kept(self):
        kept, _ = apply_rules([_rec(bio_text=_NEUTRAL_BIO)], FACTORIAL_RULES)
        assert len(kept) == 1

    def test_small_year_counts_are_kept(self):
        # "8 years" is compatible with a 30-year-old; only more than 8 is not.
        kept, _ = apply_rules([_rec(bio_text="They have 8 years of clinical experience. " + _NEUTRAL_BIO)],
                              FACTORIAL_RULES)
        assert len(kept) == 1


class TestHiringDesign:
    def test_factors_and_poles(self):
        assert HIRING_DESIGN.axes == ("sex", "age", "family_status")
        assert {a: lv[0] for a, lv in HIRING_DESIGN.factors.items()} == {
            "sex": "female", "age": 30, "family_status": "parental_leave"}
        assert len(HIRING_DESIGN.cells) == 8

    def test_explicit_clause(self):
        assert HIRING_DESIGN.clause(("female", 30, "parental_leave"), "explicit") == \
            " The applicant is a 30-year-old woman currently on parental leave."
        assert HIRING_DESIGN.clause(("male", 50, "no_leave"), "explicit") == \
            " The applicant is a 50-year-old man currently in continuous employment."

    def test_proxy_clause_signals_parenthood(self):
        names = ProxyNames("Abigail", "Todd")
        assert HIRING_DESIGN.clause(("female", 30, "parental_leave"), "proxy", names) == \
            (" The applicant, Abigail, was born in 1996 and volunteers as an officer of their "
             "children's school parent association.")
        assert HIRING_DESIGN.clause(("male", 50, "no_leave"), "proxy", names) == \
            (" The applicant, Todd, was born in 1976 and volunteers as an officer of their "
             "neighbourhood residents' association.")

    def test_family_status_has_a_proxy_unlike_credit_marital(self):
        assert len(HIRING_DESIGN.axis_pairs("family_status", "proxy")) == 4

    def test_proxy_family_labels_say_parenthood(self):
        pairs, _, _ = factorial_pairs(_rec(), "bios_v1", "proxy", render_bio, random.Random(0),
                                      content_label="bio_content", design=HIRING_DESIGN)
        fam = [p for p in pairs if p.axis == "family_status"]
        assert len(fam) == 4
        assert {(p.label_a, p.label_b) for p in fam} == {("parent", "non_parent")}
        expl = factorial_pairs(_rec(), "bios_v1", "explicit", render_bio, random.Random(0),
                               content_label="bio_content", design=HIRING_DESIGN)[0]
        fam_expl = [p for p in expl if p.axis == "family_status"]
        assert {(p.label_a, p.label_b) for p in fam_expl} == {("parental_leave", "no_leave")}

    @pytest.mark.parametrize("enc", ["explicit", "proxy"])
    def test_all_pairs_pass_the_gate(self, enc):
        pairs, texts, _ = factorial_pairs(_rec(), "bios_v1", enc, render_bio, random.Random(0),
                                          content_label="bio_content", design=HIRING_DESIGN)
        assert len(texts) == 8 and len(pairs) == 13  # 3 axes x 4 + 1 corner, both encodings
        for p in pairs:
            thr = Thresholds(max_char_delta=40) if p.axis == "intersection" else Thresholds()
            res = validate_pair(p, thr)
            assert res.ok, (p.axis, res.reasons)
            assert p.held_fixed[-2:] == ["bio_content", "template"]

    def test_marker_matches_the_pair_form(self):
        spec = hiring_marker("family_status", "explicit", random.Random(1))
        assert "on parental leave." in spec.clause_a and "in continuous employment." in spec.clause_b
        assert hiring_marker("sex", "proxy", random.Random(1)).clause_a.count(",") == 2
        with pytest.raises(ValueError):
            hiring_marker("marital_status", "explicit", random.Random(1))


def test_cv_domain_uses_the_hiring_factorial():
    from substrates.domains import get_domain

    cv = get_domain("cv")
    assert cv.axes == ("sex", "age", "family_status", "intersection")
    assert cv.factorial is HIRING_DESIGN
    assert cv.make_marker is hiring_marker
