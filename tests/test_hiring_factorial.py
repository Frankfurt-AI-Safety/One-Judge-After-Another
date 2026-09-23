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
from substrates.bios_ingest import with_article
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


# --------------------------------------------------------------------------- role label (audit 2026-09-23)
def _bio(row: int, profession: str):
    return dataclasses.replace(_fake_record(f"bios-{row}"), profession=profession,
                               target_role=profession, qualified=True)


class TestProfessionCap:
    def test_no_profession_exceeds_the_cap_of_the_capped_pool(self):
        from substrates.bios_clean import cap_professions

        recs = ([_bio(i, "professor") for i in range(600)] + [_bio(1000 + i, "physician") for i in range(150)]
                + [_bio(2000 + i, p) for i, p in enumerate(["nurse", "dentist", "poet", "dj"] * 50)])
        kept, rep = cap_professions(recs, 0.25)
        counts = {p: sum(r.profession == p for r in kept) for p in {r.profession for r in kept}}
        assert max(counts.values()) <= 0.25 * len(kept)
        assert counts["professor"] == counts["physician"] == rep["limit"]  # both capped
        assert counts["nurse"] == 50  # small professions untouched

    def test_first_bios_in_order_are_kept(self):
        from substrates.bios_clean import cap_professions

        recs = [_bio(i, "professor") for i in range(10)] + [_bio(100 + i, "nurse") for i in range(10)]
        kept, rep = cap_professions(recs, 0.4)
        assert [r.source_record_id for r in kept if r.profession == "professor"] == \
            [f"bios-{i}" for i in range(rep["limit"])]

    def test_uncapped_when_already_under(self):
        from substrates.bios_clean import cap_professions

        recs = [_bio(i, p) for i, p in enumerate(["nurse", "poet", "dj", "model"] * 5)]
        kept, rep = cap_professions(recs, 0.3)
        assert kept == recs and rep["limit"] is None


class TestRoleAssignment:
    _PROFS = (["professor"] * 40 + ["teacher"] * 20 + ["physician"] * 30 + ["surgeon"] * 10
              + ["attorney"] * 12 + ["paralegal"] * 6 + ["nurse"] * 15 + ["dj"] * 7 + ["yoga_teacher"] * 5)

    def _recs(self):
        return [_bio(i, p) for i, p in enumerate(self._PROFS)]

    def test_exactly_half_of_every_profession_is_qualified(self):
        from substrates.bios_clean import assign_roles

        out = assign_roles(self._recs(), seed=42)
        for prof in set(self._PROFS):
            members = [r for r in out if r.profession == prof]
            assert abs(2 * sum(r.qualified for r in members) - len(members)) <= 1, prof

    def test_role_name_says_nothing_about_qualified(self):
        from substrates.bios_clean import assign_roles, role_leak_report

        out = assign_roles(self._recs(), seed=42)
        for r in out:
            assert r.qualified == (r.profession == r.target_role)
            assert r.role == with_article(r.target_role)  # the rendered header phrase follows the role
        by_role = {}
        for r in out:
            by_role.setdefault(r.target_role, []).append(r.qualified)
        # every role is used by exactly as many qualified as unqualified bios (odd counts: off by one)
        for role, qs in by_role.items():
            assert abs(2 * sum(qs) - len(qs)) <= 1, role
        assert role_leak_report(out)["role_only_accuracy"] <= 0.53

    def test_near_synonyms_are_never_a_mismatch(self):
        from substrates.bios_clean import NEAR_SYNONYM_ROLES, assign_roles

        for seed in range(5):
            out = assign_roles(self._recs(), seed=seed)
            for r in out:
                if not r.qualified:
                    assert r.target_role != r.profession
                    assert frozenset({r.profession, r.target_role}) not in NEAR_SYNONYM_ROLES

    def test_deterministic_and_independent_of_input_order(self):
        from substrates.bios_clean import assign_roles

        a = {r.source_record_id: r.qualified for r in assign_roles(self._recs(), seed=42)}
        b = {r.source_record_id: r.qualified for r in assign_roles(list(reversed(self._recs())), seed=42)}
        assert a == b  # which bios are qualified depends on (seed, row), not on the list order

    def test_impossible_assignment_raises(self):
        from substrates.bios_clean import assign_roles

        # 10 teachers and 2 professors: the unqualified teachers have nowhere legal to go
        recs = [_bio(i, "teacher") for i in range(10)] + [_bio(50 + i, "professor") for i in range(2)]
        with pytest.raises(ValueError, match="near-synonym"):
            assign_roles(recs, seed=42)
