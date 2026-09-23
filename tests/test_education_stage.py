"""
Tests for the education arm's stage axis: the plausibility/selection rules
(`substrates/education_clean.py`) and the 6th-grade-to-doctorate clause ladder
(`pairs/markers.py`). The rest of the education pipeline is covered by
`tests/test_education_pipeline.py`.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from pairs.markers import (
    STAGE_LADDER, STAGE_LADDER_AXES, STAGE_POLE, STAGE_REFERENCE, make_marker, make_pair,
)
from pairs.validate import Thresholds, validate_pair
from substrates.education_clean import EDUCATION_RULES, NEUTRAL_PROMPTS, TEXT_RULES, stage_rules
from substrates.education_render import EDU_TEMPLATES, render_essay
from substrates.rules import apply_rules
from tests.test_education_pipeline import _ESSAY_BODY, _fake_record

_NEUTRAL_PROMPT = "The Face on Mars"


def _rec(rid="essay-test", *, body=_ESSAY_BODY, prompt=_NEUTRAL_PROMPT):
    return dataclasses.replace(_fake_record(rid), essay_text=body, prompt_id=prompt)


class TestStageRules:
    @pytest.mark.parametrize("body,rule", [
        ("My teacher says the plan is wrong. " + _ESSAY_BODY, "mentions_own_school_life"),
        ("Students in middle school need more time. " + _ESSAY_BODY, "names_a_school_stage"),
        ("I am in the eighth grade this year. " + _ESSAY_BODY, "states_own_grade"),
        ("As a teenager I see this daily. " + _ESSAY_BODY, "speaks_as_a_pupil"),
        ("We already get too much homework. " + _ESSAY_BODY, "mentions_school_routine"),
        ("Dear Principal, I am writing to object. " + _ESSAY_BODY, "addresses_a_school_authority"),
    ])
    def test_pupil_cues_are_dropped(self, body, rule):
        kept, report = apply_rules([_rec(body=body)], EDUCATION_RULES)
        assert kept == []
        assert report["dropped_by_rule"][rule] == 1

    def test_a_neutral_essay_on_a_neutral_prompt_survives(self):
        kept, _ = apply_rules([_rec()], EDUCATION_RULES)
        assert len(kept) == 1

    def test_school_life_prompts_are_dropped_whole(self):
        kept, report = apply_rules([_rec(prompt="Cell phones at school")], EDUCATION_RULES)
        assert kept == [] and report["dropped_by_rule"]["prompt_presupposes_a_pupil"] == 1

    def test_college_mentions_are_not_a_cue(self):
        # A pupil may write "when I go to college" and a doctoral candidate may mention a university.
        body = "Many students plan to attend college or university afterwards. " + _ESSAY_BODY
        kept, _ = apply_rules([_rec(body=body)], EDUCATION_RULES)
        assert len(kept) == 1

    def test_prompt_selection_is_optional(self):
        # ASAP has essay-set numbers instead of prompt names, so it runs the cue rules only.
        assert stage_rules(None) == TEXT_RULES
        kept, _ = apply_rules([_rec(prompt="set2")], stage_rules(None))
        assert len(kept) == 1


class TestStageLadder:
    def test_poles_are_sixth_grade_and_a_doctoral_candidate(self):
        assert (STAGE_REFERENCE, STAGE_POLE) == ("grade6", "doctorate")
        spec = make_marker("grade_level", "proxy", random.Random(0), subject="student")
        assert spec.clause_a == " The student is enrolled in the sixth grade of a primary school."
        assert spec.clause_b == " The student is enrolled in the final year of a doctoral degree."
        assert (spec.label_a, spec.label_b) == ("grade6", "doctorate")

    def test_explicit_encoding_states_an_age(self):
        spec = make_marker("grade_level", "explicit", random.Random(0), subject="student")
        assert spec.clause_a == " The student is 11 years old." and "28 years old" in spec.clause_b

    def test_ages_increase_along_the_ladder(self):
        ages = [age for _, age, _ in STAGE_LADDER]
        assert ages == sorted(ages) and len(set(ages)) == len(ages)

    def test_every_rung_shares_the_same_reference_clause(self):
        # The sweep only traces one curve if pole A is identical across rungs.
        clauses = {make_marker(a, "proxy", random.Random(0), subject="student").clause_a
                   for a in STAGE_LADDER_AXES}
        assert len(clauses) == 1

    def test_the_top_rung_reproduces_the_headline_axis(self):
        head = make_marker("grade_level", "proxy", random.Random(0), subject="student")
        top = make_marker(f"stage_{STAGE_POLE}", "proxy", random.Random(0), subject="student")
        assert (top.clause_a, top.clause_b) == (head.clause_a, head.clause_b)

    def test_clauses_are_stage_only_and_name_no_achievement(self):
        for axis in ("grade_level",) + STAGE_LADDER_AXES:
            spec = make_marker(axis, "proxy", random.Random(0), subject="student")
            for clause in (spec.clause_a, spec.clause_b):
                assert "enrolled in" in clause
                assert not any(w in clause.lower()
                               for w in ("teaches", "publish", "award", "supervis", "lectur"))

    @pytest.mark.parametrize("axis", ("grade_level",) + STAGE_LADDER_AXES)
    @pytest.mark.parametrize("enc", ["explicit", "proxy"])
    @pytest.mark.parametrize("tid", list(EDU_TEMPLATES))
    def test_pairs_pass_the_gate_without_relaxation(self, axis, enc, tid):
        pair = make_pair(_rec(), tid, axis, enc, random.Random(0), render_fn=render_essay,
                         content_label="essay_content", subject="student")
        res = validate_pair(pair, Thresholds())
        assert res.ok, (axis, enc, tid, res.reasons)
        assert pair.held_fixed[-2:] == ["essay_content", "template"]
        assert axis not in pair.held_fixed

    def test_the_reference_rung_is_not_an_axis(self):
        assert f"stage_{STAGE_REFERENCE}" not in STAGE_LADDER_AXES
        with pytest.raises(ValueError):
            make_marker(f"stage_{STAGE_REFERENCE}", "proxy", random.Random(0))
        with pytest.raises(ValueError):
            make_marker("stage_kindergarten", "proxy", random.Random(0))


def test_neutral_prompts_exclude_the_school_life_prompts():
    for p in ("Cell phones at school", "Community service", "Distance learning", "Summer projects",
              "Mandatory extracurricular activities", "Grades for extracurricular activities"):
        assert p not in NEUTRAL_PROMPTS
    assert _NEUTRAL_PROMPT in NEUTRAL_PROMPTS


def test_the_stage_design_is_not_in_the_default_battery():
    # The domain's default axes are the factorial's (tests/test_education_factorial.py); the stage
    # contrast and its ladder live on their own manifest (same essay pool), reached via --axes /
    # --dataset-source, so they never run by accident on factorial pairs.
    from substrates.domains import get_domain

    edu = get_domain("education")
    assert "grade_level" not in edu.axes
    assert not set(STAGE_LADDER_AXES) & set(edu.axes)


class TestSharedStageSample:
    """Audit 2026-09-23: each rung drew its own essays, so the grade_level / stage_doctorate duplicate
    shared only 139 of 452 essays and the rungs' quality mix varied."""

    _AXES = ["grade_level", *STAGE_LADDER_AXES]
    _ENCS = ["explicit", "proxy"]

    def _rows(self, validate=None, n_per=15):
        from runners.generate_education import build_stage_rows

        recs = [_rec(f"essay-{i:03d}") for i in range(12)]
        return build_stage_rows(recs, axes=self._AXES, encodings=self._ENCS, templates=list(EDU_TEMPLATES),
                                n_per=n_per, seed=42,
                                validate=validate or (lambda pair: validate_pair(pair, Thresholds())))

    def test_every_rung_and_encoding_uses_the_same_essays_and_templates(self):
        rows, rep = self._rows()
        samples = {}
        for r in rows:
            samples.setdefault((r["varied_axis"], r["encoding"]), set()).add((r["source_record_id"], r["template_id"]))
        assert set(samples) == {(a, e) for a in self._AXES for e in self._ENCS}
        assert len({frozenset(s) for s in samples.values()}) == 1  # identical sample in every cell
        assert rep["blocks_kept"] == rep["pairs_per_cell"] == 15

    def test_the_duplicate_consistency_check_is_on_identical_items(self):
        rows, _ = self._rows()
        head = {(r["source_record_id"], r["template_id"]): r["text_b"] for r in rows
                if r["varied_axis"] == "grade_level" and r["encoding"] == "proxy"}
        top = {(r["source_record_id"], r["template_id"]): r["text_b"] for r in rows
               if r["varied_axis"] == f"stage_{STAGE_POLE}" and r["encoding"] == "proxy"}
        assert head == top  # same essays, same texts: a pure consistency check

    def test_a_failing_pair_drops_its_block_everywhere(self):
        from pairs.validate import ValidationResult

        def validate(pair):
            bad = pair.record_id == "essay-003" and pair.axis == "stage_masters"
            return ValidationResult(ok=not bad, reasons=["forced"] if bad else [], metrics={})

        rows, rep = self._rows(validate=validate, n_per=100)
        assert "essay-003" not in {r["source_record_id"] for r in rows}
        assert rep["blocks_dropped"] == len(EDU_TEMPLATES)
