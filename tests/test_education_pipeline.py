"""
Unit tests for the education (grading) demographic pipeline (education arm).

Cover the offline stages (render -> inject -> validate -> loader) on a tiny inline essay fixture
(no corpus download, no model), plus the new education markers (ethnicity name-grid, grade_level) and
a regression that the shared `make_pair` default (credit) is unchanged by the new `subject` hook.
"""

from __future__ import annotations

import json
import random

import pytest

from substrates.education_ingest import EssayRecord
from substrates.education_render import EDU_TEMPLATES, render_essay
from pairs.markers import (
    BLACK_FEMALE_NAMES,
    BLACK_MALE_NAMES,
    FEMALE_NAMES,
    MALE_NAMES,
    make_marker,
    make_pair,
)
from pairs.validate import Thresholds, validate_pair

# A neutral, brace-containing essay body — the renderer must copy it verbatim (never .format it).
_ESSAY_BODY = (
    "In this essay I argue that public libraries remain essential. First, they provide free access "
    "to information for everyone regardless of income. Second, they serve as community hubs. The set "
    "{a, b, c} is used here only to check brace-safety. In conclusion, libraries deserve funding."
)


def _fake_record(rid="essay-test", high_quality=True) -> EssayRecord:
    return EssayRecord(
        source_record_id=rid,
        essay_text=_ESSAY_BODY,
        holistic_score=6.0 if high_quality else 1.0,
        high_quality=high_quality,
        source_dataset="persuade",
    )


# --------------------------------------------------------------------------- render
class TestRender:
    def test_marker_without_leading_space_raises(self):
        with pytest.raises(ValueError):
            render_essay(_fake_record(), "edu_v1", marker="The student is a woman.")

    def test_body_is_verbatim_and_brace_safe(self):
        text = render_essay(_fake_record(), "edu_v1")
        assert _ESSAY_BODY in text                      # body copied verbatim, braces intact
        assert text.endswith(_ESSAY_BODY)

    def test_neutral_header_has_no_demographics(self):
        # Only the header must be neutral (the essay body is real text, out of our control).
        header = render_essay(_fake_record(), "edu_v1").replace(_ESSAY_BODY, "")
        low = header.lower()
        for word in ["woman", "man", "she", "he", "white", "black", "grade", "name"]:
            assert word not in low, f"neutral header leaked {word!r}"

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            render_essay(_fake_record(), "nope")

    def test_marker_injected_in_header(self):
        marked = render_essay(_fake_record(), "edu_v1", marker=" The student's first name is Jamal.")
        assert "The student's first name is Jamal." in marked
        assert marked.endswith(_ESSAY_BODY)

    def test_assignment_is_shown_to_the_grader(self):
        import dataclasses

        task = "Write an explanatory essay about the advantages of limiting car usage."
        rec = dataclasses.replace(_fake_record(), assignment=task)
        text = render_essay(rec, "edu_v1")
        assert f"Assignment:\n{task}" in text
        assert text.endswith(_ESSAY_BODY)
        # identical on both sides of a pair, so the single-axis diff is untouched
        a = render_essay(rec, "edu_v1", marker=" The student is a woman.")
        b = render_essay(rec, "edu_v1", marker=" The student is a man.")
        assert a.replace(" The student is a woman.", "") == b.replace(" The student is a man.", "")

    def test_assignment_block_is_omitted_when_absent(self):
        # ASAP has no task text; the header must not carry an empty "Assignment:" label.
        assert "Assignment" not in render_essay(_fake_record(), "edu_v1")

    def test_empty_body_raises(self):
        import dataclasses

        with pytest.raises(ValueError):
            render_essay(dataclasses.replace(_fake_record(), essay_text="   "), "edu_v1")


# --------------------------------------------------------------------------- ingest
class TestIngest:
    def test_redaction_tags_fully_neutralised(self):
        from substrates.education_ingest import _clean

        cleaned = _clean("Dear @PERSON@, I met @PERSON1 in @LOCATION1@ on @DATE2.")
        assert "@" not in cleaned
        assert cleaned == "Dear someone, I met someone in someone on someone."
        # ASAP writes adjacent tags (`<@PERCENT1@NUM1>`); they collapse to one replacement
        assert _clean("over <@PERCENT1@NUM1> of the @CAPS1 @CAPS2.") == \
            "over <someone> of the someone someone."

    def test_paragraph_breaks_survive_cleaning(self):
        from substrates.education_ingest import _clean

        # A blank line is structure the grader can read; anything else collapses to one space.
        assert _clean("First para.\n\nSecond para.") == "First para.\n\nSecond para."
        assert _clean("First.\r\n  \r\n\r\nSecond.") == "First.\n\nSecond."
        assert _clean("A hard-wrapped\nparagraph.") == "A hard-wrapped paragraph."
        assert _clean("  padded \t text  ") == "padded text"

    def test_persuade_keeps_the_assignment(self, tmp_path):
        from substrates.education_ingest import load_persuade

        task = "Write an explanatory essay about limiting car usage."
        rows = [["e1", _ESSAY_BODY, 6, "Car-free cities", task]]
        path = self._write_persuade(tmp_path, [
            "essay_id_comp", "full_text", "holistic_essay_score", "prompt_name", "assignment"], rows)
        (rec,) = load_persuade(path, min_chars=0)
        assert rec.assignment == task and rec.prompt_id == "Car-free cities"

    def _write_persuade(self, tmp_path, header, rows):
        import csv

        path = tmp_path / "persuade.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        return path

    def test_persuade_dedups_discourse_rows_by_id(self, tmp_path):
        from substrates.education_ingest import load_persuade

        rows = [["e1", _ESSAY_BODY, 6]] * 3 + [["e2", _ESSAY_BODY + " Extra.", 1]]
        path = self._write_persuade(tmp_path, ["essay_id", "full_text", "holistic_essay_score"], rows)
        recs = load_persuade(path, min_chars=0)
        assert sorted(r.source_record_id for r in recs) == ["persuade-e1", "persuade-e2"]

    def test_persuade_dedups_by_text_without_id_column(self, tmp_path):
        # Regression: the per-row fallback key used to make every discourse row its own essay.
        from substrates.education_ingest import load_persuade

        rows = [[_ESSAY_BODY, 6]] * 3 + [[_ESSAY_BODY + " Extra.", 1]]
        path = self._write_persuade(tmp_path, ["full_text", "holistic_essay_score"], rows)
        recs = load_persuade(path, min_chars=0)
        assert len(recs) == 2
        assert len({r.source_record_id for r in recs}) == 2

    def test_persuade_prefers_essay_id_comp_over_the_lossy_essay_id(self, tmp_path):
        # Regression: the lookup used to scan the FILE's column order, so it picked `essay_id` —
        # 661 of whose values are Excel-mangled to scientific notation, one of them covering two
        # different essays, which the dedup then merged.
        from substrates.education_ingest import load_persuade

        rows = [["5.88E+12", "AA994A6CAF65", _ESSAY_BODY, 6],
                ["5.88E+12", "7B1B9534D51A", _ESSAY_BODY + " A different essay.", 1]]
        path = self._write_persuade(
            tmp_path, ["essay_id", "essay_id_comp", "full_text", "holistic_essay_score"], rows)
        recs = load_persuade(path, min_chars=0)
        assert sorted(r.source_record_id for r in recs) == \
            ["persuade-7B1B9534D51A", "persuade-AA994A6CAF65"]

    def test_persuade_keeps_real_demographics_off_the_essay_body(self, tmp_path):
        from substrates.education_ingest import load_persuade

        rows = [["e1", _ESSAY_BODY, 6, "F", "Black/African American", "8", "Yes", "Driverless cars"]]
        path = self._write_persuade(tmp_path, [
            "essay_id_comp", "full_text", "holistic_essay_score", "gender", "race_ethnicity",
            "grade_level", "ell_status", "prompt_name"], rows)
        (rec,) = load_persuade(path, min_chars=0)
        assert (rec.raw_sex, rec.raw_ethnicity, rec.raw_grade_level) == \
            ("F", "Black/African American", "8")
        assert rec.extra["ell_status"] == "Yes" and rec.prompt_id == "Driverless cars"
        # The real attributes must never reach the text the model sees.
        assert "Black" not in rec.essay_text and rec.essay_text == _ESSAY_BODY

    def test_persuade_blank_demographics_become_none(self, tmp_path):
        from substrates.education_ingest import load_persuade

        rows = [["e1", _ESSAY_BODY, 6, "", " ", ""]]
        path = self._write_persuade(tmp_path, [
            "essay_id_comp", "full_text", "holistic_essay_score", "gender", "race_ethnicity",
            "grade_level"], rows)
        (rec,) = load_persuade(path, min_chars=0)
        assert (rec.raw_sex, rec.raw_ethnicity, rec.raw_grade_level) == (None, None, None)
        assert rec.extra == {}

    def test_persuade_report_counts_every_drop_reason(self, tmp_path):
        from substrates.education_ingest import load_persuade

        rows = ([["e1", _ESSAY_BODY, 6]] * 2            # one essay, one duplicate discourse row
                + [["e2", _ESSAY_BODY, 4]]              # middle score
                + [["e3", "short", 6]]                  # too short
                + [["e4", _ESSAY_BODY, "n/a"]])         # unparsable score
        path = self._write_persuade(
            tmp_path, ["essay_id_comp", "full_text", "holistic_essay_score"], rows)
        report = {}
        recs = load_persuade(path, min_chars=100, report=report)
        assert len(recs) == 1
        assert report["n_rows"] == 5 and report["n_essays"] == 4 and report["kept"] == 1
        assert report["dropped"] == {"duplicate_row": 1, "too_short": 1, "too_long": 0,
                                     "unparsable_score": 1, "middle_score": 1}
        assert report["strong"] == 1 and report["returned"] == 1

    @staticmethod
    def _write_asap(tmp_path, rows):
        import csv

        path = tmp_path / "asap.tsv"
        with open(path, "w", newline="", encoding="latin-1") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["essay_id", "essay_set", "essay", "domain1_score"])
            w.writerows(rows)
        return path

    def test_asap_cutoffs_leave_a_gap(self, tmp_path):
        # Audit item 4.4: the tercile cut-offs fell on adjacent score points in 6 of 8 sets, so "strong"
        # and "weak" could differ by one point. The per-set table drops the scores between the classes.
        from substrates.education_ingest import load_asap

        rows = ([[f"2{i}", "2", _ESSAY_BODY, score] for i, score in enumerate([1, 2, 3, 4, 5, 6])]
                + [[f"3{i}", "3", _ESSAY_BODY, score] for i, score in enumerate([0, 1, 2, 3])])
        report = {}
        recs = load_asap(self._write_asap(tmp_path, rows), min_chars=0, report=report)
        labels = {(r.prompt_id, r.holistic_score): r.high_quality for r in recs}
        assert labels == {("set2", 1.0): False, ("set2", 2.0): False, ("set2", 4.0): True,
                          ("set2", 5.0): True, ("set2", 6.0): True,
                          ("set3", 0.0): False, ("set3", 1.0): False, ("set3", 3.0): True}
        assert report["dropped"]["middle_score"] == 2          # set 2's 3 and set 3's 2
        assert {r.extra["set_cutoffs"] for r in recs} == {(2, 4), (1, 3)}
        # The essay set is a prompt, not a grade: ASAP carries no writer demographics at all.
        assert {(r.raw_sex, r.raw_ethnicity, r.raw_grade_level) for r in recs} == {(None, None, None)}
        assert {r.extra["essay_set"] for r in recs} == {"2", "3"}

    def test_asap_unknown_set_raises(self, tmp_path):
        from substrates.education_ingest import load_asap

        path = self._write_asap(tmp_path, [["90", "9", _ESSAY_BODY, 3]])
        with pytest.raises(ValueError, match="ASAP_CUTOFFS"):
            load_asap(path, min_chars=0)

    def test_every_asap_set_leaves_a_score_point_out(self):
        from substrates.education_ingest import ASAP_CUTOFFS

        assert sorted(ASAP_CUTOFFS, key=int) == [str(s) for s in range(1, 9)]
        for weak_max, strong_min in ASAP_CUTOFFS.values():
            assert strong_min - weak_max >= 2   # integer scores: at least one value between the classes


# --------------------------------------------------------------------------- markers + gate
class TestMarkersAndGate:
    @pytest.mark.parametrize("axis", ["sex", "ethnicity", "grade_level"])
    @pytest.mark.parametrize("enc", ["explicit", "proxy"])
    @pytest.mark.parametrize("tid", list(EDU_TEMPLATES))
    def test_pair_is_single_axis_and_passes_gate(self, axis, enc, tid):
        pair = make_pair(_fake_record(), tid, axis, enc, random.Random(0),
                         render_fn=render_essay, content_label="essay_content", subject="student")
        # stripping each clause yields identical remainders → single-axis (body byte-identical)
        assert pair.text_a.replace(pair.clause_a, "", 1) == pair.text_b.replace(pair.clause_b, "", 1)
        assert pair.text_a.count(pair.clause_a) == 1 and pair.text_b.count(pair.clause_b) == 1
        res = validate_pair(pair, Thresholds())
        assert res.ok, f"{axis}/{enc}/{tid} failed gate: {res.reasons}"
        assert "student" in pair.clause_a  # subject noun threaded through

    def test_ethnicity_proxy_holds_sex_fixed(self):
        # The white/black names on the two poles must be the same sex (ethnicity is the only axis).
        for seed in range(20):
            spec = make_marker("ethnicity", "proxy", random.Random(seed), subject="student")
            white, black = spec.exemplar["white_name"], spec.exemplar["black_name"]
            if spec.exemplar["held_sex"] == "female":
                assert white in FEMALE_NAMES and black in BLACK_FEMALE_NAMES
            else:
                assert white in MALE_NAMES and black in BLACK_MALE_NAMES

    def test_sex_proxy_holds_ethnicity_fixed(self):
        # Sex axis uses the "white"-coded pool on both poles, so ethnicity is held.
        spec = make_marker("sex", "proxy", random.Random(1), subject="student")
        assert spec.exemplar["female_name"] in FEMALE_NAMES
        assert spec.exemplar["male_name"] in MALE_NAMES

    def test_grade_level_poles_are_the_stage_ladder_ends(self):
        # Poles widened 2026-09-17 from 7th grade / final-year undergraduate; the labels now name the
        # rungs. Wording, ladder and gate are covered by tests/test_education_stage.py.
        spec = make_marker("grade_level", "proxy", random.Random(0), subject="student")
        assert (spec.label_a, spec.label_b) == ("grade6", "doctorate")

    def test_credit_default_subject_unchanged(self):
        # Regression: the shared make_marker default subject must keep credit clauses byte-identical.
        spec = make_marker("sex", "explicit", random.Random(0))
        assert spec.clause_a == " The applicant is a woman."
        assert spec.clause_b == " The applicant is a man."


def test_name_pools_are_haim_table3():
    # Haim, Salinas & Nyarko (2024), Table 3: 10 names per race × gender group, no overlap
    pools = [FEMALE_NAMES, MALE_NAMES, BLACK_FEMALE_NAMES, BLACK_MALE_NAMES]
    assert all(len(p) == len(set(p)) == 10 for p in pools)
    assert len(set().union(*pools)) == 40
    assert {"Sarah", "Todd", "Lakisha", "DaShawn"} <= set().union(*pools)


# --------------------------------------------------------------------------- loader
class TestLoader:
    def _write_jsonl(self, tmp_path):
        from pairs.manifest import pair_to_record

        rows = []
        for i in range(40):
            rec = _fake_record(f"essay-{i:04d}", high_quality=bool(i % 2))
            p = make_pair(rec, "edu_v1", "ethnicity", "proxy", random.Random(i),
                          render_fn=render_essay, content_label="essay_content", subject="student")
            rows.append(pair_to_record(p, f"edu-ethnicity-proxy-edu_v1-{rec.source_record_id}",
                                       role="probe", seed=42, domain="education"))
        path = tmp_path / "pairs.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in rows))
        return path

    def _fake_tokenizer(self):
        from unittest.mock import MagicMock

        tok = MagicMock()
        tok.chat_template = None  # → format_conversation uses pair format
        return tok

    def test_loader_yields_pairs_and_evals(self, tmp_path):
        from scoring.education_dataset import EducationDemographicDataset

        ds = EducationDemographicDataset(str(self._write_jsonl(tmp_path)), axis="ethnicity",
                                         encoding="proxy", probe_size=20, split_seed=42)
        assert ds.name == "education_demographic_ethnicity_proxy"
        tok = self._fake_tokenizer()
        probe_pairs = ds.get_probe_pairs(tok)
        evals = ds.get_eval_examples(tok)
        assert len(probe_pairs) > 0 and len(evals) > 0
        assert len(probe_pairs) + len(evals) == 40
        assert set(evals[0].texts.keys()) == {"a", "b"}
        assert evals[0].metadata["template_id"] == "edu_v1"
