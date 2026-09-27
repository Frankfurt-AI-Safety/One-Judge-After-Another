"""
Unit tests for the education (grading) demographic pipeline (education arm).

Cover the offline stages (render -> inject -> validate -> loader) on a tiny inline essay fixture
(no corpus download, no model), plus the grade_level poles and the name pools. Injection and the
Tier-1 gate are tested on the education factorial (tests/test_education_factorial.py) and the stage design
(tests/test_education_stage.py).
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
)
from pairs.factorial import EDUCATION_DESIGN, factorial_pairs

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
        source_dataset="asap2",
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
        # A record without task text: the header must not carry an empty "Assignment:" label.
        assert "Assignment" not in render_essay(_fake_record(), "edu_v1")

    def test_empty_body_raises(self):
        import dataclasses

        with pytest.raises(ValueError):
            render_essay(dataclasses.replace(_fake_record(), essay_text="   "), "edu_v1")


# --------------------------------------------------------------------------- ingest
class TestIngest:
    def test_paragraph_breaks_survive_cleaning(self):
        from substrates.education_ingest import _clean

        # A blank line is structure the grader can read; anything else collapses to one space.
        assert _clean("First para.\n\nSecond para.") == "First para.\n\nSecond para."
        assert _clean("First.\r\n  \r\n\r\nSecond.") == "First.\n\nSecond."
        assert _clean("A hard-wrapped\nparagraph.") == "A hard-wrapped paragraph."
        assert _clean("  padded \t text  ") == "padded text"

    @staticmethod
    def _write(tmp_path, header, rows):
        import csv

        path = tmp_path / "asap2.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        return path

    _HEADER = ["essay_id", "score", "full_text", "prompt_name", "assignment", "gender", "race_ethnicity",
               "grade_level", "ell_status", "economically_disadvantaged"]

    def test_asap2_keeps_the_assignment_and_the_rubric(self, tmp_path):
        from substrates.education_ingest import load_asap2

        task = "Write an explanatory essay about limiting car usage."
        rows = [["X1", 6, _ESSAY_BODY, "Car-free cities", task, "F", "White", 10, "No", "Yes"],
                ["X2", 2, _ESSAY_BODY + " B.", "Car-free cities", task, "M", "White", 10, "No", "Yes"],
                ["X3", 5, _ESSAY_BODY + " C.", "Car-free cities", task, "M", "White", 10, "No", "Yes"]]
        recs = load_asap2(self._write(tmp_path, self._HEADER, rows), min_chars=0)
        by = {r.source_record_id: r for r in recs}
        assert by["asap2-X1"].assignment == task and by["asap2-X1"].prompt_id == "Car-free cities"
        assert by["asap2-X1"].high_quality and by["asap2-X3"].high_quality and not by["asap2-X2"].high_quality
        assert {r.source_dataset for r in recs} == {"asap2"}

    def test_asap2_keeps_real_demographics_off_the_essay_body(self, tmp_path):
        from substrates.education_ingest import load_asap2, real_fields

        rows = [["X1", 6, _ESSAY_BODY, "Driverless cars", "Task.", "F", "Black/African American", 8, "Yes",
                 "Economically disadvantaged"]]
        (rec,) = load_asap2(self._write(tmp_path, self._HEADER, rows), min_chars=0)
        assert (rec.raw_sex, rec.raw_ethnicity, rec.raw_grade_level) == ("F", "Black/African American", "8")
        assert rec.extra == {"ell_status": "Yes", "economically_disadvantaged": "Economically disadvantaged"}
        # The real attributes must never reach the text the model sees.
        assert "Black" not in rec.essay_text and rec.essay_text == _ESSAY_BODY
        # ... but they, and the prompt, go onto every pair row for the validity checks and the breakdown
        fields = real_fields(rec)
        assert fields["prompt_id"] == "Driverless cars" and fields["ethnicity"] == "Black/African American"

    def test_asap2_blank_demographics_become_none(self, tmp_path):
        from substrates.education_ingest import load_asap2

        rows = [["X1", 6, _ESSAY_BODY, "Driverless cars", "Task.", "", " ", "", "", ""]]
        (rec,) = load_asap2(self._write(tmp_path, self._HEADER, rows), min_chars=0)
        assert (rec.raw_sex, rec.raw_ethnicity, rec.raw_grade_level) == (None, None, None)
        assert rec.extra == {}

    def test_asap2_report_counts_every_drop_reason(self, tmp_path):
        from substrates.education_ingest import load_asap2

        rows = ([["X1", 6, _ESSAY_BODY]] * 2               # a repeated id
                + [["X2", 4, _ESSAY_BODY]]                 # middle score
                + [["X3", 6, "short"]]                     # too short
                + [["X4", "n/a", _ESSAY_BODY]]             # unparsable score
                + [["", 6, _ESSAY_BODY]])                  # missing id
        report = {}
        recs = load_asap2(self._write(tmp_path, ["essay_id", "score", "full_text"], rows), min_chars=100,
                          report=report)
        assert len(recs) == 1
        assert report["n_rows"] == 6 and report["n_essays"] == 4 and report["kept"] == 1
        assert report["dropped"] == {"missing_or_repeated_id": 2, "too_short": 1, "too_long": 0,
                                     "unparsable_score": 1, "middle_score": 1}
        assert report["strong"] == 1 and report["returned"] == 1

    def test_missing_file_explains_the_download(self, tmp_path):
        from substrates.education_ingest import load_asap2

        with pytest.raises(FileNotFoundError, match="scrosseye/ASAP_2.0"):
            load_asap2(tmp_path / "raw" / "missing.csv")


# --------------------------------------------------------------------------- stage marker
class TestMarkersAndGate:
    def test_grade_level_poles_are_the_stage_ladder_ends(self):
        # Poles widened 2026-09-17 from 7th grade / final-year undergraduate; the labels now name the
        # rungs. Wording, ladder and gate are covered by tests/test_education_stage.py.
        spec = make_marker("grade_level", "proxy", random.Random(0), subject="student")
        assert (spec.label_a, spec.label_b) == ("grade6", "doctorate")


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
            p = factorial_pairs(rec, "edu_v1", "proxy", render_essay, random.Random(i), axes=("ethnicity",),
                                content_label="essay_content", subject="student",
                                design=EDUCATION_DESIGN)[0][0]
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
