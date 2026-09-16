"""
Unit tests for the hiring (CV-screening) demographic pipeline on the real Bias-in-Bios substrate.

Cover the offline stages (scrub -> render -> inject -> validate -> loader) on tiny inline fixtures
(no corpus download, no model). The scrub tests matter most: the Tier-1 gate *cannot* catch a bad
scrub, because it only compares the two poles to each other, so a leaked "she" is present on both
sides and passes. Empirical leakage is measured separately by experiments/validate_bios_scrub.py.
"""

from __future__ import annotations

import json
import random

import pytest

from substrates.bios_ingest import (
    GENDERED_RE,
    PROFESSIONS,
    RealCVRecord,
    scrub,
    scrub_detailed,
    with_article,
)
from substrates.bios_render import BIOS_TEMPLATES, render_bio
from pairs.markers import make_pair
from pairs.validate import Thresholds, validate_pair

# A neutral, brace-containing biography body — the renderer must copy it verbatim (never .format it).
_BIO_BODY = (
    "The applicant is a data professional with twelve years in industry, specialising in distributed "
    "systems and applied statistics. They have led teams at two firms and published on scheduling. "
    "The set {a, b, c} is used here only to check brace-safety. They mentor junior colleagues."
)


def _fake_record(rid="bios-test", qualified=True, target="surgeon") -> RealCVRecord:
    return RealCVRecord(
        source_record_id=rid,
        bio_text=_BIO_BODY,
        profession=target if qualified else "nurse",
        target_role=target,
        role=with_article(target),
        qualified=qualified,
        gender=0,
    )


# --------------------------------------------------------------------------- scrub
class TestScrub:
    def test_neutralises_pronouns_titles_and_leading_name(self):
        raw = ("Dr. Jane Smith is a surgeon at Mass General. She completed her residency in 2009, "
               "and his colleagues praise Mrs. Smith for her work.")
        out = scrub(raw)
        assert not GENDERED_RE.search(out), f"scrub left gendered text: {GENDERED_RE.findall(out)}"
        assert "Jane" not in out          # leading name span removed
        assert "surgeon" in out           # substantive content preserved

    def test_is_idempotent(self):
        raw = "John Doe is a teacher. He loves his students."
        assert scrub(scrub(raw)) == scrub(raw)

    def test_preserves_case_when_substituting(self):
        # Mid-text sentence-initial pronoun: capitalisation must carry over to the replacement.
        out = scrub("The applicant is a poet. She writes daily, and he edits at night.")
        assert "They writes daily" in out and "they edits at night" in out

    def test_leading_pronoun_is_absorbed_by_the_name_strip(self):
        # A bio opening with a pronoun rather than a name still loses the sex signal: the leading
        # span regex consumes it, which is the desired outcome via a different path.
        assert scrub("She is a poet.").startswith("The applicant is a poet")

    def test_collapses_whitespace(self):
        assert "  " not in scrub("A   b\n\nc   d is a poet.")

    def test_gendered_re_is_the_shared_leak_detector(self):
        # The loader drops any bio whose scrubbed body still trips this, so it must actually fire.
        assert GENDERED_RE.search("her work")
        assert GENDERED_RE.search("the husband")
        assert not GENDERED_RE.search("they mentor junior colleagues")

    def test_removes_every_occurrence_of_the_name_not_just_the_opening(self):
        # Regression for a real leak found in the corpus: stripping only the opening span left a
        # sex-coded first name mid-text ("Call Valorie Knoop on ..."), which would have confounded
        # the sex axis while passing the Tier-1 gate (it appears identically on both poles).
        raw = ("Valorie Knoop graduated with honors in 2003. Having 13 years of experience, "
               "Valorie Knoop affiliates with no hospital. Call Valorie Knoop for an appointment, "
               "or read Valorie's notes.")
        out = scrub(raw)
        assert "Valorie" not in out, f"first name survived: {out}"
        assert "Knoop" not in out
        assert "the applicant the applicant" not in out.lower()

    def test_reports_when_the_name_could_not_be_resolved(self):
        # No leading name span -> the loader must drop the bio rather than trust the body.
        _, resolved = scrub_detailed("Award-winning coverage of the city council since 2011.")
        assert resolved is False
        _, resolved = scrub_detailed("Maria Gonzalez is a dentist in Leeds.")
        assert resolved is True

    def test_strips_phone_numbers(self):
        out = scrub("Jane Doe is a dentist. Call Jane on (909) 427-3910 today.")
        assert "427-3910" not in out and "[phone]" in out


class TestArticle:
    def test_article_agreement_and_underscores(self):
        assert with_article("architect") == "an architect"
        assert with_article("surgeon") == "a surgeon"
        assert with_article("software_engineer") == "a software engineer"

    def test_every_profession_renders(self):
        for p in PROFESSIONS:
            assert with_article(p).startswith(("a ", "an "))


# --------------------------------------------------------------------------- render
class TestRender:
    def test_body_is_verbatim_and_brace_safe(self):
        text = render_bio(_fake_record(), "bios_v1")
        assert _BIO_BODY in text                       # body copied verbatim, braces intact
        assert text.endswith(_BIO_BODY)

    def test_header_states_the_target_role(self):
        header = render_bio(_fake_record(target="architect"), "bios_v1").replace(_BIO_BODY, "")
        assert "an architect" in header

    def test_neutral_header_has_no_demographics(self):
        header = render_bio(_fake_record(), "bios_v1").replace(_BIO_BODY, "")
        low = header.lower()
        for word in ["woman", "man", "she", "he", "white", "black", "name"]:
            assert word not in low, f"neutral header leaked {word!r}"

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            render_bio(_fake_record(), "nope")

    def test_marker_injected_in_header(self):
        marked = render_bio(_fake_record(), "bios_v1", marker=" The applicant is a woman.")
        assert "The applicant is a woman." in marked
        assert marked.endswith(_BIO_BODY)

    def test_gender_label_is_never_rendered(self):
        # The real gender label exists for validity checks only; it must not reach the text.
        for g in (0, 1):
            rec = _fake_record()
            rec.gender = g
            assert render_bio(rec, "bios_v1") == render_bio(_fake_record(), "bios_v1")


# --------------------------------------------------------------------------- markers + gate
class TestMarkersAndGate:
    @pytest.mark.parametrize("axis", ["sex", "age", "family_status", "intersection"])
    @pytest.mark.parametrize("enc", ["explicit", "proxy"])
    @pytest.mark.parametrize("tid", list(BIOS_TEMPLATES))
    def test_pair_is_single_axis_and_passes_gate(self, axis, enc, tid):
        pair = make_pair(_fake_record(), tid, axis, enc, random.Random(0),
                         render_fn=render_bio, content_label="bio_content", subject="applicant")
        # stripping each clause yields identical remainders → single-axis (body byte-identical)
        assert pair.text_a.replace(pair.clause_a, "", 1) == pair.text_b.replace(pair.clause_b, "", 1)
        assert pair.text_a.count(pair.clause_a) == 1 and pair.text_b.count(pair.clause_b) == 1
        # intersection composes three clauses, so it gets the same relaxed bound the generator uses
        thr = Thresholds(max_char_delta=40) if axis == "intersection" else Thresholds()
        res = validate_pair(pair, thr)
        assert res.ok, f"{axis}/{enc}/{tid} failed gate: {res.reasons}"
        assert "applicant" in pair.clause_a  # subject noun threaded through

    def test_held_fixed_records_the_bio_content(self):
        pair = make_pair(_fake_record(), "bios_v1", "sex", "explicit", random.Random(0),
                         render_fn=render_bio, content_label="bio_content", subject="applicant")
        assert "bio_content" in pair.held_fixed


# --------------------------------------------------------------------------- loader
class TestLoader:
    def _write_jsonl(self, tmp_path):
        from pairs.manifest import pair_to_record

        rows = []
        for i in range(40):
            rec = _fake_record(f"bios-{i:04d}", qualified=bool(i % 2))
            p = make_pair(rec, "bios_v1", "sex", "explicit", random.Random(i),
                          render_fn=render_bio, content_label="bio_content", subject="applicant")
            rows.append(pair_to_record(p, f"bios-sex-explicit-bios_v1-{rec.source_record_id}",
                                       role="probe", seed=42, domain="cv"))
        path = tmp_path / "pairs.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in rows))
        return path

    def _fake_tokenizer(self):
        from unittest.mock import MagicMock

        tok = MagicMock()
        tok.chat_template = None  # → format_conversation uses pair format
        return tok

    def test_loader_yields_pairs_and_evals(self, tmp_path):
        from scoring.bios_dataset import BiosDemographicDataset

        ds = BiosDemographicDataset(str(self._write_jsonl(tmp_path)), axis="sex",
                                    encoding="explicit", probe_size=20, split_seed=42)
        assert ds.name == "cv_demographic_sex_explicit"
        tok = self._fake_tokenizer()
        probe_pairs = ds.get_probe_pairs(tok)
        evals = ds.get_eval_examples(tok)
        assert len(probe_pairs) > 0 and len(evals) > 0
        assert len(probe_pairs) + len(evals) == 40
        assert set(evals[0].texts.keys()) == {"a", "b"}
        assert evals[0].metadata["template_id"] == "bios_v1"
