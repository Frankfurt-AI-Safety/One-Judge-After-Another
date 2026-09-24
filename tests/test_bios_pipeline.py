"""
Unit tests for the hiring (CV-screening) demographic pipeline on the real Bias-in-Bios substrate.

Cover the offline stages (scrub -> render -> inject -> validate -> loader) on tiny inline fixtures
(no corpus download, no model). The scrub tests matter most: the Tier-1 gate *cannot* catch a bad
scrub, because it only compares the two poles to each other, so a leaked "she" is present on both
sides and passes. Empirical leakage is measured separately by runners/validate_bios_scrub.py.
"""

from __future__ import annotations

import json
import random

import pytest

from substrates.bios_ingest import (
    GENDERED_RE,
    PROFESSIONS,
    RealCVRecord,
    first_name_hit,
    first_names,
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

    def test_leading_pronoun_is_not_taken_for_a_name(self):
        # Most corpus bios open with "He"/"She". The pronoun is neutralised like any other, and the
        # name flag stays False (it used to report "She" as the subject's name).
        out, resolved = scrub_detailed("She is a poet. She writes daily.")
        assert out == "They are a poet. They writes daily."
        assert resolved is False

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
        # Informational only: a name mentioned later is caught by the loader's first-name drop.
        _, resolved = scrub_detailed("Award-winning coverage of the city council since 2011.")
        assert resolved is False
        _, resolved = scrub_detailed("Maria Gonzalez is a dentist in Leeds.")
        assert resolved is True

    def test_strips_phone_numbers(self):
        out = scrub("Jane Doe is a dentist. Call Jane on (909) 427-3910 today.")
        assert "427-3910" not in out and "[phone]" in out

    def test_strips_emails_and_urls(self):
        out = scrub("Jane Doe is a dentist. Mail jane.doe@smile-clinic.com or see www.smile.com/jane.")
        assert "@" not in out and "smile" not in out
        assert "[email]" in out and out.endswith("[url].")

    def test_titles_only_before_a_name(self):
        # "MS" (the degree) and the verb "miss" are not titles; "Mrs. Smith" is.
        out = scrub("They hold an MS in Nursing and never miss a deadline. Ask Mrs. Smith.")
        assert "an MS in Nursing" in out and "never miss a deadline" in out
        assert "Mrs" not in out and "Ask Smith." in out
        assert not GENDERED_RE.search(out)

    def test_surname_that_is_a_word_is_kept(self):
        # Only the full name and the first name are replaced; surnames are not sex-coded and often
        # double as ordinary words ("Young Adults").
        out = scrub("Jane Young is a teacher. Jane works with Young Adults.")
        assert out == "The applicant is a teacher. The applicant works with Young Adults."

    def test_initials_and_possessives(self):
        out = scrub("J. Robert Smith is an attorney. Robert earned a J.D. at Yale. Call Robert's office.")
        assert "J.D." in out
        assert "Robert" not in out
        assert "Call the applicant's office." in out

    def test_repeated_placeholder_is_collapsed_before_punctuation(self):
        out = scrub("Jane Doe is a nurse. Ask Jane Doe Jane.")
        assert "the applicant the applicant" not in out.lower()


class TestLeakFilters:
    @pytest.mark.parametrize("word", ["actress", "chairman", "brother", "Girl", "mom", "fiancée",
                                      "spokeswoman", "grandmother"])
    def test_gendered_words_are_flagged(self, word):
        assert GENDERED_RE.search(f"They worked as a {word} for years.")

    @pytest.mark.parametrize("text", ["They advocate for women in science.", "an MS in Nursing",
                                      "They never miss a class.", "a manager", "the Hispanic community"])
    def test_topical_and_innocent_words_are_not_flagged(self, text):
        assert not GENDERED_RE.search(text)

    def test_name_list_is_loaded_and_sex_coded_only(self):
        names = first_names()
        assert len(names) > 1000
        assert {"John", "Mary", "Steven", "Jennifer"} <= names
        # gender-neutral, word-like, calendar and place names are not listed
        assert not {"Jordan", "Taylor", "Will", "Grace", "June", "Virginia", "Austin"} & names

    @pytest.mark.parametrize("text,hit", [
        ("The applicant is licensed in Arkansas. John holds a Juris Doctorate.", "John"),
        ("They trained with Patricia Pierce at a clinic.", "Patricia"),
        ("They studied in Santa Barbara and at Stanford.", None),
        ("They practise in St. Louis and San Francisco.", None),
        ("They trained at the University of Wisconsin-Madison.", None),
        ("They graduated from Cornell University.", None),
        ("They will join the practice in June.", None),
        ("They trained under DeAndre at the clinic.", "DeAndre"),
        ("They advise on ADA compliance and gave TED talks.", None),
    ])
    def test_first_name_hit(self, text, hit):
        assert first_name_hit(text) == hit


class TestLoaderFilters:
    _FILLER = " They have practised for many years and enjoy teaching students about the field." * 5

    def _write(self, tmp_path, rows):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        path = tmp_path / "bios.parquet"
        pd.DataFrame(rows, columns=["hard_text", "profession", "gender"]).to_parquet(path)
        return path

    def _clean_rows(self, n):
        # n clean bios cycling through four professions (nurse, surgeon, poet, teacher)
        profs = [13, 25, 20, 26]
        return [(f"She is a professional, case {k}." + self._FILLER, profs[k % 4], k % 2) for k in range(n)]

    def test_drop_reasons_and_women_men_report(self, tmp_path):
        from substrates.bios_ingest import load_bias_in_bios

        rows = self._clean_rows(8) + [
            ("He is a nurse. He advocates for women in medicine." + self._FILLER, 13, 0),  # kept, women
            ("He is a nurse.", 13, 0),                                                    # too short
            ("She is a nurse. She worked as an actress." + self._FILLER, 13, 1),          # gendered word
            ("She is a nurse. Contact Patricia for details." + self._FILLER, 13, 1),      # first name
        ]
        report = {}
        recs = load_bias_in_bios(self._write(tmp_path, rows), seed=0, report=report)
        assert len(recs) == report["kept"] == 9
        assert report["dropped"]["length"] == {0: 1, 1: 0}
        assert report["dropped"]["gendered_word"] == {0: 0, 1: 1}
        assert report["dropped"]["first_name"] == {0: 0, 1: 1}
        assert report["kept_by_gender"] == {0: 5, 1: 4}
        assert report["kept_mentioning_women_or_men"] == 1
        assert report["kept_qualified"] == sum(r.qualified for r in recs)
        assert all("raw_bio" not in r.extra for r in recs)

    def test_role_match_is_consistent_and_never_self_for_unqualified(self, tmp_path):
        from substrates.bios_ingest import load_bias_in_bios

        recs = load_bias_in_bios(self._write(tmp_path, self._clean_rows(40)), seed=1, keep_raw=True)
        for r in recs:
            assert r.qualified == (r.profession == r.target_role)
            assert r.role.endswith(r.target_role.replace("_", " "))
            assert r.extra["raw_bio"].startswith("She is a professional")
        # unqualified targets are the unqualified group's own professions, reshuffled
        unq = [r for r in recs if not r.qualified]
        assert sorted(r.target_role for r in unq) == sorted(r.profession for r in unq)

    def test_unknown_or_single_profession_pool_raises(self, tmp_path):
        from substrates.bios_ingest import load_bias_in_bios

        path = self._write(tmp_path, self._clean_rows(8))
        with pytest.raises(ValueError):
            load_bias_in_bios(path, professions=["nurse", "astronaut"])
        with pytest.raises(ValueError):
            load_bias_in_bios(path, professions=["nurse"])

    def test_coin_is_fixed_per_row(self):
        from substrates.bios_ingest import _coin

        flips = [_coin(42, i) for i in range(2000)]
        assert flips == [_coin(42, i) for i in range(2000)]
        assert 0.45 < sum(flips) / len(flips) < 0.55
        assert flips != [_coin(7, i) for i in range(2000)]


class TestMismatchedRoles:
    def test_no_fixed_points_and_same_multiset(self):
        from substrates.bios_ingest import _assign_mismatched_roles

        profs = ["professor"] * 40 + ["nurse"] * 30 + ["dj"] * 20 + ["poet"] * 10
        out = _assign_mismatched_roles(profs, random.Random(0))
        assert all(t != p for t, p in zip(out, profs))
        assert sorted(out) == sorted(profs)

    def test_majority_profession_is_rejected(self):
        from substrates.bios_ingest import _assign_mismatched_roles

        with pytest.raises(ValueError):
            _assign_mismatched_roles(["professor"] * 6 + ["nurse"] * 4, random.Random(0))
        with pytest.raises(ValueError):
            _assign_mismatched_roles(["nurse"], random.Random(0))
        assert _assign_mismatched_roles([], random.Random(0)) == []


class TestArticle:
    def test_article_agreement_and_underscores(self):
        assert with_article("architect") == "an architect"
        assert with_article("surgeon") == "a surgeon"
        assert with_article("software_engineer") == "a software engineer"

    def test_dj_display_name(self):
        assert with_article("dj") == "a DJ"

    def test_every_profession_renders(self):
        for p in PROFESSIONS:
            assert with_article(p).startswith(("a ", "an "))


# --------------------------------------------------------------------------- render
class TestRender:
    def test_marker_without_leading_space_raises(self):
        with pytest.raises(ValueError):
            render_bio(_fake_record(), "bios_v1", marker="The applicant is a woman.")

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
