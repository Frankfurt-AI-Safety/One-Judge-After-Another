"""
Tests for the review site's redistribution boundary (`review_site/redact.py`).

This is the one part of the site build where a bug has consequences outside the repo: the site
is public, and the pairs embed CC BY-NC-SA essays and biographies of identifiable people. The
tests below therefore check the *negative* property --- that restricted bodies do not come out
--- rather than only that excerpts look right.

Where the generated datasets are present locally, the last class runs the same check against
real corpus text; it skips rather than fails when they are not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_site.redact import (
    CORPORA,
    DEFAULT_WINDOW,
    Excerpt,
    assert_no_leak,
    excerpt,
    get_corpus,
    probe_slice,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "demographic"

_CLAUSE = " The applicant is a woman."
_LONG_BODY = (
    "Opening sentence that sets up the profile and runs for a while. "
    + "Distinctive midsection alpha bravo charlie delta echo foxtrot golf hotel. " * 12
    + "Closing sentence that wraps the profile up."
)
_RESTRICTED = [k for k, c in CORPORA.items() if not c.redistributable]


class TestPolicyTable:
    def test_only_credit_is_redistributable(self):
        # If this ever changes, it must be a deliberate edit with a licence reason attached.
        assert [k for k, c in CORPORA.items() if c.redistributable] == ["credit"]

    def test_unknown_dataset_fails_closed(self):
        with pytest.raises(KeyError):
            get_corpus("some/new/dataset")
        with pytest.raises(KeyError):
            excerpt("text", "clause", corpus_key="some/new/dataset")

    def test_positioned_education_has_its_own_entry(self):
        # education/persuade and education_positioned/persuade both carry domain=="education"
        # in their records, so policy must be keyed per dataset. Both must resolve.
        assert not get_corpus("education/persuade").redistributable
        assert not get_corpus("education_positioned/persuade").redistributable


class TestExcerpt:
    @pytest.mark.parametrize("key", _RESTRICTED)
    def test_restricted_body_is_bounded(self, key):
        text = _LONG_BODY[:400] + _CLAUSE + _LONG_BODY[400:]
        ex = excerpt(text, _CLAUSE, corpus_key=key, window=DEFAULT_WINDOW)
        assert ex.withheld is True
        assert len(ex.before) + len(ex.after) <= DEFAULT_WINDOW
        assert ex.body_chars == len(text)
        assert ex.clause == _CLAUSE          # the clause itself is always shown -- it is the point
        assert ex.elided_before and ex.elided_after

    @pytest.mark.parametrize("key", _RESTRICTED)
    def test_restricted_midsection_never_survives(self, key):
        text = _LONG_BODY[:400] + _CLAUSE + _LONG_BODY[400:]
        ex = excerpt(text, _CLAUSE, corpus_key=key)
        needle = probe_slice(text)
        assert needle and needle not in (ex.before + ex.clause + ex.after)
        assert_no_leak(ex.to_html(), [text], corpus_key=key)   # must not raise

    def test_credit_is_shown_in_full(self):
        text = "Credit application summary." + _CLAUSE + " The applicant requests 1943 EUR."
        ex = excerpt(text, _CLAUSE, corpus_key="credit")
        assert ex.withheld is False
        assert ex.before + ex.clause + ex.after == text
        assert not ex.elided_before and not ex.elided_after

    @pytest.mark.parametrize("key", _RESTRICTED)
    def test_missing_clause_drops_the_body_entirely(self, key):
        # Fail closed: returning the raw text here would defeat the module.
        ex = excerpt(_LONG_BODY, " a clause that is not present.", corpus_key=key)
        assert ex.before == "" and ex.after == ""
        assert_no_leak(ex.to_html(), [_LONG_BODY], corpus_key=key)

    def test_clause_at_the_very_start_and_end(self):
        # A2 places the clause at the opening or the conclusion; the window must follow it.
        for text in (_CLAUSE + _LONG_BODY, _LONG_BODY + _CLAUSE):
            ex = excerpt(text, _CLAUSE, corpus_key="education_positioned/persuade")
            assert ex.clause == _CLAUSE
            assert len(ex.before) + len(ex.after) <= DEFAULT_WINDOW

    def test_html_is_escaped_and_clause_marked(self):
        text = "<script>x</script>" + _CLAUSE + " tail & more"
        out = excerpt(text, _CLAUSE, corpus_key="credit").to_html()
        assert "<script>" not in out and "&lt;script&gt;" in out
        assert '<mark class="clause">' in out


class TestLeakDetector:
    def test_detects_a_real_leak(self):
        # The detector must actually fire, or the build-time assertion is decoration.
        with pytest.raises(AssertionError, match="leaked"):
            assert_no_leak(_LONG_BODY, [_LONG_BODY], corpus_key="education/persuade")

    def test_is_whitespace_insensitive(self):
        collapsed = " ".join(_LONG_BODY.split())
        with pytest.raises(AssertionError):
            assert_no_leak(collapsed.replace(" ", "\n"), [_LONG_BODY], corpus_key="cv")

    def test_credit_is_exempt(self):
        assert_no_leak(_LONG_BODY, [_LONG_BODY], corpus_key="credit")  # must not raise

    def test_short_texts_are_not_sampled(self):
        assert probe_slice("too short to sample") is None


class TestAgainstRealCorpora:
    """The same guarantee, against the actual generated datasets when they are present."""

    @pytest.mark.parametrize("key", _RESTRICTED)
    def test_real_pairs_are_bounded_and_do_not_leak(self, key):
        path = DATA / key / "pairs.jsonl"
        if not path.exists():
            pytest.skip(f"{key} not generated locally ({path})")
        rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                rows.append(json.loads(line))

        texts, rendered = [], []
        for r in rows:
            for side in ("a", "b"):
                text, clause = r[f"text_{side}"], r[f"clause_{side}"]
                ex = excerpt(text, clause, corpus_key=key)
                assert ex.withheld is True
                assert len(ex.before) + len(ex.after) <= DEFAULT_WINDOW
                texts.append(text)
                rendered.append(ex.to_html())
        assert_no_leak("\n".join(rendered), texts, corpus_key=key)
