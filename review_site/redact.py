"""
The redistribution boundary for the public review site.

The site's whole point is showing what actually reaches the reward model, but the matched
pairs embed corpora we may not republish:

- **German Credit** (CC-BY-4.0) --- redistributable, shown in full.
- **PERSUADE 2.0** (CC BY-NC-SA 4.0) --- our own data README commits to *not* redistributing
  derived essays.
- **ASAP-AES** (Kaggle 2012 competition terms).
- **Bias-in-Bios** (MIT) --- the licence permits it, but these are biographies of
  *identifiable real people*; republishing them on a public site is a privacy call, not a
  licensing one, and we decline.

So restricted corpora get a bounded window around the injected clause instead of the body.
That costs almost nothing analytically: the body is held **byte-identical** across the A/B
pair, so it carries none of the contrast a reviewer is judging. What it does cost is topical
context, which pages restore with corpus *metadata* (essay prompt, quality tier, length;
target role and occupation) rather than prose.

Two invariants this module exists to hold:

1. **Policy is keyed on the dataset, not the record.** `education/persuade` and
   `education_positioned/persuade` both carry ``domain == "education"`` in their records, so
   the record's own domain field cannot distinguish corpora and must never be the key.
2. **It fails closed.** If the clause cannot be located in the text, we emit *no body at all*
   rather than falling back to the raw string.

`assert_no_leak` re-checks the result against rendered HTML, and the build calls it, so a bad
page cannot be published even if the test suite is skipped.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

DEFAULT_WINDOW = 200


@dataclass(frozen=True)
class Corpus:
    """Redistribution policy for one dataset. `key` is the dataset path under data/demographic/."""

    key: str
    label: str
    licence: str
    redistributable: bool
    note: str


CORPORA: Dict[str, Corpus] = {
    "credit": Corpus(
        key="credit", label="UCI German Credit", licence="CC-BY-4.0",
        redistributable=True,
        note="Redistributable; profiles are rendered from decoded attribute codes and shown in full.",
    ),
    "cv": Corpus(
        key="cv", label="Bias in Bios", licence="MIT",
        redistributable=False,
        note="Biographies of identifiable real people; excerpted on privacy grounds, not licensing.",
    ),
    "education/persuade": Corpus(
        key="education/persuade", label="PERSUADE 2.0", licence="CC BY-NC-SA 4.0",
        redistributable=False,
        note="Derived essays are not redistributed, per the corpus licence.",
    ),
    "education/asap": Corpus(
        key="education/asap", label="ASAP-AES", licence="Kaggle 2012 competition terms",
        redistributable=False,
        note="Derived essays are not redistributed, per the competition terms.",
    ),
    "education_positioned/persuade": Corpus(
        key="education_positioned/persuade", label="PERSUADE 2.0 (positioned argument)",
        licence="CC BY-NC-SA 4.0", redistributable=False,
        note="Derived essays are not redistributed, per the corpus licence.",
    ),
}


def get_corpus(key: str) -> Corpus:
    if key not in CORPORA:
        # Fail closed: an unknown dataset is treated as restricted rather than published.
        raise KeyError(
            f"No redistribution policy for dataset {key!r}. Add it to CORPORA before rendering "
            f"it; known: {sorted(CORPORA)}"
        )
    return CORPORA[key]


@dataclass(frozen=True)
class Excerpt:
    """A publishable view of one rendered side of a pair."""

    before: str            # text preceding the clause (possibly truncated)
    clause: str            # the injected clause, shown verbatim -- this is the point of the page
    after: str             # text following the clause (possibly truncated)
    elided_before: bool
    elided_after: bool
    body_chars: int        # length of the FULL text, so reviewers know what is hidden
    withheld: bool         # True when the body was excerpted
    corpus: Corpus

    @property
    def shown_chars(self) -> int:
        return len(self.before) + len(self.clause) + len(self.after)

    def to_html(self) -> str:
        """Escaped HTML with the clause marked up. The only place excerpt text becomes markup."""
        parts = []
        if self.elided_before:
            parts.append('<span class="elided">[…]</span> ')
        parts.append(html.escape(self.before))
        parts.append(f'<mark class="clause">{html.escape(self.clause)}</mark>')
        parts.append(html.escape(self.after))
        if self.elided_after:
            parts.append(' <span class="elided">[…]</span>')
        return "".join(parts)

    def provenance_note(self) -> str:
        if not self.withheld:
            return f"{self.corpus.label} ({self.corpus.licence}) — shown in full."
        return (
            f"body: {self.body_chars:,} chars, held byte-identical across A/B; "
            f"showing {self.shown_chars:,}. Full text withheld — "
            f"{self.corpus.label}, {self.corpus.licence}."
        )


def _snap_left(text: str, budget: int) -> tuple:
    """Take up to `budget` trailing chars, snapped forward to a word boundary. -> (text, elided)"""
    if len(text) <= budget:
        return text, False
    cut = text[-budget:]
    space = cut.find(" ")
    if space != -1:
        cut = cut[space + 1:]
    return cut, True


def _snap_right(text: str, budget: int) -> tuple:
    """Take up to `budget` leading chars, snapped back to a word boundary. -> (text, elided)"""
    if len(text) <= budget:
        return text, False
    cut = text[:budget]
    space = cut.rfind(" ")
    if space != -1:
        cut = cut[:space]
    return cut, True


def excerpt(text: str, clause: str, *, corpus_key: str, window: int = DEFAULT_WINDOW) -> Excerpt:
    """Publishable view of `text`, centred on `clause`.

    Redistributable corpora are returned whole. Restricted corpora get at most `window` chars
    of body, split either side of the clause. If `clause` is absent, the body is dropped
    entirely rather than guessed at.
    """
    corpus = get_corpus(corpus_key)
    body_chars = len(text)

    idx = text.find(clause) if clause else -1
    if idx == -1:
        # Fail closed. Emitting the raw text here would silently defeat the whole module.
        return Excerpt(before="", clause=clause or "", after="", elided_before=bool(text),
                       elided_after=False, body_chars=body_chars,
                       withheld=not corpus.redistributable, corpus=corpus)

    head, tail = text[:idx], text[idx + len(clause):]

    if corpus.redistributable:
        return Excerpt(before=head, clause=clause, after=tail, elided_before=False,
                       elided_after=False, body_chars=body_chars, withheld=False, corpus=corpus)

    half = max(window // 2, 0)
    before, elided_before = _snap_left(head, half)
    after, elided_after = _snap_right(tail, half)
    return Excerpt(before=before, clause=clause, after=after, elided_before=elided_before,
                   elided_after=elided_after, body_chars=body_chars, withheld=True, corpus=corpus)


def probe_slice(text: str, *, length: int = 60) -> Optional[str]:
    """A distinctive mid-body slice, used to detect leaks.

    Taken from the centre of the text, which is the region an excerpt window around a clause at
    the start or end can never legitimately reach. Returns None for texts too short to sample.
    """
    if len(text) < length * 3:
        return None
    mid = len(text) // 2
    start = max(0, mid - length // 2)
    return text[start:start + length]


def assert_no_leak(rendered: str, texts: Iterable[str], *, corpus_key: str) -> None:
    """Raise if any restricted body leaked into `rendered` (HTML or plain).

    Called at build time as well as from the tests, so a mis-rendered page cannot ship just
    because someone skipped pytest.
    """
    corpus = get_corpus(corpus_key)
    if corpus.redistributable:
        return
    haystack = re.sub(r"\s+", " ", html.unescape(rendered))
    for text in texts:
        needle = probe_slice(re.sub(r"\s+", " ", text))
        if needle and needle in haystack:
            raise AssertionError(
                f"Corpus text leaked into the built site for {corpus_key!r} "
                f"({corpus.label}, {corpus.licence}). Offending fragment: {needle[:80]!r}"
            )
