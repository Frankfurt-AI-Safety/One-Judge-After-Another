"""
Tier-1 structural validation gate — a matched pair may differ in its marker slot only.

Deterministic checks, no model involved:
1. **Single-slot difference** — each side's marker clause occurs exactly once, the two clauses differ,
   removing them leaves byte-identical remainders, and both start at the same index (so the text before
   and after the clause is identical). This is a slot check, not an axis check: a corner pair varies
   three attributes inside its one clause and passes; what fails is any difference outside the slot.
2. **Length parity** — |Δcharacters| and |Δwords| within bounds, so the marker is not a gross length cue.
   ``max_token_delta`` counts whitespace-separated words, not model tokens (the name is kept: it is a CLI
   flag and a manifest key).
3. **Readability parity** — |ΔFlesch reading ease| of the two full texts within bound (``textstat``, a
   required dependency: a gate that skipped the check would pass pairs without saying so). Measured on the
   whole text, it loosens as items grow: in the essay builds on the earlier corpora one clause moved the
   score by at most 2.5 points (ASAP-AES, essays from 63 words) and under 1 on PERSUADE's longer essays,
   against a bound of 8.

The generators gate block by block: a block (one record × template × encoding of a factorial, one stage
block, one positioned essay) with any failing pair is dropped whole, so the design stays balanced, and the
drop counts per reason code (`tally_reasons`) go into the manifest's discard report. Nothing is
over-generated to replace a dropped block. As of 2026-09-26 no block has failed in any build: the gate
guards against a builder mistake rather than filtering pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import textstat

from pairs.markers import GeneratedPair


@dataclass
class Thresholds:
    """Largest allowed |Δ| between the two sides: characters, whitespace words, Flesch reading ease."""

    max_char_delta: int = 12
    max_token_delta: int = 3
    max_flesch_delta: float = 8.0


@dataclass
class ValidationResult:
    """``reasons`` are the messages; ``codes`` the matching stable keys (one of `REASON_CODES`) that
    discard reports count by (`tally_reasons`), since a message carries the measured value."""

    ok: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    codes: List[str] = field(default_factory=list)


REASON_CODES = ("clause_not_once", "content_differs", "clause_position", "no_contrast", "char_delta",
                "token_delta", "flesch_delta")


def _strip_once(text: str, clause: str) -> Tuple[str, int]:
    """Return (text with the first occurrence of clause removed, occurrence count)."""
    return text.replace(clause, "", 1), text.count(clause)


def validate_pair(pair: GeneratedPair, thr: Optional[Thresholds] = None) -> ValidationResult:
    """Run the three checks of the module docstring on one pair; ``ok`` only if none fails."""
    thr = thr or Thresholds()
    reasons: List[str] = []
    codes: List[str] = []

    def fail(code: str, message: str) -> None:
        codes.append(code)
        reasons.append(message)

    # 1. single-slot difference
    rem_a, n_a = _strip_once(pair.text_a, pair.clause_a)
    rem_b, n_b = _strip_once(pair.text_b, pair.clause_b)
    if n_a != 1 or n_b != 1:
        fail("clause_not_once", f"marker clause not found exactly once (a={n_a}, b={n_b})")
    if rem_a != rem_b:
        fail("content_differs", "non-marker content differs between the pair (not single-slot)")
    elif n_a == 1 and n_b == 1 and pair.text_a.index(pair.clause_a) != pair.text_b.index(pair.clause_b):
        # equal remainders + equal start = the same slot (a clause moved elsewhere leaves equal remainders)
        fail("clause_position", "marker clause sits at a different position on the two sides")
    if pair.clause_a == pair.clause_b:
        fail("no_contrast", "the two marker clauses are identical (nothing varies)")

    # 2. length parity
    char_delta = abs(len(pair.text_a) - len(pair.text_b))
    token_delta = abs(len(pair.text_a.split()) - len(pair.text_b.split()))
    if char_delta > thr.max_char_delta:
        fail("char_delta", f"char-length delta {char_delta} > {thr.max_char_delta}")
    if token_delta > thr.max_token_delta:
        fail("token_delta", f"token-length delta {token_delta} > {thr.max_token_delta}")

    # 3. readability parity
    flesch_delta = abs(textstat.flesch_reading_ease(pair.text_a) - textstat.flesch_reading_ease(pair.text_b))
    if flesch_delta > thr.max_flesch_delta:
        fail("flesch_delta", f"Flesch delta {flesch_delta:.1f} > {thr.max_flesch_delta}")

    return ValidationResult(
        ok=not reasons,
        reasons=reasons,
        metrics={"char_delta": char_delta, "token_delta": token_delta, "flesch_delta": flesch_delta},
        codes=codes,
    )


def tally_reasons(results: Iterable[ValidationResult], counts: Dict[str, int]) -> None:
    """Add the reason codes of ``results`` to ``counts`` (code -> number of failing pairs), in place."""
    for res in results:
        for code in res.codes:
            counts[code] = counts.get(code, 0) + 1
