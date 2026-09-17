"""
Plausibility filters for Bias-in-Bios records in the hiring factorial (`pairs/factorial.py`).

Every biography is rendered at all levels of sex × age × family status, so it must fit each level:
a clause saying the applicant is 30 must not sit next to a degree earned in 2005 or "25 years of
experience". Unlike credit, the fields are free text, so the rules are text patterns and err on the
side of dropping (about 37% of scrubbed bios on a 20k sample; ~60k of 94k remain).

- ``mentions_year_before_2014``: a 30-year-old in 2026 was born in 1996, so any event year up to 2013
  (before age 18) is implausible, and a 1900s/2000s date is usually a degree or first job.
- ``mentions_more_than_8_years``: "N years" with N > 8 (most often experience) exceeds a career that
  started after a degree at about 22.
- ``mentions_many_years_in_words``: "fifteen years", "two decades", ...
- ``mentions_retirement``: incompatible with "currently in continuous employment".

The family-status levels need no rule: the explicit clause ("currently on parental leave") fits a
continuing career, and the proxy (a parent-association role) is a volunteering activity.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from substrates.bios_ingest import DEFAULT_BIOS_PATH, RealCVRecord, load_bias_in_bios
from substrates.rules import Rule, apply_rules

_YEAR_RE = re.compile(r"\b(?:19\d\d|200\d|201[0-3])\b")
_YEARS_RE = re.compile(r"\b(\d{1,2})\+?\s*(?:-\s*)?(?:plus\s+)?years?\b", re.IGNORECASE)
_WORD_YEARS_RE = re.compile(
    r"\b(?:nine|ten|eleven|twelve|fifteen|twenty|thirty|forty)\b[\w\s-]{0,12}\byears?\b|\bdecades?\b",
    re.IGNORECASE,
)
_RETIRED_RE = re.compile(r"\bretire(?:d|ment)\b", re.IGNORECASE)

FACTORIAL_RULES: Tuple[Rule, ...] = (
    ("mentions_year_before_2014", lambda r: bool(_YEAR_RE.search(r.bio_text))),
    ("mentions_more_than_8_years",
     lambda r: any(int(m.group(1)) > 8 for m in _YEARS_RE.finditer(r.bio_text))),
    ("mentions_many_years_in_words", lambda r: bool(_WORD_YEARS_RE.search(r.bio_text))),
    ("mentions_retirement", lambda r: bool(_RETIRED_RE.search(r.bio_text))),
)


def load_factorial_bios(
    path: str | Path = DEFAULT_BIOS_PATH,
    *,
    n: Optional[int] = None,
    report: Optional[Dict[str, object]] = None,
    **kwargs,
) -> List[RealCVRecord]:
    """Scrubbed biographies that fit every level of the hiring factorial.

    `kwargs` go to `load_bias_in_bios`. The `n` cap is applied after the rules, so it counts usable
    records; the order is the loader's seeded shuffle. If `report` is a dict it is filled with the
    loader's own report plus ``factorial_rules`` (the counts of this module's rules).
    """
    records = load_bias_in_bios(path, report=report, **kwargs)
    kept, rules_report = apply_rules(records, FACTORIAL_RULES)
    if report is not None:
        report["factorial_rules"] = rules_report
    return kept[:n] if n else kept
