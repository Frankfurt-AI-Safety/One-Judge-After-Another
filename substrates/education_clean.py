"""
Plausibility filters for the education arm's two designs.

`load_factorial_essays` serves the **sex × ethnicity × economic-status factorial**
(`pairs/factorial.py`): its factors fit any argumentative essay, so it keeps the whole corpus bar one
rule. `load_stage_essays` serves the single-axis **stage** contrast below. They are different
populations by design; every education result must say which.

The rest of this module is the stage filter (``grade_level`` in `pairs/markers.py`, poles: 6th grade vs
doctoral candidate).

Every essay is rendered at both poles, so the body must not contradict either. PERSUADE essays are
written by pupils in grades 6–12, and the cues that pin a writer to school age come overwhelmingly
from the *prompt* rather than from scattered phrasing: the six prompts that ask pupils about their own
school life draw first-person pupil talk in 46–67% of essays, while the source-text prompts
("The Face on Mars", "Exploring Venus", "Driverless cars", …) are essentially free of it (0–10%).
Nothing about arguing over the electoral college presupposes a child wrote it.

So there are two steps, both reported through `substrates/rules.py`:

1. **Prompt selection** (``prompt_presupposes_a_pupil``) — keep only the stage-neutral prompts. This
   is selection, not plausibility: those essays are fine, they just cannot carry a doctoral pole.
2. **Cue rules** on the remaining bodies, as a safety net for the residual few percent: first-person
   school life, a named school stage, the writer's own grade, pupil self-reference, school routine,
   and letters addressed to a school authority (PERSUADE contains many "Dear Principal" letters).

"college"/"university" mentions are deliberately *not* a rule: a pupil writing "when I go to college"
is plausible, and so is a doctoral candidate mentioning a university.

What this cannot fix is **register**: the essays read like grades 6–12 whatever the marker claims.
That is why the low pole is 6th grade — the corpus's own floor, so at least some essays genuinely are
that young — and why the stage axis is read together with the quality interaction (see the working
notes): norm-referenced grading predicts a gap that grows with essay quality, while a flat offset is
a prior about the writer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Collection, Dict, List, Optional, Tuple

from substrates.education_ingest import (
    DEFAULT_ASAP_PATH, DEFAULT_PERSUADE_PATH, EssayRecord, load_asap, load_persuade,
)
from substrates.rules import Rule, apply_rules

# PERSUADE prompts that do not presuppose a school pupil (source-text arguments and general policy
# questions). The excluded six ask pupils about their own schooling: "Cell phones at school",
# "Community service", "Mandatory extracurricular activities", "Grades for extracurricular
# activities", "Distance learning", "Summer projects".
NEUTRAL_PROMPTS: frozenset = frozenset({
    "Does the electoral college work?",
    "Facial action coding system",
    "Exploring Venus",
    "Driverless cars",
    '"A Cowboy Who Rode the Waves"',
    "Car-free cities",
    "The Face on Mars",
    "Seeking multiple opinions",
    "Phones and driving",
})

_OWN_SCHOOL_RE = re.compile(
    r"\b(?:my|our)\s+(?:teacher|teachers|school|schools|class|classes|classmate|classmates|"
    r"principal|homeroom|parents|mom|dad)\b", re.IGNORECASE)
_STAGE_NAMED_RE = re.compile(
    r"\b(?:elementary|middle school|high school|junior high|primary school|grade school)\b",
    re.IGNORECASE)
_OWN_GRADE_RE = re.compile(
    r"\b(?:sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|[6-9]th|1[0-2]th)\s+grade\b",
    re.IGNORECASE)
_PUPIL_VOICE_RE = re.compile(
    r"\b(?:we|us)\s+(?:kids|students|teens|teenagers)\b|\bas\s+a\s+(?:kid|teen|teenager|student)\b",
    re.IGNORECASE)
_ROUTINE_RE = re.compile(r"\b(?:homework|recess|detention|report card)\b", re.IGNORECASE)
_SCHOOL_LETTER_RE = re.compile(
    r"\bdear\s+(?:principal|teacher|mr|mrs|ms|superintendent)\b", re.IGNORECASE)

TEXT_RULES: Tuple[Rule, ...] = (
    ("mentions_own_school_life", lambda r: bool(_OWN_SCHOOL_RE.search(r.essay_text))),
    ("names_a_school_stage", lambda r: bool(_STAGE_NAMED_RE.search(r.essay_text))),
    ("states_own_grade", lambda r: bool(_OWN_GRADE_RE.search(r.essay_text))),
    ("speaks_as_a_pupil", lambda r: bool(_PUPIL_VOICE_RE.search(r.essay_text))),
    ("mentions_school_routine", lambda r: bool(_ROUTINE_RE.search(r.essay_text))),
    ("addresses_a_school_authority", lambda r: bool(_SCHOOL_LETTER_RE.search(r.essay_text))),
)

# --- factorial (sex x ethnicity x economic status) -------------------------------------------------
# The factorial's third factor is economic status, not stage, so it needs neither the prompt selection
# nor the pupil-cue rules above: an injected income level contradicts nothing in an argumentative essay.
# The one rule is for the handful of essays that talk about their own family's money (5 of 6,404) — kept
# as a named rule so the drop is counted rather than assumed away.
_OWN_MONEY_RE = re.compile(
    r"\b(?:my|our)\s+(?:family|parents|mom|dad|household)\b[^.]{0,40}"
    r"\b(?:poor|rich|wealthy|afford|money|income|broke|struggl)\w*", re.IGNORECASE)

FACTORIAL_RULES: Tuple[Rule, ...] = (
    ("mentions_own_household_money", lambda r: bool(_OWN_MONEY_RE.search(r.essay_text))),
)

_SOURCES = {"persuade": (load_persuade, DEFAULT_PERSUADE_PATH),
            "asap": (load_asap, DEFAULT_ASAP_PATH)}

# `prompts` default differs per corpus, and None is a meaningful value, so it needs a sentinel.
_PER_SOURCE = object()


def stage_rules(prompts: Optional[Collection[str]] = NEUTRAL_PROMPTS) -> Tuple[Rule, ...]:
    """The stage rules, with prompt selection first when `prompts` is given (None = cue rules only,
    which is all ASAP can do: its `prompt_id` is an essay-set number, not a prompt name)."""
    if not prompts:
        return TEXT_RULES
    keep = frozenset(prompts)
    return (("prompt_presupposes_a_pupil", lambda r: r.prompt_id not in keep),) + TEXT_RULES


STAGE_RULES: Tuple[Rule, ...] = stage_rules()


def load_stage_essays(
    path: str | Path | None = None,
    *,
    source: str = "persuade",
    n: Optional[int] = None,
    report: Optional[Dict[str, object]] = None,
    prompts: Optional[Collection[str]] = _PER_SOURCE,  # type: ignore[assignment]
    **kwargs,
) -> List[EssayRecord]:
    """Essays that can carry either pole of the stage axis.

    `source` picks the corpus ("persuade" | "asap"); `path` overrides its default location; `kwargs`
    go to that loader. `prompts` defaults to `NEUTRAL_PROMPTS` for PERSUADE and to None for ASAP
    (which has no prompt names); pass an explicit collection or None to override. The `n` cap is
    applied after the rules, so it counts usable essays. If `report` is a dict it is filled with the
    loader's own counts plus ``stage_rules``.
    """
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {sorted(_SOURCES)}, got {source!r}")
    loader, default_path = _SOURCES[source]
    if prompts is _PER_SOURCE:
        prompts = NEUTRAL_PROMPTS if source == "persuade" else None
    records = loader(path or default_path, report=report, **kwargs)
    kept, rules_report = apply_rules(records, stage_rules(prompts))
    if report is not None:
        report["stage_rules"] = rules_report
    return kept[:n] if n else kept


def load_factorial_essays(
    path: str | Path | None = None,
    *,
    source: str = "persuade",
    n: Optional[int] = None,
    report: Optional[Dict[str, object]] = None,
    **kwargs,
) -> List[EssayRecord]:
    """Essays that can carry every cell of the sex × ethnicity × economic-status factorial.

    Same shape as `load_stage_essays`, but with `FACTORIAL_RULES` and no prompt selection: the
    factorial's factors are compatible with any argumentative essay, so the full corpus is usable. The
    two loaders therefore return **different populations** — state which one a result came from.
    """
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {sorted(_SOURCES)}, got {source!r}")
    loader, default_path = _SOURCES[source]
    records = loader(path or default_path, report=report, **kwargs)
    kept, rules_report = apply_rules(records, FACTORIAL_RULES)
    if report is not None:
        report["factorial_rules"] = rules_report
    return kept[:n] if n else kept
