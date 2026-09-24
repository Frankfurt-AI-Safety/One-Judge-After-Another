"""
Plausibility filters for the education arm — **one shared essay pool** for every education design.

Four consumers read essays: the A1 sex × ethnicity × economic-status factorial (`pairs/factorial.py`),
the A1 single-axis stage contrast and its ladder (``grade_level`` / ``stage_<rung>``), the A2
positioned-argument arm (`pairs/positionality.py`) and the cross-marker decision design (via the
factorial's cells.jsonl). They all go through
`load_education_essays`, so every education result is on the same essays. That matters most for the
A1-vs-A2 comparison: the two arms test the same attributes and differ only in whether the identity is
incidental metadata or load-bearing for the argument, and different pools would confound that with
population. (Until 2026-09-23 the factorial kept the whole corpus and the stage design a filtered
subset; the factorial costs its 40% strong rate for 25%, but 900+ strong essays remain.)

The pool must fit the most demanding claim any design makes about the writer: a 6th-grade pupil and a
doctoral candidate (stage), an adult standpoint such as "a retired teacher" (A2), and a household
income level (factorial).

Every essay is rendered at both poles, so the body must not contradict either. PERSUADE essays are
written by pupils in grades 6–12, and the cues that pin a writer to school age come overwhelmingly
from the *prompt* rather than from scattered phrasing: the six prompts that ask pupils about their own
school life draw first-person pupil talk in 46–67% of essays, while the source-text prompts
("The Face on Mars", "Exploring Venus", "Driverless cars", …) are essentially free of it (0–10%).
Nothing about arguing over the electoral college presupposes a child wrote it.

So the rules run in four steps, all reported through `substrates/rules.py`:

1. **Prompt selection** (``prompt_presupposes_a_pupil``) — keep only the stage-neutral prompts. This
   is selection, not plausibility: those essays are fine, they just cannot carry a doctoral pole.
2. **Cue rules** on the remaining bodies, as a safety net for the residual few percent: first-person
   school life, a named school stage, the writer's own grade, pupil self-reference, school routine,
   and letters addressed to a school authority (PERSUADE contains many "Dear Principal" letters).
   These matter for A2 as much as for stage: "As a retired teacher who has lived these realities
   firsthand" is incoherent on an essay that says "my teacher won't let us", and that is the
   `pos_control` pole, the axis whose job is to prove an effect is identity-specific.
3. **The household-money rule** for the economic factor (``mentions_own_household_money``).
4. **The ending rule** (``ends_with_a_signoff``, since 2026-09-24): the essay must end on a sentence, not a
   sign-off or a fragment (see `ENDING_RULES`).

"college"/"university" mentions are deliberately *not* a rule: a pupil writing "when I go to college"
is plausible, and so is a doctoral candidate mentioning a university.

What this cannot fix is **register**: the pool's essays read like grades 8–10 (the real grade of
1,344 of the 1,352 with a known grade) whatever the marker claims. The poles therefore do not rest on
the writers — the low pole, 6th grade, is where argument writing enters the curriculum (see
`pairs.markers.STAGE_LADDER`) — and the true register sits between them, so both poles are
counterfactual, in opposite directions. The stage axis is read together with the quality interaction
(see the working notes): norm-referenced grading predicts a gap that grows with essay quality, while a
flat offset is a prior about the writer.
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

# --- economic status (the factorial's third factor; also A2's pos_class) ------------------------------
# An injected income level contradicts almost nothing in an argumentative essay; the one rule is for the
# handful that talk about their own family's money (5 of 6,397) — kept as a named rule so the drop is
# counted rather than assumed away.
_OWN_MONEY_RE = re.compile(
    r"\b(?:my|our)\s+(?:family|parents|mom|dad|household)\b[^.]{0,40}"
    r"\b(?:poor|rich|wealthy|afford|money|income|broke|struggl)\w*", re.IGNORECASE)

ECONOMIC_RULES: Tuple[Rule, ...] = (
    ("mentions_own_household_money", lambda r: bool(_OWN_MONEY_RE.search(r.essay_text))),
)

# --- how the essay ends --------------------------------------------------------------------------------
# About 4% of essays end in a sign-off or a fragment instead of a sentence of the argument: "Sincerely,
# PROPER_NAME", a bare "PROPER_NAME", a real first name or surname with a page number ("Gabe", "Tellez 2"),
# "The End", a stray title, a cut-off line ("It h"). A2's conclusion paragraph would follow the signature,
# and a real name in a signature can contradict the sex or ethnicity marker A1 injects. The rate is about the
# same in both classes (4.1% of weak, 4.5% of strong essays before balancing).
_CLOSING_RE = re.compile(
    r"^\W*(?:sincerely|sincerly|regards|best regards|kind regards|respectfully|yours truly|"
    r"yours sincerely|thank you|thanks|signed)\b|,\s*sincere?ly\b", re.IGNORECASE)
_CLOSING_QUOTES = "\"')\u201d\u2019\x94"   # incl. a closing quote mis-decoded from cp1252


def ends_with_a_signoff(text: str) -> bool:
    """True if the last non-empty line opens with a closing formula ("Sincerely", "Thank you", …) or is a
    fragment: at most four words without terminal punctuation (closing quotes ignored)."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return False
    last = lines[-1]
    return bool(_CLOSING_RE.search(last)) or (
        len(last.split()) <= 4 and not last.rstrip(_CLOSING_QUOTES).endswith((".", "!", "?")))


ENDING_RULES: Tuple[Rule, ...] = (
    ("ends_with_a_signoff", lambda r: ends_with_a_signoff(r.essay_text)),
)

_SOURCES = {"persuade": (load_persuade, DEFAULT_PERSUADE_PATH),
            "asap": (load_asap, DEFAULT_ASAP_PATH)}

# `prompts` default differs per corpus, and None is a meaningful value, so it needs a sentinel.
_PER_SOURCE = object()


def stage_rules(prompts: Optional[Collection[str]] = NEUTRAL_PROMPTS) -> Tuple[Rule, ...]:
    """The pupil-cue rules, with prompt selection first when `prompts` is given (None = cue rules only,
    which is all ASAP can do: its `prompt_id` is an essay-set number, not a prompt name)."""
    if not prompts:
        return TEXT_RULES
    keep = frozenset(prompts)
    return (("prompt_presupposes_a_pupil", lambda r: r.prompt_id not in keep),) + TEXT_RULES


def education_rules(prompts: Optional[Collection[str]] = NEUTRAL_PROMPTS) -> Tuple[Rule, ...]:
    """Every rule the shared pool applies: prompt selection, pupil cues, household money, the ending."""
    return stage_rules(prompts) + ECONOMIC_RULES + ENDING_RULES


EDUCATION_RULES: Tuple[Rule, ...] = education_rules()


def load_education_essays(
    path: str | Path | None = None,
    *,
    source: str = "persuade",
    n: Optional[int] = None,
    report: Optional[Dict[str, object]] = None,
    prompts: Optional[Collection[str]] = _PER_SOURCE,  # type: ignore[assignment]
    balance: bool = True,
    **kwargs,
) -> List[EssayRecord]:
    """The shared education pool: essays that fit every claim any education design makes.

    `source` picks the corpus ("persuade" | "asap"); `path` overrides its default location; `kwargs`
    go to that loader (including its `seed`). `prompts` defaults to `NEUTRAL_PROMPTS` for PERSUADE and
    to None for ASAP (which has no prompt names); pass an explicit collection or None to override.

    `balance` (default on) equalises strong and weak essays **within each prompt**, so the quality label
    is independent of the prompt. Balancing the pool as a whole is not enough: after the pupil-voice rules
    the label is so uneven across prompts ("Seeking multiple opinions" and "Phones and driving" 88% strong,
    "A Cowboy Who Rode the Waves" 3%) that the prompt alone would predict it 67% of the time, and the
    rendered header names the assignment — the same leak the hiring role name had (76.5%). Per prompt
    (per essay set for ASAP) the first ``min(strong, weak)`` essays of each class are kept, in the
    loader's seeded order, as matched strong/weak couples; `n` then keeps the first ``n // 2`` couples
    (an odd `n` rounds down), so a capped sample is balanced overall and within every prompt. If `report`
    is a dict it is filled with the loader's own counts plus ``education_rules`` and ``balance``.
    """
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {sorted(_SOURCES)}, got {source!r}")
    loader, default_path = _SOURCES[source]
    if prompts is _PER_SOURCE:
        prompts = NEUTRAL_PROMPTS if source == "persuade" else None
    records = loader(path or default_path, report=report, **kwargs)
    kept, rules_report = apply_rules(records, education_rules(prompts))
    if report is not None:
        report["education_rules"] = rules_report
    if not balance:
        return kept[:n] if n else kept
    out, balance_report = _balance_classes(kept, n)
    if report is not None:
        report["balance"] = balance_report
    return out


def _balance_classes(records: List[EssayRecord],
                     n: Optional[int]) -> Tuple[List[EssayRecord], Dict[str, object]]:
    """Equal strong/weak counts within each prompt, kept in the input order (see `load_education_essays`)."""
    order = {id(r): i for i, r in enumerate(records)}
    by_prompt: Dict[object, Tuple[List[EssayRecord], List[EssayRecord]]] = {}
    for r in records:
        by_prompt.setdefault(r.prompt_id, ([], []))[0 if r.high_quality else 1].append(r)
    # zip() cuts each prompt to min(strong, weak) couples; order couples by their earlier member
    couples = [c for strong, weak in by_prompt.values() for c in zip(strong, weak)]
    couples.sort(key=lambda c: min(order[id(c[0])], order[id(c[1])]))
    if n:
        couples = couples[: n // 2]
    chosen = {id(r) for c in couples for r in c}
    out = [r for r in records if id(r) in chosen]
    per_prompt: Dict[str, int] = {}
    for strong, _ in couples:
        per_prompt[str(strong.prompt_id)] = per_prompt.get(str(strong.prompt_id), 0) + 1
    return out, {"strong_in": sum(r.high_quality for r in records),
                 "weak_in": sum(not r.high_quality for r in records),
                 "per_class": len(couples), "n_out": len(out), "per_prompt": per_prompt}
