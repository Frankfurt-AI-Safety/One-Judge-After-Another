"""
Real-biography substrate for the hiring (CV-screening) arm — replaces the synthetic CV generator.

Source: **Bias in Bios** (De-Arteaga et al., 2019), HF mirror `LabHC/bias_in_bios` (MIT).
~400k third-person professional biographies scraped from Common Crawl, each labelled with one of 28
occupations and a binary gender inferred from pronouns/names.

Two things make this a better substrate than the synthetic CV generator it replaced (since removed):

1. **It is real text.** The erasure, reasoning and decision-response arms were all sitting on
   synthetic CVs, i.e. our most load-bearing evidence rested on the least externally valid data.
2. **It carries a real gender label.** That is what lets us ask whether our *injected* sex-marker
   direction points anywhere near the direction real demographic signal occupies — the check that,
   on German Credit, returned a *negative* cosine.

Quality label — **role-match**. Bias-in-Bios has no ordinal quality label, so `qualified` is defined
against the dataset's own occupation label: the rendered header names a *target role*, and
`qualified = (profession == target_role)`. Roughly half the records are assigned their own
profession (qualified) and half a different one (not qualified). Unlike a length/seniority heuristic
this is not invented by us, and it gives hiring a quality axis a reward model can genuinely judge —
which is exactly what credit and CV lack, where `acc_baseline` sits at chance and makes
cross-influence uninterpretable.

**Scrubbing is mandatory and is the risky part of this module.** The HF mirror ships only
`hard_text`, which *preserves* names and pronouns. Our design needs a neutral body so the injected
marker is the only demographic signal, so we neutralise pronouns/titles and strip the leading name
here. Note the Tier-1 validation gate *cannot* catch a bad scrub: it only compares the two poles to
each other, so a leaked "she" appears on both sides and passes. Residual leakage is therefore
quantified separately by `runners/validate_bios_scrub.py`, which probes the *real* gender label
out of scrubbed-bio activations. Known residual risk: a first name appearing mid-text (not at the
start) is not removed.

Privacy: these are biographies of identifiable real people. Raw and derived text stay uncommitted,
as with PERSUADE/ASAP — only aggregate metrics are shared.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

HF_DATASET_ID = "LabHC/bias_in_bios"
DEFAULT_BIOS_PATH = "data/demographic/cv/raw/bias_in_bios_train.parquet"

# Occupation label ids 0-27, per the dataset card. Index == label id.
PROFESSIONS: tuple = (
    "accountant", "architect", "attorney", "chiropractor", "comedian", "composer", "dentist",
    "dietitian", "dj", "filmmaker", "interior_designer", "journalist", "model", "nurse", "painter",
    "paralegal", "pastor", "personal_trainer", "photographer", "physician", "poet", "professor",
    "psychologist", "rapper", "software_engineer", "surgeon", "teacher", "yoga_teacher",
)

# Gender label ids, per the dataset card. Kept as the *real* label for validity checks only —
# never rendered into the text (the sex signal must come from the injected marker alone).
GENDER_LABELS: Dict[int, str] = {0: "male", 1: "female"}

# --- scrubbing ------------------------------------------------------------------------------------
# Ordered longest-first so "herself" is consumed before "her". Values are deliberately singular-they.
_PRONOUN_MAP = (
    ("herself", "themself"), ("himself", "themself"),
    ("hers", "theirs"), ("she", "they"), ("her", "their"), ("his", "their"),
    ("him", "them"), ("he", "they"),
)
_TITLE_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx)\.?\s+", re.IGNORECASE)
# Any of these surviving the scrub means the body still leaks sex. Shared with the test suite.
GENDERED_RE = re.compile(
    r"\b(?:she|he|her|hers|him|his|herself|himself|woman|man|female|male|"
    r"mr|mrs|ms|miss|daughter|son|wife|husband|mother|father)\b",
    re.IGNORECASE,
)
# A leading "Firstname Lastname is/was/has ..." span: 1-4 capitalised tokens before the first verb.
# The name itself is captured so every LATER occurrence can be removed too — stripping only the
# opening span is not enough. Real bios refer back to the person by name ("Call Valorie Knoop on
# ..."), which would leave a strongly sex-coded first name in a body we are claiming is neutral.
_LEADING_NAME_RE = re.compile(
    r"^(?:Dr\.?\s+|Prof\.?\s+|Professor\s+)?"
    r"(?P<name>(?:[A-Z][\w'`-]*\.?\s+){0,3}[A-Z][\w'`-]*)\s+"
    r"(?=(?:is|was|has|had|works|serves|received|earned|holds|graduated|joined|began|started|"
    r"currently|specialises|specializes)\b)"
)
_NAME_TITLES = {"Dr", "Dr.", "Prof", "Prof.", "Professor", "Mr", "Mrs", "Ms", "Miss", "Mx"}
# Phone numbers are direct contact details for identifiable people; drop them on privacy grounds.
_PHONE_RE = re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")


# Singular-they leaves third-person-singular auxiliaries stranded ("they has been", "they is").
# Only the high-frequency auxiliaries are repaired; lexical verbs ("they carries out") are left as
# they are. This is cosmetic either way: the body is byte-identical across an A/B pair, so it cannot
# bias any measured quantity — it only affects how natural the profile reads in absolute terms.
_AGREEMENT = ((r"\bthey has\b", "they have"), (r"\bThey has\b", "They have"),
              (r"\bthey is\b", "they are"), (r"\bThey is\b", "They are"),
              (r"\bthey was\b", "they were"), (r"\bThey was\b", "They were"),
              (r"\bthey does\b", "they do"), (r"\bThey does\b", "They do"),
              (r"\bthey hasn't\b", "they haven't"), (r"\bThey hasn't\b", "They haven't"),
              (r"\bthey isn't\b", "they aren't"), (r"\bThey isn't\b", "They aren't"),
              (r"\bthey wasn't\b", "they weren't"), (r"\bThey wasn't\b", "They weren't"),
              (r"\bthey doesn't\b", "they don't"), (r"\bThey doesn't\b", "They don't"))


def _tidy(text: str) -> str:
    """Repair the two artifacts the substitutions introduce: stranded singular auxiliaries, and a
    lowercase 'the applicant' left sitting at the start of a sentence."""
    for pat, repl in _AGREEMENT:
        text = re.sub(pat, repl, text)
    text = re.sub(r"(^|[.!?]\s+)the applicant\b",
                  lambda m: f"{m.group(1)}The applicant", text)
    return text


def _sub_case_preserving(text: str, word: str, repl: str) -> str:
    """Replace whole-word `word` with `repl`, keeping the original capitalisation pattern."""
    def _r(m: re.Match) -> str:
        return repl.capitalize() if m.group(0)[0].isupper() else repl
    return re.sub(rf"\b{word}\b", _r, text, flags=re.IGNORECASE)


def scrub_detailed(text: str) -> tuple:
    """Neutralise the sex signal carried by the biography itself.

    Returns ``(scrubbed_text, name_resolved)``. `name_resolved` is False when we could not identify
    the subject's name from the opening span — those bios are dropped by the loader, because an
    unidentified name is exactly the case where a sex-coded first name survives somewhere in the body.

    Steps: strip titles and phone numbers, identify the leading name span, remove *every* occurrence
    of that name (not just the opening one), map gendered pronouns to singular *they*, collapse
    whitespace. Idempotent. Best-effort, not a guarantee — see `runners/validate_bios_scrub.py`.
    """
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = _PHONE_RE.sub("[phone]", text)
    text = _TITLE_RE.sub("", text)

    m = _LEADING_NAME_RE.match(text)
    name_resolved = m is not None
    if m:
        tokens = [t.strip(".,;:") for t in m.group("name").split()]
        tokens = [t for t in tokens if t and t not in _NAME_TITLES and t[0].isupper()]
        text = text[m.end():]
        text = "The applicant " + text
        # Remove the full name first (so "Valorie Knoop" does not become "the applicant the
        # applicant"), then any remaining standalone token, including possessives.
        if len(tokens) > 1:
            full = r"\s+".join(re.escape(t) for t in tokens)
            text = re.sub(rf"\b{full}\b(?:'s)?", "the applicant", text)
        for tok in tokens:
            text = re.sub(rf"\b{re.escape(tok)}\b(?:'s)?", "the applicant", text)
        text = re.sub(r"(?:\bthe applicant\b\s+){2,}", "the applicant ", text, flags=re.IGNORECASE)

    for word, repl in _PRONOUN_MAP:
        text = _sub_case_preserving(text, word, repl)
    text = _tidy(text)
    return re.sub(r"\s+", " ", text).strip(), name_resolved


def scrub(text: str) -> str:
    """Scrubbed body only. See :func:`scrub_detailed` for the name-resolution flag."""
    return scrub_detailed(text)[0]


def readable(profession: str) -> str:
    """`software_engineer` -> `software engineer`."""
    return profession.replace("_", " ")


def with_article(profession: str) -> str:
    """`architect` -> `an architect`; `surgeon` -> `a surgeon`."""
    name = readable(profession)
    return f"{'an' if name[0].lower() in 'aeiou' else 'a'} {name}"


@dataclass
class RealCVRecord:
    """One real biography, scrubbed, with a role-match quality label.

    `qualified` and `role` are named to match what the downstream runners already read:
    `runners/run_reasoning_*.py` filter on `getattr(r, "qualified", True)` and `pairs/verdicts.py`
    reads `.role`.
    Renaming either would fail *silently* rather than raise.
    """

    source_record_id: str
    bio_text: str          # scrubbed body, held byte-identical across an A/B pair
    profession: str        # the person's true occupation
    target_role: str       # the role being screened for
    role: str              # target role with article, e.g. "a surgeon" (read by pairs/verdicts.py)
    qualified: bool        # True iff profession == target_role  (the quality ground truth)
    gender: int            # REAL label, 0=male / 1=female — validity checks only, never rendered
    extra: Dict[str, object] = field(default_factory=dict)


def _missing(path: Path) -> None:
    raise FileNotFoundError(
        f"Bias-in-Bios corpus not found at {path}. It is user-downloaded (not committed).\n"
        f"  Fetch and cache it once with:\n"
        f"    python runners/generate_bios.py --from-hub\n"
        f"  (downloads {HF_DATASET_ID}, MIT licence, and writes the parquet to {DEFAULT_BIOS_PATH})"
    )


def fetch_from_hub(dest: str | Path = DEFAULT_BIOS_PATH, split: str = "train") -> Path:
    """Download the HF mirror once and cache it locally as parquet, so loads are hermetic after."""
    from datasets import load_dataset  # imported lazily: only the fetch path needs it

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET_ID, split=split)
    ds.to_parquet(str(dest))
    return dest


def load_bias_in_bios(
    path: str | Path = DEFAULT_BIOS_PATH,
    *,
    n: Optional[int] = None,
    seed: int = 42,
    min_chars: int = 300,
    max_chars: int = 6000,
    professions: Optional[List[str]] = None,
) -> List[RealCVRecord]:
    """Load scrubbed biographies with a balanced role-match `qualified` label.

    Bios outside `[min_chars, max_chars]` are dropped, as are any whose scrubbed body still trips
    `GENDERED_RE` (cheap insurance against the most obvious leakage; the probe check in
    `validate_bios_scrub.py` is the real test). Order, sampling and role assignment are all
    deterministic in `seed`.
    """
    import pandas as pd  # lazy: keeps module import cheap for the pure-render tests

    path = Path(path)
    if not path.exists():
        _missing(path)
    df = pd.read_parquet(path)
    for col in ("hard_text", "profession", "gender"):
        if col not in df.columns:
            raise KeyError(
                f"Bias-in-Bios: expected column {col!r}; found {list(df.columns)}. "
                "The HF mirror schema may have changed."
            )

    pool = list(professions) if professions else list(PROFESSIONS)
    rng = random.Random(seed)
    records: List[RealCVRecord] = []
    for i, row in enumerate(df.itertuples(index=False)):
        prof = PROFESSIONS[int(row.profession)] if isinstance(row.profession, (int,)) else str(row.profession)
        if prof not in pool:
            continue
        body, name_resolved = scrub_detailed(str(row.hard_text))
        if not name_resolved:
            # We could not identify the subject's name from the opening span, so we cannot guarantee
            # it was removed from the rest of the body. With ~257k bios available we can afford to
            # drop these rather than risk a sex-coded first name confounding the sex axis.
            continue
        if not (min_chars <= len(body) <= max_chars):
            continue
        if GENDERED_RE.search(body):
            continue  # residual sex signal in the body would confound the injected marker
        # Balanced role-match: half keep their own profession, half get a different one.
        if rng.random() < 0.5:
            target, qualified = prof, True
        else:
            alternatives = [p for p in pool if p != prof]
            if not alternatives:
                continue
            target, qualified = rng.choice(alternatives), False
        records.append(RealCVRecord(
            source_record_id=f"bios-{i}",
            bio_text=body,
            profession=prof,
            target_role=target,
            role=with_article(target),
            qualified=qualified,
            gender=int(row.gender),
            # Raw (unscrubbed) body kept in memory only, as the reference arm for
            # runners/validate_bios_scrub.py. It is never rendered and never reaches the
            # manifest (pair_to_record serialises the GeneratedPair, not the source record).
            extra={"raw_bio": re.sub(r"\s+", " ", str(row.hard_text)).strip()},
        ))
    return _finalize(records, n, seed)


def _finalize(records: List[RealCVRecord], n: Optional[int], seed: int) -> List[RealCVRecord]:
    """Deterministic shuffle (+ optional truncate) so runs are reproducible from (n, seed)."""
    if not records:
        raise ValueError(
            "No biographies survived the length / scrub filters — check the corpus file "
            "(and whether the scrub is rejecting nearly everything)."
        )
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:n] if n else records
