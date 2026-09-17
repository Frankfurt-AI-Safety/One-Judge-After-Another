"""
Real-biography substrate for the hiring (CV-screening) arm — replaces the synthetic CV generator.

Source: **Bias in Bios** (De-Arteaga et al., 2019), HF mirror `LabHC/bias_in_bios` (MIT).
~257k third-person professional biographies (train split) scraped from Common Crawl, each labelled with
one of 28 occupations and a binary gender inferred from pronouns/names. The corpus's `hard_text` has the
first sentence (the one naming the occupation) removed, so most bios open with "He"/"She".

Two things make this a better substrate than the synthetic CV generator it replaced (since removed):

1. **It is real text.** The erasure, reasoning and decision-response arms were all sitting on
   synthetic CVs, i.e. our most load-bearing evidence rested on the least externally valid data.
2. **It carries a real gender label.** That is what lets us ask whether our *injected* sex-marker
   direction points anywhere near the direction real demographic signal occupies.

Quality label — **role-match**. Bias-in-Bios has no ordinal quality label, so `qualified` is defined
against the dataset's own occupation label: the rendered header names a *target role*, and
`qualified = (profession == target_role)`. Roughly half the records are assigned their own
profession (qualified) and half a different one (not qualified). Unlike a length/seniority heuristic
this is not invented by us, and it gives hiring a quality axis a reward model can plausibly judge.

**Scrubbing is mandatory and is the risky part of this module.** The body must carry no sex signal, so
the injected marker is the only one. `scrub_detailed` neutralises pronouns and titles, removes the
subject's name where the opening sentence gives it, and replaces contact details; the loader then
drops every bio whose scrubbed body still contains a gendered word (`GENDERED_RE`) or a sex-coded
first name (`first_name_hit`, list in `resources/first_names.txt`). The first-name drop is the main
guard: most bios open with a pronoun, so the subject's name is usually *not* identified up front, and
checked on a 20k sample, about 36% of otherwise clean bodies still named a person (mostly the subject).
Known residual risks: first names missing from the list, occupational gender priors (surgeon vs
nurse), and topical words ("women's health"), whose rate is reported, not filtered. The Tier-1
validation gate *cannot* catch a bad scrub (it compares the two poles to each other, so a leaked
"she" appears on both sides and passes); residual leakage is quantified by
`runners/validate_bios_scrub.py`, which probes the *real* gender label out of scrubbed activations.

Privacy: these are biographies of identifiable real people. Raw and derived text stay uncommitted,
as with PERSUADE/ASAP — only aggregate metrics are shared.
"""

from __future__ import annotations

import hashlib
import numbers
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, NoReturn, Optional, Sequence, Tuple

HF_DATASET_ID = "LabHC/bias_in_bios"
DEFAULT_BIOS_PATH = "data/demographic/cv/raw/bias_in_bios_train.parquet"
# Resolved relative to this file (not the cwd); built by runners/build_first_names.py.
FIRST_NAMES_PATH = Path(__file__).resolve().parent / "resources" / "first_names.txt"

# Occupation label ids 0-27, per the dataset card. Index == label id.
PROFESSIONS: Tuple[str, ...] = (
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
# Case-sensitive and only before a capitalised word: with IGNORECASE this also deleted "MS" (the
# degree) and the verb "miss".
_TITLE_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx)\.?\s+(?=[A-Z])")
# Any of these surviving the scrub means the body still leaks sex; the loader drops such bios.
# Shared with the test suite. Titles are matched case-sensitively for the same reason as above.
# "women"/"men" are deliberately NOT listed (mostly topical, e.g. "women's health"); the loader
# reports how many kept bios mention them.
GENDERED_RE = re.compile(
    r"(?-i:\b(?:Mr|Mrs|Ms|Miss|MR|MRS|MS)\b\.?(?=\s+[A-Z]))|"
    r"\b(?:she|he|her|hers|him|his|herself|himself|woman|man|female|male|"
    r"daughter|son|wife|husband|mother|father|mom|mum|dad|sister|brother|girl|boy|lady|gentleman|"
    r"grandmother|grandfather|grandma|grandpa|aunt|uncle|niece|nephew|girlfriend|boyfriend|"
    r"stepmother|stepfather|fianc[eé]e?|actress|waitress|hostess|businessman|businesswoman|"
    r"chairman|chairwoman|spokesman|spokeswoman|congressman|congresswoman)\b",
    re.IGNORECASE,
)
WOMEN_MEN_RE = re.compile(r"\b(?:women|men)\b", re.IGNORECASE)
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
# Capitalised words the name pattern would otherwise take for a name ("She has ...", "This is ...").
# Most bios open with a pronoun; treating it as the subject's name made the name check meaningless.
_NOT_A_NAME = {"He", "She", "His", "Her", "Hers", "Him", "They", "Their", "It", "Its", "We", "Our",
               "I", "My", "This", "That", "These", "Those", "The", "A", "An", "In", "As", "At", "For",
               "With", "After", "Since", "Currently", "Today"}
# Contact details identify real people and often contain their name; drop them on privacy grounds.
_PHONE_RE = re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S*[^\s.,;:!?)\]]", re.IGNORECASE)


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


def _replace_name(text: str, name_pattern: str) -> str:
    """Replace a name with "the applicant", keeping a possessive ("Knoop's" -> "the applicant's")."""
    return re.sub(rf"\b{name_pattern}\b('s)?", lambda m: "the applicant" + (m.group(1) or ""), text)


def scrub_detailed(text: str) -> Tuple[str, bool]:
    """Neutralise the sex signal carried by the biography itself.

    Returns ``(scrubbed_text, name_resolved)``. `name_resolved` is True when the opening sentence gave
    the subject's name ("Jane Doe is ..."), which is then removed throughout. Most bios open with a
    pronoun instead, so this is informational: a name mentioned later is caught by the loader's
    first-name drop (:func:`first_name_hit`), not here.

    Steps: collapse whitespace; replace phone numbers, emails and URLs; strip titles before a name;
    if the opening names the subject, replace that span with "The applicant" and every later use of
    the full name and of the first name (surnames are not sex-coded and are left alone, since they
    often double as ordinary words); map gendered pronouns to singular *they*; tidy. The text output
    is idempotent. Best-effort, not a guarantee — see `runners/validate_bios_scrub.py`.
    """
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = _PHONE_RE.sub("[phone]", text)
    text = _EMAIL_RE.sub("[email]", text)
    text = _URL_RE.sub("[url]", text)
    text = _TITLE_RE.sub("", text)

    m = _LEADING_NAME_RE.match(text)
    tokens: List[str] = []
    if m:
        tokens = [t.strip(".,;:") for t in m.group("name").split()]
        tokens = [t for t in tokens if t and t not in _NAME_TITLES and t[0].isupper()]
    name_resolved = bool(tokens) and tokens[0] not in _NOT_A_NAME
    if name_resolved:
        text = "The applicant " + text[m.end():]
        # Full name first (so "Valorie Knoop" does not become "the applicant Knoop"), then the first
        # name on its own. Single-letter initials are skipped ("J." would mangle "J.D.").
        if len(tokens) > 1:
            text = _replace_name(text, r"\s+".join(re.escape(t) for t in tokens))
        first = next((t for t in tokens if len(t) > 1), None)
        if first:
            text = _replace_name(text, re.escape(first))
        text = re.sub(r"\b(the applicant)(?:\s+the applicant\b)+", r"\1", text, flags=re.IGNORECASE)

    for word, repl in _PRONOUN_MAP:
        text = _sub_case_preserving(text, word, repl)
    text = _tidy(text)
    return re.sub(r"\s+", " ", text).strip(), name_resolved


def scrub(text: str) -> str:
    """Scrubbed body only. See :func:`scrub_detailed` for the name-resolution flag."""
    return scrub_detailed(text)[0]


# --- first-name drop ------------------------------------------------------------------------------
# A listed name in these contexts is a place or an institution, not a person: "Santa Barbara",
# "St. Louis", "University of Denver", "Wisconsin-Madison", "Cornell University". Contexts not covered
# here (e.g. "MD Anderson") simply cost the bio.
_PLACE_BEFORE = {"in", "at", "of", "from", "near", "uc", "san", "santa", "santo", "st", "saint", "north",
                 "south", "east", "west", "new", "fort", "lake", "mount", "puerto", "costa", "los", "las",
                 "el", "port"}
_PLACE_AFTER = {"university", "college", "institute", "school", "county", "street", "avenue", "hospital",
                "state", "tech", "city", "island", "river", "valley", "beach", "park", "center", "centre",
                "medical", "hall", "road", "drive", "square", "district", "province", "bay", "springs",
                "heights", "clinic", "foundation", "cancer", "health", "memorial", "children", "award",
                "prize", "fellowship", "scholarship", "lab", "laboratory", "library", "museum", "theatre",
                "theater", "airport", "law", "business"}
# Capitalised words, including mixed case ("DeAndre", "JoAnne"). All-caps words are not checked: in
# this corpus they are almost always acronyms (ADA, TED, RICO, IRA), rarely names.
_CAP_WORD_RE = re.compile(r"\b[A-Z][A-Za-z]*[a-z][A-Za-z]*\b")
_PREV_WORD_RE = re.compile(r"([A-Za-z]+)\.?\s*$")
_NEXT_WORD_RE = re.compile(r"\s+([A-Za-z]+)")
_first_names: Optional[FrozenSet[str]] = None
_first_names_lc: Optional[FrozenSet[str]] = None


def first_names() -> FrozenSet[str]:
    """The sex-coded first-name list (loaded once)."""
    global _first_names
    if _first_names is None:
        lines = FIRST_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        _first_names = frozenset(line.strip() for line in lines
                                 if line.strip() and not line.startswith("#"))
    return _first_names


def _first_names_lower() -> FrozenSet[str]:
    global _first_names_lc
    if _first_names_lc is None:
        _first_names_lc = frozenset(n.lower() for n in first_names())
    return _first_names_lc


def first_name_hit(text: str) -> Optional[str]:
    """The first sex-coded first name in `text` used as a person's name, or None.

    Only capitalised tokens count, compared case-insensitively ("DeAndre" matches "Deandre"); all-caps
    tokens are skipped (see ``_CAP_WORD_RE``). A match right after a place cue (``_PLACE_BEFORE`` or a
    hyphen) or right before an institution cue (``_PLACE_AFTER``) is skipped.
    """
    names = _first_names_lower()
    for m in _CAP_WORD_RE.finditer(text):
        if m.group(0).lower() not in names:
            continue
        before = text[max(0, m.start() - 20):m.start()]
        if before.endswith("-"):
            continue
        prev = _PREV_WORD_RE.search(before)
        if prev and prev.group(1).lower() in _PLACE_BEFORE:
            continue
        nxt = _NEXT_WORD_RE.match(text, m.end())
        if nxt and nxt.group(1).lower() in _PLACE_AFTER:
            continue
        return m.group(0)
    return None


_DISPLAY_NAMES = {"dj": "DJ"}


def readable(profession: str) -> str:
    """`software_engineer` -> `software engineer`; `dj` -> `DJ`."""
    return _DISPLAY_NAMES.get(profession, profession.replace("_", " "))


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
    extra: Dict[str, object] = field(default_factory=dict)  # "raw_bio" only with keep_raw=True


def _missing(path: Path) -> NoReturn:
    raise FileNotFoundError(
        f"Bias-in-Bios corpus not found at {path}. It is user-downloaded (not committed).\n"
        f"  Fetch and cache it once with:\n"
        f"    python runners/generate_bios.py --from-hub [--raw-path {path}]\n"
        f"  (downloads {HF_DATASET_ID}, MIT licence, and writes the parquet to --raw-path, default "
        f"{DEFAULT_BIOS_PATH})"
    )


def fetch_from_hub(dest: str | Path = DEFAULT_BIOS_PATH, split: str = "train") -> Path:
    """Download the HF mirror once and cache it locally as parquet, so loads are hermetic after."""
    from datasets import load_dataset  # imported lazily: only the fetch path needs it

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET_ID, split=split)
    ds.to_parquet(str(dest))
    return dest


def _coin(seed: int, row: int) -> bool:
    """A per-bio fair coin, fixed by (seed, row): filter changes elsewhere do not flip it."""
    digest = hashlib.sha256(f"{seed}|{row}".encode("utf-8")).digest()
    return digest[0] < 128


def _assign_mismatched_roles(professions: Sequence[str], rng: random.Random) -> List[str]:
    """A target role for each unqualified bio: the group's own professions, shuffled so that nobody
    keeps their own. The target roles of the unqualified group then have the same distribution as
    its professions -- the same as the qualified group's roles, since the coin is independent of the
    profession -- so the role name alone says nothing about `qualified`.

    Possible iff no profession is more than half of the group.
    """
    n = len(professions)
    top = max((professions.count(p) for p in set(professions)), default=0)
    if 2 * top > n:
        raise ValueError(
            f"cannot give {n} unqualified bios a different profession: one profession covers "
            f"{top} of them (more than half); widen the profession pool"
        )
    targets = list(professions)
    rng.shuffle(targets)
    for k in range(n):
        if targets[k] != professions[k]:
            continue
        # Swap with a position j where both ends stay mismatched; never creates a new fixed point.
        start = rng.randrange(n)
        for step in range(n):
            j = (start + step) % n
            if targets[j] != professions[k] and targets[k] != professions[j]:
                targets[k], targets[j] = targets[j], targets[k]
                break
        else:  # pragma: no cover - excluded by the majority check above
            raise ValueError("no valid swap for a mismatched role assignment")
    return targets


def load_bias_in_bios(
    path: str | Path = DEFAULT_BIOS_PATH,
    *,
    n: Optional[int] = None,
    seed: int = 42,
    min_chars: int = 300,
    max_chars: int = 6000,
    professions: Optional[List[str]] = None,
    keep_raw: bool = False,
    report: Optional[Dict[str, object]] = None,
) -> List[RealCVRecord]:
    """Load scrubbed biographies with a balanced role-match `qualified` label.

    After scrubbing, a bio is dropped if its body is outside `[min_chars, max_chars]`, still matches
    `GENDERED_RE`, or still names a person by a sex-coded first name (:func:`first_name_hit`). The
    probe check in `validate_bios_scrub.py` is the real test of what remains.

    Role match: each kept bio gets a fair coin fixed by (seed, row). Heads -> its own profession is
    the target role (`qualified=True`). Tails -> it gets another tails bio's profession
    (:func:`_assign_mismatched_roles`), so target roles are distributed alike in both groups and the
    role name carries no information about the label. (Drawing mismatched roles uniformly, as before
    2026-09-17, let the role name alone predict `qualified` with 76.5% accuracy: common professions
    were almost always matches, rare ones almost never.)

    `keep_raw=True` keeps the unscrubbed body in `extra["raw_bio"]` (only the scrub check needs it).
    If `report` is a dict it is filled with the drop counts per reason (overall and per real gender),
    the number kept, and the share of kept bios mentioning "women"/"men" (not filtered, see
    `GENDERED_RE`). Everything is deterministic in `seed`.
    """
    import pandas as pd  # lazy: keeps module import cheap for the pure-render tests

    pool = list(professions) if professions else list(PROFESSIONS)
    unknown = sorted(set(pool) - set(PROFESSIONS))
    if unknown:
        raise ValueError(f"unknown professions {unknown}; known: {list(PROFESSIONS)}")
    if len(set(pool)) < 2:
        raise ValueError("role match needs at least two professions in the pool")

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

    reasons = ("profession_pool", "length", "gendered_word", "first_name")
    dropped = {r: {0: 0, 1: 0} for r in reasons}

    def drop(reason: str, gender: int) -> None:
        dropped[reason][gender] = dropped[reason].get(gender, 0) + 1

    # Pass 1: filter.
    kept: List[Tuple[int, str, str, int, str]] = []  # (row, profession, body, gender, raw text)
    for i, row in enumerate(df.itertuples(index=False)):
        gender = int(row.gender)
        label = row.profession
        prof = PROFESSIONS[int(label)] if isinstance(label, numbers.Integral) else str(label)
        if prof not in pool:
            drop("profession_pool", gender)
            continue
        raw = str(row.hard_text)
        body, _ = scrub_detailed(raw)
        if not (min_chars <= len(body) <= max_chars):
            drop("length", gender)
            continue
        if GENDERED_RE.search(body):
            drop("gendered_word", gender)  # residual sex signal would confound the injected marker
            continue
        if first_name_hit(body):
            drop("first_name", gender)  # a named person, most often the subject: a direct sex cue
            continue
        kept.append((i, prof, body, gender, raw))

    # Pass 2: balanced role match.
    heads = [_coin(seed, i) for i, *_ in kept]
    tails = [k for k, h in enumerate(heads) if not h]
    mismatched = _assign_mismatched_roles([kept[k][1] for k in tails], random.Random(seed))
    target_of = dict(zip(tails, mismatched))

    records: List[RealCVRecord] = []
    for k, (i, prof, body, gender, raw) in enumerate(kept):
        target = prof if heads[k] else target_of[k]
        records.append(RealCVRecord(
            source_record_id=f"bios-{i}",
            bio_text=body,
            profession=prof,
            target_role=target,
            role=with_article(target),
            qualified=heads[k],
            gender=gender,
            # Unscrubbed body, in memory only and only on request: the reference arm of
            # runners/validate_bios_scrub.py. Never rendered and never written to a manifest
            # (pair_to_record serialises the GeneratedPair, not the source record).
            extra={"raw_bio": re.sub(r"\s+", " ", raw).strip()} if keep_raw else {},
        ))

    if report is not None:
        kept_by_gender = {0: 0, 1: 0}
        for r in records:
            kept_by_gender[r.gender] = kept_by_gender.get(r.gender, 0) + 1
        women_men = sum(bool(WOMEN_MEN_RE.search(r.bio_text)) for r in records)
        report.update({
            "n_rows": len(df),
            "dropped": {r: dict(v) for r, v in dropped.items()},
            "kept": len(records),
            "kept_by_gender": kept_by_gender,
            "kept_qualified": sum(heads),
            "kept_mentioning_women_or_men": women_men,
            "kept_mentioning_women_or_men_rate": round(women_men / max(len(records), 1), 4),
        })
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
    return records if n is None else records[:n]
