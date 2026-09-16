"""
Real-essay substrate for the education (grading) arm (education domain).

Like the hiring arm (Bias-in-Bios), this arm *loads* essays from an established
corpus and holds the essay body fixed — the demographic marker is injected only as a header clause
downstream (see `substrates/education_render.py` / `pairs/markers.py`), so a matched A/B pair differs by exactly the marker.

Two corpora (locked with the user; both user-downloaded, gitignored under
`data/demographic/education/raw/`):
- **PERSUADE 2.0** (primary): ~25k argumentative essays, grades 6–12, holistic score 1–6.
  CC BY-NC-SA 4.0 → research/measurement use only; do **not** redistribute derived essays.
- **ASAP-AES** (license-cleaner cross-check): 8 essay sets, grades 7–10, per-set score scales; names
  pre-redacted to `@PERSON@`/`@LOCATION@` tags (neutralized here so injected names are the only signal).

`high_quality` is the education analog of CV `qualified` / credit `credit_good` — the quality ground
truth the deferred/`crossinf` arm needs. Because essay quality is *what reward models natively score*,
this is the domain where cross-influence may finally be interpretable on a small RM. We threshold the
holistic score into a clean strong/weak contrast and **drop the middle** so the label is unambiguous.

No demographic fields are stored (matched-pair injection only, per the locked design).
"""

from __future__ import annotations

import csv
import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Default local paths (user downloads the corpora here; gitignored via /data).
DEFAULT_PERSUADE_PATH = "data/demographic/education/raw/persuade_2.0.csv"
DEFAULT_ASAP_PATH = "data/demographic/education/raw/asap_training_set_rel3.tsv"

# PERSUADE holistic score is 1–6; strong = top, weak = bottom, middle dropped for a clean contrast.
PERSUADE_STRONG_MIN = 5
PERSUADE_WEAK_MAX = 2

# ASAP scores are per-essay-set on different scales → normalize per set to [0,1], then tercile-split.
ASAP_STRONG_Q = 0.67
ASAP_WEAK_Q = 0.33

# Redaction tags in ASAP (and occasionally elsewhere) to neutralize so injected names are the only cue.
# Covers `@PERSON1`, the closed `@PERSON@` form (which used to become "someone@") and runs of adjacent
# tags such as ASAP's `@PERCENT1@NUM1`, which become a single "someone".
_REDACTION_RE = re.compile(r"(?:@\w+)+@?")

csv.field_size_limit(10_000_000)  # essays can be long


@dataclass
class EssayRecord:
    """One real essay. `essay_text` is the held-fixed body; `high_quality` is the quality label.
    No demographic fields — sex/ethnicity/grade markers are injected downstream via `pairs/markers.py`."""

    source_record_id: str
    essay_text: str
    holistic_score: float
    high_quality: bool  # True = strong essay (top tier); the ground-truth quality label
    source_dataset: str  # "persuade" | "asap"
    grade_level: Optional[str] = None
    prompt_id: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)


def _clean(text: str) -> str:
    """Neutralize redaction tags and collapse whitespace so the essay body is clean, single-spaced."""
    text = _REDACTION_RE.sub("someone", text)
    return re.sub(r"\s+", " ", text).strip()


def _missing(path: Path, corpus: str, how: str) -> None:
    # `data/` is gitignored, so on a fresh clone the raw directory does not exist and any
    # suggested download command would fail on the missing parent rather than on the download.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raise FileNotFoundError(
        f"{corpus} corpus not found at {path}. It is user-downloaded (not committed).\n  {how}\n"
        f"  (the directory {Path(path).parent} has just been created for you)"
    )


def _find_col(fieldnames: List[str], candidates: List[str], corpus: str, what: str) -> str:
    lut = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lut:
            return lut[cand.lower()]
    raise KeyError(
        f"{corpus}: could not find the {what} column (tried {candidates}). "
        f"Available columns: {fieldnames}. Pass the right name explicitly."
    )


def load_persuade(
    path: str | Path = DEFAULT_PERSUADE_PATH,
    *,
    n: Optional[int] = None,
    seed: int = 42,
    min_chars: int = 300,
    max_chars: int = 6000,
    text_col: Optional[str] = None,
    score_col: Optional[str] = None,
) -> List[EssayRecord]:
    """Load PERSUADE 2.0 essays with a strong/weak `high_quality` label from the holistic score.

    Column names are auto-detected (with sensible defaults) and can be overridden. Essays outside
    `[min_chars, max_chars]` and middle-score essays are dropped. Deterministic order/sample by `seed`.
    """
    path = Path(path)
    if not path.exists():
        _missing(path, "PERSUADE 2.0",
                 "Download from https://github.com/scrosseye/persuade_corpus_2.0 (CC BY-NC-SA 4.0) "
                 f"and place the CSV at {DEFAULT_PERSUADE_PATH}.")
    records: List[EssayRecord] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fn = reader.fieldnames or []
        tcol = text_col or _find_col(fn, ["full_text", "essay", "text"], "PERSUADE", "essay-text")
        scol = score_col or _find_col(fn, ["holistic_essay_score", "score", "holistic_score"],
                                      "PERSUADE", "holistic-score")
        gcol = next((c for c in fn if c.lower() in ("grade_level", "grade")), None)
        pcol = next((c for c in fn if c.lower() in ("prompt_name", "prompt", "assignment")), None)
        idcol = next((c for c in fn if c.lower() in ("essay_id_comp", "essay_id", "id")), None)
        # PERSUADE 2.0 is discourse-element-level (~11 rows/essay) → dedup by essay id. Without an id
        # column (or for a blank id) dedup on the cleaned body instead: a per-row fallback key would
        # silently turn every discourse row into its own copy of the essay.
        seen: set = set()
        for i, row in enumerate(reader):
            body = _clean(row.get(tcol, ""))
            eid = row.get(idcol) if idcol else None
            key = ("id", eid) if eid else ("text", body)
            if key in seen:
                continue
            seen.add(key)
            eid = eid or f"row{i}"
            if not (min_chars <= len(body) <= max_chars):
                continue
            try:
                score = float(row.get(scol, ""))
            except (TypeError, ValueError):
                continue
            if score >= PERSUADE_STRONG_MIN:
                hq = True
            elif score <= PERSUADE_WEAK_MAX:
                hq = False
            else:
                continue  # drop the middle for a clean strong/weak contrast
            records.append(EssayRecord(
                source_record_id=f"persuade-{eid}",
                essay_text=body, holistic_score=score, high_quality=hq, source_dataset="persuade",
                grade_level=row.get(gcol) if gcol else None,
                prompt_id=row.get(pcol) if pcol else None,
            ))
    return _finalize(records, n, seed)


def load_asap(
    path: str | Path = DEFAULT_ASAP_PATH,
    *,
    n: Optional[int] = None,
    seed: int = 42,
    min_chars: int = 300,
    max_chars: int = 6000,
) -> List[EssayRecord]:
    """Load ASAP-AES essays. Scores differ per essay-set, so we normalize within each set and take the
    top/bottom terciles as strong/weak (dropping the middle). ASAP is a latin-1 TSV with @-redactions."""
    path = Path(path)
    if not path.exists():
        _missing(path, "ASAP-AES",
                 "Download training_set_rel3.tsv from the Kaggle 2012 ASAP competition and place it at "
                 f"{DEFAULT_ASAP_PATH}.")
    # Pass 1: gather raw rows + per-set score ranges.
    raw: List[Dict[str, str]] = []
    set_scores: Dict[str, List[float]] = {}
    with open(path, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fn = reader.fieldnames or []
        tcol = _find_col(fn, ["essay"], "ASAP", "essay-text")
        scol = _find_col(fn, ["domain1_score", "score"], "ASAP", "score")
        setcol = _find_col(fn, ["essay_set", "set"], "ASAP", "essay-set")
        idcol = _find_col(fn, ["essay_id", "id"], "ASAP", "id")
        for row in reader:
            try:
                sc = float(row[scol])
            except (TypeError, ValueError, KeyError):
                continue
            row["_text"], row["_score"], row["_set"], row["_id"] = (
                row.get(tcol, ""), sc, row.get(setcol, "?"), row.get(idcol, ""))
            raw.append(row)
            set_scores.setdefault(row["_set"], []).append(sc)
    # Per-set tercile cutoffs.
    def _q(vals: List[float], q: float) -> float:
        s = sorted(vals)
        return s[min(int(q * (len(s) - 1)), len(s) - 1)]
    lo = {s: _q(v, ASAP_WEAK_Q) for s, v in set_scores.items()}
    hi = {s: _q(v, ASAP_STRONG_Q) for s, v in set_scores.items()}
    records: List[EssayRecord] = []
    for row in raw:
        body = _clean(row["_text"])
        if not (min_chars <= len(body) <= max_chars):
            continue
        eset, sc = row["_set"], row["_score"]
        if hi[eset] <= lo[eset]:
            continue  # degenerate set: no strong/weak contrast exists, so label none of it
        if sc >= hi[eset]:
            hq = True
        elif sc <= lo[eset]:
            hq = False
        else:
            continue
        records.append(EssayRecord(
            source_record_id=f"asap-{eset}-{row['_id']}",
            essay_text=body, holistic_score=sc, high_quality=hq, source_dataset="asap",
            grade_level=f"set{eset}", prompt_id=f"set{eset}",
        ))
    return _finalize(records, n, seed)


def _finalize(records: List[EssayRecord], n: Optional[int], seed: int) -> List[EssayRecord]:
    """Deterministic shuffle (+ optional truncate) so runs are reproducible from (n, seed)."""
    if not records:
        raise ValueError("No essays survived the length/score filters — check the corpus file.")
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:n] if n else records
