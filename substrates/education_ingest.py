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

The **assignment** (the task the essay answers) is kept and rendered into the header, the way the hiring
header names the target role: most PERSUADE prompts are text-dependent, so without it the model grades an
answer to a question it cannot see. It is constant per prompt and identical across an A/B pair.

`high_quality` is the education analog of CV `qualified` / credit `credit_good` — the quality ground
truth the cross-marker design's decision accuracy needs (a strong essay should pass, a weak one fail).
Essay quality is *what reward models natively score*, so this is the domain where that accuracy is most
likely to be interpretable on a small RM. We threshold the holistic score into a clean strong/weak
contrast and **drop the middle** so the label is unambiguous.

**Real demographics are kept on the record but never rendered.** PERSUADE 2.0 ships the writer's
`gender`, `race_ethnicity` and `grade_level` (and ELL / economic / disability status, in `extra`);
they are per-essay consistent across the corpus's discourse rows. They reach `raw_sex`,
`raw_ethnicity` and `raw_grade_level`, the same way credit keeps `raw_sex`/`raw_marital` and
Bias-in-Bios keeps `gender`: matched-pair injection is still the only thing the model ever sees, but
the real labels let the generator write `real_fields` into `cells.jsonl` and let a validity check ask
whether an injected marker behaves differently on essays actually written by that group. Education is
the only domain where all three injected axes have a real counterpart. ASAP carries no demographics,
so all three are ``None`` there.
"""

from __future__ import annotations

import csv
import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# It also fires on the handful of PERSUADE essays containing a bare @-handle ("@TimmyTurner", "@dot"):
# that is wanted for a name, coarse for the rest, and the count is reported as `redacted_bodies`.
_REDACTION_RE = re.compile(r"(?:@\w+)+@?")


def _raise_field_size_limit() -> None:
    """Essays can exceed the csv default field size. Called by the loaders rather than at import, so
    importing this module does not change a process-global setting for unrelated readers."""
    csv.field_size_limit(10_000_000)


@dataclass
class EssayRecord:
    """One real essay. `essay_text` is the held-fixed body; `high_quality` is the quality label.

    The `raw_*` fields are the writer's REAL attributes from the corpus (PERSUADE only) — validity
    checks and stratified analysis only, **never rendered**. The sex/ethnicity/grade markers the model
    sees are injected downstream via `pairs/markers.py` and are independent of these.
    """

    source_record_id: str
    essay_text: str
    holistic_score: float
    high_quality: bool  # True = strong essay (top tier); the ground-truth quality label
    source_dataset: str  # "persuade" | "asap"
    prompt_id: Optional[str] = None
    # The task the essay answers, rendered into the header by `education_render.py` so the grader is
    # not judging an answer to an invisible question. Constant per prompt; None where the corpus has
    # no task text (ASAP), and the renderer then omits the block.
    assignment: Optional[str] = None
    raw_sex: Optional[str] = None        # REAL: "F" | "M"      (PERSUADE `gender`)
    raw_ethnicity: Optional[str] = None  # REAL: "White", "Black/African American", …
    raw_grade_level: Optional[str] = None  # REAL: school grade "6".."12" — NOT the ASAP essay set
    extra: Dict[str, object] = field(default_factory=dict)


_PARAGRAPH_SPLIT_RE = re.compile(r"\n[^\S\n]*\n\s*")


def _clean(text: str) -> str:
    """Neutralize redaction tags and collapse whitespace, **keeping paragraph breaks**.

    15,004 of PERSUADE's 15,594 essays are paragraphed (median 4 blank-line breaks). Flattening them
    into one line, as this used to, removes structure the grader can legitimately read — identically
    on both sides of a pair, so it never biased a contrast, but it degraded every essay's apparent
    quality. A blank line survives as ``\\n\\n``; every other whitespace run inside a paragraph
    becomes a single space, so a lone newline from a hard-wrapped corpus does not become a false
    paragraph break (PERSUADE has none).
    """
    text = _REDACTION_RE.sub("someone", text).replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = (re.sub(r"\s+", " ", p).strip() for p in _PARAGRAPH_SPLIT_RE.split(text))
    return "\n\n".join(p for p in paragraphs if p)


def _missing(path: Path, corpus: str, how: str) -> None:
    # `data/` is gitignored, so on a fresh clone the raw directory does not exist and any
    # suggested download command would fail on the missing parent rather than on the download.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raise FileNotFoundError(
        f"{corpus} corpus not found at {path}. It is user-downloaded (not committed).\n  {how}\n"
        f"  (the directory {Path(path).parent} has just been created for you)"
    )


def _find_col(fieldnames: List[str], candidates: List[str], corpus: str, what: str,
              *, optional: bool = False) -> Optional[str]:
    """First *candidate* (in the given priority order) that the file provides, else raise.

    The priority order is the point: it must not degrade to the column order in the file. PERSUADE
    lists `essay_id` before `essay_id_comp`, but 661 of its `essay_id` values are Excel-mangled to
    scientific notation (`5.88194E+12`) and one such value covers two different essays — so a
    file-order lookup would silently merge them. `optional=True` returns None instead of raising, for
    metadata columns a corpus may simply not have.
    """
    lut = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lut:
            return lut[cand.lower()]
    if optional:
        return None
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
    id_col: Optional[str] = None,
    report: Optional[Dict[str, object]] = None,
) -> List[EssayRecord]:
    """Load PERSUADE 2.0 essays with a strong/weak `high_quality` label from the holistic score.

    Column names are auto-detected (with sensible defaults) and can be overridden. Essays outside
    `[min_chars, max_chars]` and middle-score essays are dropped. Deterministic order/sample by `seed`.
    If `report` is a dict it is filled with the row/essay counts and the per-reason drop counts, the
    way `bios_clean`/`credit_clean` report theirs, so the generator can log them into the manifest.
    """
    path = Path(path)
    if not path.exists():
        _missing(path, "PERSUADE 2.0",
                 "Download from https://github.com/scrosseye/persuade_corpus_2.0 (CC BY-NC-SA 4.0) "
                 f"and place the CSV at {DEFAULT_PERSUADE_PATH}.")
    _raise_field_size_limit()
    records: List[EssayRecord] = []
    dropped = {"duplicate_row": 0, "too_short": 0, "too_long": 0, "unparsable_score": 0,
               "middle_score": 0}
    n_rows = 0
    redacted_bodies = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fn = reader.fieldnames or []
        tcol = text_col or _find_col(fn, ["full_text", "essay", "text"], "PERSUADE", "essay-text")
        scol = score_col or _find_col(fn, ["holistic_essay_score", "score", "holistic_score"],
                                      "PERSUADE", "holistic-score")
        # `essay_id_comp` first: see `_find_col` — `essay_id` is lossy in the published CSV.
        idcol = id_col or _find_col(fn, ["essay_id_comp", "essay_id", "id"], "PERSUADE", "essay-id",
                                    optional=True)
        gcol = _find_col(fn, ["grade_level", "grade"], "PERSUADE", "grade-level", optional=True)
        pcol = _find_col(fn, ["prompt_name", "prompt"], "PERSUADE", "prompt", optional=True)
        acol = _find_col(fn, ["assignment", "task_text"], "PERSUADE", "assignment", optional=True)
        sexcol = _find_col(fn, ["gender", "sex"], "PERSUADE", "writer-sex", optional=True)
        ethcol = _find_col(fn, ["race_ethnicity", "ethnicity"], "PERSUADE", "writer-ethnicity",
                           optional=True)
        # PERSUADE 2.0 is discourse-element-level (~11 rows/essay) → dedup by essay id. Without an id
        # column (or for a blank id) dedup on the cleaned body instead: a per-row fallback key would
        # silently turn every discourse row into its own copy of the essay.
        seen: set = set()
        for i, row in enumerate(reader):
            n_rows += 1
            raw = row.get(tcol, "") or ""
            eid = (row.get(idcol) or "").strip() if idcol else ""
            # Only the id-less fallback needs the body up front; cleaning every discourse row would
            # do ~11x the work for nothing.
            body = None if eid else _clean(raw)
            key = ("id", eid) if eid else ("text", body)
            if key in seen:
                dropped["duplicate_row"] += 1
                continue
            seen.add(key)
            if body is None:
                body = _clean(raw)
            eid = eid or f"row{i}"
            if len(body) < min_chars:
                dropped["too_short"] += 1
                continue
            if len(body) > max_chars:
                dropped["too_long"] += 1
                continue
            try:
                score = float(row.get(scol, ""))
            except (TypeError, ValueError):
                dropped["unparsable_score"] += 1
                continue
            if score >= PERSUADE_STRONG_MIN:
                hq = True
            elif score <= PERSUADE_WEAK_MAX:
                hq = False
            else:
                dropped["middle_score"] += 1  # drop the middle for a clean strong/weak contrast
                continue
            if _REDACTION_RE.search(raw):
                redacted_bodies += 1
            records.append(EssayRecord(
                source_record_id=f"persuade-{eid}",
                essay_text=body, holistic_score=score, high_quality=hq, source_dataset="persuade",
                prompt_id=_get(row, pcol),
                assignment=(_clean(row.get(acol, "")) or None) if acol else None,
                raw_sex=_get(row, sexcol),
                raw_ethnicity=_get(row, ethcol),
                raw_grade_level=_get(row, gcol),
                extra=_persuade_extra(row, fn),
            ))
    if report is not None:
        report.update({"corpus": "persuade", "n_rows": n_rows, "n_essays": n_rows - dropped["duplicate_row"],
                       "dropped": dropped, "kept": len(records),
                       "strong": sum(r.high_quality for r in records),
                       "redacted_bodies": redacted_bodies})
    return _finalize(records, n, seed, report)


def _get(row: Dict[str, str], col: Optional[str]) -> Optional[str]:
    """Column value, with blanks (PERSUADE leaves some demographics empty) normalised to None."""
    if not col:
        return None
    val = (row.get(col) or "").strip()
    return val or None


_PERSUADE_EXTRA_COLS = ("ell_status", "economically_disadvantaged", "student_disability_status")


def _persuade_extra(row: Dict[str, str], fieldnames: List[str]) -> Dict[str, object]:
    """The remaining real writer attributes. Same rule as the `raw_*` fields: never rendered."""
    lut = {c.lower(): c for c in fieldnames}
    out: Dict[str, object] = {}
    for name in _PERSUADE_EXTRA_COLS:
        val = _get(row, lut.get(name))
        if val:
            out[name] = val
    return out


def load_asap(
    path: str | Path = DEFAULT_ASAP_PATH,
    *,
    n: Optional[int] = None,
    seed: int = 42,
    min_chars: int = 300,
    max_chars: int = 6000,
    report: Optional[Dict[str, object]] = None,
) -> List[EssayRecord]:
    """Load ASAP-AES essays. Scores differ per essay-set, so we normalize within each set and take the
    top/bottom terciles as strong/weak (dropping the middle). ASAP is a latin-1 TSV with @-redactions.

    ASAP ships no writer demographics, so `raw_sex`/`raw_ethnicity`/`raw_grade_level` stay None — in
    particular the essay set is *not* a grade level; it is kept as `prompt_id` and `extra["essay_set"]`.
    """
    path = Path(path)
    if not path.exists():
        _missing(path, "ASAP-AES",
                 "Download training_set_rel3.tsv from the Kaggle 2012 ASAP competition and place it at "
                 f"{DEFAULT_ASAP_PATH}.")
    _raise_field_size_limit()
    dropped = {"unparsable_score": 0, "too_short": 0, "too_long": 0, "degenerate_set": 0,
               "middle_tercile": 0}
    # Pass 1: gather (id, set, text, score) + per-set score ranges. Kept in a tuple rather than
    # written back into the reader's row dict, which would both lie about its type and risk colliding
    # with a real column of the same name.
    raw: List[Tuple[str, str, str, float]] = []
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
                dropped["unparsable_score"] += 1
                continue
            eset = row.get(setcol, "?")
            raw.append((row.get(idcol, ""), eset, row.get(tcol, ""), sc))
            set_scores.setdefault(eset, []).append(sc)
    lo = {s: _nearest_rank(v, ASAP_WEAK_Q) for s, v in set_scores.items()}
    hi = {s: _nearest_rank(v, ASAP_STRONG_Q) for s, v in set_scores.items()}
    records: List[EssayRecord] = []
    for eid, eset, text, sc in raw:
        body = _clean(text)
        if len(body) < min_chars:
            dropped["too_short"] += 1
            continue
        if len(body) > max_chars:
            dropped["too_long"] += 1
            continue
        if hi[eset] <= lo[eset]:
            dropped["degenerate_set"] += 1  # no strong/weak contrast exists, so label none of it
            continue
        if sc >= hi[eset]:
            hq = True
        elif sc <= lo[eset]:
            hq = False
        else:
            dropped["middle_tercile"] += 1
            continue
        records.append(EssayRecord(
            source_record_id=f"asap-{eset}-{eid}",
            essay_text=body, holistic_score=sc, high_quality=hq, source_dataset="asap",
            prompt_id=f"set{eset}",
            extra={"essay_set": eset, "set_cutoffs": (lo[eset], hi[eset])},
        ))
    if report is not None:
        report.update({"corpus": "asap", "n_rows": len(raw) + dropped["unparsable_score"],
                       "n_essays": len(raw), "dropped": dropped, "kept": len(records),
                       "strong": sum(r.high_quality for r in records),
                       "sets": sorted(set_scores)})
    return _finalize(records, n, seed, report)


def _nearest_rank(vals: List[float], q: float) -> float:
    """Nearest-rank cutoff: the value at position `q` of the sorted scores. Not an interpolated
    quantile — ASAP scores are coarse integers, so the tercile split is approximate by construction
    and a set whose two cutoffs coincide is dropped whole by the caller."""
    s = sorted(vals)
    return s[min(int(q * (len(s) - 1)), len(s) - 1)]


def _finalize(records: List[EssayRecord], n: Optional[int], seed: int,
              report: Optional[Dict[str, object]] = None) -> List[EssayRecord]:
    """Deterministic shuffle (+ optional truncate) so runs are reproducible from (n, seed)."""
    if not records:
        raise ValueError("No essays survived the length/score filters — check the corpus file.")
    rng = random.Random(seed)
    rng.shuffle(records)
    out = records[:n] if n else records
    if report is not None:
        report["returned"] = len(out)
    return out
