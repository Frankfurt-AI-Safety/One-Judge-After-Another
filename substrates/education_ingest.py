"""
Real-essay substrate for the education (grading) arm (education domain).

Like the hiring arm (Bias-in-Bios), this arm *loads* essays from an established
corpus and holds the essay body fixed — the demographic marker is injected only as a header clause
downstream (see `substrates/education_render.py` / `pairs/markers.py`), so a matched A/B pair differs by exactly the marker.

Two corpora (locked with the user; both user-downloaded, gitignored under
`data/demographic/education/raw/`):
- **PERSUADE 2.0** (primary): ~25k argumentative essays, grades 6–12, holistic score 1–6.
  CC BY-NC-SA 4.0 → research/measurement use only; do **not** redistribute derived essays.
- **ASAP 2.0** (the second sample; since 2026-09-27): ~24k source-based argumentative essays, grades 6–10,
  holistic score 1–6 on PERSUADE's rubric, the same seven text-dependent prompts, with assignment text and
  writer demographics; CC BY 4.0. 7,726 of the 17,307 essays in its public training file are PERSUADE
  essays under other ids, so `load_asap2` drops them by text and keeps the ~9,600 new ones: **new writers
  on the same tasks**, not an independent corpus. It replaced ASAP-AES (Kaggle 2012), whose release had
  replaced every capitalised word, name and number with a tag (`@CAPS1`, `@PERSON1`, …) in 64% of its
  essays, more densely in strong ones, which no rendering could repair.

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
the only domain where all three injected axes have a real counterpart. ASAP 2.0 carries the same fields.
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
DEFAULT_ASAP2_PATH = "data/demographic/education/raw/ASAP_2_Final_github_train.csv"

# The holistic score is 1–6 in both corpora (one rubric); strong = top, weak = bottom, middle dropped for a
# clean contrast.
PERSUADE_STRONG_MIN = 5
PERSUADE_WEAK_MAX = 2

# @-handles to neutralize so injected names are the only cue. PERSUADE contains a handful of bare @-handles
# ("@TimmyTurner", "@dot"): wanted for a name, coarse for the rest, and the count is reported as
# `redacted_bodies`. (It was written for the tags of the old ASAP-AES corpus, `@PERSON1`, `@CAPS1`, …,
# replaced by ASAP 2.0 on 2026-09-27.)
_REDACTION_RE = re.compile(r"(?:@\w+)+@?")


def _raise_field_size_limit() -> None:
    """Essays can exceed the csv default field size. Called by the loaders rather than at import, so
    importing this module does not change a process-global setting for unrelated readers."""
    csv.field_size_limit(10_000_000)


@dataclass
class EssayRecord:
    """One real essay. `essay_text` is the held-fixed body; `high_quality` is the quality label.

    The `raw_*` fields are the writer's REAL attributes from the corpus — validity
    checks and stratified analysis only, **never rendered**. The sex/ethnicity/grade markers the model
    sees are injected downstream via `pairs/markers.py` and are independent of these.
    """

    source_record_id: str
    essay_text: str
    holistic_score: float
    high_quality: bool  # True = strong essay (top tier); the ground-truth quality label
    source_dataset: str  # "persuade" | "asap2"
    prompt_id: Optional[str] = None
    # The task the essay answers, rendered into the header by `education_render.py` so the grader is
    # not judging an answer to an invisible question. Constant per prompt; None where the corpus has
    # no task text, and the renderer then omits the block.
    assignment: Optional[str] = None
    raw_sex: Optional[str] = None        # REAL: "F" | "M"      (PERSUADE `gender`)
    raw_ethnicity: Optional[str] = None  # REAL: "White", "Black/African American", …
    raw_grade_level: Optional[str] = None  # REAL: school grade "6".."12"
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
    records = _load_holistic_csv(path, corpus="PERSUADE", source_dataset="persuade", min_chars=min_chars,
                                 max_chars=max_chars, text_col=text_col, score_col=score_col, id_col=id_col,
                                 report=report)
    return _finalize(records, n, seed, report)


def load_asap2(
    path: str | Path = DEFAULT_ASAP2_PATH,
    *,
    persuade_path: str | Path = DEFAULT_PERSUADE_PATH,
    n: Optional[int] = None,
    seed: int = 42,
    min_chars: int = 300,
    max_chars: int = 6000,
    report: Optional[Dict[str, object]] = None,
) -> List[EssayRecord]:
    """Load the ASAP 2.0 essays that are **not** in PERSUADE 2.0 (see the module docstring).

    ASAP 2.0 has PERSUADE's columns, rubric (holistic 1-6) and prompts, so it is parsed exactly like
    PERSUADE, with the same strong/weak cut-offs. 7,726 of the 17,307 essays in the public training file
    are PERSUADE essays under other ids; they are matched by text (`_overlap_key`) and dropped, counted as
    ``in_persuade``, so the second sample shares no essay with the first. That needs the PERSUADE file.
    """
    path, persuade_path = Path(path), Path(persuade_path)
    if not path.exists():
        _missing(path, "ASAP 2.0",
                 "Download ASAP_2_Final_github_train.zip from https://github.com/scrosseye/ASAP_2.0 "
                 f"(CC BY 4.0), unzip it and place the CSV at {DEFAULT_ASAP2_PATH}.")
    if not persuade_path.exists():
        _missing(persuade_path, "PERSUADE 2.0 (needed to drop ASAP 2.0's PERSUADE essays)",
                 f"Place the PERSUADE CSV at {DEFAULT_PERSUADE_PATH} (see load_persuade).")
    in_persuade = {_overlap_key(r.essay_text) for r in
                   _load_holistic_csv(persuade_path, corpus="PERSUADE", source_dataset="persuade",
                                      min_chars=0, max_chars=10 ** 9, keep_all_scores=True)}
    records = _load_holistic_csv(path, corpus="ASAP 2.0", source_dataset="asap2", min_chars=min_chars,
                                 max_chars=max_chars, exclude=in_persuade, report=report)
    return _finalize(records, n, seed, report)


def _overlap_key(body: str) -> str:
    """The first 200 letters of a cleaned body, lower-cased: identifies the same essay across the two
    corpora despite whitespace and punctuation differences (7,725 exact matches, 7,726 by this key)."""
    return re.sub(r"[^a-z]", "", body.lower())[:200]


def _load_holistic_csv(
    path: Path,
    *,
    corpus: str,
    source_dataset: str,
    min_chars: int,
    max_chars: int,
    text_col: Optional[str] = None,
    score_col: Optional[str] = None,
    id_col: Optional[str] = None,
    exclude: Optional[set] = None,
    keep_all_scores: bool = False,
    report: Optional[Dict[str, object]] = None,
) -> List[EssayRecord]:
    """The rows of a PERSUADE-format CSV (PERSUADE 2.0, ASAP 2.0) as records, unshuffled. ``exclude`` holds
    `_overlap_key`s to drop (``in_persuade``); ``keep_all_scores`` keeps the middle scores too (only for
    building that key set)."""
    _raise_field_size_limit()
    records: List[EssayRecord] = []
    dropped = {"duplicate_row": 0, "too_short": 0, "too_long": 0, "unparsable_score": 0,
               "middle_score": 0}
    if exclude is not None:
        dropped["in_persuade"] = 0
    n_rows = 0
    redacted_bodies = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fn = reader.fieldnames or []
        tcol = text_col or _find_col(fn, ["full_text", "essay", "text"], corpus, "essay-text")
        scol = score_col or _find_col(fn, ["holistic_essay_score", "score", "holistic_score"],
                                      corpus, "holistic-score")
        # `essay_id_comp` first: see `_find_col` — `essay_id` is lossy in the published PERSUADE CSV.
        idcol = id_col or _find_col(fn, ["essay_id_comp", "essay_id", "id"], corpus, "essay-id",
                                    optional=True)
        gcol = _find_col(fn, ["grade_level", "grade"], corpus, "grade-level", optional=True)
        pcol = _find_col(fn, ["prompt_name", "prompt"], corpus, "prompt", optional=True)
        acol = _find_col(fn, ["assignment", "task_text"], corpus, "assignment", optional=True)
        sexcol = _find_col(fn, ["gender", "sex"], corpus, "writer-sex", optional=True)
        ethcol = _find_col(fn, ["race_ethnicity", "ethnicity"], corpus, "writer-ethnicity", optional=True)
        # PERSUADE 2.0 is discourse-element-level (~11 rows/essay) → dedup by essay id. Without an id
        # column (or for a blank id) dedup on the cleaned body instead: a per-row fallback key would
        # silently turn every discourse row into its own copy of the essay. ASAP 2.0 has one row per essay.
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
            if exclude is not None and _overlap_key(body) in exclude:
                dropped["in_persuade"] += 1
                continue
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
            elif score <= PERSUADE_WEAK_MAX or keep_all_scores:
                hq = False
            else:
                dropped["middle_score"] += 1  # drop the middle for a clean strong/weak contrast
                continue
            if _REDACTION_RE.search(raw):
                redacted_bodies += 1
            records.append(EssayRecord(
                source_record_id=f"{source_dataset}-{eid}",
                essay_text=body, holistic_score=score, high_quality=hq, source_dataset=source_dataset,
                prompt_id=_get(row, pcol),
                assignment=(_clean(row.get(acol, "")) or None) if acol else None,
                raw_sex=_get(row, sexcol),
                raw_ethnicity=_get(row, ethcol),
                raw_grade_level=_get(row, gcol),
                extra=_persuade_extra(row, fn),
            ))
    if report is not None:
        report.update({"corpus": source_dataset, "n_rows": n_rows,
                       "n_essays": n_rows - dropped["duplicate_row"],
                       "dropped": dropped, "kept": len(records),
                       "strong": sum(r.high_quality for r in records),
                       "redacted_bodies": redacted_bodies})
    return records


def _get(row: Dict[str, str], col: Optional[str]) -> Optional[str]:
    """Column value, with blanks (PERSUADE leaves some demographics empty) normalised to None."""
    if not col:
        return None
    val = (row.get(col) or "").strip()
    return val or None


_PERSUADE_EXTRA_COLS = ("ell_status", "economically_disadvantaged", "student_disability_status")


def real_fields(record: EssayRecord) -> Dict[str, object]:
    """The writer's REAL attributes and the quality label, as the generators write them into
    cells.jsonl and onto every pair row. Never rendered: covariates for the validity checks (does an
    injected marker move the score differently on essays actually written by that group?) and the
    stratum of the probe_records split. One definition for the A1 factorial, the stage design and A2."""
    return {"sex": record.raw_sex, "ethnicity": record.raw_ethnicity,
            "grade_level": record.raw_grade_level, "high_quality": record.high_quality,
            **{k: record.extra.get(k) for k in _PERSUADE_EXTRA_COLS}}


def _persuade_extra(row: Dict[str, str], fieldnames: List[str]) -> Dict[str, object]:
    """The remaining real writer attributes. Same rule as the `raw_*` fields: never rendered."""
    lut = {c.lower(): c for c in fieldnames}
    out: Dict[str, object] = {}
    for name in _PERSUADE_EXTRA_COLS:
        val = _get(row, lut.get(name))
        if val:
            out[name] = val
    return out


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
