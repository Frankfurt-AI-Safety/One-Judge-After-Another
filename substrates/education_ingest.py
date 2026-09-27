"""
Real-essay substrate for the education (grading) arm (education domain).

Like the hiring arm (Bias-in-Bios), this arm *loads* essays from an established
corpus and holds the essay body fixed — the demographic marker is injected only as a header clause
downstream (see `substrates/education_render.py` / `pairs/markers.py`), so a matched A/B pair differs by exactly the marker.

One corpus (since 2026-09-27; user-downloaded, gitignored under `data/demographic/education/raw/`):
**ASAP 2.0** (Crossley et al., 2025; https://github.com/scrosseye/ASAP_2.0; CC BY 4.0): source-based
argumentative essays from US state writing tests, grades 6, 8, 9 and 10, one holistic score 1–6 on a single
rubric, seven text-dependent prompts, with the assignment text and the writers' demographics. Its public
training file has 17,307 essays. 7,726 of them were also published in PERSUADE 2.0, with the same score and
demographics in 99.9% of cases; ASAP 2.0 replaced both PERSUADE 2.0 (CC BY-NC-SA; its two extra prompts in the
pool were independent-writing tasks, and its grade 11–12 essays were all on prompts the pool excludes) and
ASAP-AES (Kaggle 2012), whose release had replaced every capitalised word, name and number with a tag in 64% of
its essays, more densely in strong ones. See the working notes of 2026-09-27.

The **assignment** (the task the essay answers) is kept and rendered into the header, the way the hiring
header names the target role: every prompt is text-dependent, so without it the model grades an answer to a
question it cannot see. It is constant per prompt and identical across an A/B pair.

`high_quality` is the education analog of CV `qualified` / credit `credit_good` — the quality ground
truth the cross-marker design's decision accuracy needs (a strong essay should pass, a weak one fail).
Essay quality is *what reward models natively score*, so this is the domain where that accuracy is most
likely to be interpretable on a small RM. We threshold the holistic score into a clean strong/weak
contrast and **drop the middle** so the label is unambiguous.

**Real demographics are kept on the record but never rendered.** ASAP 2.0 ships the writer's `gender`,
`race_ethnicity` and `grade_level` (and ELL / economic / disability status, in `extra`). They reach
`raw_sex`, `raw_ethnicity` and `raw_grade_level`, the same way credit keeps `raw_sex`/`raw_marital` and
Bias-in-Bios keeps `gender`: matched-pair injection is still the only thing the model ever sees, but
the real labels let the generator write `real_fields` into `cells.jsonl` and let a validity check ask
whether an injected marker behaves differently on essays actually written by that group. Education is
the only domain where all three injected axes have a real counterpart.
"""

from __future__ import annotations

import csv
import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Default local path (the user downloads the corpus here; gitignored via /data).
DEFAULT_ASAP2_PATH = "data/demographic/education/raw/ASAP_2_Final_github_train.csv"

# The holistic score is 1–6; strong = top, weak = bottom, middle dropped for a clean contrast.
STRONG_MIN = 5
WEAK_MAX = 2


def _raise_field_size_limit() -> None:
    """Essays can exceed the csv default field size. Called by the loader rather than at import, so
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
    source_dataset: str  # "asap2"
    prompt_id: Optional[str] = None
    # The task the essay answers, rendered into the header by `education_render.py` so the grader is
    # not judging an answer to an invisible question. Constant per prompt; None where a record has no
    # task text, and the renderer then omits the block.
    assignment: Optional[str] = None
    raw_sex: Optional[str] = None        # REAL: "F" | "M"      (`gender`)
    raw_ethnicity: Optional[str] = None  # REAL: "White", "Black/African American", …
    raw_grade_level: Optional[str] = None  # REAL: school grade "6", "8", "9", "10"
    extra: Dict[str, object] = field(default_factory=dict)


_PARAGRAPH_SPLIT_RE = re.compile(r"\n[^\S\n]*\n\s*")


def _clean(text: str) -> str:
    """Collapse whitespace, **keeping paragraph breaks**.

    16,375 of ASAP 2.0's 17,307 training essays are paragraphed. Flattening them into one line removes
    structure the grader can legitimately read — identically on both sides of a pair, so it never biases a
    contrast, but it degrades every essay's apparent quality. A blank line survives as ``\\n\\n``; every
    other whitespace run inside a paragraph becomes a single space, so a lone newline from a hard-wrapped
    text does not become a false paragraph break.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
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
    `optional=True` returns None instead of raising, for metadata columns a file may not have."""
    lut = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lut:
            return lut[cand.lower()]
    if optional:
        return None
    raise KeyError(
        f"{corpus}: could not find the {what} column (tried {candidates}). "
        f"Available columns: {fieldnames}."
    )


def load_asap2(
    path: str | Path = DEFAULT_ASAP2_PATH,
    *,
    n: Optional[int] = None,
    seed: int = 42,
    min_chars: int = 300,
    max_chars: int = 6000,
    report: Optional[Dict[str, object]] = None,
) -> List[EssayRecord]:
    """Load ASAP 2.0 essays with a strong/weak `high_quality` label from the holistic score.

    Essays outside `[min_chars, max_chars]` and middle-score essays are dropped; a repeated or blank id
    is dropped and counted too (the public file has none). Deterministic order/sample by `seed`. If
    `report` is a dict it is filled with the row counts and the per-reason drop counts, the way
    `bios_clean`/`credit_clean` report theirs, so the generator can log them into the manifest.
    """
    path = Path(path)
    if not path.exists():
        _missing(path, "ASAP 2.0",
                 "Download ASAP_2_Final_github_train.zip from https://github.com/scrosseye/ASAP_2.0 "
                 f"(CC BY 4.0), unzip it and place the CSV at {DEFAULT_ASAP2_PATH}.")
    _raise_field_size_limit()
    records: List[EssayRecord] = []
    dropped = {"missing_or_repeated_id": 0, "too_short": 0, "too_long": 0, "unparsable_score": 0,
               "middle_score": 0}
    n_rows = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fn = reader.fieldnames or []

        def col(candidates: List[str], what: str, optional: bool = False) -> Optional[str]:
            return _find_col(fn, candidates, "ASAP 2.0", what, optional=optional)

        tcol, scol, idcol = col(["full_text"], "essay-text"), col(["score"], "score"), col(["essay_id"], "id")
        pcol, acol = col(["prompt_name"], "prompt", True), col(["assignment"], "assignment", True)
        sexcol, ethcol = col(["gender"], "writer-sex", True), col(["race_ethnicity"], "writer-ethnicity", True)
        gcol = col(["grade_level"], "grade-level", True)
        seen: set = set()
        for row in reader:
            n_rows += 1
            eid = (row.get(idcol) or "").strip()
            if not eid or eid in seen:
                dropped["missing_or_repeated_id"] += 1
                continue
            seen.add(eid)
            body = _clean(row.get(tcol, "") or "")
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
            if score >= STRONG_MIN:
                hq = True
            elif score <= WEAK_MAX:
                hq = False
            else:
                dropped["middle_score"] += 1  # drop the middle for a clean strong/weak contrast
                continue
            records.append(EssayRecord(
                source_record_id=f"asap2-{eid}",
                essay_text=body, holistic_score=score, high_quality=hq, source_dataset="asap2",
                prompt_id=_get(row, pcol),
                assignment=(_clean(row.get(acol, "")) or None) if acol else None,
                raw_sex=_get(row, sexcol),
                raw_ethnicity=_get(row, ethcol),
                raw_grade_level=_get(row, gcol),
                extra=_extra(row, fn),
            ))
    if report is not None:
        report.update({"corpus": "asap2", "n_rows": n_rows,
                       "n_essays": n_rows - dropped["missing_or_repeated_id"], "dropped": dropped,
                       "kept": len(records),
                       "strong": sum(r.high_quality for r in records)})
    return _finalize(records, n, seed, report)


def _get(row: Dict[str, str], col: Optional[str]) -> Optional[str]:
    """Column value, with blanks (some demographics are empty) normalised to None."""
    if not col:
        return None
    val = (row.get(col) or "").strip()
    return val or None


_EXTRA_COLS = ("ell_status", "economically_disadvantaged", "student_disability_status")


def real_fields(record: EssayRecord) -> Dict[str, object]:
    """The writer's REAL attributes, the quality label and the prompt, as the generators write them into
    cells.jsonl and onto every pair row. Never rendered: covariates for the validity checks (does an
    injected marker move the score differently on essays actually written by that group?), the stratum of
    the probe_records split, and the prompt for the per-prompt breakdown. One definition for the A1
    factorial, the stage design and A2."""
    return {"sex": record.raw_sex, "ethnicity": record.raw_ethnicity,
            "grade_level": record.raw_grade_level, "high_quality": record.high_quality,
            "prompt_id": record.prompt_id,
            **{k: record.extra.get(k) for k in _EXTRA_COLS}}


def _extra(row: Dict[str, str], fieldnames: List[str]) -> Dict[str, object]:
    """The remaining real writer attributes. Same rule as the `raw_*` fields: never rendered."""
    lut = {c.lower(): c for c in fieldnames}
    out: Dict[str, object] = {}
    for name in _EXTRA_COLS:
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
