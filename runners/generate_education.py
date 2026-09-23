#!/usr/bin/env python3
"""
Generate the demographic education (grading) matched-pair dataset (education arm).

Two designs, two manifests, ONE essay pool (`--design`). Both load the shared education pool
(`substrates/education_clean.load_education_essays`: stage-neutral prompts, no in-essay pupil cues, no
talk of the writer's own household money), which the A2 positioned arm and cross-influence use too.

**factorial** (default) — sex × ethnicity × economic status as a 2×2×2, all 8 cells rendered per
essay/template/encoding and the matched pairs cut from them (pairs/factorial.py). Sex and ethnicity
share one carrier in the proxy encoding (the first name), so the proxy is the index-matched Haim name
grid; the economic proxy is the school's free/reduced-price-lunch share.

**stage** — the single-axis 6th-grade-vs-doctoral-candidate contrast, plus `--include-ladder` for the
monotonicity rungs against the same reference clause.

Both: load real essays (PERSUADE 2.0 or ASAP-AES) -> strong/weak `high_quality` from the holistic score
-> render as a gradable submission with a neutral header -> inject the marker clause -> Tier-1
structural gate -> write manifest. Approach A1 (header proxies over real essays); the essay body is
held byte-identical, so only the marker differs.

The corpora are user-downloaded (not committed) into data/demographic/education/raw/ — see the module
docstrings in substrates/education_ingest.py for the exact download instructions.

Usage:
    python runners/generate_education.py --source persuade          # factorial -> <source>/
    python runners/generate_education.py --design stage --include-ladder   # -> <source>_stage/
    python runners/run_battery.py --config configs/<edu cfg> \
        --dataset-source data/demographic/education/persuade_stage/pairs.jsonl \
        --axes grade_level,stage_grade12,stage_undergrad,stage_masters,stage_doctorate
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from substrates.education_clean import load_education_essays
from substrates.education_render import EDU_TEMPLATES, render_essay
from pairs.factorial import EDUCATION_DESIGN, build_factorial_rows, stable_rng
from pairs.markers import STAGE_LADDER_AXES, make_pair
from pairs.validate import Thresholds, validate_pair
from pairs.manifest import EDU_ATTRIBUTION, pair_to_record, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("gen-edu")

SOURCES = ("persuade", "asap")
# PERSUADE's remaining real writer attributes, carried into cells.jsonl as covariates (never rendered).
REAL_COVARIATES = ("ell_status", "economically_disadvantaged", "student_disability_status")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", choices=("factorial", "stage"), default="factorial",
                    help="factorial: sex x ethnicity x economic status, all 8 cells per essay (the "
                         "domain's main manifest). stage: the single-axis 6th-grade-vs-doctorate "
                         "contrast (+ --include-ladder), on its own manifest. Same essay pool.")
    ap.add_argument("--source", choices=SOURCES, default="persuade")
    ap.add_argument("--raw-path", default=None, help="Override the corpus file path (else the default).")
    ap.add_argument("--axes", default=None,
                    help="Default: the factorial's axes + intersection, or grade_level for --design stage.")
    ap.add_argument("--include-ladder", action="store_true",
                    help=f"--design stage only: also emit the monotonicity rungs "
                         f"({','.join(STAGE_LADDER_AXES)}), each against the same 6th-grade reference.")
    ap.add_argument("--encodings", default="explicit,proxy")
    ap.add_argument("--templates", default=",".join(sorted(EDU_TEMPLATES)))
    ap.add_argument("--n-per", type=int, default=None,
                    help="--design stage only: target clean pairs per (axis, encoding). Default 500.")
    ap.add_argument("--n-essays", type=int, default=1500, help="Essays to sample from the corpus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: data/demographic/education/<source>[_stage]")
    ap.add_argument("--max-char-delta", type=int, default=12)
    ap.add_argument("--max-token-delta", type=int, default=3)
    ap.add_argument("--max-flesch-delta", type=float, default=8.0)
    args = ap.parse_args()

    if args.design == "factorial":
        if args.n_per is not None:
            ap.error("--n-per applies to --design stage only: the factorial emits a fixed set of pairs "
                     "per essay")
        if args.include_ladder:
            ap.error("--include-ladder applies to --design stage only")
        run_factorial(args)
    else:
        run_stage(args)


def run_stage(args) -> None:
    """The single-axis stage design: `n_per` shared (essay, template) blocks, one pair per axis and
    encoding from each (see `build_stage_rows`)."""
    out_dir = args.out_dir or Path(f"data/demographic/education/{args.source}_stage")
    axes = ([a.strip() for a in args.axes.split(",") if a.strip()] if args.axes else ["grade_level"])
    if args.include_ladder:
        axes += [a for a in STAGE_LADDER_AXES if a not in axes]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    templates = [t.strip() for t in args.templates.split(",") if t.strip()]
    n_per = args.n_per if args.n_per is not None else 500

    corpus_report: Dict[str, Any] = {}
    # raises FileNotFoundError with download instructions if absent
    records = load_education_essays(args.raw_path, source=args.source, n=args.n_essays, seed=args.seed,
                                    report=corpus_report)
    rules_report = corpus_report.pop("education_rules")
    n_strong = sum(r.high_quality for r in records)
    logger.info("Corpus filters: %d rows -> %d essays -> kept %d; dropped %s",
                corpus_report["n_rows"], corpus_report["n_essays"], corpus_report["kept"],
                corpus_report["dropped"])
    logger.info("Education rules: %d -> %d %s", rules_report["n_in"], rules_report["n_out"],
                rules_report["dropped_by_rule"])
    logger.info("Loaded %d %s essays (%d strong / %d weak); templates=%s",
                len(records), args.source, n_strong, len(records) - n_strong, templates)

    thr = Thresholds(args.max_char_delta, args.max_token_delta, args.max_flesch_delta)
    out_records, discards = build_stage_rows(records, axes=axes, encodings=encodings,
                                             templates=templates, n_per=n_per, seed=args.seed,
                                             validate=lambda pair: validate_pair(pair, thr))
    logger.info("Stage sample: %d (essay, template) blocks shared by %d axis/encoding cells; "
                "dropped %d blocks %s", discards["blocks_kept"], len(axes) * len(encodings),
                discards["blocks_dropped"], discards["failure_reasons"])
    if discards["blocks_kept"] < n_per:
        logger.warning("only %d/%d clean blocks (need more essays: raise --n-essays)",
                       discards["blocks_kept"], n_per)

    paths = write_manifest(
        out_dir, out_records, seed=args.seed,
        discard_report={"corpus_filters": corpus_report, "education_rules": rules_report,
                        "n_records_used": len(records), **discards},
        thresholds={"max_char_delta": args.max_char_delta, "max_token_delta": args.max_token_delta,
                    "max_flesch_delta": args.max_flesch_delta},
        domain="education", attribution=f"{EDU_ATTRIBUTION} Source corpus: {args.source}.",
    )
    logger.info("Wrote %d records -> %s", len(out_records), paths["pairs"])
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


def build_stage_rows(records, *, axes, encodings, templates, n_per, seed, validate):
    """Stage pairs for every axis and encoding from ONE shared sample of (essay, template) blocks.

    The ladder's rungs (and the `grade_level`/`stage_doctorate` duplicate that checks them) are only
    comparable if they are measured on the same essays. They used to draw a separate sample per axis and
    encoding (duplicate pair shared 139 of 452 essays; strong share 0.47-0.53 across rungs). Now the
    blocks are shuffled once; each block yields one pair per (axis, encoding), and a block with any pair
    failing `validate` is dropped for all of them. The first `n_per` clean blocks are kept.
    """
    combos = [(r, t) for r in records for t in templates]
    stable_rng(seed, "stage").shuffle(combos)
    rows: List[Dict[str, Any]] = []
    kept, dropped = 0, 0
    fail_reasons: Dict[str, int] = {}
    for rec, tid in combos:
        if kept >= n_per:
            break
        block = [(axis, enc, make_pair(rec, tid, axis, enc, stable_rng(seed, rec.source_record_id, tid),
                                       render_fn=render_essay, content_label="essay_content",
                                       subject="student"))
                 for axis in axes for enc in encodings]
        failures = [res for res in (validate(pair) for _, _, pair in block) if not res.ok]
        if failures:
            dropped += 1
            for res in failures:
                for rsn in res.reasons:
                    k = rsn.split(" (")[0].split(" >")[0]
                    fail_reasons[k] = fail_reasons.get(k, 0) + 1
            continue
        kept += 1
        for axis, enc, pair in block:
            item_id = f"edu-{axis}-{enc}-{tid}-{rec.source_record_id}"
            rows.append(pair_to_record(pair, item_id, role="probe", seed=seed, domain="education"))
    return rows, {"blocks_kept": kept, "blocks_dropped": dropped, "failure_reasons": fail_reasons,
                  "pairs_per_cell": kept}


def run_factorial(args) -> None:
    """The sex × ethnicity × economic-status factorial: all 8 cells per essay/template/encoding."""
    out_dir = args.out_dir or Path(f"data/demographic/education/{args.source}")
    axes = ([a.strip() for a in args.axes.split(",") if a.strip()] if args.axes
            else list(EDUCATION_DESIGN.axes + ("intersection",)))
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    templates = [t.strip() for t in args.templates.split(",") if t.strip()]

    corpus_report: Dict[str, Any] = {}
    # raises FileNotFoundError with download instructions if absent
    records = load_education_essays(args.raw_path, source=args.source, n=args.n_essays,
                                    seed=args.seed, report=corpus_report)
    rules_report = corpus_report.pop("education_rules")
    n_strong = sum(r.high_quality for r in records)
    logger.info("Corpus filters: %d rows -> %d essays -> kept %d; dropped %s",
                corpus_report["n_rows"], corpus_report["n_essays"], corpus_report["kept"],
                corpus_report["dropped"])
    logger.info("Education rules: %d -> %d %s", rules_report["n_in"], rules_report["n_out"],
                rules_report["dropped_by_rule"])
    logger.info("Using %d %s essays (%d strong / %d weak); templates=%s",
                len(records), args.source, n_strong, len(records) - n_strong, templates)

    thr = Thresholds(args.max_char_delta, args.max_token_delta, args.max_flesch_delta)
    pair_rows, cell_rows, gate = build_factorial_rows(
        records,
        design=EDUCATION_DESIGN,
        render_fn=render_essay,
        id_prefix="edu",
        domain="education",
        # The writer's REAL attributes, never rendered: covariates for the validity checks (does an
        # injected marker move the score differently on essays actually written by that group?).
        real_fields=lambda r: {"sex": r.raw_sex, "ethnicity": r.raw_ethnicity,
                               "grade_level": r.raw_grade_level, "high_quality": r.high_quality,
                               **{k: r.extra.get(k) for k in REAL_COVARIATES}},
        axes=axes,
        encodings=encodings,
        templates=templates,
        seed=args.seed,
        validate=lambda pair: validate_pair(pair, thr),
        content_label="essay_content",
        subject="student",
    )
    for enc, g in gate.items():
        logger.info("Gate %s: kept %d blocks, dropped %d %s", enc, g["blocks_kept"],
                    g["blocks_dropped"], g["failure_reasons"])

    paths = write_manifest(
        out_dir=out_dir, records=pair_rows, seed=args.seed,
        discard_report={"corpus_filters": corpus_report, "education_rules": rules_report,
                        "n_records_used": len(records), "gate": gate},
        thresholds={"max_char_delta": args.max_char_delta, "max_token_delta": args.max_token_delta,
                    "max_flesch_delta": args.max_flesch_delta},
        domain="education", attribution=f"{EDU_ATTRIBUTION} Source corpus: {args.source}.",
    )
    cells_path = out_dir / "cells.jsonl"
    with open(cells_path, "w") as f:
        for row in cell_rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d pairs -> %s", len(pair_rows), paths["pairs"])
    logger.info("Wrote %d factorial blocks -> %s", len(cell_rows), cells_path)
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


if __name__ == "__main__":
    main()
