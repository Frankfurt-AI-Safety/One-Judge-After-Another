#!/usr/bin/env python3
"""
Generate the demographic education (grading) matched-pair dataset (education arm).

Two designs, two manifests, ONE essay pool (`--design`). Both load the shared education pool
(`substrates/education_clean.load_education_essays`: stage-neutral prompts, no in-essay pupil cues, no
talk of the writer's own household money), which the A2 positioned arm uses too; the cross-marker
decision design reads the factorial's cells.jsonl.

**factorial** (default) — sex × ethnicity × economic status as a 2×2×2, all 8 cells rendered per
essay/template/encoding and the matched pairs cut from them (pairs/factorial.py). Sex and ethnicity
share one carrier in the proxy encoding (the first name), so the proxy is a Haim sex × ethnicity name
grid, one name per cell; the economic proxy is the school's free/reduced-price-lunch share.

**stage** — the single-axis 6th-grade-vs-doctoral-candidate contrast, plus `--include-ladder` for the
monotonicity rungs against the same reference clause. Every essay of the pool × every template is one block,
as in the factorial, so both designs rest on the same records (and the same probe split); a block yields one
pair per axis and encoding. (Until 2026-09-28 the stage drew 500 (essay, template) blocks: 462 of the 2,018
essays, 38 of them twice, strong share 0.488.)

Both: load real essays (ASAP 2.0; PERSUADE 2.0 until 2026-09-27) -> strong/weak `high_quality` from the
holistic score -> render as a gradable submission with a neutral header -> inject the marker clause ->
Tier-1 structural gate (a block with any failing pair is dropped whole) -> write pairs.jsonl (+ cells.jsonl
for the factorial), spotcheck.csv and manifest.json (`pairs/manifest.py`; it names the data files and the
corpus file by SHA-256). Approach A1 (header proxies over real essays); the essay body is held
byte-identical, so only the marker differs.

Every pair row (and cells row) carries the essay's ``real_fields`` (`education_ingest.real_fields`), never
rendered: ``high_quality``, the quality label the probe split stratifies on and the cross-marker design groups
by; ``prompt_id``, for the per-prompt breakdown; and the writer's real sex, ethnicity, grade level, ELL,
economic and disability status — pupils' data — kept for the validity checks (does a marker move the score
differently on essays that group actually wrote?; decided 2026-09-28). The datasets keep all but the quality
label and the prompt out of every result (`scoring/pair_dataset.py`).

The corpora are user-downloaded (not committed) into data/demographic/education/raw/ — see the module
docstrings in substrates/education_ingest.py for the exact download instructions.

Usage:
    python runners/generate_education.py                            # factorial -> <source>/ (asap2)
    python runners/generate_education.py --design stage --include-ladder   # -> <source>_stage/
    python runners/run_battery.py --config configs/<edu cfg> \
        --dataset-source data/demographic/education/asap2_stage/pairs.jsonl \
        --axes grade_level,stage_grade12,stage_undergrad,stage_masters,stage_doctorate
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from substrates.education_clean import load_education_essays, source_path
from substrates.education_ingest import real_fields
from substrates.education_render import EDU_TEMPLATES, render_essay
from pairs.factorial import EDUCATION_DESIGN, build_factorial_rows, stable_rng
from pairs.markers import STAGE_LADDER_AXES, make_pair
from pairs.validate import add_threshold_args, tally_reasons, thresholds_from_args, validate_pair
from pairs.manifest import EDU_ATTRIBUTION, pair_to_record, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("gen-edu")

SOURCES = ("asap2",)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", choices=("factorial", "stage"), default="factorial",
                    help="factorial: sex x ethnicity x economic status, all 8 cells per essay (the "
                         "domain's main manifest). stage: the single-axis 6th-grade-vs-doctorate "
                         "contrast (+ --include-ladder), on its own manifest. Same essay pool.")
    ap.add_argument("--source", choices=SOURCES, default="asap2")
    ap.add_argument("--raw-path", default=None, help="Override the corpus file path (else the default).")
    ap.add_argument("--axes", default=None,
                    help="Default: the factorial's axes + intersection, or grade_level for --design stage.")
    ap.add_argument("--include-ladder", action="store_true",
                    help=f"--design stage only: also emit the monotonicity rungs "
                         f"({','.join(STAGE_LADDER_AXES)}), each against the same 6th-grade reference.")
    ap.add_argument("--encodings", default="explicit,proxy")
    ap.add_argument("--templates", default=",".join(sorted(EDU_TEMPLATES)))
    ap.add_argument("--n-essays", type=int, default=None,
                    help="Cap on essays (default: the whole shared pool, the essays every education design uses)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: data/demographic/education/<source>[_stage]")
    add_threshold_args(ap)
    args = ap.parse_args()

    if args.design == "factorial":
        if args.include_ladder:
            ap.error("--include-ladder applies to --design stage only")
        run_factorial(args)
    else:
        run_stage(args)


def run_stage(args) -> None:
    """The single-axis stage design: every (essay, template) block, one pair per axis and encoding from each
    (see `build_stage_rows`)."""
    out_dir = args.out_dir or Path(f"data/demographic/education/{args.source}_stage")
    axes = ([a.strip() for a in args.axes.split(",") if a.strip()] if args.axes else ["grade_level"])
    if args.include_ladder:
        axes += [a for a in STAGE_LADDER_AXES if a not in axes]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    templates = [t.strip() for t in args.templates.split(",") if t.strip()]

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

    thr = thresholds_from_args(args)
    out_records, discards = build_stage_rows(records, axes=axes, encodings=encodings,
                                             templates=templates, seed=args.seed,
                                             validate=lambda pair: validate_pair(pair, thr))
    logger.info("Stage: %d (essay, template) blocks shared by %d axis/encoding cells; dropped %d blocks %s",
                discards["blocks_kept"], len(axes) * len(encodings), discards["blocks_dropped"],
                discards["failure_reasons"])

    paths = write_manifest(
        out_dir, out_records, seed=args.seed,
        discard_report={"corpus_filters": corpus_report, "education_rules": rules_report,
                        "n_records_used": len(records), **discards},
        thresholds=dataclasses.asdict(thr),
        domain="education", attribution=f"{EDU_ATTRIBUTION} Source corpus: {args.source}.",
        sources={f"{args.source}.csv": source_path(args.source, args.raw_path)},
    )
    logger.info("Wrote %d records -> %s", len(out_records), paths["pairs"])
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


def build_stage_rows(records, *, axes, encodings, templates, seed, validate):
    """Stage pairs for every axis and encoding from the same (essay, template) blocks: every essay × every
    template, in pool order.

    The ladder's rungs (and the `grade_level`/`stage_doctorate` duplicate that checks them) are only
    comparable if they are measured on the same essays. They once drew a separate sample per axis and
    encoding (duplicate pair shared 139 of 452 essays; strong share 0.47-0.53 across rungs). Each block yields
    one pair per (axis, encoding), and a block with any pair failing `validate` is dropped for all of them.
    """
    rows: List[Dict[str, Any]] = []
    kept, dropped = 0, 0
    fail_reasons: Dict[str, int] = {}
    for rec, tid in ((r, t) for r in records for t in templates):
        block = [(axis, enc, make_pair(rec, tid, axis, enc, stable_rng(seed, rec.source_record_id, tid),
                                       render_fn=render_essay, content_label="essay_content",
                                       subject="student"))
                 for axis in axes for enc in encodings]
        failures = [res for res in (validate(pair) for _, _, pair in block) if not res.ok]
        if failures:
            dropped += 1
            tally_reasons(failures, fail_reasons)
            continue
        kept += 1
        for axis, enc, pair in block:
            item_id = f"edu-{axis}-{enc}-{tid}-{rec.source_record_id}"
            rows.append(pair_to_record(pair, item_id, seed=seed, domain="education",
                                       real_fields=real_fields(rec)))
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

    thr = thresholds_from_args(args)
    pair_rows, cell_rows, gate = build_factorial_rows(
        records,
        design=EDUCATION_DESIGN,
        render_fn=render_essay,
        id_prefix="edu",
        domain="education",
        # The writer's REAL attributes, never rendered (see `education_ingest.real_fields`).
        real_fields=real_fields,
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
        thresholds=dataclasses.asdict(thr),
        domain="education", attribution=f"{EDU_ATTRIBUTION} Source corpus: {args.source}.",
        cells=cell_rows, sources={f"{args.source}.csv": source_path(args.source, args.raw_path)},
    )
    logger.info("Wrote %d pairs -> %s", len(pair_rows), paths["pairs"])
    logger.info("Wrote %d factorial blocks -> %s", len(cell_rows), paths["cells"])
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


if __name__ == "__main__":
    main()
