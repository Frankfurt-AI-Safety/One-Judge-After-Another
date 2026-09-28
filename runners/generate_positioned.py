#!/usr/bin/env python3
"""
Generate the positioned-argument (A2) matched-pair dataset (education / standpoint-credibility).

Pipeline: load real argumentative essays (ASAP 2.0) from the **shared education pool**
(`substrates/education_clean.load_education_essays` — the same essays as the A1 factorial and stage designs,
so A1 and A2 are comparable) -> keep one **standpoint-fit group** (`--standpoint-fit`, decided 2026-09-27:
``plausible`` = the civic prompts where a standpoint is at least arguable, ``implausible`` = the control
group from prompts where it is clearly not, equally large; `pairs.positionality.STANDPOINT_FIT`) ->
inject a first-person positionality sentence whose claimed identity is the only thing that varies A<->B
(essay body held byte-identical) at a chosen position (conclusion [v1] / opening / middle / random) -> frame
it in the A1 submission header with the assignment (empty marker) -> Tier-1 structural validation gate (the
default bounds; the identity phrases differ by at most 8 characters, 1 word and 0.72 Flesch points over both
groups, all positions and the paraphrases, measured 2026-09-28) -> pairs.jsonl, spotcheck.csv and
manifest.json (`pairs/manifest.py`; it names the data file and the corpus file by SHA-256). The position is
stored in the manifest ``encoding`` field so the existing loader/runners select it via ``--encodings``.

Every axis uses the same essays; so does every position, except that ``middle`` and ``random`` drop the
essays without a sentence boundary to insert at (4 plausible, 2 implausible; counted as
``no_insertion_point``). One manifest per group.

Every pair row's ``real_fields`` carries the essay's `education_ingest.real_fields` plus its
``standpoint_fit`` group: the quality label, the prompt, and the writer's real sex, ethnicity, grade level,
ELL, economic and disability status (pupils' data), never rendered and kept for the validity checks (decided
2026-09-28); the datasets keep all but the quality label and the prompt out of every result.

The exported A2 numbers come from `runners/run_positioned_maineffect.py`, which builds its pairs itself; this
manifest feeds the battery (`configs/demographic_edupos_qwen06.yaml`).

Usage:
    python runners/generate_positioned.py --standpoint-fit plausible     # -> .../asap2_plausible
    python runners/generate_positioned.py --standpoint-fit implausible   # -> .../asap2_implausible
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from substrates.education_clean import load_education_essays, source_path
from substrates.education_ingest import real_fields
from substrates.education_render import EDU_TEMPLATES
from pairs.factorial import stable_rng
from pairs.positionality import (
    DEFAULT_HEADER_TEMPLATE, POSITIONED_AXES, POSITIONS, STANDPOINT_GROUPS, NoInsertionPoint, block_id_suffix,
    make_positioned_pairs, select_standpoint_essays, standpoint_fit,
)
from pairs.validate import add_threshold_args, tally_reasons, thresholds_from_args, validate_pair
from pairs.manifest import EDU_ATTRIBUTION, pair_to_record, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("gen-pos")

SOURCES = ("asap2",)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES, default="asap2")
    ap.add_argument("--raw-path", default=None)
    ap.add_argument("--axes", default=",".join(POSITIONED_AXES))
    ap.add_argument("--positions", default="conclusion",
                    help=f"Comma-separated; any of {POSITIONS}. Stored in the manifest 'encoding' field.")
    ap.add_argument("--paraphrase", choices=["off", "sample"], default="off",
                    help="off=base wording; sample=rng-picked paraphrase per pair (diversified).")
    ap.add_argument("--header-template", choices=sorted(EDU_TEMPLATES), default=DEFAULT_HEADER_TEMPLATE,
                    help="A1 submission shell the positioned essay is framed in (identical A<->B).")
    ap.add_argument("--standpoint-fit", choices=STANDPOINT_GROUPS, default="plausible",
                    help="Which essays: the prompts where a standpoint is plausible, or the control group")
    ap.add_argument("--n-essays", type=int, default=None,
                    help="Cap per group (default: every plausible essay, and as many implausible ones); "
                         "every axis uses the same essays (middle/random drop those without a sentence boundary)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None)
    add_threshold_args(ap)
    args = ap.parse_args()

    out_dir = args.out_dir or Path(f"data/demographic/education_positioned/{args.source}_{args.standpoint_fit}")
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]

    corpus_report: Dict[str, Any] = {}
    # raises FileNotFoundError with download instructions if absent
    pool = load_education_essays(args.raw_path, source=args.source, seed=args.seed, report=corpus_report)
    records = select_standpoint_essays(pool, args.standpoint_fit, args.seed, n=args.n_essays)
    by_prompt = dict(sorted(Counter(r.prompt_id for r in records).items()))
    logger.info("Standpoint fit %s: %d of %d pool essays, per prompt %s", args.standpoint_fit, len(records),
                len(pool), by_prompt)
    rules_report = corpus_report.pop("education_rules")
    logger.info("Corpus filters: %d rows -> %d essays -> kept %d; dropped %s",
                corpus_report["n_rows"], corpus_report["n_essays"], corpus_report["kept"],
                corpus_report["dropped"])
    logger.info("Education rules: %d -> %d %s", rules_report["n_in"], rules_report["n_out"],
                rules_report["dropped_by_rule"])
    logger.info("Using %d %s essays; axes=%s positions=%s header=%s", len(records), args.source, axes,
                positions, args.header_template)

    def fields(rec):  # the record's real fields (incl. prompt_id), plus its standpoint-fit group
        return {**real_fields(rec), "standpoint_fit": standpoint_fit(rec)}

    thr = thresholds_from_args(args)
    out_records: List[Dict[str, Any]] = []
    discards: Dict[str, Any] = {}

    # Every axis (and position) runs on the SAME essays, like the A1 factorial: one block per essay, holding
    # all of that essay's pairs for the axis (4 for a per-attribute axis, 1 otherwise). A block with any
    # pair failing the gate is dropped whole, so the four settings of the other attributes stay balanced.
    for axis in axes:
        for position in positions:
            kept_blocks, dropped_blocks, n_pairs = 0, 0, 0
            fail_reasons: Dict[str, int] = {}
            for rec in records:
                rng = stable_rng(args.seed, rec.source_record_id, axis, position)
                try:
                    pairs = make_positioned_pairs(rec, axis, position, rng,
                                                  variant="sample" if args.paraphrase == "sample" else None,
                                                  header_template=args.header_template)
                except NoInsertionPoint:  # middle/random in an essay with no sentence boundary
                    dropped_blocks += 1
                    fail_reasons["no_insertion_point"] = fail_reasons.get("no_insertion_point", 0) + 1
                    continue
                failures = [res for res in (validate_pair(p, thr) for p in pairs) if not res.ok]
                if failures:
                    dropped_blocks += 1
                    tally_reasons(failures, fail_reasons)
                    continue
                kept_blocks += 1
                for pair in pairs:
                    item_id = f"pos-{axis}-{position}-{rec.source_record_id}{block_id_suffix(pair)}"
                    out_records.append(pair_to_record(pair, item_id, seed=args.seed,
                                                       domain="education", real_fields=fields(rec)))
                    n_pairs += 1
            seen = kept_blocks + dropped_blocks
            discards[f"{axis}/{position}"] = {
                "blocks_kept": kept_blocks, "blocks_dropped": dropped_blocks, "pairs": n_pairs,
                "discard_rate": round(dropped_blocks / max(seen, 1), 4), "failure_reasons": fail_reasons,
            }
            logger.info("%s/%s: kept %d essays -> %d pairs (dropped %d essays)",
                        axis, position, kept_blocks, n_pairs, dropped_blocks)

    paths = write_manifest(
        out_dir, out_records, seed=args.seed,
        discard_report={"corpus_filters": corpus_report, "education_rules": rules_report,
                        "n_records_used": len(records), "standpoint_fit": args.standpoint_fit,
                        "essays_per_prompt": by_prompt, **discards},
        thresholds=dataclasses.asdict(thr),
        domain="education",
        sources={f"{args.source}.csv": source_path(args.source, args.raw_path)},
        attribution=f"{EDU_ATTRIBUTION} Positioned-argument (A2) arm; source corpus: {args.source}; "
                    f"standpoint-fit group: {args.standpoint_fit}.",
    )
    logger.info("Wrote %d records -> %s", len(out_records), paths["pairs"])
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


if __name__ == "__main__":
    main()
