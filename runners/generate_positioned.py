#!/usr/bin/env python3
"""
Generate the positioned-argument (A2) matched-pair dataset (education / standpoint-credibility).

Pipeline: load real argumentative essays (PERSUADE 2.0 / ASAP-AES) from the **shared education pool**
(`substrates/education_clean.load_education_essays` — the same essays as the A1 factorial and stage designs,
so A1 and A2 are comparable) -> inject a first-person positionality sentence whose claimed identity is the
only thing that varies A<->B (essay body held byte-identical) at a chosen position (conclusion [v1] /
opening / middle / random) -> frame it in the A1 submission header with the assignment (empty marker) ->
Tier-1 structural validation gate (relaxed char/token bounds, since identity phrases legitimately differ a
little) -> manifest. The position is stored in the manifest ``encoding`` field so the existing
loader/runners select it via ``--encodings``.

Usage:
    python runners/generate_positioned.py --source persuade \
        --axes pos_sex,pos_race,pos_class,pos_origin,pos_intersection,pos_control \
        --positions conclusion --n-essays 500 \
        --out-dir data/demographic/education_positioned/persuade
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from substrates.education_clean import load_education_essays
from substrates.education_render import EDU_TEMPLATES
from pairs.factorial import stable_rng
from pairs.positionality import (
    DEFAULT_HEADER_TEMPLATE, POSITIONED_AXES, POSITIONS, block_id_suffix, make_positioned_pairs,
)
from pairs.validate import Thresholds, validate_pair
from pairs.manifest import EDU_ATTRIBUTION, pair_to_record, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("gen-pos")

SOURCES = ("persuade", "asap")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES, default="persuade")
    ap.add_argument("--raw-path", default=None)
    ap.add_argument("--axes", default=",".join(POSITIONED_AXES))
    ap.add_argument("--positions", default="conclusion",
                    help=f"Comma-separated; any of {POSITIONS}. Stored in the manifest 'encoding' field.")
    ap.add_argument("--paraphrase", choices=["off", "sample"], default="off",
                    help="off=base wording; sample=rng-picked paraphrase per pair (diversified).")
    ap.add_argument("--header-template", choices=sorted(EDU_TEMPLATES), default=DEFAULT_HEADER_TEMPLATE,
                    help="A1 submission shell the positioned essay is framed in (identical A<->B).")
    ap.add_argument("--n-per", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--n-essays", type=int, default=500,
                    help="Essays from the shared pool; every axis and position uses the same ones")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None)
    # Relaxed bounds: identity phrases differ a little; the single-slot strip is the real guarantee.
    ap.add_argument("--max-char-delta", type=int, default=20)
    ap.add_argument("--max-token-delta", type=int, default=5)
    ap.add_argument("--max-flesch-delta", type=float, default=12.0)
    args = ap.parse_args()
    if args.n_per is not None:
        ap.error("--n-per was replaced by --n-essays: every axis and position now uses the same essays "
                 "(as the A1 factorial does), and a factorial axis emits 4 pairs per essay")

    out_dir = args.out_dir or Path(f"data/demographic/education_positioned/{args.source}")
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]

    corpus_report: Dict[str, Any] = {}
    # raises FileNotFoundError with download instructions if absent
    records = load_education_essays(args.raw_path, source=args.source, n=args.n_essays, seed=args.seed,
                                    report=corpus_report)
    rules_report = corpus_report.pop("education_rules")
    logger.info("Corpus filters: %d rows -> %d essays -> kept %d; dropped %s",
                corpus_report["n_rows"], corpus_report["n_essays"], corpus_report["kept"],
                corpus_report["dropped"])
    logger.info("Education rules: %d -> %d %s", rules_report["n_in"], rules_report["n_out"],
                rules_report["dropped_by_rule"])
    logger.info("Using %d %s essays; axes=%s positions=%s header=%s", len(records), args.source, axes,
                positions, args.header_template)

    thr = Thresholds(args.max_char_delta, args.max_token_delta, args.max_flesch_delta)
    out_records: List[Dict[str, Any]] = []
    discards: Dict[str, Any] = {}

    # Every axis and position runs on the SAME essays, like the A1 factorial: one block per essay, holding
    # all of that essay's pairs for the axis (4 for a per-attribute axis, 1 otherwise). A block with any
    # pair failing the gate is dropped whole, so the four settings of the other attributes stay balanced.
    for axis in axes:
        for position in positions:
            kept_blocks, dropped_blocks, n_pairs = 0, 0, 0
            fail_reasons: Dict[str, int] = {}
            for rec in records:
                rng = stable_rng(args.seed, rec.source_record_id, axis, position)
                pairs = make_positioned_pairs(rec, axis, position, rng,
                                              variant="sample" if args.paraphrase == "sample" else None,
                                              header_template=args.header_template)
                failures = [res for res in (validate_pair(p, thr) for p in pairs) if not res.ok]
                if failures:
                    dropped_blocks += 1
                    for res in failures:
                        for rsn in res.reasons:
                            k = rsn.split(" (")[0].split(" >")[0]
                            fail_reasons[k] = fail_reasons.get(k, 0) + 1
                    continue
                kept_blocks += 1
                for pair in pairs:
                    item_id = f"pos-{axis}-{position}-{rec.source_record_id}{block_id_suffix(pair)}"
                    out_records.append(pair_to_record(pair, item_id, role="probe", seed=args.seed,
                                                       domain="education"))
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
                        "n_records_used": len(records), **discards},
        thresholds={"max_char_delta": args.max_char_delta, "max_token_delta": args.max_token_delta,
                    "max_flesch_delta": args.max_flesch_delta},
        domain="education",
        attribution=f"{EDU_ATTRIBUTION} Positioned-argument (A2) arm; source corpus: {args.source}.",
    )
    logger.info("Wrote %d records -> %s", len(out_records), paths["pairs"])
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


if __name__ == "__main__":
    main()
