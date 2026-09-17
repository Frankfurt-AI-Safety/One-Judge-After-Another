#!/usr/bin/env python3
"""
Generate the demographic credit-bias matched-pair dataset (credit arm).

Pipeline:
  1. load German Credit (corrected codebook, `substrates/credit_ingest.py`)
  2. drop self-contradictory records            (`credit_clean.RECORD_RULES`)
  3. drop records that cannot carry every level  (`credit_clean.FACTORIAL_RULES`)
  4. render each remaining record in all 8 cells of the sex × age × marital-status factorial and cut
     the matched pairs from them (`pairs/factorial.py`), per template and encoding
  5. Tier-1 structural gate on every pair; a record/template/encoding block with any failing pair is
     dropped whole, so the factorial stays balanced
  6. write pairs.jsonl (+ manifest.json, spotcheck.csv) and cells.jsonl (all 8 texts per block, plus
     the record's real sex/marital/age fields for stratified analysis — never rendered)

Every drop is counted in manifest.json's discard report. The train/eval split happens at load time
and is grouped by record (`scoring/pair_dataset.py`).

Usage:
    python runners/generate_credit.py --encodings explicit,proxy --seed 42 \
        --out-dir data/demographic/credit
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from substrates.credit_clean import FACTORIAL_RULES, RECORD_RULES, apply_rules
from substrates.credit_ingest import DEFAULT_RAW_PATH, GermanCreditRecord, load_german_credit
from substrates.credit_render import TEMPLATES, render_profile
from pairs.factorial import AXES, CREDIT_DESIGN, build_factorial_rows
from pairs.validate import Thresholds, validate_pair
from pairs.manifest import write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("gen-credit")

DEFAULT_AXES = AXES + ("intersection",)


def build_dataset(
    records: Sequence[GermanCreditRecord],
    *,
    axes: Sequence[str] = DEFAULT_AXES,
    encodings: Sequence[str] = ("explicit", "proxy"),
    templates: Sequence[str] = tuple(sorted(TEMPLATES)),
    seed: int = 42,
    thr: Optional[Thresholds] = None,
    n_records: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Steps 2-5 above. Returns ``(pair_rows, cell_rows, discard_report)``."""
    thr = thr or Thresholds()
    clean, record_report = apply_rules(records, RECORD_RULES)
    eligible, factorial_report = apply_rules(clean, FACTORIAL_RULES)
    order = list(eligible)
    random.Random(seed).shuffle(order)
    if n_records is not None:
        order = order[:n_records]

    pair_rows, cell_rows, gate = build_factorial_rows(
        order,
        design=CREDIT_DESIGN,
        render_fn=render_profile,
        id_prefix="credit",
        domain="credit",
        real_fields=lambda r: {"sex": r.raw_sex, "marital": r.raw_marital, "age": r.raw_age_years,
                               "credit_good": r.credit_good},
        axes=axes,
        encodings=encodings,
        templates=templates,
        seed=seed,
        validate=lambda pair: validate_pair(pair, thr),
        content_label="financial_content",
    )
    report = {
        "record_rules": record_report,
        "factorial_rules": factorial_report,
        "n_records_used": len(order),
        "gate": gate,
    }
    return pair_rows, cell_rows, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", default=",".join(DEFAULT_AXES),
                    help="Which pair types to write (the 8 cells are always rendered)")
    ap.add_argument("--encodings", default="explicit,proxy")
    ap.add_argument("--templates", default=",".join(sorted(TEMPLATES)))
    ap.add_argument("--n-records", type=int, default=None,
                    help="Cap on records used (default: every record that passes the rules)")
    ap.add_argument("--n-per", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("data/demographic/credit"))
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    ap.add_argument("--max-char-delta", type=int, default=12)
    ap.add_argument("--max-token-delta", type=int, default=3)
    ap.add_argument("--max-flesch-delta", type=float, default=8.0)
    args = ap.parse_args()
    if args.n_per is not None:
        ap.error("--n-per was replaced by --n-records: the factorial design emits a fixed set of "
                 "pairs per record")

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    templates = [t.strip() for t in args.templates.split(",") if t.strip()]
    thr = Thresholds(args.max_char_delta, args.max_token_delta, args.max_flesch_delta)

    records = load_german_credit(args.raw)
    logger.info("Loaded %d German Credit records; templates=%s", len(records), templates)
    pair_rows, cell_rows, report = build_dataset(
        records, axes=axes, encodings=encodings, templates=templates, seed=args.seed, thr=thr,
        n_records=args.n_records,
    )
    rr, fr = report["record_rules"], report["factorial_rules"]
    logger.info("Record rules: %d -> %d %s", rr["n_in"], rr["n_out"], rr["dropped_by_rule"])
    logger.info("Factorial rules: %d -> %d %s", fr["n_in"], fr["n_out"], fr["dropped_by_rule"])
    for enc, g in report["gate"].items():
        logger.info("Gate %s: kept %d blocks, dropped %d %s", enc, g["blocks_kept"],
                    g["blocks_dropped"], g["failure_reasons"])
        if g["blocks_dropped"]:
            logger.warning("Gate dropped %d %s blocks; the factorial is balanced per block, not "
                           "per record, for these", g["blocks_dropped"], enc)

    paths = write_manifest(
        args.out_dir, pair_rows, seed=args.seed, discard_report=report,
        thresholds={"max_char_delta": thr.max_char_delta, "max_token_delta": thr.max_token_delta,
                    "max_flesch_delta": thr.max_flesch_delta},
    )
    cells_path = args.out_dir / "cells.jsonl"
    with open(cells_path, "w") as f:
        for row in cell_rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d pairs → %s", len(pair_rows), paths["pairs"])
    logger.info("Wrote %d factorial blocks → %s", len(cell_rows), cells_path)
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


if __name__ == "__main__":
    main()
