#!/usr/bin/env python3
"""
Generate the demographic hiring (CV-screening) matched-pair dataset from **real biographies**.

Pipeline:
  1. load Bias-in-Bios and scrub the body (strip the leading name, neutralise gendered pronouns and
     titles, replace contact details), dropping bios that still contain a gendered word or a sex-coded
     first name (`substrates/bios_ingest.py`)
  2. drop bios that cannot carry every level of the factorial plausibly — a stated age of 30 next to
     "25 years of experience" or a degree dated 2005 (`substrates/bios_clean.py`)
  3. cap every profession at 10% of the pool, then assign the role-match `qualified` label on the bios
     actually used: exactly half of each profession qualified, unqualified bios screened for another
     unqualified bio's profession, never a near-synonym (`substrates/bios_clean.py`)
  4. render each remaining bio in all 8 cells of the sex × age × family-status factorial (a neutral
     header naming the target role + one composite marker clause) and cut the matched pairs from them
     (`pairs/factorial.py`), per template and encoding
  5. Tier-1 structural gate on every pair; a bio/template/encoding block with any failing pair is
     dropped whole, so the factorial stays balanced
  6. write pairs.jsonl (+ manifest.json, spotcheck.csv) and cells.jsonl (all 8 texts per block, plus
     the corpus's real gender label for stratified analysis — never rendered)

The biography body is byte-identical across an A/B pair, so only the marker differs. Every drop is
counted in manifest.json's discard report; the train/eval split happens at load time.

The corpus is user-downloaded (not committed) into data/demographic/cv/raw/. Fetch it once with
`--from-hub`; note these are biographies of identifiable real people, so neither the raw corpus nor
the derived pairs are redistributed.

Usage:
    # one-time fetch of the HF mirror into data/demographic/cv/raw/ (MIT licence)
    python runners/generate_bios.py --from-hub

    python runners/generate_bios.py --encodings explicit,proxy --n-bios 4000 --seed 42 \
        --out-dir data/demographic/cv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from substrates.bios_clean import FACTORIAL_RULES, load_factorial_bios
from substrates.bios_ingest import DEFAULT_BIOS_PATH, fetch_from_hub
from substrates.bios_render import BIOS_TEMPLATES, render_bio
from pairs.factorial import HIRING_DESIGN, build_factorial_rows
from pairs.validate import Thresholds, validate_pair
from pairs.manifest import BIOS_ATTRIBUTION, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("gen-bios")

DEFAULT_AXES = HIRING_DESIGN.axes + ("intersection",)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-hub", action="store_true",
                    help="Download the HF mirror once and cache it as parquet, then continue.")
    ap.add_argument("--raw-path", default=None, help="Override the corpus file path (else the default).")
    ap.add_argument("--axes", default=",".join(DEFAULT_AXES),
                    help="Which pair types to write (the 8 cells are always rendered)")
    ap.add_argument("--encodings", default="explicit,proxy")
    ap.add_argument("--templates", default=",".join(sorted(BIOS_TEMPLATES)))
    ap.add_argument("--n-bios", type=int, default=4000,
                    help="Biographies to use (after the scrub and plausibility filters)")
    ap.add_argument("--n-per", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("data/demographic/cv"))
    ap.add_argument("--max-char-delta", type=int, default=12)
    ap.add_argument("--max-token-delta", type=int, default=3)
    ap.add_argument("--max-flesch-delta", type=float, default=8.0)
    # The composed 3-way intersection clause is inherently longer than a marginal one, so the gate
    # gets a relaxed char bound for that axis only.
    ap.add_argument("--intersection-char-delta", type=int, default=40)
    args = ap.parse_args()
    if args.n_per is not None:
        ap.error("--n-per was replaced by --n-bios: the factorial design emits a fixed set of pairs "
                 "per biography")

    raw_path = args.raw_path or DEFAULT_BIOS_PATH
    if args.from_hub:
        logger.info("Fetching the Bias-in-Bios HF mirror -> %s (one-time)", raw_path)
        fetch_from_hub(raw_path)

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    templates = [t.strip() for t in args.templates.split(",") if t.strip()]
    base = Thresholds(args.max_char_delta, args.max_token_delta, args.max_flesch_delta)
    wide = Thresholds(args.intersection_char_delta, args.max_token_delta, args.max_flesch_delta)

    # raises FileNotFoundError with fetch instructions if absent
    corpus_report: Dict[str, Any] = {}
    records = load_factorial_bios(raw_path, n=args.n_bios, seed=args.seed, report=corpus_report)
    rules_report = corpus_report.pop("factorial_rules")
    cap_report = corpus_report.pop("profession_cap")
    role_report = corpus_report.pop("role_leak")
    mix = corpus_report.pop("profession_mix")
    n_strong = sum(r.qualified for r in records)
    logger.info("Corpus filters: kept %d of %d bios; dropped %s; %.1f%% of kept mention women/men",
                corpus_report["kept"], corpus_report["n_rows"], corpus_report["dropped"],
                100 * corpus_report["kept_mentioning_women_or_men_rate"])
    logger.info("Factorial rules: %d -> %d %s", rules_report["n_in"], rules_report["n_out"],
                rules_report["dropped_by_rule"])
    logger.info("Profession cap %s: limit %s per profession, %d -> %d bios; capped %s", cap_report["cap"],
                cap_report.get("limit"), cap_report.get("n_in", 0), cap_report.get("n_out", 0),
                sorted(cap_report.get("capped", {})))
    logger.info("Role label on the bios used: %s (0.5 = the role name says nothing about qualified)",
                role_report)
    by_keep = sorted(mix, key=lambda p: mix[p]["keep_rate"])
    by_shift = sorted(mix, key=lambda p: mix[p]["share_used"] - mix[p]["share_loaded"])
    logger.info("Age rules keep %s of bios by profession (lowest %s, highest %s); largest share shifts: %s",
                "/".join(f"{mix[p]['keep_rate']:.0%}" for p in (by_keep[0], by_keep[-1])), by_keep[0],
                by_keep[-1], ", ".join(f"{p} {mix[p]['share_loaded']:.1%}->{mix[p]['share_used']:.1%}"
                                       for p in (by_shift[0], by_shift[1], by_shift[-2], by_shift[-1])))
    logger.info("Using %d biographies (%d role-matched / %d mismatched); templates=%s",
                len(records), n_strong, len(records) - n_strong, templates)

    pair_rows, cell_rows, gate = build_factorial_rows(
        records,
        design=HIRING_DESIGN,
        render_fn=render_bio,
        id_prefix="bios",
        domain="cv",
        # `role` (the target role with its article, as the header renders it) is read by the
        # cross-marker decision prompt ("You are screening a candidate for {role}.").
        real_fields=lambda r: {"gender": r.gender, "profession": r.profession,
                               "target_role": r.target_role, "role": r.role, "qualified": r.qualified},
        axes=axes,
        encodings=encodings,
        templates=templates,
        seed=args.seed,
        validate=lambda pair: validate_pair(pair, wide if pair.axis == "intersection" else base),
        content_label="bio_content",
    )
    for enc, g in gate.items():
        logger.info("Gate %s: kept %d blocks, dropped %d %s", enc, g["blocks_kept"],
                    g["blocks_dropped"], g["failure_reasons"])

    discards: Dict[str, Any] = {"corpus_filters": corpus_report, "factorial_rules": rules_report,
                                "profession_cap": cap_report, "role_leak": role_report,
                                "profession_mix": mix,
                                "n_records_used": len(records), "gate": gate}
    paths = write_manifest(
        out_dir=args.out_dir, records=pair_rows, seed=args.seed, discard_report=discards,
        thresholds={"max_char_delta": args.max_char_delta, "max_token_delta": args.max_token_delta,
                    "max_flesch_delta": args.max_flesch_delta,
                    "intersection_char_delta": args.intersection_char_delta},
        domain="cv", attribution=BIOS_ATTRIBUTION,
    )
    cells_path = args.out_dir / "cells.jsonl"
    with open(cells_path, "w") as f:
        for row in cell_rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d pairs -> %s", len(pair_rows), paths["pairs"])
    logger.info("Wrote %d factorial blocks -> %s", len(cell_rows), cells_path)
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


if __name__ == "__main__":
    main()
