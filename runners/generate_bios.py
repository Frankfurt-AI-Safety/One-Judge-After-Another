#!/usr/bin/env python3
"""
Generate the demographic hiring (CV-screening) matched-pair dataset from **real biographies**.

Pipeline: load Bias-in-Bios -> scrub the body (strip leading name, neutralise gendered pronouns and
titles) -> assign a balanced role-match `qualified` label (profession == target role) -> render as a
candidate profile with a neutral header naming the target role -> inject a single demographic header
marker (explicit + proxy) -> Tier-1 structural validation gate -> write manifest. The biography body
is held byte-identical across an A/B pair, so only the marker differs.

This replaces the synthetic CV substrate for the `cv` domain. `generate_cv_data.py` is retained to
reproduce pre-2026-08 hiring results.

The corpus is user-downloaded (not committed) into data/demographic/cv/raw/. Fetch it once with
`--from-hub`; note these are biographies of identifiable real people, so neither the raw corpus nor
the derived pairs are redistributed.

Usage:
    # one-time fetch of the HF mirror into data/demographic/cv/raw/ (MIT licence)
    python experiments/generate_bios_data.py --from-hub

    python experiments/generate_bios_data.py \
        --axes sex,age,family_status,intersection --encodings explicit,proxy \
        --n-per 500 --seed 42 --out-dir data/demographic/cv
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from substrates.bios_ingest import (
    DEFAULT_BIOS_PATH, fetch_from_hub, load_bias_in_bios,
)
from substrates.bios_render import BIOS_TEMPLATES, render_bio
from pairs.markers import make_pair
from pairs.validate import Thresholds, validate_pair
from pairs.manifest import BIOS_ATTRIBUTION, pair_to_record, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("gen-bios")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-hub", action="store_true",
                    help="Download the HF mirror once and cache it as parquet, then continue.")
    ap.add_argument("--raw-path", default=None, help="Override the corpus file path (else the default).")
    ap.add_argument("--axes", default="sex,age,family_status,intersection")
    ap.add_argument("--encodings", default="explicit,proxy")
    ap.add_argument("--templates", default=",".join(sorted(BIOS_TEMPLATES)))
    ap.add_argument("--n-per", type=int, default=500, help="Target clean pairs per (axis, encoding)")
    ap.add_argument("--n-bios", type=int, default=4000, help="Biographies to sample from the corpus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("data/demographic/cv"))
    ap.add_argument("--max-char-delta", type=int, default=12)
    ap.add_argument("--max-token-delta", type=int, default=3)
    ap.add_argument("--max-flesch-delta", type=float, default=8.0)
    # The composed 3-way intersection clause is inherently longer than a marginal one; same relaxation
    # the synthetic CV generator uses so the gate does not reject the whole cell on length alone.
    ap.add_argument("--intersection-char-delta", type=int, default=40)
    args = ap.parse_args()

    raw_path = args.raw_path or DEFAULT_BIOS_PATH
    if args.from_hub:
        logger.info("Fetching the Bias-in-Bios HF mirror -> %s (one-time)", raw_path)
        fetch_from_hub(raw_path)

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    templates = [t.strip() for t in args.templates.split(",") if t.strip()]

    # raises FileNotFoundError with fetch instructions if absent
    records = load_bias_in_bios(raw_path, n=args.n_bios, seed=args.seed)
    n_strong = sum(r.qualified for r in records)
    logger.info("Loaded %d scrubbed biographies (%d role-matched / %d mismatched); templates=%s",
                len(records), n_strong, len(records) - n_strong, templates)

    out_records: List[Dict[str, Any]] = []
    discards: Dict[str, Any] = {}

    for axis in axes:
        # Per-axis thresholds: only the intersection cell gets the relaxed char bound.
        thr = Thresholds(
            args.intersection_char_delta if axis == "intersection" else args.max_char_delta,
            args.max_token_delta,
            args.max_flesch_delta,
        )
        for enc in encodings:
            rng = random.Random(hash((args.seed, axis, enc)) & 0xFFFFFFFF)
            combos = [(r, t) for r in records for t in templates]
            rng.shuffle(combos)
            kept, n_fail = 0, 0
            fail_reasons: Dict[str, int] = {}
            for rec, tid in combos:
                if kept >= args.n_per:
                    break
                pair = make_pair(rec, tid, axis, enc, rng,
                                 render_fn=render_bio, content_label="bio_content",
                                 subject="applicant")
                res = validate_pair(pair, thr)
                if not res.ok:
                    n_fail += 1
                    for rsn in res.reasons:
                        k = rsn.split(" (")[0].split(" >")[0]
                        fail_reasons[k] = fail_reasons.get(k, 0) + 1
                    continue
                item_id = f"bios-{axis}-{enc}-{tid}-{rec.source_record_id}"
                out_records.append(pair_to_record(pair, item_id, role="probe", seed=args.seed,
                                                  domain="cv"))
                kept += 1
            seen = kept + n_fail
            discards[f"{axis}/{enc}"] = {
                "kept": kept, "examined": seen,
                "discard_rate": round(n_fail / max(seen, 1), 4),
                "failure_reasons": fail_reasons,
            }
            logger.info("%s/%s: kept %d (examined %d, discard %.2f)",
                        axis, enc, kept, seen, n_fail / max(seen, 1))
            if kept < args.n_per:
                logger.warning("%s/%s: only %d/%d clean pairs (need more bios: raise --n-bios)",
                               axis, enc, kept, args.n_per)

    paths = write_manifest(
        out_dir=args.out_dir, records=out_records, seed=args.seed, discard_report=discards,
        thresholds={"max_char_delta": args.max_char_delta, "max_token_delta": args.max_token_delta,
                    "max_flesch_delta": args.max_flesch_delta,
                    "intersection_char_delta": args.intersection_char_delta},
        domain="cv", attribution=BIOS_ATTRIBUTION,
    )
    logger.info("Wrote %d records -> %s", len(out_records), paths["pairs"])
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


if __name__ == "__main__":
    main()
