#!/usr/bin/env python3
"""
Generate the positioned-argument (A2) matched-pair dataset (education / standpoint-credibility).

Pipeline: load real argumentative essays (PERSUADE 2.0 / ASAP-AES) -> inject a first-person positionality
sentence whose claimed identity is the only thing that varies A<->B (essay body held byte-identical) at a
chosen position (conclusion [v1] / opening / middle / random) -> Tier-1 structural validation gate (relaxed
char/token bounds, since identity phrases legitimately differ a little) -> manifest. The position is stored
in the manifest ``encoding`` field so the existing loader/runners select it via ``--encodings``.

Usage:
    python experiments/generate_positioned_data.py --source persuade \
        --axes pos_sex,pos_race,pos_class,pos_origin,pos_intersection,pos_control \
        --positions conclusion --n-per 500 --n-essays 4000 \
        --out-dir data/demographic/education_positioned/persuade
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

from substrates.education_ingest import load_asap, load_persuade
from pairs.positionality import IDENTITY_AXES, POSITIONS, make_positioned_pair
from pairs.validate import Thresholds, validate_pair
from pairs.manifest import EDU_ATTRIBUTION, pair_to_record, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("gen-pos")

_LOADERS = {"persuade": load_persuade, "asap": load_asap}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(_LOADERS), default="persuade")
    ap.add_argument("--raw-path", default=None)
    ap.add_argument("--axes", default=",".join(IDENTITY_AXES))
    ap.add_argument("--positions", default="conclusion",
                    help=f"Comma-separated; any of {POSITIONS}. Stored in the manifest 'encoding' field.")
    ap.add_argument("--paraphrase", choices=["off", "sample"], default="off",
                    help="off=base wording; sample=rng-picked paraphrase per pair (diversified).")
    ap.add_argument("--n-per", type=int, default=500, help="Target clean pairs per (axis, position)")
    ap.add_argument("--n-essays", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None)
    # Relaxed bounds: identity phrases differ a little; the single-slot strip is the real guarantee.
    ap.add_argument("--max-char-delta", type=int, default=20)
    ap.add_argument("--max-token-delta", type=int, default=5)
    ap.add_argument("--max-flesch-delta", type=float, default=12.0)
    args = ap.parse_args()

    out_dir = args.out_dir or Path(f"data/demographic/education_positioned/{args.source}")
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]

    load_kwargs: Dict[str, Any] = dict(n=args.n_essays, seed=args.seed)
    if args.raw_path:
        load_kwargs["path"] = args.raw_path
    records = _LOADERS[args.source](**load_kwargs)
    logger.info("Loaded %d %s essays; axes=%s positions=%s", len(records), args.source, axes, positions)

    thr = Thresholds(args.max_char_delta, args.max_token_delta, args.max_flesch_delta)
    out_records: List[Dict[str, Any]] = []
    discards: Dict[str, Any] = {}

    for axis in axes:
        for position in positions:
            rng = random.Random(hash((args.seed, axis, position)) & 0xFFFFFFFF)
            recs = list(records)
            rng.shuffle(recs)
            kept, n_fail = 0, 0
            fail_reasons: Dict[str, int] = {}
            for rec in recs:
                if kept >= args.n_per:
                    break
                pair = make_positioned_pair(rec, axis, position, rng,
                                            variant="sample" if args.paraphrase == "sample" else None)
                res = validate_pair(pair, thr)
                if not res.ok:
                    n_fail += 1
                    for rsn in res.reasons:
                        k = rsn.split(" (")[0].split(" >")[0]
                        fail_reasons[k] = fail_reasons.get(k, 0) + 1
                    continue
                item_id = f"pos-{axis}-{position}-{rec.source_record_id}"
                out_records.append(pair_to_record(pair, item_id, role="probe", seed=args.seed,
                                                   domain="education"))
                kept += 1
            seen = kept + n_fail
            discards[f"{axis}/{position}"] = {
                "kept": kept, "examined": seen,
                "discard_rate": round(n_fail / max(seen, 1), 4), "failure_reasons": fail_reasons,
            }
            logger.info("%s/%s: kept %d (examined %d, discard %.2f)",
                        axis, position, kept, seen, n_fail / max(seen, 1))
            if kept < args.n_per:
                logger.warning("%s/%s: only %d/%d clean pairs (raise --n-essays)", axis, position,
                               kept, args.n_per)

    paths = write_manifest(
        out_dir, out_records, seed=args.seed, discard_report=discards,
        thresholds={"max_char_delta": args.max_char_delta, "max_token_delta": args.max_token_delta,
                    "max_flesch_delta": args.max_flesch_delta},
        domain="education",
        attribution=f"{EDU_ATTRIBUTION} Positioned-argument (A2) arm; source corpus: {args.source}.",
    )
    logger.info("Wrote %d records -> %s", len(out_records), paths["pairs"])
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


if __name__ == "__main__":
    main()
