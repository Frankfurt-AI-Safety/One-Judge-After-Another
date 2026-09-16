#!/usr/bin/env python3
"""
Generate the demographic education (grading) matched-pair dataset (education arm).

Pipeline: load real essays (PERSUADE 2.0 or ASAP-AES) -> derive a strong/weak `high_quality` label
from the holistic score -> render as a gradable submission with a neutral header -> inject a single
demographic header marker (sex/ethnicity via name; grade_level via stage cue; explicit + proxy) ->
Tier-1 structural validation gate -> write manifest. Approach A1 (header proxies over real essays);
the essay body is held byte-identical, so only the marker differs.

The corpora are user-downloaded (not committed) into data/demographic/education/raw/ — see the module
docstrings in substrates/education_ingest.py for the exact download instructions.

Usage:
    python experiments/generate_edu_data.py --source persuade \
        --axes sex,ethnicity,grade_level --encodings explicit,proxy \
        --n-per 200 --seed 42 --out-dir data/demographic/education/persuade
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
from substrates.education_render import EDU_TEMPLATES, render_essay
from pairs.markers import make_pair
from pairs.validate import Thresholds, validate_pair
from pairs.manifest import EDU_ATTRIBUTION, pair_to_record, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("gen-edu")

_LOADERS = {"persuade": load_persuade, "asap": load_asap}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(_LOADERS), default="persuade")
    ap.add_argument("--raw-path", default=None, help="Override the corpus file path (else the default).")
    ap.add_argument("--axes", default="sex,ethnicity,grade_level")
    ap.add_argument("--encodings", default="explicit,proxy")
    ap.add_argument("--templates", default=",".join(sorted(EDU_TEMPLATES)))
    ap.add_argument("--n-per", type=int, default=500, help="Target clean pairs per (axis, encoding)")
    ap.add_argument("--n-essays", type=int, default=1500, help="Essays to sample from the corpus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: data/demographic/education/<source>")
    ap.add_argument("--max-char-delta", type=int, default=12)
    ap.add_argument("--max-token-delta", type=int, default=3)
    ap.add_argument("--max-flesch-delta", type=float, default=8.0)
    args = ap.parse_args()

    out_dir = args.out_dir or Path(f"data/demographic/education/{args.source}")
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    templates = [t.strip() for t in args.templates.split(",") if t.strip()]

    loader = _LOADERS[args.source]
    load_kwargs = dict(n=args.n_essays, seed=args.seed)
    if args.raw_path:
        load_kwargs["path"] = args.raw_path
    records = loader(**load_kwargs)  # raises FileNotFoundError with download instructions if absent
    n_strong = sum(r.high_quality for r in records)
    logger.info("Loaded %d %s essays (%d strong / %d weak); templates=%s",
                len(records), args.source, n_strong, len(records) - n_strong, templates)

    thr = Thresholds(args.max_char_delta, args.max_token_delta, args.max_flesch_delta)
    out_records: List[Dict[str, Any]] = []
    discards: Dict[str, Any] = {}

    for axis in axes:
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
                                 render_fn=render_essay, content_label="essay_content",
                                 subject="student")
                res = validate_pair(pair, thr)
                if not res.ok:
                    n_fail += 1
                    for rsn in res.reasons:
                        k = rsn.split(" (")[0].split(" >")[0]
                        fail_reasons[k] = fail_reasons.get(k, 0) + 1
                    continue
                item_id = f"edu-{axis}-{enc}-{tid}-{rec.source_record_id}"
                out_records.append(pair_to_record(pair, item_id, role="probe", seed=args.seed,
                                                   domain="education"))
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
                logger.warning("%s/%s: only %d/%d clean pairs (need more essays: raise --n-essays)",
                               axis, enc, kept, args.n_per)

    paths = write_manifest(
        out_dir, out_records, seed=args.seed, discard_report=discards,
        thresholds={"max_char_delta": args.max_char_delta, "max_token_delta": args.max_token_delta,
                    "max_flesch_delta": args.max_flesch_delta},
        domain="education", attribution=f"{EDU_ATTRIBUTION} Source corpus: {args.source}.",
    )
    logger.info("Wrote %d records -> %s", len(out_records), paths["pairs"])
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


if __name__ == "__main__":
    main()
