#!/usr/bin/env python3
"""
Did the Bias-in-Bios scrub actually remove the sex signal from the biography body?

This exists because **the Tier-1 validation gate cannot answer that question**. The gate only checks
that the two poles of a matched pair differ by exactly the injected clause; a leaked "she" sits on
*both* sides, so it cancels in the diff and passes silently. Yet residual sex signal in the body is
precisely what would confound the sex axis — the injected marker would no longer be the only cue.

So we ask the question empirically, against the dataset's own **real** gender label:

    extract activations for scrubbed bios -> train a linear AND a non-linear (MLP) probe to predict
    the real gender -> compare held-out accuracy against the majority-class chance rate.

  * both probes ≈ chance   -> the scrub held; the body carries no recoverable sex signal.
  * linear ≈ chance, MLP ≫ -> sex is still there, just not linearly; the sex axis is confounded.
  * both ≫ chance          -> the scrub failed outright; fix it before trusting any sex-axis number.

As a reference point the same probes are run on the **unscrubbed** bodies, which should be strongly
decodable — if they are not, the probe setup itself is broken and the scrubbed result means nothing.

This reuses `probe_recoverability` from probes/erasure.py (the same discriminator the
LEACE/TaCo arm uses) rather than introducing a second probe implementation.

Usage:
    python runners/validate_bios_scrub.py --config configs/demographic_cv_sex_qwen06.yaml \
        --probe-items 400 --eval-items 400
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scoring.dataset_base import format_conversation
from scoring.bios_dataset import BIOS_ASSESSMENT_PROMPT
from substrates.bios_ingest import DEFAULT_BIOS_PATH, load_bias_in_bios
from substrates.bios_render import render_bio
from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment
from probes.erasure import probe_recoverability
from probes.probe import get_embeddings


def occupation_baseline(train, evalr) -> Dict[str, float]:
    """How much of the real gender label is predictable from occupation alone, with no model.

    This is the control that decides how to *read* a leak. Bias-in-Bios exists because
    occupation correlates with gender (surgeon vs nurse), so a scrubbed body still carries a
    gender prior no name/pronoun scrub can remove. If the probe's margin over chance is roughly
    this baseline, the residual is the dataset's own phenomenon rather than a scrub defect.

    `target_role` is included as a design check: it is randomised for the mismatched half of the
    records, so it should sit at chance. If it does not, the role-match header is leaking.
    """
    from collections import Counter, defaultdict

    def majority(rows, key):
        agg = defaultdict(Counter)
        for r in rows:
            agg[getattr(r, key)][r.gender] += 1
        return {k: c.most_common(1)[0][0] for k, c in agg.items()}

    chance = max(Counter(r.gender for r in evalr).values()) / len(evalr)
    out = {"chance": round(chance, 4)}
    for key in ("profession", "target_role"):
        m = majority(train, key)
        acc = sum(m.get(getattr(r, key), 0) == r.gender for r in evalr) / len(evalr)
        out[f"{key}_acc"] = round(acc, 4)
        out[f"{key}_margin"] = round(acc - chance, 4)
    return out


def _embed(exp, cfg, records, *, scrubbed: bool) -> torch.Tensor:
    """Activations for the rendered profiles. `scrubbed=False` restores the raw body as a control."""
    texts = []
    for r in records:
        rec = r
        if not scrubbed:
            raw = str(r.extra.get("raw_bio", r.bio_text))
            rec = type(r)(**{**r.__dict__, "bio_text": raw})
        rendered = render_bio(rec, "bios_v1")
        texts.append(format_conversation(exp.tokenizer, BIOS_ASSESSMENT_PROMPT, rendered))
    return get_embeddings(exp.model, exp.tokenizer, texts, batch_size=cfg.batch_size,
                          device=cfg.device, max_length=cfg.max_length)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_cv_sex_qwen06.yaml"))
    ap.add_argument("--raw-path", default=DEFAULT_BIOS_PATH)
    ap.add_argument("--probe-items", type=int, default=400)
    ap.add_argument("--eval-items", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-control", action="store_true",
                    help="Skip the unscrubbed reference arm (faster, but loses the sanity check).")
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/results/demographic/scrubcheck_bios_qwen06.json"))
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    need = args.probe_items + args.eval_items
    records = load_bias_in_bios(args.raw_path, n=need, seed=args.seed, keep_raw=True)
    if len(records) < need:
        raise SystemExit(f"need {need} bios, loaded {len(records)} — raise the corpus sample or lower --*-items")
    random.Random(args.seed).shuffle(records)
    probe_recs, eval_recs = records[:args.probe_items], records[args.probe_items:need]

    y_tr = [r.gender for r in probe_recs]
    y_ev = [r.gender for r in eval_recs]

    arms = ["scrubbed"] if args.no_control else ["scrubbed", "unscrubbed"]
    results: Dict[str, Any] = {}
    for arm in arms:
        scrubbed = arm == "scrubbed"
        if not scrubbed and "raw_bio" not in (probe_recs[0].extra or {}):
            print("[scrubcheck] no raw_bio retained on records; skipping the unscrubbed control arm.")
            continue
        print(f"[scrubcheck] embedding {arm} ...", flush=True)
        Xtr = _embed(exp, cfg, probe_recs, scrubbed=scrubbed)
        Xev = _embed(exp, cfg, eval_recs, scrubbed=scrubbed)
        results[arm] = probe_recoverability(Xtr, y_tr, Xev, y_ev, seed=args.seed)

    occ = occupation_baseline(probe_recs, eval_recs)

    print("\n" + "=" * 84)
    print(f"BIAS-IN-BIOS SCRUB CHECK — real gender recoverability — {cfg.model_path}")
    print("=" * 84)
    print(f"{'arm':12} {'linear':>9} {'MLP':>9} {'chance':>9}  verdict")
    for arm, r in results.items():
        margin = max(r["linear_acc"], r["mlp_acc"]) - r["chance"]
        if arm == "scrubbed":
            verdict = "clean (<=0.05 over chance)" if margin <= 0.05 else "residual signal present"
        else:
            verdict = "decodable (expected)" if margin > 0.05 else "PROBE SETUP SUSPECT"
        r["margin_over_chance"] = round(margin, 4)
        print(f"{arm:12} {r['linear_acc']:>9.3f} {r['mlp_acc']:>9.3f} {r['chance']:>9.3f}  {verdict}")
    print("-" * 84)
    print("No-model baselines — how much of gender is predictable from a single field:")
    print(f"  occupation   {occ['profession_acc']:.3f}  ({occ['profession_margin']:+.3f} over chance)"
          "   <- irreducible: this is what Bias-in-Bios is *about*")
    print(f"  target role  {occ['target_role_acc']:.3f}  ({occ['target_role_margin']:+.3f})"
          "   <- design check: should sit at chance")

    scrub_margin = results.get("scrubbed", {}).get("margin_over_chance", 0.0)
    unexplained = scrub_margin - occ["profession_margin"]
    print("-" * 84)
    if scrub_margin <= 0.05:
        print("VERDICT: scrub clean — the injected marker is the only sex cue.")
    elif unexplained <= 0.05:
        print(f"VERDICT: residual {scrub_margin:+.3f} is essentially the occupational prior "
              f"({occ['profession_margin']:+.3f}); unexplained {unexplained:+.3f}.")
        print("  Not a scrub defect: occupation cannot be removed without destroying the substrate.")
        print("  The matched-pair contrast is still clean (the body is identical across A/B), but")
        print("  the item is NOT sex-neutral — report congruent vs incongruent pairs separately.")
    else:
        print(f"VERDICT: residual {scrub_margin:+.3f} exceeds the occupational prior "
              f"({occ['profession_margin']:+.3f}) by {unexplained:+.3f} — investigate the scrub.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": cfg.model_path, "seed": args.seed,
        "probe_items": args.probe_items, "eval_items": args.eval_items,
        "results": results, "occupation_baseline": occ,
        "unexplained_margin": round(unexplained, 4),
    }, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
