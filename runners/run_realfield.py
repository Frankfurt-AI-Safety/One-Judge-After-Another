#!/usr/bin/env python3
"""
Real-field marital-status arm (external-validity cross-check), credit arm, one RM (default Qwen3-0.6B).

Uses German Credit's ACTUAL `personal_status_sex` field (neutralized in the synthetic arm) to read a
real-data marital-status signal. Contrast holds sex = male:
    divorced/separated males (A91)  vs  married/widowed males (A93).

Codebook per Grömping (2019), see `substrates/credit_ingest.py`. Single males cannot be isolated —
they share code A92 with non-single women — so the only clean within-sex marital contrast is the one
above. A91 has only 50 records, 47 after the record-consistency rules, so both groups are n=47 and every number here is low-powered.
(Before 2026-09-16 this runner used the wrong UCI codebook and compared "single males" that were in
fact married/widowed males against a mix of divorced males and single women; those results are void.)

Builds the real-field marital difference-of-means direction and reports:
  1. cross-check cosines vs the SYNTHETIC marital-status and sex directions (does the real field encode
     the same direction as the synthetic injection? does the known sex/marital entanglement surface?);
  2. the divorced-vs-married mean reward gap, baseline vs nulled (project out the real-field direction).

HONEST LIMITATION: the two marital groups are different applicants whose financials also differ, so
this direction/gap is **confounded** with marital-correlated financials — a cross-check, not a clean
single-axis result. (Deferred refinement: balance/match the groups on financials.)

Usage:
    python runners/run_realfield.py --config configs/demographic_credit_sex_qwen06.yaml
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

from scoring.dataset_base import ContrastivePair, format_conversation
from scoring.pair_dataset import ASSESSMENT_PROMPT, CreditDemographicDataset
from substrates.credit_clean import RECORD_RULES, apply_rules
from substrates.credit_ingest import load_german_credit
from pairs.markers import real_field_clause
from substrates.credit_render import render_profile, TEMPLATES
from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment
from probes.probe import build_probe_direction, get_rewards_both


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a / (a.norm() + 1e-8)) @ (b / (b.norm() + 1e-8)))


def _synthetic_dir(exp, cfg, axis):
    ds = CreditDemographicDataset(cfg.dataset_source, axis=axis, encoding="explicit",
                                  probe_size=cfg.probe_size, split_seed=cfg.split_seed,
                                  probe_records=cfg.probe_records)
    probe, _ = build_probe_direction(exp.model, exp.tokenizer, ds.get_probe_pairs(exp.tokenizer),
                                     batch_size=cfg.batch_size, device=cfg.device, max_length=cfg.max_length)
    return probe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_credit_sex_qwen06.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/results/demographic/realfield_marital_qwen06.json"))
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    tok = exp.tokenizer
    fmt = lambda r, t: format_conversation(tok, ASSESSMENT_PROMPT,
                                           render_profile(r, t, marker=real_field_clause(r)))

    # Select males, split by real marital status; balance groups (sample married to divorced count).
    records, _ = apply_rules(load_german_credit(), RECORD_RULES)
    males = [r for r in records if r.raw_sex == "male"]
    divorced = [r for r in males if r.raw_marital == "divorced/separated"]
    married = [r for r in males if r.raw_marital == "married/widowed"]
    rng = random.Random(args.seed)
    rng.shuffle(divorced)
    rng.shuffle(married)
    n = min(len(divorced), len(married))
    divorced, married = divorced[:n], married[:n]
    tids = sorted(TEMPLATES)
    divorced_txt = [fmt(r, tids[i % len(tids)]) for i, r in enumerate(divorced)]
    married_txt = [fmt(r, tids[i % len(tids)]) for i, r in enumerate(married)]

    # Real-field marital direction (divorced − married); DiffMean uses only group means.
    pairs = [ContrastivePair(positive_text=d, negative_text=m) for d, m in zip(divorced_txt, married_txt)]
    real_dir, meta = build_probe_direction(exp.model, tok, pairs, batch_size=cfg.batch_size,
                                           device=cfg.device, max_length=cfg.max_length)

    # Cross-check cosines vs synthetic directions.
    # Synthetic marital direction is married − single; the real one is divorced − married, so a
    # shared "married" component shows up as a NEGATIVE cosine.
    marital_dir = _synthetic_dir(exp, cfg, "marital_status")
    sex_dir = _synthetic_dir(exp, cfg, "sex")
    cos_marital, cos_sex = _cos(real_dir, marital_dir), _cos(real_dir, sex_dir)

    # Group reward gap (divorced − married), baseline vs nulled (project out real_dir).
    d_base, d_null = get_rewards_both(exp.model, tok, divorced_txt, real_dir, batch_size=cfg.batch_size,
                                      device=cfg.device, max_length=cfg.max_length, null_alpha=1.0,
                                      show_progress=False)
    m_base, m_null = get_rewards_both(exp.model, tok, married_txt, real_dir, batch_size=cfg.batch_size,
                                      device=cfg.device, max_length=cfg.max_length, null_alpha=1.0,
                                      show_progress=False)
    gap_base = float(d_base.mean() - m_base.mean())
    gap_null = float(d_null.mean() - m_null.mean())

    print("\n" + "=" * 78)
    print(f"REAL-FIELD MARITAL STATUS — {cfg.model_path}")
    print("=" * 78)
    print(f"groups: divorced/separated males n={n}  vs  married/widowed males n={n}  (sex held = male)")
    print(f"real-field probe: accuracy={meta.get('probe_accuracy', 0):.2%}  separation={meta.get('separation', 0):.3f}")
    print(f"cosine(real_marital, synthetic MARITAL dir)       = {cos_marital:+.4f}   (external validity)")
    print(f"cosine(real_marital, synthetic SEX dir)           = {cos_sex:+.4f}   (sex/marital entanglement)")
    print(f"reward gap divorced−married:  baseline={gap_base:+.4f}   nulled={gap_null:+.4f}")
    print("=" * 78)
    print("Caveat: groups differ in financials too → confounded cross-check, not clean single-axis.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": cfg.model_path, "n_per_group": n, "seed": args.seed,
        "probe_accuracy": meta.get("probe_accuracy"), "probe_separation": meta.get("separation"),
        "cosine_real_vs_synthetic_marital": cos_marital, "cosine_real_vs_synthetic_sex": cos_sex,
        "reward_gap_divorced_minus_married": {"baseline": gap_base, "nulled": gap_null},
        "divorced_ids": [r.source_record_id for r in divorced],
        "married_ids": [r.source_record_id for r in married],
    }, indent=2))
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
