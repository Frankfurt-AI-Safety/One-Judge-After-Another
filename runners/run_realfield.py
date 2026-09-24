#!/usr/bin/env python3
"""
Real-field marital-status arm (external-validity cross-check), credit arm, one RM (default Qwen3-0.6B).

Uses German Credit's ACTUAL `personal_status_sex` field (neutralized in the synthetic arm) to read a
real-data marital-status signal. Contrast holds sex = male:
    divorced/separated males (A91)  vs  married/widowed males (A93).

Codebook per Grömping (2019), see `substrates/credit_ingest.py`. Single males cannot be isolated —
they share code A92 with non-single women — so the only clean within-sex marital contrast is the one
above. A91 has only 50 records, 47 after the record-consistency rules, so both groups are n=47 and every number here is low-powered.

**Matched groups (2026-09-24, audit item 4.2).** The two groups are different applicants. Drawn at random,
the married men had a good-credit rate 14 points higher and differed in rendered fields (dependents,
checking account), so the direction and the gap mixed marital status with quality. Each divorced man is now
paired with a married man of the SAME `MATCH_KEYS` (good-credit label, checking account, dependents;
exact, without replacement, seeded): 47 of 47 find one. Savings is left out of the keys, since it would
cost 4 of the 47; the remaining imbalance on every rendered field is reported (`balance_table`), next to
that of a random draw of the same size, so the gain is visible. The gap is the mean of the per-pair
differences, with a bootstrap CI over pairs.
(Before 2026-09-16 this runner used the wrong UCI codebook and compared "single males" that were in
fact married/widowed males against a mix of divorced males and single women; those results are void.)

Builds the real-field marital difference-of-means direction and reports:
  1. cross-check cosines vs the SYNTHETIC marital-status and sex directions (does the real field encode
     the same direction as the synthetic injection? does the known sex/marital entanglement surface?);
  2. the divorced-vs-married mean reward gap, baseline vs nulled (project out the real-field direction).

LIMITATION: matching removes the label and the two most unequal fields, not every difference between two
sets of real applicants (see the balance table) — a cross-check, not a clean single-axis result.

Usage:
    python runners/run_realfield.py --config configs/demographic_credit_sex_qwen06.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from pairs.factorial import stable_rng
from scoring.cross_marker_metrics import summarize
from scoring.dataset_base import ContrastivePair, format_conversation
from scoring.pair_dataset import ASSESSMENT_PROMPT, CreditDemographicDataset
from substrates.credit_clean import RECORD_RULES, apply_rules
from substrates.credit_ingest import load_german_credit
from pairs.markers import real_field_clause
from substrates.credit_render import render_profile, TEMPLATES
from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment
from probes.probe import build_probe_direction, get_rewards_both


# Exact-match keys: the quality label and the two rendered fields the random groups differed on most.
MATCH_KEYS: Tuple[str, ...] = ("credit_good", "checking", "dependents")
# Every field the profile renders (`substrates.credit_render._slots`), plus the label.
CATEGORICAL_FIELDS: Tuple[str, ...] = (
    "credit_good", "purpose", "checking", "savings", "employment_since", "job", "existing_credits",
    "credit_history", "housing", "property", "installment_rate", "other_installment_plans", "dependents")
NUMERIC_FIELDS: Tuple[str, ...] = ("duration_months", "credit_amount_dm")


def match_groups(treated: Sequence[Any], pool: Sequence[Any], keys: Sequence[str],
                 seed: int) -> Tuple[List[Tuple[Any, Any]], List[Any]]:
    """1:1 exact matching without replacement: the treated records, in a seeded order, each take a record
    drawn (seeded) from the still-unused pool records with the same values of ``keys``. Returns the
    ``(treated, control)`` pairs and the treated records left without a match."""
    key = lambda r: tuple(getattr(r, k) for k in keys)
    by_key: Dict[Tuple[Any, ...], List[Any]] = defaultdict(list)
    for r in sorted(pool, key=lambda r: r.source_record_id):
        by_key[key(r)].append(r)
    order = sorted(treated, key=lambda r: r.source_record_id)
    stable_rng(seed, "realfield", "order").shuffle(order)
    rng = stable_rng(seed, "realfield", "controls")
    pairs: List[Tuple[Any, Any]] = []
    unmatched: List[Any] = []
    for t in order:
        candidates = by_key.get(key(t), [])
        if candidates:
            pairs.append((t, candidates.pop(rng.randrange(len(candidates)))))
        else:
            unmatched.append(t)
    return pairs, unmatched


def balance_table(a: Sequence[Any], b: Sequence[Any]) -> Dict[str, Dict[str, float]]:
    """How far two groups differ on each field: the total-variation distance between the level
    distributions for a categorical field (0 = identical, 1 = disjoint) and the standardised mean
    difference (pooled SD) for a numeric one."""
    tv: Dict[str, float] = {}
    for f in CATEGORICAL_FIELDS:
        ca, cb = Counter(getattr(r, f) for r in a), Counter(getattr(r, f) for r in b)
        tv[f] = 0.5 * sum(abs(ca[v] / len(a) - cb[v] / len(b)) for v in set(ca) | set(cb))
    smd: Dict[str, float] = {}
    for f in NUMERIC_FIELDS:
        va = np.array([getattr(r, f) for r in a], dtype=float)
        vb = np.array([getattr(r, f) for r in b], dtype=float)
        pooled = float(np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)) if min(len(a), len(b)) > 1 else 0.0
        smd[f] = float((va.mean() - vb.mean()) / pooled) if pooled else 0.0
    return {"total_variation": tv, "standardised_mean_difference": smd}


def chance_balance(pool: Sequence[Any], n: int, seed: int, draws: int = 200) -> Dict[str, Dict[str, float]]:
    """What `balance_table` shows between two random samples of ``n`` from ONE population (the mean over
    ``draws`` pairs of samples, |SMD| for numeric fields): the floor the matched groups are read against.
    At n=47 a many-level field differs by a TV distance of ~0.1 by chance alone."""
    pool = sorted(pool, key=lambda r: r.source_record_id)
    tables = [balance_table(stable_rng(seed, "realfield", "chance", k, "a").sample(pool, n),
                            stable_rng(seed, "realfield", "chance", k, "b").sample(pool, n))
              for k in range(draws)]
    return {part: {f: float(np.mean([abs(t[part][f]) for t in tables])) for f in tables[0][part]}
            for part in tables[0]}


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
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/results/demographic/realfield_marital_qwen06.json"))
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    tok = exp.tokenizer
    fmt = lambda r, t: format_conversation(tok, ASSESSMENT_PROMPT,
                                           render_profile(r, t, marker=real_field_clause(r)))

    # Males split by real marital status; each divorced man matched to a married man on MATCH_KEYS.
    records, _ = apply_rules(load_german_credit(), RECORD_RULES)
    males = [r for r in records if r.raw_sex == "male"]
    divorced_all = [r for r in males if r.raw_marital == "divorced/separated"]
    married_all = [r for r in males if r.raw_marital == "married/widowed"]
    matched, unmatched = match_groups(divorced_all, married_all, MATCH_KEYS, args.seed)
    n = len(matched)
    divorced = [d for d, _ in matched]
    married = [m for _, m in matched]
    # the pre-2026-09-24 design, for comparison only: the same divorced men against a random draw
    random_draw = stable_rng(args.seed, "realfield", "random_draw").sample(
        sorted(married_all, key=lambda r: r.source_record_id), n)
    balance = {"matched": balance_table(divorced, married),
               "random_draw_same_size": balance_table(divorced, random_draw),
               "chance_same_population": chance_balance(married_all, n, args.seed)}
    tids = sorted(TEMPLATES)
    # both members of a pair are rendered with the same template
    divorced_txt = [fmt(r, tids[i % len(tids)]) for i, r in enumerate(divorced)]
    married_txt = [fmt(r, tids[i % len(tids)]) for i, r in enumerate(married)]

    # Real-field marital direction (divorced − married) from the matched pairs.
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
    gap = {"baseline": summarize((d_base - m_base).tolist(), n_boot=args.n_boot, seed=args.seed),
           "nulled": summarize((d_null - m_null).tolist(), n_boot=args.n_boot, seed=args.seed)}
    worst = lambda t: max({**t["total_variation"], **{k: abs(v) for k, v in
                                                       t["standardised_mean_difference"].items()}}.items(),
                          key=lambda kv: kv[1])

    print("\n" + "=" * 78)
    print(f"REAL-FIELD MARITAL STATUS — {cfg.model_path}")
    print("=" * 78)
    print(f"groups: divorced/separated males n={n}  vs  married/widowed males n={n}  (sex held = male), "
          f"exact-matched on {', '.join(MATCH_KEYS)} ({len(unmatched)} unmatched)")
    mean_tv = lambda t: float(np.mean(list(t["total_variation"].values())))
    print(f"imbalance (mean TV over fields | largest, TV or |SMD|): matched {mean_tv(balance['matched']):.3f} | "
          f"{worst(balance['matched'])[0]} {worst(balance['matched'])[1]:.2f};  random draw "
          f"{mean_tv(balance['random_draw_same_size']):.3f};  chance within one population "
          f"{mean_tv(balance['chance_same_population']):.3f}")
    print(f"real-field probe: accuracy={meta.get('probe_accuracy', 0):.2%}  separation={meta.get('separation', 0):.3f}")
    print(f"cosine(real_marital, synthetic MARITAL dir)       = {cos_marital:+.4f}   (external validity)")
    print(f"cosine(real_marital, synthetic SEX dir)           = {cos_sex:+.4f}   (sex/marital entanglement)")
    fmt_gap = lambda g: f"{g['mean']:+.4f} [{g['ci_low']:+.3f},{g['ci_high']:+.3f}] d_z={g['d_z']:+.2f}"
    print(f"reward gap divorced−married (per matched pair):  baseline={fmt_gap(gap['baseline'])}   "
          f"nulled={fmt_gap(gap['nulled'])}")
    print("=" * 78)
    print("Caveat: matched on label, checking and dependents only; see the balance table for the rest.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": cfg.model_path, "n_per_group": n, "seed": args.seed,
        "match_keys": list(MATCH_KEYS), "n_unmatched": len(unmatched),
        "unmatched_ids": [r.source_record_id for r in unmatched],
        "balance": balance,
        "probe_accuracy": meta.get("probe_accuracy"), "probe_separation": meta.get("separation"),
        "cosine_real_vs_synthetic_marital": cos_marital, "cosine_real_vs_synthetic_sex": cos_sex,
        "reward_gap_divorced_minus_married": gap,
        "pairs": [[d.source_record_id, m.source_record_id] for d, m in matched],
    }, indent=2))
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
