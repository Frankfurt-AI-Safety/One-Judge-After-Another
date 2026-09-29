#!/usr/bin/env python3
"""
Real-field marital-status arm (external-validity cross-check), credit arm, one RM.

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

Builds the real-field marital difference-of-means direction (divorced − married, from the matched pairs) and
reports:
  1. cross-check cosines vs the SYNTHETIC marital-status and sex directions (fitted on the direct manifest's
     probe records: does the real field encode the same direction as the synthetic injection? does the known
     sex/marital entanglement surface?), with each direction's full-sample reliability (split-half cosine
     stepped up), since the real one rests on 47 pairs of different applicants: a cosine is read against
     √(rel_real · rel_synthetic), not against 1;
  2. the divorced-vs-married mean reward gap per matched pair: baseline; nulled HELD OUT (each pair with the
     real-field direction fitted on the other pairs, leave one out); and nulled with the SYNTHETIC marital
     direction — the external-validity question: does the injected marker's direction carry the real field's
     gap? Until 2026-09-29 the pairs were nulled with the direction fitted on themselves, which removes their mean
     state difference exactly, so for a linear head the "nulled" gap was 0 by construction.

``probe_accuracy_in_sample`` is measured on the 47 pairs the direction was fitted on. The result
``realfield_marital_{model}.json`` (never replaced without ``--overwrite``) carries ``meta``: the config with the
loaded model commit, the code commit, the direct manifest's and ``german.data``'s SHA-256 (`scoring.experiment`).

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from pairs.factorial import stable_rng
from pairs.manifest import file_sha256
from scoring.cross_marker_metrics import summarize
from scoring.dataset_base import format_conversation
from scoring.intervals import DEFAULT_N_BOOT
from scoring.pair_dataset import ASSESSMENT_PROMPT, CreditDemographicDataset
from substrates.credit_clean import RECORD_RULES, apply_rules
from substrates.credit_ingest import DEFAULT_RAW_PATH, load_german_credit
from pairs.markers import real_field_clause
from substrates.credit_render import render_profile, TEMPLATES
from scoring.experiment import ExperimentConfig, add_override_args, apply_overrides, data_file, run_metadata
from scoring.demographic_experiment import DemographicBiasExperiment
from probes.cross_marker_directions import full_sample_reliability, split_half_cosine
from probes.probe import embed_states, embed_with_gates, rewards_from_hidden
from runners.run_cross_marker import record_contrast_matrix

RESULTS_DIR = Path("artifacts/results/demographic")


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


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def _synthetic(exp, cfg, axis: str, seed: int) -> Tuple[torch.Tensor, float]:
    """The direct arm's explicit difference-of-means direction for ``axis`` (the manifest's probe records) and its
    full-sample reliability (split-half over records, stepped up); the states are embedding-cache hits."""
    ds = CreditDemographicDataset(cfg.dataset_source, axis=axis, encoding="explicit", split_seed=cfg.split_seed,
                                  probe_records=cfg.probe_records)
    pairs = ds.get_probe_pairs(exp.tokenizer)
    pos, _ = embed_states(exp.model, exp.tokenizer, [p.positive_text for p in pairs], batch_size=cfg.batch_size,
                          max_length=cfg.max_length, show_progress=False)
    neg, _ = embed_states(exp.model, exp.tokenizer, [p.negative_text for p in pairs], batch_size=cfg.batch_size,
                          max_length=cfg.max_length, show_progress=False)
    contrasts = record_contrast_matrix(pairs, pos, neg)
    return _unit(contrasts.mean(0)), full_sample_reliability(split_half_cosine(contrasts, seed))


def pair_gaps(model, h_a: torch.Tensor, h_b: torch.Tensor, dtype, gates_a=None, gates_b=None,
              direction: Optional[torch.Tensor] = None) -> List[float]:
    """Reward gap a − b of each pair (row k of ``h_a`` against row k of ``h_b``), with ``direction`` projected out
    of both (None = baseline)."""
    ra = rewards_from_hidden(model, h_a, dtype, direction, gates=gates_a)[1 if direction is not None else 0]
    rb = rewards_from_hidden(model, h_b, dtype, direction, gates=gates_b)[1 if direction is not None else 0]
    return (ra - rb).tolist()


def leave_one_out_gaps(model, h_a: torch.Tensor, h_b: torch.Tensor, dtype, gates_a=None, gates_b=None) -> List[float]:
    """Each pair's gap with the difference-of-means direction of the OTHER pairs projected out: the held-out
    nulled gap. (Fitted on all pairs, the direction is their mean state difference, and projecting it out of
    them sets their mean gap to 0 for any linear head, by construction.)"""
    diffs = (h_a - h_b).float()
    total = diffs.sum(0)
    gaps = []
    for k in range(diffs.shape[0]):
        pick = lambda g: None if g is None else g[k:k + 1]
        gaps += pair_gaps(model, h_a[k:k + 1], h_b[k:k + 1], dtype, pick(gates_a), pick(gates_b),
                          _unit(total - diffs[k]))
    return gaps


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_credit_sex_qwen06.yaml"))
    ap.add_argument("--raw", type=Path, default=Path(DEFAULT_RAW_PATH), help="German Credit's german.data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"Default {RESULTS_DIR}/realfield_marital_{{model}}.json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result")
    add_override_args(ap)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    # everything that can fail on the inputs fails here, before the model loads
    out = args.out or RESULTS_DIR / f"realfield_marital_{Path(cfg.model_path).name}.json"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or --out")
    data = {"pairs.jsonl": data_file(cfg.dataset_source),
            "german.data": {"path": str(args.raw), "sha256": file_sha256(args.raw)}}

    # Males split by real marital status; each divorced man matched to a married man on MATCH_KEYS.
    records, _ = apply_rules(load_german_credit(args.raw), RECORD_RULES)
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

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    tok = exp.tokenizer
    fmt = lambda r, t: format_conversation(tok, ASSESSMENT_PROMPT, render_profile(r, t, marker=real_field_clause(r)))
    tids = sorted(TEMPLATES)
    # both members of a pair are rendered with the same template
    divorced_txt = [fmt(r, tids[i % len(tids)]) for i, r in enumerate(divorced)]
    married_txt = [fmt(r, tids[i % len(tids)]) for i, r in enumerate(married)]
    h_d, dtype, g_d = embed_with_gates(exp.model, tok, divorced_txt, batch_size=cfg.batch_size,
                                       max_length=cfg.max_length, show_progress=False)
    h_m, _, g_m = embed_with_gates(exp.model, tok, married_txt, batch_size=cfg.batch_size,
                                   max_length=cfg.max_length, show_progress=False)

    # Real-field marital direction (divorced − married) from the matched pairs: the difference of means.
    diffs = (h_d - h_m).float()
    real_dir = _unit(diffs.mean(0))
    rel_real = full_sample_reliability(split_half_cosine(diffs, args.seed))
    proj_d, proj_m = h_d.float() @ real_dir, h_m.float() @ real_dir
    threshold = (proj_d.mean() + proj_m.mean()) / 2
    probe_accuracy = float(((proj_d > threshold).sum() + (proj_m <= threshold).sum()) / (2 * n))   # in-sample

    # Cross-check cosines vs synthetic directions.
    # Synthetic marital direction is married − single; the real one is divorced − married, so a
    # shared "married" component shows up as a NEGATIVE cosine.
    marital_dir, rel_marital = _synthetic(exp, cfg, "marital_status", args.seed)
    sex_dir, rel_sex = _synthetic(exp, cfg, "sex", args.seed)
    cos_marital, cos_sex = _cos(real_dir, marital_dir), _cos(real_dir, sex_dir)

    # Group reward gap (divorced − married) per matched pair: baseline, held-out nulled, synthetic-nulled.
    summ = lambda gaps: summarize(gaps, n_boot=args.n_boot, seed=args.seed)
    gap = {"baseline": summ(pair_gaps(exp.model, h_d, h_m, dtype, g_d, g_m)),
           "nulled_heldout": summ(leave_one_out_gaps(exp.model, h_d, h_m, dtype, g_d, g_m)),
           "nulled_synthetic_marital": summ(pair_gaps(exp.model, h_d, h_m, dtype, g_d, g_m, marital_dir))}
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
    print(f"real-field direction: reliability {rel_real:.3f}; in-sample probe accuracy {probe_accuracy:.2%}")
    print(f"cosine(real_marital, synthetic MARITAL dir) = {cos_marital:+.4f}   (external validity; "
          f"ceiling √(rel·rel) {np.sqrt(max(rel_real, 0) * max(rel_marital, 0)):.3f})")
    print(f"cosine(real_marital, synthetic SEX dir)     = {cos_sex:+.4f}   (sex/marital entanglement; "
          f"ceiling {np.sqrt(max(rel_real, 0) * max(rel_sex, 0)):.3f})")
    fmt_gap = lambda g: f"{g['mean']:+.4f} [{g['ci_low']:+.3f},{g['ci_high']:+.3f}] d_z={g['d_z']:+.2f}"
    print("reward gap divorced−married (per matched pair):")
    print(f"  baseline                         {fmt_gap(gap['baseline'])}")
    print(f"  nulled, real direction held out  {fmt_gap(gap['nulled_heldout'])}")
    print(f"  nulled, synthetic marital dir    {fmt_gap(gap['nulled_synthetic_marital'])}")
    print("=" * 78)
    print("Caveat: matched on label, checking and dependents only; see the balance table for the rest.")

    settings = {"match_keys": list(MATCH_KEYS), "seed": args.seed, "n_boot": args.n_boot}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": run_metadata(cfg, data, settings),
        "model": cfg.model_path, "n_per_group": n, "seed": args.seed,
        "match_keys": list(MATCH_KEYS), "n_unmatched": len(unmatched),
        "unmatched_ids": [r.source_record_id for r in unmatched],
        "balance": balance,
        "probe_accuracy_in_sample": probe_accuracy,
        "reliability": {"real_marital": rel_real, "synthetic_marital": rel_marital, "synthetic_sex": rel_sex},
        "cosine_real_vs_synthetic_marital": cos_marital, "cosine_real_vs_synthetic_sex": cos_sex,
        "reward_gap_divorced_minus_married": gap,
        "pairs": [[d.source_record_id, m.source_record_id] for d, m in matched],
    }, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
