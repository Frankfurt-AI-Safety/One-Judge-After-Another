#!/usr/bin/env python3
"""
First look at intersectional **additivity** (RQ1-b / H1-b).

Builds difference-of-means probe directions for the domain's three marginal axes (credit: sex, age,
marital_status; cv: sex, age, family_status) and for the combined **intersection** axis on one RM, all
oriented A-pole − B-pole and built from the SAME contrast poles, then reports:

    cosine( intersection_dir , normalize(sum of the three marginal dirs) )

High cosine ⇒ the intersectional direction ≈ the sum of its marginals (**additive / low-complexity**);
low cosine ⇒ a distinct interaction (**non-additive**, the more interesting & harder-to-fix case).

This is a first linear look only — the rigorous test (LEACE + non-linear-probe recoverability) is
deferred. Caveat: difference-of-means is validated mostly on binary attributes; the combined cell is
multi-attribute (cardinality caveat).

Usage:
    python runners/run_additivity.py --domain credit --encoding explicit --probe-records 150

Every direction is fitted on the same ``--probe-records`` records (stratified by quality), so the
intersection and its marginals are compared at equal precision. (Counting pairs, as ``--probe-size``
does, gave the marginals ~38 records and the intersection 150: a record contributes 8 pairs to a
marginal axis but 2 to the intersection.)

Every cosine also gets a 95% interval from a bootstrap over the probe records (``intervals``): the
directions are refitted on each resample of the shared records, from the per-record contrasts (a record's
pairs share its content). The interval shows whether a cosine clears a verdict threshold or only its point
estimate does.

Credit has no proxy for marital status (see pairs/factorial.py), so credit additivity is explicit only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment
from substrates.domains import DOMAINS, get_domain
from probes.probe import build_probe_direction, embed_states
from runners.run_cross_marker import record_contrasts
from scoring.intervals import DEFAULT_N_BOOT

# Short names for the output keys; `family` keeps the historical CV keys (cos_sex_family, ...).
_SHORT = {"family_status": "family", "marital_status": "marital"}


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a / a.norm()) @ (b / b.norm()))


def _unit_rows(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def additivity_intervals(contrasts: Mapping[str, Mapping[str, torch.Tensor]], marginal_axes: Sequence[str],
                         n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Record-bootstrap intervals of cos(intersection, Σ unit marginals) and of the pairwise marginal
    cosines. ``contrasts[axis][record]`` is the record's mean pair contrast; every direction is the unit
    mean over the resampled records (the difference of means, as `build_probe_direction` fits it when
    every record has the same number of pairs). Only records present on every axis are used."""
    axes = list(marginal_axes) + ["intersection"]
    records = sorted(set.intersection(*(set(contrasts[a]) for a in axes)))
    k = len(records)
    mats = {a: torch.stack([contrasts[a][r] for r in records]).float().numpy() for a in axes}
    draws = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    weights = np.stack([np.bincount(row, minlength=k) for row in draws]) / k        # (n_boot, k)
    full = np.full((1, k), 1.0 / k)
    short = [_SHORT.get(a, a) for a in marginal_axes]

    def cosines(w: np.ndarray) -> Dict[str, np.ndarray]:
        dirs = {a: _unit_rows(w @ mats[a]) for a in axes}                             # (b, d) each
        total = sum(dirs[a] for a in marginal_axes)
        out = {"cos_intersection_vs_marginal_sum": np.sum(dirs["intersection"] * _unit_rows(total), -1)}
        for i in range(len(marginal_axes)):
            for j in range(i + 1, len(marginal_axes)):
                out[f"cos_{short[i]}_{short[j]}"] = np.sum(dirs[marginal_axes[i]] * dirs[marginal_axes[j]], -1)
        return out

    point, boot = cosines(full), cosines(weights)
    return {name: {"estimate": float(point[name][0]), "ci_low": float(np.percentile(boot[name], 2.5)),
                   "ci_high": float(np.percentile(boot[name], 97.5)), "n_records": k}
            for name in point}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    ap.add_argument("--domain", default="credit", choices=sorted(DOMAINS))
    ap.add_argument("--pairs", default=None, help="Pairs manifest (defaults per --domain)")
    ap.add_argument("--encoding", default="explicit", choices=["explicit", "proxy"])
    ap.add_argument("--probe-records", type=int, default=150,
                    help="Probe records per direction, the same for every axis (0 = count pairs instead)")
    ap.add_argument("--probe-size", type=int, default=300, help="Probe PAIRS; only with --probe-records 0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON output (feeds runners/export_paper_numbers.py)")
    args = ap.parse_args()

    spec = get_domain(args.domain)
    marginal_axes = [a for a in spec.axes if a != "intersection"]
    if "intersection" not in spec.axes or len(marginal_axes) != 3:
        raise SystemExit(f"domain {spec.name!r} has no three-marginal intersection design")
    if spec.name == "credit" and args.encoding != "explicit":
        raise SystemExit("credit additivity is explicit only: marital status has no proxy encoding")
    dataset_cls = spec.dataset_cls
    pairs_path = args.pairs or spec.default_pairs

    # Load the model/backend once via the experiment's loader (auto→CUDA when present).
    cfg = ExperimentConfig(name="additivity", bias_type="demographic", model_path=args.model,
                           device=args.device, batch_size=args.batch_size, max_length=args.max_length)
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    probes = {}
    contrasts: Dict[str, Dict[str, torch.Tensor]] = {}
    for axis in marginal_axes + ["intersection"]:
        ds = dataset_cls(pairs_path, axis=axis, encoding=args.encoding,
                         probe_size=args.probe_size, split_seed=cfg.split_seed,
                         probe_records=args.probe_records or None)
        pairs = ds.get_probe_pairs(exp.tokenizer)
        probe, meta = build_probe_direction(exp.model, exp.tokenizer, pairs,
                                            batch_size=args.batch_size, device=args.device,
                                            max_length=args.max_length)
        probes[axis] = probe
        # the same states again (embedding-cache hits), per record for the bootstrap
        pos, _ = embed_states(exp.model, exp.tokenizer, [p.positive_text for p in pairs],
                              batch_size=args.batch_size, max_length=args.max_length, show_progress=False)
        neg, _ = embed_states(exp.model, exp.tokenizer, [p.negative_text for p in pairs],
                              batch_size=args.batch_size, max_length=args.max_length, show_progress=False)
        ids, rows = record_contrasts(pairs, pos, neg)
        contrasts[axis] = dict(zip(ids, rows))
        split = ds.split_report()
        print(f"  {axis:14} probe: n={len(pairs)} pairs / {split.get('probe_records')} records "
              f"acc={meta.get('probe_accuracy', 0):.2%} sep={meta.get('separation', 0):.3f}")

    marg_sum = sum(probes[a] for a in marginal_axes)
    cos_inter_sum = _cos(probes["intersection"], marg_sum)
    short = [_SHORT.get(a, a) for a in marginal_axes]
    pairwise = {f"cos_{short[i]}_{short[j]}": _cos(probes[marginal_axes[i]], probes[marginal_axes[j]])
                for i in range(3) for j in range(i + 1, 3)}

    print("\n" + "=" * 64)
    print(f"ADDITIVITY ({args.encoding}, {args.model})")
    print("=" * 64)
    intervals = additivity_intervals(contrasts, marginal_axes, args.n_boot, args.seed)
    ci = lambda key: f"[{intervals[key]['ci_low']:+.4f}, {intervals[key]['ci_high']:+.4f}]"
    print(f"cosine(intersection, {'+'.join(short)}) = {cos_inter_sum:.4f}   95% {ci('cos_intersection_vs_marginal_sum')}"
          f"  ({intervals['cos_intersection_vs_marginal_sum']['n_records']} records, bootstrap)")
    print("pairwise marginal cosines (overlap):")
    for key, val in pairwise.items():
        print(f"  {key[4:].replace('_', '·'):18} = {val:+.4f}   95% {ci(key)}")
    verdict = ("ADDITIVE (≈ low-complexity)" if cos_inter_sum >= 0.9
               else "PARTIALLY ADDITIVE" if cos_inter_sum >= 0.6
               else "NON-ADDITIVE (distinct interaction)")
    print(f"\nverdict: {verdict}  [first linear look; LEACE/MLP test deferred]")
    print("=" * 64)

    if args.out is not None:
        import json
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "model": args.model, "domain": args.domain, "encoding": args.encoding,
            "marginal_axes": marginal_axes,
            "probe_records": args.probe_records or None,
            "cos_intersection_vs_marginal_sum": cos_inter_sum,
            **pairwise,
            "verdict": verdict,
            "intervals": intervals,
        }, indent=2))
        print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
