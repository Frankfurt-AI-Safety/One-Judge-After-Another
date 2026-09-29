#!/usr/bin/env python3
"""
Intersectional additivity of the direct arm's directions, one RM (RQ1.1, in state space).

For the domain's three marginal axes (credit: sex, age, marital_status; cv: sex, age, family_status; education:
sex, ethnicity, economic_status) and the intersection (the all-pole-A cell against the all-pole-B cell), each
record's mean pair contrast (positive − negative last-token state) is taken on the SAME probe records
(``--probe-records``, stratified by quality; the split ignores the axis). With the direction of an axis the mean
of those contrasts (the difference of means), the runner reports:

    cos( corner , Σ marginal contrasts )            the corner against the VECTOR SUM (RQ1.1's definition)
    residual r = corner − Σ marginal contrasts       per record; its mean estimates twice the three-way term

The residual is reported as a share of the corner's length three ways: raw ‖r̄‖/‖corner‖; the noise floor
√(tr S / n)/‖corner‖, what the raw share reaches on average with no three-way term at all (S = the records'
residual covariance); and the debiased share √max(0, ‖r̄‖² − tr S / n)/‖corner‖. "No three-way term in state
space" is tested by sign flips: under it each record's residual is centred on 0, so flipping the records' signs
at random gives the null distribution of ‖r̄‖² (valid if each residual is symmetric about 0; p = (1 + #{flip ≥
observed}) / (1 + flips)). The cosine is a point value, descriptive. The pairwise marginal cosines get 95%
bootstrap intervals over the records.

Length-type statistics are not given bootstrap intervals: resampling noise lengthens a residual and shortens a
cosine, so their percentile intervals miss the estimate (on Qwen3-0.6B, credit: cosine 0.9994 with "interval"
[0.9989, 0.9993], raw share 0.034 with [0.038, 0.047]); nor a reliability ceiling, which assumes independent noise
while the corner and the marginals share cell states (the cosine exceeded it). Both were tried and dropped
2026-09-29.

**Reading rule.** In effects coding the corner contrast is 2·Σ main effects + 2·(the three-way term): the two-way
interactions cancel in it (as in reward space, `scoring/cross_marker_metrics.py`). So the corner equals the vector
sum of the marginals — cosine at 1 up to noise, debiased residual at 0, sign-flip p uniform — whenever there is no
THREE-WAY structure, however strong a two-way interaction is (a "young women" effect leaves the cosine at 1). A
small p means three-way structure (its size: the debiased share); it says nothing about two-way
intersectionality, which is read from the cross-marker design's two-way interactions (reward space). Until
2026-09-29 the corner was compared with the sum of the UNIT marginal directions, which is not the additive
prediction when the contrasts differ in length and always biased the cosine down (Qwen3-0.6B, credit: 0.958 vs
0.9994 against the vector sum), and a verdict (≥ 0.9 "additive", ≥ 0.6 "partially") was read off the point
estimate; both are gone.

This is a first linear look only (the LEACE + non-linear test is deferred); difference-of-means is validated
mostly on binary attributes, and the corner is a multi-attribute contrast. Credit has no proxy for marital
status (pairs/factorial.py), so credit additivity is explicit only.

The result ``additivity_{domain}_{model}_{encoding}.json`` (never replaced without ``--overwrite``) carries
``meta`` (the config with the loaded model commit, the code commit, the manifest's SHA-256; `scoring.experiment`).

Usage:
    python runners/run_additivity.py --domain credit --encoding explicit --probe-records 150
    python runners/run_additivity.py --domain cv --encoding proxy --model Skywork/Skywork-Reward-V2-Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scoring.experiment import ExperimentConfig, add_override_args, apply_overrides, data_file, run_metadata
from scoring.demographic_experiment import DemographicBiasExperiment
from substrates.domains import DOMAINS, get_domain
from probes.probe import embed_states
from runners.run_cross_marker import record_contrasts
from scoring.intervals import DEFAULT_N_BOOT

DEFAULT_MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
DEFAULT_PROBE_RECORDS = 150
RESULTS_DIR = Path("artifacts/results/demographic")
# Short names for the output keys; `family` keeps the historical CV keys (cos_sex_family, ...).
_SHORT = {"family_status": "family", "marital_status": "marital", "economic_status": "economic"}


def _unit_rows(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def _matrices(contrasts: Mapping[str, Mapping[str, torch.Tensor]], axes: Sequence[str]):
    records = sorted(set.intersection(*(set(contrasts[a]) for a in axes)))
    return records, {a: torch.stack([contrasts[a][r] for r in records]).double().numpy() for a in axes}


def threeway_residual(contrasts: Mapping[str, Mapping[str, torch.Tensor]], marginal_axes: Sequence[str],
                      n_flips: int = 10_000, seed: int = 0) -> Dict[str, float]:
    """The corner against the vector sum of the marginal contrasts (module docstring): the cosine (a point value),
    the residual's raw, noise-floor and debiased shares of the corner's length, and the sign-flip p-value of "no
    three-way term". ``contrasts[axis][record]`` is the record's mean pair contrast; only records present on every
    axis are used."""
    axes = list(marginal_axes) + ["intersection"]
    records, mats = _matrices(contrasts, axes)
    n = len(records)
    corner = mats["intersection"].mean(0)
    total = sum(mats[a] for a in marginal_axes).mean(0)
    resid = mats["intersection"] - sum(mats[a] for a in marginal_axes)               # (n, d), one row per record
    rbar = resid.mean(0)
    observed = float(rbar @ rbar)
    floor = float(np.sum(np.var(resid, axis=0, ddof=1))) / n                          # E‖r̄‖² with no three-way term
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, n_flips, 500):                                              # chunks bound the memory
        signs = rng.choice((-1.0, 1.0), size=(min(500, n_flips - start), n))
        exceed += int(np.sum(np.sum((signs @ resid / n) ** 2, axis=1) >= observed))
    length = float(np.linalg.norm(corner))
    return {"cos_intersection_vs_marginal_sum": float(corner @ total / (length * np.linalg.norm(total))),
            "residual_share_raw": float(np.sqrt(observed)) / length,
            "noise_floor_share": float(np.sqrt(floor)) / length,
            "residual_share_debiased": float(np.sqrt(max(0.0, observed - floor))) / length,
            "sign_flip_p": (1 + exceed) / (1 + n_flips), "n_flips": n_flips, "n_records": n}


def pairwise_intervals(contrasts: Mapping[str, Mapping[str, torch.Tensor]], marginal_axes: Sequence[str],
                       n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """The pairwise cosines of the marginal contrasts (difference-of-means directions, as `build_probe_direction`
    fits them: every record has the same number of pairs per axis), each with a record-bootstrap 95% interval (the
    directions refitted on every resample)."""
    records, mats = _matrices(contrasts, marginal_axes)
    k = len(records)
    draws = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    weights = np.stack([np.bincount(row, minlength=k) for row in draws]) / k        # (n_boot, k)
    short = [_SHORT.get(a, a) for a in marginal_axes]

    def cosines(w: np.ndarray) -> Dict[str, np.ndarray]:
        dirs = {a: _unit_rows(w @ mats[a]) for a in marginal_axes}
        return {f"cos_{short[i]}_{short[j]}": np.sum(dirs[marginal_axes[i]] * dirs[marginal_axes[j]], -1)
                for i in range(len(marginal_axes)) for j in range(i + 1, len(marginal_axes))}

    point, boot = cosines(np.full((1, k), 1.0 / k)), cosines(weights)
    return {name: {"estimate": float(point[name][0]), "ci_low": float(np.percentile(boot[name], 2.5)),
                   "ci_high": float(np.percentile(boot[name], 97.5)), "n_records": k}
            for name in point}


def contrast_lengths(contrasts: Mapping[str, Mapping[str, torch.Tensor]], axes: Sequence[str]) -> Dict[str, float]:
    """‖mean contrast‖ of each axis: how far apart its two poles are in state space (what the unit sum ignored)."""
    _, mats = _matrices(contrasts, axes)
    return {a: float(np.linalg.norm(mats[a].mean(0))) for a in axes}


def default_out(domain: str, model_path: str, encoding: str) -> Path:
    return RESULTS_DIR / f"additivity_{domain}_{Path(model_path).name}_{encoding}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="credit", choices=sorted(DOMAINS))
    ap.add_argument("--pairs", default=None, help="Pairs manifest (defaults per --domain)")
    ap.add_argument("--encoding", default="explicit", choices=["explicit", "proxy"])
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="Bootstrap resamples (pairwise cosines)")
    ap.add_argument("--n-flips", type=int, default=10_000, help="Sign flips for the three-way test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"Default {RESULTS_DIR}/additivity_{{domain}}_{{model}}_{{encoding}}.json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result")
    add_override_args(ap)   # --model (default Qwen3-0.6B), --revision, --batch-size, --device, --probe-records
    return ap


def main() -> None:
    args = build_parser().parse_args()
    spec = get_domain(args.domain)
    marginal_axes = [a for a in spec.axes if a != "intersection"]
    if "intersection" not in spec.axes or len(marginal_axes) != 3:
        raise SystemExit(f"domain {spec.name!r} has no three-marginal intersection design")
    if spec.name == "credit" and args.encoding != "explicit":
        raise SystemExit("credit additivity is explicit only: marital status has no proxy encoding")
    pairs_path = args.pairs or spec.default_pairs
    cfg = apply_overrides(ExperimentConfig(name="additivity", bias_type="demographic", model_path=DEFAULT_MODEL,
                                           dataset_source=pairs_path, probe_records=DEFAULT_PROBE_RECORDS,
                                           max_length=args.max_length), args)
    # everything that can fail on the inputs fails here, before the model loads
    out = args.out or default_out(spec.name, cfg.model_path, args.encoding)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or --out")
    data = {"pairs.jsonl": data_file(pairs_path)}

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    contrasts: Dict[str, Dict[str, torch.Tensor]] = {}
    splits = {}
    for axis in marginal_axes + ["intersection"]:
        ds = spec.dataset_cls(pairs_path, axis=axis, encoding=args.encoding,
                              probe_records=cfg.probe_records, split_seed=cfg.split_seed)
        pairs = ds.get_probe_pairs(exp.tokenizer)
        pos, _ = embed_states(exp.model, exp.tokenizer, [p.positive_text for p in pairs],
                              batch_size=cfg.batch_size, max_length=cfg.max_length, show_progress=False)
        neg, _ = embed_states(exp.model, exp.tokenizer, [p.negative_text for p in pairs],
                              batch_size=cfg.batch_size, max_length=cfg.max_length, show_progress=False)
        ids, rows = record_contrasts(pairs, pos, neg)
        contrasts[axis] = dict(zip(ids, rows))
        splits[axis] = ds.split_report()
        print(f"  {axis:16} probe: {len(pairs)} pairs / {len(ids)} records")

    three = threeway_residual(contrasts, marginal_axes, args.n_flips, args.seed)
    pairwise = pairwise_intervals(contrasts, marginal_axes, args.n_boot, args.seed)
    lengths = contrast_lengths(contrasts, marginal_axes + ["intersection"])
    short = [_SHORT.get(a, a) for a in marginal_axes]
    print("\n" + "=" * 72)
    print(f"ADDITIVITY ({spec.name}, {args.encoding}, {cfg.model_path}; {three['n_records']} records)")
    print("=" * 72)
    print(f"cos(corner, {'+'.join(short)} vector sum) = {three['cos_intersection_vs_marginal_sum']:.4f}  (point value)")
    print(f"three-way residual, share of the corner: raw {three['residual_share_raw']:.4f} | noise floor "
          f"{three['noise_floor_share']:.4f} | debiased {three['residual_share_debiased']:.4f} | "
          f"sign-flip p {three['sign_flip_p']:.4f} ({three['n_flips']} flips)")
    print("contrast lengths: " + "  ".join(f"{_SHORT.get(a, a)} {v:.3f}" for a, v in lengths.items()))
    print("pairwise marginal cosines (overlap):")
    for key, v in pairwise.items():
        print(f"  {key[4:].replace('_', '·'):18} = {v['estimate']:+.4f}  95% [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")
    print("reading: the corner sees main effects and the THREE-WAY term only (two-way interactions cancel in it);")
    print("two-way intersectionality is read from the cross-marker design's interactions, not from this cosine.")
    print("=" * 72)

    settings = {"domain": spec.name, "encoding": args.encoding, "marginal_axes": marginal_axes,
                "n_boot": args.n_boot, "n_flips": args.n_flips, "seed": args.seed}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": run_metadata(cfg, data, settings),
        "model": cfg.model_path, "domain": spec.name, "encoding": args.encoding, "marginal_axes": marginal_axes,
        "probe_records": cfg.probe_records, "probe_splits": splits,
        "cos_intersection_vs_marginal_sum": three["cos_intersection_vs_marginal_sum"], "threeway": three,
        "contrast_lengths": lengths, **{k: v["estimate"] for k, v in pairwise.items()}, "pairwise_intervals": pairwise,
    }, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
