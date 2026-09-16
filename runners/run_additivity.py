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
    python runners/run_additivity.py --domain credit --encoding explicit --probe-size 300

Credit has no proxy for marital status (see pairs/factorial.py), so credit additivity is explicit only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment
from substrates.domains import DOMAINS, get_domain
from probes.probe import build_probe_direction

# Short names for the output keys; `family` keeps the historical CV keys (cos_sex_family, ...).
_SHORT = {"family_status": "family", "marital_status": "marital"}


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a / a.norm()) @ (b / b.norm()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    ap.add_argument("--domain", default="credit", choices=sorted(DOMAINS))
    ap.add_argument("--pairs", default=None, help="Pairs manifest (defaults per --domain)")
    ap.add_argument("--encoding", default="explicit", choices=["explicit", "proxy"])
    ap.add_argument("--probe-size", type=int, default=300)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON output (feeds experiments/export_paper_numbers.py)")
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
    for axis in marginal_axes + ["intersection"]:
        ds = dataset_cls(pairs_path, axis=axis, encoding=args.encoding,
                         probe_size=args.probe_size, split_seed=cfg.split_seed)
        pairs = ds.get_probe_pairs(exp.tokenizer)
        probe, meta = build_probe_direction(exp.model, exp.tokenizer, pairs,
                                            batch_size=args.batch_size, device=args.device,
                                            max_length=args.max_length)
        probes[axis] = probe
        print(f"  {axis:14} probe: n={len(pairs)} acc={meta.get('probe_accuracy', 0):.2%} "
              f"sep={meta.get('separation', 0):.3f}")

    marg_sum = sum(probes[a] for a in marginal_axes)
    cos_inter_sum = _cos(probes["intersection"], marg_sum)
    short = [_SHORT.get(a, a) for a in marginal_axes]
    pairwise = {f"cos_{short[i]}_{short[j]}": _cos(probes[marginal_axes[i]], probes[marginal_axes[j]])
                for i in range(3) for j in range(i + 1, 3)}

    print("\n" + "=" * 64)
    print(f"ADDITIVITY ({args.encoding}, {args.model})")
    print("=" * 64)
    print(f"cosine(intersection, {'+'.join(short)}) = {cos_inter_sum:.4f}")
    print("pairwise marginal cosines (overlap):")
    for key, val in pairwise.items():
        print(f"  {key[4:].replace('_', '·'):18} = {val:+.4f}")
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
            "cos_intersection_vs_marginal_sum": cos_inter_sum,
            **pairwise,
            "verdict": verdict,
        }, indent=2))
        print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
