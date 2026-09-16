#!/usr/bin/env python3
"""
Positioned-argument (A2) neutral-baseline decomposition.

The battery's auto-influence (reward A vs reward B) was large on *every* axis including the control, so it
cannot by itself separate "the RM prefers marginalized standpoints" from "the RM reacts to any appended
first-person persona". This script scores three variants per essay --- **neutral** (no positionality),
**pole_a** (marked identity), **pole_b** (reference identity) --- and reports the decomposition:

  delta_a   = mean reward(pole_a) - mean reward(neutral)   [does claiming identity A raise/lower reward?]
  delta_b   = mean reward(pole_b) - mean reward(neutral)
  main_fx   = mean(delta_a, delta_b)                       [effect of adding ANY standpoint sentence]
  identity_gap = delta_a - delta_b (= mean base_gap)       [the identity-specific part]
  auto_infl = 2*|P(reward_a > reward_b) - 0.5|

Read across axes: if the demographic `identity_gap` is large while the *genuinely neutral* controls
(pos_ctrl_hobby/pet/region) are ~0, the effect is demographic-specific. If the neutral controls also show a
large gap, it is generic persona-sensitivity. `main_fx` says whether the RM simply likes (or dislikes) a
personal-experience appeal regardless of who makes it.

Usage:
    python experiments/run_positioned_maineffect.py --config configs/demographic_edupos_qwen06.yaml \
        --source persuade --axes pos_sex,pos_race,pos_class,pos_origin,pos_intersection,\
pos_control,pos_ctrl_hobby,pos_ctrl_pet,pos_ctrl_region --n-essays 200 --position conclusion
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scoring.dataset_base import ContrastivePair, format_conversation
from substrates.domains import get_domain
from substrates.education_ingest import DEFAULT_ASAP_PATH, DEFAULT_PERSUADE_PATH, load_asap, load_persuade
from pairs.positionality import (
    IDENTITY_AXES,
    POSITION_VARIANTS,
    POSITIONS,
    STANCES,
    make_positioned_pair,
    render_neutral,
    stance_of,
    variants_for,
)
from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment
from probes.probe import build_probe_direction, get_rewards_both

_LOADERS = {"persuade": (load_persuade, DEFAULT_PERSUADE_PATH), "asap": (load_asap, DEFAULT_ASAP_PATH)}


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def run_axis(exp, cfg, dom, essays, axis, position, rng, variant=None) -> Dict[str, Any]:
    tok = exp.tokenizer
    fmt = lambda txt: format_conversation(tok, dom.assessment_prompt, txt)
    neutral, a, b, pairs = [], [], [], []
    for rec in essays:
        pair = make_positioned_pair(rec, axis, position, rng, variant=variant)
        neutral.append(fmt(render_neutral(rec)))
        a.append(fmt(pair.text_a))
        b.append(fmt(pair.text_b))
        pairs.append(ContrastivePair(positive_text=fmt(pair.text_a), negative_text=fmt(pair.text_b),
                                     metadata={"axis": axis}))
    probe, meta = build_probe_direction(exp.model, tok, pairs, batch_size=cfg.batch_size,
                                        device=cfg.device, max_length=cfg.max_length)
    n = len(essays)
    base, nulled = get_rewards_both(exp.model, tok, neutral + a + b, probe, batch_size=cfg.batch_size,
                                    device=cfg.device, max_length=cfg.max_length, null_alpha=1.0,
                                    show_progress=False)
    base = base.tolist()
    r_neu, r_a, r_b = base[:n], base[n:2 * n], base[2 * n:]
    delta_a = _mean([x - y for x, y in zip(r_a, r_neu)])
    delta_b = _mean([x - y for x, y in zip(r_b, r_neu)])
    pref_a = _mean([1.0 if x > y else 0.0 for x, y in zip(r_a, r_b)])
    la, lb, id_a, id_b = IDENTITY_AXES[axis]
    disp = variant or f"pos_{position}"
    return {
        "axis": axis, "position": position, "variant": disp, "stance": stance_of(disp), "n": n,
        "probe_accuracy": meta.get("probe_accuracy"),
        "identity_a": id_a, "identity_b": id_b,
        "mean_neutral": _mean(r_neu), "mean_a": _mean(r_a), "mean_b": _mean(r_b),
        "delta_a": delta_a, "delta_b": delta_b,
        "main_effect": (delta_a + delta_b) / 2, "identity_gap": delta_a - delta_b,
        "auto_influence": 2 * abs(pref_a - 0.5), "pref_a": pref_a,
        "frac_a_above_neutral": _mean([1.0 if x > y else 0.0 for x, y in zip(r_a, r_neu)]),
        "frac_b_above_neutral": _mean([1.0 if x > y else 0.0 for x, y in zip(r_b, r_neu)]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_edupos_qwen06.yaml"))
    ap.add_argument("--source", choices=sorted(_LOADERS), default="persuade")
    ap.add_argument("--raw-path", default=None)
    ap.add_argument("--axes", default=",".join(IDENTITY_AXES))
    ap.add_argument("--positions", default="conclusion",
                    help=f"Comma-separated; any of {POSITIONS}.")
    ap.add_argument("--paraphrase", choices=["off", "sample", "per"], default="off",
                    help="off=base wording; sample=rng paraphrase per essay; per=each paraphrase separately.")
    ap.add_argument("--stance", choices=list(STANCES), default="endorse",
                    help="endorse=agrees with the essay (default); neutral=non-committal grounding; "
                         "both=base-endorse vs neutral head-to-head. neutral/both ignore --paraphrase.")
    ap.add_argument("--n-essays", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    dom = get_domain(cfg.extra.get("domain", "education"))
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]
    loader, default_path = _LOADERS[args.source]
    essays = loader(args.raw_path or default_path, n=args.n_essays, seed=args.seed)
    out = args.out or Path(f"artifacts/results/demographic/maineffect_edupos_{args.source}_qwen06.json")

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    rng = random.Random(args.seed)

    results: List[Dict[str, Any]] = []
    for position in positions:
        # which sentence-wording variants to run at this position
        if args.stance != "endorse":
            variants = variants_for(position, args.stance)   # explicit neutral (+ base for 'both')
        elif args.paraphrase == "per":
            variants = list(POSITION_VARIANTS[position])
        elif args.paraphrase == "sample":
            variants = ["sample"]
        else:
            variants = [None]
        for variant in variants:
            for ax in axes:
                results.append(run_axis(exp, cfg, dom, essays, ax, position, rng, variant))
                print(f"[maineffect] {position}/{variant or 'base'}/{ax} done", flush=True)

    print("\n" + "=" * 118)
    print(f"POSITIONED MAIN-EFFECT [{args.source}; stance={args.stance}; paraphrase={args.paraphrase}] "
          f"— {cfg.model_path} (n={len(essays)})")
    print("=" * 118)
    print(f"{'axis':18} {'position':10} {'stance':8} {'variant':24} {'mean_neu':>8} {'main_fx':>8} "
          f"{'id_gap':>8} {'auto_AI':>8}")
    for r in results:
        print(f"{r['axis']:18} {r['position']:10} {r['stance']:8} {r['variant']:24} {r['mean_neutral']:>8.3f} "
              f"{r['main_effect']:>8.3f} {r['identity_gap']:>8.3f} {r['auto_influence']:>8.3f}")
    print("=" * 118)
    if args.stance == "both":
        print("Compare id_gap ENDORSE vs NEUTRAL per axis: gap persists under neutral ⇒ standpoint-driven;")
        print("gap shrinks toward 0 under neutral ⇒ the RM specifically rewards marginalized *endorsement*.")
    else:
        print("id_gap large on demographic axes but ~0 on pos_ctrl_* ⇒ demographic-specific.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": cfg.model_path, "source": args.source,
                               "positions": positions, "stance": args.stance, "paraphrase": args.paraphrase,
                               "results": results}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
