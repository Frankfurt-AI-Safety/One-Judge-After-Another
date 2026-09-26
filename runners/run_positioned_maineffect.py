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

Every quantity also gets a 95% interval from a bootstrap over **essays** (an essay's positioned pairs
share the essay; `scoring/intervals.py`), in ``intervals``.

Read across axes: if the demographic `identity_gap` is large while the *genuinely neutral* controls
(pos_ctrl_hobby/pet/region) are ~0, the effect is demographic-specific. If the neutral controls also show a
large gap, it is generic persona-sensitivity. `main_fx` says whether the RM simply likes (or dislikes) a
personal-experience appeal regardless of who makes it.

Usage:
    python runners/run_positioned_maineffect.py --config configs/demographic_edupos_qwen06.yaml \
        --source persuade --axes pos_sex,pos_race,pos_class,pos_origin,pos_intersection,\
pos_control,pos_ctrl_hobby,pos_ctrl_pet,pos_ctrl_region --n-essays 200 --position conclusion
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scoring.dataset_base import ContrastivePair, format_conversation
from scoring.intervals import DEFAULT_N_BOOT, cluster_bootstrap, clusters_of
from substrates.domains import get_domain
from substrates.education_clean import load_education_essays
from substrates.education_render import EDU_TEMPLATES
from pairs.factorial import pair_suffix, stable_rng
from pairs.positionality import (
    DEFAULT_HEADER_TEMPLATE,
    POSITIONED_AXES,
    POSITION_VARIANTS,
    POSITIONS,
    STANCES,
    make_positioned_pairs,
    render_neutral,
    stance_of,
    variants_for,
)
from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment
from probes.probe import build_probe_direction, get_rewards_both

SOURCES = ("persuade", "asap")


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


# items are (reward_neutral, reward_a, reward_b) of one positioned pair (the neutral is its essay's)
POSITIONED_STATS = {
    "delta_a": lambda s: _mean([a - n for n, a, _ in s]),
    "delta_b": lambda s: _mean([b - n for n, _, b in s]),
    "main_effect": lambda s: _mean([(a + b) / 2 - n for n, a, b in s]),
    "identity_gap": lambda s: _mean([a - b for _, a, b in s]),
    "auto_influence": lambda s: 2 * abs(_mean([1.0 if a > b else 0.0 for _, a, b in s]) - 0.5),
}


def positioned_intervals(r_neu: List[float], r_a: List[float], r_b: List[float], owner: List[int],
                         n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> Dict[str, Any]:
    """Essay-bootstrap 95% intervals of the decomposition (pair k belongs to essay ``owner[k]``)."""
    items = [(r_neu[owner[k]], r_a[k], r_b[k]) for k in range(len(r_a))]
    return cluster_bootstrap(clusters_of(items, owner), POSITIONED_STATS, n_boot, seed)


def run_axis(exp, cfg, dom, essays, axis, position, seed, variant=None,
             header_template=DEFAULT_HEADER_TEMPLATE) -> Dict[str, Any]:
    """Score one axis at one position. A per-attribute axis has 4 pairs per essay (one per setting of the
    other two attributes); each is compared with that essay's single neutral rendering, and the gaps
    are reported both averaged (``identity_gap``, the marginal, as A1 averages it) and per setting
    (``by_cell``, for stratified analysis)."""
    tok = exp.tokenizer
    fmt = lambda txt: format_conversation(tok, dom.assessment_prompt, txt)
    # Same header as the pairs, so main_fx measures the positioned sentence, not the frame.
    neutral = [fmt(render_neutral(rec, header_template)) for rec in essays]
    a, b, owner, cells, probe_pairs, identities = [], [], [], [], [], []
    for i, rec in enumerate(essays):
        rng = stable_rng(seed, rec.source_record_id, axis, position, variant)
        for pair in make_positioned_pairs(rec, axis, position, rng, variant=variant,
                                          header_template=header_template):
            a.append(fmt(pair.text_a))
            b.append(fmt(pair.text_b))
            owner.append(i)
            cells.append(pair_suffix(pair.intersectional_cell) if pair.intersectional_cell else "all")
            identities.append((pair.exemplar["identity_a"], pair.exemplar["identity_b"]))
            probe_pairs.append(ContrastivePair(positive_text=a[-1], negative_text=b[-1],
                                               metadata={"axis": axis}))
    probe, meta = build_probe_direction(exp.model, tok, probe_pairs, batch_size=cfg.batch_size,
                                        device=cfg.device, max_length=cfg.max_length)
    n, m = len(essays), len(a)
    base, nulled = get_rewards_both(exp.model, tok, neutral + a + b, probe, batch_size=cfg.batch_size,
                                    device=cfg.device, max_length=cfg.max_length, null_alpha=1.0,
                                    show_progress=False)
    base = base.tolist()
    r_neu, r_a, r_b = base[:n], base[n:n + m], base[n + m:]
    d_a = [r_a[k] - r_neu[owner[k]] for k in range(m)]
    d_b = [r_b[k] - r_neu[owner[k]] for k in range(m)]
    delta_a, delta_b = _mean(d_a), _mean(d_b)
    pref_a = _mean([1.0 if x > y else 0.0 for x, y in zip(r_a, r_b)])
    by_cell: Dict[str, float] = {}
    for c in sorted(set(cells)):
        idx = [k for k in range(m) if cells[k] == c]
        by_cell[c] = _mean([d_a[k] - d_b[k] for k in idx])
    disp = variant or f"pos_{position}"
    return {
        "axis": axis, "position": position, "variant": disp, "stance": stance_of(disp),
        "n": n, "n_pairs": m,
        "probe_accuracy": meta.get("probe_accuracy"),
        # the first pair's identities, plus every distinct pair for the per-attribute axes
        "identity_a": identities[0][0], "identity_b": identities[0][1],
        "identities": sorted(set(identities)),
        "mean_neutral": _mean(r_neu), "mean_a": _mean(r_a), "mean_b": _mean(r_b),
        "delta_a": delta_a, "delta_b": delta_b,
        "main_effect": (delta_a + delta_b) / 2, "identity_gap": delta_a - delta_b,
        "by_cell": by_cell,
        "auto_influence": 2 * abs(pref_a - 0.5), "pref_a": pref_a,
        "frac_a_above_neutral": _mean([1.0 if d > 0 else 0.0 for d in d_a]),
        "frac_b_above_neutral": _mean([1.0 if d > 0 else 0.0 for d in d_b]),
        "intervals": positioned_intervals(r_neu, r_a, r_b, owner, int(cfg.extra.get("n_boot", DEFAULT_N_BOOT)),
                                          seed),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_edupos_qwen06.yaml"))
    ap.add_argument("--source", choices=SOURCES, default="persuade")
    ap.add_argument("--raw-path", default=None)
    ap.add_argument("--axes", default=",".join(POSITIONED_AXES))
    ap.add_argument("--positions", default="conclusion",
                    help=f"Comma-separated; any of {POSITIONS}.")
    ap.add_argument("--paraphrase", choices=["off", "sample", "per"], default="off",
                    help="off=base wording; sample=rng paraphrase per essay; per=each paraphrase separately.")
    ap.add_argument("--stance", choices=list(STANCES), default="endorse",
                    help="endorse=agrees with the essay (default); neutral=non-committal grounding; "
                         "both=base-endorse vs neutral head-to-head. neutral/both ignore --paraphrase.")
    ap.add_argument("--header-template", choices=sorted(EDU_TEMPLATES), default=DEFAULT_HEADER_TEMPLATE,
                    help="A1 submission shell for the neutral and positioned texts alike.")
    ap.add_argument("--n-essays", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    dom = get_domain(cfg.extra.get("domain", "education"))
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]
    # The shared education pool: the same essays as the A1 factorial and stage designs.
    essays = load_education_essays(args.raw_path, source=args.source, n=args.n_essays, seed=args.seed)
    out = args.out or Path(f"artifacts/results/demographic/maineffect_edupos_{args.source}_qwen06.json")

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

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
                results.append(run_axis(exp, cfg, dom, essays, ax, position, args.seed, variant,
                                        args.header_template))
                print(f"[maineffect] {position}/{variant or 'base'}/{ax} done", flush=True)

    print("\n" + "=" * 118)
    print(f"POSITIONED MAIN-EFFECT [{args.source}; stance={args.stance}; paraphrase={args.paraphrase}] "
          f"— {cfg.model_path} (n={len(essays)})")
    print("=" * 118)
    print(f"{'axis':18} {'position':10} {'stance':8} {'variant':24} {'mean_neu':>8} {'main_fx':>8} "
          f"{'id_gap':>8} {'auto_AI':>8}  id_gap 95% (essay bootstrap)")
    for r in results:
        ci = r["intervals"]["identity_gap"]
        print(f"{r['axis']:18} {r['position']:10} {r['stance']:8} {r['variant']:24} {r['mean_neutral']:>8.3f} "
              f"{r['main_effect']:>8.3f} {r['identity_gap']:>8.3f} {r['auto_influence']:>8.3f}  "
              f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]")
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
