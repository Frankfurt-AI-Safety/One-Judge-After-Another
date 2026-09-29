#!/usr/bin/env python3
"""
Positioned-argument (A2) neutral-baseline decomposition.

The battery's auto-influence (reward A vs reward B) was large on *every* axis including the control, so it
cannot by itself separate "the RM prefers marginalized standpoints" from "the RM reacts to any appended
first-person persona". This script scores three variants per essay --- **neutral** (no positionality),
**pole_a** (marked identity), **pole_b** (reference identity) --- and reports the decomposition:

  delta_a      = mean reward(pole_a) - reward(neutral)       [does claiming identity A raise/lower reward?]
  delta_b      = mean reward(pole_b) - reward(neutral)
  main_effect  = mean of delta_a and delta_b                 [effect of adding ANY standpoint sentence]
  identity_gap = delta_a - delta_b (= mean reward gap A − B)  [the identity-specific part]
  pref_a_rate  = P(reward_a > reward_b), a tie counting ½; auto_influence = 2·|pref_a_rate − ½|

Each is defined once (``POSITIONED_STATS``); its point estimate is its interval's estimate, a 95% bootstrap over
**essays** (an essay's pairs share the essay and its neutral; `scoring/intervals.py`). Read the SIGNED
``identity_gap`` interval (covering 0 or not) and ``pref_a_rate`` against ½; ``auto_influence`` is folded
(positive under noise alone). ``by_cell`` gives the gap per setting of the other two attributes.

Read across axes: if the demographic `identity_gap` is large while the *genuinely neutral* controls
(pos_ctrl_hobby/pet/region) are ~0, the effect is demographic-specific. If the neutral controls also show a
large gap, it is generic persona-sensitivity. `main_effect` says whether the RM simply likes (or dislikes) a
personal-experience appeal regardless of who makes it.

Pairs: built and gated exactly as `runners/generate_positioned.py` builds the positioned manifest
(`pairs.positionality.positioned_block`: the same seeded block per essay, axis and position, the default gate
bounds, a failing block dropped whole), from the same essays: the shared education pool narrowed to one
standpoint-fit group, ALL of the group's essays by default (``--n-essays`` caps them for a smoke run). So the
default run scores the manifest's texts. Until 2026-09-29 the random position drew its own insertion points
(93% of its pairs differed from the manifest's), the default was a 200-essay sample, and a direction fitted on the
evaluated pairs themselves fed only an in-sample probe accuracy (its nulling was computed and discarded); the
held-out nulling of A2 is the battery's, on the positioned manifest.

Run it once per standpoint-fit group (`--standpoint-fit plausible|implausible`, decided 2026-09-27): the
plausible group is the result, the implausible one the control for whether the effect needs a topic where a
standpoint could matter.

The result ``maineffect_edupos_{source}_{group}_{model}[__...].json`` names every non-default setting (positions,
stance, paraphrase, axes, essay cap), is never replaced without ``--overwrite``, and carries ``meta`` (the config
with the loaded model commit, the code commit, the corpus file's SHA-256, the settings; `scoring.experiment`).

Usage:
    python runners/run_positioned_maineffect.py --config configs/demographic_edupos_qwen06.yaml \
        --standpoint-fit plausible                   # every axis, conclusion, all 622 essays of the group
    python runners/run_positioned_maineffect.py --config ... --standpoint-fit implausible
    python runners/run_positioned_maineffect.py --config ... --n-essays 20 --device mps   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pairs.manifest import file_sha256
from scoring.dataset_base import format_conversation
from scoring.intervals import DEFAULT_N_BOOT, cluster_bootstrap, clusters_of, win
from substrates.domains import get_domain
from substrates.education_clean import load_education_essays, source_path
from substrates.education_render import EDU_TEMPLATES
from pairs.factorial import pair_suffix
from pairs.positionality import (
    DEFAULT_HEADER_TEMPLATE,
    POSITIONED_AXES,
    POSITION_VARIANTS,
    POSITIONS,
    STANCES,
    STANDPOINT_GROUPS,
    positioned_block,
    render_neutral,
    select_standpoint_essays,
    stance_of,
    variants_for,
)
from scoring.experiment import ExperimentConfig, add_override_args, apply_overrides, run_metadata
from scoring.demographic_experiment import DemographicBiasExperiment
from probes.probe import embed_with_gates, rewards_from_hidden

SOURCES = ("asap2",)
RESULTS_DIR = Path("artifacts/results/demographic")


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _pref(s) -> float:
    return _mean([win(a, b) for _, a, b in s])


# items are (reward_neutral, reward_a, reward_b) of one positioned pair (the neutral is its essay's); every
# statistic is defined here once, so a point estimate is its interval's estimate
POSITIONED_STATS = {
    "delta_a": lambda s: _mean([a - n for n, a, _ in s]),
    "delta_b": lambda s: _mean([b - n for n, _, b in s]),
    "main_effect": lambda s: _mean([(a + b) / 2 - n for n, a, b in s]),
    "identity_gap": lambda s: _mean([a - b for _, a, b in s]),
    "pref_a_rate": _pref,
    "auto_influence": lambda s: 2 * abs(_pref(s) - 0.5),
    "frac_a_above_neutral": lambda s: _mean([win(a, n) for n, a, _ in s]),
    "frac_b_above_neutral": lambda s: _mean([win(b, n) for n, _, b in s]),
}


def positioned_intervals(r_neu: List[float], r_a: List[float], r_b: List[float], owner: List[int],
                         n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> Dict[str, Any]:
    """Essay-bootstrap 95% intervals of the decomposition (pair k belongs to essay ``owner[k]``)."""
    items = [(r_neu[owner[k]], r_a[k], r_b[k]) for k in range(len(r_a))]
    return cluster_bootstrap(clusters_of(items, owner), POSITIONED_STATS, n_boot, seed)


def _rewards(exp, cfg, texts: List[Any]) -> List[float]:
    hidden, dtype, gates = embed_with_gates(exp.model, exp.tokenizer, texts, batch_size=cfg.batch_size,
                                            max_length=cfg.max_length, show_progress=False)
    return rewards_from_hidden(exp.model, hidden, dtype, None, gates=gates)[0].tolist()


def run_axis(exp, cfg, dom, essays, axis, position, seed, variant=None,
             header_template=DEFAULT_HEADER_TEMPLATE) -> Dict[str, Any]:
    """Score one axis at one position. A per-attribute axis has 4 pairs per essay (one per setting of the
    other two attributes); each is compared with that essay's single neutral rendering, and the gaps
    are reported both averaged (``identity_gap``, the marginal, as A1 averages it) and per setting
    (``by_cell``, for stratified analysis). An essay whose block is dropped (no insertion point, or the gate)
    contributes neither pairs nor its neutral."""
    fmt = lambda txt: format_conversation(exp.tokenizer, dom.assessment_prompt, txt)
    kept: List[Any] = []
    a, b, owner, cells, identities = [], [], [], [], []
    dropped: Dict[str, int] = {}
    for rec in essays:
        pairs, failures = positioned_block(rec, axis, position, seed, variant=variant, header_template=header_template)
        if not pairs:
            for code, k in failures.items():
                dropped[code] = dropped.get(code, 0) + k
            continue
        for pair in pairs:
            a.append(fmt(pair.text_a))
            b.append(fmt(pair.text_b))
            owner.append(len(kept))
            cells.append(pair_suffix(pair.intersectional_cell) if pair.intersectional_cell else "all")
            identities.append((pair.exemplar["identity_a"], pair.exemplar["identity_b"]))
        kept.append(rec)
    disp = variant or f"pos_{position}"
    out: Dict[str, Any] = {"axis": axis, "position": position, "variant": disp, "stance": stance_of(disp),
                           "n": len(kept), "n_pairs": len(a), "dropped": dropped}
    if not a:
        return out
    # Same header as the pairs, so main_effect measures the positioned sentence, not the frame.
    rewards = _rewards(exp, cfg, [fmt(render_neutral(rec, header_template)) for rec in kept] + a + b)
    n, m = len(kept), len(a)
    r_neu, r_a, r_b = rewards[:n], rewards[n:n + m], rewards[n + m:]
    intervals = positioned_intervals(r_neu, r_a, r_b, owner, int(cfg.extra.get("n_boot", DEFAULT_N_BOOT)), seed)
    by_cell = {c: _mean([r_a[k] - r_b[k] for k in range(m) if cells[k] == c]) for c in sorted(set(cells))}
    out.update({
        # the first pair's identities, plus every distinct pair for the per-attribute axes
        "identity_a": identities[0][0], "identity_b": identities[0][1], "identities": sorted(set(identities)),
        "mean_neutral": _mean(r_neu), "mean_a": _mean(r_a), "mean_b": _mean(r_b),   # levels, descriptive
        **{k: v["estimate"] for k, v in intervals.items() if k in POSITIONED_STATS},
        "by_cell": by_cell, "intervals": intervals,
    })
    return out


def default_out(source: str, group: str, model_path: str, *, positions: Sequence[str], stance: str,
                paraphrase: str, axes: Sequence[str], n_essays: Optional[int]) -> Path:
    """``maineffect_edupos_{source}_{group}_{model}``, plus every setting that differs from the default run, so
    no variant run takes the default run's name."""
    extras = []
    if list(positions) != ["conclusion"]:
        extras.append("-".join(positions))
    if stance != "endorse":
        extras.append(f"stance-{stance}")
    if paraphrase != "off":
        extras.append(f"paraphrase-{paraphrase}")
    if list(axes) != list(POSITIONED_AXES):
        extras.append("-".join(axes))
    if n_essays is not None:
        extras.append(f"n{n_essays}")
    name = f"maineffect_edupos_{source}_{group}_{Path(model_path).name}" + "".join(f"__{e}" for e in extras)
    return RESULTS_DIR / f"{name}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_edupos_qwen06.yaml"))
    ap.add_argument("--source", choices=SOURCES, default="asap2")
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
    ap.add_argument("--standpoint-fit", choices=STANDPOINT_GROUPS, default="plausible",
                    help="Which essays (pairs.positionality.STANDPOINT_FIT): the prompts where a standpoint is "
                         "plausible, or the control group; the same selection as generate_positioned.py")
    ap.add_argument("--n-essays", type=int, default=None,
                    help="Cap per group, for smoke runs (default: every essay of the group, as the manifest)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result")
    add_override_args(ap)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    dom = get_domain(cfg.extra.get("domain", "education"))
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]
    out = args.out or default_out(args.source, args.standpoint_fit, cfg.model_path, positions=positions,
                                  stance=args.stance, paraphrase=args.paraphrase, axes=axes, n_essays=args.n_essays)
    # everything that can fail on the inputs fails here, before the model loads
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or --out")
    corpus = source_path(args.source, args.raw_path)
    data = {f"{args.source}.csv": {"path": str(corpus), "sha256": file_sha256(corpus)}}
    # The shared education pool (the same essays as the A1 factorial and stage designs), narrowed to one
    # standpoint-fit group exactly as the positioned generator does.
    pool = load_education_essays(args.raw_path, source=args.source, seed=args.seed)
    essays = select_standpoint_essays(pool, args.standpoint_fit, args.seed, n=args.n_essays)

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
    print(f"POSITIONED MAIN-EFFECT [{args.source}/{args.standpoint_fit}; stance={args.stance}; "
          f"paraphrase={args.paraphrase}] — {cfg.model_path} (n={len(essays)})")
    print("=" * 118)
    print(f"{'axis':18} {'position':10} {'stance':8} {'variant':24} {'mean_neu':>8} {'main_fx':>8} "
          f"{'id_gap':>8} {'pref_a':>7}  id_gap 95% (essay bootstrap)   dropped")
    for r in results:
        if "intervals" not in r:
            print(f"{r['axis']:18} {r['position']:10} no pairs (dropped {r['dropped']})")
            continue
        ci = r["intervals"]["identity_gap"]
        print(f"{r['axis']:18} {r['position']:10} {r['stance']:8} {r['variant']:24} {r['mean_neutral']:>8.3f} "
              f"{r['main_effect']:>8.3f} {r['identity_gap']:>8.3f} {r['pref_a_rate']:>7.3f}  "
              f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]   {sum(r['dropped'].values()) or ''}")
    print("=" * 118)
    if args.stance == "both":
        print("Compare id_gap ENDORSE vs NEUTRAL per axis: gap persists under neutral ⇒ standpoint-driven;")
        print("gap shrinks toward 0 under neutral ⇒ the RM specifically rewards marginalized *endorsement*.")
    else:
        print("id_gap (signed interval) large on demographic axes but ~0 on pos_ctrl_* ⇒ demographic-specific.")

    settings = {"source": args.source, "standpoint_fit": args.standpoint_fit, "axes": axes, "positions": positions,
                "stance": args.stance, "paraphrase": args.paraphrase, "header_template": args.header_template,
                "n_essays": args.n_essays, "seed": args.seed, "n_boot": int(cfg.extra.get("n_boot", DEFAULT_N_BOOT))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": run_metadata(cfg, data, settings), "model": cfg.model_path,
                               "source": args.source, "standpoint_fit": args.standpoint_fit,
                               "n_essays": len(essays), "positions": positions, "stance": args.stance,
                               "paraphrase": args.paraphrase, "essays": [r.source_record_id for r in essays],
                               "results": results}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
