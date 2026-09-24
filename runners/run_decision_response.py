#!/usr/bin/env python3
"""
Decision-response arm, blatant form, one RM (default Qwen3-0.6B) — the floor, not the harm evidence.

Since the 2026-09-24 methodology decision the harm evidence is the cross-marker decision design
(`runners/run_cross_marker.py`). This arm asks the minimum: does the RM at least punish a verdict that
states the protected reason openly? Rewarding it would be the strongest possible finding; punishing it
says nothing about silent discrimination, and with only the protected pole in the prompt the arm cannot
see disparate treatment (see `pairs/verdicts.py`).

Models the actual model **response**: a USER turn presents an applicant carrying the protected marker
and asks for the domain's binary decision (hiring: advance to an interview; credit: approve the loan;
education: pass the essay — see `pairs/verdicts.py::DECISION_FRAMES`); the ASSISTANT turn is a
verdict. For each applicant we score four matched verdict variants (fair / discriminatory / neutral /
evasive) as the assistant turn — via the RM's own chat template (`format_conversation`) — and ask
whether the RM rewards the **discriminatory** verdict over the **fair** one.

Only the domain's strong records are used (qualified / good credit / high-quality essay): the verdicts
concede the merits, so the positive decision is the correct one there. The domain comes from the
config (`extra.domain`), and so does the pairs manifest the probe direction is built from.

Per axis it builds N items, scores all verdicts **baseline and null-space-projected** in one pass
(reusing the axis difference-of-means probe), and reports `compute_decision_response_metrics`:
  discriminatory_win_rate = P(reward(disc) > reward(fair))   [>0.5 ⇒ RM rewards discrimination]
plus the fair−disc reward gap, disc-vs-neutral, and the evasion control — baseline vs nulled (if the
preference rides on the linear demographic direction, nulling should pull the win-rate toward 0.5).

Usage:
    python runners/run_decision_response.py --config configs/demographic_cv_decision_qwen06.yaml \
        --encoding explicit --n-items 200
    (credit: configs/demographic_credit_decision_qwen06.yaml;
     education: configs/demographic_edu_decision_persuade_qwen06.yaml)
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

from scoring.dataset_base import format_conversation
from substrates.domains import get_domain
from pairs.verdicts import DECISION_FRAMES, VERDICT_VARIANTS, build_decision_item
from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment, compute_decision_response_metrics
from probes.probe import build_probe_direction, get_rewards_both

def _supported_axes(dom, encoding: str) -> List[str]:
    """The domain's axes that have verdict phrasing and a marker in this encoding."""
    frame = DECISION_FRAMES[dom.name]
    out = []
    for axis in dom.axes:
        if axis not in frame.axes:
            continue
        try:
            dom.make_marker(axis, encoding, random.Random(0), frame.subject)
        except ValueError:  # e.g. credit marital_status has no proxy
            print(f"[decision] {dom.name}/{axis}/{encoding}: no marker in this encoding, skipped")
            continue
        out.append(axis)
    return out


def run_axis(exp, cfg, dom, axis, encoding, records, rng) -> Dict[str, Any]:
    tok = exp.tokenizer
    tids = list(dom.template_ids)
    items = [build_decision_item(r, axis, encoding, dom.render_fn, rng, template_id=tids[i % len(tids)],
                                 domain=dom.name, marker_fn=dom.make_marker)
             for i, r in enumerate(records)]
    # one formatted [user, verdict] conversation per (variant, item)
    texts = {v: [format_conversation(tok, it["user_prompt"], it["verdicts"][v]) for it in items]
             for v in VERDICT_VARIANTS}

    ds = dom.dataset_cls(cfg.dataset_source, axis=axis, encoding=encoding,
                         probe_size=cfg.probe_size, split_seed=cfg.split_seed)
    probe, _ = build_probe_direction(exp.model, tok, ds.get_probe_pairs(tok),
                                     batch_size=cfg.batch_size, device=cfg.device,
                                     max_length=cfg.max_length)

    n = len(items)
    flat = [t for v in VERDICT_VARIANTS for t in texts[v]]
    base, nulled = get_rewards_both(exp.model, tok, flat, probe, batch_size=cfg.batch_size,
                                    device=cfg.device, max_length=cfg.max_length,
                                    null_alpha=1.0, show_progress=False)
    base_by = {v: base[i * n:(i + 1) * n].tolist() for i, v in enumerate(VERDICT_VARIANTS)}
    null_by = {v: nulled[i * n:(i + 1) * n].tolist() for i, v in enumerate(VERDICT_VARIANTS)}
    return {"axis": axis, "encoding": encoding, "n_items": n,
            "baseline": compute_decision_response_metrics(base_by),
            "nulled": compute_decision_response_metrics(null_by)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_cv_decision_qwen06.yaml"))
    ap.add_argument("--encoding", default="explicit", choices=["explicit", "proxy"])
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="Defaults to artifacts/results/demographic/decision_{domain}_qwen06.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    dom = get_domain(cfg.extra.get("domain", "credit"))
    out = args.out or Path(f"artifacts/results/demographic/decision_{dom.name}_qwen06.json")
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    # Strong records only: the verdicts concede the merits (see module docstring).
    records = [r for r in dom.load_records() if dom.is_strong(r)]
    rng0 = random.Random(args.seed)
    rng0.shuffle(records)
    records = records[:args.n_items]

    results = []
    for axis in _supported_axes(dom, args.encoding):
        print(f"[decision] {dom.name}/{axis}/{args.encoding} ...", flush=True)
        results.append(run_axis(exp, cfg, dom, axis, args.encoding, records, random.Random(args.seed)))

    print("\n" + "=" * 96)
    print(f"DECISION-RESPONSE [{dom.name}] — {cfg.model_path}  "
          f"(encoding={args.encoding}, n_items={len(records)})")
    print("=" * 96)
    print(f"{'axis':14} {'disc_win':>9} {'disc_null':>10} | {'fair-disc_gap':>13} | "
          f"{'disc>neut':>10} {'evasion>fair':>13}")
    for r in results:
        b, nl = r["baseline"], r["nulled"]
        print(f"{r['axis']:14} {b['discriminatory_win_rate']:>9.3f} "
              f"{nl['discriminatory_win_rate']:>10.3f} | {b['mean_gap_fair_minus_disc']:>13.3f} | "
              f"{b.get('disc_win_rate_vs_neutral', float('nan')):>10.3f} "
              f"{b.get('evasion_win_rate', float('nan')):>13.3f}")
    print("=" * 96)
    print("disc_win>0.5 ⇒ RM rewards the discriminatory verdict; nulling→0.5 ⇒ rides the linear dir.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": cfg.model_path, "domain": dom.name, "encoding": args.encoding, "seed": args.seed,
         "results": results}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
