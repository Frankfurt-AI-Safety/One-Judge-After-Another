#!/usr/bin/env python3
"""
Reasoning-flip 2×2: does the RM reward factually-CORRECT reasoning, or the conclusion?

For a strong applicant carrying a premise (parental-leave / intersection / a non-demographic commute
control), score four matched verdicts crossing {causal-claim correctness} × {conclusion}:
    true_reject   — "leave reduces availability, so do not advance"   (sound, harmful = the discriminatory verdict)
    true_advance  — "leave reduces availability, but advance anyway"
    false_advance — "leave INCREASES availability, so advance"        (unsound, favorable — logical flip i)
    false_reject  — "leave INCREASES availability, but do not advance" (logical flip ii)
and report the 2×2 factorial effects (correctness vs conclusion) baseline + null-space-projected.

If the **correctness effect** matches on the non-demographic commute control, the RM is doing general
causal-reasoning checking; if it's larger on the demographic premises, it's demographic-specific.

Usage:
    python experiments/run_reasoning_flip.py --config configs/demographic_cv_reasoning_qwen06.yaml \
        --n-items 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scoring.dataset_base import format_conversation
from substrates.domains import get_domain
from pairs.verdicts import REASONING_CELLS, build_reasoning_item
from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment, compute_reasoning_metrics
from probes.probe import build_probe_direction, get_rewards_both

# premise → axis whose difference-of-means probe to null with (None = baseline only, e.g. the control)
PREMISES = {"parental_leave": "family_status", "intersection": "intersection", "commute": None}


def _probe(exp, cfg, dom, axis):
    ds = dom.dataset_cls(cfg.dataset_source, axis=axis, encoding="explicit",
                         probe_size=cfg.probe_size, split_seed=cfg.split_seed)
    probe, _ = build_probe_direction(exp.model, exp.tokenizer, ds.get_probe_pairs(exp.tokenizer),
                                     batch_size=cfg.batch_size, device=cfg.device, max_length=cfg.max_length)
    return probe


def run_premise(exp, cfg, dom, premise, null_axis, records, rng) -> Dict[str, Any]:
    tok = exp.tokenizer
    tids = list(dom.template_ids)
    items = [build_reasoning_item(r, premise, dom.render_fn, rng, template_id=tids[i % len(tids)])
             for i, r in enumerate(records)]
    texts = {c: [format_conversation(tok, it["user_prompt"], it["cells"][c]) for it in items]
             for c in REASONING_CELLS}
    n = len(items)
    flat = [t for c in REASONING_CELLS for t in texts[c]]

    # baseline scores are probe-independent; for the control premise we build a dummy probe
    # (family_status) only to obtain the baseline and we do NOT report its nulled scores.
    probe = _probe(exp, cfg, dom, null_axis or "family_status")
    base, nulled = get_rewards_both(exp.model, tok, flat, probe, batch_size=cfg.batch_size,
                                    device=cfg.device, max_length=cfg.max_length,
                                    null_alpha=1.0, show_progress=False)
    base_by = {c: base[i * n:(i + 1) * n].tolist() for i, c in enumerate(REASONING_CELLS)}
    out = {"premise": premise, "demographic": items[0]["meta"]["demographic"], "n_items": n,
           "baseline": compute_reasoning_metrics(base_by)}
    if null_axis is not None:
        null_by = {c: nulled[i * n:(i + 1) * n].tolist() for i, c in enumerate(REASONING_CELLS)}
        out["nulled"] = compute_reasoning_metrics(null_by)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_cv_reasoning_qwen06.yaml"))
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="Defaults to artifacts/results/demographic/reasoning_{domain}_qwen06.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    dom = get_domain(cfg.extra.get("domain", "credit"))
    out = args.out or Path(f"artifacts/results/demographic/reasoning_{dom.name}_qwen06.json")
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    # strong/qualified applicants only → the sole reason to reject is the availability claim
    records = [r for r in dom.load_records() if getattr(r, "qualified", True)]
    random.Random(args.seed).shuffle(records)
    records = records[:args.n_items]

    results = []
    for premise, null_axis in PREMISES.items():
        print(f"[reasoning] {dom.name}/{premise} ...", flush=True)
        results.append(run_premise(exp, cfg, dom, premise, null_axis, records, random.Random(args.seed)))

    print("\n" + "=" * 100)
    print(f"REASONING-FLIP 2×2 [{dom.name}] — {cfg.model_path}  (n_items={len(records)})")
    print("=" * 100)
    print(f"{'premise':16} {'demog':>6} | {'correct_eff':>11} {'concl_eff':>10} {'interact':>9} | "
          f"{'C1>C2_rate':>10} {'gap_C1-C2':>10}")
    for r in results:
        b = r["baseline"]
        print(f"{r['premise']:16} {str(r['demographic']):>6} | {b['correctness_effect']:>11.3f} "
              f"{b['conclusion_effect']:>10.3f} {b['interaction']:>9.3f} | "
              f"{b['prefers_correct_over_favorable_rate']:>10.3f} {b['gap_correct_minus_favorable']:>10.3f}")
    # demographic-vs-control contrast on the correctness effect
    eff = {r["premise"]: r["baseline"]["correctness_effect"] for r in results}
    if "commute" in eff:
        for p in ("parental_leave", "intersection"):
            if p in eff:
                print(f"  demographic-specific correctness ({p} − commute) = {eff[p] - eff['commute']:+.3f}")
    print("=" * 100)
    print("correct_eff>0 ⇒ RM rewards factually-correct reasoning; concl_eff>0 ⇒ rewards 'advance'.")
    print("C1>C2_rate = P(reward(correct-harmful) > reward(wrong-favorable)).")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": cfg.model_path, "domain": dom.name, "seed": args.seed,
                               "results": results}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
