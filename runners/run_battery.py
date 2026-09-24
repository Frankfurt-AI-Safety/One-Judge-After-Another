#!/usr/bin/env python3
"""
Robustness battery for the demographic credit bias on ONE RM (default Qwen3-0.6B).

Loads the model once (auto→CUDA) and, reusing the existing probe machinery, reports for every
(axis × encoding) cell: probe quality + baseline vs fully-nulled **auto-influence**, plus a
**per-template** breakdown and a **null_alpha sweep** curve for sex & intersection. The point is to
check the striking baseline result is a *real attribute signal* (explicit → proxy should shrink but
not vanish; effect holds across templates) and a *low-complexity* one (auto-influence drops sharply
toward 0 as α→1), before scaling to other RMs.

Direct-scoring arm = the mechanism layer (methodology decision 2026-09-24): its numbers show that the
reward is sensitive to protected attributes under controlled substitution, not that the RM assesses
applicants in a biased way. The harm evidence is `runners/run_cross_marker.py`, whose placement check
scores these same cells with the marker in the prompt.

Usage:
    python experiments/run_demographic_battery.py --config configs/demographic_credit_sex_qwen06.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment, compute_auto_influence_metrics
from substrates.domains import get_domain
from probes.embedding_cache import CACHE_ATTR
from probes.probe import build_probe_direction, get_embeddings, get_rewards_both, rewards_from_hidden

ENCODINGS = ["explicit", "proxy"]
SWEEP_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SWEEP_AXES = ["sex", "intersection"]
# Default axes come from the domain spec (markers are already baked into pairs.jsonl).
DOMAIN_SWEEP = {"education": ["grade_level", "sex"]}


def _subgroup_auto_influence(base_org, eval_examples, key="template_id") -> Dict[str, float]:
    groups = sorted({e.metadata.get(key) for e in eval_examples})
    out = {}
    for g in groups:
        idx = [i for i, e in enumerate(eval_examples) if e.metadata.get(key) == g]
        sub = {"a": [base_org["a"][i] for i in idx], "b": [base_org["b"][i] for i in idx]}
        out[str(g)] = compute_auto_influence_metrics(sub).get("auto_influence", float("nan"))
    return out


def run_cell(exp, cfg, axis, encoding, dataset_cls, sweep_axes=SWEEP_AXES) -> Dict[str, Any]:
    ds = dataset_cls(cfg.dataset_source, axis=axis, encoding=encoding,
                     probe_size=cfg.probe_size, split_seed=cfg.split_seed,
                     max_test_examples=cfg.max_test_examples, probe_records=cfg.probe_records)
    probe, meta = build_probe_direction(exp.model, exp.tokenizer, ds.get_probe_pairs(exp.tokenizer),
                                        batch_size=cfg.batch_size, device=cfg.device,
                                        max_length=cfg.max_length)
    eval_examples = ds.get_eval_examples(exp.tokenizer)
    all_texts, text_meta = exp._get_all_texts_and_variants(eval_examples)
    n = len(eval_examples)
    baseline, nulled = get_rewards_both(exp.model, exp.tokenizer, all_texts, probe,
                                        batch_size=cfg.batch_size, device=cfg.device,
                                        max_length=cfg.max_length, null_alpha=1.0, show_progress=False)
    base_org = exp._organize_rewards(baseline, text_meta, n)
    null_org = exp._organize_rewards(nulled, text_meta, n)
    cell = {
        "axis": axis, "encoding": encoding, "n_eval": n,
        "probe_accuracy": meta.get("probe_accuracy"), "probe_separation": meta.get("separation"),
        "baseline": compute_auto_influence_metrics(base_org),
        "nulled": compute_auto_influence_metrics(null_org),
        "baseline_by_template": _subgroup_auto_influence(base_org, eval_examples),
    }
    # α-sweep: the texts' states come from the embedding cache; only the head runs per α.
    if axis in sweep_axes:
        cell["alpha_sweep"] = _alpha_sweep(exp, cfg, all_texts, text_meta, n, probe)
    return cell


def _alpha_sweep(exp, cfg, all_texts, text_meta, n, probe) -> Dict[str, float]:
    """auto_influence at each α. One embedding pass (served from the embedding cache — these texts
    were just scored), then only the score head per α. The old fast path checked for an MLX-era
    backend method that no longer exists and silently re-ran the model once per α."""
    hidden = get_embeddings(exp.model, exp.tokenizer, all_texts, batch_size=cfg.batch_size,
                            device=cfg.device, max_length=cfg.max_length, show_progress=False)
    state_dtype = next(exp.model.parameters()).dtype
    cache = getattr(exp.model, CACHE_ATTR, None)
    if cache is not None and cache.state_dtype is not None:
        state_dtype = cache.state_dtype
    curve: Dict[str, float] = {}
    for a in SWEEP_ALPHAS:
        _, scores = rewards_from_hidden(exp.model, hidden, state_dtype, probe, null_alpha=a)
        org = exp._organize_rewards(scores, text_meta, n)
        curve[str(a)] = compute_auto_influence_metrics(org).get("auto_influence", float("nan"))
    return curve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_credit_sex_qwen06.yaml"))
    ap.add_argument("--axes", default=None, help="Comma-separated axes; default is domain-appropriate.")
    ap.add_argument("--encodings", default=None,
                    help="Comma-separated encodings (default explicit,proxy). A2 uses positions here.")
    ap.add_argument("--sweep-axes", default=None, help="Comma-separated axes to alpha-sweep; overrides default.")
    ap.add_argument("--dataset-source", default=None, help="Override the matched-pair manifest (pairs.jsonl).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Defaults to artifacts/results/demographic/battery_{domain}_qwen06.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.dataset_source:
        cfg.dataset_source = args.dataset_source
    spec = get_domain(cfg.extra.get("domain", "credit"))
    axes = [a.strip() for a in args.axes.split(",")] if args.axes else list(spec.axes)
    encodings = [e.strip() for e in args.encodings.split(",")] if args.encodings else ENCODINGS
    sweep_axes = ([a.strip() for a in args.sweep_axes.split(",")] if args.sweep_axes
                  else DOMAIN_SWEEP.get(spec.name, SWEEP_AXES))
    out = args.out or Path(f"artifacts/results/demographic/battery_{spec.name}_qwen06.json")
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    cells: List[Dict[str, Any]] = []
    for axis in axes:
        for enc in encodings:
            # Only a factorial axis can be absent from a factorial; education also has single-axis
            # stage items (`grade_level`, `stage_<rung>`) on their own manifest, which pass through.
            if (spec.factorial and axis in spec.factorial.axes
                    and not spec.factorial.axis_pairs(axis, enc)):
                print(f"[battery] {spec.name}/{axis}/{enc}: no such pairs in the factorial, skipped")
                continue
            print(f"[battery] {spec.name}/{axis}/{enc} ...", flush=True)
            cells.append(run_cell(exp, cfg, axis, enc, spec.dataset_cls, sweep_axes))

    # ---- report ----
    print("\n" + "=" * 86)
    print(f"DEMOGRAPHIC ROBUSTNESS BATTERY — {cfg.model_path}")
    print("=" * 86)
    print(f"{'axis':14} {'enc':9} {'probe_acc':>9} {'base_AI':>8} {'null_AI':>8} {'base_gap':>9}  per-template(base_AI)")
    for c in cells:
        bt = "  ".join(f"{k}={v:.2f}" for k, v in c["baseline_by_template"].items())
        print(f"{c['axis']:14} {c['encoding']:9} {c['probe_accuracy']:>9.2%} "
              f"{c['baseline']['auto_influence']:>8.3f} {c['nulled']['auto_influence']:>8.3f} "
              f"{c['baseline']['mean_gap']:>9.3f}  {bt}")
    print("-" * 86)
    for c in cells:
        if "alpha_sweep" in c:
            curve = "  ".join(f"α={a}:{v:.2f}" for a, v in c["alpha_sweep"].items())
            print(f"sweep {c['axis']}/{c['encoding']}: {curve}")
    print("=" * 86)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": cfg.model_path, "domain": spec.name, "cells": cells}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
