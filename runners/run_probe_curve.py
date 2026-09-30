#!/usr/bin/env python3
"""
Probe-size curve for the direct directions, one RM — the pilot's tool for fixing ``probe_records``
(pilot-then-freeze; the rule is stated in the working notes, 2026-09-25, before any pilot run).

For every (encoding, axis) of the domain's direct manifest, the direction is fitted on the first N probe
records for each N in ``--grid``. The ``probe_records`` split is nested in N (two quality strata, largest
remainder; `ProbeDataset._record_split`), so every prefix sits inside the largest probe set, and the test
split of the largest set is one **fixed eval set**, disjoint from every prefix. Per N:

- ``split_half``: the mean cosine between directions fitted on two random halves of the N records
  (`split_half_cosine`, records as the unit) — the reliability of a direction from N/2 records;
- ``cos_to_max``: the cosine to the direction from the largest N (descriptive: the sets are nested);
- ``nulled_gap`` / ``nulled_abs_gap``: the eval pairs' reward gap (A − B) with the direction projected out,
  averaged per record and summarised over records (mean, SD, 95% bootstrap CI).

``probe_rule`` applies the pre-stated rule: a grid point passes when its split-half cosine is at least
``--threshold`` and its nulled abs gap lies inside the 95% CI of the value at the largest N; the direction's
answer is the smallest N from which every larger grid point passes too, and a direction that fails even at the
largest N is reported and answers the largest N. The domain's ``probe_records`` is the largest answer. The grid
runs to N = 500 (amendment to the rule, confirmed 2026-09-30), so the reference is the value at 500. The eval
baseline (per-record SD of the gap) is reported as well; it sizes the direct arm's evaluation.

The eval set and the largest probe set go through the model once each; every prefix's direction (the difference
of means, as `build_probe_direction`) and contrasts come from the largest set's states. The result (default
``pilot/probecurve_{domain}_{model}.json``, never replaced without ``--overwrite``) carries ``meta`` (the config
with the loaded model commit, the code commit, the manifest's SHA-256; `scoring.experiment`): ids and numbers only,
never text.

Usage:
    python runners/run_probe_curve.py --config configs/demographic_credit_crossmarker_qwen06.yaml
    python runners/run_probe_curve.py --config ... --grid 8,16 --max-eval 40 --device mps   # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from runners.run_cross_marker import PhaseTimer, record_contrast_matrix, timing_report
from scoring.experiment import add_override_args, apply_overrides, data_file, run_metadata
from scoring.cross_marker_metrics import summarize
from scoring.intervals import DEFAULT_N_BOOT

logger = logging.getLogger(__name__)

DEFAULT_GRID = (25, 50, 75, 100, 150, 200, 300, 500)
DEFAULT_THRESHOLD = 0.90


# --------------------------------------------------------------------------- rule --------------------
def probe_rule(curve: Sequence[Mapping[str, Any]], threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """The pre-stated rule on one direction's curve (sorted by N; the last entry is the largest N).
    A point passes when ``split_half >= threshold`` and its ``nulled_abs_gap`` mean lies inside the 95% CI
    of the largest N's. Returns the per-N verdicts, ``smallest_passing_n``: the smallest N from which every
    larger point passes too (None when even the largest fails), and ``answer``: that N, or the largest N for a
    direction that fails even there (the rule: it is reported, and keeps the largest N)."""
    if not curve:
        return {"passes": {}, "smallest_passing_n": None, "answer": None, "threshold": threshold}
    ref = curve[-1]["nulled_abs_gap"]
    passes: Dict[int, bool] = {}
    for point in curve:
        stable = point["split_half"] == point["split_half"] and point["split_half"] >= threshold
        gap = point["nulled_abs_gap"]["mean"]
        passes[int(point["n"])] = bool(stable and ref["ci_low"] <= gap <= ref["ci_high"])
    smallest = None
    for point in reversed(curve):
        if not passes[int(point["n"])]:
            break
        smallest = int(point["n"])
    return {"passes": passes, "smallest_passing_n": smallest,
            "answer": smallest if smallest is not None else int(curve[-1]["n"]), "threshold": threshold}


# --------------------------------------------------------------------------- eval --------------------
def per_record_gaps(examples: Sequence[Any], gaps: Sequence[float]) -> Dict[str, List[float]]:
    """The eval pairs' gaps grouped by record (a record's pairs are correlated; the record is the unit)."""
    by_record: Dict[str, List[float]] = defaultdict(list)
    for example, gap in zip(examples, gaps):
        by_record[str(example.metadata["source_record_id"])].append(float(gap))
    return dict(by_record)


def gap_summaries(by_record: Mapping[str, Sequence[float]], n_boot: int, seed: int) -> Dict[str, Any]:
    """Mean gap and mean |gap| per record, each summarised over records (mean, SD, d_z, 95% CI)."""
    mean = [sum(g) / len(g) for g in by_record.values()]
    absolute = [sum(abs(x) for x in g) / len(g) for g in by_record.values()]
    return {"gap": summarize(mean, n_boot, seed), "abs_gap": summarize(absolute, n_boot, seed)}


# --------------------------------------------------------------------------- curve -------------------
def direction_curve(model: Any, tokenizer: Any, dataset_cls: Any, source: str, axis: str, encoding: str, *,
                    grid: Sequence[int], max_eval: Optional[int], batch_size: int,
                    max_length: int, n_boot: int, seed: int, split_seed: int,
                    threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """The probe-size curve of one (encoding, axis) direction; see the module docstring."""
    from probes.cross_marker_directions import cosine, split_half_cosine, unit
    from probes.probe import embed_states, embed_with_gates, rewards_from_hidden

    grid = sorted(set(grid))
    make = lambda n: dataset_cls(source, axis=axis, encoding=encoding, split_seed=split_seed,
                                 probe_records=n, max_test_examples=max_eval)
    ds_max = make(grid[-1])
    max_ids = ds_max.probe_record_ids()
    if len(max_ids) != grid[-1]:
        raise ValueError(f"{encoding}/{axis}: N={grid[-1]} leaves too few records for evaluation "
                         f"(the split kept {len(max_ids)} probe records); lower the grid")
    examples = ds_max.get_eval_examples(tokenizer)
    eval_ids = {str(e.metadata["source_record_id"]) for e in examples}
    if eval_ids & max_ids:
        raise AssertionError(f"{encoding}/{axis}: eval records overlap the probe records")
    h_a, dtype, g_a = embed_with_gates(model, tokenizer, [e.texts["a"] for e in examples],
                                       batch_size=batch_size, max_length=max_length, show_progress=False)
    h_b, _, g_b = embed_with_gates(model, tokenizer, [e.texts["b"] for e in examples],
                                   batch_size=batch_size, max_length=max_length, show_progress=False)

    def eval_gaps(direction: Optional[Any]) -> Dict[str, Any]:
        _, r_a = rewards_from_hidden(model, h_a, dtype, direction, gates=g_a)
        _, r_b = rewards_from_hidden(model, h_b, dtype, direction, gates=g_b)
        return gap_summaries(per_record_gaps(examples, (r_a - r_b).tolist()), n_boot, seed)

    # the largest probe set's states, once; every prefix's pairs are a subset of its pairs (its records' rows)
    max_pairs = ds_max.get_probe_pairs(tokenizer)
    pos_all, _ = embed_states(model, tokenizer, [p.positive_text for p in max_pairs], batch_size=batch_size,
                              max_length=max_length, show_progress=False)
    neg_all, _ = embed_states(model, tokenizer, [p.negative_text for p in max_pairs], batch_size=batch_size,
                              max_length=max_length, show_progress=False)
    row_of = {(p.positive_text, p.negative_text): i for i, p in enumerate(max_pairs)}

    points: List[Dict[str, Any]] = []
    directions: Dict[int, Any] = {}
    for n in grid:
        ds = ds_max if n == grid[-1] else make(n)
        ids = ds.probe_record_ids()
        if len(ids) != n or not ids <= max_ids:
            raise AssertionError(f"{encoding}/{axis}: the N={n} probe set is not a prefix of N={grid[-1]}")
        pairs = ds.get_probe_pairs(tokenizer)
        rows = [row_of[(p.positive_text, p.negative_text)] for p in pairs]
        pos, neg = pos_all[rows], neg_all[rows]
        u = unit(pos.mean(0) - neg.mean(0))          # the difference of means, as build_probe_direction
        nulled = eval_gaps(u)
        directions[n] = u
        points.append({"n": n, "n_pairs": len(pairs), "split": ds.split_report(),
                       "split_half": split_half_cosine(record_contrast_matrix(pairs, pos, neg), seed),
                       "nulled_gap": nulled["gap"], "nulled_abs_gap": nulled["abs_gap"]})
    u_max = directions[grid[-1]]
    for point in points:
        point["cos_to_max"] = cosine(directions[point["n"]], u_max)
    return {"eval_records": len(eval_ids), "eval_pairs": len(examples), "baseline": eval_gaps(None),
            "curve": points, "rule": probe_rule(points, threshold)}


# --------------------------------------------------------------------------- main --------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_credit_crossmarker_qwen06.yaml"))
    ap.add_argument("--grid", default=",".join(map(str, DEFAULT_GRID)), help="Probe records per point")
    ap.add_argument("--encodings", default="explicit,proxy")
    ap.add_argument("--axes", default=None, help="Comma-separated; default every axis with pairs")
    ap.add_argument("--max-eval", type=int, default=2000, help="Eval pairs (about one per record)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default artifacts/results/demographic/pilot/probecurve_{domain}_{model}.json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result")
    add_override_args(ap)
    return ap


def default_out(domain: str, model_path: str) -> Path:
    return Path(f"artifacts/results/demographic/pilot/probecurve_{domain}_{Path(model_path).name}.json")


def domain_answer(directions: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """The domain's ``probe_records``: the largest answer over its directions (a failing direction answers the
    largest N), and the directions that fail even there."""
    answers = [d["rule"]["answer"] for d in directions.values()]
    return {"probe_records": max((a for a in answers if a is not None), default=None),
            "failing_directions": sorted(k for k, d in directions.items()
                                         if d["rule"]["smallest_passing_n"] is None)}


def main() -> None:
    from scoring.demographic_experiment import DemographicBiasExperiment
    from scoring.experiment import ExperimentConfig
    from substrates.domains import get_domain

    timer = PhaseTimer()
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    if args.probe_records is not None:
        raise SystemExit("--probe-records: the grid sets the probe records here (--grid)")
    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    dom = get_domain(cfg.extra.get("domain", "credit"))
    source = cfg.dataset_source or dom.default_pairs
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    grid = sorted({int(x) for x in split(args.grid)})
    axes = split(args.axes) if args.axes else list(dom.axes)
    unknown = sorted(set(axes) - set(dom.axes))
    if unknown:
        raise SystemExit(f"{dom.name} has no axes {unknown}; its axes are {list(dom.axes)}")
    # everything that can fail on the inputs fails here, before the model loads
    out = args.out or default_out(dom.name, cfg.model_path)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or --out")
    data = {"pairs.jsonl": data_file(source)}
    cells = [(e, a) for e in split(args.encodings) for a in axes if dom.factorial.axis_pairs(a, e)]
    for encoding, axis in cells:
        kept = dom.dataset_cls(source, axis=axis, encoding=encoding, split_seed=cfg.split_seed,
                               probe_records=grid[-1]).probe_record_ids()
        if len(kept) != grid[-1]:
            raise SystemExit(f"{encoding}/{axis}: N={grid[-1]} leaves too few records for evaluation (the split "
                             f"kept {len(kept)} probe records); lower the grid")

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    model, tok = exp.model, exp.tokenizer
    timer.lap("load_model")

    results: Dict[str, Any] = {}
    for encoding, axis in cells:
        logger.info("probe curve %s/%s over N=%s", encoding, axis, grid)
        results[f"{encoding}/{axis}"] = direction_curve(
            model, tok, dom.dataset_cls, source, axis, encoding, grid=grid, max_eval=args.max_eval,
            batch_size=cfg.batch_size, max_length=cfg.max_length,
            n_boot=args.n_boot, seed=args.seed, split_seed=cfg.split_seed, threshold=args.threshold)
        timer.lap("curves")

    cache = getattr(model, "_onejudge_embedding_cache", None)
    timing = timing_report(timer, batch_size=cfg.batch_size, texts=None)
    timing["texts_through_model"] = None if cache is None else cache.misses
    settings = {"grid": grid, "max_eval": args.max_eval, "threshold": args.threshold, "seed": args.seed,
                "n_boot": args.n_boot, "cells": [list(c) for c in cells]}
    summary = {
        "meta": run_metadata(cfg, data, settings),
        "model": cfg.model_path, "domain": dom.name, "direct_manifest": source, "grid": grid,
        "max_eval": args.max_eval, "threshold": args.threshold, "seed": args.seed,
        "split_seed": cfg.split_seed, "max_length": cfg.max_length,
        "directions": results,
        **domain_answer(results),
        "timing": timing,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print_report(summary)
    print(f"saved → {out}")


def print_report(summary: Mapping[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"PROBE-SIZE CURVE [{summary['domain']}] — {summary['model']}  (split-half ≥ {summary['threshold']}, "
          f"nulled |gap| inside the CI at N={summary['grid'][-1]})")
    for name, r in summary["directions"].items():
        cells = "  ".join(f"N={p['n']}: {p['split_half']:.3f}/{p['nulled_abs_gap']['mean']:.3f}"
                          f"{'✓' if r['rule']['passes'][p['n']] else '✗'}" for p in r["curve"])
        fails = "" if r["rule"]["smallest_passing_n"] is not None else " (fails at the largest N)"
        print(f"  {name:28} {cells}  → {r['rule']['answer']}{fails}")
    print(f"domain probe_records (max over directions): {summary['probe_records']}; failing even at "
          f"N={summary['grid'][-1]}: {summary['failing_directions'] or 'none'}")
    t = summary["timing"]
    print(f"timing: {' | '.join(f'{k} {v:.0f}s' for k, v in t['seconds'].items())} | total {t['total_s']:.0f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
