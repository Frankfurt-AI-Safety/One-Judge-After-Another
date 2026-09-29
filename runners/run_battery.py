#!/usr/bin/env python3
"""
The direct-scoring arm on ONE RM: every (axis × encoding) cell of one matched-pair manifest (credit, hiring, the
education factorial, the education stage design or an A2 positioned group).

Per cell: the difference-of-means direction from the probe records' pairs (record split, stratified by quality;
``probe_accuracy``/``probe_separation`` are **in-sample**, on those pairs), then on the held-out eval pairs the
baseline and fully nulled metrics, each with its record-bootstrap interval, what nulling changed, and the
baseline broken down by template, record quality (strong/weak) and, for education, essay prompt. An α-sweep
(the texts' cached states, only the score head per α) for the sweep axes.

Reading (`scoring/intervals.py`): whether a preference is left after nulling is read from the **signed**
intervals (``mean_gap`` covering 0, ``pref_a_rate`` covering ½); ``auto_influence`` and ``abs_mean_gap`` are
folded (positive under noise alone), so neither the nulled values nor the α-sweep's end point go to 0. The sweep
reports the signed metrics at each α too.

Direct-scoring arm = the mechanism layer (methodology decision 2026-09-24): its numbers show that the
reward is sensitive to protected attributes under controlled substitution, not that the RM assesses
applicants in a biased way. The harm evidence is `runners/run_cross_marker.py`, whose placement check
scores these same cells with the marker in the prompt.

Cells: by default every (axis, encoding) the manifest lists (``manifest.json``); ``--axes``/``--encodings``
filter them, and a name the manifest does not have stops the run before the model loads. The result file is
``battery_{domain}[_{manifest folder}]_{model}.json`` (a filtered run adds ``__{axes}__{encodings}``), never
overwritten without ``--overwrite``; its ``meta`` names the config with the loaded model commit, the code
commit and the manifest's SHA-256 (`scoring.experiment.run_metadata`).

Usage:
    python runners/run_battery.py --config configs/demographic_credit_sex_qwen06.yaml
    python runners/run_battery.py --config configs/demographic_credit_sex_qwen06.yaml --model Skywork/Skywork-Reward-V2-Llama-3.1-8B
    python runners/run_battery.py --config configs/demographic_edu_grade_level_asap2_qwen06.yaml   # the stage manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scoring.experiment import ExperimentConfig, add_override_args, apply_overrides, data_file, run_metadata
from scoring.demographic_experiment import (
    DemographicBiasExperiment, auto_influence_with_intervals, compute_auto_influence_metrics, nulling_change,
    organize_rewards, subgroup_metrics, texts_and_variants,
)
from scoring.intervals import DEFAULT_N_BOOT
from substrates.domains import get_domain
from probes.probe import build_probe_direction, embed_with_gates, get_rewards_both, rewards_from_hidden

RESULTS_DIR = Path("artifacts/results/demographic")
SWEEP_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SWEEP_AXES = ["sex", "intersection"]
DOMAIN_SWEEP = {"education": ["grade_level", "sex"]}
SWEEP_METRICS = ("mean_gap", "pref_a_rate", "abs_mean_gap", "auto_influence")


def manifest_cells(source: Path | str) -> List[Tuple[str, str]]:
    """The (axis, encoding) cells a manifest holds, in its order (``counts_by_axis_encoding``)."""
    manifest = json.loads((Path(source).parent / "manifest.json").read_text())
    return [tuple(key.split("/", 1)) for key in manifest["counts_by_axis_encoding"]]


def select_cells(available: Sequence[Tuple[str, str]], axes: Optional[Sequence[str]] = None,
                 encodings: Optional[Sequence[str]] = None) -> List[Tuple[str, str]]:
    """The manifest's cells, filtered by ``axes`` and ``encodings``. A named axis or encoding the manifest does not
    have at all is an error; a combination it lacks (credit's marital status has no proxy) is simply not run."""
    for what, wanted, have in (("axis", axes, {a for a, _ in available}), ("encoding", encodings, {e for _, e in available})):
        missing = sorted(set(wanted or ()) - have)
        if missing:
            raise SystemExit(f"{what} {missing} not in the manifest, which has {sorted(have)}")
    return [(a, e) for a, e in available if (not axes or a in axes) and (not encodings or e in encodings)]


def default_out(domain: str, source: Path | str, model_path: str,
                cells: Sequence[Tuple[str, str]], available: Sequence[Tuple[str, str]]) -> Path:
    """``battery_{domain}[_{manifest folder}]_{model}.json``; a run on part of the manifest adds
    ``__{axes}__{encodings}``, so it never takes the name of the full battery."""
    folder = Path(source).parent.name
    name = f"battery_{domain if folder == domain else f'{domain}_{folder}'}_{Path(model_path).name}"
    if list(cells) != list(available):
        axes = list(dict.fromkeys(a for a, _ in cells))
        encodings = list(dict.fromkeys(e for _, e in cells))
        name += f"__{'-'.join(axes)}__{'-'.join(encodings)}"
    return RESULTS_DIR / f"{name}.json"


def run_cell(exp, cfg, axis, encoding, dataset_cls, sweep_axes=SWEEP_AXES) -> Dict[str, Any]:
    ds = dataset_cls(cfg.dataset_source, axis=axis, encoding=encoding,
                     split_seed=cfg.split_seed,
                     max_test_examples=cfg.max_test_examples, probe_records=cfg.probe_records)
    probe, meta = build_probe_direction(exp.model, exp.tokenizer, ds.get_probe_pairs(exp.tokenizer),
                                        batch_size=cfg.batch_size, max_length=cfg.max_length)
    eval_examples = ds.get_eval_examples(exp.tokenizer)
    all_texts, text_meta = texts_and_variants(eval_examples)
    n = len(eval_examples)
    baseline, nulled = get_rewards_both(exp.model, exp.tokenizer, all_texts, probe,
                                        batch_size=cfg.batch_size,
                                        max_length=cfg.max_length, null_alpha=1.0, show_progress=False)
    base_org = organize_rewards(baseline, text_meta, n)
    null_org = organize_rewards(nulled, text_meta, n)
    n_boot, seed = int(cfg.extra.get("n_boot", DEFAULT_N_BOOT)), cfg.split_seed
    cell = {
        "axis": axis, "encoding": encoding, "n_eval": n,
        # in-sample: on the probe pairs the direction was fitted on
        "probe_accuracy": meta.get("probe_accuracy"), "probe_separation": meta.get("separation"),
        # each metric with its record-bootstrap interval (a record's pairs are correlated)
        "baseline": auto_influence_with_intervals(base_org, eval_examples, n_boot, seed),
        "nulled": auto_influence_with_intervals(null_org, eval_examples, n_boot, seed),
        "baseline_vs_nulled": nulling_change(base_org, null_org, eval_examples, n_boot, seed),
        # the full metric set per template, record quality (strong/weak) and, for education, essay prompt
        "baseline_by_template": subgroup_metrics(base_org, eval_examples, "template_id"),
        "baseline_by_quality": subgroup_metrics(base_org, eval_examples, "strong"),
        "baseline_by_prompt": subgroup_metrics(base_org, eval_examples, "prompt_id"),
    }
    # α-sweep: the texts' states come from the embedding cache; only the head runs per α.
    if axis in sweep_axes:
        cell["alpha_sweep"] = _alpha_sweep(exp, cfg, all_texts, text_meta, n, probe)
    return cell


def _alpha_sweep(exp, cfg, all_texts, text_meta, n, probe) -> Dict[str, Dict[str, float]]:
    """The signed and the folded metrics at each α (point estimates; the intervals at α = 1 are the cell's
    ``nulled``). One embedding pass (served from the embedding cache: these texts were just scored), then only
    the score head per α."""
    hidden, state_dtype, gates = embed_with_gates(exp.model, exp.tokenizer, all_texts, batch_size=cfg.batch_size,
                                                  max_length=cfg.max_length, show_progress=False)
    curve: Dict[str, Dict[str, float]] = {}
    for a in SWEEP_ALPHAS:
        _, scores = rewards_from_hidden(exp.model, hidden, state_dtype, probe, null_alpha=a, gates=gates)
        metrics = compute_auto_influence_metrics(organize_rewards(scores, text_meta, n))
        curve[str(a)] = {k: metrics.get(k, float("nan")) for k in SWEEP_METRICS}
    return curve


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_credit_sex_qwen06.yaml"))
    ap.add_argument("--axes", default=None, help="Comma-separated; default every axis the manifest holds")
    ap.add_argument("--encodings", default=None,
                    help="Comma-separated; default every encoding the manifest holds (A2: the positions)")
    ap.add_argument("--sweep-axes", default=None, help="Comma-separated axes to alpha-sweep; overrides default.")
    ap.add_argument("--dataset-source", default=None, help="Override the matched-pair manifest (pairs.jsonl).")
    ap.add_argument("--out", type=Path, default=None, help="Default: battery_{domain}[_{folder}]_{model}.json "
                                                            f"in {RESULTS_DIR}")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result file")
    add_override_args(ap)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None

    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    if args.dataset_source:
        cfg.dataset_source = args.dataset_source
    spec = get_domain(cfg.extra.get("domain", "credit"))
    cfg.dataset_source = cfg.dataset_source or spec.default_pairs
    # everything that can fail on the inputs fails here, before the model loads
    data = {Path(cfg.dataset_source).name: data_file(cfg.dataset_source)}
    available = manifest_cells(cfg.dataset_source)
    cells_to_run = select_cells(available, split(args.axes), split(args.encodings))
    sweep_axes = split(args.sweep_axes) or DOMAIN_SWEEP.get(spec.name, SWEEP_AXES)
    out = args.out or default_out(spec.name, cfg.dataset_source, cfg.model_path, cells_to_run, available)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or --out")

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    cells: List[Dict[str, Any]] = []
    for axis, enc in cells_to_run:
        print(f"[battery] {spec.name}/{axis}/{enc} ...", flush=True)
        cells.append(run_cell(exp, cfg, axis, enc, spec.dataset_cls, sweep_axes))

    # ---- report ----
    print("\n" + "=" * 86)
    print(f"DEMOGRAPHIC ROBUSTNESS BATTERY — {cfg.model_path}")
    print("=" * 86)
    print(f"{'axis':14} {'enc':9} {'probe_acc':>9} {'base_AI':>8} {'null_AI':>8} {'base_gap':>9}  per-template(base_AI)")
    for c in cells:
        bt = "  ".join(f"{k}={v.get('auto_influence', float('nan')):.2f}" for k, v in c["baseline_by_template"].items())
        print(f"{c['axis']:14} {c['encoding']:9} {c['probe_accuracy']:>9.2%} "
              f"{c['baseline']['auto_influence']:>8.3f} {c['nulled']['auto_influence']:>8.3f} "
              f"{c['baseline']['mean_gap']:>9.3f}  {bt}")
    print("(probe_acc is in-sample, on the probe pairs)")
    print("-" * 86)
    print("abs_mean_gap with its record-bootstrap 95% interval, and what nulling changed on the same pairs")
    for c in cells:
        ci = c["baseline"]["intervals"]["abs_mean_gap"]
        ch = c["baseline_vs_nulled"]["nulled_minus_baseline"]["abs_mean_gap_change"]
        print(f"{c['axis']:14} {c['encoding']:9} |gap| {ci['estimate']:.4f} [{ci['ci_low']:.4f}, "
              f"{ci['ci_high']:.4f}]   nulled − baseline {ch['estimate']:+.4f} "
              f"[{ch['ci_low']:+.4f}, {ch['ci_high']:+.4f}]   ({ci['n_clusters']} records)")
    print("-" * 86)
    print("is a preference left after nulling? the signed intervals (auto-influence and |gap| are positive under noise)")
    for c in cells:
        g, r = c["nulled"]["intervals"]["mean_gap"], c["nulled"]["intervals"]["pref_a_rate"]
        print(f"{c['axis']:14} {c['encoding']:9} nulled mean_gap {g['estimate']:+.4f} [{g['ci_low']:+.4f}, "
              f"{g['ci_high']:+.4f}]   pref_a_rate {r['estimate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print("-" * 86)
    for c in cells:
        if "alpha_sweep" in c:
            curve = "  ".join(f"α={a}: gap {v['mean_gap']:+.3f} pref_a {v['pref_a_rate']:.2f}"
                              for a, v in c["alpha_sweep"].items())
            print(f"sweep {c['axis']}/{c['encoding']}: {curve}")
    print("=" * 86)

    settings = {"cells": [list(c) for c in cells_to_run], "sweep_axes": sweep_axes, "sweep_alphas": SWEEP_ALPHAS,
                "n_boot": int(cfg.extra.get("n_boot", DEFAULT_N_BOOT)), "bootstrap_seed": cfg.split_seed}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": run_metadata(cfg, data, settings), "model": cfg.model_path,
                               "domain": spec.name, "cells": cells}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
