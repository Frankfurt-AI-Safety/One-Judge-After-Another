#!/usr/bin/env python3
"""
Cross-influence (EU-AI-Act reliability harm) on the credit arm, one RM (default Qwen3-0.6B).

Pairs a **good-credit** record (strong) with a **bad-credit** record (weak) from German Credit, then
asks: does attaching the **protected** demographic marker to the *weaker* applicant degrade the RM's
ability to still rank the stronger one first?

Per axis × encoding it renders 5 variants per pair (strong_neutral, weak_neutral, weak_protected,
weak_reference, strong_protected), scores them **baseline and null-space-projected** in one pass
(reusing the axis difference-of-means probe), and reports:
  acc_baseline      = P(strong_neutral > weak_neutral)   [does the RM track creditworthiness at all?]
  cross_influence   = acc_baseline − P(strong_neutral > weak_protected)   [signed reliability harm]
plus reference-pole and marker-on-strong controls — for baseline vs nulled scores.

Usage:
    python experiments/run_crossinfluence.py --config configs/demographic_credit_sex_qwen06.yaml \
        --encoding explicit --n-pairs 300
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
from substrates.bios_ingest import DEFAULT_BIOS_PATH, load_bias_in_bios
from substrates.credit_clean import load_factorial_records
from substrates.education_ingest import DEFAULT_ASAP_PATH, DEFAULT_PERSUADE_PATH, load_asap, load_persuade
from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment, compute_cross_influence
from probes.probe import build_probe_direction, get_rewards_both

_EDU_LOADERS = {"persuade": (load_persuade, DEFAULT_PERSUADE_PATH), "asap": (load_asap, DEFAULT_ASAP_PATH)}
VARIANTS = ["strong_neutral", "weak_neutral", "weak_protected", "weak_reference", "strong_protected"]


def _build_pairs(records, n_pairs, seed, is_strong, template_ids):
    strong = [r for r in records if is_strong(r)]
    weak = [r for r in records if not is_strong(r)]
    rng = random.Random(seed)
    rng.shuffle(strong)
    rng.shuffle(weak)
    n = min(n_pairs, len(strong), len(weak))
    tids = list(template_ids)
    return [(strong[i], weak[i], tids[i % len(tids)]) for i in range(n)]


def run_axis(exp, cfg, dom, axis, encoding, pairs, rng, subject="applicant") -> Dict[str, Any]:
    tok = exp.tokenizer
    fmt = lambda txt: format_conversation(tok, dom.assessment_prompt, txt)
    render = dom.render_fn
    texts: Dict[str, List[Any]] = {v: [] for v in VARIANTS}
    for good, bad, tid in pairs:
        spec = dom.make_marker(axis, encoding, rng, subject)  # clause_a = protected pole, clause_b = reference
        texts["strong_neutral"].append(fmt(render(good, tid, "")))
        texts["weak_neutral"].append(fmt(render(bad, tid, "")))
        texts["weak_protected"].append(fmt(render(bad, tid, spec.clause_a)))
        texts["weak_reference"].append(fmt(render(bad, tid, spec.clause_b)))
        texts["strong_protected"].append(fmt(render(good, tid, spec.clause_a)))

    # axis difference-of-means probe (from the existing matched-pair manifest)
    ds = dom.dataset_cls(cfg.dataset_source, axis=axis, encoding=encoding,
                         probe_size=cfg.probe_size, split_seed=cfg.split_seed)
    probe, _ = build_probe_direction(exp.model, tok, ds.get_probe_pairs(tok),
                                     batch_size=cfg.batch_size, device=cfg.device,
                                     max_length=cfg.max_length)

    # score all variants at once → baseline + nulled
    n = len(pairs)
    flat = [t for v in VARIANTS for t in texts[v]]
    base, nulled = get_rewards_both(exp.model, tok, flat, probe, batch_size=cfg.batch_size,
                                    device=cfg.device, max_length=cfg.max_length,
                                    null_alpha=1.0, show_progress=False)
    base_by = {v: base[i * n:(i + 1) * n].tolist() for i, v in enumerate(VARIANTS)}
    null_by = {v: nulled[i * n:(i + 1) * n].tolist() for i, v in enumerate(VARIANTS)}
    return {"axis": axis, "encoding": encoding, "n_pairs": n,
            "baseline": compute_cross_influence(base_by),
            "nulled": compute_cross_influence(null_by)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_credit_sex_qwen06.yaml"))
    ap.add_argument("--encoding", default="explicit", choices=["explicit", "proxy"])
    ap.add_argument("--axes", default=None, help="Comma-separated axes; default is domain-appropriate.")
    ap.add_argument("--dataset-source", default=None, help="Override the matched-pair manifest (probe pairs).")
    ap.add_argument("--source", default="persuade", choices=sorted(_EDU_LOADERS),
                    help="Education only: which corpus to load strong/weak records from.")
    ap.add_argument("--raw-path", default=None,
                    help="Override the corpus file path (credit: german.data; cv/education: the corpus).")
    ap.add_argument("--n-pairs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="Defaults to artifacts/results/demographic/crossinf_{domain}_qwen06.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.dataset_source:
        cfg.dataset_source = args.dataset_source
    dom = get_domain(cfg.extra.get("domain", "credit"))
    subject = "student" if dom.name == "education" else "applicant"
    axes = [a.strip() for a in args.axes.split(",")] if args.axes else list(dom.axes)
    out = args.out or Path(f"artifacts/results/demographic/crossinf_{dom.name}_qwen06.json")
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    # Every domain honours --raw-path (and education also --source); the registry loader is zero-arg,
    # so the override has to happen here.
    if dom.name == "education":
        loader, default_path = _EDU_LOADERS[args.source]
        records = loader(args.raw_path or default_path)
    elif dom.name == "cv":
        records = load_bias_in_bios(args.raw_path or DEFAULT_BIOS_PATH)
    elif dom.name == "credit":
        # same population as the registry loader (both consistency rule sets), from the given file
        records = load_factorial_records(args.raw_path)
    else:
        records = dom.load_records()
    pairs = _build_pairs(records, args.n_pairs, args.seed, dom.is_strong, dom.template_ids)
    pairing = [{"strong": g.source_record_id, "weak": b.source_record_id, "template_id": t}
               for g, b, t in pairs]
    rng = random.Random(args.seed)

    results = []
    for axis in axes:
        print(f"[cross-influence] {dom.name}/{axis}/{args.encoding} ...", flush=True)
        results.append(run_axis(exp, cfg, dom, axis, args.encoding, pairs, rng, subject))

    print("\n" + "=" * 96)
    print(f"CROSS-INFLUENCE [{dom.name}] — {cfg.model_path}  "
          f"(encoding={args.encoding}, n_pairs={len(pairs)})")
    print("=" * 96)
    print(f"{'axis':14} {'acc_base':>9} {'tracks?':>8} | {'CI_base':>8} {'CI_null':>8} | "
          f"{'CI_ref':>8} {'mark_on_strong':>15}")
    for r in results:
        b, nl = r["baseline"], r["nulled"]
        print(f"{r['axis']:14} {b['acc_baseline']:>9.3f} {str(b['baseline_tracks_quality']):>8} | "
              f"{b['cross_influence']:>8.3f} {nl['cross_influence']:>8.3f} | "
              f"{b.get('cross_influence_reference', float('nan')):>8.3f} "
              f"{b.get('marker_on_strong_effect', float('nan')):>15.3f}")
    print("=" * 96)
    print("acc_base≈0.5 ⇒ RM doesn't track applicant quality ⇒ cross-influence not interpretable.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": cfg.model_path, "domain": dom.name, "encoding": args.encoding, "seed": args.seed,
         "pairing": pairing, "results": results}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
