#!/usr/bin/env python3
"""
LEACE + non-linear-probe test: is the **reasoning-correctness** concept (and **conclusion
polarity**) genuinely *low-complexity* in the RM's activations, or only *linearly* erasable while still
**non-linearly recoverable** (high-complexity / entangled — the TaCo signature)?

For each premise we build diversified verdict items (`vary=True`: paraphrases + multiple claim types, so
the concept is decorrelated from any single phrase), split train/eval by applicant, extract verdict
activations (`get_embeddings`) labeled by **correctness** (TRUE-claim cells=1) and **conclusion**
(advance cells=1), and for each concept report held-out **linear vs non-linear (MLP) probe accuracy**
under three conditions:
  - **none**     — no erasure (sanity: both probes ≫ chance ⇒ decodable);
  - **diffmean** — project out the difference-of-means direction (our method);
  - **leace**    — provably-optimal linear erasure.
Linear→chance after LEACE is guaranteed; **MLP accuracy after LEACE is the answer**: ≈chance ⇒ low-complexity,
≫chance ⇒ non-linearly recoverable / entangled.

Usage:
    python experiments/run_reasoning_erasure.py --config configs/demographic_cv_reasoning_qwen06.yaml \
        --probe-items 200 --eval-items 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scoring.dataset_base import format_conversation
from substrates.domains import get_domain
from pairs.verdicts import REASONING_CELLS, build_reasoning_item
from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment
from probes.probe import get_embeddings
from probes.erasure import (
    apply_diffmean, apply_eraser, diffmean_erase, leace_erase, probe_recoverability,
)

PREMISES = ["parental_leave", "commute"]
CONCEPTS = ("correctness", "conclusion")


def _label(cell: str, concept: str) -> int:
    if concept == "correctness":
        return int(cell.startswith("true_"))
    return int(cell.endswith("_advance"))  # conclusion: advance=1, reject=0


def _embed_and_label(exp, cfg, dom, premise, records, rng) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    tok = exp.tokenizer
    tids = list(dom.template_ids)
    items = [build_reasoning_item(r, premise, dom.render_fn, rng, template_id=tids[i % len(tids)], vary=True)
             for i, r in enumerate(records)]
    texts, labels = [], {c: [] for c in CONCEPTS}
    for it in items:
        for cell in REASONING_CELLS:
            texts.append(format_conversation(tok, it["user_prompt"], it["cells"][cell]))
            for c in CONCEPTS:
                labels[c].append(_label(cell, c))
    X = get_embeddings(exp.model, tok, texts, cfg.batch_size, cfg.device, cfg.max_length)
    return X, {c: torch.tensor(labels[c]) for c in CONCEPTS}


def _erasure_row(Xtr, ytr, Xev, yev, method: str) -> Dict[str, float]:
    if method == "none":
        return probe_recoverability(Xtr, ytr, Xev, yev)
    if method == "diffmean":
        d = diffmean_erase(Xtr, ytr)
        return probe_recoverability(apply_diffmean(d, Xtr), ytr, apply_diffmean(d, Xev), yev)
    if method == "leace":
        er = leace_erase(Xtr, ytr)
        return probe_recoverability(apply_eraser(er, Xtr), ytr, apply_eraser(er, Xev), yev)
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_cv_reasoning_qwen06.yaml"))
    ap.add_argument("--probe-items", type=int, default=200)
    ap.add_argument("--eval-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    dom = get_domain(cfg.extra.get("domain", "credit"))
    out = args.out or Path(f"artifacts/results/demographic/erasure_{dom.name}_qwen06.json")
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    records = [r for r in dom.load_records() if getattr(r, "qualified", True)]
    random.Random(args.seed).shuffle(records)
    probe_recs = records[:args.probe_items]
    eval_recs = records[args.probe_items:args.probe_items + args.eval_items]
    assert len(eval_recs) == args.eval_items, "not enough records for probe+eval split"

    results: List[Dict[str, Any]] = []
    for premise in PREMISES:
        print(f"[erasure] embedding {dom.name}/{premise} ...", flush=True)
        Xtr, ytr = _embed_and_label(exp, cfg, dom, premise, probe_recs, random.Random(args.seed))
        Xev, yev = _embed_and_label(exp, cfg, dom, premise, eval_recs, random.Random(args.seed + 1))
        for concept in CONCEPTS:
            rows = {m: _erasure_row(Xtr, ytr[concept], Xev, yev[concept], m)
                    for m in ("none", "diffmean", "leace")}
            results.append({"premise": premise, "concept": concept, "rows": rows})
            print(f"  [{premise}/{concept}] done", flush=True)

    print("\n" + "=" * 96)
    print(f"REASONING ERASURE — LEACE + non-linear probe — {cfg.model_path}")
    print("=" * 96)
    print(f"{'premise':15} {'concept':12} {'method':9} {'linear':>8} {'MLP':>8} {'chance':>8}")
    for r in results:
        for m in ("none", "diffmean", "leace"):
            row = r["rows"][m]
            print(f"{r['premise']:15} {r['concept']:12} {m:9} {row['linear_acc']:>8.3f} "
                  f"{row['mlp_acc']:>8.3f} {row['chance']:>8.3f}")
        print("-" * 96)
    print("MLP after LEACE ≈ chance ⇒ low-complexity; ≫ chance ⇒ non-linearly recoverable (entangled).")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": cfg.model_path, "domain": dom.name, "seed": args.seed,
                               "probe_items": args.probe_items, "eval_items": args.eval_items,
                               "results": results}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
