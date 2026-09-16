#!/usr/bin/env python3
"""
Mechanistic probe of the reasoning-flip drivers, **held-out probe/eval split**: are
**reasoning correctness** and the **advance/reject conclusion** each represented as a *linear direction*
in the RM's activations that can be nulled — measured on items NOT used to build the direction?

The 2×2 gives perfectly matched contrastive pairs:
  - **correctness direction** = DiffMean(true-claim − false-claim verdicts), holding the conclusion fixed.
  - **conclusion direction**  = DiffMean(advance − reject verdicts), holding correctness fixed.

We build each direction on a **probe split** of applicants and measure, on a **disjoint eval split**:
(1) **held-out decodability** — paired accuracy of the direction on unseen contrastive pairs; (2) **held-out
nulling** — does projecting out `corr_dir` collapse the eval correctness effect toward 0 while leaving the
conclusion effect (and vice versa)? Now a genuine generalization result, not the in-sample mechanical
collapse. (3) **cross-premise transfer** — null parental-leave eval items with the *commute* correctness
direction (and vice versa): breaks the shared-wording shortcut, testing one general direction (cos≈0.94).

Usage:
    python experiments/run_reasoning_probe.py --config configs/demographic_cv_reasoning_qwen06.yaml \
        --probe-items 200 --eval-items 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scoring.dataset_base import ContrastivePair, format_conversation
from substrates.domains import get_domain
from pairs.verdicts import REASONING_CELLS, build_reasoning_item
from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment, compute_reasoning_metrics
from probes.probe import build_probe_direction, get_embeddings, get_rewards_both

PREMISES = ["parental_leave", "commute"]  # demographic + non-demographic control

# matched contrastive pairs (pos_key, neg_key): two per item, holding the other factor fixed
CORRECTNESS_PAIRS = [("true_reject", "false_reject"), ("true_advance", "false_advance")]
CONCLUSION_PAIRS = [("true_advance", "true_reject"), ("false_advance", "false_reject")]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a / (a.norm() + 1e-8)) @ (b / (b.norm() + 1e-8)))


def _split(records, n_probe: int, n_eval: int) -> Tuple[List, List]:
    """Disjoint probe/eval splits from an (already-shuffled) record list."""
    probe = records[:n_probe]
    eval_ = records[n_probe:n_probe + n_eval]
    return probe, eval_


def contrastive_pairs(items, fmt: Callable[[str, str], Any], cell_pairs) -> List[ContrastivePair]:
    """Build DiffMean contrastive pairs from reasoning items. ``fmt(user_prompt, verdict)`` formats one
    [user, assistant] conversation; ``cell_pairs`` lists (positive_cell, negative_cell) keys."""
    out = []
    for it in items:
        for pos, neg in cell_pairs:
            out.append(ContrastivePair(
                positive_text=fmt(it["user_prompt"], it["cells"][pos]),
                negative_text=fmt(it["user_prompt"], it["cells"][neg]),
            ))
    return out


def _items(dom, premise, records, rng):
    tids = list(dom.template_ids)
    return [build_reasoning_item(r, premise, dom.render_fn, rng, template_id=tids[i % len(tids)])
            for i, r in enumerate(records)]


def _heldout_paired_acc(exp, cfg, pairs: List[ContrastivePair], direction) -> float:
    """Paired decodability on unseen pairs: mean[ proj(positive) > proj(negative) ]."""
    pos = get_embeddings(exp.model, exp.tokenizer, [p.positive_text for p in pairs],
                         cfg.batch_size, cfg.device, cfg.max_length)
    neg = get_embeddings(exp.model, exp.tokenizer, [p.negative_text for p in pairs],
                         cfg.batch_size, cfg.device, cfg.max_length)
    return float(((pos @ direction) > (neg @ direction)).float().mean())


def _effects_on(exp, cfg, eval_flat, n, direction):
    base, nulled = get_rewards_both(exp.model, exp.tokenizer, eval_flat, direction,
                                    batch_size=cfg.batch_size, device=cfg.device,
                                    max_length=cfg.max_length, null_alpha=1.0, show_progress=False)
    by = lambda s: {c: s[i * n:(i + 1) * n].tolist() for i, c in enumerate(REASONING_CELLS)}
    return compute_reasoning_metrics(by(base)), compute_reasoning_metrics(by(nulled))


def run_premise(exp, cfg, dom, premise, probe_recs, eval_recs, rng) -> Dict[str, Any]:
    tok = exp.tokenizer
    fmt = lambda u, v: format_conversation(tok, u, v)
    probe_items = _items(dom, premise, probe_recs, rng)
    eval_items = _items(dom, premise, eval_recs, rng)

    # directions from the PROBE split only
    corr_dir, corr_meta = build_probe_direction(
        exp.model, tok, contrastive_pairs(probe_items, fmt, CORRECTNESS_PAIRS),
        batch_size=cfg.batch_size, device=cfg.device, max_length=cfg.max_length)
    concl_dir, concl_meta = build_probe_direction(
        exp.model, tok, contrastive_pairs(probe_items, fmt, CONCLUSION_PAIRS),
        batch_size=cfg.batch_size, device=cfg.device, max_length=cfg.max_length)

    # held-out decodability on the EVAL split
    corr_acc_ho = _heldout_paired_acc(exp, cfg, contrastive_pairs(eval_items, fmt, CORRECTNESS_PAIRS), corr_dir)
    concl_acc_ho = _heldout_paired_acc(exp, cfg, contrastive_pairs(eval_items, fmt, CONCLUSION_PAIRS), concl_dir)

    # held-out nulling on the EVAL split
    n = len(eval_items)
    eval_flat = [fmt(it["user_prompt"], it["cells"][c]) for c in REASONING_CELLS for it in eval_items]
    base_c, null_c = _effects_on(exp, cfg, eval_flat, n, corr_dir)
    _, null_k = _effects_on(exp, cfg, eval_flat, n, concl_dir)

    return {
        "premise": premise, "demographic": probe_items[0]["meta"]["demographic"],
        "n_probe": len(probe_items), "n_eval": n,
        "correctness_probe": {"acc_insample": corr_meta.get("probe_accuracy"),
                              "acc_heldout": corr_acc_ho, "separation": corr_meta.get("separation")},
        "conclusion_probe": {"acc_insample": concl_meta.get("probe_accuracy"),
                             "acc_heldout": concl_acc_ho, "separation": concl_meta.get("separation")},
        "cos_correctness_conclusion": _cos(corr_dir, concl_dir),
        "heldout_baseline": {"correctness_effect": base_c["correctness_effect"],
                             "conclusion_effect": base_c["conclusion_effect"]},
        "heldout_null_correctness_dir": {"correctness_effect": null_c["correctness_effect"],
                                         "conclusion_effect": null_c["conclusion_effect"]},
        "heldout_null_conclusion_dir": {"correctness_effect": null_k["correctness_effect"],
                                        "conclusion_effect": null_k["conclusion_effect"]},
        "_corr_dir": corr_dir, "_concl_dir": concl_dir, "_eval_flat": eval_flat, "_n_eval": n,
    }


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
    out = args.out or Path(f"artifacts/results/demographic/reasoning_probe_{dom.name}_qwen06.json")
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    records = [r for r in dom.load_records() if getattr(r, "qualified", True)]
    random.Random(args.seed).shuffle(records)
    probe_recs, eval_recs = _split(records, args.probe_items, args.eval_items)
    assert len(eval_recs) == args.eval_items, "not enough records for the requested probe+eval split"

    results = []
    for premise in PREMISES:
        print(f"[reasoning-probe/held-out] {dom.name}/{premise} ...", flush=True)
        results.append(run_premise(exp, cfg, dom, premise, probe_recs, eval_recs, random.Random(args.seed)))

    # cross-premise transfer: null premise X's eval cells with premise Y's correctness direction
    dirs = {r["premise"]: r["_corr_dir"] for r in results}
    cdirs = {r["premise"]: r["_concl_dir"] for r in results}
    transfer = {}
    by_premise = {r["premise"]: r for r in results}
    if {"parental_leave", "commute"} <= set(by_premise):
        for tgt, src in (("parental_leave", "commute"), ("commute", "parental_leave")):
            r = by_premise[tgt]
            base, nulled = _effects_on(exp, cfg, r["_eval_flat"], r["_n_eval"], dirs[src])
            transfer[f"{src}_dir__on__{tgt}_eval"] = {
                "correctness_effect_base": base["correctness_effect"],
                "correctness_effect_null": nulled["correctness_effect"],
            }
        transfer["cos_correctness_parental_vs_commute"] = _cos(dirs["parental_leave"], dirs["commute"])
        transfer["cos_conclusion_parental_vs_commute"] = _cos(cdirs["parental_leave"], cdirs["commute"])

    for r in results:  # drop tensors before serialization
        for k in ("_corr_dir", "_concl_dir", "_eval_flat", "_n_eval"):
            r.pop(k, None)

    print("\n" + "=" * 104)
    print(f"REASONING-PROBE — HELD-OUT split — {cfg.model_path}  "
          f"(probe={args.probe_items}, eval={args.eval_items} applicants, disjoint)")
    print("=" * 104)
    for r in results:
        cp, kp = r["correctness_probe"], r["conclusion_probe"]
        print(f"\n[{r['premise']}]  (demographic={r['demographic']})")
        print(f"  decodable (held-out paired acc): correctness={cp['acc_heldout']:.2%} "
              f"(in-sample {cp['acc_insample']:.2%}) | conclusion={kp['acc_heldout']:.2%} "
              f"(in-sample {kp['acc_insample']:.2%})")
        print(f"  cos(correctness_dir, conclusion_dir) = {r['cos_correctness_conclusion']:+.3f}")
        b, nc, nk = r["heldout_baseline"], r["heldout_null_correctness_dir"], r["heldout_null_conclusion_dir"]
        print(f"  HELD-OUT correctness_effect: base={b['correctness_effect']:+.3f}  "
              f"null(corr)={nc['correctness_effect']:+.3f}  null(concl)={nk['correctness_effect']:+.3f}")
        print(f"  HELD-OUT conclusion_effect:  base={b['conclusion_effect']:+.3f}  "
              f"null(corr)={nc['conclusion_effect']:+.3f}  null(concl)={nk['conclusion_effect']:+.3f}")
    if transfer:
        print("\n" + "-" * 104)
        print(f"cross-premise cos(correctness dir): parental vs commute = "
              f"{transfer['cos_correctness_parental_vs_commute']:+.3f}")
        for k in ("commute_dir__on__parental_leave_eval", "parental_leave_dir__on__commute_eval"):
            t = transfer[k]
            print(f"  transfer {k}: correctness_effect base={t['correctness_effect_base']:+.3f} "
                  f"→ null={t['correctness_effect_null']:+.3f}")
    print("=" * 104)
    print("held-out null(corr) collapses correctness_effect & spares conclusion_effect ⇒ genuine, separable.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": cfg.model_path, "domain": dom.name, "seed": args.seed,
                               "probe_items": args.probe_items, "eval_items": args.eval_items,
                               "transfer": transfer, "results": results}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
