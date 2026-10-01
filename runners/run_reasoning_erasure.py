#!/usr/bin/env python3
"""
LEACE + non-linear-probe test (hiring, credit, education): is the **reasoning-correctness** concept (and
**conclusion polarity**) *low-complexity* in the RM's activations, or only *linearly* erasable while still
**non-linearly recoverable** (high-complexity / entangled — the TaCo signature)?

Items (`pairs.verdicts.REASONING_FRAMES`; the domain is the config's ``extra.domain``): per premise (the demographic
premise ``frame.primary`` and the control), one item per applicant, its four cells labeled by **correctness**
(TRUE-claim cells = 1) and **conclusion** (cells of the domain's favourable decision = 1). Two design choices keep
surface features from answering the question (decided 2026-09-30, after the same pipeline on bag-of-words vectors of
the earlier verdicts recovered correctness after LEACE at 0.98):
  - **no connective**: the decision is a sentence of its own. With "so" / "but", correctness = XOR("so", advance),
    so a non-linear probe recovers it from the connective and the decision once the linear information is gone;
  - **held-out wording**: the probe split's verdicts draw entries ``FIT_PARAPHRASES`` of every pool, the eval split's
    ``EVAL_PARAPHRASES``. Each class is a union of several phrasings, and on shared pools a non-linear probe maps a
    phrasing to its label after LEACE; on unseen phrasings it cannot.
Every item's draws are seeded by (seed, premise, record) (`run_reasoning_probe.reasoning_items`).

For each premise and concept, held-out **linear vs non-linear (MLP) probe accuracy** (`probes.erasure`) under three
conditions:
  - **none**     — no erasure (sanity: both probes ≫ chance ⇒ decodable);
  - **diffmean** — project out the difference-of-means direction (our method);
  - **leace**    — provably-optimal linear erasure.
LEACE guarantees linear → chance on the states it was fitted on (the probe split); on the held-out states it is
approximate, so the linear row after LEACE is a check, not a given. **MLP accuracy after LEACE is the answer**, read
from the 95% interval of ``mlp_above_chance`` (clustered by applicant: the four cells of one item are correlated),
one-sided: clearly above 0 ⇒ non-linearly recoverable / entangled; reaching 0 or below ⇒ low-complexity (the
baseline is the majority-class share, >= 0.5, so a probe at chance sits at or below it). Each row comes with the
**lexical control** (``lexical_control``): the same pipeline on bag-of-words vectors of the same verdicts
(`probes.erasure.bag_of_words`). The model's recovery says more than the text's surface only where it exceeds it.

Records: the domain's strong records of the manifest's pool that no premise contradicts
(`run_reasoning_flip.reasoning_records`), in seeded order: the first ``--probe-items`` are the probe split, the next
``--eval-items`` the eval split. The result ``erasure_{domain}_{model}{variant}.json`` (``variant`` names every
setting the CLI changed; never replaced without ``--overwrite``) carries ``meta`` (the config with the loaded model
commit, the code commit, the SHA-256 of the manifest's cells and of the corpus file; `scoring.experiment`) and both
splits' record ids.

Usage:
    python runners/run_reasoning_erasure.py --config configs/demographic_cv_reasoning_qwen06.yaml \
        --probe-items 200 --eval-items 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scoring.dataset_base import format_conversation
from substrates.domains import get_domain
from pairs.verdicts import EVAL_PARAPHRASES, FIT_PARAPHRASES, REASONING_CELLS, REASONING_FRAMES
from scoring.experiment import (
    ExperimentConfig, add_override_args, apply_overrides, run_metadata, variant_suffix,
)
from scoring.demographic_experiment import DemographicBiasExperiment
from scoring.intervals import DEFAULT_N_BOOT
from probes.probe import get_embeddings
from probes.erasure import (
    apply_diffmean, apply_eraser, bag_of_words, diffmean_direction, leace_erase, probe_recoverability,
)
from runners.run_decision_response import select_items
from runners.run_reasoning_flip import reasoning_records
from runners.run_reasoning_probe import probe_premises, reasoning_items

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("artifacts/results/demographic")
CONCEPTS = ("correctness", "conclusion")
METHODS = ("none", "diffmean", "leace")


def label(cell: str, concept: str) -> int:
    if concept == "correctness":
        return int(cell.startswith("true_"))
    return int(cell.endswith("_advance"))  # conclusion: the favourable decision=1, reject=0


class Split:
    """A split's items flattened to one row per (item, cell): the verdicts, the formatted conversations, the labels per
    concept and each row's item (applicant) index, the cluster of the intervals."""

    def __init__(self, tokenizer, items: List[Dict]):
        self.verdicts, self.texts, self.groups = [], [], []
        self.labels: Dict[str, List[int]] = {c: [] for c in CONCEPTS}
        for n, it in enumerate(items):
            for cell in REASONING_CELLS:
                self.verdicts.append(it["cells"][cell])
                self.texts.append(format_conversation(tokenizer, it["user_prompt"], it["cells"][cell]))
                self.groups.append(n)
                for c in CONCEPTS:
                    self.labels[c].append(label(cell, c))


def erasure_rows(Xtr: torch.Tensor, ytr: List[int], Xev: torch.Tensor, yev: List[int], groups_ev: List[int],
                 n_boot: int, seed: int) -> Dict[str, Any]:
    """Held-out linear and MLP recoverability without erasure, after the diffmean projection and after LEACE (both
    fitted on the training states)."""
    rows = {}
    for method in METHODS:
        if method == "none":
            tr, ev = Xtr, Xev
        elif method == "diffmean":
            d = diffmean_direction(Xtr, ytr)
            tr, ev = apply_diffmean(d, Xtr), apply_diffmean(d, Xev)
        else:
            eraser = leace_erase(Xtr, ytr)
            tr, ev = apply_eraser(eraser, Xtr), apply_eraser(eraser, Xev)
        rows[method] = probe_recoverability(tr, ytr, ev, yev, seed=seed, groups_ev=groups_ev, n_boot=n_boot)
    return rows


def run_premise(exp, cfg, dom, premise: str, probe_recs: List[Any], eval_recs: List[Any], seed: int, n_boot: int
                ) -> List[Dict[str, Any]]:
    """One premise: per concept, the erasure rows on the model's states and on the lexical control."""
    train = Split(exp.tokenizer, reasoning_items(dom, premise, probe_recs, FIT_PARAPHRASES, seed, connective=False))
    evals = Split(exp.tokenizer, reasoning_items(dom, premise, eval_recs, EVAL_PARAPHRASES, seed, connective=False))
    Xtr = get_embeddings(exp.model, exp.tokenizer, train.texts, cfg.batch_size, cfg.max_length, show_progress=False)
    Xev = get_embeddings(exp.model, exp.tokenizer, evals.texts, cfg.batch_size, cfg.max_length, show_progress=False)
    Ltr, Lev = bag_of_words(train.verdicts, evals.verdicts)
    return [{"premise": premise, "concept": c, "n_train": len(train.texts), "n_eval": len(evals.texts),
             "rows": erasure_rows(Xtr, train.labels[c], Xev, evals.labels[c], evals.groups, n_boot, seed),
             "lexical_control": erasure_rows(Ltr, train.labels[c], Lev, evals.labels[c], evals.groups, n_boot, seed)}
            for c in CONCEPTS]


def default_out(domain: str, model_path: str, variant: str = "") -> Path:
    return RESULTS_DIR / f"erasure_{domain}_{Path(model_path).name}{variant}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_cv_reasoning_qwen06.yaml"))
    ap.add_argument("--probe-items", type=int, default=200)
    ap.add_argument("--eval-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"Default {RESULTS_DIR}/erasure_{{domain}}_{{model}}{{variant}}.json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result")
    add_override_args(ap)
    return ap


def _gap(row: Dict[str, Any]) -> Tuple[float, float, float]:
    g = row["intervals"]["mlp_above_chance"]
    return g["estimate"], g["ci_low"], g["ci_high"]


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    configured_cfg = ExperimentConfig.from_yaml(args.config)
    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    domain = cfg.extra.get("domain")
    if domain not in REASONING_FRAMES:
        raise SystemExit(f"the reasoning arm runs on {sorted(REASONING_FRAMES)}: set extra.domain (the config has "
                         f"{domain!r})")
    if args.probe_items < 2 or args.eval_items < 2:
        raise SystemExit("--probe-items and --eval-items must be at least 2")
    dom = get_domain(domain)
    premises = probe_premises(domain)
    # everything that can fail on the inputs fails here, before the model loads
    keys = ("probe_items", "eval_items", "seed")
    configured = {**{k: ap.get_default(k) for k in keys}, "revision": configured_cfg.model_revision}
    used = {**{k: getattr(args, k) for k in keys}, "revision": cfg.model_revision}
    out = args.out or default_out(domain, cfg.model_path, variant_suffix(configured, used))
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or --out")
    cfg.dataset_source = cfg.dataset_source or dom.default_pairs
    pool, data, left_out = reasoning_records(dom, REASONING_FRAMES[domain], cfg.dataset_source)
    wanted = args.probe_items + args.eval_items
    records, selection = select_items(pool, dom.is_strong, set(), wanted, args.seed)
    selection.update(left_out)
    if len(records) < wanted:
        raise SystemExit(f"{selection['available']} eligible strong records, fewer than --probe-items + "
                         f"--eval-items = {wanted}")
    probe_recs, eval_recs = records[:args.probe_items], records[args.probe_items:]
    n_boot = int(cfg.extra.get("n_boot", DEFAULT_N_BOOT))

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    results: List[Dict[str, Any]] = []
    for premise in premises:
        print(f"[erasure] {dom.name}/{premise} ...", flush=True)
        results += run_premise(exp, cfg, dom, premise, probe_recs, eval_recs, args.seed, n_boot)

    print("\n" + "=" * 112)
    print(f"REASONING ERASURE — LEACE + non-linear probe — {cfg.model_path}  (probe={len(probe_recs)}, "
          f"eval={len(eval_recs)} applicants; held-out wording, no connective)")
    print("=" * 112)
    print(f"{'premise':15} {'concept':12} {'method':9} {'linear':>7} {'MLP':>7}  {'MLP − chance [95% CI]':>28}  "
          f"{'lexical control':>28}")
    for r in results:
        for m in METHODS:
            row, lex = r["rows"][m], r["lexical_control"][m]
            print(f"{r['premise']:15} {r['concept']:12} {m:9} {row['linear_acc']:>7.3f} {row['mlp_acc']:>7.3f}  "
                  f"{'{:+.3f} [{:+.3f}, {:+.3f}]'.format(*_gap(row)):>28}  "
                  f"{'{:+.3f} [{:+.3f}, {:+.3f}]'.format(*_gap(lex)):>28}")
        print("-" * 112)
    print("MLP after LEACE: interval clearly above 0 (and above the lexical control) ⇒ non-linearly recoverable "
          "(entangled); reaching 0 or below ⇒ low-complexity.")

    settings = {"probe_items": args.probe_items, "eval_items": args.eval_items, "seed": args.seed, "n_boot": n_boot,
                "premises": list(premises), "fit_paraphrases": list(FIT_PARAPHRASES),
                "eval_paraphrases": list(EVAL_PARAPHRASES), "connective": False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": run_metadata(cfg, data, settings), "model": cfg.model_path, "domain": dom.name, "seed": args.seed,
         "selection": selection, "probe_records": [str(r.source_record_id) for r in probe_recs],
         "eval_records": [str(r.source_record_id) for r in eval_recs], "results": results}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
