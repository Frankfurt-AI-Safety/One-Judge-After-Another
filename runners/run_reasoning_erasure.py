#!/usr/bin/env python3
"""
LEACE + non-linear-probe test (hiring, credit, education): are **claim correctness**, **conclusion polarity** and
the claim's **valence** *low-complexity* in the RM's activations, or only *linearly* erasable while still
**non-linearly recoverable** (high-complexity / entangled — the TaCo signature)?

Items (`pairs.verdicts.REASONING_FRAMES`; the domain is the config's ``extra.domain``): per premise (the demographic
premise ``frame.primary``, the control and the favourable-truth premise; decided 2026-10-01), one item per
applicant, its four cells labeled by **correctness** (TRUE-claim cells = 1), **conclusion** (cells of the domain's
favourable decision = 1) and **valence** (cells whose claim favours the applicant = 1;
`pairs.verdicts.claim_is_favourable`). The probes are trained on the control's and the favourable-truth premise's
items pooled (opposite truth configurations, so correctness and valence are crossed in the training labels; within
one premise the valence labels are the correctness labels or their complement) and evaluated on every premise's
held-out items (`run_domain`). Two design choices keep surface features from answering the question (decided
2026-09-30, after the same pipeline on bag-of-words vectors of the earlier verdicts recovered correctness after LEACE
at 0.98):
  - **no connective**: the decision is a sentence of its own. With "so" / "but", correctness = XOR("so", advance),
    so a non-linear probe recovers it from the connective and the decision once the linear information is gone;
  - **held-out wording, cross-fitted**: each fold trains on verdicts drawing four entries of every six-entry pool and
    evaluates on the other two (`pairs.verdicts.PARAPHRASE_FOLDS`). Each class is a union of several phrasings, and
    on shared pools a non-linear probe maps a phrasing to its label after LEACE; on unseen phrasings it cannot (no
    word that separates true from false recurs across entries; tested on the words alone, both ways, every fold).
Every item's draws are seeded by (seed, premise, record) (`run_reasoning_probe.reasoning_items`).

For each evaluated premise and concept, held-out **linear vs non-linear (MLP) probe accuracy** (`probes.erasure`)
under three conditions:
  - **none**     — no erasure (sanity: both probes ≫ chance ⇒ decodable);
  - **diffmean** — project out the difference-of-means direction (our method);
  - **leace**    — provably-optimal linear erasure.
LEACE erases correctness and valence jointly (pooled over the two training premises each is the other XOR the
premise, so erasing one alone leaves a route an MLP can take: a model with no truth code at all would read
"entangled"; `probes.erasure.leace_erase`, tests/test_erasure.py), conclusion alone; the MLP is a majority vote of
three seeds. LEACE guarantees linear → chance on the states it was fitted on (the probe split); on the held-out states
it is approximate, so the linear row after LEACE is a check, not a given. **MLP accuracy after LEACE is the
answer**, read from the 95% interval of ``mlp_above_chance`` (clustered by applicant: the four cells of one item are
correlated), one-sided: clearly above 0 ⇒ non-linearly recoverable / entangled; reaching 0 or below ⇒ low-complexity
(the baseline is the majority-class share, >= 0.5, so a probe at chance sits at or below it) — but only where the
concept is decodable without erasure (the "none" row's linear probe above chance): the demographic premise is never
trained on, so there a row at chance may only mean the probes do not transfer to it (``verdict``, `erasure_verdict`).
Every row pools the three folds (each fold's probes on the eval states of its held-out wording; the interval resamples
applicants with all their folds; `probes.erasure.recoverability_summary`), and ``by_fold`` gives each fold's point
values. Each row comes
with the **lexical control** (``lexical_control``): the same pipeline on bag-of-words vectors of the same verdicts
(`probes.erasure.bag_of_words`). The model's recovery says more than the text's surface only where it exceeds it.
The held-out wording varies within a fold (two entries per pool), so the lexical control varies over applicants
too; ``by_fold`` gives the spread over the three pairs of held-out entries.

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
from pairs.verdicts import PARAPHRASE_FOLDS, REASONING_CELLS, REASONING_FRAMES, claim_is_favourable
from scoring.experiment import (
    ExperimentConfig, add_override_args, apply_overrides, run_metadata, variant_suffix,
)
from scoring.demographic_experiment import DemographicBiasExperiment
from scoring.intervals import DEFAULT_N_BOOT
from probes.probe import get_embeddings
from probes.erasure import (
    apply_diffmean, apply_eraser, bag_of_words, diffmean_direction, leace_erase, probe_recoverability,
    recoverability_summary,
)
from runners.run_decision_response import select_items
from runners.run_reasoning_flip import reasoning_records
from runners.run_reasoning_probe import probe_premises, reasoning_items

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("artifacts/results/demographic")
CONCEPTS = ("correctness", "conclusion", "valence")
METHODS = ("none", "diffmean", "leace")
# correctness and valence are erased together: pooled over the two training premises, each is the other XOR the
# premise, so erasing one alone leaves a route a non-linear probe can take (`probes.erasure.leace_erase`)
JOINT = ("correctness", "valence")
MLP_SEEDS = 3                    # the MLP's majority vote over three seeds (`probes.erasure.probe_recoverability`)


def label(cell: str, concept: str, favourable_true: bool = False) -> int:
    if concept == "correctness":
        return int(cell.startswith("true_"))
    if concept == "conclusion":
        return int(cell.endswith("_advance"))  # the favourable decision=1, reject=0
    if concept == "valence":
        return int(claim_is_favourable(cell, favourable_true))
    raise ValueError(f"concept must be one of {CONCEPTS}, got {concept!r}")


class Split:
    """A split's items flattened to one row per (item, cell): the verdicts, the formatted conversations, the labels per
    concept and each row's item (applicant) index, the cluster of the intervals (the same index in every fold, as
    every fold reads the same applicants)."""

    def __init__(self, tokenizer, items: List[Dict]):
        self.record_ids = [it["meta"]["record_id"] for it in items]
        self.verdicts, self.texts, self.groups = [], [], []
        self.labels: Dict[str, List[int]] = {c: [] for c in CONCEPTS}
        for n, it in enumerate(items):
            for cell in REASONING_CELLS:
                self.verdicts.append(it["cells"][cell])
                self.texts.append(format_conversation(tokenizer, it["user_prompt"], it["cells"][cell]))
                self.groups.append(n)
                for c in CONCEPTS:
                    self.labels[c].append(label(cell, c, it["meta"]["favourable_true"]))


def erasure_rows(Xtr: torch.Tensor, ytr: List[int], Xev: torch.Tensor, yev: List[int], groups_ev: List[int],
                 seed: int, erase: Any = None) -> Dict[str, Any]:
    """Held-out linear and MLP recoverability without erasure, after the diffmean projection and after LEACE (both
    fitted on the training states; LEACE on ``erase``, the training labels of every concept to erase jointly, or on
    ``ytr`` alone): per method the per-state ``items`` (the intervals come from the pooled folds, `pool_folds`, so
    none is bootstrapped here)."""
    rows = {}
    for method in METHODS:
        if method == "none":
            tr, ev = Xtr, Xev
        elif method == "diffmean":
            d = diffmean_direction(Xtr, ytr)
            tr, ev = apply_diffmean(d, Xtr), apply_diffmean(d, Xev)
        else:
            eraser = leace_erase(Xtr, ytr if erase is None else erase)
            tr, ev = apply_eraser(eraser, Xtr), apply_eraser(eraser, Xev)
        rows[method] = probe_recoverability(tr, ytr, ev, yev, seed=seed, groups_ev=groups_ev, n_boot=0,
                                            return_items=True, mlp_seeds=MLP_SEEDS)
    return rows


def pool_folds(folds: List[List[Any]], groups: List[int], n_boot: int, seed: int) -> Dict[str, Any]:
    """The eval states' items of every fold pooled (an applicant's states of every fold resampled together; the same
    applicant index in every fold) with intervals, and each fold's point values (``by_fold``)."""
    items = [it for f in folds for it in f]
    point = lambda its: {k: v for k, v in recoverability_summary(its, list(range(len(its))), seed=seed, n_boot=0)
                         .items() if k != "intervals"}
    return {**recoverability_summary(items, groups * len(folds), seed=seed, n_boot=n_boot),
            "by_fold": [point(f) for f in folds]}


def run_domain(exp, cfg, dom, premises: Tuple[str, str, str], probe_recs: List[Any], eval_recs: List[Any], seed: int,
               n_boot: int) -> List[Dict[str, Any]]:
    """Per fold and concept, the probes trained on the control's and the favourable-truth premise's fit states pooled
    (their truth configurations are opposite, so valence and correctness are crossed in the training labels; within
    one premise the valence labels are the correctness labels or their complement) and evaluated on each premise's
    held-out states; the same on the lexical control. Rows per (evaluated premise, concept), pooled over the folds."""
    frame = REASONING_FRAMES[dom.name]
    fitted_on = (frame.control, frame.favourable)
    model = {(p, c, m): [] for p in premises for c in CONCEPTS for m in METHODS}
    lexical = {(p, c, m): [] for p in premises for c in CONCEPTS for m in METHODS}
    eval_ids = None
    for fit, held in PARAPHRASE_FOLDS:
        train = Split(exp.tokenizer, [it for p in fitted_on for it in reasoning_items(dom, p, probe_recs, fit, seed)])
        evals = {p: Split(exp.tokenizer, reasoning_items(dom, p, eval_recs, held, seed)) for p in premises}
        # pooling pairs applicant k of every premise and fold: the same applicants in the same order everywhere
        ids = {tuple(e.record_ids) for e in evals.values()}
        if len(ids) != 1 or (eval_ids is not None and ids != {eval_ids}):
            raise ValueError("every premise and fold must read the same applicants in the same order")
        eval_ids = ids.pop()
        texts = [t for p in premises for t in evals[p].texts]
        Xtr = get_embeddings(exp.model, exp.tokenizer, train.texts, cfg.batch_size, cfg.max_length,
                             show_progress=False)
        Xev = get_embeddings(exp.model, exp.tokenizer, texts, cfg.batch_size, cfg.max_length, show_progress=False)
        Ltr, Lev = bag_of_words(train.verdicts, [v for p in premises for v in evals[p].verdicts])
        size = len(evals[premises[0]].texts)               # every premise reads the same applicants
        for c in CONCEPTS:
            labels = [y for p in premises for y in evals[p].labels[c]]
            groups = [g for p in premises for g in evals[p].groups]
            erase = [train.labels[j] for j in JOINT] if c in JOINT else None
            for store, (X, E) in ((model, (Xtr, Xev)), (lexical, (Ltr, Lev))):
                rows = erasure_rows(X, train.labels[c], E, labels, groups, seed, erase)
                for m in METHODS:
                    for k, p in enumerate(premises):
                        store[(p, c, m)].append(rows[m]["items"][k * size:(k + 1) * size])
    groups = evals[premises[0]].groups
    out = []
    for p in premises:
        for c in CONCEPTS:
            rows = {m: pool_folds(model[(p, c, m)], groups, n_boot, seed) for m in METHODS}
            out.append({"premise": p, "concept": c, "fitted_on": list(fitted_on),
                        "erased_jointly": list(JOINT) if c in JOINT else [c],
                        "favourable_true": frame.premises[p].favourable_true, "n_train": len(train.texts),
                        "n_eval": size, "folds": len(PARAPHRASE_FOLDS), "rows": rows,
                        "lexical_control": {m: pool_folds(lexical[(p, c, m)], groups, n_boot, seed) for m in METHODS},
                        "verdict": erasure_verdict(rows)})
    return out


def erasure_verdict(rows: Dict[str, Any]) -> str:
    """Read a row: the concept must be decodable without erasure (the linear probe's interval above chance), else the
    row says nothing — on the demographic premise, which is never trained on, that means the probes do not transfer
    to it; then the MLP after LEACE: above chance ⇒ non-linearly recoverable (entangled), not ⇒ low-complexity."""
    if not rows["none"]["intervals"]["linear_above_chance"]["ci_low"] > 0:
        return "not decodable without erasure: says nothing"
    if rows["leace"]["intervals"]["mlp_above_chance"]["ci_low"] > 0:
        return "non-linearly recoverable after LEACE (entangled)"
    return "not recoverable after LEACE (low-complexity)"


def default_out(domain: str, model_path: str, variant: str = "") -> Path:
    return RESULTS_DIR / f"erasure_{domain}_{Path(model_path).name}{variant}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_cv_reasoning_qwen06.yaml"))
    ap.add_argument("--probe-items", type=int, default=200, help="Records of the probe (training) split")
    ap.add_argument("--eval-items", type=int, default=200, help="Records of the eval split")
    ap.add_argument("--seed", type=int, default=42, help="Seeds the record order, the paraphrase draws and the MLP")
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
    print(f"[erasure] {dom.name} ...", flush=True)
    results += run_domain(exp, cfg, dom, premises, probe_recs, eval_recs, args.seed, n_boot)

    print("\n" + "=" * 112)
    print(f"REASONING ERASURE — LEACE + non-linear probe — {cfg.model_path}  (probe={len(probe_recs)}, "
          f"eval={len(eval_recs)} applicants; {len(PARAPHRASE_FOLDS)} folds of two held-out entries pooled, no "
          f"connective)")
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
    for r in results:
        print(f"{r['premise']:15} {r['concept']:12} ⇒ {r['verdict']}")
    print("MLP after LEACE: interval clearly above 0 (and above the lexical control) ⇒ non-linearly recoverable "
          "(entangled); reaching 0 or below ⇒ low-complexity — read only where the row is decodable without "
          "erasure.")

    settings = {"probe_items": args.probe_items, "eval_items": args.eval_items, "seed": args.seed, "n_boot": n_boot,
                "premises": list(premises), "paraphrase_folds": [list(map(list, f)) for f in PARAPHRASE_FOLDS],
                "connective": False, "erased_jointly": list(JOINT), "mlp_seeds": MLP_SEEDS}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": run_metadata(cfg, data, settings), "model": cfg.model_path, "domain": dom.name, "seed": args.seed,
         "selection": selection, "probe_records": [str(r.source_record_id) for r in probe_recs],
         "eval_records": [str(r.source_record_id) for r in eval_recs], "results": results}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
