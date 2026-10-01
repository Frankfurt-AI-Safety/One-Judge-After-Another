#!/usr/bin/env python3
"""
Mechanistic probe of the reasoning-flip drivers (hiring, credit, education): are **claim correctness**, the
**advance/reject conclusion** and the claim's **valence** each represented as a *linear direction* in the RM's
activations that can be nulled, measured on applicants AND wording the direction was not fitted on?

Items (`pairs.verdicts.REASONING_FRAMES`; the domain is the config's ``extra.domain``): per premise, one 2×2 item per
applicant, its decision a sentence of its own (no "so"/"but": with a connective, correctness = XOR(connective,
decision), readable without the claim; decided 2026-10-01). Three premises per domain: the demographic one
(``frame.primary``), the non-demographic control and the non-demographic favourable-truth premise. The first two make
the unfavourable claim true, the third the favourable one, so truth and valence (is the claim favourable to the
applicant?) are crossed across premises. Matched contrastive pairs, two per item ("advance" = the domain's
favourable decision; `concept_pairs`):
  - **correctness** = true-claim − false-claim verdict, the conclusion held fixed;
  - **conclusion**  = advance − reject verdict, the claim held fixed;
  - **valence**     = favourable-claim − unfavourable-claim verdict, the conclusion held fixed. Within one premise
    these are the correctness pairs or their reverse, so valence is fitted only across premises.

Directions, per paraphrase fold:
  - per premise: correctness and conclusion, fitted on the premise's fit split;
  - **crossed** (`crossed_directions`): correctness, conclusion and valence fitted on the control's and the
    favourable-truth premise's fit splits pooled with equal weight. Their truth configurations are opposite, so
    valence cancels from the crossed correctness direction and truth from the crossed valence direction — exactly so
    where the two premises carry valence (and truth) with the same size, which they share a claim to make likely;
    neither premise is demographic.

**Held out twice, cross-fitted over the wording.** Every pool has six paraphrase entries; each fold fits on four and
evaluates on the other two (`pairs.verdicts.PARAPHRASE_FOLDS`), so every entry is held out once and the held-out
wording varies within a fold (each eval item draws one of its two entries per pool, seeded by record: the same draw
in every fold's index position, which leaves the held-out entries unseen). The headline pools the three folds (each
eval item read with the direction fitted without its wording), with intervals over records (a record's items of all
folds resampled together; `scoring.demographic_experiment._record_clusters`); ``by_fold`` gives the spread over the
three pairs of held-out entries. The pooled nulling change is the effect of the cross-fitted procedure, conditional
on the probe split (the interval resamples the eval records, not the directions' fits; ``cosines.folds`` shows how
alike the folds' directions are). On the eval split:
(1) **held-out decodability** — the paired accuracy of a direction (P(proj(positive) > proj(negative)), a tie ½),
pooled, per fold and per pair kind;
(2) **held-out nulling** — the 2×2 effects with the direction projected out, with intervals, and nulled − baseline.
The statistics are each premise's own (`scoring.demographic_experiment.reasoning_stats`): interaction is
coherence-relative, and the favourable-truth premise has no correct-but-harmful headline pair;
(3) **cross-premise transfer** — every premise's eval items read and nulled with every other premise's correctness
and conclusion directions, and every premise with the crossed directions;
(4) the **valence check** (`valence_verdict`) of each unfavourable-truth premise's correctness direction: on the other
unfavourable-truth premise (positive control: the same truth configuration) and on the favourable-truth premise
(opposite truth; the control's claim, and in hiring and education the primary's too — credit's primary makes
another claim, so its check also crosses the claim). A truth code stays above ½ there, a valence code falls below; a
positive control at or below ½ means the direction does not transfer between premises and the check says nothing;
an interval covering ½ is a mix of the two or neither. Cosines are point values per fold (no interval).

The eval items' cells share their prompt, so a gated head (QRM) reads the same gate for all four: the effects
compare cells within an item, and no gate-fixed column is needed.

Records: the domain's strong records of the manifest's pool that no premise contradicts
(`run_reasoning_flip.reasoning_records`), in seeded order: the first ``--probe-items`` are the probe split, the next
``--eval-items`` the eval split. No demographic direction is used, so the demographic probe records are not excluded.
The result ``reasoning_probe_{domain}_{model}{variant}.json`` (``variant`` names every setting the CLI changed;
never replaced without ``--overwrite``) carries ``meta`` (the config with the loaded model commit, the code commit,
the SHA-256 of the manifest's cells and of the corpus file; `scoring.experiment`) and both splits' record ids.

Usage:
    python runners/run_reasoning_probe.py --config configs/demographic_cv_reasoning_qwen06.yaml \\
        --probe-items 200 --eval-items 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scoring.dataset_base import format_conversation
from substrates.domains import get_domain
from pairs.verdicts import PARAPHRASE_FOLDS, REASONING_CELLS, REASONING_FRAMES, build_reasoning_item
from scoring.experiment import (
    ExperimentConfig, add_override_args, apply_overrides, run_metadata, variant_suffix,
)
from scoring.demographic_experiment import (
    DemographicBiasExperiment, compute_reasoning_metrics, reasoning_effects, reasoning_intervals,
)
from scoring.intervals import DEFAULT_N_BOOT, cluster_bootstrap, win
from probes.cross_marker_directions import cosine, unit
from probes.probe import embed_with_gates, rewards_from_hidden
from runners.run_decision_response import select_items
from runners.run_reasoning_flip import reasoning_records

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("artifacts/results/demographic")

CONCEPTS = ("correctness", "conclusion", "valence")
# per premise only correctness and conclusion: within one premise the valence pairs are the correctness pairs
# reversed, so valence is fitted only across premises (`crossed_directions`)
PREMISE_CONCEPTS = ("correctness", "conclusion")
# the two pair kinds of each concept, in `concept_pairs` order (what is held fixed)
PAIR_KINDS = {"correctness": ("reject", "advance"), "conclusion": ("true", "false"), "valence": ("reject", "advance")}


def probe_premises(domain: str) -> Tuple[str, str, str]:
    """The demographic premise, the control and the favourable-truth premise of the domain's frame."""
    frame = REASONING_FRAMES[domain]
    return frame.primary, frame.control, frame.favourable


def concept_pairs(concept: str, favourable_true: bool) -> List[Tuple[str, str]]:
    """The concept's matched contrastive pairs (positive cell, negative cell), two per item, holding the other factor
    fixed. Valence depends on which claim the premise makes true."""
    if concept == "correctness":
        return [("true_reject", "false_reject"), ("true_advance", "false_advance")]
    if concept == "conclusion":
        return [("true_advance", "true_reject"), ("false_advance", "false_reject")]
    if concept == "valence":
        fav, unf = ("true", "false") if favourable_true else ("false", "true")
        return [(f"{fav}_reject", f"{unf}_reject"), (f"{fav}_advance", f"{unf}_advance")]
    raise ValueError(f"concept must be one of {CONCEPTS}, got {concept!r}")


def reasoning_items(dom, premise: str, records: List[Any], paraphrases: Sequence[int], seed: int) -> List[Dict]:
    """One varied item per record, its decision a sentence of its own (pools restricted to ``paraphrases``;
    `pairs.verdicts.build_reasoning_item`), its draws seeded by (seed, premise, record) so an item does not depend
    on the others."""
    tids = list(dom.template_ids)
    return [build_reasoning_item(r, premise, dom.render_fn, random.Random(f"{seed}:{premise}:{r.source_record_id}"),
                                 template_id=tids[i % len(tids)], vary=True, paraphrases=paraphrases,
                                 domain=dom.name)
            for i, r in enumerate(records)]


class SplitStates:
    """The states (and gates) of a split's items, embedded once: one [user, verdict] text per (cell, item)."""

    def __init__(self, exp, cfg, items: List[Dict]):
        self.items, self.n = items, len(items)
        self.favourable_true = items[0]["meta"]["favourable_true"]
        flat = [format_conversation(exp.tokenizer, it["user_prompt"], it["cells"][c])
                for c in REASONING_CELLS for it in items]
        self.hidden, self.dtype, self.gates = embed_with_gates(exp.model, exp.tokenizer, flat,
                                                               batch_size=cfg.batch_size, max_length=cfg.max_length,
                                                               show_progress=False)

    def cell(self, c: str) -> torch.Tensor:
        i = REASONING_CELLS.index(c)
        return self.hidden[i * self.n:(i + 1) * self.n]

    def pair_differences(self, concept: str) -> List[torch.Tensor]:
        """positive − negative state of each of the concept's pairs, [n, d] per pair kind."""
        return [self.cell(pos) - self.cell(neg) for pos, neg in concept_pairs(concept, self.favourable_true)]


def fit_direction(splits: Sequence[SplitStates], concept: str) -> torch.Tensor:
    """The unit difference-of-means direction over every pair of ``splits`` (the pairs are matched, so it is the mean
    pair difference)."""
    return unit(torch.cat([d for s in splits for d in s.pair_differences(concept)]).mean(0))


def fold_splits(exp, cfg, dom, premise: str, probe_recs: List[Any], eval_recs: List[Any], seed: int
                ) -> List[Tuple[SplitStates, SplitStates]]:
    """(fit split, eval split) per paraphrase fold: the probe records with the fold's fitted entries, the eval
    records with its held-out entry."""
    return [(SplitStates(exp, cfg, reasoning_items(dom, premise, probe_recs, fit, seed)),
             SplitStates(exp, cfg, reasoning_items(dom, premise, eval_recs, held, seed)))
            for fit, held in PARAPHRASE_FOLDS]


Read = Sequence[Tuple[SplitStates, torch.Tensor]]   # per fold: (states, the direction they are read with)


def _same_records(reads: Read) -> None:
    """Every fold of ``reads`` must read the same records in the same order (pooling pairs record k across folds)."""
    ids = [[it["meta"]["record_id"] for it in states.items] for states, _ in reads]
    if any(i != ids[0] for i in ids[1:]):
        raise ValueError("every fold must read the same records in the same order")


def paired_accuracy(reads: Read, concept: str, n_boot: int, seed: int) -> Dict[str, Any]:
    """P(proj(positive) > proj(negative)) over the concept's pairs, a tie ½, pooled over the folds of ``reads`` (the
    same records in every fold) with its bootstrap interval over records (a record's pairs of every fold resampled
    together); ``by_fold`` (point values) and ``by_pair`` (each pair kind pooled over the folds, with its interval)."""
    _same_records(reads)
    wins = []                                       # fold → record → [win per pair kind]
    for states, u in reads:
        proj = [(states.cell(p) @ u, states.cell(q) @ u) for p, q in concept_pairs(concept, states.favourable_true)]
        wins.append([[win(float(a[k]), float(b[k])) for a, b in proj] for k in range(states.n)])
    if len({len(f) for f in wins}) != 1:
        raise ValueError("every fold must read the same records")
    mean = lambda s: sum(s) / len(s)
    records = range(len(wins[0]))
    out = cluster_bootstrap([[w for f in wins for w in f[k]] for k in records], {"acc": mean}, n_boot, seed)["acc"]
    out["by_fold"] = [mean([w for rec in f for w in rec]) for f in wins]
    out["by_pair"] = {kind: cluster_bootstrap([[f[k][j] for f in wins] for k in records], {"acc": mean}, n_boot,
                                              seed)["acc"]
                      for j, kind in enumerate(PAIR_KINDS[concept])}
    return out


def _by_cell(values: Any, n: int) -> Dict[str, List[float]]:
    return {c: values[i * n:(i + 1) * n].tolist() for i, c in enumerate(REASONING_CELLS)}


def _pooled(folds: Sequence[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    return {c: [x for f in folds for x in f[c]] for c in REASONING_CELLS}


def fold_rewards(exp, reads: Read) -> Tuple[List[Dict[str, List[float]]], List[Dict[str, List[float]]]]:
    """The baseline and the nulled rewards by cell, per fold (each fold's states nulled with its direction)."""
    _same_records(reads)
    base, null = [], []
    for states, u in reads:
        b, x = rewards_from_hidden(exp.model, states.hidden, states.dtype, u, gates=states.gates)
        base.append(_by_cell(b, states.n))
        null.append(_by_cell(x, states.n))
    return base, null


def nulled_effects(exp, reads: Read, n_boot: int, seed: int, with_baseline: bool = True) -> Dict[str, Any]:
    """The 2×2 metrics with each fold's direction projected out, pooled over the folds, and the intervals (baseline,
    nulled, nulled − baseline; records resampled with all their folds). The pooled change is the effect of the
    cross-fitted procedure (three directions, one per fold), conditional on the probe split: the interval resamples
    the eval records, not the directions' fits. ``nulled_minus_baseline_by_fold`` gives the change per fold (point
    values). The statistics are those of the read premise (`scoring.demographic_experiment.reasoning_stats`)."""
    favourable_true = reads[0][0].favourable_true
    base, null = fold_rewards(exp, reads)
    effects = reasoning_effects(favourable_true)
    by_fold = []
    for b, x in zip(base, null):
        mb, mx = (compute_reasoning_metrics(v, favourable_true) for v in (b, x))
        by_fold.append({k: mx[k] - mb[k] for k in effects})
    return {"nulled": compute_reasoning_metrics(_pooled(null), favourable_true),
            "intervals": reasoning_intervals(base, null, n_boot, seed, favourable_true, with_baseline),
            "nulled_minus_baseline_by_fold": by_fold}


def read_with(splits: Sequence[Tuple[SplitStates, SplitStates]], directions: Sequence[Dict[str, torch.Tensor]],
              concept: str) -> Read:
    """Each fold's eval split with the concept's direction of the same fold."""
    return [(ev, d[concept]) for (_, ev), d in zip(splits, directions)]


def run_premise(exp, cfg, dom, premise: str, probe_recs: List[Any], eval_recs: List[Any], seed: int, n_boot: int
                ) -> Tuple[Dict[str, Any], List[Dict[str, torch.Tensor]], List[Tuple[SplitStates, SplitStates]]]:
    """One premise: its correctness and conclusion directions per fold (fitted on the fold's probe split), their
    held-out accuracy and nulling, pooled over the folds. Returns the result, the directions per fold and the splits
    (for the crossed directions and the cross-premise transfer)."""
    splits = fold_splits(exp, cfg, dom, premise, probe_recs, eval_recs, seed)
    directions = [{c: fit_direction([fit], c) for c in PREMISE_CONCEPTS} for fit, _ in splits]
    spec = REASONING_FRAMES[dom.name].premises[premise]
    base = [_by_cell(rewards_from_hidden(exp.model, ev.hidden, ev.dtype, None, gates=ev.gates)[0], ev.n)
            for _, ev in splits]
    out: Dict[str, Any] = {"premise": premise, "demographic": spec.demographic,
                           "favourable_true": spec.favourable_true, "n_probe": len(probe_recs),
                           "n_eval": len(eval_recs),
                           "baseline": compute_reasoning_metrics(_pooled(base), spec.favourable_true),
                           "directions": {}}
    for c in PREMISE_CONCEPTS:
        fitted = [(fit, d[c]) for (fit, _), d in zip(splits, directions)]
        out["directions"][c] = {
            # on the pairs the direction was fitted on, for comparison only
            "paired_acc_fit_split": paired_accuracy(fitted, c, n_boot, seed)["estimate"],
            "paired_acc_heldout": paired_accuracy(read_with(splits, directions, c), c, n_boot, seed),
            **nulled_effects(exp, read_with(splits, directions, c), n_boot, seed)}
    out["cosines"] = {
        "correctness_conclusion": [cosine(d["correctness"], d["conclusion"]) for d in directions],
        # how alike the three folds' directions are (each fitted without one pair of wordings)
        "folds": {c: [cosine(directions[i][c], directions[j][c]) for i, j in ((0, 1), (0, 2), (1, 2))]
                  for c in PREMISE_CONCEPTS}}
    return out, directions, splits


def crossed_directions(control: Sequence[Tuple[SplitStates, SplitStates]],
                       favourable: Sequence[Tuple[SplitStates, SplitStates]]) -> List[Dict[str, torch.Tensor]]:
    """Per fold, every concept's direction fitted on the control's and the favourable-truth premise's fit splits
    pooled (equal sizes, so equal weight): their truth configurations are opposite, so valence cancels from the
    correctness direction and truth from the valence direction; neither premise is demographic."""
    return [{c: fit_direction([ctl_fit, fav_fit], c) for c in CONCEPTS}
            for (ctl_fit, _), (fav_fit, _) in zip(control, favourable)]


def valence_verdict(positive: Dict[str, Any], test: Dict[str, Any]) -> str:
    """Read the valence check from two paired accuracies of one premise's correctness direction: ``positive`` on a
    premise with the same truth configuration (the positive control) and ``test`` on the favourable-truth premise
    (truth and valence come apart). A direction of truth stays above ½ on ``test``, one of valence falls below; a
    positive control at or below ½ means the direction does not transfer between premises at all, and the check says
    nothing."""
    if not positive["ci_low"] > 0.5:
        return "inconclusive: the direction does not transfer to a premise with the same truth configuration"
    if test["ci_low"] > 0.5:
        return "truth: above ½ where truth and valence come apart"
    if test["ci_high"] < 0.5:
        return "valence: below ½ where truth and valence come apart"
    return "inconclusive: covers ½ where truth and valence come apart (a mix of truth and valence, or neither)"


def default_out(domain: str, model_path: str, variant: str = "") -> Path:
    return RESULTS_DIR / f"reasoning_probe_{domain}_{Path(model_path).name}{variant}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_cv_reasoning_qwen06.yaml"))
    ap.add_argument("--probe-items", type=int, default=200, help="Records of the probe (fit) split")
    ap.add_argument("--eval-items", type=int, default=200, help="Records of the eval split")
    ap.add_argument("--seed", type=int, default=42, help="Seeds the record order and the paraphrase draws")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"Default {RESULTS_DIR}/reasoning_probe_{{domain}}_{{model}}{{variant}}.json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result")
    add_override_args(ap)
    return ap


def _ci(entry: Dict[str, float]) -> str:
    return f"{entry['estimate']:+.3f} [{entry['ci_low']:+.3f}, {entry['ci_high']:+.3f}]"


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

    results, dirs, splits = [], {}, {}
    for premise in premises:
        print(f"[reasoning-probe] {dom.name}/{premise} ...", flush=True)
        result, dirs[premise], splits[premise] = run_premise(exp, cfg, dom, premise, probe_recs, eval_recs,
                                                             args.seed, n_boot)
        results.append(result)

    # cross-premise transfer: every premise's eval items read and nulled with every other premise's directions
    transfer: Dict[str, Any] = {}
    for src in premises:
        for tgt in premises:
            if src != tgt:
                transfer[f"{src}_on_{tgt}"] = {
                    c: {"paired_acc": paired_accuracy(read_with(splits[tgt], dirs[src], c), c, n_boot, args.seed),
                        **nulled_effects(exp, read_with(splits[tgt], dirs[src], c), n_boot, args.seed)}
                    for c in PREMISE_CONCEPTS}
    transfer["cosines"] = {c: {a: {b: [cosine(x[c], y[c]) for x, y in zip(dirs[a], dirs[b])] for b in premises}
                               for a in premises} for c in PREMISE_CONCEPTS}

    primary, control, favourable = premises
    # the crossed directions (control + favourable-truth premise pooled), read on every premise
    crossed_dirs = crossed_directions(splits[control], splits[favourable])
    crossed = {"fitted_on": [control, favourable], "on": {
        p: {c: {"paired_acc": paired_accuracy(read_with(splits[p], crossed_dirs, c), c, n_boot, args.seed),
                **nulled_effects(exp, read_with(splits[p], crossed_dirs, c), n_boot, args.seed)} for c in CONCEPTS}
        for p in premises},
        "cosines": {"correctness_valence": [cosine(d["correctness"], d["valence"]) for d in crossed_dirs],
                    "correctness_to_premise": {p: [cosine(d["correctness"], x["correctness"])
                                                   for d, x in zip(crossed_dirs, dirs[p])] for p in premises}}}
    # the valence check of each unfavourable-truth premise's correctness direction: on the other unfavourable-truth
    # premise (positive control) and on the favourable-truth premise (truth and valence come apart)
    valence_check = {}
    for src, other in ((control, primary), (primary, control)):
        pos = transfer[f"{src}_on_{other}"]["correctness"]["paired_acc"]
        test = transfer[f"{src}_on_{favourable}"]["correctness"]["paired_acc"]
        valence_check[src] = {"positive_control": f"{src}_on_{other}", "test": f"{src}_on_{favourable}",
                              "verdict": valence_verdict(pos, test)}

    print("\n" + "=" * 104)
    print(f"REASONING-PROBE — held-out applicants and wording, {len(PARAPHRASE_FOLDS)} folds — {cfg.model_path}  "
          f"(probe={len(probe_recs)}, eval={len(eval_recs)} records)")
    print("=" * 104)
    for r in results:
        print(f"\n[{r['premise']}]  favourable claim true: {r['favourable_true']}")
        for c, d in r["directions"].items():
            iv = d["intervals"]["nulled_minus_baseline"]
            print(f"  {c:11} held-out paired acc {_ci(d['paired_acc_heldout'])} by fold "
                  f"{[round(x, 3) for x in d['paired_acc_heldout']['by_fold']]}")
            print(f"    nulled − baseline: correctness {_ci(iv['correctness_effect'])}  "
                  f"conclusion {_ci(iv['conclusion_effect'])}")
        for c in ("correctness", "valence"):
            print(f"  crossed {c:11} paired acc {_ci(crossed['on'][r['premise']][c]['paired_acc'])}")
    print("\n" + "-" * 104)
    for src, v in valence_check.items():
        print(f"valence check — {src}'s correctness direction: {v['positive_control']} "
              f"{_ci(transfer[v['positive_control']]['correctness']['paired_acc'])}, {v['test']} "
              f"{_ci(transfer[v['test']]['correctness']['paired_acc'])} ⇒ {v['verdict']}")
    print("Intervals resample the eval records; the held-out wording varies within each fold over two paraphrase "
          "entries, and by_fold shows the spread over the three pairs of entries.")
    print("=" * 104)

    settings = {"probe_items": args.probe_items, "eval_items": args.eval_items, "seed": args.seed, "n_boot": n_boot,
                "premises": list(premises), "paraphrase_folds": [list(map(list, f)) for f in PARAPHRASE_FOLDS],
                "connective": False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": run_metadata(cfg, data, settings), "model": cfg.model_path, "domain": dom.name, "seed": args.seed,
         "selection": selection, "probe_records": [str(r.source_record_id) for r in probe_recs],
         "eval_records": [str(r.source_record_id) for r in eval_recs], "results": results, "transfer": transfer,
         "crossed": crossed, "valence_check": valence_check},
        indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
