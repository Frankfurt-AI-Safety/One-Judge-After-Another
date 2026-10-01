#!/usr/bin/env python3
"""
Mechanistic probe of the reasoning-flip drivers (hiring, credit, education): are **reasoning correctness** and the
**advance/reject conclusion** each represented as a *linear direction* in the RM's activations that can be nulled,
measured on applicants AND wording the direction was not fitted on?

The 2×2 (`pairs.verdicts.REASONING_FRAMES`; the domain is the config's ``extra.domain``) gives matched contrastive
pairs, two per item ("advance" = the domain's favourable decision):
  - **correctness direction** = DiffMean(true-claim − false-claim verdicts), holding the conclusion fixed;
  - **conclusion direction**  = DiffMean(advance − reject verdicts), holding correctness fixed.
Each pair pairs a "so" with a "but" connective and the two pairs swap them, so the connective averages out.

**Held out twice.** The directions are fitted on a probe split of applicants whose verdicts draw paraphrases
``FIT_PARAPHRASES`` of every pool (the claim stems and the four connective pools; entry 0 is close to the flip's
fixed wording), and measured on a disjoint eval split whose verdicts use ``EVAL_PARAPHRASES``, wording the
direction never saw (until 2026-09-30 both splits used the one fixed wording, so "held out" held only the
applicant out and the direction was the difference of two fixed strings). On the eval split:
(1) **held-out decodability** — the paired accuracy of the direction (P(proj(positive) > proj(negative)), a tie ½),
with a record-bootstrap interval (two pairs per record);
(2) **held-out nulling** — does projecting out the correctness direction collapse the eval correctness effect
while leaving the conclusion effect (and vice versa)? Each metric with its interval, and nulled − baseline;
(3) **cross-premise transfer** — the demographic premise's eval items (``frame.primary``: hiring parental leave,
credit age, education low income) nulled with the *control's* direction (six months abroad, an unpaid sabbatical, the
out-of-district fee) and vice
versa: another premise subject on top of the held-out wording. In hiring and education the two premises share the
claim's paraphrase pools, so this does not test another claim form, and the directions' cosine across premises is
expected to be high for that reason; credit's control makes another claim (income vs credit history), so there the
transfer also crosses the claim. Cosines are point values (length-type, no bootstrap interval).

The eval items' cells share their prompt, so a gated head (QRM) reads the same gate for all four: the effects
compare cells within an item, and no gate-fixed column is needed.

Records: the domain's strong records of the manifest's pool that no premise contradicts
(`run_reasoning_flip.reasoning_records`), in seeded order: the first ``--probe-items`` are the probe split, the next
``--eval-items`` the eval split. No demographic direction is used, so the demographic probe records are not excluded.
The result ``reasoning_probe_{domain}_{model}{variant}.json`` (``variant`` names every setting the CLI changed;
never replaced without ``--overwrite``) carries ``meta`` (the config with the loaded model commit, the code commit,
the SHA-256 of the manifest's cells and of the corpus file; `scoring.experiment`) and both splits' record ids.

Usage:
    python runners/run_reasoning_probe.py --config configs/demographic_cv_reasoning_qwen06.yaml \
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
from pairs.verdicts import (
    EVAL_PARAPHRASES, FIT_PARAPHRASES, REASONING_CELLS, REASONING_FRAMES, build_reasoning_item,
)
from scoring.experiment import (
    ExperimentConfig, add_override_args, apply_overrides, run_metadata, variant_suffix,
)
from scoring.demographic_experiment import DemographicBiasExperiment, compute_reasoning_metrics, reasoning_intervals
from scoring.intervals import DEFAULT_N_BOOT, cluster_bootstrap, win
from probes.cross_marker_directions import cosine, unit
from probes.probe import embed_with_gates, rewards_from_hidden
from runners.run_decision_response import select_items
from runners.run_reasoning_flip import reasoning_records

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("artifacts/results/demographic")


def probe_premises(domain: str) -> Tuple[str, str]:
    """The demographic premise and the non-demographic control of the domain's frame."""
    frame = REASONING_FRAMES[domain]
    return frame.primary, frame.control


# matched contrastive pairs (positive cell, negative cell): two per item, holding the other factor fixed
CONCEPTS = {"correctness": [("true_reject", "false_reject"), ("true_advance", "false_advance")],
            "conclusion": [("true_advance", "true_reject"), ("false_advance", "false_reject")]}


def reasoning_items(dom, premise: str, records: List[Any], paraphrases: Sequence[int], seed: int,
                    connective: bool = True) -> List[Dict]:
    """One varied item per record (pools restricted to ``paraphrases``; `pairs.verdicts.build_reasoning_item`), its
    draws seeded by (seed, premise, record) so an item does not depend on the others."""
    tids = list(dom.template_ids)
    return [build_reasoning_item(r, premise, dom.render_fn, random.Random(f"{seed}:{premise}:{r.source_record_id}"),
                                 template_id=tids[i % len(tids)], vary=True, paraphrases=paraphrases,
                                 connective=connective, domain=dom.name)
            for i, r in enumerate(records)]


class SplitStates:
    """The states (and gates) of a split's items, embedded once: one [user, verdict] text per (cell, item)."""

    def __init__(self, exp, cfg, items: List[Dict]):
        self.n = len(items)
        flat = [format_conversation(exp.tokenizer, it["user_prompt"], it["cells"][c])
                for c in REASONING_CELLS for it in items]
        self.hidden, self.dtype, self.gates = embed_with_gates(exp.model, exp.tokenizer, flat, batch_size=cfg.batch_size,
                                                               max_length=cfg.max_length, show_progress=False)

    def cell(self, c: str) -> torch.Tensor:
        i = REASONING_CELLS.index(c)
        return self.hidden[i * self.n:(i + 1) * self.n]

    def pair_differences(self, concept: str) -> List[torch.Tensor]:
        """positive − negative state of each of the concept's pairs, [n, d] per pair kind."""
        return [self.cell(pos) - self.cell(neg) for pos, neg in CONCEPTS[concept]]


def fit_direction(states: SplitStates, concept: str) -> torch.Tensor:
    """The unit difference-of-means direction (the pairs are matched, so it is the mean pair difference)."""
    return unit(torch.cat(states.pair_differences(concept)).mean(0))


def paired_accuracy(states: SplitStates, concept: str, direction: torch.Tensor, n_boot: int, seed: int
                    ) -> Dict[str, float]:
    """P(proj(positive) > proj(negative)) over the concept's pairs, a tie ½, with its bootstrap interval over
    records (a record's two pairs resampled together)."""
    proj = [(pos @ direction, neg @ direction) for pos, neg in
            ((states.cell(p), states.cell(q)) for p, q in CONCEPTS[concept])]
    clusters = [[win(float(a[k]), float(b[k])) for a, b in proj] for k in range(states.n)]
    return cluster_bootstrap(clusters, {"acc": lambda s: sum(s) / len(s)}, n_boot, seed)["acc"]


def _by_cell(values: Any, n: int) -> Dict[str, List[float]]:
    return {c: values[i * n:(i + 1) * n].tolist() for i, c in enumerate(REASONING_CELLS)}


def nulled_effects(exp, states: SplitStates, direction: torch.Tensor, n_boot: int, seed: int) -> Dict[str, Any]:
    """The 2×2 metrics on ``states`` with ``direction`` projected out, and the intervals (baseline, nulled,
    nulled − baseline)."""
    base, nulled = rewards_from_hidden(exp.model, states.hidden, states.dtype, direction, gates=states.gates)
    null_by = _by_cell(nulled, states.n)
    return {"nulled": compute_reasoning_metrics(null_by),
            "intervals": reasoning_intervals(_by_cell(base, states.n), null_by, n_boot, seed)}


def run_premise(exp, cfg, dom, premise: str, probe_recs: List[Any], eval_recs: List[Any], seed: int, n_boot: int
                ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], SplitStates]:
    """One premise: its directions (fitted on the probe split), their held-out accuracy and nulling on the eval
    split. Returns the result, the directions and the eval states (for the cross-premise transfer)."""
    probe = SplitStates(exp, cfg, reasoning_items(dom, premise, probe_recs, FIT_PARAPHRASES, seed))
    evals = SplitStates(exp, cfg, reasoning_items(dom, premise, eval_recs, EVAL_PARAPHRASES, seed))
    base, _ = rewards_from_hidden(exp.model, evals.hidden, evals.dtype, None, gates=evals.gates)
    dirs = {concept: fit_direction(probe, concept) for concept in CONCEPTS}
    out: Dict[str, Any] = {"premise": premise,
                           "demographic": REASONING_FRAMES[dom.name].premises[premise].demographic,
                           "n_probe": probe.n, "n_eval": evals.n,
                           "baseline": compute_reasoning_metrics(_by_cell(base, evals.n)), "directions": {}}
    for concept, u in dirs.items():
        out["directions"][concept] = {
            # on the pairs the direction was fitted on, for comparison only
            "paired_acc_probe_split": paired_accuracy(probe, concept, u, n_boot, seed)["estimate"],
            "paired_acc_heldout": paired_accuracy(evals, concept, u, n_boot, seed),
            **nulled_effects(exp, evals, u, n_boot, seed)}
    out["cos_correctness_conclusion"] = cosine(dirs["correctness"], dirs["conclusion"])
    return out, dirs, evals


def default_out(domain: str, model_path: str, variant: str = "") -> Path:
    return RESULTS_DIR / f"reasoning_probe_{domain}_{Path(model_path).name}{variant}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_cv_reasoning_qwen06.yaml"))
    ap.add_argument("--probe-items", type=int, default=200)
    ap.add_argument("--eval-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
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
    primary, control = probe_premises(domain)
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

    results, dirs, evals = [], {}, {}
    for premise in (primary, control):
        print(f"[reasoning-probe] {dom.name}/{premise} ...", flush=True)
        result, dirs[premise], evals[premise] = run_premise(exp, cfg, dom, premise, probe_recs, eval_recs, args.seed,
                                                            n_boot)
        results.append(result)

    # cross-premise transfer: premise X's eval cells nulled with premise Y's directions
    transfer: Dict[str, Any] = {
        f"{src}_on_{tgt}": {concept: nulled_effects(exp, evals[tgt], dirs[src][concept], n_boot, args.seed)
                            for concept in CONCEPTS}
        for tgt, src in ((primary, control), (control, primary))}
    transfer["cosines"] = {concept: cosine(dirs[primary][concept], dirs[control][concept]) for concept in CONCEPTS}

    print("\n" + "=" * 104)
    print(f"REASONING-PROBE — held-out applicants and wording — {cfg.model_path}  "
          f"(probe={len(probe_recs)} with paraphrases {FIT_PARAPHRASES}, eval={len(eval_recs)} with "
          f"{EVAL_PARAPHRASES})")
    print("=" * 104)
    for r in results:
        print(f"\n[{r['premise']}]  cos(correctness_dir, conclusion_dir) = {r['cos_correctness_conclusion']:+.3f}")
        for concept, d in r["directions"].items():
            iv = d["intervals"]
            print(f"  {concept:11} held-out paired acc {_ci(d['paired_acc_heldout'])} "
                  f"(probe split {d['paired_acc_probe_split']:.3f})")
            print(f"    nulled: correctness_effect {_ci(iv['nulled']['correctness_effect'])}  "
                  f"conclusion_effect {_ci(iv['nulled']['conclusion_effect'])}")
        b = r["directions"]["correctness"]["intervals"]["baseline"]
        print(f"  baseline: correctness_effect {_ci(b['correctness_effect'])}  "
              f"conclusion_effect {_ci(b['conclusion_effect'])}")
    print("\n" + "-" * 104)
    print(f"cross-premise cosines: {transfer['cosines']}")
    for key in (f"{control}_on_{primary}", f"{primary}_on_{control}"):
        ch = transfer[key]["correctness"]["intervals"]["nulled_minus_baseline"]["correctness_effect"]
        print(f"  {key}: correctness_effect nulled − baseline {_ci(ch)}")
    print("=" * 104)
    print("held-out null(correctness) collapses correctness_effect and spares conclusion_effect ⇒ a separable "
          "direction that carries over to unseen wording.")

    settings = {"probe_items": args.probe_items, "eval_items": args.eval_items, "seed": args.seed, "n_boot": n_boot,
                "premises": [primary, control], "fit_paraphrases": list(FIT_PARAPHRASES),
                "eval_paraphrases": list(EVAL_PARAPHRASES)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": run_metadata(cfg, data, settings), "model": cfg.model_path, "domain": dom.name, "seed": args.seed,
         "selection": selection, "probe_records": [str(r.source_record_id) for r in probe_recs],
         "eval_records": [str(r.source_record_id) for r in eval_recs], "results": results, "transfer": transfer},
        indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
