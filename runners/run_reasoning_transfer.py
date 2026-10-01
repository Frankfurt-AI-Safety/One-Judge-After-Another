#!/usr/bin/env python3
"""
Cross-domain transfer of the reasoning directions: does a **correctness** (or **conclusion**, or **valence**)
direction fitted on one domain's reasoning 2×2 work on another domain's items, where the claim, the decision, the
record and every phrase differ?

Within a domain the reasoning probe (`run_reasoning_probe.py`) holds the applicants and the wording out; across
domains nothing of the claim or the decision phrase is shared. Two confounds are designed out (decided 2026-10-01):
with a connective ("so"/"but") correctness = XOR(connective, decision), the same structure in every domain and
readable without the claim, so none is used; and every domain has a favourable-truth premise beside the two whose
unfavourable claim is true, so a direction of *valence* (is the claim good for the applicant?) cannot pass for one of
correctness: the correctness direction is fitted valence-balanced, and the favourable-truth premise's rows show
whether it holds where truth and valence come apart.

Per domain the items are exactly the probe runner's (`run_reasoning_probe.reasoning_items`, `fold_splits`: the same
records, split, seed and paraphrase folds), so run after it with the same settings on the same embedding cache this
runner needs no forward pass of its own. Each fold fits on four of every pool's six paraphrase entries and evaluates
on the other two (`pairs.verdicts.PARAPHRASE_FOLDS`); every statistic pools the folds (each eval item read with the
direction fitted without its wording; intervals resample records with all their folds) and gives ``by_fold``.

Fit sources, per fold (`source_members`, `source_direction`):
  - ``{domain}``: the domain's three premises, valence-balanced (`domain_direction`: the mean pair difference of the
    two unfavourable-truth premises and that of the favourable-truth premise, averaged). Valence cancels exactly where
    every premise makes one claim (hiring, education); credit's unfavourable-truth group pools two claims (credit
    history, income) against one (income), so a valence residual of the two claims' difference remains;
  - ``{domain}:nondemographic``: the control and the favourable-truth premise only — the same claim with opposite
    truth, so exactly balanced, and no demographic content (a pair holds the premise fixed, but a premise × claim
    interaction such as "a negative claim about a protected group" could otherwise ride along). Credit's clean source;
  - ``others`` (with three or more domains): the unit directions of every domain but the evaluated one, averaged, so
    each domain weighs the same (leave one domain out).

Every (source, evaluated domain, evaluated premise, concept) row, on the evaluated domain's eval split:
  - ``paired_acc``: P(proj(positive) > proj(negative)) of the source's direction, a tie ½, pooled with its
    record-bootstrap interval, ``by_fold`` and ``by_pair`` (each pair kind; `run_reasoning_probe.paired_accuracy`);
  - ``lexical_paired_acc``: the same on bag-of-words vectors of the verdicts (`expected_lexical`): the vocabulary from
    the source's fitted wording, the direction from its *expected* bag-of-words (every cell averaged over every
    combination of the fold's fitted stem and decision entries), so it carries no noise from which entries the
    records drew; where valence balancing cancels the words of true and false claims it is 0 and every pair ties (½).
    Words recur across domains ("reduce"/"increase" in hiring's and credit's income claims, "shrink", "more"/"less"
    with the polarity of their claim), and "not" in most reject sentences carries the conclusion in every domain;
  - ``nulled`` / ``intervals``: the evaluated items' 2×2 effects with the source's direction projected out
    (`run_reasoning_probe.nulled_effects`; the evaluated premise's own statistics); the baseline intervals once per
    (domain, premise) in ``baselines``;
  - ``transfer_gap`` (any source but the evaluated domain itself): the evaluated domain's own direction's nulling
    change minus the source's, paired by record (`scoring.demographic_experiment.reasoning_nulling_contrast_intervals`):
    a transfer read on the evaluated domain's own scale. Read only where the diagonal row's own change clearly differs
    from 0 (a direction that nulls nothing leaves a gap near 0 for any source).
Read: a transfer means more than shared wording where the model's paired accuracy exceeds the lexical control's (at
least; ties stated). The source directions are valence-balanced, so a pure valence code reads ½ on every premise
rather than below ½ on the favourable-truth one (the probe's single-premise check does that); here a difference
between the unfavourable- and favourable-truth premises' rows is residual valence. ``cosines``: the domains'
directions per concept, per fold (point values).

The four cells of an item share their prompt, so a gated head (QRM) reads one gate per item and no gate-fixed column
is needed (as in the probe runner).

Inputs: one reasoning config per domain (``--configs``), all naming the same model and revision after the CLI
overrides and the same cache, device, remote-code setting and ``n_boot``; each domain reads its own manifest, corpus,
``max_length`` and ``batch_size``. The model loads once; an input longer than its domain's ``max_length`` is refused
when embedded, after the load (as in the probe runner). The result ``reasoning_transfer_{model}{variant}.json``
(``variant`` names every setting the CLI changed, the domain set included; never replaced without ``--overwrite``)
carries ``meta`` (the first config with the loaded model commit, every config, the code commit, each domain's cells
and corpus SHA-256) and every split's record ids.

Usage:
    python runners/run_reasoning_transfer.py \\
        --configs configs/demographic_cv_reasoning_qwen06.yaml configs/demographic_credit_reasoning_qwen06.yaml \\
                  configs/demographic_edu_reasoning_asap2_qwen06.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from substrates.domains import get_domain
from pairs.verdicts import (
    PARAPHRASE_FOLDS, REASONING_CELLS, REASONING_FRAMES, reasoning_cell_text, reasoning_stem,
)
from scoring.experiment import (
    ExperimentConfig, add_override_args, apply_overrides, run_metadata, variant_suffix,
)
from scoring.demographic_experiment import (
    DemographicBiasExperiment, reasoning_intervals, reasoning_nulling_contrast_intervals,
)
from scoring.intervals import DEFAULT_N_BOOT
from probes.cross_marker_directions import cosine, unit
from probes.erasure import bag_of_words
from probes.probe import rewards_from_hidden
from runners.run_decision_response import select_items
from runners.run_reasoning_flip import reasoning_records
from runners.run_reasoning_probe import (
    CONCEPTS, SplitStates, _by_cell, fold_rewards, fold_splits, nulled_effects, paired_accuracy, probe_premises,
)

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("artifacts/results/demographic")
DEFAULT_CONFIGS = [Path("configs/demographic_cv_reasoning_qwen06.yaml"),
                   Path("configs/demographic_credit_reasoning_qwen06.yaml"),
                   Path("configs/demographic_edu_reasoning_asap2_qwen06.yaml")]
DEFAULT_DOMAINS = ["cv", "credit", "education"]
OTHERS = "others"                 # the fit source averaging every domain but the evaluated one
NONDEMOGRAPHIC = ":nondemographic"
# settings every config must share: they are read from the first config only
SHARED = ("embedding_cache_dir", "device", "trust_remote_code")


class Vectors(SplitStates):
    """A split's vectors given directly (cell-major, ``n`` items per cell): the lexical control's bag-of-words."""

    def __init__(self, hidden: torch.Tensor, n: int, favourable_true: bool):
        self.hidden, self.n, self.favourable_true = hidden, n, favourable_true
        self.items, self.dtype, self.gates = [], hidden.dtype, None


def domain_direction(splits: Sequence[SplitStates], concept: str) -> torch.Tensor:
    """The unit direction of ``splits`` (one fold's fit splits of a domain's premises), valence-balanced: the mean pair
    difference of the unfavourable-truth splits and that of the favourable-truth splits, averaged (one group alone
    if only one is given). Valence cancels exactly where every premise shares one claim (hiring, education); in
    credit the unfavourable-truth group pools two claims (credit history, income) and the favourable one has only
    income, so a residual valence of the two claims' difference remains (``credit:nondemographic`` has none)."""
    groups: Dict[bool, List[torch.Tensor]] = {}
    for s in splits:
        groups.setdefault(s.favourable_true, []).extend(s.pair_differences(concept))
    return unit(torch.stack([torch.cat(g).mean(0) for g in groups.values()]).mean(0))


def others_direction(directions: Sequence[torch.Tensor]) -> torch.Tensor:
    """The unit mean of several domains' unit directions: every domain weighs the same."""
    return unit(torch.stack([unit(d) for d in directions]).mean(0))


def source_members(source: str, evaluated: str, domains: Sequence[str]) -> List[Tuple[str, Tuple[str, ...]]]:
    """The (domain, premises) a source's direction is fitted on: a domain's three premises; its control and
    favourable-truth premise (``:nondemographic``); or every other domain's three premises (``others``)."""
    if source == OTHERS:
        return [(d, probe_premises(d)) for d in domains if d != evaluated]
    if source.endswith(NONDEMOGRAPHIC):
        d = source[:-len(NONDEMOGRAPHIC)]
        return [(d, probe_premises(d)[1:])]
    return [(source, probe_premises(source))]


def source_direction(members: Sequence[Sequence[SplitStates]], concept: str) -> torch.Tensor:
    """A source's direction from its members' fit splits (one list per member domain): one member's
    valence-balanced direction, or several members' averaged with equal weight (`others_direction`)."""
    directions = [domain_direction(m, concept) for m in members]
    return directions[0] if len(directions) == 1 else others_direction(directions)


def expected_lexical(members: Sequence[Tuple[str, Sequence[str]]], fit_entries: Sequence[int],
                     evaluated: Sequence[List[Dict]]) -> Tuple[List[List[Vectors]], List[Vectors]]:
    """The lexical control's vectors: bag-of-words, the vocabulary from the source's fit wording. Each fit premise is
    one expected item — every cell's vector averaged over every combination of its fold's fitted stem and decision
    entries — so the lexical direction carries no noise from which entries the records happened to draw (with the
    drawn items, a direction that cancels in expectation was mostly that noise, blown up to unit length). Returns the
    fit vectors (one list per member domain) and the evaluated items' vectors (per item list)."""
    fit_texts, layout = [], []
    for d, premises in members:
        frame = REASONING_FRAMES[d]
        for p in premises:
            for cell in REASONING_CELLS:
                decisions = frame.advance_decision if cell.endswith("_advance") else frame.reject_decision
                fit_texts += [reasoning_cell_text(reasoning_stem(p, d, cell.startswith("true_"), e), decisions[k])
                              for e in fit_entries for k in fit_entries]
            layout.append((d, frame.premises[p].favourable_true))
    combos = len(fit_entries) ** 2
    ev_texts = [it["cells"][c] for items in evaluated for c in REASONING_CELLS for it in items]   # cell-major
    Xfit, Xev = bag_of_words(fit_texts, ev_texts)
    per_premise = Xfit.reshape(len(layout), len(REASONING_CELLS), combos, -1).mean(2)
    fit: List[List[Vectors]] = []
    for (d, favourable_true), X in zip(layout, per_premise):
        if not fit or fit[-1][0] != d:
            fit.append([d, []])
        fit[-1][1].append(Vectors(X, 1, favourable_true))
    out, start = [], 0
    for items in evaluated:
        size = len(REASONING_CELLS) * len(items)
        out.append(Vectors(Xev[start:start + size], len(items), items[0]["meta"]["favourable_true"]))
        start += size
    return [vectors for _, vectors in fit], out


def transfer_gap(exp, own: Sequence[tuple], base: Sequence[Dict[str, List[float]]],
                 null_own: Sequence[Dict[str, List[float]]], other: Sequence[tuple], n_boot: int, seed: int
                 ) -> Dict[str, Any]:
    """The evaluated domain's own direction's nulling change minus another's, on the same items and folds, paired by
    record (``base``/``null_own``: the cached rewards of the own direction's read ``own``; ``other``: per fold the
    eval split and the source's direction)."""
    _, null_other = fold_rewards(exp, other)
    return reasoning_nulling_contrast_intervals(base, null_own, base, null_other, n_boot, seed,
                                                own[0][0].favourable_true)


def default_out(model_path: str, variant: str = "") -> Path:
    return RESULTS_DIR / f"reasoning_transfer_{Path(model_path).name}{variant}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", type=Path, nargs="+", default=DEFAULT_CONFIGS,
                    help="One reasoning config per domain (default: hiring, credit, education)")
    ap.add_argument("--probe-items", type=int, default=200, help="Records of each domain's probe (fit) split")
    ap.add_argument("--eval-items", type=int, default=200, help="Records of each domain's eval split")
    ap.add_argument("--seed", type=int, default=42, help="Seeds the record order and the paraphrase draws")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"Default {RESULTS_DIR}/reasoning_transfer_{{model}}{{variant}}.json")
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
    configured = [ExperimentConfig.from_yaml(p) for p in args.configs]
    cfgs = [apply_overrides(ExperimentConfig.from_yaml(p), args) for p in args.configs]
    domains = [c.extra.get("domain") for c in cfgs]
    # everything that can fail on the inputs fails here, before the model loads
    unknown = [d for d in domains if d not in REASONING_FRAMES]
    if unknown:
        raise SystemExit(f"the reasoning arm runs on {sorted(REASONING_FRAMES)}; the configs name {unknown}")
    if len(set(domains)) != len(domains) or len(domains) < 2:
        raise SystemExit(f"--configs needs at least two configs of distinct domains, got {domains}")
    if len({(c.model_path, c.model_revision) for c in cfgs}) != 1:
        raise SystemExit("the configs must name one model and revision (directions live in one model's states): "
                         f"{sorted({(c.model_path, str(c.model_revision)) for c in cfgs})}; use --model/--revision")
    differing = [k for k in SHARED if len({str(getattr(c, k)) for c in cfgs}) != 1] + \
        (["extra.n_boot"] if len({int(c.extra.get("n_boot", DEFAULT_N_BOOT)) for c in cfgs}) != 1 else [])
    if differing:
        raise SystemExit(f"the configs differ in {differing}, which this runner takes from the first config only")
    if args.probe_items < 2 or args.eval_items < 2:
        raise SystemExit("--probe-items and --eval-items must be at least 2")
    keys = ("probe_items", "eval_items", "seed")
    variant = variant_suffix(
        {**{k: ap.get_default(k) for k in keys}, "revision": configured[0].model_revision, "domains": DEFAULT_DOMAINS},
        {**{k: getattr(args, k) for k in keys}, "revision": cfgs[0].model_revision, "domains": domains})
    out = args.out or default_out(cfgs[0].model_path, variant)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or --out")

    wanted = args.probe_items + args.eval_items
    data: Dict[str, Any] = {}
    doms: Dict[str, Dict[str, Any]] = {}
    for d, cfg in zip(domains, cfgs):
        dom = get_domain(d)
        cfg.dataset_source = cfg.dataset_source or dom.default_pairs
        pool, pool_data, left_out = reasoning_records(dom, REASONING_FRAMES[d], cfg.dataset_source)
        data.update({f"{d}/{k}": v for k, v in pool_data.items()})
        records, selection = select_items(pool, dom.is_strong, set(), wanted, args.seed)
        selection.update(left_out)
        if len(records) < wanted:
            raise SystemExit(f"{d}: {selection['available']} eligible strong records, fewer than --probe-items + "
                             f"--eval-items = {wanted}")
        doms[d] = {"dom": dom, "cfg": cfg, "selection": selection, "premises": probe_premises(d),
                   "probe_recs": records[:args.probe_items], "eval_recs": records[args.probe_items:]}
    n_boot = int(cfgs[0].extra.get("n_boot", DEFAULT_N_BOOT))

    exp = DemographicBiasExperiment(cfgs[0])
    exp.load_model()
    for c in cfgs[1:]:
        c.model_revision = cfgs[0].model_revision            # the commit actually loaded, for every config's record
    for d, sp in doms.items():
        print(f"[reasoning-transfer] embedding {d} ...", flush=True)
        # premise → [(fit split, eval split) per fold]
        sp["splits"] = {p: fold_splits(exp, sp["cfg"], sp["dom"], p, sp["probe_recs"], sp["eval_recs"], args.seed)
                        for p in sp["premises"]}

    folds = range(len(PARAPHRASE_FOLDS))
    fit = lambda d, f, premises: [doms[d]["splits"][p][f][0] for p in premises]
    # per fold, each domain's own direction (its three premises, valence-balanced)
    directions = {d: [{c: domain_direction(fit(d, f, doms[d]["premises"]), c) for c in CONCEPTS} for f in folds]
                  for d in domains}
    sources = [*domains, *(d + NONDEMOGRAPHIC for d in domains)] + ([OTHERS] if len(domains) > 2 else [])
    # per (evaluated domain, premise): the baseline intervals once, and the own direction's rewards (transfer gaps)
    baselines: Dict[str, Dict[str, Any]] = {}
    own_rewards: Dict[tuple, tuple] = {}
    for e in domains:
        baselines[e] = {}
        for p in doms[e]["premises"]:
            evals = [doms[e]["splits"][p][f][1] for f in folds]
            base = [_by_cell(rewards_from_hidden(exp.model, ev.hidden, ev.dtype, None, gates=ev.gates)[0], ev.n)
                    for ev in evals]
            baselines[e][p] = reasoning_intervals(base, None, n_boot, args.seed,
                                                  evals[0].favourable_true)["baseline"]
            for c in CONCEPTS:
                own = [(evals[f], directions[e][f][c]) for f in folds]
                own_rewards[(e, p, c)] = (own, *fold_rewards(exp, own))

    rows: List[Dict[str, Any]] = []
    for source in sources:
        for e in domains:
            members = source_members(source, e, domains)
            dirs = [{c: source_direction([fit(d, f, ps) for d, ps in members], c) for c in CONCEPTS} for f in folds]
            # the lexical control per fold: the expected bag-of-words of the source's fitted wording
            lexical = []
            for f, (fit_entries, _) in enumerate(PARAPHRASE_FOLDS):
                lf, le = expected_lexical(members, fit_entries,
                                          [doms[e]["splits"][p][f][1].items for p in doms[e]["premises"]])
                lexical.append({"eval": dict(zip(doms[e]["premises"], le)),
                                "dirs": {c: source_direction(lf, c) for c in CONCEPTS}})
            for p in doms[e]["premises"]:
                for c in CONCEPTS:
                    reads = [(doms[e]["splits"][p][f][1], dirs[f][c]) for f in folds]
                    row = {"source": source, "fit_domains": [d for d, _ in members],
                           "fit_premises": {d: list(ps) for d, ps in members}, "evaluated": e, "premise": p,
                           "demographic": REASONING_FRAMES[e].premises[p].demographic,
                           "favourable_true": REASONING_FRAMES[e].premises[p].favourable_true, "concept": c,
                           "paired_acc": paired_accuracy(reads, c, n_boot, args.seed),
                           "lexical_paired_acc": paired_accuracy(
                               [(lexical[f]["eval"][p], lexical[f]["dirs"][c]) for f in folds], c, n_boot, args.seed),
                           **nulled_effects(exp, reads, n_boot, args.seed, with_baseline=False)}
                    row["transfer_gap"] = None if source == e else \
                        transfer_gap(exp, *own_rewards[(e, p, c)], reads, n_boot, args.seed)
                    rows.append(row)
    cosines = {c: {a: {b: [cosine(directions[a][f][c], directions[b][f][c]) for f in folds] for b in domains}
                   for a in domains} for c in CONCEPTS}

    print("\n" + "=" * 120)
    print(f"REASONING TRANSFER across domains — {cfgs[0].model_path}  (fit {args.probe_items} / eval "
          f"{args.eval_items} records per domain; {len(PARAPHRASE_FOLDS)} held-out wordings pooled)")
    print("=" * 120)
    print(f"{'source':26} {'evaluated':10} {'premise':16} {'paired acc (model)':>28} {'paired acc (words)':>28}")
    for r in rows:
        if r["concept"] == "correctness":
            print(f"{r['source']:26} {r['evaluated']:10} {r['premise']:16} {_ci(r['paired_acc']):>28} "
                  f"{_ci(r['lexical_paired_acc']):>28}")
    print("-" * 120)
    print("A transfer means more than the wording where the model's paired accuracy exceeds the words' (at least, "
          "ties stated); the source direction is valence-balanced, so a valence code reads ½ on every premise and a "
          "difference between the unfavourable- and favourable-truth premises' rows is residual valence.")

    settings = {"probe_items": args.probe_items, "eval_items": args.eval_items, "seed": args.seed, "n_boot": n_boot,
                "domains": domains, "sources": sources, "fit": "valence-balanced (unfavourable- and favourable-truth "
                "premises weighted equally); others: domains weighted equally",
                "paraphrase_folds": [list(map(list, f)) for f in PARAPHRASE_FOLDS], "connective": False,
                "configs": {d: c.to_dict() for d, c in zip(domains, cfgs)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": run_metadata(cfgs[0], data, settings), "model": cfgs[0].model_path, "domains": domains,
         "seed": args.seed, "selection": {d: sp["selection"] for d, sp in doms.items()},
         "records": {d: {"probe": [str(r.source_record_id) for r in sp["probe_recs"]],
                         "eval": [str(r.source_record_id) for r in sp["eval_recs"]]} for d, sp in doms.items()},
         "baselines": baselines, "rows": rows, "cosines": cosines}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
