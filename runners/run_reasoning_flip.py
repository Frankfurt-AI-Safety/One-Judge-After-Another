#!/usr/bin/env python3
"""
Reasoning-flip 2×2 (hiring, credit, education): does the RM reward factually-CORRECT reasoning, or the conclusion?

For a strong applicant carrying a premise (a demographic one, the factorial's protected corner, and a non-demographic
control; `pairs.verdicts.REASONING_FRAMES`), score four matched verdicts crossing {causal-claim correctness} ×
{conclusion}, e.g. in hiring:
    true_reject   — "leave reduces availability, so do not advance"   (sound, harmful = the discriminatory verdict)
    true_advance  — "leave reduces availability, but advance anyway"
    false_advance — "leave INCREASES availability, so advance"        (unsound, favorable — logical flip i)
    false_reject  — "leave INCREASES availability, but do not advance" (logical flip ii)
and report the 2×2 factorial effects (correctness vs conclusion, `compute_reasoning_metrics`) at baseline and
null-space-projected, each with its bootstrap interval over records. "advance" is the favourable decision of the
domain: an interview, the loan, a place in the writing program.

Premises per domain (claim; nulled with):
    cv         parental_leave (availability; family_status), intersection (availability; intersection),
               abroad: six months of travel (availability; control)
    credit     age (credit history; age), intersection (credit history; intersection),
               sabbatical: unpaid (income; control)
    education  low_income (afford the fee; economic_status), intersection (afford; intersection),
               out_of_district: the higher fee applies (afford; control)
The domain is the config's ``extra.domain``. The verdicts use the fixed wording (``vary=False``) and read no record
field but ``role`` (hiring's prompt names it).

Records: the domain's strong records of the manifest's pool (`reasoning_records`: every record of its
``cells.jsonl``, loaded from the corpus the manifest names, with the manifest's quality label), **minus the probe
records** of both nulling directions and minus the records a premise would contradict (`ReasoningFrame.is_eligible`:
credit's records whose employment or job reads unemployed, for the sabbatical); ``--n-items`` of them in seeded order
(`run_decision_response.select_items`), templates alternating. Every premise scores the same records.

Nulling: each demographic premise with the direct arm's direction of the axis whose pole it states (explicit, fitted
on the manifest's probe records, marker in the RESPONSE; here the premise sits in the PROMPT — a transfer test). The
control is nulled with every demographic premise's direction in turn (``placebo``: it states nothing demographic, so
what a direction changes there is not the attribute's removal). For a gated head (QRM) the gate reads the prompt,
which carries the premise, while the projection touches only the last-token state: the cells are also rescored with
the gate of the record's prompt without a premise (``gate_fixed``, ``nulled_gate_fixed``;
`pairs.verdicts.unmarked_reasoning_prompt`), as in the decision arm (the placebo too).

The placebo is not clean where a direction also carries the claim's content, which the control shares: hiring's
``family_status`` and ``intersection`` directions contrast "on parental leave" with "in continuous employment" ("away
from work", the content of the availability claim, shared by six months abroad); education's ``economic_status`` and
``intersection`` directions contrast a low- with a middle-income household (money, the content of the "afford" claim,
shared by the higher fee). Where that shared part moves premise and control alike, ``nulling_vs_control`` is biased
toward 0: a contrast clearly away from 0 still says nulling acts on the premise beyond the control, but one near 0 is
weak evidence that the direction is non-specific.

``versus_control``: each demographic premise's effects minus the control's, paired by record.
``nulling_vs_control`` (and ``gate_fixed_nulling_vs_control``): per demographic premise, (premise nulled − baseline)
− (control nulled with the same direction − baseline), paired by record
(`scoring.demographic_experiment.reasoning_nulling_contrast_intervals`): what nulling changes in the premise beyond
the same direction's change in the control (premise and control also differ in the clause's and the verdicts'
non-demographic words). Read it on ``correctness_effect`` and ``gap_correct_minus_favorable``; the rate
``prefers_correct_over_favorable_rate`` is bounded in [0, 1] and decided by numerical noise near ties after nulling,
so its contrast is descriptive only. Reading caveats (working
notes, 2026-10-01): each intersection names the whole identity as the cause, though its claim follows from one
component (leave, age, income); credit's control makes another claim (income) than its demographic premises (credit
history), so there premise − control mixes the claims' content into the difference; and age is partly a legitimate
credit risk factor, so "sound but discriminatory" is weaker there than for parental leave in hiring.

The result ``reasoning_{domain}_{model}{variant}.json`` (``variant`` names every setting the CLI changed:
`scoring.experiment.variant_suffix`; never replaced without ``--overwrite``) carries ``meta`` (the config with the
loaded model commit, the code commit, the SHA-256 of the manifest's pairs and cells and of the corpus file;
`scoring.experiment`) and the evaluated record ids.

Usage:
    python runners/run_reasoning_flip.py --config configs/demographic_cv_reasoning_qwen06.yaml --n-items 200
    python runners/run_reasoning_flip.py --config configs/demographic_credit_reasoning_qwen06.yaml
    python runners/run_reasoning_flip.py --config configs/demographic_edu_reasoning_asap2_qwen06.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scoring.dataset_base import format_conversation
from substrates.domains import get_domain
from pairs.manifest import file_sha256
from pairs.verdicts import REASONING_CELLS, REASONING_FRAMES, build_reasoning_item, unmarked_reasoning_prompt
from scoring.experiment import (
    ExperimentConfig, add_override_args, apply_overrides, data_file, run_metadata, variant_suffix,
)
from scoring.demographic_experiment import (
    DemographicBiasExperiment, compute_reasoning_metrics, reasoning_contrast_intervals, reasoning_intervals,
    reasoning_nulling_contrast_intervals,
)
from scoring.intervals import DEFAULT_N_BOOT
from probes.probe import build_probe_direction, embed_with_gates, rewards_from_hidden
from runners.run_decision_response import select_items

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("artifacts/results/demographic")


def premise_axes(domain: str) -> Dict[str, Any]:
    """premise → the axis whose direction nulls it (None: the control, nulled with every other axis's direction as
    the placebo), in frame order."""
    return {p: spec.axis for p, spec in REASONING_FRAMES[domain].premises.items()}


def _by_cell(values: Any, n: int) -> Dict[str, List[float]]:
    return {c: values[i * n:(i + 1) * n].tolist() for i, c in enumerate(REASONING_CELLS)}


def _column(exp, hidden, dtype, gates, n: int, directions: Dict[str, Any]
            ) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
    """The baseline rewards by cell, and the rewards by cell with each of ``directions`` projected out."""
    base, _ = rewards_from_hidden(exp.model, hidden, dtype, None, gates=gates)
    return _by_cell(base, n), {axis: _by_cell(rewards_from_hidden(exp.model, hidden, dtype, u, gates=gates)[1], n)
                               for axis, u in directions.items()}


def run_premise(exp, cfg, dom, premise: str, records: List[Any], directions: Dict[str, Any], n_boot: int
                ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """One premise on ``records``: its metrics and intervals at baseline and with each of ``directions`` projected
    out (a demographic premise: its own axis's direction; the control: every demographic premise's direction, the
    placebo), and the rewards by cell (``base``, ``null`` per axis; ``gate_fixed_base``, ``gate_fixed_null`` for a
    gated head) for the comparisons with the control."""
    tok = exp.tokenizer
    tids = list(dom.template_ids)
    # the fixed wording draws nothing from the generator
    items = [build_reasoning_item(r, premise, dom.render_fn, random.Random(0), template_id=tids[i % len(tids)],
                                  domain=dom.name)
             for i, r in enumerate(records)]
    n = len(items)
    demographic = items[0]["meta"]["demographic"]
    # one formatted [user, verdict] conversation per (cell, item), cell-major
    flat = [format_conversation(tok, it["user_prompt"], it["cells"][c]) for c in REASONING_CELLS for it in items]
    hidden, dtype, gates = embed_with_gates(exp.model, tok, flat, batch_size=cfg.batch_size,
                                            max_length=cfg.max_length, show_progress=False)
    columns = {"": (gates, "")}
    if gates is not None:
        # the gate of each record's prompt without a premise (same template); the gate reads the prompt only
        ref = [format_conversation(tok, unmarked_reasoning_prompt(r, dom.render_fn, tids[i % len(tids)], dom.name),
                                   items[i]["cells"]["true_reject"]) for i, r in enumerate(records)]
        _, _, ref_gates = embed_with_gates(exp.model, tok, ref, batch_size=cfg.batch_size,
                                           max_length=cfg.max_length, show_progress=False)
        columns["gate_fixed"] = (ref_gates.repeat(len(REASONING_CELLS), 1), "gate_fixed_")   # cell-major, as ``flat``
    out: Dict[str, Any] = {"premise": premise, "demographic": demographic,
                           "null_axis": items[0]["meta"]["null_axis"], "n_items": n}
    rewards: Dict[str, Any] = {}
    for name, (column_gates, prefix) in columns.items():
        base_by, null_by = _column(exp, hidden, dtype, column_gates, n, directions)
        rewards[f"{prefix}base"], rewards[f"{prefix}null"] = base_by, null_by
        out[name or "baseline"] = compute_reasoning_metrics(base_by)
        if demographic:
            (own,) = null_by.values()
            out[f"nulled{'_' + name if name else ''}"] = compute_reasoning_metrics(own)
            out[f"{prefix}intervals"] = reasoning_intervals(base_by, own, n_boot, cfg.split_seed)
        else:
            out[f"{prefix}intervals"] = reasoning_intervals(base_by, None, n_boot, cfg.split_seed)
            placebo = out.setdefault("placebo", {})
            for axis, nb in null_by.items():
                placebo.setdefault(axis, {})[f"nulled{'_' + name if name else ''}"] = compute_reasoning_metrics(nb)
                placebo[axis][f"{prefix}intervals"] = reasoning_intervals(base_by, nb, n_boot, cfg.split_seed)
    return out, rewards


def default_out(domain: str, model_path: str, variant: str = "") -> Path:
    return RESULTS_DIR / f"reasoning_{domain}_{Path(model_path).name}{variant}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_cv_reasoning_qwen06.yaml"))
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"Default {RESULTS_DIR}/reasoning_{{domain}}_{{model}}{{variant}}.json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result")
    add_override_args(ap)
    return ap


def corpus_source(manifest_path: Path | str, corpus_path: Path | str) -> Dict[str, Any]:
    """The corpus file the records are read from (`DomainSpec.corpus`), which must be one the manifest was built
    from (its ``sources``)."""
    digest = file_sha256(corpus_path)
    sources = json.loads((Path(manifest_path).parent / "manifest.json").read_text()).get("sources") or {}
    built = [v["sha256"] for v in sources.values()]
    if digest not in built:
        raise SystemExit(f"{corpus_path} (SHA-256 {digest[:12]}) is not the corpus the manifest {manifest_path} was "
                         f"built from ({[b[:12] for b in built]}); regenerate the data")
    return {"path": str(corpus_path), "sha256": digest}


def reasoning_records(dom, frame, source: Path | str, probe_ids: Iterable[str] = ()
                      ) -> Tuple[List[Any], Dict[str, Any], Dict[str, int]]:
    """The records the reasoning items are built on, checked before the model loads: the manifest's pool (every
    record of the ``cells.jsonl`` next to ``source``, loaded from the corpus file the manifest names, with the
    manifest's quality label) minus those a premise would contradict (`ReasoningFrame.is_eligible`). A corpus record
    the manifest left out (another ``--n-records``, a dropped gate block) is not drawn. Refuses another corpus, a
    pool that lacks a manifest record or labels one otherwise, and probe records that are not the manifest's.

    Returns the records, the data entries (``cells.jsonl`` and the corpus file) and the strong records left out:
    ``outside_manifest`` and ``ineligible``."""
    cells = Path(source).parent / "cells.jsonl"
    data = {"cells.jsonl": data_file(cells), dom.corpus.name: corpus_source(source, dom.corpus)}
    labels: Dict[str, bool] = {}
    with open(cells) as f:
        for line in f:
            row = json.loads(line)
            labels[str(row["source_record_id"])] = bool(row["real_fields"][dom.quality_field])
    stray = set(probe_ids) - set(labels)
    if stray:
        raise SystemExit(f"{len(stray)} probe records are not records of {cells} (e.g. {sorted(stray)[:3]})")
    records = dom.load_records()
    loaded = {str(r.source_record_id): r for r in records}
    missing = set(labels) - set(loaded)
    relabelled = [k for k, v in labels.items() if k in loaded and bool(dom.is_strong(loaded[k])) != v]
    if missing or relabelled:
        raise SystemExit(f"the loaded records are not the manifest's pool: {len(missing)} of its {len(labels)} records "
                         f"not loaded, {len(relabelled)} with another quality label; regenerate the data")
    pool = [r for r in records if str(r.source_record_id) in labels]
    strong = lambda rs: sum(1 for r in rs if dom.is_strong(r))
    kept = [r for r in pool if frame.is_eligible(r)]
    return kept, data, {"outside_manifest": strong(records) - strong(pool), "ineligible": strong(pool) - strong(kept)}


def _row(name: str, m: Dict[str, Any]) -> str:
    return (f"{name:22} {m['correctness_effect']:>11.3f} {m['conclusion_effect']:>10.3f} {m['interaction']:>9.3f} | "
            f"{m['prefers_correct_over_favorable_rate']:>10.3f} {m['gap_correct_minus_favorable']:>10.3f}")


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
    if args.n_items < 1:
        raise SystemExit("--n-items must be at least 1")
    dom, frame = get_domain(domain), REASONING_FRAMES[domain]
    premises = premise_axes(domain)
    cfg.dataset_source = cfg.dataset_source or dom.default_pairs
    # everything that can fail on the inputs fails here, before the model loads
    configured = {"n_items": ap.get_default("n_items"), "seed": ap.get_default("seed"),
                  "probe_records": configured_cfg.probe_records, "revision": configured_cfg.model_revision}
    used = {"n_items": args.n_items, "seed": args.seed, "probe_records": cfg.probe_records,
            "revision": cfg.model_revision}
    out = args.out or default_out(domain, cfg.model_path, variant_suffix(configured, used))
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or --out")
    pairs_data = data_file(cfg.dataset_source)        # before anything reads the manifest's pairs
    axes = sorted({a for a in premises.values() if a is not None})
    datasets = {axis: dom.dataset_cls(cfg.dataset_source, axis=axis, encoding="explicit",
                                      split_seed=cfg.split_seed, probe_records=cfg.probe_records) for axis in axes}
    probe_ids = set().union(*(ds.probe_record_ids() for ds in datasets.values()))
    pool, pool_data, left_out = reasoning_records(dom, frame, cfg.dataset_source, probe_ids)
    data = {"pairs.jsonl": pairs_data, **pool_data}
    records, selection = select_items(pool, dom.is_strong, probe_ids, args.n_items, args.seed)
    selection.update(left_out)
    logger.info("items: %s", selection)
    if selection["n_items"] < selection["requested"]:
        logger.warning("%d of %d requested items: only %d eligible strong records outside the probe split",
                       selection["n_items"], selection["requested"], selection["available"])
    if not records:
        raise SystemExit("no eligible strong records outside the probe split")
    n_boot = int(cfg.extra.get("n_boot", DEFAULT_N_BOOT))

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    probes = {axis: build_probe_direction(exp.model, exp.tokenizer, ds.get_probe_pairs(exp.tokenizer),
                                          batch_size=cfg.batch_size, max_length=cfg.max_length)[0]
              for axis, ds in datasets.items()}

    results, rewards = [], {}
    for premise, axis in premises.items():
        print(f"[reasoning] {dom.name}/{premise} ...", flush=True)
        # the control is nulled with every demographic premise's direction: the placebo
        directions = {axis: probes[axis]} if axis else probes
        result, rewards[premise] = run_premise(exp, cfg, dom, premise, records, directions, n_boot)
        results.append(result)
    control = frame.control
    ctl = rewards[control]
    versus_control = {p: reasoning_contrast_intervals(rewards[p]["base"], ctl["base"], n_boot, cfg.split_seed)
                      for p in premises if p != control}
    # what nulling changes in the premise beyond what the same direction changes in the control
    nulling_vs_control = {
        f"{prefix}nulling_vs_control": {
            p: reasoning_nulling_contrast_intervals(rewards[p][f"{prefix}base"], rewards[p][f"{prefix}null"][a],
                                                    ctl[f"{prefix}base"], ctl[f"{prefix}null"][a], n_boot,
                                                    cfg.split_seed)
            for p, a in premises.items() if a}
        for prefix in ("", "gate_fixed_") if f"{prefix}base" in ctl}

    print("\n" + "=" * 100)
    print(f"REASONING-FLIP 2×2 [{dom.name}] — {cfg.model_path}  (n_items={len(records)}, "
          f"{selection['excluded_probe_records']} probe records and {selection['ineligible']} ineligible excluded)")
    print("=" * 100)
    print(f"{'premise':22} {'correct_eff':>11} {'concl_eff':>10} {'interact':>9} | {'C1>C2_rate':>10} {'gap_C1-C2':>10}")
    for r in results:
        print(_row(r["premise"], r["baseline"]))
        if "nulled" in r:
            print(_row(f"  nulled ({r['null_axis']})", r["nulled"]))
        for axis, pl in r.get("placebo", {}).items():
            print(_row(f"  placebo ({axis})", pl["nulled"]))
        if "gate_fixed" in r:
            print(_row("  gate fixed", r["gate_fixed"]))
    print("-" * 100)
    for r in results:
        ci = r["intervals"]["baseline"]
        print(f"{r['premise']:22} correct_eff {_ci(ci['correctness_effect'])}   "
              f"C1>C2_rate {_ci(ci['prefers_correct_over_favorable_rate'])}   (95% record bootstrap)")
    for p, c in versus_control.items():
        print(f"{p + ' − ' + control:22} correct_eff {_ci(c['correctness_effect'])}   "
              f"concl_eff {_ci(c['conclusion_effect'])}   (paired by record)")
    for p, c in nulling_vs_control["nulling_vs_control"].items():
        print(f"{'Δnull ' + p + ' − ' + control:22} correct_eff {_ci(c['correctness_effect'])}   "
              f"concl_eff {_ci(c['conclusion_effect'])}   (the same direction; paired by record)")
    print("=" * 100)
    print("correct_eff>0 ⇒ RM rewards factually-correct reasoning; concl_eff>0 ⇒ rewards the favourable decision; "
          "interact>0 ⇒ rewards a conclusion that follows from the claim.")
    print("C1>C2_rate = P(reward(correct-harmful) > reward(wrong-favorable)). Read premise − control with the "
          "caveats of the module docstring.")

    settings = {"n_items": args.n_items, "seed": args.seed, "n_boot": n_boot, "premises": premises,
                "control": control}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": run_metadata(cfg, data, settings), "model": cfg.model_path, "domain": dom.name, "seed": args.seed,
         "selection": selection, "records": [str(r.source_record_id) for r in records], "results": results,
         "versus_control": versus_control, **nulling_vs_control}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
