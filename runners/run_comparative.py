#!/usr/bin/env python3
"""
Comparative (two-applicant) decision design, one RM — the third placement of the marker (design 2026-09-26,
decisions 2026-09-30; `pairs/comparative.py`). Exploratory until the headline family is fixed.

The USER turn shows two applications of the same domain (hiring: the last interview slot for one role; credit:
the one loan the budget allows; education: the one writing award, for two essays on the same assignment) and
the ASSISTANT turn chooses one, with a merit reason or a coded, stereotype-fitting doubt about the other.
Record pairs are strong–strong and weak–weak (does the marker break a tie of merit?) and strong–weak (a
merit-correct answer exists: can the marker overturn it?). Per pair and contrast axis both marker assignments ×
both orders are scored, so the records and the position cancel (`scoring/comparative_metrics.py`: marker effect,
preference rate, position effect; for strong–weak the quality margin, the overturn as an accuracy gap, and the
exchange rate; intervals over record pairs).

Reward columns:

- ``baseline``; ``gate_fixed`` for gated heads (QRM): every row rescored with the gate of its pair's unmarked
  prompt in the same order and template, so the marker acts through the last-token state alone;
- ``null_direct:{axis}`` / ``null_direct:joint``: the direct arm's directions (marker in the response), fitted on
  its probe records, which are never paired here — the direct → comparative transfer (RQ5);
- ``null_own:{axis}``: this design's "the chosen applicant is protected" direction
  (`probes/comparative_directions.py`), cross-fitted over ``n_folds`` folds of pairs stratified by pairing.

The full 3×3 transfer matrix (direct / cross-marker / comparative directions, each nulled in the others) and
LEACE with a non-linear probe on the comparative states are later steps; ``geometry`` gives the comparative ↔
direct cosines with their reliability ceilings.

Items come from the generator's ``cells.jsonl`` next to the config's ``dataset_source``; both files are checked
against their manifest hashes before the model loads (`scoring.experiment.data_file`). A pair is formed only if
every one of its texts fits ``max_length`` (two documents per prompt: the education config sets 4096) and the
two documents share no proxy first name; the pairing report counts what was skipped. Settings: ``extra.comparative``
in the config, overridden by the CLI.

Outputs (no texts): ``<out>.json`` (``meta``, pairing report, the pairs' record ids, metrics per encoding and
reward column, geometry, timing), ``<out>_rewards.jsonl`` (one row per scored text) and ``<out>_directions.pt``.
The default ``<out>`` is ``comparative_{domain}[_{manifest folder}]_{model}.json``; an existing result is never
replaced without ``--overwrite``.

Usage:
    python runners/run_comparative.py --config configs/demographic_credit_comparative_qwen06.yaml
    python runners/run_comparative.py --config ... --n-pairs 4 --device mps          # smoke
    python runners/run_comparative.py --config ... --n-pairs 100,50,100             # ss,sw,ww
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pairs.comparative import (
    COMPARATIVE_FRAMES, KINDS, MATCH_FIELDS, PAIRINGS, UNMARKED, RecordPair, build_pair_items, contrast_axes,
    draw_pairs,
)
from pairs.cross_marker import CellBlock, load_cell_blocks
from runners.run_cross_marker import (
    PhaseTimer, cells_path, check_requested, direct_directions, timing_report, token_counter,
)
from scoring.comparative_metrics import POOLED, comparative_metrics
from scoring.dataset_base import format_conversation
from scoring.experiment import add_override_args, apply_overrides, data_file, run_metadata
from scoring.intervals import DEFAULT_N_BOOT

logger = logging.getLogger(__name__)

DIRECTION_SOURCES = ("direct", "own")
DEFAULTS: Dict[str, Any] = {
    "n_pairs": {p: 150 for p in PAIRINGS},
    "encodings": ["explicit", "proxy"],
    "templates": None,          # None = every template in cells.jsonl
    "include_unmarked": True,   # the unmarked prompts: the gate for gate_fixed and the scale of the effects
    "paraphrases": 3,
    "seed": 42,
    "n_boot": DEFAULT_N_BOOT,
    "directions": list(DIRECTION_SOURCES),   # [] = baseline only
    "n_folds": 5,
}

BlockMap = Dict[str, Dict[Tuple[str, str], CellBlock]]     # record -> (encoding, template) -> block


# --------------------------------------------------------------------------- settings ----------------
def parse_n_pairs(value: Any) -> Dict[str, int]:
    """``150`` (every pairing), ``"100,50,100"`` (strong_strong, strong_weak, weak_weak) or a mapping."""
    if isinstance(value, Mapping):
        unknown = set(value) - set(PAIRINGS)
        if unknown:
            raise ValueError(f"unknown pairings {sorted(unknown)}; known: {PAIRINGS}")
        return {p: int(value.get(p, 0)) for p in PAIRINGS}
    parts = [int(v) for v in str(value).split(",")]
    if len(parts) == 1:
        parts *= len(PAIRINGS)
    if len(parts) != len(PAIRINGS):
        raise ValueError(f"n_pairs takes 1 or {len(PAIRINGS)} counts ({','.join(PAIRINGS)}), got {value!r}")
    return dict(zip(PAIRINGS, parts))


def resolve_settings(extra: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """DEFAULTS < config ``extra.comparative`` < CLI (``None`` in ``overrides`` = not given)."""
    settings = dict(DEFAULTS)
    configured = extra.get("comparative") or {}
    unknown = set(configured) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown comparative settings {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    settings.update(configured)
    settings.update({k: v for k, v in overrides.items() if v is not None})
    settings["n_pairs"] = parse_n_pairs(settings["n_pairs"])
    bad = set(settings["directions"]) - set(DIRECTION_SOURCES)
    if bad:
        raise ValueError(f"directions must be drawn from {DIRECTION_SOURCES}, got {sorted(bad)}")
    return settings


def default_out(domain: str, source: Path | str, model_path: str) -> Path:
    """``comparative_{domain}[_{manifest folder}]_{model}.json``, as the cross-marker runner names its results."""
    folder = Path(source).parent.name
    stem = domain if folder == domain else f"{domain}_{folder}"
    return Path("artifacts/results/demographic") / f"comparative_{stem}_{Path(model_path).name}.json"


# --------------------------------------------------------------------------- pairing -----------------
def candidates_from(blocks: Sequence[CellBlock], *, quality_field: str, match_field: Optional[str],
                    encodings: Sequence[str], templates: Sequence[str], exclude: Set[str]
                    ) -> Tuple[BlockMap, Dict[str, Tuple[bool, Any]], Dict[str, int]]:
    """The records that can be paired: every requested (encoding, template) block present and not a probe
    record. Returns their blocks, ``{record: (strong, match group)}`` and a count of what was removed."""
    wanted = {(e, t) for e in encodings for t in templates}
    by_record: BlockMap = defaultdict(dict)
    for b in blocks:
        if (b.encoding, b.template_id) in wanted:
            by_record[b.record_id][(b.encoding, b.template_id)] = b
    report = {"records_in_cells": len(by_record), "excluded_probe_records": 0, "incomplete_records": 0}
    candidates: Dict[str, Tuple[bool, Any]] = {}
    for rid, bs in by_record.items():
        if rid in exclude:
            report["excluded_probe_records"] += 1
        elif set(bs) != wanted:
            report["incomplete_records"] += 1
        else:
            first = next(iter(bs.values()))
            group = first.real_fields.get(match_field) if match_field else None
            if match_field and group is None:
                raise KeyError(f"{rid}: real_fields has no {match_field!r} — regenerate the manifest")
            candidates[rid] = (first.is_strong(quality_field), group)
    return dict(by_record), candidates, report


def pair_items(pair: RecordPair, by_record: BlockMap, domain: str, key: Tuple[str, str],
               settings: Mapping[str, Any]) -> List[Any]:
    return build_pair_items(pair, by_record[pair.x][key], by_record[pair.y][key], domain, seed=settings["seed"],
                            n_paraphrases=settings["paraphrases"], include_unmarked=settings["include_unmarked"])


def compatibility(by_record: BlockMap, domain: str, settings: Mapping[str, Any],
                  format_fn: Callable[[str, str], Any], count: Callable[[Any], int], max_length: int
                  ) -> Callable[[RecordPair], Optional[str]]:
    """The pairing's check (`pairs.comparative.draw_pairs`): ``"shared_name"`` when the two records' blocks
    share a proxy first name, ``"too_long"`` when any text of the pair exceeds ``max_length`` tokens (the forward
    pass would refuse it), else None."""
    def check(pair: RecordPair) -> Optional[str]:
        names_x = set().union(*(b.names for b in by_record[pair.x].values()))
        if names_x & set().union(*(b.names for b in by_record[pair.y].values())):
            return "shared_name"
        convs = [format_fn(i.prompt, i.text) for key in by_record[pair.x]
                 for i in pair_items(pair, by_record, domain, key, settings)]
        counts = count.batch(convs) if hasattr(count, "batch") else [count(c) for c in convs]
        return "too_long" if max(counts) > max_length else None
    return check


def build_rows(pairs: Sequence[RecordPair], by_record: BlockMap, domain: str, encoding: str,
               templates: Sequence[str], settings: Mapping[str, Any], format_fn: Callable[[str, str], Any]
               ) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """One row per (pair, template, axis, assignment, order, kind, choice) of one encoding and its formatted
    conversation (same order). Rows carry ids and labels only, never text."""
    rows: List[Dict[str, Any]] = []
    convs: List[Any] = []
    for pair in pairs:
        for template in templates:
            for item in pair_items(pair, by_record, domain, (encoding, template), settings):
                rows.append({"pair_id": pair.pair_id, "pairing": pair.pairing, "template_id": template,
                             "encoding": encoding, "axis": item.axis, "protected": item.protected,
                             "order": item.order, "kind": item.kind, "chosen": item.chosen,
                             "paraphrase": item.paraphrase})
                convs.append(format_fn(item.prompt, item.text))
    return rows, convs


# --------------------------------------------------------------------------- scoring -----------------
def score_encoding(model: Any, tokenizer: Any, rows: List[Dict[str, Any]], convs: List[Any],
                   direct_dirs: Mapping[str, Any], axes: Sequence[str], settings: Mapping[str, Any], *,
                   batch_size: int, max_length: int, direct_reliability: Optional[Mapping[str, float]] = None,
                   show_progress: bool = True) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    """Add every reward column to ``rows`` (one encoding) in place; return ``(columns, geometry, directions)``."""
    import torch

    from probes import cross_marker_directions as cmd
    from probes.comparative_directions import pair_contrasts
    from probes.heads import get_head
    from probes.probe import embed_with_gates, rewards_from_hidden

    hx, dtype, gx = embed_with_gates(model, tokenizer, convs, batch_size=batch_size, max_length=max_length,
                                     show_progress=show_progress)

    def apply(column: str, basis: Optional[torch.Tensor], sub: Optional[Sequence[int]] = None,
              gates: Optional[torch.Tensor] = None) -> None:
        idx = list(range(len(rows))) if sub is None else list(sub)
        if not idx:
            return
        g = gx if gates is None else gates
        _, r = rewards_from_hidden(model, hx[idx], dtype, basis, gates=None if g is None else g[idx])
        for k, i in enumerate(idx):
            rows[i][column] = float(r[k])

    apply("baseline", None)
    columns = ["baseline"]
    if gx is not None:
        ref: Dict[Tuple[str, str, str], int] = {}
        for i, r in enumerate(rows):
            if r["axis"] == UNMARKED:
                ref.setdefault((r["pair_id"], r["template_id"], r["order"]), i)
        keys = [(r["pair_id"], r["template_id"], r["order"]) for r in rows]
        if all(k in ref for k in keys):
            apply("gate_fixed", None, gates=gx[[ref[k] for k in keys]])
            columns.append("gate_fixed")
        else:
            logger.info("gate_fixed skipped: no unmarked prompt to take the gate from")

    directions: Dict[str, torch.Tensor] = {}
    for axis, u in direct_dirs.items():
        apply(f"null_direct:{axis}", u)
        columns.append(f"null_direct:{axis}")
        directions[f"direct:{axis}"] = u
    if len(direct_dirs) > 1:
        apply("null_direct:joint", torch.stack(list(direct_dirs.values())))
        columns.append("null_direct:joint")

    fitted: Dict[str, Dict[int, torch.Tensor]] = {}
    split_half: Dict[str, float] = {}
    pairing_of = {r["pair_id"]: r["pairing"] for r in rows}
    folds: Dict[str, int] = {}
    if "own" in settings["directions"]:
        folds = cmd.fold_assignment(sorted(pairing_of), pairing_of, settings["n_folds"], settings["seed"])
        if len(set(folds.values())) < 2:
            logger.warning("own directions skipped: the %d pairs fill one fold; cross-fitting needs two",
                           len(pairing_of))
            folds = {}
        else:
            by_fold: Dict[int, List[int]] = defaultdict(list)
            for i, r in enumerate(rows):
                by_fold[folds[r["pair_id"]]].append(i)
            for axis in axes:
                ids, contrasts = pair_contrasts(hx, rows, axis)
                name = f"own:{axis}"
                fitted[name] = cmd.cross_fitted(contrasts, ids, folds)
                for f, u in fitted[name].items():
                    apply(f"null_{name}", u, sub=by_fold[f])
                columns.append(f"null_{name}")
                directions[name] = cmd.unit(contrasts.mean(0))
                split_half[name] = cmd.split_half_cosine(contrasts, settings["seed"])

    head = get_head(model)
    w = head.effective_weights(gx).mean(0)
    reliability = {n: cmd.full_sample_reliability(h) for n, h in split_half.items()}
    for axis in direct_dirs:
        reliability[f"direct:{axis}"] = float((direct_reliability or {}).get(axis, float("nan")))
    geometry: Dict[str, Any] = {"head_alignment": {n: cmd.head_alignment(w, u) for n, u in directions.items()},
                                "split_half_cosine": split_half, "reliability": reliability,
                                "own_vs_direct": {}, "n_folds": settings["n_folds"],
                                "head": head.kind if not head.gated else f"{head.kind} (mean effective head)"}
    for axis in axes:
        own, direct = f"own:{axis}", f"direct:{axis}"
        if own in directions and direct in directions:
            geometry["own_vs_direct"][axis] = {
                "cosine": cmd.cosine(directions[own], directions[direct]),
                "ceiling": cmd.cosine_ceiling(reliability[own], reliability[direct])}
    saved = {"full_data": directions, "cross_fitted": fitted, "folds": folds}
    return columns, geometry, saved


# --------------------------------------------------------------------------- report ------------------
def _fmt(s: Optional[Mapping[str, Any]]) -> str:
    if not s or s.get("mean") is None or s["mean"] != s["mean"]:
        return "—"
    lo, hi = s.get("ci_low"), s.get("ci_high")
    return f"{s['mean']:+.3f} [{lo:+.2f},{hi:+.2f}]" if lo == lo and hi == hi else f"{s['mean']:+.3f}"


def print_report(summary: Mapping[str, Any]) -> None:
    print("\n" + "=" * 110)
    counts = ", ".join(f"{p} {summary['pairing'][p]['n']}/{summary['pairing'][p]['requested']}" for p in PAIRINGS)
    print(f"COMPARATIVE [{summary['domain']}] — {summary['model']}  (pairs: {counts})")
    print("marker effect = r(choose protected) − r(choose reference), merit response; negative ⇒ the protected "
          "applicant is disfavoured")
    for encoding, by_col in summary["metrics"].items():
        base = by_col["baseline"]
        print("-" * 110)
        print(f"[{encoding}] baseline marker effect (merit | coded) and with each direction projected out (merit)")
        for pairing, block in base.items():
            for axis, entry in block.items():
                if axis in ("n_pairs", UNMARKED) or "merit" not in entry:
                    continue
                nulls = [c for c in by_col if c.startswith("null_") and c.endswith(f":{axis}")]
                nulled = "  ".join(f"{c[5:]} {by_col[c][pairing][axis]['merit']['marker_effect']['mean']:+.3f}"
                                   for c in nulls)
                print(f"  {pairing:13} {axis:16} {_fmt(entry['merit']['marker_effect']):>24} | "
                      f"{_fmt(entry.get('coded', {}).get('marker_effect')):>24}   {nulled}")
                if pairing == "strong_weak":
                    m = entry["merit"]
                    print(f"  {'':13} {'':16} accuracy gap {_fmt(m['accuracy_gap'])}  exchange rate "
                          f"{_fmt(m['exchange_rate'])}  quality margin {_fmt(m['quality_margin'])}")
        geo = summary["geometry"].get(encoding, {}).get("own_vs_direct", {})
        if geo:
            print("  cos(own, direct) / ceiling: " + "  ".join(f"{a} {v['cosine']:+.2f}/{v['ceiling']:.2f}"
                                                          for a, v in geo.items()))
    timing = summary.get("timing")
    if timing:
        print(f"timing (batch {timing['batch_size']}): " +
              " | ".join(f"{k} {v:.0f}s" for k, v in timing["seconds"].items()) + f" | total {timing['total_s']:.0f}s")
    print("=" * 110)


# --------------------------------------------------------------------------- main --------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_credit_comparative_qwen06.yaml"))
    ap.add_argument("--n-pairs", default=None, help="Pairs per pairing: N, or ss,sw,ww counts")
    ap.add_argument("--encodings", default=None, help="Comma-separated, e.g. explicit,proxy")
    ap.add_argument("--templates", default=None, help="Comma-separated template ids (default: all)")
    ap.add_argument("--paraphrases", type=int, default=None)
    ap.add_argument("--directions", default=None,
                    help=f"Comma-separated subset of {','.join(DIRECTION_SOURCES)}, or 'none'")
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default artifacts/results/demographic/comparative_{domain}[_{folder}]_{model}.json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result (and its side files)")
    add_override_args(ap)   # --model, --revision, --batch-size, --device, --probe-records
    return ap


def main() -> None:
    import torch

    from scoring.backend import model_revision
    from scoring.demographic_experiment import DemographicBiasExperiment
    from scoring.experiment import ExperimentConfig
    from substrates.domains import get_domain

    timer = PhaseTimer()
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    dom = get_domain(cfg.extra.get("domain", "credit"))
    if dom.name not in COMPARATIVE_FRAMES:
        raise SystemExit(f"no comparative frame for domain {dom.name!r}")
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None
    settings = resolve_settings(cfg.extra, {
        "n_pairs": args.n_pairs, "encodings": split(args.encodings), "templates": split(args.templates),
        "paraphrases": args.paraphrases, "directions": ([] if args.directions == "none" else split(args.directions)),
        "n_folds": args.n_folds, "n_boot": args.n_boot, "seed": args.seed})
    source = cfg.dataset_source = cfg.dataset_source or dom.default_pairs
    path = cells_path(source, dom.default_pairs)
    out = args.out or default_out(dom.name, source, cfg.model_path)
    # everything that can fail on the inputs fails here, before the model loads
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it (and its side files), or --out")
    data = {"pairs.jsonl": data_file(source), "cells.jsonl": data_file(path)}
    cells_report: Dict[str, Any] = {}
    blocks = load_cell_blocks(path, dom.factorial, cells_report)
    templates = settings["templates"] or sorted({b.template_id for b in blocks})
    check_requested(blocks, settings["encodings"], templates)
    if cells_report.get("n_dropped_mismatch"):
        logger.warning("%d cells.jsonl blocks dropped as not matching the design", cells_report["n_dropped_mismatch"])

    timer.lap("setup")
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    model, tok = exp.model, exp.tokenizer
    cache = getattr(model, "_onejudge_embedding_cache", None)
    timer.lap("load_model")

    direct_dirs: Dict[str, Dict[str, Any]] = {}
    probe_ids: Set[str] = set()
    probe_meta: Dict[str, Any] = {}
    if "direct" in settings["directions"]:
        direct_dirs, probe_ids, probe_meta = direct_directions(
            model, tok, dom, source, settings["encodings"], probe_records=cfg.probe_records,
            split_seed=cfg.split_seed, batch_size=cfg.batch_size, max_length=cfg.max_length,
            reliability_seed=settings["seed"])
    direct_misses = None if cache is None else cache.misses
    timer.lap("direct_directions")

    format_fn = lambda prompt, response: format_conversation(tok, prompt, response)
    by_record, candidates, selection = candidates_from(
        blocks, quality_field=dom.quality_field, match_field=MATCH_FIELDS[dom.name],
        encodings=settings["encodings"], templates=templates, exclude=probe_ids)
    check = compatibility(by_record, dom.name, settings, format_fn, token_counter(tok), cfg.max_length)
    pairs, pairing = draw_pairs(candidates, settings["n_pairs"], settings["seed"], check)
    logger.info("records: %d in cells.jsonl, %d probe records excluded, %d incomplete; pairs %s",
                selection["records_in_cells"], selection["excluded_probe_records"],
                selection["incomplete_records"], {p: pairing[p]["n"] for p in PAIRINGS})
    for p in PAIRINGS:
        if pairing[p]["n"] < pairing[p]["requested"]:
            logger.warning("%s: %d of %d requested pairs (skipped: %s)", p, pairing[p]["n"],
                           pairing[p]["requested"], pairing[p]["skipped"])
    if not pairs:
        raise SystemExit("no pairs could be formed")
    timer.lap("prepare")

    metrics: Dict[str, Dict[str, Any]] = {}
    geometry: Dict[str, Any] = {}
    saved: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []
    all_columns: List[str] = []
    for encoding in settings["encodings"]:
        rows, convs = build_rows(pairs, by_record, dom.name, encoding, templates, settings, format_fn)
        axes = contrast_axes(dom.factorial, encoding)
        timer.lap("prepare")
        columns, geometry[encoding], saved[encoding] = score_encoding(
            model, tok, rows, convs, direct_dirs.get(encoding, {}), axes, settings,
            batch_size=cfg.batch_size, max_length=cfg.max_length,
            direct_reliability={a: probe_meta[f"{encoding}/{a}"]["reliability"] for a in direct_dirs.get(encoding, {})})
        del convs
        timer.lap("embed_and_null")
        metrics[encoding] = {c: comparative_metrics(rows, c, baseline_key=None if c == "baseline" else "baseline",
                                                    n_boot=settings["n_boot"], seed=settings["seed"])
                             for c in columns}
        all_columns += [c for c in columns if c not in all_columns]
        all_rows += rows
        timer.lap("metrics")

    summary = {
        "meta": run_metadata(cfg, data, {**settings, "templates": templates}),
        "model": cfg.model_path, "model_revision": model_revision(cfg), "domain": dom.name,
        "settings": {**settings, "templates": templates}, "max_length": cfg.max_length,
        "probe_records": cfg.probe_records, "cells_path": str(path), "cells_report": cells_report,
        "direct_manifest": source, "probe_directions": probe_meta, "probe_record_ids": sorted(probe_ids),
        "selection": selection, "pairing": pairing,
        "pairs": [{"pair_id": p.pair_id, "pairing": p.pairing, "x": p.x, "y": p.y, "index": p.index} for p in pairs],
        "n_texts": len(all_rows), "reward_columns": all_columns, "kinds": list(KINDS), "pooled": POOLED,
        "embedding_cache": None if cache is None else {"hits": cache.hits, "misses": cache.misses},
        "metrics": metrics, "geometry": geometry,
        "timing": timing_report(timer, batch_size=cfg.batch_size, texts=None if cache is None else {
            "direct_directions": direct_misses, "scoring": cache.misses - direct_misses}),
        "caveats": [
            "Exploratory until the headline family is fixed; intervals are uncorrected.",
            "Direct directions are fitted on the direct arm's probe records, none of which is paired here.",
            "Own directions are cross-fitted over folds of pairs; geometry uses full-data fits (descriptive).",
            "Binding: the RM must bind the marker to the right applicant; a single linear direction may not capture "
            "that, so a failed projection can be structural (RQ3).",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    stem = out.with_suffix("")
    with open(f"{stem}_rewards.jsonl", "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    torch.save(saved, f"{stem}_directions.pt")
    print_report(summary)
    print(f"saved → {out} (+ _rewards.jsonl, _directions.pt)")


if __name__ == "__main__":
    main()
