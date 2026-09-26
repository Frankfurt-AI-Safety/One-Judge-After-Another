#!/usr/bin/env python3
"""
Cross-marker decision design, one RM — the primary harm evidence (methodology decision 2026-09-24) and
its mechanism layer.

For each evaluated record the USER turn presents the document in all 8 factorial cells plus an unmarked
control and asks for the domain's decision; the ASSISTANT turn is one of five responses that state no
attribute value (approve / neutral decline / coded decline / overt decline / evasive;
`pairs/cross_marker.py`). `scoring/cross_marker_metrics.py` turns the rewards into decision-margin
disparities (D = r(approve) − r(decline); a negative D-disparity = the RM favours declining the protected
pole more), the 2x2x2 decomposition, decision accuracy and cross-influence (accuracy-based and
threshold-free) on strong and weak records.

Mechanism layer (`probes/cross_marker_directions.py`), all from one embedding pass per encoding:

- **directions**: the direct arm's per axis (marker in the response; RQ5 transfer) and their joint span,
  plus three fitted on this design's own states — ``prompt`` and ``interaction`` per axis, ``unfair``
  (overt vs neutral decline). The latter are **cross-fitted** (records in ``n_folds`` quality-stratified
  folds; each fold nulled with the direction fitted on the others);
- **cross-nulling**: every direction projected out, every metric recomputed (one reward column each);
- **geometry**: cosines between all directions, each direction's split-half reliability ceiling, its
  head alignment, and the share of each axis's decision disparity it carries;
- **α-sweep**: the own-axis D-disparity at each ``alphas`` value, for the direct, prompt and interaction
  directions;
- **placement check**: the same records' byte-identical cells with the marker in the RESPONSE (the direct
  arm's format) — direct gap vs prompt main effect vs DiD, in one unit;
- **quality tracking**: the AUC of D between strong and weak records next to what document length alone
  reaches, within length strata and length-residualised (``length_bins``; each row carries ``doc_tokens``).

Items come from the generator's ``cells.jsonl`` next to the config's ``dataset_source`` (the direct
manifest). Every record in any direct probe split is excluded from evaluation; a record is evaluated only
if all its requested blocks fit ``max_length`` untruncated (right truncation would cut the response, which
carries the decision). Settings: ``extra.cross_marker`` in the config, overridden by the CLI.

Outputs (no texts — the corpora's licences): ``<out>.json`` (settings, selection, probe metadata, metrics,
placement check, geometry, α-sweep), ``<out>_rewards.jsonl`` and ``<out>_direct_rewards.jsonl`` (one row
per scored text, one reward column per variant) and ``<out>_directions.pt`` (every direction, per fold).

Usage:
    python runners/run_cross_marker.py --config configs/demographic_credit_crossmarker_qwen06.yaml
    python runners/run_cross_marker.py --config ... --n-strong 8 --n-weak 8 --device mps   # smoke
    python runners/run_cross_marker.py --config ... --model Skywork/Skywork-Reward-V2-Llama-3.1-8B --batch-size 32

``summary["timing"]`` (and the report's last line) gives the seconds per phase, the texts that went through
the model, the scoring throughput and the peak GPU memory — the numbers that size the cluster runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pairs.cross_marker import CellBlock, build_block_items, fits_max_length, load_cell_blocks
from pairs.factorial import FactorialDesign, stable_rng
from scoring.cross_marker_metrics import cross_marker_metrics, placement_check
from scoring.dataset_base import format_conversation

logger = logging.getLogger(__name__)

DIRECTION_SOURCES = ("direct", "prompt", "interaction", "unfair")
DEFAULTS: Dict[str, Any] = {
    "n_strong": 300,
    "n_weak": 300,
    "encodings": ["explicit", "proxy"],
    "templates": None,          # None = every template in cells.jsonl
    "include_unmarked": True,
    "paraphrases": 3,
    "seed": 42,
    "n_boot": 2000,
    "directions": list(DIRECTION_SOURCES),   # [] = baseline only
    "n_folds": 5,
    "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
    "placement_check": True,
    "length_bins": 5,           # quantile strata for the within-length AUC of D (quality tracking)
}


class PhaseTimer:
    """Wall-clock seconds per phase of a run, summed over repeats: ``lap(name)`` books the time since the
    previous lap (or the start) to ``name``. Feeds ``summary["timing"]``, the throughput numbers that size
    the cluster runs."""

    def __init__(self) -> None:
        self.start = self._last = time.perf_counter()
        self.seconds: Dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.seconds[name] = self.seconds.get(name, 0.0) + now - self._last
        self._last = now

    def total(self) -> float:
        return time.perf_counter() - self.start


# --------------------------------------------------------------------------- settings ----------------
def resolve_settings(extra: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """DEFAULTS < config ``extra.cross_marker`` < CLI (``None`` in ``overrides`` = not given)."""
    settings = dict(DEFAULTS)
    configured = extra.get("cross_marker") or {}
    unknown = set(configured) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown cross_marker settings {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    settings.update(configured)
    settings.update({k: v for k, v in overrides.items() if v is not None})
    bad = set(settings["directions"]) - set(DIRECTION_SOURCES)
    if bad:
        raise ValueError(f"directions must be drawn from {DIRECTION_SOURCES}, got {sorted(bad)}")
    return settings


def cells_path(dataset_source: Optional[str], default_pairs: str, override: Optional[str]) -> Path:
    """The generator writes cells.jsonl next to pairs.jsonl."""
    if override:
        return Path(override)
    return Path(dataset_source or default_pairs).parent / "cells.jsonl"


# --------------------------------------------------------------------------- selection ---------------
def select_records(
    blocks: Iterable[CellBlock],
    *,
    quality_field: str,
    encodings: Sequence[str],
    templates: Sequence[str],
    exclude: Set[str],
    n_strong: int,
    n_weak: int,
    seed: int,
    fits: Callable[[List[CellBlock]], bool] = lambda blocks: True,
) -> Tuple[Dict[str, List[CellBlock]], Dict[str, Any]]:
    """Seeded choice of ``n_strong`` strong and ``n_weak`` weak records, each with a block for every
    requested (encoding, template), none in ``exclude`` (probe records), and all of whose blocks pass
    ``fits`` (the length guard; checked lazily, only for candidates reached). Returns
    ``{record_id: blocks}`` and a report of how many records each filter removed."""
    wanted = {(e, t) for e in encodings for t in templates}
    by_record: Dict[str, List[CellBlock]] = defaultdict(list)
    for b in blocks:
        if (b.encoding, b.template_id) in wanted:
            by_record[b.record_id].append(b)
    report: Dict[str, Any] = {"records_in_cells": len(by_record), "excluded_probe_records": 0,
                              "incomplete_records": 0, "too_long_records": 0}
    pools: Dict[bool, List[str]] = {True: [], False: []}
    for rid, bs in by_record.items():
        if rid in exclude:
            report["excluded_probe_records"] += 1
        elif {(b.encoding, b.template_id) for b in bs} != wanted:
            report["incomplete_records"] += 1
        else:
            pools[bs[0].is_strong(quality_field)].append(rid)
    selected: Dict[str, List[CellBlock]] = {}
    for strong, n in ((True, n_strong), (False, n_weak)):
        order = sorted(pools[strong])
        stable_rng(seed, "cross_marker_records", strong).shuffle(order)
        taken = 0
        for rid in order:
            if taken >= n:
                break
            if not fits(by_record[rid]):
                report["too_long_records"] += 1
                continue
            selected[rid] = sorted(by_record[rid], key=lambda b: (b.encoding, b.template_id))
            taken += 1
        group = "strong" if strong else "weak"
        report[f"available_{group}"] = len(pools[strong])
        report[f"n_{group}"] = taken
        report[f"requested_{group}"] = n
    return selected, report


# --------------------------------------------------------------------------- items -> rows -----------
def build_rows(selected: Dict[str, List[CellBlock]], domain: str, quality_field: str,
               format_fn: Callable[[str, str], Any], settings: Dict[str, Any]
               ) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """One row per (record, encoding, template, cell, response) and its formatted conversation (same
    order). Rows carry ids and labels only, never text."""
    rows: List[Dict[str, Any]] = []
    convs: List[Any] = []
    for rid, blocks in selected.items():
        for block in blocks:
            strong = block.is_strong(quality_field)
            for item in build_block_items(block, domain, seed=settings["seed"],
                                          n_paraphrases=settings["paraphrases"],
                                          include_unmarked=settings["include_unmarked"]):
                rows.append({"record_id": rid, "template_id": item.template_id, "encoding": item.encoding,
                             "cell": "unmarked" if item.cell is None else list(item.cell),
                             "response": item.response, "paraphrase": item.paraphrase, "strong": strong})
                convs.append(format_fn(item.prompt, item.text))
    return rows, convs


def build_direct_rows(selected: Dict[str, List[CellBlock]], design: FactorialDesign, quality_field: str,
                      assessment_prompt: str, format_fn: Callable[[str, str], Any]
                      ) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """The placement check's other side: each block's 8 cells with the marker in the RESPONSE, exactly as
    the direct arm formats them (the assessment prompt, the document as the assistant turn)."""
    rows: List[Dict[str, Any]] = []
    convs: List[Any] = []
    for rid, blocks in selected.items():
        for block in blocks:
            for cell in design.cells:
                rows.append({"record_id": rid, "template_id": block.template_id, "encoding": block.encoding,
                             "cell": list(cell), "strong": block.is_strong(quality_field)})
                convs.append(format_fn(assessment_prompt, block.texts[cell]))
    return rows, convs


def document_lengths(selected: Dict[str, List[CellBlock]], tokenizer: Any) -> Dict[Tuple[str, str], int]:
    """Tokens of each record's document per template (the unmarked text, the same under every encoding):
    the length reference for quality tracking. The decision request around it is constant, so it would
    shift every length alike and change no AUC."""
    out: Dict[Tuple[str, str], int] = {}
    for rid, blocks in selected.items():
        for block in blocks:
            key = (rid, block.template_id)
            if key not in out:
                out[key] = len(tokenizer(block.unmarked, add_special_tokens=False)["input_ids"])
    return out


def token_counter(tokenizer: Any) -> Callable[[Any], int]:
    """Tokens of one formatted conversation as the forward pass tokenizes it (special tokens included,
    no truncation); pair-format inputs are (prompt, response) tuples."""
    def count(conv: Any) -> int:
        ids = tokenizer(*conv)["input_ids"] if isinstance(conv, tuple) else tokenizer(conv)["input_ids"]
        return len(ids)
    return count


def block_fits(domain: str, settings: Dict[str, Any], format_fn: Callable[[str, str], Any],
               count: Callable[[Any], int], max_length: int) -> Callable[[List[CellBlock]], bool]:
    def fits(blocks: List[CellBlock]) -> bool:
        return all(fits_max_length(
            (format_fn(i.prompt, i.text) for i in build_block_items(
                b, domain, seed=settings["seed"], n_paraphrases=settings["paraphrases"],
                include_unmarked=settings["include_unmarked"])),
            count, max_length) for b in blocks)
    return fits


# --------------------------------------------------------------------------- directions --------------
def record_contrasts(pairs: Sequence[Any], pos: Any, neg: Any) -> Tuple[List[str], Any]:
    """The records (first-seen order) and, one row each, the mean over the record's pairs of the state
    difference positive − negative. The unit that split-half reliabilities and bootstraps resample,
    since a record's pairs share its content."""
    import torch

    by_record: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        by_record[str(p.metadata["source_record_id"])].append(i)
    return list(by_record), torch.stack([(pos[idx] - neg[idx]).mean(0) for idx in by_record.values()])


def record_contrast_matrix(pairs: Sequence[Any], pos: Any, neg: Any) -> Any:
    """`record_contrasts` without the record ids."""
    return record_contrasts(pairs, pos, neg)[1]


def direct_directions(model: Any, tokenizer: Any, dom: Any, source: str, encodings: Sequence[str], *,
                      probe_size: int, split_seed: int, batch_size: int, device: str, max_length: int,
                      reliability_seed: int = 0, probe_records: Optional[int] = None
                      ) -> Tuple[Dict[str, Dict[str, Any]], Set[str], Dict[str, Any]]:
    """The direct arm's difference-of-means direction for every (encoding, axis) the factorial has pairs
    for, the union of their probe records, and per-direction metadata (incl. the split report and a
    split-half reliability over the probe records, from the same states — embedding-cache hits).

    With ``probe_records`` every direction rests on that many records, stratified by quality and the
    same for every axis (the split ignores the axis), so the union is those records."""
    import torch

    from probes.cross_marker_directions import split_half_cosine
    from probes.probe import build_probe_direction, embed_states

    directions: Dict[str, Dict[str, Any]] = {}
    probe_ids: Set[str] = set()
    meta: Dict[str, Any] = {}
    for encoding in encodings:
        for axis in dom.axes:
            if not dom.factorial.axis_pairs(axis, encoding):
                continue
            ds = dom.dataset_cls(source, axis=axis, encoding=encoding, probe_size=probe_size,
                                 split_seed=split_seed, probe_records=probe_records)
            pairs = ds.get_probe_pairs(tokenizer)
            if not pairs:
                raise ValueError(f"{dom.name}/{axis}/{encoding}: no probe pairs in {source}")
            probe, m = build_probe_direction(model, tokenizer, pairs, batch_size=batch_size, device=device,
                                             max_length=max_length)
            ids = ds.probe_record_ids()
            pos, _ = embed_states(model, tokenizer, [p.positive_text for p in pairs], batch_size=batch_size,
                                  max_length=max_length, show_progress=False)
            neg, _ = embed_states(model, tokenizer, [p.negative_text for p in pairs], batch_size=batch_size,
                                  max_length=max_length, show_progress=False)
            contrasts = record_contrast_matrix(pairs, pos, neg)
            directions.setdefault(encoding, {})[axis] = probe
            probe_ids |= ids
            meta[f"{encoding}/{axis}"] = {"n_pairs": len(pairs), "n_records": len(ids),
                                          "split": ds.split_report(),
                                          "probe_accuracy": m.get("probe_accuracy"),
                                          "separation": m.get("separation"),
                                          "split_half_cosine": split_half_cosine(contrasts, reliability_seed)}
    return directions, probe_ids, meta


# --------------------------------------------------------------------------- scoring -----------------
def _set_column(rows: List[Dict[str, Any]], idx: Sequence[int], values: Any, column: str) -> None:
    for k, i in enumerate(idx):
        rows[i][column] = float(values[k])


def score_encoding(model: Any, tokenizer: Any, encoding: str, design: FactorialDesign,
                   rows: List[Dict[str, Any]], convs: List[Any], direct_rows: List[Dict[str, Any]],
                   direct_convs: List[Any], direct_dirs: Mapping[str, Any], settings: Dict[str, Any], *,
                   batch_size: int, max_length: int, show_progress: bool = True,
                   timer: Optional[PhaseTimer] = None
                   ) -> Tuple[List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Score one encoding: add every reward column to its ``rows`` and ``direct_rows`` in place, and
    return ``(columns, geometry, alpha_sweep, directions)``. See the module docstring for the variants.
    ``timer`` books the forward pass to ``embed`` and everything after it to ``mechanism``."""
    import torch

    from probes import cross_marker_directions as cmd
    from probes.probe import embed_states, get_score_head, rewards_from_hidden

    x_idx = [i for i, r in enumerate(rows) if r["encoding"] == encoding]
    d_idx = [i for i, r in enumerate(direct_rows) if r["encoding"] == encoding]
    hx, dtype = embed_states(model, tokenizer, [convs[i] for i in x_idx], batch_size=batch_size,
                             max_length=max_length, show_progress=show_progress)
    hd = None
    if d_idx:
        hd, _ = embed_states(model, tokenizer, [direct_convs[i] for i in d_idx], batch_size=batch_size,
                             max_length=max_length, show_progress=show_progress)
    timer = timer or PhaseTimer()
    timer.lap("embed")

    def apply(column: str, basis: Optional[torch.Tensor], alpha: float = 1.0,
              x_sub: Optional[Sequence[int]] = None, d_sub: Optional[Sequence[int]] = None) -> None:
        """Rewards of (a subset of) this encoding's rows with ``basis`` projected out (``None`` = all
        rows of that side, an empty list = none)."""
        for h, sub, idx, table in ((hx, x_sub, x_idx, rows), (hd, d_sub, d_idx, direct_rows)):
            if h is None or (sub is not None and not len(sub)):
                continue
            _, r = rewards_from_hidden(model, h if sub is None else h[list(sub)], dtype, basis,
                                       null_alpha=alpha)
            _set_column(table, idx if sub is None else [idx[k] for k in sub], r, column)

    apply("baseline", None)
    columns = ["baseline"]
    directions: Dict[str, Any] = {}

    # the direct arm's directions, fitted on its probe records (never evaluated here)
    for axis, u in direct_dirs.items():
        apply(f"null_direct:{axis}", u)
        columns.append(f"null_direct:{axis}")
        directions[f"direct:{axis}"] = u
    if len(direct_dirs) > 1:
        apply("null_direct:joint", torch.stack(list(direct_dirs.values())))
        columns.append("null_direct:joint")

    # this design's own directions, cross-fitted over quality-stratified record folds
    x_rows = [rows[i] for i in x_idx]
    index = cmd.state_index(x_rows)
    record_ids = sorted({r["record_id"] for r in x_rows})
    strong = {r["record_id"]: bool(r["strong"]) for r in x_rows}
    folds = cmd.fold_assignment(record_ids, strong, settings["n_folds"], settings["seed"])
    x_by_fold: Dict[int, List[int]] = defaultdict(list)
    for k, r in enumerate(x_rows):
        x_by_fold[folds[r["record_id"]]].append(k)
    d_by_fold: Dict[int, List[int]] = defaultdict(list)
    for k, i in enumerate(d_idx):
        d_by_fold[folds[direct_rows[i]["record_id"]]].append(k)

    axes = [a for a in design.axes if design.axis_pairs(a, encoding)] + ["intersection"]
    own: List[Tuple[str, Optional[str]]] = []
    for kind in ("prompt", "interaction"):
        if kind in settings["directions"]:
            own += [(kind, axis) for axis in axes]
    if "unfair" in settings["directions"]:
        own.append(("unfair", None))
    contrasts: Dict[str, torch.Tensor] = {}
    fitted: Dict[str, Dict[int, torch.Tensor]] = {}
    for kind, axis in own:
        name = f"{kind}:{axis}" if axis else kind
        contrasts[name] = cmd.record_contrasts(hx, index, record_ids, kind, design, encoding, axis)
        fitted[name] = cmd.cross_fitted(contrasts[name], record_ids, folds)
        for f, u in fitted[name].items():
            apply(f"null_{name}", u, x_sub=x_by_fold[f], d_sub=d_by_fold.get(f, []))
        columns.append(f"null_{name}")
        directions[name] = cmd.unit(contrasts[name].mean(0))

    # geometry: cosines, reliability ceilings, head alignment, share of each axis's disparity carried
    w = get_score_head(model).weight.detach().float().cpu().reshape(-1)
    names = list(directions)
    reliability = {n: cmd.split_half_cosine(contrasts[n], settings["seed"]) for n in contrasts}
    # Δ_int over the headline group (strong records), so w·Δ_int reproduces the reported D-disparity
    group_ids = [r for r in record_ids if strong[r]] or record_ids
    delta_int = {axis: cmd.record_contrasts(hx, index, group_ids, "interaction", design, encoding,
                                            axis).mean(0) for axis in axes}
    geometry = {
        "directions": names,
        "cosine": [[cmd.cosine(directions[a], directions[b]) for b in names] for a in names],
        "split_half_cosine": reliability,
        "head_alignment": {n: cmd.head_alignment(w, directions[n]) for n in names},
        "did_share": {axis: {n: cmd.did_share(w, delta_int[axis], directions[n]) for n in names}
                      for axis in axes},
        # w·Δ_int in float32: equals the reward-based D-disparity of the same group up to bf16 rounding
        "did_from_states": {axis: float(w @ delta_int[axis]) for axis in axes},
        "did_group": "strong" if any(strong[r] for r in record_ids) else "weak",
        "n_folds": settings["n_folds"],
    }

    # α-sweep of the own-axis directions (states in memory: only the head runs per α)
    sweep: Dict[str, Any] = {}
    for kind in ("direct", "prompt", "interaction"):
        for axis in axes:
            name = f"{kind}:{axis}"
            if name not in directions:
                continue
            curve = {}
            for alpha in settings["alphas"]:
                if kind == "direct":
                    apply("_alpha", directions[name], alpha, d_sub=[])
                else:
                    for f, u in fitted[name].items():
                        apply("_alpha", u, alpha, x_sub=x_by_fold[f], d_sub=[])
                m = cross_marker_metrics(x_rows, design, encoding, reward_key="_alpha", n_boot=1,
                                         seed=settings["seed"])
                d = m["margins"]["D"].get("strong") or m["margins"]["D"]["weak"]
                curve[str(alpha)] = {"disparity": d["disparity"][axis]["mean"],
                                     "auc_marked": m["accuracy"]["auc_marked"]}
            sweep[name] = curve
    for r in x_rows:
        r.pop("_alpha", None)
    saved = {"direct": {n: v for n, v in directions.items() if n.startswith("direct:")},
             "full_data": {n: v for n, v in directions.items() if not n.startswith("direct:")},
             "cross_fitted": fitted, "folds": folds}
    timer.lap("mechanism")
    return columns, geometry, sweep, saved


def timing_report(timer: PhaseTimer, *, batch_size: int, texts: Optional[Mapping[str, int]]
                  ) -> Dict[str, Any]:
    """Seconds per phase, the texts that went through the model (embedding-cache misses; None with the
    cache off), the scoring throughput and, on CUDA, the peak GPU memory summed over the visible devices."""
    import torch

    seconds = {k: round(v, 1) for k, v in timer.seconds.items()}
    report: Dict[str, Any] = {"seconds": seconds, "total_s": round(timer.total(), 1),
                              "batch_size": batch_size, "texts_through_model": texts}
    if texts and timer.seconds.get("embed"):
        report["scoring_texts_per_s"] = round(texts["scoring"] / timer.seconds["embed"], 1)
    if torch.cuda.is_available():
        devices = range(torch.cuda.device_count())
        report["gpus"] = [torch.cuda.get_device_name(i) for i in devices]
        report["peak_gpu_memory_gib"] = {
            "allocated": round(sum(torch.cuda.max_memory_allocated(i) for i in devices) / 2**30, 2),
            "reserved": round(sum(torch.cuda.max_memory_reserved(i) for i in devices) / 2**30, 2)}
    return report


def credit_reference(dom: Any, selected: Dict[str, List[CellBlock]]) -> Optional[Dict[str, Any]]:
    """Credit only: the common-sense scorer's AUC on the evaluated records (`substrates/credit_reference.py`)."""
    if dom.name != "credit":
        return None
    from scoring.cross_marker_metrics import auc
    from substrates.credit_reference import common_sense_scores

    by_id = {r.source_record_id: r for r in dom.load_records()}
    recs = [by_id[rid] for rid in selected if rid in by_id]
    scores = common_sense_scores(recs)
    pos = [s for s, r in zip(scores, recs) if dom.is_strong(r)]
    neg = [s for s, r in zip(scores, recs) if not dom.is_strong(r)]
    return {"common_sense_auc": auc(pos, neg), "n_strong": len(pos), "n_weak": len(neg),
            "n_missing": len(selected) - len(recs)}


# --------------------------------------------------------------------------- report ------------------
def _num(s: Mapping[str, Any], key: str = "mean", ci: bool = True) -> str:
    v = s.get(key) if s else None
    if v is None or v != v:
        return "—"
    lo, hi = (("ci_low", "ci_high") if key == "mean" else ("scaled_ci_low", "scaled_ci_high"))
    return f"{v:+.3f} [{s[lo]:+.2f},{s[hi]:+.2f}]" if ci and lo in s else f"{v:+.3f}"


def print_report(summary: Dict[str, Any]) -> None:
    sel = summary["selection"]
    print("\n" + "=" * 118)
    print(f"CROSS-MARKER [{summary['domain']}] — {summary['model']}  "
          f"(strong {sel['n_strong']}/{sel['requested_strong']}, weak {sel['n_weak']}/{sel['requested_weak']})")
    print("D-disparity = D(protected) − D(reference), D = r(approve) − r(decline); negative ⇒ the RM favours "
          "declining the protected pole")
    for encoding, by_col in summary["metrics"].items():
        base = by_col["baseline"]
        group = "strong" if "strong" in base["margins"].get("D", {}) else "weak"
        d = base["margins"].get("D", {}).get(group)
        if not d:
            continue
        acc = base.get("accuracy", {})
        print("-" * 118)
        print(f"[{encoding}] {group} records — D-disparity, baseline and with each direction projected out "
              f"(cross-fitted where fitted here)")
        heads = ["baseline", "direct", "direct:joint", "prompt", "interaction", "unfair"]
        print(f"  {'axis':14}" + "".join(f"{h:>16}" for h in heads) + f"{'scaled base':>16}{'CI-AUC':>10}")
        for axis, s in d["disparity"].items():
            cols = ["baseline", f"null_direct:{axis}", "null_direct:joint", f"null_prompt:{axis}",
                    f"null_interaction:{axis}", "null_unfair"]
            vals = []
            for c in cols:
                m = by_col.get(c, {}).get("margins", {}).get("D", {}).get(group, {})
                vals.append(_num(m.get("disparity", {}).get(axis, {}), ci=False))
            ci_auc = acc.get(f"cross_influence_auc:{axis}", {}).get("mean")
            print(f"  {axis:14}" + "".join(f"{v:>16}" for v in vals) +
                  f"{_num(s, 'scaled_mean', ci=False):>16}" + (f"{ci_auc:>+10.3f}" if ci_auc is not None else f"{'—':>10}"))
        if acc:
            print(f"  balanced acc (unmarked) {_num(acc.get('acc_unmarked', {}).get('balanced', {}))}   "
                  f"AUC(D) unmarked {acc.get('auc_unmarked', float('nan')):.3f}  "
                  f"marked {acc.get('auc_marked', float('nan')):.3f}")
        qt = base.get("quality_tracking", {})
        if "auc_d" in qt:
            strata = qt["auc_d_within_length_strata"]
            print(f"  quality tracking: AUC(D) {_num(qt['auc_d'])} | length only {_num(qt['auc_length'])} | "
                  f"within length strata {_num(strata)} (coverage {strata['coverage']:.2f}) | "
                  f"length-residualised {_num(qt['auc_d_length_residualised'])}")
        geo = summary.get("geometry", {}).get(encoding)
        if geo:
            names = geo["directions"]
            cos = {(a, b): geo["cosine"][i][j] for i, a in enumerate(names) for j, b in enumerate(names)}
            rel = {**geo["split_half_cosine"], **{f"direct:{k.split('/')[1]}": v["split_half_cosine"]
                                                   for k, v in summary["probe_directions"].items()
                                                   if k.startswith(encoding + "/")}}
            print(f"  geometry (reliability = split-half cosine; share = part of the DiD the direction carries)")
            for axis in d["disparity"]:
                dn, pn, iname = f"direct:{axis}", f"prompt:{axis}", f"interaction:{axis}"
                c = lambda a, b: f"{cos[(a, b)]:+.2f}" if (a, b) in cos else "—"
                r = lambda n: f"{rel[n]:.2f}" if n in rel and rel[n] == rel[n] else "—"
                share = geo["did_share"].get(axis, {})
                sh = lambda n: f"{share[n]:+.2f}" if n in share and share[n] == share[n] else "—"
                # the full-data interaction direction is Δ_int itself, so its share is 1 by construction;
                # its held-out effect is the cross-fitted "interaction" column above
                print(f"    {axis:14} rel d/p/i {r(dn)}/{r(pn)}/{r(iname)}  cos(d,p) {c(dn, pn)} "
                      f"cos(d,i) {c(dn, iname)} cos(i,unfair) {c(iname, 'unfair')}  "
                      f"share d/p/unfair {sh(dn)}/{sh(pn)}/{sh('unfair')}")
        pc = summary.get("placement", {}).get(encoding, {}).get("baseline", {}).get(group)
        if pc:
            print("  placement (scaled): marker in response (direct gap) | in prompt, approve fixed | DiD")
            for axis, v in pc["axes"].items():
                pe = v.get("prompt_effect", {}).get("approve", {})
                print(f"    {axis:14} {_num(v.get('direct_gap', {}), 'scaled_mean', ci=False):>10} "
                      f"{_num(pe, 'scaled_mean', ci=False):>10} {_num(v.get('did', {}), 'scaled_mean', ci=False):>10}")
    ref = summary.get("credit_reference")
    if ref:
        print(f"credit common-sense reference AUC {ref['common_sense_auc']:.3f} (read AUC(D) against it)")
    probes = summary.get("probe_directions", {})
    if probes:
        records = sorted({v["n_records"] for v in probes.values()})
        strata = {k: v["split"].get("probe_strata") for k, v in probes.items()}
        print(f"direct directions: {'/'.join(map(str, records))} probe records per direction "
              f"({summary['selection']['excluded_probe_records']} excluded from evaluation); "
              f"strata (strong=True) {next(iter(strata.values()))}")
    timing = summary.get("timing")
    if timing:
        phases = " | ".join(f"{k} {v:.0f}s" for k, v in timing["seconds"].items())
        texts = timing.get("texts_through_model")
        rate = timing.get("scoring_texts_per_s")
        mem = timing.get("peak_gpu_memory_gib")
        print(f"timing (batch {timing['batch_size']}): {phases} | total {timing['total_s']:.0f}s"
              + (f" | through the model {texts['direct_directions']} + {texts['scoring']} texts" if texts else "")
              + (f", scoring {rate:.0f} texts/s" if rate else "")
              + (f" | peak GPU {mem['allocated']:.1f} GiB allocated, {mem['reserved']:.1f} reserved" if mem else ""))
    print("=" * 118)


# --------------------------------------------------------------------------- main --------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_credit_crossmarker_qwen06.yaml"))
    ap.add_argument("--cells", default=None, help="cells.jsonl (default: next to the config's dataset_source)")
    ap.add_argument("--encodings", default=None, help="Comma-separated, e.g. explicit,proxy")
    ap.add_argument("--templates", default=None, help="Comma-separated template ids (default: all)")
    ap.add_argument("--n-strong", type=int, default=None)
    ap.add_argument("--n-weak", type=int, default=None)
    ap.add_argument("--paraphrases", type=int, default=None)
    ap.add_argument("--no-unmarked", action="store_true", help="Skip the unmarked control prompt")
    ap.add_argument("--no-placement-check", action="store_true", help="Skip the marker-in-response side")
    ap.add_argument("--directions", default=None,
                    help=f"Comma-separated subset of {','.join(DIRECTION_SOURCES)}, or 'none'")
    ap.add_argument("--probe-records", type=int, default=None,
                    help="Overrides the config's probe_records (records per direct direction)")
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None, help="Overrides the config's device (e.g. cpu, mps)")
    ap.add_argument("--model", default=None,
                    help="Overrides the config's model_path (e.g. an 8B RM on a 0.6B config)")
    ap.add_argument("--batch-size", type=int, default=None, help="Overrides the config's batch_size")
    ap.add_argument("--out", type=Path, default=None,
                    help="Default artifacts/results/demographic/crossmarker_{domain}_{model}.json")
    return ap


def apply_overrides(cfg: Any, args: argparse.Namespace) -> Any:
    """The CLI over the config (config precedence: YAML < CLI), for the keys the runner reads from ``cfg``."""
    for attr, value in (("device", args.device), ("model_path", args.model),
                        ("batch_size", args.batch_size), ("probe_records", args.probe_records)):
        if value is not None:
            setattr(cfg, attr, value)
    return cfg


def main() -> None:
    import torch

    from scoring.demographic_experiment import DemographicBiasExperiment
    from scoring.experiment import ExperimentConfig
    from substrates.domains import get_domain

    timer = PhaseTimer()
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    dom = get_domain(cfg.extra.get("domain", "credit"))
    design = dom.factorial
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None
    settings = resolve_settings(cfg.extra, {
        "encodings": split(args.encodings), "templates": split(args.templates),
        "n_strong": args.n_strong, "n_weak": args.n_weak, "paraphrases": args.paraphrases,
        "include_unmarked": False if args.no_unmarked else None,
        "placement_check": False if args.no_placement_check else None,
        "directions": ([] if args.directions == "none" else split(args.directions)),
        "n_folds": args.n_folds, "n_boot": args.n_boot, "seed": args.seed})
    source = cfg.dataset_source or dom.default_pairs
    path = cells_path(cfg.dataset_source, dom.default_pairs, args.cells)
    slug = Path(cfg.model_path).name
    out = args.out or Path(f"artifacts/results/demographic/crossmarker_{dom.name}_{slug}.json")

    cells_report: Dict[str, Any] = {}
    blocks = load_cell_blocks(path, design, cells_report)
    templates = settings["templates"] or sorted({b.template_id for b in blocks})

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
            model, tok, dom, source, settings["encodings"], probe_size=cfg.probe_size,
            split_seed=cfg.split_seed, batch_size=cfg.batch_size, device=cfg.device,
            max_length=cfg.max_length, reliability_seed=settings["seed"], probe_records=cfg.probe_records)
    direct_misses = None if cache is None else cache.misses
    timer.lap("direct_directions")

    format_fn = lambda prompt, response: format_conversation(tok, prompt, response)
    fits = block_fits(dom.name, settings, format_fn, token_counter(tok), cfg.max_length)
    selected, selection = select_records(
        blocks, quality_field=dom.quality_field, encodings=settings["encodings"], templates=templates,
        exclude=probe_ids, n_strong=settings["n_strong"], n_weak=settings["n_weak"],
        seed=settings["seed"], fits=fits)
    rows, convs = build_rows(selected, dom.name, dom.quality_field, format_fn, settings)
    lengths = document_lengths(selected, tok)
    for row in rows:                      # stored with the rewards, so quality tracking recomputes offline
        row["doc_tokens"] = lengths[(row["record_id"], row["template_id"])]
    direct_rows, direct_convs = ([], [])
    if settings["placement_check"]:
        direct_rows, direct_convs = build_direct_rows(selected, design, dom.quality_field,
                                                      dom.assessment_prompt, format_fn)
    logger.info("Scoring %d decision texts + %d direct-placement texts for %d records",
                len(rows), len(direct_rows), len(selected))
    timer.lap("prepare")

    metrics: Dict[str, Dict[str, Any]] = {}
    placement: Dict[str, Dict[str, Any]] = {}
    geometry: Dict[str, Any] = {}
    sweeps: Dict[str, Any] = {}
    saved: Dict[str, Any] = {}
    all_columns: List[str] = []
    for encoding in settings["encodings"]:
        columns, geometry[encoding], sweeps[encoding], saved[encoding] = score_encoding(
            model, tok, encoding, design, rows, convs, direct_rows, direct_convs,
            direct_dirs.get(encoding, {}), settings, batch_size=cfg.batch_size, max_length=cfg.max_length,
            timer=timer)
        all_columns += [c for c in columns if c not in all_columns]
        enc_rows = [r for r in rows if r["encoding"] == encoding]
        enc_direct = [r for r in direct_rows if r["encoding"] == encoding]
        metrics[encoding] = {c: cross_marker_metrics(enc_rows, design, encoding, reward_key=c,
                                                     n_boot=settings["n_boot"], seed=settings["seed"],
                                                     n_length_bins=settings["length_bins"])
                             for c in columns}
        if enc_direct:
            placement[encoding] = {c: placement_check(enc_direct, enc_rows, design, encoding, reward_key=c,
                                                      n_boot=settings["n_boot"], seed=settings["seed"])
                                   for c in columns}
        timer.lap("metrics")

    reference = credit_reference(dom, selected)
    timer.lap("metrics")
    summary = {
        "model": cfg.model_path, "domain": dom.name, "settings": {**settings, "templates": templates},
        "max_length": cfg.max_length, "probe_records": cfg.probe_records,
        "cells_path": str(path), "cells_report": cells_report,
        "direct_manifest": source, "probe_directions": probe_meta,
        "selection": selection,
        "records": {"strong": [r for r, b in selected.items() if b[0].is_strong(dom.quality_field)],
                    "weak": [r for r, b in selected.items() if not b[0].is_strong(dom.quality_field)]},
        "n_texts": {"decision": len(rows), "direct_placement": len(direct_rows)},
        "reward_columns": all_columns,
        "embedding_cache": None if cache is None else {"hits": cache.hits, "misses": cache.misses},
        "metrics": metrics,
        "placement": placement,
        "geometry": geometry,
        "alpha_sweep": sweeps,
        "credit_reference": reference,
        "timing": timing_report(timer, batch_size=cfg.batch_size, texts=None if cache is None else {
            "direct_directions": direct_misses, "scoring": cache.misses - direct_misses}),
        "caveats": [
            "Direct probe directions are fitted on probe_records records per direction, stratified by "
            "quality and shared by every axis (probe_directions[*].split); none of them is evaluated. "
            "If probe_records is None they are counted in PAIRS (probe_size), ~38 records per single axis.",
            "Last-token projection only: an RM that also reads the prompt elsewhere (QRM's gate) keeps a "
            "second pathway.",
            "Directions fitted on this design (prompt, interaction, unfair) are cross-fitted for nulling; "
            "geometry and did_share use full-data fits (descriptive, in-sample).",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    stem = out.with_suffix("")
    for suffix, table in (("_rewards.jsonl", rows), ("_direct_rewards.jsonl", direct_rows)):
        with open(f"{stem}{suffix}", "w") as f:
            for row in table:
                f.write(json.dumps(row) + "\n")
    torch.save(saved, f"{stem}_directions.pt")
    print_report(summary)
    print(f"saved → {out} (+ _rewards.jsonl, _direct_rewards.jsonl, _directions.pt)")


if __name__ == "__main__":
    main()
