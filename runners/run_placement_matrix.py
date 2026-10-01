#!/usr/bin/env python3
"""
Placement matrix, one RM (design 2026-10-01, working notes): does the demographic effect of one placement of the
marker live along the direction another placement found? Exploratory until the headline family is fixed.

The three placements, all scored on the same records:

- **direct** (D): the document with the marker as the scored response (the direct arm's format: the domain's
  assessment prompt, the document as the assistant turn), every factorial cell;
- **cross** (X): the cross-marker decision design: the marked document in the user turn, an attribute-free
  approve / decline as the response (`pairs/cross_marker.py`), every cell plus the unmarked control;
- **comparative** (C): two applicants in the user turn, a response that chooses one (`pairs/comparative.py`).

**One record set, one fold assignment.** The records are the comparative arm's pairs, drawn by the same calls as
`runners/run_comparative.py` with the same settings (so its texts are embedding-cache hits), and the folds are its
pair folds (stratified by pairing). Each record is in at most one pair and takes its pair's fold, so every item of
every placement belongs to one fold, and a direction fitted outside fold f nulls fold f in all three placements: no
item is ever nulled by a direction that saw its record.

Rows (directions, per contrast axis — the comparative contrast axes plus the intersection corner):

- the matrix that is read, all fitted on the paired records and cross-fitted over the same folds:
  ``d_shared`` (the direct contrast h(A document) − h(B document) per record), ``x_interaction`` (the part of the
  state that produces the decision disparity, `probes.cross_marker_directions`), ``c_own`` ("the chosen applicant is
  protected", `probes.comparative_directions`);
- side rows, exploratory: ``d_probe`` (the direct arm's direction, fitted on the manifest's probe split, which is
  never paired: the link to the arms' ``null_direct`` columns) and ``x_prompt`` (the attribute of the person judged,
  approve and decline alike).

Targets (`scoring/placement_matrix.py`; the unit is the pair, one bootstrap shared by every cell of an encoding):
D the direct gap r(A) − r(B) over the paired records; X the decision disparity on the strong records; C the merit
marker effect, pairings pooled. A direction is read against its own axis's effect; within one encoding only. Per
cell: baseline, nulled, change, gap (own change − this row's change) and the shortfall, the gap oriented so that
positive = this row removed less than the own one.

Reward columns: ``baseline`` and ``null:{row}:{axis}``; for gated heads (QRM) also ``gate_fixed`` (each row with the
gate of its unmarked prompt: C the pair's in the same order and template, X the record's in the same template; D's
prompt is the assessment prompt for every row, so its gate is fixed already) and ``null_gf:{row}:{axis}`` (nulled
with the fixed gate), read against ``gate_fixed`` — the projection acts on the last-token state only, so the
gate-fixed matrix is the clean transfer reading (listed first). Cross-marker items use the approve and decline
responses only.

Inputs as `run_comparative.py` (its config, its pre-load checks). The pairs, folds and text lengths are checked before
any forward pass; X and D texts over ``max_length`` are refused, not dropped, since dropping would change the pairs.
Settings: ``extra.comparative`` (the pairing; ``directions`` is not read) and ``extra.placement`` (``x_paraphrases``:
the cross-marker paraphrase setting, default the cross-marker runner's code default, 3, which every cross-marker
config uses; if a domain's config ever changes it, set it here too, or its texts stop matching). Cross-marker texts are
embedding-cache hits from the cross-marker arm only under the same ``max_length``: education's comparative config has
4096 and its cross-marker config 2048 (the cache key holds max_length), so there they are embedded anew — the states
are the same, inputs are refused, never truncated, and no single-essay input exceeds ~1,500 tokens, so no record is
accepted here that the cross-marker arm would refuse.

Outputs (no texts): ``<out>.json`` (``meta``, pairing report, the pairs and folds, the matrix per encoding and gate
mode, geometry, timing), ``<out>_rewards.jsonl`` (one row per scored text, tagged with its placement; streamed per
encoding into a ``.partial`` file, renamed at the end) and ``<out>_directions.pt``; the JSON is written last. The default ``<out>`` is ``placement_{domain}[_{manifest folder}]_{model}.json``, plus
``__{setting}-{value}`` for every result-relevant setting the CLI changed; an existing result is never replaced
without ``--overwrite``.

Usage:
    python runners/run_placement_matrix.py --config configs/demographic_credit_comparative_qwen06.yaml
    python runners/run_placement_matrix.py --config ... --n-pairs 4 --device mps          # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pairs.comparative import COMPARATIVE_FRAMES, MATCH_FIELDS, PAIRINGS, UNMARKED, RecordPair, assign_pools, \
    contrast_axes, draw_pairs
from pairs.cross_marker import DECISION_RESPONSES, build_block_items, load_cell_blocks
from pairs.factorial import FactorialDesign
from runners import run_comparative as rc
from runners.run_cross_marker import DEFAULTS as CROSS_MARKER_DEFAULTS
from runners.run_cross_marker import PhaseTimer, cells_path, check_requested, direct_directions, timing_report, \
    token_counter
from scoring.comparative_metrics import pair_values
from scoring.cross_marker_metrics import RewardIndex, record_axis_effect
from scoring.dataset_base import format_conversation
from scoring.experiment import add_override_args, apply_overrides, data_file, run_metadata, variant_suffix
from scoring.placement_matrix import cell_table, shared_draws, unit_sums

logger = logging.getLogger(__name__)

PLACEMENTS = ("direct", "cross", "comparative")
MATRIX_ROWS = ("d_shared", "x_interaction", "c_own")     # same records, n and folds: the matrix that is read
SIDE_ROWS = ("d_probe", "x_prompt")                      # exploratory
ROWS = MATRIX_ROWS + SIDE_ROWS
OWN_ROW = {"direct": "d_shared", "cross": "x_interaction", "comparative": "c_own"}
X_RESPONSES = ("approve", "decline")
DIRECT_RESPONSE = "document"       # the direct placement's one "response": the document is the scored text
PLACEMENT_DEFAULTS: Dict[str, Any] = {"x_paraphrases": CROSS_MARKER_DEFAULTS["paraphrases"]}
# gated heads: the gate-fixed matrix first, the clean transfer reading (user decision 2026-10-01)
GATE_MODES = {"gate_fixed": ("gate_fixed", "null_gf"), "own_gates": ("baseline", "null")}


# --------------------------------------------------------------------------- settings ----------------
def resolve_placement(extra: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """PLACEMENT_DEFAULTS < config ``extra.placement`` < CLI (``None`` = not given)."""
    settings = dict(PLACEMENT_DEFAULTS)
    configured = extra.get("placement") or {}
    unknown = set(configured) - set(PLACEMENT_DEFAULTS)
    if unknown:
        raise KeyError(f"unknown placement settings {sorted(unknown)}; known: {sorted(PLACEMENT_DEFAULTS)}")
    settings.update(configured)
    settings.update({k: v for k, v in overrides.items() if v is not None})
    return settings


def resolve_all(extra: Mapping[str, Any], comparative: Mapping[str, Any], placement: Mapping[str, Any]
                ) -> Dict[str, Any]:
    """The comparative pairing settings (without ``directions``, which the matrix does not read) and the
    placement settings, in one dict (what the result name and ``meta`` record)."""
    settings = rc.resolve_settings(dict(extra), dict(comparative))
    settings.pop("directions")
    return {**settings, **resolve_placement(extra, placement)}


def check_placement(settings: Mapping[str, Any], domain: str) -> None:
    """The comparative checks plus the placement settings, before the model loads."""
    rc.check_settings({**settings, "directions": []}, domain)
    size = DECISION_RESPONSES[domain].size
    if not 1 <= settings["x_paraphrases"] <= size:
        raise SystemExit(f"bad placement settings: x_paraphrases must be in 1..{size}")


def default_out(domain: str, source: Path | str, model_path: str, variant: str = "") -> Path:
    """``placement_{domain}[_{manifest folder}]_{model}{variant}.json``, as the comparative runner names its
    results."""
    folder = Path(source).parent.name
    stem = domain if folder == domain else f"{domain}_{folder}"
    return Path("artifacts/results/demographic") / f"placement_{stem}_{Path(model_path).name}{variant}.json"


# --------------------------------------------------------------------------- items -> rows -----------
def build_cross_rows(pairs: Sequence[RecordPair], by_record: rc.BlockMap, domain: str, encoding: str,
                     templates: Sequence[str], quality_field: str, settings: Mapping[str, Any],
                     format_fn: Callable[[str, str], Any]) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """The cross-marker items of every paired record: per template the 8 cells and the unmarked control, each with
    the approve and decline responses at the record's paraphrase index (the cross-marker arm's texts)."""
    rows: List[Dict[str, Any]] = []
    convs: List[Any] = []
    for pair in pairs:
        for rid in (pair.x, pair.y):
            for template in templates:
                block = by_record[rid][(encoding, template)]
                strong = block.is_strong(quality_field)
                for item in build_block_items(block, domain, seed=settings["seed"],
                                              n_paraphrases=settings["x_paraphrases"], include_unmarked=True,
                                              responses=X_RESPONSES):
                    rows.append({"pair_id": pair.pair_id, "record_id": rid, "template_id": template,
                                 "encoding": encoding, "cell": "unmarked" if item.cell is None else list(item.cell),
                                 "response": item.response, "paraphrase": item.paraphrase, "strong": strong})
                    convs.append(format_fn(item.prompt, item.text))
    return rows, convs


def build_direct_rows(pairs: Sequence[RecordPair], by_record: rc.BlockMap, design: FactorialDesign, encoding: str,
                      templates: Sequence[str], quality_field: str, assessment_prompt: str,
                      format_fn: Callable[[str, str], Any]) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """The direct placement of every paired record: per template the 8 cells with the marked document as the
    response to the assessment prompt (as `run_cross_marker.build_direct_rows`, the direct arm's format)."""
    rows: List[Dict[str, Any]] = []
    convs: List[Any] = []
    for pair in pairs:
        for rid in (pair.x, pair.y):
            for template in templates:
                block = by_record[rid][(encoding, template)]
                for cell in design.cells:
                    rows.append({"pair_id": pair.pair_id, "record_id": rid, "template_id": template,
                                 "encoding": encoding, "cell": list(cell), "response": DIRECT_RESPONSE,
                                 "strong": block.is_strong(quality_field)})
                    convs.append(format_fn(assessment_prompt, block.texts[cell]))
    return rows, convs


def check_lengths(convs: Mapping[str, Sequence[Any]], count: Callable[[Any], int], max_length: int) -> None:
    """Every text within ``max_length`` (the comparative texts passed the pairing's check). Refused, not dropped:
    dropping a record would change the pairs."""
    for placement, cs in convs.items():
        counts = count.batch(list(cs)) if hasattr(count, "batch") else [count(c) for c in cs]
        over = sum(n > max_length for n in counts)
        if over:
            raise SystemExit(f"{over} {placement} texts exceed max_length {max_length} (longest {max(counts)}); "
                             f"raise max_length — dropping them would change the pairs")


def gate_refs(placement: str, rows: Sequence[Mapping[str, Any]]) -> List[int]:
    """Row → the row whose gate it is scored with in ``gate_fixed``: C the pair's first unmarked prompt in the same
    order and template, X the record's unmarked control in the same template, D itself (one prompt for all)."""
    if placement == "direct":
        return list(range(len(rows)))
    ref: Dict[Tuple[str, ...], int] = {}
    if placement == "comparative":
        key = lambda r: (r["pair_id"], r["template_id"], r["order"])
        for i, r in enumerate(rows):
            if r["axis"] == UNMARKED:
                ref.setdefault(key(r), i)
    else:
        key = lambda r: (r["record_id"], r["template_id"])
        for i, r in enumerate(rows):
            if r["cell"] == "unmarked":
                ref.setdefault(key(r), i)
    missing = [i for i, r in enumerate(rows) if key(r) not in ref]
    if missing:
        raise ValueError(f"{placement}: {len(missing)} rows have no unmarked prompt to take the gate from")
    return [ref[key(r)] for r in rows]


# --------------------------------------------------------------------------- directions --------------
def fit_rows(states: Mapping[str, Any], rows: Mapping[str, Sequence[Mapping[str, Any]]], axes: Sequence[str],
             design: FactorialDesign, encoding: str, folds: Mapping[str, int], seed: int,
             probe: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Dict[int, Any]], Dict[str, Any],
                                                                   Dict[str, float]]:
    """Every row's direction per axis: ``{"{row}:{axis}": {fold: unit direction fitted outside the fold}}``, the
    full-data directions (geometry) and their split-half cosines. ``folds`` maps pairs to folds; the record rows
    (D, X) take their pair's. ``probe`` (axis → the direct arm's direction) gives ``d_probe``, the same in every
    fold: fitted on the probe split, which is never paired."""
    from probes import cross_marker_directions as cmd
    from probes.comparative_directions import pair_contrasts

    record_fold: Dict[str, int] = {}
    for placement in ("direct", "cross"):
        for r in rows[placement]:
            if record_fold.setdefault(r["record_id"], folds[r["pair_id"]]) != folds[r["pair_id"]]:
                raise ValueError(f"record {r['record_id']} is in two pairs")
    indexes = {p: cmd.state_index(rows[p]) for p in ("direct", "cross")}
    record_ids = {p: sorted({r["record_id"] for r in rows[p]}) for p in ("direct", "cross")}
    fitted: Dict[str, Dict[int, Any]] = {}
    full: Dict[str, Any] = {}
    split_half: Dict[str, float] = {}
    for axis in axes:
        sources = (("d_shared", "direct", "prompt"), ("x_interaction", "cross", "interaction"),
                   ("x_prompt", "cross", "prompt"))
        contrasts: Dict[str, Tuple[List[str], Any, Mapping[str, int]]] = {}
        for row, placement, kind in sources:
            ids = record_ids[placement]
            contrasts[row] = (ids, cmd.record_contrasts(states[placement], indexes[placement], ids, kind, design,
                                                        encoding, axis), record_fold)
        pair_ids, c = pair_contrasts(states["comparative"], rows["comparative"], axis)
        contrasts["c_own"] = (pair_ids, c, folds)
        for row, (ids, c, fold_of) in contrasts.items():
            name = f"{row}:{axis}"
            fitted[name] = cmd.cross_fitted(c, ids, fold_of)
            full[name] = cmd.unit(c.mean(0))
            split_half[name] = cmd.split_half_cosine(c, seed)
        if probe is not None:
            u = probe[axis]
            fitted[f"d_probe:{axis}"] = {f: u for f in sorted(set(folds.values()))}
            full[f"d_probe:{axis}"] = u
    return fitted, full, split_half


# --------------------------------------------------------------------------- scoring -----------------
def null_columns(model: Any, states: Mapping[str, Any], gates: Mapping[str, Any], dtype: Any,
                 rows: Mapping[str, List[Dict[str, Any]]], refs: Mapping[str, Sequence[int]],
                 fitted: Mapping[str, Mapping[int, Any]], folds: Mapping[str, int]) -> List[str]:
    """Add the reward columns to every placement's rows in place; return their names. Each row is nulled with the
    direction of its pair's fold. Gated heads (``gates`` not None) also get ``gate_fixed`` and ``null_gf:*``."""
    import torch

    from probes.probe import rewards_from_hidden

    gated = any(g is not None for g in gates.values())
    columns = ["baseline"] + (["gate_fixed"] if gated else [])
    columns += [f"null:{name}" for name in fitted] + ([f"null_gf:{name}" for name in fitted] if gated else [])
    for placement in PLACEMENTS:
        h, g, table = states[placement], gates[placement], rows[placement]
        fixed = None if g is None else g[torch.tensor(list(refs[placement]), dtype=torch.long)]
        by_fold: Dict[int, List[int]] = defaultdict(list)
        for i, r in enumerate(table):
            by_fold[folds[r["pair_id"]]].append(i)

        def put(column: str, idx: Sequence[int], values: Any) -> None:
            for k, i in enumerate(idx):
                table[i][column] = float(values[k])

        everything = list(range(len(table)))
        put("baseline", everything, rewards_from_hidden(model, h, dtype, None, gates=g)[0])
        if gated:
            put("gate_fixed", everything, rewards_from_hidden(model, h, dtype, None, gates=fixed)[0])
        for f, idx in sorted(by_fold.items()):
            sel = torch.tensor(idx, dtype=torch.long)
            hf = h[sel]
            for name, per_fold in fitted.items():
                u = per_fold[f]
                put(f"null:{name}", idx, rewards_from_hidden(model, hf, dtype, u,
                                                              gates=None if g is None else g[sel])[1])
                if gated:
                    put(f"null_gf:{name}", idx, rewards_from_hidden(model, hf, dtype, u, gates=fixed[sel])[1])
    return columns


# --------------------------------------------------------------------------- the matrix --------------
def target_values(placement: str, rows: Sequence[Mapping[str, Any]], column: str, axis: str,
                  design: FactorialDesign, encoding: str, index: Optional[RewardIndex] = None,
                  comparative_values: Optional[Mapping[Tuple[str, str], Mapping[str, Mapping[str, float]]]] = None
                  ) -> List[Tuple[str, float]]:
    """``(pair, value)`` entries of one target effect: D the direct gap of each paired record, X the decision
    disparity of each strong record, C the pair's merit marker effect (``comparative_values`` = `pair_values` of the
    column, computed once per column)."""
    if placement == "comparative":
        values = comparative_values if comparative_values is not None else pair_values(rows, column)[0]
        return [(pid, v["marker_effect"]) for pid, v in values[(axis, "merit")].items()]
    index = index or RewardIndex(rows, design, encoding)
    effect = record_axis_effect(index, index.values(rows, column), axis,
                                margin=None if placement == "direct" else "D")
    pair_of = {r["record_id"]: r["pair_id"] for r in rows}
    return [(pair_of[rid], float(effect[k])) for k, rid in enumerate(index.records)
            if placement == "direct" or index.strong[k]]


def matrix(rows: Mapping[str, Sequence[Mapping[str, Any]]], columns: Sequence[str], fitted_names: Sequence[str],
           axes: Sequence[str], design: FactorialDesign, encoding: str, pair_ids: Sequence[str], n_boot: int,
           seed: int) -> Dict[str, Any]:
    """``{gate mode: {target: {axis: cell_table}}}`` with one set of draws over ``pair_ids`` for every cell."""
    draws = shared_draws(len(pair_ids), n_boot, seed)
    indexes = {p: RewardIndex(rows[p], design, encoding) for p in ("direct", "cross")}
    out: Dict[str, Any] = {}
    for mode, (base_col, prefix) in GATE_MODES.items():
        if base_col not in columns:
            continue
        out[mode] = {}
        for placement in PLACEMENTS:
            table = rows[placement]
            cache: Dict[str, Any] = {}

            def values(column: str, axis: str) -> List[Tuple[str, float]]:
                if placement == "comparative" and column not in cache:
                    cache[column] = pair_values(table, column)[0]
                return target_values(placement, table, column, axis, design, encoding, indexes.get(placement),
                                     cache.get(column))

            out[mode][placement] = {}
            for axis in axes:
                base = unit_sums(values(base_col, axis), pair_ids)
                if not base[1].any():
                    logger.warning("%s/%s/%s/%s: no item counts for this target (e.g. no strong record among the "
                                   "pairs); its cells are NaN", encoding, mode, placement, axis)
                nulled = {}
                for name in fitted_names:
                    row, row_axis = name.split(":", 1)
                    if row_axis == axis:
                        sums, counts = unit_sums(values(f"{prefix}:{name}", axis), pair_ids)
                        if not (counts == base[1]).all():
                            raise ValueError(f"{placement}/{axis}/{name}: other units than the baseline's")
                        nulled[row] = sums
                nulled = {r: nulled[r] for r in ROWS if r in nulled}       # the matrix rows first, then the side rows
                out[mode][placement][axis] = cell_table(base, nulled, OWN_ROW[placement], draws)
    return out


def geometry(model: Any, gates: Mapping[str, Any], full: Mapping[str, Any], split_half: Mapping[str, float],
             axes: Sequence[str], probe_reliability: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    """Per axis: the cosines between the rows' full-data directions, each against its ceiling √(rel_a · rel_b)
    (``rel`` = full-sample reliability: split-half stepped up; ``d_probe``'s from its probe records), and each
    direction's head alignment (gated heads: the mean of the three placements' mean effective heads, so each
    placement counts once)."""
    import torch

    from probes import cross_marker_directions as cmd
    from probes.heads import get_head

    head = get_head(model)
    if head.gated:      # each placement counts once, however many rows it has
        w = torch.stack([head.effective_weights(g).mean(0) for g in gates.values()]).mean(0)
    else:
        w = head.effective_weights(None).mean(0)
    reliability = {n: cmd.full_sample_reliability(v) for n, v in split_half.items()}
    for axis in axes:
        if f"d_probe:{axis}" in full:
            reliability[f"d_probe:{axis}"] = float((probe_reliability or {}).get(axis, float("nan")))
    out: Dict[str, Any] = {"head": head.kind if not head.gated else
                           f"{head.kind} (mean effective head, each placement weighted equally)",
                           "split_half_cosine": dict(split_half), "reliability": reliability, "axes": {}}
    for axis in axes:
        names = [r for r in ROWS if f"{r}:{axis}" in full]
        dirs = {r: full[f"{r}:{axis}"] for r in names}
        rel = {r: reliability[f"{r}:{axis}"] for r in names}
        out["axes"][axis] = {
            "rows": names,
            "cosine": [[cmd.cosine(dirs[a], dirs[b]) for b in names] for a in names],
            "cosine_ceiling": [[cmd.cosine_ceiling(rel[a], rel[b]) for b in names] for a in names],
            "head_alignment": {r: cmd.head_alignment(w, dirs[r]) for r in names}}
    return out


def score_encoding(model: Any, tokenizer: Any, rows: Mapping[str, List[Dict[str, Any]]],
                   convs: Mapping[str, List[Any]], axes: Sequence[str], design: FactorialDesign, encoding: str,
                   folds: Mapping[str, int], settings: Mapping[str, Any], *, batch_size: int, max_length: int,
                   probe: Optional[Mapping[str, Any]] = None, probe_reliability: Optional[Mapping[str, float]] = None,
                   show_progress: bool = True, timer: Optional[PhaseTimer] = None
                   ) -> Tuple[List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Embed the three placements of one encoding, fit every row, add the reward columns to ``rows`` in place;
    return ``(columns, matrix, geometry, directions)``."""
    from probes.probe import embed_with_gates

    timer = timer or PhaseTimer()
    states, gates = {}, {}
    dtype = None
    for placement in PLACEMENTS:
        states[placement], dtype, gates[placement] = embed_with_gates(
            model, tokenizer, convs[placement], batch_size=batch_size, max_length=max_length,
            show_progress=show_progress)
    timer.lap("embed")
    fitted, full, split_half = fit_rows(states, rows, axes, design, encoding, folds, settings["seed"], probe)
    refs = {p: gate_refs(p, rows[p]) for p in PLACEMENTS}
    columns = null_columns(model, states, gates, dtype, rows, refs, fitted, folds)
    pair_ids = sorted(folds)
    table = matrix(rows, columns, list(fitted), axes, design, encoding, pair_ids, settings["n_boot"],
                   settings["seed"])
    geo = geometry(model, gates, full, split_half, axes, probe_reliability)
    timer.lap("mechanism")
    return columns, table, geo, {"full_data": full, "cross_fitted": fitted, "folds": dict(folds)}


# --------------------------------------------------------------------------- report ------------------
def _fmt(s: Mapping[str, Any]) -> str:
    m, lo, hi = s.get("mean"), s.get("ci_low"), s.get("ci_high")
    if m is None or m != m:
        return "—"
    return f"{m:+.3f} [{lo:+.2f},{hi:+.2f}]" if lo == lo and hi == hi else f"{m:+.3f}"


def print_report(summary: Mapping[str, Any]) -> None:
    print("\n" + "=" * 110)
    counts = ", ".join(f"{p} {summary['pairing'][p]['n']}" for p in PAIRINGS)
    print(f"PLACEMENT MATRIX [{summary['domain']}] — {summary['model']}  (pairs: {counts})")
    print("per target and axis: baseline, then per row the change (nulled − baseline) and the shortfall (positive = "
          "removed less than the own row); "
          "rows " + ", ".join(MATRIX_ROWS) + " (read) | " + ", ".join(SIDE_ROWS) + " (side)")
    for encoding, modes in summary["matrix"].items():
        for mode, targets in modes.items():
            print("-" * 110)
            print(f"[{encoding} | {mode}]")
            for placement, by_axis in targets.items():
                for axis, cell in by_axis.items():
                    print(f"  {placement:12} {axis:16} baseline {_fmt(cell['baseline'])}  (own: {cell['own_row']})")
                    for row in ROWS:
                        if row in cell["rows"]:
                            c = cell["rows"][row]
                            print(f"  {'':12} {'':16}   {row:14} change {_fmt(c['change']):>24}  "
                                  f"shortfall {_fmt(c['shortfall']):>24}")
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
    ap.add_argument("--paraphrases", type=int, default=None, help="The comparative responses' paraphrases")
    ap.add_argument("--x-paraphrases", type=int, default=None, help="The cross-marker responses' paraphrases")
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default artifacts/results/demographic/placement_{domain}[_{folder}]_{model}"
                         "[__{setting}-{value} per setting changed on the CLI].json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result (and its side files)")
    add_override_args(ap)   # --model, --revision, --batch-size, --device, --probe-records
    return ap


def main() -> None:
    import torch

    from probes import cross_marker_directions as cmd
    from scoring.backend import model_revision
    from scoring.demographic_experiment import DemographicBiasExperiment
    from scoring.experiment import ExperimentConfig
    from substrates.domains import get_domain

    timer = PhaseTimer()
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    configured_cfg = ExperimentConfig.from_yaml(args.config)
    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    dom = get_domain(cfg.extra.get("domain", "credit"))
    if dom.name not in COMPARATIVE_FRAMES:
        raise SystemExit(f"no comparative frame for domain {dom.name!r}")
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None
    settings = resolve_all(cfg.extra, {
        "n_pairs": args.n_pairs, "encodings": split(args.encodings), "templates": split(args.templates),
        "paraphrases": args.paraphrases, "n_folds": args.n_folds, "n_boot": args.n_boot, "seed": args.seed},
        {"x_paraphrases": args.x_paraphrases})
    check_placement(settings, dom.name)
    source = cfg.dataset_source = cfg.dataset_source or dom.default_pairs
    path = cells_path(source, dom.default_pairs)
    configured = {**resolve_all(configured_cfg.extra, {}, {}), "probe_records": configured_cfg.probe_records,
                  "revision": configured_cfg.model_revision}
    used = {**settings, "probe_records": cfg.probe_records, "revision": cfg.model_revision}
    out = args.out or default_out(dom.name, source, cfg.model_path, variant_suffix(configured, used))
    # everything that can fail on the inputs fails here, before the model loads
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it (and its side files), or --out")
    data = {"pairs.jsonl": data_file(source), "cells.jsonl": data_file(path)}
    cells_report: Dict[str, Any] = {}
    blocks = load_cell_blocks(path, dom.factorial, cells_report)
    templates = settings["templates"] or sorted({b.template_id for b in blocks})
    check_requested(blocks, settings["encodings"], templates)
    rc.check_blocks(blocks, dom)
    if cells_report.get("n_dropped_mismatch"):
        logger.warning("%d cells.jsonl blocks dropped as not matching the design", cells_report["n_dropped_mismatch"])
    probe_ids = rc.probe_split_ids(dom, source, cfg.probe_records, cfg.split_seed)
    by_record, complete, candidates, selection = rc.candidates_from(
        blocks, quality_field=dom.quality_field, match_field=MATCH_FIELDS[dom.name], exclude=probe_ids)
    if not candidates:
        raise SystemExit("no record can be paired (every record is a probe record or incomplete)")
    pools = assign_pools(complete, settings["seed"])

    timer.lap("setup")
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    model, tok = exp.model, exp.tokenizer
    cache = getattr(model, "_onejudge_embedding_cache", None)
    timer.lap("load_model")

    # the pairs: the comparative arm's calls with its settings, so the same pairs (and cache hits)
    format_fn = lambda prompt, response: format_conversation(tok, prompt, response)
    count = token_counter(tok)
    check = rc.compatibility(by_record, dom.name, settings, format_fn, count, cfg.max_length)
    pairs, pairing = draw_pairs(candidates, settings["n_pairs"], settings["seed"], check, pools=pools)
    if not pairs:
        raise SystemExit("no pairs could be formed")
    pairing_of = {p.pair_id: p.pairing for p in pairs}
    folds = cmd.fold_assignment(sorted(pairing_of), pairing_of, settings["n_folds"], settings["seed"])
    if len(set(folds.values())) < 2:
        raise SystemExit(f"the {len(pairs)} pairs fill one fold; the matrix's directions are cross-fitted")
    logger.info("records: %d in cells.jsonl, %d probe records excluded; pairs %s", selection["records_in_cells"],
                selection["excluded_probe_records"], {p: pairing[p]["n"] for p in PAIRINGS})

    design = dom.factorial

    def placement_rows(encoding: str, which: Sequence[str] = PLACEMENTS
                       ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Any]]]:
        built = {
            "comparative": lambda: rc.build_rows(pairs, by_record, dom.name, encoding, templates, settings, format_fn),
            "cross": lambda: build_cross_rows(pairs, by_record, dom.name, encoding, templates, dom.quality_field,
                                              settings, format_fn),
            "direct": lambda: build_direct_rows(pairs, by_record, design, encoding, templates, dom.quality_field,
                                                dom.assessment_prompt, format_fn)}
        rows: Dict[str, List[Dict[str, Any]]] = {}
        convs: Dict[str, List[Any]] = {}
        for placement in which:
            rows[placement], convs[placement] = built[placement]()
        return rows, convs

    # every text within max_length before any forward pass (the comparative ones passed the pairing's check); the
    # texts are rebuilt per encoding below, so only one encoding's are held at a time
    for encoding in settings["encodings"]:
        check_lengths(placement_rows(encoding, ("cross", "direct"))[1], count, cfg.max_length)
    timer.lap("prepare")

    probe_dirs, fitted_on, probe_meta = direct_directions(
        model, tok, dom, source, settings["encodings"], probe_records=cfg.probe_records,
        split_seed=cfg.split_seed, batch_size=cfg.batch_size, max_length=cfg.max_length,
        reliability_seed=settings["seed"])
    if not fitted_on <= probe_ids:
        raise RuntimeError("the direct directions were fitted on records outside the excluded probe split")
    direct_misses = None if cache is None else cache.misses
    timer.lap("direct_directions")

    table: Dict[str, Any] = {}
    geo: Dict[str, Any] = {}
    saved: Dict[str, Any] = {}
    columns: List[str] = []
    n_texts: Counter = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    # the reward rows are streamed per encoding into a partial file, renamed once the run is complete
    partial = Path(f"{stem}_rewards.jsonl.partial")
    with open(partial, "w") as f:
        for encoding in settings["encodings"]:
            rows, convs = placement_rows(encoding)
            timer.lap("prepare")
            axes = contrast_axes(design, encoding)
            cols, table[encoding], geo[encoding], saved[encoding] = score_encoding(
                model, tok, rows, convs, axes, design, encoding, folds, settings, batch_size=cfg.batch_size,
                max_length=cfg.max_length, probe=probe_dirs[encoding], timer=timer,
                probe_reliability={a: probe_meta[f"{encoding}/{a}"]["reliability"] for a in axes})
            del convs
            columns += [c for c in cols if c not in columns]
            for placement in PLACEMENTS:
                n_texts[placement] += len(rows[placement])
                for r in rows[placement]:
                    f.write(json.dumps({"placement": placement, **r}) + "\n")
            del rows
            timer.lap("metrics")

    summary = {
        "meta": run_metadata(cfg, data, {**settings, "templates": templates}),
        "model": cfg.model_path, "model_revision": model_revision(cfg), "domain": dom.name,
        "settings": {**settings, "templates": templates}, "max_length": cfg.max_length,
        "probe_records": cfg.probe_records, "cells_path": str(path), "cells_report": cells_report,
        "direct_manifest": source, "probe_directions": probe_meta, "probe_record_ids": sorted(probe_ids),
        "selection": selection, "pool_sizes": dict(sorted(Counter(pools.values()).items())), "pairing": pairing,
        "pairs": [{"pair_id": p.pair_id, "pairing": p.pairing, "x": p.x, "y": p.y, "index": p.index} for p in pairs],
        "folds": folds, "rows": {"matrix": list(MATRIX_ROWS), "side": list(SIDE_ROWS)}, "own_row": OWN_ROW,
        "targets": {"direct": "direct gap r(A) − r(B), paired records",
                    "cross": "decision disparity D(A) − D(B), strong records",
                    "comparative": "merit marker effect, pairings pooled"},
        "n_texts": dict(n_texts), "reward_columns": columns,
        "embedding_cache": None if cache is None else {"hits": cache.hits, "misses": cache.misses},
        "matrix": table, "geometry": geo,
        "timing": timing_report(timer, batch_size=cfg.batch_size, texts=None if cache is None else {
            "direct_directions": direct_misses, "scoring": cache.misses - direct_misses}),
        "caveats": [
            "Exploratory until the headline family is fixed; intervals are uncorrected.",
            "The read matrix is d_shared × x_interaction × c_own (same records, n and folds); d_probe and x_prompt "
            "are side rows.",
            "Every row but d_probe is cross-fitted over the comparative pair folds; d_probe is fitted on the probe "
            "split, which is never paired. Geometry uses full-data fits (descriptive).",
            "Within one encoding; a direction is read against its own axis's effect only.",
            "Binding: the comparative RM must bind the marker to the right applicant; a single linear direction may "
            "not capture that (RQ3).",
        ],
    }
    partial.replace(f"{stem}_rewards.jsonl")
    torch.save(saved, f"{stem}_directions.pt")
    out.write_text(json.dumps(summary, indent=2))      # last: the result exists only once its side files do
    print_report(summary)
    print(f"saved → {out} (+ _rewards.jsonl, _directions.pt)")


if __name__ == "__main__":
    main()
