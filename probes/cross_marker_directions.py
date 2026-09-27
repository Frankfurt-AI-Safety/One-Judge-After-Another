"""
Directions fitted on the cross-marker decision design's own hidden states — the mechanism layer's second
source of directions next to the direct arm's (`probes/probe.py::build_probe_direction`).

For a linear head the reward is linear in the pooled state, r = w·h + b, so every reward effect is the head
reading a difference of states, and projecting out a unit direction u at strength α changes the decision
disparity (D(A) − D(B), the metrics' main effect: mean over pole-A cells − mean over pole-B cells) by exactly

    ΔDiD = −α (w·u)(Δ_int·u),     Δ_int = (h(A, approve) − h(A, decline)) − (h(B, approve) − h(B, decline)).

(For a gated head, QRM, w is each text's effective head ``gate @ R̄``, and the runner's geometry uses their
mean, so there the identity is approximate.) A direction only removes the disparity if the head reads it
(w·u ≠ 0) *and* it carries the interaction (Δ_int·u ≠ 0). The direct arm's direction can therefore be
reward-relevant and still leave the decision bias untouched. Three directions are fitted here, per record as a
difference of states under the same record and template, then averaged over records; each points to pole A
(``prompt``, ``interaction``) or to the overt decline (``unfair``), which the signed geometry (cosines, w·u)
depends on:

- ``prompt``      (per axis) — h(A-prompt, resp) − h(B-prompt, resp), every response: the attribute of the
                  person being judged, as the final token represents it with the marker in the prompt;
- ``interaction`` (per axis) — Δ_int above: the part of the state that produces the decision disparity;
- ``unfair``      (one per encoding) — h(c, overt) − h(c, decline) under the same prompt, every cell and the
                  unmarked control: the RM's representation of an openly attribute-based decision against a
                  neutral one.

Directions are compared with the direct arm's by cosine and by head alignment. A direction fitted on few
records is partly the records' own content, so a cosine is read against its **ceiling**: with each record's
contrast = signal + independent noise, a direction fitted on n records has reliability rel = the expected
cosine between two independent n-record fits, and the cosine between two directions is at most
√(rel_a · rel_b) (it equals that when their signals coincide). ``split_half_cosine`` measures the reliability
of *half* samples; `full_sample_reliability` steps it up to n records (Spearman–Brown), and `cosine_ceiling`
combines two. Nulling uses **cross-fitting**: records are split into folds
(stratified by quality), and each fold's rewards are nulled with the direction fitted on the other folds,
so no record is ever nulled by a direction it helped fit — in-sample, nulling the mean interaction
removes the mean disparity by construction.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from pairs.factorial import Cell, FactorialDesign, stable_rng

DIRECTION_KINDS = ("prompt", "interaction", "unfair")
StateKey = Tuple[str, str, Optional[Cell], str]   # (record, template, cell or None, response)


def _cell(value: Any) -> Optional[Cell]:
    return None if value in (None, "unmarked") else tuple(value)


def state_index(rows: Sequence[Mapping[str, Any]]) -> Dict[StateKey, int]:
    """(record, template, cell, response) -> row position, for one encoding's rows."""
    index: Dict[StateKey, int] = {}
    for i, r in enumerate(rows):
        key = (str(r["record_id"]), str(r["template_id"]), _cell(r["cell"]), r["response"])
        if key in index:
            raise ValueError(f"duplicate state key {key}")
        index[key] = i
    return index


def _terms(kind: str, axis: Optional[str], design: FactorialDesign, encoding: str, templates: Iterable[str],
           responses: Sequence[str]) -> List[Tuple[List[Tuple[str, Optional[Cell], str]],
                                                   List[Tuple[str, Optional[Cell], str]]]]:
    """The (plus, minus) state keys of one record's contrast units (without the record id)."""
    units = []
    for t in templates:
        if kind == "prompt":
            for a, b in design.axis_pairs(axis, encoding):
                for resp in responses:
                    units.append(([(t, a, resp)], [(t, b, resp)]))
        elif kind == "interaction":
            for a, b in design.axis_pairs(axis, encoding):
                units.append(([(t, a, "approve"), (t, b, "decline")], [(t, a, "decline"), (t, b, "approve")]))
        elif kind == "unfair":
            for c in list(design.cells) + [None]:
                units.append(([(t, c, "overt")], [(t, c, "decline")]))
        else:
            raise ValueError(f"kind must be one of {DIRECTION_KINDS}, got {kind!r}")
    return units


def record_contrasts(states: torch.Tensor, index: Mapping[StateKey, int], record_ids: Sequence[str],
                     kind: str, design: FactorialDesign, encoding: str,
                     axis: Optional[str] = None) -> torch.Tensor:
    """[n_records, d] per-record contrast vectors (float32): the mean over the record's templates and
    contrast units of Σ plus − Σ minus. Keys absent from ``index`` (e.g. no unmarked control) are skipped
    unit by unit, so the unfair contrast works with or without the control."""
    responses = sorted({k[3] for k in index})
    templates_of: Dict[str, List[str]] = {}
    for rid, tid, _, _ in index:
        templates_of.setdefault(rid, [])
        if tid not in templates_of[rid]:
            templates_of[rid].append(tid)
    plus_idx: List[int] = []
    plus_of: List[int] = []
    minus_idx: List[int] = []
    minus_of: List[int] = []
    units = torch.zeros(len(record_ids), dtype=torch.float32)
    for n, rid in enumerate(record_ids):
        for plus, minus in _terms(kind, axis, design, encoding, sorted(templates_of[rid]), responses):
            keys_p = [(rid, *k) for k in plus]
            keys_m = [(rid, *k) for k in minus]
            if all(k in index for k in keys_p + keys_m):
                plus_idx += [index[k] for k in keys_p]
                minus_idx += [index[k] for k in keys_m]
                plus_of += [n] * len(keys_p)
                minus_of += [n] * len(keys_m)
                units[n] += 1
    if (units == 0).any():
        rid = record_ids[int((units == 0).nonzero()[0])]
        raise ValueError(f"record {rid}: no complete {kind} contrast for axis {axis!r}")
    # all records' sums at once (a per-record loop of tiny ops was the mechanism layer's cost, 2026-09-26)
    plus = _segment_sum(states, plus_idx, plus_of, len(record_ids))
    minus = _segment_sum(states, minus_idx, minus_of, len(record_ids))
    return (plus - minus) / units[:, None]


def _segment_sum(states: torch.Tensor, rows: List[int], owner: List[int], n: int,
                 chunk: int = 1 << 15) -> torch.Tensor:
    """float32 [n, d]: row ``owner[k]`` accumulates ``states[rows[k]]``, gathered in chunks so no more than
    ``chunk`` state rows are copied at a time."""
    out = torch.zeros(n, states.shape[1], dtype=torch.float32)
    rows_t, owner_t = torch.tensor(rows, dtype=torch.long), torch.tensor(owner, dtype=torch.long)
    for start in range(0, len(rows), chunk):
        out.index_add_(0, owner_t[start:start + chunk], states[rows_t[start:start + chunk]].float())
    return out


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def _degenerate(v: torch.Tensor) -> bool:
    """A (near-)zero direction: the contrast's two sides were identical, e.g. a marker the tokenizer
    cannot see. Nulling it is a no-op (`gram_schmidt` drops it); its geometry is undefined, not 0."""
    return float(v.float().norm()) < 1e-8


def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    if _degenerate(u) or _degenerate(v):
        return float("nan")
    return float(unit(u.float()) @ unit(v.float()))


def fold_assignment(record_ids: Sequence[str], strong: Mapping[str, bool], k: int,
                    seed: int) -> Dict[str, int]:
    """record -> fold in 0..k-1, dealt round-robin after a seeded shuffle within each quality group, so
    every fold holds strong and weak records in the pool's proportion."""
    folds: Dict[str, int] = {}
    for group in (True, False):
        ids = sorted(r for r in record_ids if strong[r] == group)
        stable_rng(seed, "cross_marker_folds", group).shuffle(ids)
        for i, rid in enumerate(ids):
            folds[rid] = i % k
    return folds


def cross_fitted(contrasts: torch.Tensor, record_ids: Sequence[str],
                 folds: Mapping[str, int]) -> Dict[int, torch.Tensor]:
    """fold -> unit direction fitted on every record outside the fold."""
    fold_of = torch.tensor([folds[r] for r in record_ids])
    out: Dict[int, torch.Tensor] = {}
    for f in sorted(set(folds[r] for r in record_ids)):
        rest = fold_of != f
        if not bool(rest.any()):
            raise ValueError(f"fold {f} holds every record; cross-fitting needs at least two folds")
        out[f] = unit(contrasts[rest].mean(0))
    return out


def split_half_cosine(contrasts: torch.Tensor, seed: int, repeats: int = 20) -> float:
    """Mean cosine between directions fitted on two random halves of the records: the reliability of an
    n/2-record fit. For the n-record direction step it up with `full_sample_reliability`; a cosine between two
    directions is read against `cosine_ceiling`, not against this. NaN below 4 records."""
    n = contrasts.shape[0]
    if n < 4:
        return float("nan")
    rng = stable_rng(seed, "split_half")
    values = []
    for _ in range(repeats):
        order = list(range(n))
        rng.shuffle(order)
        half = n // 2
        values.append(cosine(contrasts[order[:half]].mean(0), contrasts[order[half:]].mean(0)))
    return float(sum(values) / len(values))


def full_sample_reliability(split_half: float) -> float:
    """The reliability of a direction fitted on all n records from the split-half cosine ρ of n/2-record
    halves: 2ρ / (1 + ρ) (Spearman–Brown). With contrast = signal s + noise of total variance σ² per record,
    a half-sample fit has cos(half, half') ≈ |s|² / (|s|² + 2σ²/n) = ρ, and two full-sample fits
    |s|² / (|s|² + σ²/n) = 2ρ / (1 + ρ). NaN for ρ ≤ 0 or NaN (no measurable signal)."""
    if not split_half > 0:
        return float("nan")
    return 2 * split_half / (1 + split_half)


def cosine_ceiling(rel_a: float, rel_b: float) -> float:
    """The largest cosine two directions of full-sample reliabilities ``rel_a`` and ``rel_b`` can show:
    √(rel_a · rel_b), reached when their signals coincide (each fit's cosine with its own signal is √rel, and
    independent noise is orthogonal in high dimension). NaN if either reliability is NaN or ≤ 0."""
    if not (rel_a > 0 and rel_b > 0):
        return float("nan")
    return (rel_a * rel_b) ** 0.5


def head_alignment(w: torch.Tensor, u: torch.Tensor) -> Dict[str, float]:
    """How much the score head reads a direction: ``w_dot_u`` (reward per unit of state along u) and the
    scale-free ``cos_w_u``."""
    if _degenerate(u):
        return {"w_dot_u": float("nan"), "cos_w_u": float("nan")}
    w = w.float().reshape(-1)
    u = unit(u.float())
    return {"w_dot_u": float(w @ u), "cos_w_u": float(w @ u / (w.norm() + 1e-8))}


def did_share(w: torch.Tensor, delta_int: torch.Tensor, u: torch.Tensor) -> float:
    """Share of the mean decision disparity w·Δ_int carried along u: (w·u)(Δ_int·u) / (w·Δ_int) — the
    fraction projecting out u would remove (in-sample, at α = 1). NaN when the disparity is ~0; unstable
    when it is small. For u = the interaction direction fitted on the same records the share is 1 by
    construction — read its cross-fitted nulling instead."""
    if _degenerate(u):
        return float("nan")
    w = w.float().reshape(-1)
    u = unit(u.float())
    total = float(w @ delta_int.float())
    if abs(total) < 1e-12:
        return float("nan")
    return float((w @ u) * (delta_int.float() @ u)) / total
