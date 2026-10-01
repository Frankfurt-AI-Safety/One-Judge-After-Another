"""
Statistics of the placement matrix (`runners/run_placement_matrix.py`; design 2026-10-01, working notes): how much
of one placement's demographic effect a direction found in another placement removes.

**The unit is the comparative record pair.** Every target effect is a ratio of sums over pairs, Σ per-pair sums /
Σ per-pair counts: the direct gap over the pair's two records (count 2), the cross-marker decision disparity over
its strong records (count 0–2: the cross-marker headline group), the comparative marker effect (count 1). One set
of bootstrap draws over the pairs (`scoring.cross_marker_metrics.bootstrap_draws`) serves every cell of an encoding,
so the difference between two cells has a paired interval.

Per cell (row direction R projected out, target placement T, axis a):

- ``baseline`` — T's effect on a, nothing projected out (the same in every row of T);
- ``nulled``   — the same with R's a-direction projected out (each fold by the direction fitted outside it);
- ``change``   — nulled − baseline: positive = nulling raised the effect;
- ``gap``      — change(T's own row) − change(R): what T's own direction changes beyond R. Exactly 0 on the
  diagonal. Signed like the changes: for a negative baseline, removing it is a positive change, and a positive gap
  means R removed less than the own direction; for a positive baseline the signs flip;
- ``shortfall`` — the gap oriented by the sign of the baseline's full-sample estimate, −sign(baseline) × gap (removing
  an effect moves it against its sign), the same sign in every replicate: **positive = R removed less than the own
  direction**, whatever the baseline's sign
  (user decision, 2026-10-01 review). NaN when the baseline estimate is 0; when the baseline's interval covers 0
  (``orientation.baseline_ci_covers_zero``) the orientation is a coin flip and the shortfall says little.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Iterable, Mapping, Sequence, Tuple

import numpy as np

from scoring.cross_marker_metrics import bootstrap_draws

UnitSums = Tuple[np.ndarray, np.ndarray]     # (per-unit sums, per-unit counts), in the order of the units


def unit_sums(values: Iterable[Tuple[Hashable, float]], units: Sequence[Hashable]) -> UnitSums:
    """Per unit (in ``units`` order) the sum and the count of the ``(unit, value)`` entries; a unit without entries
    has count 0. Raises on an entry whose unit is not in ``units`` and on a non-finite value."""
    at = {u: k for k, u in enumerate(units)}
    if len(at) != len(units):
        raise ValueError("duplicate units")
    sums = np.zeros(len(units))
    counts = np.zeros(len(units), dtype=np.int64)
    for unit, value in values:
        if unit not in at:
            raise KeyError(f"value of unknown unit {unit!r}")
        if not np.isfinite(value):
            raise ValueError(f"non-finite value {value!r} for unit {unit!r}")
        sums[at[unit]] += value
        counts[at[unit]] += 1
    return sums, counts


def ratio_summary(sums: np.ndarray, counts: np.ndarray, draws: np.ndarray) -> Dict[str, float]:
    """Σ sums / Σ counts and its 95% percentile interval over ``draws`` ([n_boot, n_units] unit indices). A
    replicate that draws no counted unit has no value and is left out (``n_boot_valid`` counts the rest); the
    interval is NaN with fewer than two counted units (every replicate would be the same)."""
    nan = float("nan")
    total = int(counts.sum())
    out: Dict[str, Any] = {"mean": float(sums.sum() / total) if total else nan, "n": total,
                           "n_units": int((counts > 0).sum())}
    bs, bc = sums[draws].sum(axis=1), counts[draws].sum(axis=1)
    ok = bc > 0
    out["n_boot_valid"] = int(ok.sum())
    if out["n_units"] < 2 or not ok.any():
        out["ci_low"] = out["ci_high"] = nan
    else:
        boot = bs[ok] / bc[ok]
        out["ci_low"], out["ci_high"] = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return out


def cell_table(baseline: UnitSums, nulled: Mapping[str, np.ndarray], own: str, draws: np.ndarray
               ) -> Dict[str, Any]:
    """One (target, axis): the baseline, its orientation, and per row direction in ``nulled`` (row → per-unit sums
    over the same items, so the counts are the baseline's) the nulled effect, the change, the gap to ``own`` (a key
    of ``nulled``: the target's own direction) and the oriented shortfall."""
    base, counts = baseline
    if own not in nulled:
        raise KeyError(f"the own row {own!r} is not among {sorted(nulled)}")
    for name, sums in nulled.items():
        if sums.shape != base.shape:
            raise ValueError(f"{name}: {sums.shape[0]} units, the baseline has {base.shape[0]}")
    baseline_summary = ratio_summary(base, counts, draws)
    sign = float(np.sign(baseline_summary["mean"])) if baseline_summary["mean"] == baseline_summary["mean"] else 0.0
    lo, hi = baseline_summary["ci_low"], baseline_summary["ci_high"]
    orientation = {"baseline_sign": sign,
                   "baseline_ci_covers_zero": bool(not (lo == lo and hi == hi) or lo <= 0 <= hi)}
    own_change = nulled[own] - base
    rows: Dict[str, Any] = {}
    for name, sums in nulled.items():
        change = sums - base
        gap = own_change - change
        rows[name] = {"nulled": ratio_summary(sums, counts, draws),
                      "change": ratio_summary(change, counts, draws),
                      "gap": ratio_summary(gap, counts, draws),
                      "shortfall": (ratio_summary(-sign * gap, counts, draws) if sign
                                    else {**ratio_summary(gap, counts, draws), "mean": float("nan"),
                                          "ci_low": float("nan"), "ci_high": float("nan")})}
    return {"own_row": own, "baseline": baseline_summary, "orientation": orientation, "rows": rows}


def shared_draws(n_units: int, n_boot: int, seed: int) -> np.ndarray:
    """The one set of bootstrap draws over the pairs that every cell of an encoding uses."""
    if n_units < 1:
        raise ValueError("no units to resample")
    return bootstrap_draws(seed, n_units, n_boot)
