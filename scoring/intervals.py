"""
Cluster-bootstrap intervals: the direct arm (run_battery), the blatant decision arm, A2, additivity, the erasure
recoverability test (`probes/erasure.py`) and the cross-marker metrics (`scoring/cross_marker_metrics.py`).

The unit of resampling is the **cluster** whose items are correlated: a record's matched pairs share its
content (a factorial record gives 8 pairs per single axis), an essay's positioned pairs share the essay. A
percentile bootstrap over clusters, each replicate keeping all of a drawn cluster's items, gives the
interval of the pooled statistic; the point estimate is the statistic on the full data, so it equals what
the arm has always reported. Same convention as `scoring.cross_marker_metrics.summarize`: 95% percentile
interval, deterministic in ``seed``.

An interval says how precise an estimate is. It becomes a significance claim only inside the pre-registered
headline family, with its multiplicity correction; until then every interval here is uncorrected.

**Reading rule for the matched-pair metrics.** ``auto_influence`` (= |pref_a_rate − ½|·2) and ``abs_mean_gap``
(= mean |gap|) are *folded*: zero only for an exactly zero effect, positive under noise alone, so their intervals
cannot show that no preference is left. On 200 simulated no-effect data sets (2026-09-28) the 95% interval
excluded "no effect" for 25% (auto-influence) and 100% (abs_mean_gap) of them, against 6% for the signed
statistics. So: whether a preference is left (e.g. after nulling) is read from the **signed** intervals —
``mean_gap`` covering 0, ``pref_a_rate`` covering ½; ``abs_mean_gap`` is a size that includes a noise floor; how
much nulling removed is read from the change intervals (``mean_gap_change``, ``abs_mean_gap_change``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, List, Mapping, Sequence, Tuple

import numpy as np

DEFAULT_N_BOOT = 2000

Stat = Callable[[List[Any]], float]


def clusters_of(items: Sequence[Any], keys: Sequence[Hashable]) -> List[List[Any]]:
    """Group ``items`` by their cluster key (one key per item), in first-seen order."""
    order: Dict[Hashable, List[Any]] = {}
    for item, key in zip(items, keys):
        order.setdefault(key, []).append(item)
    return list(order.values())


def cluster_bootstrap(clusters: Sequence[Sequence[Any]], stats: Mapping[str, Stat],
                      n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Each statistic on all items (``estimate``) and its 95% percentile interval over ``n_boot``
    resamples of whole clusters. Every statistic is computed on the same replicates.

    Every entry has ``estimate``, ``ci_low``, ``ci_high``, ``n_clusters``, ``n_items`` and ``n_boot_valid`` (the
    replicates on which the statistic was defined; the interval is taken over those, so fewer than ``n_boot``
    means it is conditional on the statistic being defined). With fewer than two clusters a resample carries no
    information about the spread, and the interval is NaN."""
    clusters = [list(c) for c in clusters if len(c)]
    k = len(clusters)
    everything = [x for c in clusters for x in c]
    nan = float("nan")
    if k < 2:
        return {name: {"estimate": float(fn(everything)) if k else nan, "ci_low": nan, "ci_high": nan,
                       "n_clusters": k, "n_items": len(everything), "n_boot_valid": 0}
                for name, fn in stats.items()}
    draws = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    boot: Dict[str, List[float]] = {name: [] for name in stats}
    for row in draws:
        sample = [x for j in row for x in clusters[j]]
        for name, fn in stats.items():
            boot[name].append(fn(sample))
    out: Dict[str, Dict[str, float]] = {}
    for name, fn in stats.items():
        values = np.asarray(boot[name], dtype=float)
        values = values[~np.isnan(values)]
        out[name] = {"estimate": float(fn(everything)),
                     "ci_low": float(np.percentile(values, 2.5)) if values.size else nan,
                     "ci_high": float(np.percentile(values, 97.5)) if values.size else nan,
                     "n_clusters": k, "n_items": len(everything), "n_boot_valid": int(values.size)}
    return out


# --------------------------------------------------------------------------- matched pairs (A vs B) ---
def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")


def win(x: float, y: float) -> float:
    """1 if x > y, ½ on an exact tie, 0 otherwise — the Mann–Whitney convention, as in
    `scoring.cross_marker_metrics`. Rewards come out in the model's dtype (bf16, ~3 significant digits), so
    exact ties occur (7% of nulled credit sex pairs on Qwen3-0.6B, 2026-09-28); a strict ``>`` would count
    every tie against A."""
    return 1.0 if x > y else 0.5 if x == y else 0.0


def _pref_rate(s: Sequence[Tuple[float, float]]) -> float:
    return _mean([win(a, b) for a, b in s])


# The one definition of the matched-pair metrics: `compute_auto_influence_metrics` computes its point estimates
# with these, so an interval's estimate is always the reported value. auto_influence and abs_mean_gap are folded:
# read a remaining preference from mean_gap / pref_a_rate (module docstring).
PAIR_STATS: Dict[str, Stat] = {
    # items are (reward_a, reward_b)
    "mean_gap": lambda s: _mean([a - b for a, b in s]),
    "abs_mean_gap": lambda s: _mean([abs(a - b) for a, b in s]),
    "pref_a_rate": _pref_rate,
    "auto_influence": lambda s: abs(_pref_rate(s) - 0.5) * 2.0,
}

CHANGE_STATS: Dict[str, Stat] = {
    # items are (gap_baseline, gap_nulled) of one pair; nulled minus baseline, so negative = reduced
    "mean_gap_change": lambda s: _mean([n - b for b, n in s]),
    "abs_mean_gap_change": lambda s: _mean([abs(n) for _, n in s]) - _mean([abs(b) for b, _ in s]),
    "auto_influence_change": lambda s: (abs(_mean([win(n, 0.0) for _, n in s]) - 0.5)
                                        - abs(_mean([win(b, 0.0) for b, _ in s]) - 0.5)) * 2.0,
}


def pair_intervals(a: Sequence[float], b: Sequence[float], keys: Sequence[Hashable],
                   n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Intervals of the matched-pair metrics (A − B), clustered by ``keys`` (the pair's record)."""
    items = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    keys = [k for x, y, k in zip(a, b, keys) if x is not None and y is not None]
    return cluster_bootstrap(clusters_of(items, keys), PAIR_STATS, n_boot, seed)


def change_intervals(base: Tuple[Sequence[float], Sequence[float]], nulled: Tuple[Sequence[float], Sequence[float]],
                     keys: Sequence[Hashable], n_boot: int = DEFAULT_N_BOOT,
                     seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Intervals of what nulling changed (nulled − baseline) on the same pairs, clustered by ``keys`` —
    the paired comparison RQ4 needs, which two separate intervals cannot give."""
    (ba, bb), (na, nb) = base, nulled
    items, kept = [], []
    for x, y, u, v, k in zip(ba, bb, na, nb, keys):
        if None not in (x, y, u, v):
            items.append((x - y, u - v))
            kept.append(k)
    return cluster_bootstrap(clusters_of(items, kept), CHANGE_STATS, n_boot, seed)
