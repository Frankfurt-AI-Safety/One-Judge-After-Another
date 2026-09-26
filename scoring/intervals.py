"""
Cluster-bootstrap intervals for the arms that reported point estimates only (direct scoring, the blatant
decision arm, A2, additivity).

The unit of resampling is the **cluster** whose items are correlated: a record's matched pairs share its
content (a factorial record gives 8 pairs per single axis), an essay's positioned pairs share the essay. A
percentile bootstrap over clusters, each replicate keeping all of a drawn cluster's items, gives the
interval of the pooled statistic; the point estimate is the statistic on the full data, so it equals what
the arm has always reported. Same convention as `scoring.cross_marker_metrics.summarize`: 95% percentile
interval, deterministic in ``seed``.

An interval says how precise an estimate is. It becomes a significance claim only inside the pre-registered
headline family, with its multiplicity correction; until then every interval here is uncorrected.
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
    resamples of whole clusters. Every statistic is computed on the same replicates."""
    clusters = [list(c) for c in clusters if len(c)]
    k = len(clusters)
    everything = [x for c in clusters for x in c]
    out: Dict[str, Dict[str, float]] = {}
    if k == 0:
        return {name: {"n_clusters": 0, "n_items": 0} for name in stats}
    draws = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    boot: Dict[str, List[float]] = {name: [] for name in stats}
    for row in draws:
        sample = [x for j in row for x in clusters[j]]
        for name, fn in stats.items():
            boot[name].append(fn(sample))
    for name, fn in stats.items():
        values = np.asarray(boot[name], dtype=float)
        values = values[~np.isnan(values)]
        estimate = float(fn(everything))
        out[name] = {"estimate": estimate,
                     "ci_low": float(np.percentile(values, 2.5)) if values.size else float("nan"),
                     "ci_high": float(np.percentile(values, 97.5)) if values.size else float("nan"),
                     "n_clusters": k, "n_items": len(everything)}
    return out


# --------------------------------------------------------------------------- matched pairs (A vs B) ---
def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")


PAIR_STATS: Dict[str, Stat] = {
    # items are (reward_a, reward_b); the same statistics as compute_auto_influence_metrics
    "mean_gap": lambda s: _mean([a - b for a, b in s]),
    "abs_mean_gap": lambda s: _mean([abs(a - b) for a, b in s]),
    "pref_a_rate": lambda s: _mean([1.0 if a > b else 0.0 for a, b in s]),
    "auto_influence": lambda s: abs(_mean([1.0 if a > b else 0.0 for a, b in s]) - 0.5) * 2.0,
}

CHANGE_STATS: Dict[str, Stat] = {
    # items are (gap_baseline, gap_nulled) of one pair; nulled minus baseline, so negative = reduced
    "mean_gap_change": lambda s: _mean([n - b for b, n in s]),
    "abs_mean_gap_change": lambda s: _mean([abs(n) for _, n in s]) - _mean([abs(b) for b, _ in s]),
    "auto_influence_change": lambda s: (abs(_mean([1.0 if n > 0 else 0.0 for _, n in s]) - 0.5)
                                        - abs(_mean([1.0 if b > 0 else 0.0 for b, _ in s]) - 0.5)) * 2.0,
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
