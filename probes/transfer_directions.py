"""
Difference-of-means directions that can leave part of their fitting data out, for the transfer runner
(`runners/run_demographic_transfer.py`).

A direction is fitted on **contrast units**: one matched pair of one block, state(pole A) − state(pole B). The
direction is the unit-length mean of the units, which is the battery's direction (`probes.probe.build_probe_direction`:
mean(A) − mean(B) over the probe pairs). Each unit carries the folds of the first names in its two texts (none for an
explicit marker) and its document template, so a direction can be fitted without the names of the pair it will be
projected out of (``exclude_folds``) or without that pair's template (``exclude_template``).

The same class serves any vector space the units live in: the model's states, and the bag-of-words vectors of the
marker clauses (the lexical control).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Sequence, Tuple

import torch

from probes.cross_marker_directions import unit

Key = Tuple[FrozenSet[int], str]            # (the name folds of a pair, its template)


class NoFitUnits(ValueError):
    """Every contrast unit of a direction was excluded."""


class UnitStore:
    """The contrast units of one (domain, encoding, axis), summed per (name folds, template) group, so a direction
    without some folds or a template is a sum over the remaining groups. ``record_contrasts`` holds each record's
    mean unit (records in order of first appearance): what split-half reliabilities resample."""

    def __init__(self, diffs: torch.Tensor, folds: Sequence[FrozenSet[int]], templates: Sequence[str],
                 records: Sequence[str]):
        n = diffs.shape[0]
        if not (n == len(folds) == len(templates) == len(records)):
            raise ValueError(f"{n} units, {len(folds)} fold sets, {len(templates)} templates, {len(records)} records")
        if n == 0:
            raise NoFitUnits("no contrast units")
        diffs = diffs.float()
        group_at: Dict[Key, int] = {}
        owner = torch.tensor([group_at.setdefault((frozenset(f), t), len(group_at))
                              for f, t in zip(folds, templates)], dtype=torch.long)
        self.keys = list(group_at)
        self.sums = torch.zeros(len(group_at), diffs.shape[1]).index_add_(0, owner, diffs)
        self.counts = torch.bincount(owner, minlength=len(group_at))
        record_at: Dict[str, int] = {}
        rec = torch.tensor([record_at.setdefault(str(r), len(record_at)) for r in records], dtype=torch.long)
        per_record = torch.zeros(len(record_at), diffs.shape[1]).index_add_(0, rec, diffs)
        self.record_contrasts = per_record / torch.bincount(rec, minlength=len(record_at)).float()[:, None]
        self.n_units, self.n_records = n, len(record_at)
        # the length of the mean unit: 0 for a marker the tokenizer cannot see, whose direction is then the zero
        # vector (projecting it out is a no-op; its cosines are NaN)
        self.separation = float(diffs.mean(0).norm())
        self.templates = sorted({t for _, t in self.keys})
        self.named = any(f for f, _ in self.keys)
        self._fitted: Dict[Tuple[FrozenSet[int], Optional[str]], torch.Tensor] = {}

    def kept(self, exclude_folds: FrozenSet[int] = frozenset(), exclude_template: Optional[str] = None
             ) -> torch.Tensor:
        """Which groups a fit without ``exclude_folds`` and ``exclude_template`` keeps (a boolean per group)."""
        return torch.tensor([not (f & exclude_folds) and t != exclude_template for f, t in self.keys])

    def n_fit(self, exclude_folds: FrozenSet[int] = frozenset(), exclude_template: Optional[str] = None) -> int:
        """The units such a fit rests on."""
        return int(self.counts[self.kept(exclude_folds, exclude_template)].sum())

    def direction(self, exclude_folds: FrozenSet[int] = frozenset(), exclude_template: Optional[str] = None
                  ) -> torch.Tensor:
        """The unit direction fitted on every unit that shares no name fold with ``exclude_folds`` and is not of
        ``exclude_template``. Raises `NoFitUnits` when none is left."""
        key = (frozenset(exclude_folds), exclude_template)
        if key not in self._fitted:
            keep = self.kept(*key)
            n = int(self.counts[keep].sum())
            if n == 0:
                raise NoFitUnits(f"no contrast unit left without name folds {sorted(key[0])}"
                                 + (f" and template {exclude_template!r}" if exclude_template else ""))
            self._fitted[key] = unit(self.sums[keep].sum(0) / n)
        return self._fitted[key]


def mean_unit(directions: Sequence[torch.Tensor]) -> torch.Tensor:
    """The unit mean of several directions, each at unit length first: every one weighs the same."""
    return unit(torch.stack([unit(d.float()) for d in directions]).mean(0))


def random_units(dim: int, draws: int, seed: int) -> torch.Tensor:
    """``draws`` random unit vectors of R^dim (seeded), one per row."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(draws, dim, generator=g)
    return v / v.norm(dim=1, keepdim=True)
