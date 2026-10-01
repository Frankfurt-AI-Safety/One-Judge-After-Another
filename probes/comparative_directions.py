"""
The comparative design's own direction (`pairs/comparative.py`): per record pair and contrast axis, the mean
state difference between the same response under the two marker assignments, signed so that it points to "the
chosen applicant carries the protected cell". For the choice of side c in order o with response kind k:

    state(c protected, o, k, choose c) − state(other side protected, o, k, choose c)

averaged over c, o, k and the pair's templates. The prompt changes (the marker moves to the other applicant),
the response text does not, so the difference is what the marker assignment adds to the scored state; the head's
reading of it, w · contrast, is the pair's marker effect up to the pooling (linear heads). One row per pair: the
unit that cross-fitting (`probes.cross_marker_directions.cross_fitted`, folds of pairs) and split-half
reliabilities resample.

Binding caveat (RQ3, working notes 2026-09-26): the RM must bind the attribute to the right applicant; a single
linear direction may not capture that, so a failed projection here can be structural rather than "deeper" bias.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


def pair_contrasts(states: torch.Tensor, rows: Sequence[Mapping[str, Any]], axis: str
                   ) -> Tuple[List[str], torch.Tensor]:
    """The pairs (sorted) and their mean "chosen is protected" state contrast on ``axis`` ([pairs, d]); ``rows``
    are aligned with ``states``. Raises when the axis has no rows, a row occurs twice, or a row lacks its
    counterpart under the other assignment (each row is used exactly once, as one side of one difference)."""
    at: Dict[Tuple[str, str, str, str, str, str], int] = {}
    for i, r in enumerate(rows):
        if r["axis"] == axis:
            key = (r["pair_id"], r["template_id"], r["order"], r["kind"], r["protected"], r["chosen"])
            if key in at:
                raise ValueError(f"duplicate row {key}")
            at[key] = i
    if not at:
        raise ValueError(f"no rows for axis {axis!r}")
    pos: List[int] = []
    neg: List[int] = []
    owner: List[str] = []
    for (pid, template, order, kind, protected, chosen), i in at.items():
        if protected != chosen:
            continue
        other = "Y" if protected == "X" else "X"
        j = at.get((pid, template, order, kind, other, chosen))
        if j is None:
            raise ValueError(f"{pid}/{template}/{axis}: choice {chosen} lacks the swapped assignment")
        pos.append(i)
        neg.append(j)
        owner.append(pid)
    if 2 * len(pos) != len(at):
        unused = sorted(set(at.values()) - set(pos) - set(neg))
        raise ValueError(f"{axis}: {len(unused)} rows lack the swapped assignment, e.g. row {unused[0]}")
    ids = sorted(set(owner))
    index = {p: k for k, p in enumerate(ids)}
    own = torch.tensor([index[p] for p in owner], dtype=torch.long)
    diff = (states[pos] - states[neg]).float()
    sums = torch.zeros(len(ids), states.shape[1]).index_add_(0, own, diff)
    counts = torch.bincount(own, minlength=len(ids)).float()
    return ids, sums / counts[:, None]
