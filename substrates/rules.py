"""
Generic record filter shared by the per-domain consistency rules (`credit_clean.py`, `bios_clean.py`).

A rule is ``(name, predicate)`` where the predicate returns True for a record that must be DROPPED.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

Rule = Tuple[str, Callable[[Any], bool]]


def apply_rules(records: Sequence[Any], rules: Sequence[Rule]) -> Tuple[List[Any], Dict[str, object]]:
    """Drop every record that trips any rule. Order is preserved.

    Returns ``(kept, report)``; ``report["dropped_by_rule"]`` counts each rule a record trips, so a
    record tripping two rules is counted under both, while ``n_in - n_out`` is the exact number dropped.
    """
    kept: List[Any] = []
    by_rule: Dict[str, int] = {name: 0 for name, _ in rules}
    for rec in records:
        hits = [name for name, pred in rules if pred(rec)]
        for name in hits:
            by_rule[name] += 1
        if not hits:
            kept.append(rec)
    report = {"n_in": len(records), "n_out": len(kept), "dropped_by_rule": by_rule}
    return kept, report
