"""Pole orientation of the factorial pairs, for every design and encoding: side A of every pair is the
pole-A cell (the hypothesised penalised level: `pairs/factorial.py`), side B the pole-B cell.

Every disparity downstream is A − B, so a swap here would flip every sign while the Tier-1 gate and the
other factorial tests still pass (each clause would still sit in its own text). Added 2026-09-26 in the
pairs/ review (review flag: pole orientation end to end).
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from pairs.factorial import DESIGNS, ProxyNames, factorial_pairs


def _render(record, template_id, marker=""):
    """A stand-in renderer: the document is fixed text around the one marker slot."""
    return f"[{record.source_record_id}|{template_id}]{marker} [end]"


@pytest.mark.parametrize("domain", sorted(DESIGNS))
@pytest.mark.parametrize("encoding", ["explicit", "proxy"])
def test_side_a_is_the_pole_a_cell(domain, encoding):
    design = DESIGNS[domain]
    poles_a = tuple(levels[0] for levels in design.factors.values())
    poles_b = tuple(levels[1] for levels in design.factors.values())
    record = SimpleNamespace(source_record_id="rec-1")
    for seed in range(10):
        pairs, texts, exemplar = factorial_pairs(record, "t1", encoding, _render, random.Random(seed),
                                                 design=design)
        names = ProxyNames.from_exemplar(exemplar) if encoding == "proxy" else None
        cell_of = {text: cell for cell, text in texts.items()}
        assert len(cell_of) == 8  # the 8 cell texts are distinct, so a text identifies its cell
        assert pairs
        for p in pairs:
            a, b = cell_of[p.text_a], cell_of[p.text_b]
            assert p.clause_a == design.clause(a, encoding, names)
            assert p.clause_b == design.clause(b, encoding, names)
            if p.axis == "intersection":
                assert (a, b) == (poles_a, poles_b)
            else:
                i = design.axes.index(p.axis)
                assert (a[i], b[i]) == design.factors[p.axis]          # A is pole A of the varied axis
                assert [k for k in range(3) if a[k] != b[k]] == [i]    # and nothing else varies
                assert p.intersectional_cell[p.axis] == f"{a[i]}-vs-{b[i]}"
            assert (p.label_a, p.label_b) == design.labels(p.axis, encoding, a, b)
