"""
Sex × age × marital-status full factorial for the credit arm.

Every record is rendered in all 2×2×2 = 8 cells with ONE composite clause in the renderer's
``{marker}`` slot, e.g. ``" The applicant is a 30-year-old married woman."``. Matched pairs are then
cut from the cells:

- **single-axis pairs** — two cells that differ in exactly one attribute, for each of the 4 settings of
  the other two. The other attributes are *stated and equal* on both sides (not absent), and the
  analysis can average over, or stratify by, their levels (recorded in ``intersectional_cell``).
- **intersection pair** — the two corner cells (all three attributes flipped), used by the
  additivity test: intersection ≈ sex + age + marital_status if the effects are additive.

Pole A of every axis is the hypothesised penalised level (female, 30, married), so the intersection
contrast is exactly the sum of the three marginal contrasts in orientation.

Encodings:
- ``explicit`` — ``" The applicant is a {age}-year-old {marital} {woman|man}."``
- ``proxy`` — sex via a first name, age via a birth year:
  ``" The applicant, {name}, was born in {year} and is {marital}."``. Marital status has no clean
  proxy on a credit profile (a spouse or joint application would change the financial content), so it
  stays explicit here, and **no marital_status single-axis pairs are emitted for proxy** — they would
  be explicit pairs under a proxy label. The proxy intersection pair still flips it (recorded in
  ``exemplar["marital_encoding"]``).

Within one record/template/encoding the proxy names are drawn once and held across all cells, so an
age or marital pair never changes the name. The sex pair does change it, and first names carry some
age-cohort signal, so the proxy sex contrast is not perfectly age-neutral.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from pairs.markers import AGE_OLDER, AGE_YOUNG, FEMALE_NAMES, MALE_NAMES, GeneratedPair, MarkerSpec

# Pole A first (the hypothesised penalised level), pole B second.
FACTORS: Dict[str, Tuple[object, object]] = {
    "sex": ("female", "male"),
    "age": (AGE_YOUNG, AGE_OLDER),
    "marital_status": ("married", "single"),
}
AXES: Tuple[str, ...] = tuple(FACTORS)
ENCODINGS = ("explicit", "proxy")
# Axes that carry a genuine proxy; the others are explicit even inside a proxy clause.
PROXY_AXES = ("sex", "age")
# Birth years are computed from a fixed reference year so they stay reproducible.
REFERENCE_YEAR = 2026

Cell = Tuple[str, int, str]  # (sex, age, marital_status)
CELLS: Tuple[Cell, ...] = tuple(itertools.product(*FACTORS.values()))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProxyNames:
    female: str
    male: str

    @classmethod
    def draw(cls, rng: random.Random) -> "ProxyNames":
        return cls(female=rng.choice(FEMALE_NAMES), male=rng.choice(MALE_NAMES))


def factorial_clause(cell: Cell, encoding: str, names: Optional[ProxyNames] = None,
                     subject: str = "applicant") -> str:
    """The composite marker clause for one cell (leading space, as the renderers expect)."""
    sex, age, marital = cell
    if encoding == "explicit":
        noun = "woman" if sex == "female" else "man"
        return f" The {subject} is a {age}-year-old {marital} {noun}."
    if encoding == "proxy":
        if names is None:
            raise ValueError("proxy encoding needs ProxyNames")
        name = names.female if sex == "female" else names.male
        return f" The {subject}, {name}, was born in {REFERENCE_YEAR - age} and is {marital}."
    raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")


def cell_label(cell: Cell) -> Dict[str, object]:
    return dict(zip(AXES, cell))


def axis_pairs(axis: str, encoding: str) -> List[Tuple[Cell, Cell]]:
    """(pole-A cell, pole-B cell) for every setting of the other two factors; for ``intersection``,
    the single corner pair."""
    if axis == "intersection":
        a = tuple(levels[0] for levels in FACTORS.values())
        b = tuple(levels[1] for levels in FACTORS.values())
        return [(a, b)]  # type: ignore[list-item]
    if axis not in FACTORS:
        raise ValueError(f"axis must be one of {AXES + ('intersection',)}, got {axis!r}")
    if encoding == "proxy" and axis not in PROXY_AXES:
        return []  # no proxy exists for this axis; see module docstring
    i = AXES.index(axis)
    out = []
    for cell in CELLS:
        if cell[i] == FACTORS[axis][0]:
            flipped = list(cell)
            flipped[i] = FACTORS[axis][1]
            out.append((cell, tuple(flipped)))
    return out  # type: ignore[return-value]


def _pair_cell_meta(axis: str, a: Cell, b: Cell) -> Dict[str, object]:
    """``intersectional_cell`` for a pair: the varied axis as ``A-vs-B``, the others at their level."""
    meta: Dict[str, object] = {}
    for i, name in enumerate(AXES):
        meta[name] = f"{a[i]}-vs-{b[i]}" if a[i] != b[i] else a[i]
    return meta


def render_cells(record, template_id: str, encoding: str, render_fn: Callable[..., str],
                 names: Optional[ProxyNames] = None, subject: str = "applicant") -> Dict[Cell, str]:
    return {cell: render_fn(record, template_id, marker=factorial_clause(cell, encoding, names, subject))
            for cell in CELLS}


def factorial_pairs(
    record,
    template_id: str,
    encoding: str,
    render_fn: Callable[..., str],
    rng: random.Random,
    axes: Tuple[str, ...] = AXES + ("intersection",),
    content_label: str = "financial_content",
    subject: str = "applicant",
) -> Tuple[List[GeneratedPair], Dict[Cell, str], Dict[str, object]]:
    """All matched pairs for one record/template/encoding.

    Returns ``(pairs, cell_texts, exemplar)``. ``rng`` is only consumed for the proxy names.
    """
    names = ProxyNames.draw(rng) if encoding == "proxy" else None
    texts = render_cells(record, template_id, encoding, render_fn, names, subject)
    exemplar: Dict[str, object] = {"design": "factorial_2x2x2"}
    if names is not None:
        exemplar.update(female_name=names.female, male_name=names.male, marital_encoding="explicit")
    pairs: List[GeneratedPair] = []
    for axis in axes:
        for a, b in axis_pairs(axis, encoding):
            varied = [n for i, n in enumerate(AXES) if a[i] != b[i]]
            held = [n for n in AXES if n not in varied]
            pairs.append(GeneratedPair(
                record_id=record.source_record_id,
                template_id=template_id,
                axis=axis,
                encoding=encoding,
                label_a="intersectional" if axis == "intersection" else str(a[AXES.index(axis)]),
                label_b="reference" if axis == "intersection" else str(b[AXES.index(axis)]),
                text_a=texts[a],
                text_b=texts[b],
                clause_a=factorial_clause(a, encoding, names, subject),
                clause_b=factorial_clause(b, encoding, names, subject),
                held_fixed=held + [content_label, "template"],
                intersectional_cell=_pair_cell_meta(axis, a, b),
                exemplar=dict(exemplar),
            ))
    return pairs, texts, exemplar


def credit_marker(axis: str, encoding: str, rng: random.Random, subject: str = "applicant") -> MarkerSpec:
    """A factorial marker pair for runners that inject clauses on the fly (e.g. cross-influence).

    Draws one of the axis's pole pairs at random (and proxy names), so the injected clause has the
    same form as the pairs the probe direction was built from.
    """
    options = axis_pairs(axis, encoding)
    if not options:
        raise ValueError(f"no {encoding} marker for axis {axis!r} in the credit factorial")
    names = ProxyNames.draw(rng) if encoding == "proxy" else None
    a, b = rng.choice(options)
    exemplar: Dict[str, object] = {"cell": _pair_cell_meta(axis, a, b)}
    if names is not None:
        exemplar.update(female_name=names.female, male_name=names.male)
    return MarkerSpec(
        axis, encoding,
        "intersectional" if axis == "intersection" else str(a[AXES.index(axis)]),
        "reference" if axis == "intersection" else str(b[AXES.index(axis)]),
        factorial_clause(a, encoding, names, subject),
        factorial_clause(b, encoding, names, subject),
        exemplar=exemplar,
    )
