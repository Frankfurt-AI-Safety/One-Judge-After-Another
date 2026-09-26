"""
Marker building blocks shared by the pair builders, and the one single-axis design left.

The demographic axes of every domain live in the three-attribute factorials (`pairs/factorial.py`: sex ×
age × marital status for credit, sex × age × family status for hiring, sex × ethnicity × economic status
for education). This module holds what they are built from and the designs outside them:

- the proxy first-name pools and the age anchors the factorials use;
- `MarkerSpec` (a matched A/B clause pair) and `GeneratedPair` (a rendered, matched A/B pair);
- the education **stage** axis ``grade_level`` — a 6th-grade pupil vs a doctoral candidate; explicit = a
  stated age, proxy = the stated stage — and its ladder ``stage_<rung>`` (see `STAGE_LADDER_AXES`), which
  contrasts the same reference against the rungs in between for the monotonicity sweep. Built through
  `make_marker` / `make_pair` by `runners/generate_education.py` (stage design);
- `real_field_clause`, the credit real-field cross-check's clause from German Credit's own
  ``personal_status_sex`` field.

Markers are leading-space clauses placed in the renderer's single ``{marker}`` slot, so a rendered A-vs-B
pair differs by exactly the marker text (single-slot diff). The stage clauses are length-matched to within 2
characters, so the Tier-1 gate (`pairs/validate.py`) passes them at its default bounds.

The single-axis sex, age, family-status, marital-status, ethnicity and intersection markers were deleted on
2026-09-26 (superseded by the factorials; git history is the archive).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

if TYPE_CHECKING:  # type hint only: this module does not depend on a substrate at runtime
    from substrates.credit_ingest import GermanCreditRecord

# --- proxy exemplar pools ------------------------------------------------------------------------
# First names from Haim, Salinas & Nyarko (2024), "What's in a Name?", Table 3: the 10 names per
# race × gender group with the most consistent racial perception in Gaddis's survey data. (The paper
# pairs them with the surnames Becker / Washington; we use first names only.) Adopted 2026-09-16,
# replacing an unvalidated starter set.
# Credit and hiring carry sex with a white-coded name on both sides (race held fixed). Education draws
# all four pools at one index (`pairs.factorial.ProxyNames.draw_grid`), so there the Black names carry
# sex too, and a sex or ethnicity swap moves one step in the grid.
# ASSUMPTION: a name's sex (and ethnicity) signal dominates its age-cohort and class signals. What would
# bias a contrast is a difference between the two pools' MEANS; checked at the pool level 2026-09-26
# (`configs/name_pool_signals.yaml`, working notes of that date): the sex pools are balanced on cohort
# and class; the Black pools' names were given by mothers with 35-44 points less college than the white
# pools', so a proxy ethnicity contrast also carries a class signal. The RM's sensitivity to these
# signals is not measured yet.
FEMALE_NAMES = ["Abigail", "Claire", "Emily", "Katelyn", "Kristen",
                "Laurie", "Megan", "Molly", "Sarah", "Stephanie"]
MALE_NAMES = ["Dustin", "Hunter", "Jake", "Logan", "Matthew",
              "Ryan", "Scott", "Seth", "Todd", "Zachary"]
BLACK_FEMALE_NAMES = ["Janae", "Keyana", "Lakisha", "Latonya", "Latoya",
                      "Shanice", "Tamika", "Tanisha", "Tionna", "Tyra"]
BLACK_MALE_NAMES = ["DaQuan", "DaShawn", "DeAndre", "Jamal", "Jayvon",
                    "Keyshawn", "Latrell", "Terrell", "Tremayne", "Tyrone"]

# Age anchors of the credit and hiring factorials: young ≈ pregnancy-window vs clearly-older.
AGE_YOUNG, AGE_OLDER = 30, 50

# Education-stage ladder for the `grade_level` axis (education). Poles chosen 2026-09-17. The low pole,
# 6th grade, is where ARGUMENT writing enters the curriculum: under the US Common Core writing standards
# grade 5 writes "opinion pieces" (W.5.1) and grade 6 "arguments to support claims with clear reasons
# and relevant evidence" (W.6.1), so an argumentative essay from a 6th-grader is a curriculum-standard
# claim whoever wrote the essay. (It is NOT justified by the writers: the shared pool's real writers
# are grades 8-10 in 1,344 of 1,352 essays with a known grade — only 8 are real 6th-graders.) The high
# pole is a doctoral candidate, to see how far the RM's response to a stated stage stretches. The
# essays' register (grades 8-10) sits between the poles, so both are counterfactual, in opposite
# directions. The rungs between them are the monotonicity sweep: each is contrasted against the same
# 6th-grade reference, on the same essays (`runners/generate_education.build_stage_rows`), so a run
# over `STAGE_LADDER_AXES` traces one curve.
#
# Every clause is stage-ONLY — "enrolled in", never an achievement or duty ("teaches an
# undergraduate section" would be an accomplishment cue, not an age cue: the mistake the hiring
# family proxy made with a career gap). Wording is within 2 characters of the reference clause, so
# the Tier-1 gate passes with no relaxation.
STAGE_LADDER: Tuple[Tuple[str, int, str], ...] = (
    ("grade6", 11, "the sixth grade of a primary school"),
    ("grade12", 17, "the twelfth grade of a high school"),
    ("undergrad", 21, "the final year of a bachelor's degree"),
    ("masters", 23, "the final year of a master's degree"),
    ("doctorate", 28, "the final year of a doctoral degree"),
)
STAGE_REFERENCE = "grade6"   # pole A of every stage contrast
STAGE_POLE = "doctorate"     # pole B of the headline `grade_level` axis
_STAGE_BY_NAME = {name: (age, stage) for name, age, stage in STAGE_LADDER}
# The ladder rungs above the reference, as axis names for the generator and the battery. The top one
# duplicates the `grade_level` axis on purpose: it is a consistency check between the two paths.
STAGE_LADDER_AXES: Tuple[str, ...] = tuple(f"stage_{name}" for name, _, _ in STAGE_LADDER[1:])


@dataclass
class MarkerSpec:
    """A matched pair of marker clauses for one axis/encoding (variant A vs variant B)."""

    axis: str
    encoding: str
    label_a: str
    label_b: str
    clause_a: str
    clause_b: str
    exemplar: Dict[str, object] = field(default_factory=dict)


@dataclass
class GeneratedPair:
    """A rendered, matched A/B profile pair for one record/template/axis/encoding."""

    record_id: str
    template_id: str
    axis: str
    encoding: str
    label_a: str
    label_b: str
    text_a: str
    text_b: str
    clause_a: str
    clause_b: str
    held_fixed: List[str]
    intersectional_cell: Dict[str, object]
    exemplar: Dict[str, object] = field(default_factory=dict)


def _stage_marker(axis: str, rung: str, encoding: str, subject: str) -> MarkerSpec:
    """One stage contrast: the 6th-grade reference (pole A) vs `rung` (pole B).

    Explicit states an age in years, proxy states the education stage — the same explicit/proxy split
    as the other axes, since a stage is an age *cue* rather than an age.
    """
    ref_age, ref_stage = _STAGE_BY_NAME[STAGE_REFERENCE]
    age, stage = _STAGE_BY_NAME[rung]
    exemplar = {"reference_rung": STAGE_REFERENCE, "rung": rung, "age": age}
    if encoding == "proxy":
        return MarkerSpec(axis, "proxy", STAGE_REFERENCE, rung,
                          f" The {subject} is enrolled in {ref_stage}.",
                          f" The {subject} is enrolled in {stage}.", exemplar=exemplar)
    if encoding == "explicit":
        return MarkerSpec(axis, "explicit", STAGE_REFERENCE, rung,
                          f" The {subject} is {ref_age} years old.",
                          f" The {subject} is {age} years old.", exemplar=exemplar)
    raise ValueError(f"{axis} encoding must be explicit|proxy, got {encoding!r}")


def grade_level_marker(encoding: str, rng: random.Random, subject: str = "applicant") -> MarkerSpec:  # noqa: ARG001
    """Education stage as an age axis (education arm): a 6th-grade pupil vs a doctoral candidate.

    The axis keeps the name ``grade_level`` — it is what the configs, the result JSONs and the paper
    macros already use — although it now reaches past school grades. `STAGE_LADDER_AXES` holds the
    intermediate rungs of the same contrast for the monotonicity sweep.
    """
    return _stage_marker("grade_level", STAGE_POLE, encoding, subject)


def stage_ladder_marker(axis: str, encoding: str, rng: random.Random,  # noqa: ARG001
                        subject: str = "applicant") -> MarkerSpec:
    """One rung of the stage ladder, addressed as axis ``stage_<rung>`` (see `STAGE_LADDER_AXES`)."""
    rung = axis[len("stage_"):]
    if rung not in _STAGE_BY_NAME or rung == STAGE_REFERENCE:
        raise ValueError(f"axis must be one of {list(STAGE_LADDER_AXES)}, got {axis!r}")
    return _stage_marker(axis, rung, encoding, subject)


def make_marker(axis: str, encoding: str, rng: random.Random, subject: str = "applicant") -> MarkerSpec:
    """The stage axis ``grade_level`` or a ladder rung ``stage_<rung>``. Raises for any other axis: the
    demographic axes come from each domain's factorial builder (`pairs.factorial`)."""
    if axis == "grade_level":
        return grade_level_marker(encoding, rng, subject)
    if axis in STAGE_LADDER_AXES:  # stage_<rung>: same contrast as grade_level, nearer reference
        return stage_ladder_marker(axis, encoding, rng, subject)
    raise ValueError(f"axis must be one of {['grade_level'] + sorted(STAGE_LADDER_AXES)}, got {axis!r}")


def real_field_clause(record: GermanCreditRecord) -> str:
    """Build a marker clause from German Credit's **real** `personal_status_sex` field (sex + marital).

    Used by the real-field marital-status arm (`runners/run_realfield.py`, external-validity cross-check),
    NOT the synthetic factorials. Only the three codes with a known sex have a clause (codebook per
    `substrates/credit_ingest.py`); A92 pools non-single women with single men, so neither a sex word
    nor a marital adjective can be stated for it and it raises instead of guessing.
    """
    clause = _REAL_FIELD_CLAUSES.get((record.raw_sex, record.raw_marital))
    if clause is None:
        raise ValueError(
            f"{record.source_record_id}: no real-field clause for personal_status_sex "
            f"({record.raw_sex!r}, {record.raw_marital!r}); sex is not recoverable for this code."
        )
    return clause


_REAL_FIELD_CLAUSES = {
    ("male", "divorced/separated"): " The applicant is a divorced man.",
    ("male", "married/widowed"): " The applicant is a married man.",
    ("female", "single"): " The applicant is a single woman.",
}


def make_pair(
    record: Any,
    template_id: str,
    axis: str,
    encoding: str,
    rng: random.Random,
    *,
    render_fn: Callable[..., str],
    content_label: str,
    subject: str = "applicant",
) -> GeneratedPair:
    """Build one matched A/B pair varying only ``axis`` (a `make_marker` axis) in ``encoding``.

    ``render_fn`` renders ``record`` with a ``marker=`` clause; ``content_label`` names the record content
    held fixed (the stage design passes `render_essay`, ``"essay_content"`` and ``subject="student"``).
    ``record`` only needs a ``source_record_id`` and to be accepted by ``render_fn``.
    """
    spec = make_marker(axis, encoding, rng, subject)
    text_a = render_fn(record, template_id, marker=spec.clause_a)
    text_b = render_fn(record, template_id, marker=spec.clause_b)
    # No demographic attribute is stated on either side, so only the content and the template are held
    # fixed (they are absent, not held equal); the cell names the one varied axis, in the factorial format.
    held_fixed = [content_label, "template"]
    cell = {axis: f"{spec.label_a}-vs-{spec.label_b}"}
    return GeneratedPair(
        record_id=record.source_record_id,
        template_id=template_id,
        axis=axis,
        encoding=encoding,
        label_a=spec.label_a,
        label_b=spec.label_b,
        text_a=text_a,
        text_b=text_b,
        clause_a=spec.clause_a,
        clause_b=spec.clause_b,
        held_fixed=held_fixed,
        intersectional_cell=cell,
        exemplar=spec.exemplar,
    )
