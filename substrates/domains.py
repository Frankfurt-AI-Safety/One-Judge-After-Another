"""
Domain registry for the demographic-bias arms.

A single source of truth for everything that differs between the **credit**, **hiring (CV-screening)**
and **education (grading)** domains: the matched-pair dataset class + default manifest, the neutral
renderer + framing prompt + template ids, how to load the underlying records, and which records count
as the "stronger" applicant (the quality ground truth for cross-influence). All three now sit on real
substrates. The runners (`run_battery`, `run_crossinfluence`, the reasoning/decision-response arms, …)
and `DemographicBiasExperiment._create_dataset` all resolve a `DomainSpec` from
`cfg.extra["domain"]` (or a `--domain` flag) so adding a domain is one entry here.

Imports only the dataset classes from `scoring/` and the marker builders from `pairs/` (never
`scoring.demographic_experiment`, which imports this module) → no import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from pairs.factorial import (
    AXES as CREDIT_FACTORIAL_AXES, CREDIT_DESIGN, EDUCATION_DESIGN, FactorialDesign, HIRING_DESIGN,
    credit_marker, education_marker, hiring_marker,
)
from scoring.pair_dataset import ASSESSMENT_PROMPT, CreditDemographicDataset
from scoring.bios_dataset import BIOS_ASSESSMENT_PROMPT, BiosDemographicDataset
from scoring.education_dataset import EDU_ASSESSMENT_PROMPT, EducationDemographicDataset
from substrates.bios_clean import load_factorial_bios
from substrates.credit_clean import load_factorial_records
from substrates.bios_ingest import DEFAULT_BIOS_PATH
from substrates.education_clean import load_education_essays
from substrates.credit_render import TEMPLATES, render_profile
from substrates.bios_render import BIOS_TEMPLATES, render_bio
from substrates.education_render import EDU_TEMPLATES, render_essay


@dataclass(frozen=True)
class DomainSpec:
    """Everything a runner needs to operate one domain's direct-scoring arm."""

    name: str
    dataset_cls: type
    default_pairs: str
    render_fn: Callable[..., str]          # (record, template_id, marker="") -> str
    assessment_prompt: str
    template_ids: Tuple[str, ...]
    load_records: Callable[[], List[Any]]  # () -> records (each with .source_record_id)
    is_strong: Callable[[Any], bool]       # record -> True if the "stronger" applicant
    axes: Tuple[str, ...]                  # axes present in this domain's pairs manifest
    make_marker: Callable[..., Any]        # (axis, encoding, rng, subject) -> MarkerSpec
    factorial: Optional[FactorialDesign] = None  # set where pairs come from a 2x2x2 factorial
    # Cross-influence pairs a strong with a weak record only within the same stratum (None = anywhere).
    # Education: the prompt, so both essays answer the same assignment the header shows.
    pair_stratum: Optional[Callable[[Any], Any]] = None


# Credit: sex × age × marital status as a full factorial (pairs/factorial.py). `family_status`
# (parental leave / career gap) is not used here: it contradicts the profile's employment field.
CREDIT = DomainSpec(
    name="credit",
    dataset_cls=CreditDemographicDataset,
    default_pairs="data/demographic/credit/pairs.jsonl",
    render_fn=render_profile,
    assessment_prompt=ASSESSMENT_PROMPT,
    template_ids=tuple(sorted(TEMPLATES)),
    # Only records passing the consistency rules (substrates/credit_clean.py).
    load_records=load_factorial_records,
    is_strong=lambda r: r.credit_good,
    # Derived from the factorial so the two lists cannot drift apart.
    axes=CREDIT_FACTORIAL_AXES + ("intersection",),
    make_marker=credit_marker,
    factorial=CREDIT_DESIGN,
)

# Hiring: sex × age × family status as a full factorial (pairs/factorial.py), the pregnancy-window
# combination. The substrate is REAL biographies (Bias-in-Bios), not the synthetic CV generator that
# used to back this domain (removed; pre-2026-08 results were produced on it). `qualified` here means
# role-match (profession == target_role), so it is a genuine quality axis rather than an invented
# heuristic.
CV = DomainSpec(
    name="cv",
    dataset_cls=BiosDemographicDataset,
    default_pairs="data/demographic/cv/pairs.jsonl",
    render_fn=render_bio,
    assessment_prompt=BIOS_ASSESSMENT_PROMPT,
    template_ids=tuple(sorted(BIOS_TEMPLATES)),
    # Real biographies are user-downloaded; raises with fetch instructions if absent. Only bios that
    # pass the scrub and the factorial plausibility rules (substrates/bios_clean.py).
    load_records=lambda: load_factorial_bios(DEFAULT_BIOS_PATH),
    is_strong=lambda r: r.qualified,
    axes=HIRING_DESIGN.axes + ("intersection",),
    make_marker=hiring_marker,
    factorial=HIRING_DESIGN,
)

EDUCATION = DomainSpec(
    name="education",
    dataset_cls=EducationDemographicDataset,
    default_pairs="data/demographic/education/persuade/pairs.jsonl",
    render_fn=render_essay,
    assessment_prompt=EDU_ASSESSMENT_PROMPT,
    template_ids=tuple(sorted(EDU_TEMPLATES)),
    # Real essays are user-downloaded; load the PERSUADE corpus (raises with instructions if absent).
    # The shared education pool (substrates/education_clean.py): the same essays for the factorial, the
    # stage design, the A2 positioned arm and cross-influence, so their results are comparable.
    load_records=lambda: load_education_essays(source="persuade"),
    is_strong=lambda r: r.high_quality,
    # sex × ethnicity × economic status as a full factorial. The **stage** axis (`grade_level`) and its
    # monotonicity ladder are a separate single-axis design on a separate manifest (same essays;
    # `runners/generate_education.py --design stage`), so they are not in the default battery: run them
    # with --dataset-source <stage dir>/pairs.jsonl --axes grade_level.
    axes=EDUCATION_DESIGN.axes + ("intersection",),
    make_marker=education_marker,
    factorial=EDUCATION_DESIGN,
    pair_stratum=lambda r: r.prompt_id,
)

DOMAINS = {CREDIT.name: CREDIT, CV.name: CV, EDUCATION.name: EDUCATION}


def get_domain(name: str) -> DomainSpec:
    if name not in DOMAINS:
        raise ValueError(f"domain must be one of {sorted(DOMAINS)}, got {name!r}")
    return DOMAINS[name]
