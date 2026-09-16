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
from typing import Any, Callable, List, Tuple

from pairs.factorial import credit_marker
from pairs.markers import make_marker
from scoring.pair_dataset import ASSESSMENT_PROMPT, CreditDemographicDataset
from scoring.bios_dataset import BIOS_ASSESSMENT_PROMPT, BiosDemographicDataset
from scoring.education_dataset import EDU_ASSESSMENT_PROMPT, EducationDemographicDataset
from substrates.credit_clean import load_factorial_records
from substrates.bios_ingest import DEFAULT_BIOS_PATH, load_bias_in_bios
from substrates.education_ingest import DEFAULT_PERSUADE_PATH, load_persuade
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
    axes=("sex", "age", "marital_status", "intersection"),
    make_marker=credit_marker,
)

# Hiring arm. The substrate is REAL biographies (Bias-in-Bios), not the synthetic CV generator that
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
    # Real biographies are user-downloaded; raises with fetch instructions if absent.
    load_records=lambda: load_bias_in_bios(DEFAULT_BIOS_PATH),
    is_strong=lambda r: r.qualified,
    axes=("sex", "age", "family_status", "intersection"),
    make_marker=make_marker,
)

EDUCATION = DomainSpec(
    name="education",
    dataset_cls=EducationDemographicDataset,
    default_pairs="data/demographic/education/persuade/pairs.jsonl",
    render_fn=render_essay,
    assessment_prompt=EDU_ASSESSMENT_PROMPT,
    template_ids=tuple(sorted(EDU_TEMPLATES)),
    # Real essays are user-downloaded; load the PERSUADE corpus (raises with instructions if absent).
    load_records=lambda: load_persuade(DEFAULT_PERSUADE_PATH),
    is_strong=lambda r: r.high_quality,
    axes=("sex", "ethnicity", "grade_level"),
    make_marker=make_marker,
)

DOMAINS = {CREDIT.name: CREDIT, CV.name: CV, EDUCATION.name: EDUCATION}


def get_domain(name: str) -> DomainSpec:
    if name not in DOMAINS:
        raise ValueError(f"domain must be one of {sorted(DOMAINS)}, got {name!r}")
    return DOMAINS[name]
