"""
Consistency filters for decoded German Credit records.

Two rule sets, both applied as filters (records are dropped, never repaired) and both counted, so the
discard report states exactly what was removed and why:

- :data:`RECORD_RULES` — the record contradicts itself once rendered (e.g. "Time with current employer:
  none (unemployed)" next to "Job: skilled employee or official"). Independent of any marker.
- :data:`FACTORIAL_RULES` — the record is fine on its own but cannot carry *every* level of the
  sex × age × marital-status factorial plausibly. A matched pair is only clean if both poles fit the
  profile; otherwise the reward model can punish the implausible pole instead of the attribute.

Each rule is ``(name, predicate)`` where the predicate returns True for a record that must be DROPPED.
Labels are compared through the codebook maps so a relabel in `credit_ingest` cannot silently disable
a rule.
"""

from __future__ import annotations

from typing import List, Tuple

from substrates.credit_ingest import (
    EMPLOYMENT, HOUSING, JOB, PEOPLE_LIABLE, PROPERTY, GermanCreditRecord, load_german_credit,
)
from substrates.rules import Rule, apply_rules  # noqa: F401  (apply_rules re-exported)

RECORD_RULES: Tuple[Rule, ...] = (
    # 45 records. Possibly self-employed applicants coded without an employer, but the codebook does
    # not say so, and as rendered the two fields contradict each other.
    ("unemployed_with_skilled_or_management_job",
     lambda r: r.employment_since == EMPLOYMENT["A71"] and r.job in (JOB["A173"], JOB["A174"])),
    # 4 records. Property records the most valuable asset, so owned housing implies real estate.
    ("owned_housing_without_real_estate",
     lambda r: r.housing == HOUSING["A153"] and r.property != PROPERTY["A124"]),
)

FACTORIAL_RULES: Tuple[Rule, ...] = (
    # The "single" level next to 3+ financially dependent people is an unusual combination while
    # "married" is not, so the marital pole would be asymmetric in plausibility. The factorial renders
    # every record as single too, so the whole record goes (149 records).
    # REPORTING NOTE: this shifts the population. Of the 149 records it drops, 130 are really
    # married/widowed men (A93), whose share falls from 54.5% to 48.4% of the records used. So the
    # credit results describe a sample with no large households and fewer married men than German
    # Credit as a whole. State this with every credit number; `cells.jsonl` keeps the real fields.
    ("single_level_with_3_or_more_dependents", lambda r: r.dependents == PEOPLE_LIABLE["2"]),
)


def load_factorial_records(path=None) -> List[GermanCreditRecord]:
    """German Credit records that pass both rule sets — the population every credit-arm runner
    injects factorial markers into."""
    records = load_german_credit(path) if path is not None else load_german_credit()
    clean, _ = apply_rules(records, RECORD_RULES)
    eligible, _ = apply_rules(clean, FACTORIAL_RULES)
    return eligible
