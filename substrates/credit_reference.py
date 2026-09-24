"""
A common-sense credit scorer — the reference line for decision accuracy on the credit domain.

German Credit's label is repayment among loans the bank *granted*, and several rendered fields run
against lending intuition in this population: real estate and owned housing go with worse outcomes,
managers do worse than unskilled workers, "no checking account" is the worst checking level (audit
2026-09-23). A reward model that judges an application the way a sensible lender would can therefore
score well below what a model fitted to these labels reaches, and "low accuracy" alone does not mean it
ignores creditworthiness. This scorer puts a number on that: every rendered field scored in the
direction a lender would assume, fixed a priori (not fitted), equal weights. Its AUC against
``credit_good`` on the records a run evaluates is reported next to the RM's AUC of the decision margin.

Only fields the renderer shows are used (`substrates/credit_render.py`); purpose is left out (no
common-sense direction). Duration and amount are scored by their rank within the scored records
(shorter / smaller is better), which needs no label.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from substrates.credit_ingest import GermanCreditRecord

# Level -> score in [0, 1], higher = a sensible lender's better risk. Written from lending intuition;
# the good-credit rates of this dataset were deliberately not used to order them.
INTUITIVE_LEVELS: Dict[str, Dict[str, float]] = {
    "checking": {
        "balance below 0 EUR": 0.0,
        "no checking account": 1 / 3,
        "balance of 0 to 199 EUR": 2 / 3,
        "balance of 200 EUR or more, or salary paid in for at least 1 year": 1.0,
    },
    "savings": {
        "unknown or no savings account": 0.0,
        "less than 100 EUR": 0.25,
        "100 to 499 EUR": 0.5,
        "500 to 999 EUR": 0.75,
        "1000 EUR or more": 1.0,
    },
    "credit_history": {  # bad history vs duly paid; the three duly-paid levels are not ranked
        "critical account or other credits elsewhere": 0.0,
        "delays in paying off in the past": 0.0,
        "existing credits paid back duly until now": 1.0,
        "no credits taken or all credits paid back duly": 1.0,
        "all credits at this bank paid back duly": 1.0,
    },
    "employment_since": {
        "none (unemployed)": 0.0,
        "less than 1 year": 0.25,
        "1 to under 4 years": 0.5,
        "4 to under 7 years": 0.75,
        "7 years or more": 1.0,
    },
    "installment_rate": {  # share of disposable income: a lower burden is better
        "35% or more": 0.0,
        "25% to under 35%": 1 / 3,
        "20% to under 25%": 2 / 3,
        "less than 20%": 1.0,
    },
    "property": {
        "unknown or none": 0.0,
        "a car or other property": 1 / 3,
        "building-society savings or life insurance": 2 / 3,
        "real estate": 1.0,
    },
    "housing": {"rented": 0.5, "provided for free": 0.5, "owned": 1.0},
    "job": {
        "unemployed or unskilled": 0.0,
        "unskilled": 1 / 3,
        "skilled employee or official": 2 / 3,
        "manager, self-employed or highly qualified employee": 1.0,
    },
    "existing_credits": {"1": 1.0, "2 to 3": 2 / 3, "4 to 5": 1 / 3, "6 or more": 0.0},
    "other_installment_plans": {"none": 1.0, "with stores": 0.0, "with other banks": 0.0},
    "dependents": {"0 to 2": 1.0, "3 or more": 0.0},
}
_NUMERIC = {"duration_months": "shorter", "credit_amount_dm": "smaller"}


def _low_is_good_ranks(values: Sequence[float]) -> List[float]:
    """Score in [0, 1] from the rank among ``values``: the smallest scores 1; ties share their mean rank."""
    n = len(values)
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return [1.0 - r / (n - 1) for r in ranks]


def common_sense_scores(records: Sequence[GermanCreditRecord]) -> List[float]:
    """Equal-weight intuitive score per record (higher = better risk). Raises on an unknown level, so a
    codebook change cannot silently score a level as 0."""
    records = list(records)
    totals = [0.0] * len(records)
    for field, levels in INTUITIVE_LEVELS.items():
        for i, rec in enumerate(records):
            value = getattr(rec, field)
            if value not in levels:
                raise KeyError(f"{field}: no common-sense score for level {value!r}")
            totals[i] += levels[value]
    for field in _NUMERIC:
        for i, s in enumerate(_low_is_good_ranks([float(getattr(r, field)) for r in records])):
            totals[i] += s
    return totals
