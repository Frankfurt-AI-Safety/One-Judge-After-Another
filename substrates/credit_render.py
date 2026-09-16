"""
Render a decoded :class:`GermanCreditRecord` into a demographically-neutral natural-language credit
profile via slot templates.

Design choices that make single-axis control trivial downstream:
- **One dedicated marker slot** (`{marker}`) at a fixed position. The neutral baseline passes
  `marker=""`; demographic markers (Phase 2) are injected *only* here, so a marked-vs-marked pair
  differs by exactly the marker text (structural single-diff, easy length parity).
- **Pronoun-free body** — the body refers to "the applicant" / "their", never "he/she". This means a
  proxy marker (a gendered first name in the clause) is the *only* sex signal; no pronoun agreement
  leaks the attribute across the rest of the text.
- **No names/institutions/hobbies** from the source — proxy-leakage control.
- Multiple templates (`template_id`) support the robustness battery.
"""

from __future__ import annotations

from typing import Dict

from substrates.credit_ingest import GermanCreditRecord

# Template bodies. `{marker}` is the (possibly empty) demographic clause; all other slots are
# financial. Keep wording pronoun-free. Add templates here for the robustness battery.
# Apart from {amount}, {duration} and {purpose}, every slot sits after a "Label: " prefix: the decoded
# values are codebook categories and bands ("none (unemployed)", "2 to 3", "less than 20%"), which do
# not read grammatically when spliced into running prose.
TEMPLATES: Dict[str, str] = {
    "credit_v1": (
        "Credit application summary.{marker} The applicant requests a loan of {amount} EUR "
        "over {duration} months for {purpose}. Checking account: {checking}. Savings: {savings}. "
        "Time with current employer: {employment}. Job: {job}. Credits at this bank, including "
        "this one: {existing_credits}. Credit history: {history}. Housing: {housing}. Most valuable "
        "property: {property}. Installments as a share of disposable income: {installment}. Other "
        "installment plans: {other_plans}. People financially dependent on the applicant: "
        "{dependents}."
    ),
    "credit_v2": (
        "Loan request under review.{marker} A loan of {amount} EUR is requested over a term of "
        "{duration} months, intended for {purpose}. Banking position: checking account: "
        "{checking}; savings: {savings}. Employment: time with current employer: {employment}; "
        "job category: {job}. Credit record: credits held at this bank, including this one: "
        "{existing_credits}; history: {history}. Living situation: housing: {housing}; most "
        "valuable property: {property}. Repayment burden: installments take {installment} of "
        "disposable income; other installment plans: {other_plans}; financially dependent people: "
        "{dependents}."
    ),
}


def _slots(record: GermanCreditRecord) -> Dict[str, object]:
    return {
        "amount": record.credit_amount_dm,
        "duration": record.duration_months,
        "purpose": record.purpose,
        "checking": record.checking,
        "savings": record.savings,
        "employment": record.employment_since,
        "job": record.job,
        "existing_credits": record.existing_credits,
        "history": record.credit_history,
        "housing": record.housing,
        "property": record.property,
        "installment": record.installment_rate,
        "other_plans": record.other_installment_plans,
        "dependents": record.dependents,
    }


def render_profile(
    record: GermanCreditRecord,
    template_id: str = "credit_v1",
    marker: str = "",
) -> str:
    """Render a neutral credit profile.

    Args:
        record: decoded German Credit record (financial slots used; sex/age/marital status,
            telephone and foreign-worker status NOT used).
        template_id: which template in :data:`TEMPLATES`.
        marker: demographic clause to inject at the dedicated slot. ``""`` → neutral baseline.
            A non-empty marker should be a leading-space clause, e.g. ``" The applicant is a woman."``.

    Returns:
        The rendered profile string.
    """
    if template_id not in TEMPLATES:
        raise KeyError(f"Unknown template_id {template_id!r}; known: {sorted(TEMPLATES)}")
    return TEMPLATES[template_id].format(marker=marker, **_slots(record))
