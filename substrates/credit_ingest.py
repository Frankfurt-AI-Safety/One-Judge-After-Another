"""
Ingest UCI German Credit (Statlog, CC-BY-4.0) and decode the A-coded fields into a typed record.

Source file: `data/demographic/credit/raw/german.data` — 1000 space-separated rows, 20 attributes
+ class.

**Codebook.** The mapping below is NOT the one in UCI's `german.doc`, which is wrong for most
categorical attributes. It follows the corrected code table of Grömping (2019), "South German Credit
Data: Correcting a Widely Used Data Set", Beuth Hochschule Report 04/2019, Table 1: the numeric part of
each A-code is the P2 level code there (A43 → level 3), except that A18 (people liable) and A20
(foreign worker) have their two levels swapped. The mapping was checked against the per-class
frequencies of `german.data` (see `tests/test_credit_pipeline.py::TestCodebook`). Amounts and the
checking/savings thresholds are DM in the source and are deliberately rendered as EUR, unconverted.

We deliberately separate the **financial slots** (used to render a demographically-neutral profile)
from the two protected/source fields — `personal_status_sex` (attr 9) and `age_years` (attr 13) —
which are preserved on the record (prefixed `raw_`) for the optional real-field arm but are NOT
rendered into the neutral baseline (demographics are injected synthetically downstream).
Sex is only recoverable for part of the data: A92 pools non-single women with single men, so its
`raw_sex` is ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# --- A-code → readable value maps (Grömping 2019, Table 1) ---------------------------------------
# Values are phrased to read after a "Label: " prefix in the templates (see credit_render.py).
CHECKING = {
    "A11": "no checking account",
    "A12": "balance below 0 EUR",
    "A13": "balance of 0 to 199 EUR",
    "A14": "balance of 200 EUR or more, or salary paid in for at least 1 year",
}
CREDIT_HISTORY = {
    "A30": "delays in paying off in the past",
    "A31": "critical account or other credits elsewhere",
    "A32": "no credits taken or all credits paid back duly",
    "A33": "existing credits paid back duly until now",
    "A34": "all credits at this bank paid back duly",
}
# A47 (education) and A48 (vacation) are listed for completeness; A47 never occurs in the data.
PURPOSE = {"A40": "other purposes", "A41": "a new car", "A42": "a used car",
           "A43": "furniture or equipment", "A44": "a radio or television",
           "A45": "domestic appliances", "A46": "repairs", "A47": "education",
           "A48": "a vacation", "A49": "retraining", "A410": "business"}
SAVINGS = {"A61": "unknown or no savings account", "A62": "less than 100 EUR",
           "A63": "100 to 499 EUR", "A64": "500 to 999 EUR", "A65": "1000 EUR or more"}
# Employment duration with the current employer; bounds are [lo, hi) in the codebook.
EMPLOYMENT = {"A71": "none (unemployed)", "A72": "less than 1 year", "A73": "1 to under 4 years",
              "A74": "4 to under 7 years", "A75": "7 years or more"}
# Most valuable property; the highest applicable code is recorded.
PROPERTY = {"A121": "unknown or none", "A122": "a car or other property",
            "A123": "building-society savings or life insurance", "A124": "real estate"}
OTHER_PLANS = {"A141": "with other banks", "A142": "with stores", "A143": "none"}
HOUSING = {"A151": "provided for free", "A152": "rented", "A153": "owned"}
# The codebook's A171/A172 carry "non-resident"/"resident", a nationality cue we keep out of the
# neutral profile.
JOB = {"A171": "unemployed or unskilled", "A172": "unskilled",
       "A173": "skilled employee or official",
       "A174": "manager, self-employed or highly qualified employee"}
# Attributes 8, 16 and 18 are stored as integers but are discretised bands, not counts or percents.
INSTALLMENT_RATE = {"1": "35% or more", "2": "25% to under 35%", "3": "20% to under 25%",
                    "4": "less than 20%"}
NUMBER_CREDITS = {"1": "1", "2": "2 to 3", "3": "4 to 5", "4": "6 or more"}
PEOPLE_LIABLE = {"1": "0 to 2", "2": "3 or more"}  # levels swapped relative to Grömping's P2
# protected/source fields (preserved, not rendered into the neutral baseline): code → (sex, marital)
PERSONAL_STATUS_SEX = {
    "A91": ("male", "divorced/separated"),
    "A92": (None, "female non-single or male single"),
    "A93": ("male", "married/widowed"),
    "A94": ("female", "single"),
}

_COLUMNS = [
    "checking", "duration_months", "credit_history", "purpose", "credit_amount_dm", "savings",
    "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age_years", "other_installment_plans", "housing",
    "existing_credits", "job", "people_liable", "telephone", "foreign_worker", "credit_class",
]


@dataclass
class GermanCreditRecord:
    """One decoded German Credit applicant. Financial slots feed the neutral renderer; the two
    protected fields (`raw_sex`, `raw_marital`, `raw_age_years`) are preserved for the real-field
    arm / quality signal but are not part of the demographically-neutral profile."""

    source_record_id: str
    # financial slots (rendered)
    checking: str
    duration_months: int
    credit_history: str
    purpose: str
    credit_amount_dm: int
    savings: str
    employment_since: str
    installment_rate: str   # band of disposable income, e.g. "less than 20%"
    property: str
    other_installment_plans: str
    housing: str
    existing_credits: str   # band of credits at this bank incl. this one, e.g. "2 to 3"
    job: str
    dependents: str         # band of people financially dependent: "0 to 2" | "3 or more"
    # not rendered
    telephone: bool
    foreign_worker: bool
    credit_good: bool  # class 1 = good credit (the ground-truth quality label)
    # protected / source fields (preserved, NOT rendered in the neutral baseline)
    raw_sex: Optional[str]  # None for A92, which pools non-single women with single men
    raw_marital: str
    raw_age_years: int
    extra: Dict[str, object] = field(default_factory=dict)


DEFAULT_RAW_PATH = Path("data/demographic/credit/raw/german.data")


def load_german_credit(path: Path | str = DEFAULT_RAW_PATH) -> List[GermanCreditRecord]:
    """Parse `german.data` into decoded :class:`GermanCreditRecord` objects (order preserved)."""
    path = Path(path)
    if not path.exists():
        # Create the directory before suggesting the download. `data/` is gitignored, so on a
        # fresh clone it does not exist and `curl -o` fails with "No such file or directory"
        # rather than downloading -- which reads as a broken instruction.
        path.parent.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            f"German Credit raw file not found at {path}. Download it first, e.g.:\n"
            f"  curl -L -o {path} \\\n"
            "    https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data\n"
            f"  (the directory {path.parent} has just been created for you)"
        )

    records: List[GermanCreditRecord] = []
    for i, line in enumerate(path.read_text().splitlines()):
        tok = line.split()
        if len(tok) != len(_COLUMNS):
            continue
        row = dict(zip(_COLUMNS, tok))
        sex, marital = PERSONAL_STATUS_SEX[row["personal_status_sex"]]
        records.append(
            GermanCreditRecord(
                source_record_id=f"german-{i:04d}",
                checking=CHECKING[row["checking"]],
                duration_months=int(row["duration_months"]),
                credit_history=CREDIT_HISTORY[row["credit_history"]],
                purpose=PURPOSE[row["purpose"]],
                credit_amount_dm=int(row["credit_amount_dm"]),
                savings=SAVINGS[row["savings"]],
                employment_since=EMPLOYMENT[row["employment_since"]],
                installment_rate=INSTALLMENT_RATE[row["installment_rate"]],
                property=PROPERTY[row["property"]],
                other_installment_plans=OTHER_PLANS[row["other_installment_plans"]],
                housing=HOUSING[row["housing"]],
                existing_credits=NUMBER_CREDITS[row["existing_credits"]],
                job=JOB[row["job"]],
                dependents=PEOPLE_LIABLE[row["people_liable"]],
                telephone=(row["telephone"] == "A192"),
                foreign_worker=(row["foreign_worker"] == "A202"),  # A201 = no (swapped, see above)
                credit_good=(row["credit_class"] == "1"),
                raw_sex=sex,
                raw_marital=marital,
                raw_age_years=int(row["age_years"]),
            )
        )
    if not records:
        raise ValueError(f"No valid German Credit rows parsed from {path}.")
    return records
