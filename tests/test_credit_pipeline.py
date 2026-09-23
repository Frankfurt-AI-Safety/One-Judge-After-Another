"""
Unit tests for the demographic credit data-creation pipeline.

Cover the offline stages (ingest → render → inject → validate → loader → metric) without any model:
- renderer determinism + demographic-neutrality of the baseline,
- marker injection yields a structural single-axis diff,
- Tier-1 gate accepts clean pairs and rejects content drift,
- the CreditDemographicDataset loader yields ContrastivePair/EvalExample,
- auto-influence metric arithmetic.
"""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock

import pytest

from substrates.credit_ingest import GermanCreditRecord, load_german_credit
from substrates.credit_render import render_profile, TEMPLATES
from pairs.markers import make_pair
from pairs.validate import validate_pair, validate_pairs, Thresholds


def _fake_record(rid="german-test") -> GermanCreditRecord:
    return GermanCreditRecord(
        source_record_id=rid, checking="no checking account", duration_months=12,
        credit_history="existing credits paid back duly until now", purpose="a new car",
        credit_amount_dm=2000, savings="less than 100 EUR", employment_since="1 to under 4 years",
        installment_rate="20% to under 25%", property="real estate", other_installment_plans="none",
        housing="owned", existing_credits="1", job="skilled employee or official",
        dependents="0 to 2", telephone=True, foreign_worker=False, credit_good=True,
        raw_sex="female", raw_marital="single", raw_age_years=29,
    )


# --------------------------------------------------------------------------- ingest
def test_ingest_loads_and_decodes():
    recs = load_german_credit()  # uses the downloaded raw file
    assert len(recs) == 1000
    r0 = recs[0]
    assert r0.raw_sex in ("male", "female", None)
    assert isinstance(r0.credit_amount_dm, int) and r0.credit_amount_dm > 0
    assert isinstance(r0.credit_good, bool)


# Grömping (2019), Table 1: per-level % among (bad, good) credits, keyed by the Statlog code the level
# maps to. A18 (people liable) is keyed with its levels already swapped back to the Statlog coding.
_TABLE1 = {
    "checking": ("CHECKING", {"A11": (45.00, 19.86), "A12": (35.00, 23.43),
                              "A13": (4.67, 7.00), "A14": (15.33, 49.71)}),
    "credit_history": ("CREDIT_HISTORY", {"A30": (8.33, 2.14), "A31": (9.33, 3.00),
                                          "A32": (56.33, 51.57), "A33": (9.33, 8.57),
                                          "A34": (16.67, 34.71)}),
    "purpose": ("PURPOSE", {"A40": (29.67, 20.71), "A41": (5.67, 12.29), "A42": (19.33, 17.57),
                            "A43": (20.67, 31.14), "A44": (1.33, 1.14), "A45": (2.67, 2.00),
                            "A46": (7.33, 4.00), "A47": (0.00, 0.00), "A48": (0.33, 1.14),
                            "A49": (11.33, 9.00), "A410": (1.67, 1.00)}),
    "savings": ("SAVINGS", {"A61": (72.33, 55.14), "A62": (11.33, 9.86), "A63": (3.67, 7.43),
                            "A64": (2.00, 6.00), "A65": (10.67, 21.57)}),
    "employment_since": ("EMPLOYMENT", {"A71": (7.67, 5.57), "A72": (23.33, 14.57),
                                        "A73": (34.67, 33.57), "A74": (13.00, 19.29),
                                        "A75": (21.33, 27.00)}),
    "installment_rate": ("INSTALLMENT_RATE", {"1": (11.33, 14.57), "2": (20.67, 24.14),
                                              "3": (15.00, 16.00), "4": (53.00, 45.29)}),
    "property": ("PROPERTY", {"A121": (20.00, 31.71), "A122": (23.67, 23.00),
                              "A123": (34.00, 32.86), "A124": (22.33, 12.43)}),
    "other_installment_plans": ("OTHER_PLANS", {"A141": (19.00, 11.71), "A142": (6.33, 4.00),
                                                "A143": (74.67, 84.29)}),
    "housing": ("HOUSING", {"A151": (23.33, 15.57), "A152": (62.00, 75.43),
                            "A153": (14.67, 9.00)}),
    "existing_credits": ("NUMBER_CREDITS", {"1": (66.67, 61.86), "2": (30.67, 34.43),
                                            "3": (2.00, 3.14), "4": (0.67, 0.57)}),
    "job": ("JOB", {"A171": (2.33, 2.14), "A172": (18.67, 20.57), "A173": (62.00, 63.43),
                    "A174": (17.00, 13.86)}),
    "dependents": ("PEOPLE_LIABLE", {"1": (84.67, 84.43), "2": (15.33, 15.57)}),
}
_N_BAD, _N_GOOD = 300, 700


@pytest.fixture(scope="module")
def records():
    from substrates.credit_ingest import DEFAULT_RAW_PATH

    if not DEFAULT_RAW_PATH.exists():
        pytest.skip(f"German Credit raw file not downloaded ({DEFAULT_RAW_PATH})")
    return load_german_credit()


class TestCodebook:
    """Regression against the wrong UCI `german.doc` mapping: the decoded labels must reproduce the
    per-class distribution Grömping (2019) published for the correctly coded data."""

    @staticmethod
    def _counts(records, key):
        from collections import Counter

        return (Counter(key(r) for r in records if not r.credit_good),
                Counter(key(r) for r in records if r.credit_good))

    @pytest.mark.parametrize("attr", sorted(_TABLE1))
    def test_decoded_distribution_matches_table1(self, records, attr):
        import substrates.credit_ingest as ci

        map_name, table = _TABLE1[attr]
        mapping = getattr(ci, map_name)
        bad, good = self._counts(records, lambda r: getattr(r, attr))
        for code, (pct_bad, pct_good) in table.items():
            label = mapping[code]
            # tolerance 1: Grömping reports one A15 (housing) mismatch against the LMU data
            assert abs(bad[label] - round(pct_bad * _N_BAD / 100)) <= 1, (attr, code, label)
            assert abs(good[label] - round(pct_good * _N_GOOD / 100)) <= 1, (attr, code, label)

    def test_personal_status_sex_matches_table1(self, records):
        bad, good = self._counts(records, lambda r: (r.raw_sex, r.raw_marital))
        expected = {("male", "divorced/separated"): (6.67, 4.29),
                    (None, "female non-single or male single"): (36.33, 28.71),
                    ("male", "married/widowed"): (48.67, 57.43),
                    ("female", "single"): (8.33, 9.57)}
        for key, (pct_bad, pct_good) in expected.items():
            assert abs(bad[key] - round(pct_bad * _N_BAD / 100)) <= 1, key
            assert abs(good[key] - round(pct_good * _N_GOOD / 100)) <= 1, key

    def test_binary_fields_match_table1(self, records):
        # foreign worker: "yes" is the rare level (1.33% bad, 4.71% good); telephone "no" 62.33/58.43
        bad, good = self._counts(records, lambda r: r.foreign_worker)
        assert (bad[True], good[True]) == (round(1.33 * 3), round(4.71 * 7))
        bad, good = self._counts(records, lambda r: r.telephone)
        assert (bad[False], good[False]) == (round(62.33 * 3), round(58.43 * 7))

    def test_single_row_decodes(self, tmp_path):
        # Needs no download: one hand-written row, every categorical at a non-default code.
        row = ("A11 6 A30 A40 1169 A61 A71 1 A92 A101 4 A121 67 A141 A151 4 A171 2 A191 A202 2")
        path = tmp_path / "german.data"
        path.write_text(row + "\n")
        (r,) = load_german_credit(path)
        assert r.checking == "no checking account"
        assert r.credit_history == "delays in paying off in the past"
        assert r.purpose == "other purposes"
        assert r.savings == "unknown or no savings account"
        assert r.employment_since == "none (unemployed)"
        assert r.installment_rate == "35% or more"
        assert r.property == "unknown or none"
        assert r.other_installment_plans == "with other banks"
        assert r.housing == "provided for free"
        assert r.existing_credits == "6 or more"
        assert r.job == "unemployed or unskilled"
        assert r.dependents == "3 or more"
        assert (r.telephone, r.foreign_worker, r.credit_good) == (False, True, False)
        assert (r.raw_sex, r.raw_age_years) == (None, 67)

    def test_no_residency_cue_in_job_labels(self):
        from substrates.credit_ingest import JOB

        assert not any("resident" in v for v in JOB.values())


# --------------------------------------------------------------------------- render
class TestRender:
    def test_marker_without_leading_space_raises(self):
        with pytest.raises(ValueError):
            render_profile(_fake_record(), "credit_v1", marker="The applicant is a woman.")

    def test_neutral_profile_has_no_demographics(self):
        import re

        r = _fake_record()  # raw_sex=female, raw_age_years=29
        text = render_profile(r, "credit_v1")
        low = text.lower()
        # the source age must not appear
        assert str(r.raw_age_years) not in text
        # explicit sex words / pronouns must not appear (word-boundary to avoid e.g. "management")
        for word in ["woman", "women", "female", "male", "she", "he"]:
            assert re.search(rf"\b{word}\b", low) is None, f"neutral profile leaked {word!r}"

    def test_render_is_deterministic(self):
        r = _fake_record()
        assert render_profile(r, "credit_v1") == render_profile(r, "credit_v1")

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            render_profile(_fake_record(), "nope")

    def test_every_real_record_renders_neutrally(self):
        import re

        from substrates.credit_ingest import DEFAULT_RAW_PATH

        if not DEFAULT_RAW_PATH.exists():
            pytest.skip(f"German Credit raw file not downloaded ({DEFAULT_RAW_PATH})")
        for r in load_german_credit():
            for tid in TEMPLATES:
                low = render_profile(r, tid).lower()
                assert "{" not in low and "}" not in low
                assert re.search(r"\b(?:resident|non-resident|foreign|woman|man|male|female)\b",
                                 low) is None, (r.source_record_id, tid)

    def test_marker_inserted_at_slot(self):
        r = _fake_record()
        marked = render_profile(r, "credit_v1", marker=" The applicant is a woman.")
        assert "The applicant is a woman." in marked


# --------------------------------------------------------------------------- markers + gate
class TestMarkersAndGate:
    @pytest.mark.parametrize("axis", ["sex", "age", "family_status", "intersection"])
    @pytest.mark.parametrize("enc", ["explicit", "proxy"])
    def test_pair_is_single_axis_and_passes_gate(self, axis, enc):
        pair = make_pair(_fake_record(), "credit_v1", axis, enc, random.Random(0))
        # stripping each clause yields identical remainders → single-axis
        assert pair.text_a.replace(pair.clause_a, "", 1) == pair.text_b.replace(pair.clause_b, "", 1)
        res = validate_pair(pair)
        assert res.ok, f"{axis}/{enc} failed gate: {res.reasons}"

    def test_intersection_cell_records_all_axes(self):
        pair = make_pair(_fake_record(), "credit_v1", "intersection", "explicit", random.Random(0))
        assert pair.intersectional_cell["sex"] == "female-vs-male"
        assert pair.intersectional_cell["age"] == "30-vs-50"
        assert pair.intersectional_cell["family_status"] == "parental_leave-vs-continuous"
        assert pair.held_fixed == ["financial_content", "template"]
        assert "woman" in pair.text_a and "30-year-old" in pair.text_a
        assert "man" in pair.text_b and "50-year-old" in pair.text_b

    def test_real_field_clause(self):
        import dataclasses
        from pairs.markers import real_field_clause

        base = _fake_record()

        def clause(sex, marital):
            return real_field_clause(dataclasses.replace(base, raw_sex=sex, raw_marital=marital))

        assert clause("male", "married/widowed") == " The applicant is a married man."
        assert clause("male", "divorced/separated") == " The applicant is a divorced man."
        assert clause("female", "single") == " The applicant is a single woman."
        # A92 pools non-single women with single men → sex unknown, must not guess
        with pytest.raises(ValueError):
            clause(None, "female non-single or male single")
        # the old (wrong) codebook's categories no longer exist
        with pytest.raises(ValueError):
            clause("male", "single")

    def test_gate_rejects_content_drift(self):
        pair = make_pair(_fake_record(), "credit_v1", "sex", "explicit", random.Random(0))
        # corrupt a financial fact on one side only → not single-axis anymore
        pair.text_b = pair.text_b.replace("2000 EUR", "9999 EUR")
        res = validate_pair(pair)
        assert not res.ok
        assert any("single-axis" in r or "non-marker" in r for r in res.reasons)

    def test_gate_rejects_length_blowup(self):
        pair = make_pair(_fake_record(), "credit_v1", "sex", "explicit", random.Random(0))
        pair.text_a = pair.text_a + " " + ("padding " * 50)
        res = validate_pair(pair, Thresholds(max_char_delta=12, max_token_delta=3))
        assert not res.ok

    def test_validate_pairs_report(self):
        recs = [_fake_record(f"r{i}") for i in range(5)]
        pairs = [make_pair(r, "credit_v1", "sex", "explicit", random.Random(0)) for r in recs]
        passed, failed, report = validate_pairs(pairs)
        assert report["n_passed"] == 5 and report["n_failed"] == 0


# --------------------------------------------------------------------------- loader
class TestLoader:
    def _write_jsonl(self, tmp_path):
        from pairs.markers import make_pair as mp
        from pairs.manifest import pair_to_record

        recs = [_fake_record(f"german-{i:04d}") for i in range(40)]
        rows = []
        for i, r in enumerate(recs):
            p = mp(r, "credit_v1", "sex", "explicit", random.Random(i))
            rows.append(pair_to_record(p, f"credit-sex-explicit-credit_v1-{r.source_record_id}",
                                       role="probe", seed=42))
        path = tmp_path / "pairs.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in rows))
        return path

    def _fake_tokenizer(self):
        tok = MagicMock()
        tok.chat_template = None  # → format_conversation uses pair format (returns a tuple)
        return tok

    def test_loader_yields_pairs_and_evals(self, tmp_path):
        from scoring.pair_dataset import CreditDemographicDataset

        ds = CreditDemographicDataset(str(self._write_jsonl(tmp_path)), axis="sex",
                                      encoding="explicit", probe_size=20, split_seed=42)
        tok = self._fake_tokenizer()
        probe_pairs = ds.get_probe_pairs(tok)
        evals = ds.get_eval_examples(tok)
        assert len(probe_pairs) > 0 and len(evals) > 0
        assert len(probe_pairs) + len(evals) == 40
        cp = probe_pairs[0]
        assert cp.positive_text and cp.negative_text
        assert set(evals[0].texts.keys()) == {"a", "b"}

    def test_loader_filters_axis(self, tmp_path):
        from scoring.pair_dataset import CreditDemographicDataset

        ds = CreditDemographicDataset(str(self._write_jsonl(tmp_path)), axis="age",
                                      encoding="explicit", probe_size=20)
        with pytest.raises(ValueError):  # no age rows in this manifest
            ds.get_probe_pairs(self._fake_tokenizer())


# --------------------------------------------------------------------------- metric
class TestAutoInfluence:
    def test_metric_arithmetic(self):
        from scoring.demographic_experiment import compute_auto_influence_metrics

        rewards = {"a": [1.0, 2.0, 0.0, 5.0], "b": [0.0, 1.0, 1.0, 1.0]}
        m = compute_auto_influence_metrics(rewards)
        assert m["n_examples"] == 4
        assert m["pref_a_rate"] == pytest.approx(0.75)        # a>b for 3 of 4
        assert m["mean_gap"] == pytest.approx((1 + 1 - 1 + 4) / 4)
        assert m["auto_influence"] == pytest.approx(abs(0.75 - 0.5) * 2)

    def test_metric_handles_none_and_empty(self):
        from scoring.demographic_experiment import compute_auto_influence_metrics

        assert compute_auto_influence_metrics({"a": [], "b": []})["n_examples"] == 0
        m = compute_auto_influence_metrics({"a": [1.0, None], "b": [0.0, 2.0]})
        assert m["n_examples"] == 1

    def test_cross_influence_metric(self):
        from scoring.demographic_experiment import compute_cross_influence

        scores = {
            "strong_neutral":   [5.0, 5.0, 5.0, 5.0],
            "weak_neutral":     [1.0, 1.0, 1.0, 1.0],   # strong>weak always → acc_baseline=1.0
            "weak_protected":   [6.0, 6.0, 1.0, 1.0],   # marker flips 2/4 → acc=0.5
            "weak_reference":   [1.0, 1.0, 1.0, 1.0],   # unchanged → acc=1.0
            "strong_protected": [5.0, 5.0, 5.0, 5.0],
        }
        m = compute_cross_influence(scores)
        assert m["acc_baseline"] == pytest.approx(1.0)
        assert m["acc_weak_protected"] == pytest.approx(0.5)
        assert m["cross_influence"] == pytest.approx(0.5)         # reliability harm
        assert m["cross_influence_reference"] == pytest.approx(0.0)
        assert m["protected_vs_reference"] == pytest.approx(0.5)  # protected-specific harm
        assert m["baseline_tracks_quality"] is True

    def test_protected_vs_reference_ignores_a_generic_marker_effect(self):
        # An RM that reacts to ANY marker clause on the weak text (length, or "some demographic
        # statement") in the same way: cross_influence shows harm, the protected-vs-reference contrast
        # correctly shows none.
        from scoring.demographic_experiment import compute_cross_influence

        m = compute_cross_influence({
            "strong_neutral": [5.0, 5.0, 5.0, 5.0],
            "weak_neutral":   [1.0, 1.0, 1.0, 1.0],
            "weak_protected": [6.0, 6.0, 1.0, 1.0],
            "weak_reference": [6.0, 6.0, 1.0, 1.0],
        })
        assert m["cross_influence"] == pytest.approx(0.5)
        assert m["cross_influence_reference"] == pytest.approx(0.5)
        assert m["protected_vs_reference"] == pytest.approx(0.0)
        assert m["protected_vs_reference"] == pytest.approx(
            m["cross_influence"] - m["cross_influence_reference"])

    def test_cross_influence_flags_non_tracking(self):
        from scoring.demographic_experiment import compute_cross_influence

        # RM doesn't prefer the stronger (acc_baseline≈0.5) → flag non-interpretable
        m = compute_cross_influence({
            "strong_neutral": [1.0, 0.0, 1.0, 0.0],
            "weak_neutral":   [0.0, 1.0, 0.0, 1.0],
            "weak_protected": [0.0, 1.0, 0.0, 1.0],
        })
        assert m["acc_baseline"] == pytest.approx(0.5)
        assert m["baseline_tracks_quality"] is False

    def test_per_template_subgroup_split(self):
        from scoring.dataset_base import EvalExample
        from runners.run_battery import _subgroup_auto_influence

        evs = [EvalExample(texts={}, metadata={"template_id": t})
               for t in ["credit_v1", "credit_v2", "credit_v1", "credit_v2"]]
        # v1 examples (idx 0,2): a>b both → AI=1.0; v2 examples (idx 1,3): a<b both → AI=1.0
        base_org = {"a": [2.0, 0.0, 3.0, 0.0], "b": [1.0, 5.0, 1.0, 9.0]}
        out = _subgroup_auto_influence(base_org, evs, key="template_id")
        assert set(out) == {"credit_v1", "credit_v2"}
        assert out["credit_v1"] == pytest.approx(1.0)  # both a>b
        assert out["credit_v2"] == pytest.approx(1.0)  # both a<b (pref_a_rate=0 → AI=1)
