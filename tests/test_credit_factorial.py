"""
Tests for the credit arm's sex × age × marital-status factorial: the consistency rules
(`substrates/credit_clean.py`), the cell/pair builder (`pairs/factorial.py`), the dataset generator
(`runners/generate_credit.py`) and the record-grouped probe/eval split.
"""

from __future__ import annotations

import dataclasses
import json
import random

import pytest

from pairs.factorial import (
    AXES, CELLS, FACTORS, ProxyNames, axis_pairs, credit_marker, factorial_clause, factorial_pairs,
)
from pairs.validate import validate_pair
from substrates.credit_clean import FACTORIAL_RULES, RECORD_RULES, apply_rules
from substrates.credit_ingest import EMPLOYMENT, HOUSING, JOB, PEOPLE_LIABLE, PROPERTY
from substrates.credit_render import render_profile
from tests.test_credit_pipeline import _fake_record


def _rec(rid="german-test", **changes):
    return dataclasses.replace(_fake_record(rid), **changes)


# --------------------------------------------------------------------------- rules
class TestRules:
    def test_record_rules_drop_contradictions(self):
        recs = [
            _rec("ok"),
            _rec("unemployed-manager", employment_since=EMPLOYMENT["A71"], job=JOB["A174"]),
            _rec("unemployed-skilled", employment_since=EMPLOYMENT["A71"], job=JOB["A173"]),
            _rec("unemployed-unskilled", employment_since=EMPLOYMENT["A71"], job=JOB["A172"]),
            _rec("owned-no-estate", housing=HOUSING["A153"], property=PROPERTY["A121"]),
            _rec("owned-estate", housing=HOUSING["A153"], property=PROPERTY["A124"]),
        ]
        kept, report = apply_rules(recs, RECORD_RULES)
        assert [r.source_record_id for r in kept] == ["ok", "unemployed-unskilled", "owned-estate"]
        assert report == {"n_in": 6, "n_out": 3, "dropped_by_rule": {
            "unemployed_with_skilled_or_management_job": 2, "owned_housing_without_real_estate": 1}}

    def test_factorial_rule_drops_3_or_more_dependents(self):
        recs = [_rec("few", dependents=PEOPLE_LIABLE["1"]), _rec("many", dependents=PEOPLE_LIABLE["2"])]
        kept, report = apply_rules(recs, FACTORIAL_RULES)
        assert [r.source_record_id for r in kept] == ["few"]
        assert report["dropped_by_rule"] == {"single_level_with_3_or_more_dependents": 1}

    def test_rule_counts_on_real_data(self):
        from substrates.credit_ingest import DEFAULT_RAW_PATH, load_german_credit

        if not DEFAULT_RAW_PATH.exists():
            pytest.skip(f"German Credit raw file not downloaded ({DEFAULT_RAW_PATH})")
        clean, rr = apply_rules(load_german_credit(), RECORD_RULES)
        _, fr = apply_rules(clean, FACTORIAL_RULES)
        assert rr["dropped_by_rule"] == {"unemployed_with_skilled_or_management_job": 45,
                                         "owned_housing_without_real_estate": 4}
        assert (rr["n_out"], fr["n_out"]) == (952, 803)


# --------------------------------------------------------------------------- cells + pairs
class TestFactorial:
    def test_eight_distinct_cells(self):
        assert len(CELLS) == len(set(CELLS)) == 8

    def test_pole_a_is_the_hypothesised_penalised_level(self):
        assert {a: levels[0] for a, levels in FACTORS.items()} == {
            "sex": "female", "age": 30, "marital_status": "married"}
        assert axis_pairs("intersection", "explicit") == [(("female", 30, "married"), ("male", 50, "single"))]

    @pytest.mark.parametrize("axis", AXES)
    def test_single_axis_pairs_flip_exactly_one_factor(self, axis):
        pairs = axis_pairs(axis, "explicit")
        assert len(pairs) == 4
        i = AXES.index(axis)
        for a, b in pairs:
            assert [k for k in range(3) if a[k] != b[k]] == [i]
            assert (a[i], b[i]) == FACTORS[axis]
        # the 4 backgrounds are all distinct → balanced over the other two factors
        assert len({tuple(v for k, v in enumerate(a) if k != i) for a, _ in pairs}) == 4

    def test_proxy_has_no_marital_pairs(self):
        assert axis_pairs("marital_status", "proxy") == []
        assert len(axis_pairs("sex", "proxy")) == len(axis_pairs("age", "proxy")) == 4

    def test_explicit_clause(self):
        assert factorial_clause(("female", 30, "married"), "explicit") == \
            " The applicant is a 30-year-old married woman."
        assert factorial_clause(("male", 50, "single"), "explicit") == \
            " The applicant is a 50-year-old single man."

    def test_proxy_clause(self):
        names = ProxyNames("Abigail", "Todd")
        assert factorial_clause(("female", 30, "single"), "proxy", names) == \
            " The applicant, Abigail, was born in 1996 and is single."
        assert factorial_clause(("male", 50, "married"), "proxy", names) == \
            " The applicant, Todd, was born in 1976 and is married."
        with pytest.raises(ValueError):
            factorial_clause(("male", 50, "married"), "proxy")

    @pytest.mark.parametrize("enc", ["explicit", "proxy"])
    def test_every_pair_passes_the_gate_and_matches_the_cells(self, enc):
        pairs, texts, _ = factorial_pairs(_rec(), "credit_v1", enc, render_profile, random.Random(0))
        assert len(texts) == 8
        assert len(pairs) == (13 if enc == "explicit" else 9)
        for p in pairs:
            res = validate_pair(p)
            assert res.ok, (p.axis, res.reasons)
            assert p.text_a in texts.values() and p.text_b in texts.values()
            assert p.text_a.replace(p.clause_a, "") == p.text_b.replace(p.clause_b, "")

    def test_cell_metadata_marks_varied_and_held_levels(self):
        pairs, _, _ = factorial_pairs(_rec(), "credit_v1", "explicit", render_profile, random.Random(0))
        age_pairs = [p for p in pairs if p.axis == "age"]
        assert {(p.intersectional_cell["sex"], p.intersectional_cell["marital_status"])
                for p in age_pairs} == {("female", "married"), ("female", "single"),
                                        ("male", "married"), ("male", "single")}
        assert all(p.intersectional_cell["age"] == "30-vs-50" for p in age_pairs)
        assert age_pairs[0].held_fixed == ["sex", "marital_status", "financial_content", "template"]
        (corner,) = [p for p in pairs if p.axis == "intersection"]
        assert corner.held_fixed == ["financial_content", "template"]
        assert (corner.label_a, corner.label_b) == ("intersectional", "reference")

    def test_proxy_names_held_within_a_block(self):
        pairs, _, exemplar = factorial_pairs(_rec(), "credit_v1", "proxy", render_profile, random.Random(0))
        for p in pairs:
            if p.axis == "age":
                assert p.clause_a.split(",")[1] == p.clause_b.split(",")[1]  # same name
        assert exemplar["marital_encoding"] == "explicit"

    def test_credit_marker_matches_factorial_form(self):
        spec = credit_marker("sex", "explicit", random.Random(1))
        assert spec.clause_a.endswith(" woman.") and spec.clause_b.endswith(" man.")
        assert spec.clause_a.replace("woman", "man") == spec.clause_b
        with pytest.raises(ValueError):
            credit_marker("marital_status", "proxy", random.Random(1))
        with pytest.raises(ValueError):
            credit_marker("family_status", "explicit", random.Random(1))


# --------------------------------------------------------------------------- generator
class TestGenerator:
    def _records(self):
        return [
            _rec("german-0001"),
            _rec("german-0002", credit_amount_dm=5000),
            _rec("german-0003", employment_since=EMPLOYMENT["A71"], job=JOB["A174"]),  # record rule
            _rec("german-0004", dependents=PEOPLE_LIABLE["2"]),                         # factorial rule
        ]

    def test_build_dataset_counts_and_ids(self):
        from runners.generate_credit import build_dataset

        rows, cells, report = build_dataset(self._records(), seed=7)
        assert report["record_rules"]["n_out"] == 3
        assert report["factorial_rules"]["n_out"] == 2
        assert report["n_records_used"] == 2
        # 2 records × 2 templates × (13 explicit + 9 proxy) pairs
        assert len(rows) == 2 * 2 * (13 + 9)
        assert len({r["id"] for r in rows}) == len(rows)
        assert {r["source_record_id"] for r in rows} == {"german-0001", "german-0002"}
        assert len(cells) == 2 * 2 * 2 and all(len(c["cells"]) == 8 for c in cells)
        assert report["gate"]["explicit"] == {"blocks_kept": 4, "blocks_dropped": 0, "failure_reasons": {}}

    def test_build_dataset_is_deterministic(self):
        from runners.generate_credit import build_dataset

        a = build_dataset(self._records(), seed=7)
        b = build_dataset(self._records(), seed=7)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    def test_cells_carry_real_fields_but_texts_do_not(self):
        from runners.generate_credit import build_dataset

        _, cells, _ = build_dataset([_rec("german-0001", raw_age_years=67)], encodings=("explicit",))
        assert cells[0]["real_fields"] == {"sex": "female", "marital": "single", "age": 67}
        assert all("67" not in c["text"] for c in cells[0]["cells"])

    def test_axes_filter(self):
        from runners.generate_credit import build_dataset

        rows, _, _ = build_dataset(self._records(), axes=("intersection",), encodings=("explicit",))
        assert {r["varied_axis"] for r in rows} == {"intersection"}


# --------------------------------------------------------------------------- split
class TestGroupedSplit:
    def _manifest(self, tmp_path, n_records=30):
        from runners.generate_credit import build_dataset

        recs = [_rec(f"german-{i:04d}", credit_amount_dm=1000 + i) for i in range(n_records)]
        rows, _, _ = build_dataset(recs, encodings=("explicit",))
        path = tmp_path / "pairs.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        return path

    def test_no_record_in_both_splits(self, tmp_path):
        from scoring.pair_dataset import CreditDemographicDataset

        ds = CreditDemographicDataset(str(self._manifest(tmp_path)), axis="sex", encoding="explicit",
                                      probe_size=40, split_seed=42)
        ds._ensure_loaded()
        raw = ds._raw_data
        probe = {raw[i]["source_record_id"] for i in ds._probe_indices}
        test = {raw[i]["source_record_id"] for i in ds._test_indices}
        assert probe and test and not probe & test
        assert len(ds._probe_indices) >= 40
        assert len(ds._probe_indices) + len(ds._test_indices) == len(raw)

    def test_test_cap_spreads_over_records(self, tmp_path):
        from scoring.pair_dataset import CreditDemographicDataset

        ds = CreditDemographicDataset(str(self._manifest(tmp_path)), axis="sex", encoding="explicit",
                                      probe_size=40, split_seed=42, max_test_examples=10)
        ds._ensure_loaded()
        # 10 capped examples come from 10 different records (round-robin), not from 2
        assert len({ds._raw_data[i]["source_record_id"] for i in ds._test_indices}) == 10

    def test_only_credit_is_grouped(self):
        from scoring.bios_dataset import BiosDemographicDataset
        from scoring.education_dataset import EducationDemographicDataset
        from scoring.pair_dataset import CreditDemographicDataset, MatchedPairDataset

        row = {"id": "x", "source_record_id": "r"}
        for cls in (BiosDemographicDataset, EducationDemographicDataset):
            assert issubclass(cls, MatchedPairDataset) and not issubclass(cls, CreditDemographicDataset)
            assert cls("unused", axis="sex", encoding="explicit")._get_group_key(row) is None
        assert CreditDemographicDataset("unused", axis="sex", encoding="explicit")._get_group_key(row) == "r"

    def test_base_class_needs_a_domain(self):
        from scoring.pair_dataset import MatchedPairDataset

        with pytest.raises(TypeError):
            MatchedPairDataset("unused", axis="sex", encoding="explicit")

    def test_default_prompt_and_name_per_domain(self):
        from scoring.bios_dataset import BIOS_ASSESSMENT_PROMPT, BiosDemographicDataset
        from scoring.pair_dataset import ASSESSMENT_PROMPT, CreditDemographicDataset

        credit = CreditDemographicDataset("unused", axis="sex", encoding="proxy")
        bios = BiosDemographicDataset("unused", axis="age", encoding="explicit", prompt="custom")
        assert (credit.name, credit.prompt) == ("credit_demographic_sex_proxy", ASSESSMENT_PROMPT)
        assert (bios.name, bios.prompt) == ("cv_demographic_age_explicit", "custom")
        assert BiosDemographicDataset("unused", axis="sex", encoding="explicit").prompt == BIOS_ASSESSMENT_PROMPT


# --------------------------------------------------------------------------- registry / markers
def test_credit_domain_uses_marital_status():
    from substrates.domains import get_domain

    credit = get_domain("credit")
    assert credit.axes == ("sex", "age", "marital_status", "intersection")
    assert credit.make_marker is credit_marker
    assert "family_status" in get_domain("cv").axes


def test_standalone_marital_marker_is_explicit_only():
    from pairs.markers import make_marker

    spec = make_marker("marital_status", "explicit", random.Random(0))
    assert (spec.clause_a, spec.clause_b) == (" The applicant is married.", " The applicant is single.")
    with pytest.raises(ValueError):
        make_marker("marital_status", "proxy", random.Random(0))
