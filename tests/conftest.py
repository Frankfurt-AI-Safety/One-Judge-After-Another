"""Shared fixtures."""

import pytest


@pytest.fixture
def manifest(tmp_path):
    """The credit manifest of `tests.test_run_cross_marker.credit_manifest`: its pairs.jsonl path."""
    from tests.test_run_cross_marker import credit_manifest

    return credit_manifest(tmp_path)


@pytest.fixture
def hiring(tmp_path, monkeypatch):
    """A hiring manifest from `generate_bios` on a synthetic parquet (96 bios): (pairs.jsonl, parquet)."""
    from tests.test_bios_pipeline import TestGenerateBiosCLI

    raw, out = TestGenerateBiosCLI()._main(tmp_path, monkeypatch)
    return out / "pairs.jsonl", raw


@pytest.fixture
def credit_corpus(tmp_path, monkeypatch):
    """A credit manifest from `generate_credit` on a synthetic german.data (60 records, every other one good credit;
    of the good ones, 6 unemployed with an unskilled job and 6 employed with the job "unemployed or unskilled", which
    the record rules allow and the reasoning arm does not draw): (pairs.jsonl, german.data)."""
    from runners import generate_credit

    codes = ("A91", "A92", "A93", "A94")
    raw, out = tmp_path / "credit_raw" / "german.data", tmp_path / "credit"
    raw.parent.mkdir()
    raw.write_text("".join(
        f"A14 12 A34 A43 {1000 + 37 * i} A61 {'A71' if i % 10 == 0 else 'A73'} 2 {codes[i % 4]} A101 2 A121 "
        f"{30 + i % 30} A143 A152 1 {'A172' if i % 10 == 0 else 'A171' if i % 10 == 4 else 'A173'} 1 A192 A201 "
        f"{1 + i % 2}\n"
        for i in range(60)))
    monkeypatch.setattr("sys.argv", ["generate_credit.py", "--raw", str(raw), "--out-dir", str(out)])
    generate_credit.main()
    return out / "pairs.jsonl", raw


@pytest.fixture
def education_corpus(tmp_path, monkeypatch):
    """An education manifest from `generate_education` on a synthetic ASAP 2.0 file (40 essays, every other one
    strong): (pairs.jsonl, the csv)."""
    from runners import generate_education
    from tests.test_education_stage import _asap2_csv

    (tmp_path / "edu_raw").mkdir()
    raw, out = _asap2_csv(tmp_path / "edu_raw" / "asap2.csv", n=40), tmp_path / "education"
    monkeypatch.setattr("sys.argv", ["generate_education.py", "--raw-path", str(raw), "--out-dir", str(out)])
    generate_education.main()
    return out / "pairs.jsonl", raw


def reasoning_domain(domain, raw):
    """The domain's spec with its records and corpus read from ``raw`` (a fixture's corpus) instead of data/."""
    import dataclasses

    from substrates.domains import get_domain

    dom = get_domain(domain)
    if domain == "cv":
        from substrates.bios_clean import DEFAULT_N_BIOS, load_factorial_bios
        load = lambda: load_factorial_bios(str(raw), n=DEFAULT_N_BIOS)
    elif domain == "credit":
        from substrates.credit_clean import load_factorial_records
        load = lambda: load_factorial_records(raw)
    else:
        from substrates.education_clean import load_education_essays
        load = lambda: load_education_essays(raw, source="asap2")
    return dataclasses.replace(dom, load_records=load, corpus=raw)
