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
