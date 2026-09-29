"""Shared fixtures."""

import pytest


@pytest.fixture
def manifest(tmp_path):
    """The credit manifest of `tests.test_run_cross_marker.credit_manifest`: its pairs.jsonl path."""
    from tests.test_run_cross_marker import credit_manifest

    return credit_manifest(tmp_path)
