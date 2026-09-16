"""
Validate the concept-erasure / probe-recoverability harness on SYNTHETIC data (no model), so the
LEACE + non-linear-probe verdict can be trusted: (1) LEACE drives a *linearly*-encoded concept's linear
probe to chance; (2) the metric distinguishes a non-linearly-encoded (XOR) concept — MLP recovers it
where the linear probe cannot.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("concept_erasure")

from probes.erasure import (  # noqa: E402
    apply_diffmean, apply_eraser, diffmean_erase, leace_erase, probe_recoverability,
)


def _split(X, y, n_tr):
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


def test_leace_kills_linear_for_linear_concept():
    torch.manual_seed(0)
    n, d = 600, 12
    y = torch.randint(0, 2, (n,))
    X = torch.randn(n, d)
    X[:, 0] += 3.0 * (y.float() - 0.5)  # concept on a single linear direction
    Xtr, ytr, Xev, yev = _split(X, y, 400)

    base = probe_recoverability(Xtr, ytr, Xev, yev)
    assert base["linear_acc"] > 0.85  # clearly decodable

    er = leace_erase(Xtr, ytr)
    after = probe_recoverability(apply_eraser(er, Xtr), ytr, apply_eraser(er, Xev), yev)
    assert after["linear_acc"] < 0.65  # LEACE → linear probe ≈ chance

    # diffmean erasure also removes the (single) linear direction here
    dvec = diffmean_erase(Xtr, ytr)
    aff = probe_recoverability(apply_diffmean(dvec, Xtr), ytr, apply_diffmean(dvec, Xev), yev)
    assert aff["linear_acc"] < 0.65


def test_mlp_recovers_xor_where_linear_fails():
    torch.manual_seed(1)
    n, d = 900, 8
    core0 = (torch.randint(0, 2, (n,)) * 2 - 1).float() * 1.5  # clean ±1.5 clusters
    core1 = (torch.randint(0, 2, (n,)) * 2 - 1).float() * 1.5
    X = torch.randn(n, d) * 0.5
    X[:, 0] += core0
    X[:, 1] += core1
    y = ((core0 > 0) ^ (core1 > 0)).long()  # cleanly-separable XOR (non-linear)
    Xtr, ytr, Xev, yev = _split(X, y, 650)

    r = probe_recoverability(Xtr, ytr, Xev, yev)
    assert r["linear_acc"] < 0.65   # linear probe can't represent XOR
    assert r["mlp_acc"] > 0.80      # MLP recovers it → harness detects non-linear structure
