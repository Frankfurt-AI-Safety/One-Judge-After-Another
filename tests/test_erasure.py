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
    apply_diffmean, apply_eraser, diffmean_direction, leace_erase, probe_recoverability,
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
    dvec = diffmean_direction(Xtr, ytr)
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


def _linear_concept(n=600, d=12, seed=0):
    torch.manual_seed(seed)
    y = torch.randint(0, 2, (n,))
    X = torch.randn(n, d)
    X[:, 0] += 3.0 * (y.float() - 0.5)
    return X, y


def test_the_interval_decides_at_chance():
    X, y = _linear_concept()
    Xtr, ytr, Xev, yev = _split(X, y, 400)
    before = probe_recoverability(Xtr, ytr, Xev, yev, n_boot=500)["intervals"]["mlp_above_chance"]
    assert before["ci_low"] > 0.2                           # decodable: clearly above chance
    er = leace_erase(Xtr, ytr)
    after = probe_recoverability(apply_eraser(er, Xtr), ytr, apply_eraser(er, Xev), yev, n_boot=500)["intervals"]
    # erased: neither probe is clearly above the majority-class baseline (which is >= 0.5, so a probe at
    # chance sits at or below it: the reading is one-sided)
    for probe in ("linear_above_chance", "mlp_above_chance"):
        assert after[probe]["ci_low"] < 0 and after[probe]["estimate"] < 0.05


def test_clustered_states_widen_the_interval():
    # four copies of every eval state = four correlated states per item: resampled one by one they look
    # like 4x the data; resampled by item the interval keeps the width of the underlying 200 items
    X, y = _linear_concept(n=600, d=12, seed=3)
    X[:, 0] = torch.randn(600)                              # no signal: accuracies near chance
    Xtr, ytr, Xev, yev = _split(X, y, 400)
    Xev4, yev4 = Xev.repeat_interleave(4, 0), yev.repeat_interleave(4)
    groups = [i // 4 for i in range(len(yev4))]
    width = lambda r: r["intervals"]["mlp_acc"]["ci_high"] - r["intervals"]["mlp_acc"]["ci_low"]
    naive = probe_recoverability(Xtr, ytr, Xev4, yev4, n_boot=500)
    clustered = probe_recoverability(Xtr, ytr, Xev4, yev4, groups_ev=groups, n_boot=500)
    assert naive["mlp_acc"] == clustered["mlp_acc"]          # the point estimate does not change
    assert width(clustered) > 1.6 * width(naive)
    assert clustered["intervals"]["mlp_acc"]["n_clusters"] == 200
    with pytest.raises(ValueError, match="groups"):
        probe_recoverability(Xtr, ytr, Xev4, yev4, groups_ev=groups[:-1])


@pytest.mark.parametrize("bad", [lambda y: y + 1, lambda y: y * 2 - 1])
def test_labels_other_than_zero_one_are_refused(bad):
    # 1/2 coding made the difference-of-means direction NaN (an empty class), which the projection drops:
    # the "diffmean" row would silently equal "none"
    X, y = _linear_concept(n=100)
    for fn in (lambda: diffmean_direction(X, bad(y)), lambda: leace_erase(X, bad(y)),
               lambda: probe_recoverability(X[:60], bad(y[:60]), X[60:], y[60:])):
        with pytest.raises(ValueError, match="0/1"):
            fn()


def test_a_training_split_needs_both_classes_and_bool_labels_work():
    X, y = _linear_concept(n=100)
    with pytest.raises(ValueError, match="both classes"):
        diffmean_direction(X, torch.zeros(100, dtype=torch.long))
    assert torch.equal(diffmean_direction(X, y.bool()), diffmean_direction(X, y))
    assert torch.equal(diffmean_direction(X, y.float()), diffmean_direction(X, y))
