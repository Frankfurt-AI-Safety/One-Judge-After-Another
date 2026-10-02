"""
Validate the concept-erasure / probe-recoverability harness on SYNTHETIC data (no model), so the
LEACE + non-linear-probe verdict can be trusted: (1) LEACE drives a *linearly*-encoded concept's linear
probe to chance; (2) the metric distinguishes a non-linearly-encoded (XOR) concept — MLP recovers it
where the linear probe cannot.
"""

from __future__ import annotations

import numpy as np
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


def test_joint_leace_closes_the_xor_route_single_leace_leaves_open():
    # states with no truth code at all: valence, premise and conclusion only. Pooled over two premises of opposite
    # truth configuration, truth = valence XOR premise — linearly invisible, so LEACE on truth alone removes nothing
    # and an MLP rebuilds truth from the two (a false "entangled" reading). Erasing truth and valence jointly removes
    # the valence route: the MLP falls to chance.
    import numpy as np
    from probes.erasure import apply_eraser, leace_erase, probe_recoverability

    rng = np.random.default_rng(0)
    n = 1200
    valence, premise, conclusion = (rng.integers(0, 2, n) for _ in range(3))
    truth = (valence == premise).astype(int)                 # premise 1 = favourable-truth: its true claim is favourable
    X = torch.tensor(np.column_stack([2 * valence - 1, 2 * premise - 1, 2 * conclusion - 1,
                                      rng.normal(0, 0.1, (n, 5))]), dtype=torch.float32)
    tr, ev = slice(0, 800), slice(800, n)

    def mlp_after(erase):
        eraser = leace_erase(X[tr], erase)
        return probe_recoverability(apply_eraser(eraser, X[tr]), truth[tr].tolist(), apply_eraser(eraser, X[ev]),
                                    truth[ev].tolist(), n_boot=0, mlp_seeds=3)["mlp_acc"]

    assert mlp_after(truth[tr].tolist()) > 0.9                                   # the false positive
    assert abs(mlp_after([truth[tr].tolist(), valence[tr].tolist()]) - 0.5) < 0.1   # closed


def test_the_mlp_vote_needs_an_odd_number_of_seeds():
    X = torch.randn(40, 3)
    y = [0, 1] * 20
    with pytest.raises(ValueError, match="odd"):
        from probes.erasure import probe_recoverability
        probe_recoverability(X, y, X, y, n_boot=0, mlp_seeds=2)


def test_pooling_the_folds_keeps_each_cluster_one_cluster():
    from probes.erasure import pool_folds

    # two folds of two applicants × two states; fold 0's MLP right on applicant 0 only, fold 1's on both
    item = lambda ok, label: (ok, ok, label, False)
    folds = [[item(True, 1), item(True, 0), item(False, 1), item(False, 0)],
             [item(True, 1), item(True, 0), item(True, 1), item(True, 0)]]
    groups = [0, 0, 1, 1]                                   # one fold's states → applicant
    pooled = pool_folds(folds, [groups, groups], n_boot=200, seed=0)
    iv = pooled["intervals"]["mlp_acc"]
    assert pooled["mlp_acc"] == 0.75 and (iv["n_clusters"], iv["n_items"]) == (2, 8)
    # applicant 0 scores 1, applicant 1 scores ½: resampling whole applicants gives 0.5–1, never another split
    assert iv["ci_low"] == 0.5 and iv["ci_high"] == 1.0
    assert [f["mlp_acc"] for f in pooled["by_fold"]] == [0.5, 1.0]
    with pytest.raises(ValueError, match="cluster key"):
        pool_folds(folds, [groups, groups[:3]], n_boot=10, seed=0)
    with pytest.raises(ValueError, match="cluster key"):
        pool_folds(folds, [groups], n_boot=10, seed=0)


def test_pooling_folds_that_evaluate_different_states():
    from probes.erasure import pool_folds

    # name folds: each fold scores other states of the records; a record's states of every fold form one cluster
    item = lambda ok, label: (ok, ok, label, False)
    folds = [[item(True, 1), item(False, 0)], [item(True, 0), item(True, 1), item(True, 0)], []]
    keys = [["r0", "r1"], ["r0", "r1", "r1"], []]
    pooled = pool_folds(folds, keys, n_boot=200, seed=0)
    iv = pooled["intervals"]["mlp_acc"]
    assert pooled["mlp_acc"] == 0.8 and (iv["n_clusters"], iv["n_items"]) == (2, 5)
    # r0: 2 of 2 right, r1: 2 of 3 right; resampled whole: 2/3 (r1 twice) to 1 (r0 twice)
    assert iv["ci_low"] == pytest.approx(2 / 3) and iv["ci_high"] == 1.0
    assert pooled["by_fold"][2] is None and [f["mlp_acc"] for f in pooled["by_fold"][:2]] == [0.5, 1.0]


def test_a_feature_erased_to_rounding_is_not_rescaled_into_a_separator():
    # bag-of-words-like: two features that ARE the label (one word per pole) plus label-free ones. LEACE leaves the
    # two at 0.5 ± float32 rounding; standardising that residue to unit variance made it a perfect separator
    from probes.erasure import SNAP_RTOL

    from itertools import product

    grid = np.array(list(product((0, 1), repeat=5)))                         # label × 4 label-free words, balanced
    rows = np.vstack([grid] * 12)
    y = rows[:, 0]
    X = torch.tensor(np.column_stack([y, 1 - y, rows[:, 1:]]), dtype=torch.float32)
    tr, ev = slice(0, 32 * 9), slice(32 * 9, 32 * 12)
    er = leace_erase(X[tr], y[tr])
    Etr, Eev = apply_eraser(er, X[tr]), apply_eraser(er, X[ev])
    assert 0 < Etr[:, 0].std() <= SNAP_RTOL * X[tr][:, 0].std()             # rounding, not signal
    naive = probe_recoverability(Etr, y[tr], Eev, y[ev], n_boot=0)
    assert naive["linear_acc"] > 0.95                                         # the bug, kept visible
    guarded = probe_recoverability(Etr, y[tr], Eev, y[ev], n_boot=0, unerased_tr=X[tr])
    assert guarded["snapped_features"] == 2
    assert guarded["linear_acc"] < 0.65 and guarded["mlp_acc"] < 0.65
    # without erasure nothing is snapped
    assert probe_recoverability(X[tr], y[tr], X[ev], y[ev], n_boot=0, unerased_tr=X[tr])["snapped_features"] == 0
    with pytest.raises(ValueError, match="shape"):
        probe_recoverability(Etr, y[tr], Eev, y[ev], n_boot=0, unerased_tr=X[ev])


def test_a_word_in_every_text_is_snapped_too():
    # review 2026-10-02: a feature constant BEFORE erasure (a word in every clause) gets a label-correlated rounding
    # residue from LEACE; it must be snapped as well, but it does not count as one the erasure left behind
    from itertools import product

    grid = np.array(list(product((0, 1), repeat=5)))
    rows = np.vstack([grid] * 12)
    y = rows[:, 0]
    X = torch.tensor(np.column_stack([y, 1 - y, np.ones(len(y)), rows[:, 1:]]), dtype=torch.float32)
    tr, ev = slice(0, 32 * 9), slice(32 * 9, 32 * 12)
    er = leace_erase(X[tr], y[tr])
    Etr, Eev = apply_eraser(er, X[tr]), apply_eraser(er, X[ev])
    # the residue the review measured (std ~4e-8), planted on both splits as LEACE's own residue is (a function of x)
    Etr[:, 2] = 1 + torch.tensor((y[tr] - 0.5) * 8e-8, dtype=torch.float64).float()
    Eev[:, 2] = 1 + torch.tensor((y[ev] - 0.5) * 8e-8, dtype=torch.float64).float()
    Etr[:, :2], Eev[:, :2] = 0.5, 0.5                                     # the label words: exactly erased here
    assert Etr[:, 2].std() > 0
    naive = probe_recoverability(Etr, y[tr], Eev, y[ev], n_boot=0)
    assert naive["linear_acc"] > 0.95                                     # the bug, kept visible
    guarded = probe_recoverability(Etr, y[tr], Eev, y[ev], n_boot=0, unerased_tr=X[tr])
    assert guarded["linear_acc"] < 0.65 and guarded["mlp_acc"] < 0.65
    assert guarded["snapped_features"] == 1                    # the constant word (woman/man were left exactly 0.5)


def test_chance_is_each_folds_majority_share():
    from probes.erasure import pool_folds

    # fold 0: 3 of 4 labels 1; fold 1: 1 of 4. Pooled share ½, but within each fold an uninformative probe can reach ¾
    item = lambda label: (True, True, label, False)
    folds = [[item(1), item(1), item(1), item(0)], [item(0), item(0), item(0), item(1)]]
    pooled = pool_folds(folds, [list("abcd"), list("abcd")], n_boot=50, seed=0)
    assert pooled["chance"] == 0.75
    # the same states in one fit: the plain majority share
    assert pool_folds([folds[0] + folds[1]], [list("abcdabcd")], n_boot=0, seed=0)["chance"] == 0.5


def test_names_resampled_crossed_widen_the_interval():
    from probes.erasure import pool_folds

    # 40 records × 4 names; the probe is right on every name but one, where it is always wrong: a record bootstrap
    # treats that name as fixed, the crossed one also draws names
    folds, keys, names = [[]], [[]], [[]]
    for r in range(40):
        for n in "ABCD":
            folds[0].append((n != "D", n != "D", r % 2, False))
            keys[0].append(r)
            names[0].append(n)
    records = pool_folds(folds, keys, n_boot=500, seed=0)["intervals"]["mlp_acc"]
    crossed = pool_folds(folds, keys, n_boot=500, seed=0, keys_b=names)["intervals"]["mlp_acc"]
    assert records["ci_high"] - records["ci_low"] < 1e-9           # every record scores ¾: no spread over records
    assert crossed["ci_high"] - crossed["ci_low"] > 0.2 and crossed["n_clusters_b"] == 4
    with pytest.raises(ValueError, match="second key"):
        pool_folds(folds, keys, n_boot=10, seed=0, keys_b=[names[0][:-1]])
