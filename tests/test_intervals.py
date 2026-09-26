"""Cluster-bootstrap intervals for the arms that reported point estimates only (`scoring/intervals.py`):
the estimate is always the arm's existing point estimate, and the interval resamples whole clusters."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from runners.run_additivity import additivity_intervals
from runners.run_positioned_maineffect import positioned_intervals
from scoring.demographic_experiment import (
    auto_influence_with_intervals, compute_auto_influence_metrics, compute_decision_response_metrics,
    decision_response_intervals, nulling_change,
)
from scoring.dataset_base import EvalExample
from scoring.experiment import ExperimentConfig, ExperimentResults
from scoring.intervals import change_intervals, cluster_bootstrap, clusters_of, pair_intervals

MEAN = {"mean": lambda s: float(np.mean(s))}


def test_estimate_is_the_statistic_on_all_items_and_is_bracketed():
    rng = np.random.default_rng(0)
    clusters = [list(rng.normal(1.0, 1.0, 4)) for _ in range(50)]
    out = cluster_bootstrap(clusters, MEAN, n_boot=500, seed=1)["mean"]
    assert out["estimate"] == pytest.approx(np.mean([x for c in clusters for x in c]))
    assert out["ci_low"] < out["estimate"] < out["ci_high"]
    assert (out["n_clusters"], out["n_items"]) == (50, 200)
    assert cluster_bootstrap(clusters, MEAN, n_boot=500, seed=1)["mean"] == out     # deterministic


def test_clustering_widens_the_interval_when_a_cluster_repeats_one_value():
    # 40 records, each contributing 8 identical pairs: only 40 independent values, not 320
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, 40)
    clustered = cluster_bootstrap([[v] * 8 for v in values], MEAN, n_boot=1000, seed=0)["mean"]
    naive = cluster_bootstrap([[v] for v in values for _ in range(8)], MEAN, n_boot=1000, seed=0)["mean"]
    width = lambda d: d["ci_high"] - d["ci_low"]
    assert width(clustered) > 2 * width(naive)


def test_clusters_of_groups_by_key_in_first_seen_order():
    assert clusters_of(["a", "b", "c", "d"], ["r2", "r1", "r2", "r3"]) == [["a", "c"], ["b"], ["d"]]


def _examples(records):
    return [EvalExample(texts={}, metadata={"source_record_id": r}) for r in records]


def test_pair_intervals_estimate_the_auto_influence_metrics():
    rng = np.random.default_rng(3)
    a, b = list(rng.normal(0.2, 1, 60)), list(rng.normal(0.0, 1, 60))
    keys = [f"r{i // 4}" for i in range(60)]
    ci = pair_intervals(a, b, keys, n_boot=300)
    point = compute_auto_influence_metrics({"a": a, "b": b})
    for key in ("mean_gap", "abs_mean_gap", "pref_a_rate", "auto_influence"):
        assert ci[key]["estimate"] == pytest.approx(point[key])
        assert ci[key]["ci_low"] <= ci[key]["estimate"] <= ci[key]["ci_high"]
    assert ci["mean_gap"]["n_clusters"] == 15
    # the experiment's metric carries the same point values plus the intervals
    m = auto_influence_with_intervals({"a": a, "b": b}, _examples(keys), n_boot=300)
    assert m["mean_gap"] == point["mean_gap"] and m["intervals"]["mean_gap"] == ci["mean_gap"]


def test_nulling_change_is_paired():
    a, b = [1.0, 2.0, 0.5, 1.5], [0.0, 0.5, 0.0, 1.0]
    keys = ["r0", "r0", "r1", "r1"]
    same = change_intervals((a, b), (a, b), keys, n_boot=200)
    assert all(v["estimate"] == 0 and v["ci_low"] == 0 == v["ci_high"] for v in same.values())
    # nulling that leaves only ±0.1 gaps, half each way: the preference is gone, |gap| falls to 0.1
    # (exact zeros would all count as "B preferred" and leave auto_influence at 1, as the metric defines it)
    nulled_b = [x - d for x, d in zip(a, (0.1, -0.1, 0.1, -0.1))]
    gone = nulling_change({"a": a, "b": b}, {"a": a, "b": nulled_b}, _examples(keys),
                          n_boot=200)["nulled_minus_baseline"]
    assert gone["abs_mean_gap_change"]["estimate"] == pytest.approx(0.1 - np.mean(np.abs(np.subtract(a, b))))
    assert gone["auto_influence_change"]["estimate"] == pytest.approx(-1.0)


def test_decision_intervals_estimate_the_decision_metrics():
    rng = np.random.default_rng(5)
    base = {v: list(rng.normal(mu, 1, 80)) for v, mu in
            (("fair", 1.0), ("discriminatory", 0.0), ("neutral", 0.5), ("evasive", 0.2))}
    out = decision_response_intervals(base, base, n_boot=300)
    point = compute_decision_response_metrics(base)
    for key in ("discriminatory_win_rate", "mean_gap_fair_minus_disc", "disc_win_rate_vs_neutral",
                "evasion_win_rate"):
        assert out["baseline"][key]["estimate"] == pytest.approx(point[key])
    assert out["nulled_minus_baseline"]["discriminatory_win_rate_change"]["estimate"] == 0
    # without the optional variants their metrics are not reported
    two = {k: base[k] for k in ("fair", "discriminatory")}
    assert set(decision_response_intervals(two, two, n_boot=50)["baseline"]) == {
        "discriminatory_win_rate", "mean_gap_fair_minus_disc"}


def test_positioned_intervals_estimate_the_decomposition():
    r_neu = [0.0, 1.0, 2.0]
    r_a, r_b = [0.5, 0.7, 1.4, 1.6, 2.1, 2.9], [0.1, 0.3, 1.2, 0.8, 2.3, 2.0]
    owner = [0, 0, 1, 1, 2, 2]
    out = positioned_intervals(r_neu, r_a, r_b, owner, n_boot=200)
    d_a = [a - r_neu[o] for a, o in zip(r_a, owner)]
    d_b = [b - r_neu[o] for b, o in zip(r_b, owner)]
    assert out["delta_a"]["estimate"] == pytest.approx(np.mean(d_a))
    assert out["identity_gap"]["estimate"] == pytest.approx(np.mean(d_a) - np.mean(d_b))
    assert out["main_effect"]["estimate"] == pytest.approx((np.mean(d_a) + np.mean(d_b)) / 2)
    assert out["auto_influence"]["estimate"] == pytest.approx(2 * abs(5 / 6 - 0.5))   # 5 of 6 pairs a > b
    assert out["delta_a"]["n_clusters"] == 3


def test_additivity_intervals_on_an_additive_design():
    torch.manual_seed(0)
    axes = ["sex", "age", "marital_status"]
    base = {a: torch.randn(16) for a in axes}
    contrasts = {a: {f"r{i}": base[a] + 0.1 * torch.randn(16) for i in range(30)} for a in axes}
    unit = lambda v: v / v.norm()
    # intersection = the sum of the unit marginals, per record: additive by construction
    contrasts["intersection"] = {r: sum(unit(base[a]) for a in axes) + 0.05 * torch.randn(16)
                                 for r in contrasts["sex"]}
    out = additivity_intervals(contrasts, axes, n_boot=300, seed=0)
    c = out["cos_intersection_vs_marginal_sum"]
    assert c["estimate"] > 0.95 and c["ci_low"] <= c["estimate"] <= c["ci_high"] and c["n_records"] == 30
    assert set(out) == {"cos_intersection_vs_marginal_sum", "cos_sex_age", "cos_sex_marital", "cos_age_marital"}
    mean_dir = lambda a: unit(torch.stack(list(contrasts[a].values())).mean(0))
    assert out["cos_sex_age"]["estimate"] == pytest.approx(float(mean_dir("sex") @ mean_dir("age")), abs=1e-5)


def test_results_round_trip_keeps_the_paired_metrics(tmp_path):
    cfg = ExperimentConfig(name="x", bias_type="demographic", model_path="m")
    paired = {"nulled_minus_baseline": {"mean_gap_change": {"estimate": -0.1, "ci_low": -0.2, "ci_high": 0.0}}}
    res = ExperimentResults(config=cfg, baseline_metrics={"mean_gap": 0.3}, paired_metrics=paired)
    res.save(tmp_path / "r.json")
    assert ExperimentResults.load(tmp_path / "r.json").paired_metrics == paired
    assert math.isclose(res.to_dict()["baseline"]["mean_gap"], 0.3)
