"""`runners/run_additivity.py`: the corner against the VECTOR sum of the marginal contrasts (RQ1.1), which the
two-way interactions cannot move and a three-way term does, and the runner end to end on the tiny model."""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pytest
import torch

from runners import run_additivity as ra
from runners.run_additivity import contrast_lengths, pairwise_intervals, threeway_residual
from tests.test_run_cross_marker import _model, _tokenizer

AXES = ["sex", "age", "marital_status"]
CELLS = list(itertools.product((1, -1), repeat=3))


def _contrasts(two_way=0.0, three_way=0.0, n=40, d=24, noise=0.05, seed=0):
    """Per-record contrasts from 8-cell states h(x) = Σ a_i x_i + b x0 x1 + c x0 x1 x2 + noise (effects coding):
    each marginal contrast averages its four pairs, the corner is h(+++) − h(−−−). The main effects differ in
    length (sex three times the others)."""
    rng = np.random.default_rng(seed)
    a = [rng.normal(size=d) * s for s in (3.0, 1.0, 1.0)]
    b, c = rng.normal(size=d) * two_way, rng.normal(size=d) * three_way
    out = {axis: {} for axis in AXES + ["intersection"]}
    for r in range(n):
        h = {x: sum(ai * xi for ai, xi in zip(a, x)) + b * x[0] * x[1] + c * x[0] * x[1] * x[2]
             + rng.normal(size=d) * noise for x in CELLS}
        for i, axis in enumerate(AXES):
            flip = lambda x: tuple(-v if k == i else v for k, v in enumerate(x))
            out[axis][f"r{r}"] = torch.tensor(np.mean([h[x] - h[flip(x)] for x in CELLS if x[i] == 1], 0))
        out["intersection"][f"r{r}"] = torch.tensor(h[(1, 1, 1)] - h[(-1, -1, -1)])
    return out


def test_an_additive_design_with_unequal_lengths_is_at_one():
    c = _contrasts()
    out = threeway_residual(c, AXES, n_flips=2000, seed=0)
    assert out["cos_intersection_vs_marginal_sum"] > 0.999 and out["n_records"] == 40
    assert out["residual_share_debiased"] < out["noise_floor_share"] and out["sign_flip_p"] > 0.05
    # the old comparison with the sum of UNIT directions calls this additive design non-additive
    mean = {a: torch.stack(list(c[a].values())).mean(0) for a in c}
    unit = lambda v: v / v.norm()
    assert float(unit(mean["intersection"]) @ unit(sum(unit(mean[a]) for a in AXES))) < 0.95


def test_two_way_interactions_are_invisible_a_three_way_term_is_not():
    two = threeway_residual(_contrasts(two_way=2.0), AXES, n_flips=2000, seed=0)
    three = threeway_residual(_contrasts(three_way=0.3), AXES, n_flips=2000, seed=0)
    assert two["cos_intersection_vs_marginal_sum"] > 0.999 and two["sign_flip_p"] > 0.05   # a strong two-way term cancels
    assert three["sign_flip_p"] < 0.001 and three["residual_share_debiased"] > 10 * three["noise_floor_share"]


def test_the_debiased_share_removes_the_noise_floor():
    # pure noise residuals: the raw share sits near the floor, the debiased one near 0, p roughly uniform
    noisy = threeway_residual(_contrasts(noise=0.5), AXES, n_flips=2000, seed=0)
    assert noisy["residual_share_raw"] == pytest.approx(noisy["noise_floor_share"], rel=0.3)
    assert noisy["residual_share_debiased"] < noisy["residual_share_raw"] / 2
    ps = [threeway_residual(_contrasts(noise=0.5, seed=s), AXES, n_flips=500, seed=s)["sign_flip_p"] for s in range(20)]
    assert 0.1 < np.mean(ps) < 0.9 and sum(p < 0.05 for p in ps) <= 3


def test_pairwise_cosines_cover_their_estimates_and_the_lengths():
    c = _contrasts()
    out = pairwise_intervals(c, AXES, n_boot=200, seed=0)
    assert set(out) == {"cos_sex_age", "cos_sex_marital", "cos_age_marital"}
    mean = {a: torch.stack(list(c[a].values())).mean(0) for a in c}
    unit = lambda v: v / v.norm()
    assert out["cos_sex_age"]["estimate"] == pytest.approx(float(unit(mean["sex"]) @ unit(mean["age"])), abs=1e-9)
    assert all(v["ci_low"] <= v["estimate"] <= v["ci_high"] for v in out.values())
    lengths = contrast_lengths(c, AXES + ["intersection"])
    assert lengths["sex"] == pytest.approx(float(mean["sex"].norm()), rel=1e-9) and lengths["sex"] > 2 * lengths["age"]


def test_end_to_end_on_a_tiny_model(manifest, tmp_path, monkeypatch):
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    loads = []

    def load_model(self):
        loads.append(1)
        self.model, self.tokenizer = _model(), _tokenizer()
        self.config.model_revision = "abc123"

    monkeypatch.setattr(ra.DemographicBiasExperiment, "load_model", load_model)
    argv = ["run_additivity.py", "--domain", "credit", "--pairs", str(manifest), "--model", "org/Tiny-RM",
            "--probe-records", "6", "--max-length", "1024", "--batch-size", "16", "--n-boot", "50",
            "--n-flips", "200"]
    monkeypatch.setattr("sys.argv", argv)
    ra.main()
    result = json.loads((tmp_path / "artifacts/results/demographic/additivity_credit_Tiny-RM_explicit.json").read_text())
    assert result["meta"]["config"]["model_revision"] == "abc123" and result["meta"]["config"]["probe_records"] == 6
    assert result["meta"]["data"]["pairs.jsonl"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result["cos_intersection_vs_marginal_sum"] == result["threeway"]["cos_intersection_vs_marginal_sum"]
    assert {"threeway", "contrast_lengths", "cos_sex_age", "pairwise_intervals", "probe_splits"} <= set(result)
    assert {"verdict", "intervals", "cosine_ceiling"}.isdisjoint(result)
    with pytest.raises(SystemExit, match="exists"):
        ra.main()
    assert len(loads) == 1
