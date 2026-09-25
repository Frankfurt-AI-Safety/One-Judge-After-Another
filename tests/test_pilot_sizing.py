"""Pilot sizing (`runners/pilot_sizing.py`): records per group from the pilot's per-record SDs — blinded to
every mean — and the probe-records answer from the probe-size curves."""

from __future__ import annotations

import copy

import pytest

from runners.pilot_sizing import precision, probe_answers, required_n, size_crossmarker


def _stat(sd, scale_sd, mean=0.3, n=500):
    return {"n": n, "mean": mean, "sd": sd, "d_z": mean / sd, "ci_low": mean - 0.1, "ci_high": mean + 0.1,
            "share_negative": 0.4, "scale_sd": scale_sd, "scaled_mean": mean / scale_sd if scale_sd else float("nan"),
            "scaled_ci_low": 0.0, "scaled_ci_high": 1.0}


def _summary():
    group = lambda k: {"disparity": {"sex": _stat(1.0 * k, 2.0), "age": _stat(3.0 * k, 2.0),
                                     "intersection": _stat(2.0 * k, 2.0)},
                       "additivity_gap": _stat(1.0 * k, 2.0),
                       "interactions": {"sex_x_age": _stat(4.0 * k, 2.0)}}
    return {"model": "Skywork/Skywork-Reward-V2-Qwen3-0.6B", "domain": "credit",
            "selection": {"available_strong": 459, "available_weak": 194},
            "metrics": {"explicit": {"baseline": {"margins": {"D": {"strong": group(1), "weak": group(1)}}}}}}


def test_required_n_by_hand():
    assert required_n(1.0, 0.2, 1) == 197           # ((1.95996 + 0.84162) / 0.2)^2 = 196.2
    assert required_n(1.0, 0.1, 1) == 785           # halving delta quadruples n
    assert required_n(1.0, 0.2, 4) > required_n(1.0, 0.2, 1)
    assert required_n(float("nan"), 0.2, 1) is None and required_n(None, 0.2, 1) is None


def test_precision_is_sd_over_scale_sd_and_nothing_else():
    assert precision(_stat(1.5, 3.0)) == {"n": 500, "r": 0.5}
    assert precision({"n": 3, "sd": 1.0})["r"] is None           # unmarked control not scored
    assert precision(_stat(1.0, 0.0))["r"] is None


def test_the_binding_contrast_sets_n_and_the_pool_caps_it():
    rows = size_crossmarker(_summary(), deltas=[0.2], families=[1])
    main = {r["group"]: r for r in rows if r["family"] == "main"}
    # the largest r among the main contrasts: age, 3.0 / 2.0 = 1.5
    assert main["strong"]["binding_contrast"] == "disparity:age" and main["strong"]["r"] == pytest.approx(1.5)
    assert main["strong"]["required"]["delta=0.2,m=1"] == required_n(1.5, 0.2, 1) == 442
    assert main["strong"]["exceeds_available"]["delta=0.2,m=1"] is False      # 442 <= 459
    assert main["weak"]["exceeds_available"]["delta=0.2,m=1"] is True         # 442 > 194
    inter = [r for r in rows if r["family"] == "interactions"]
    assert {r["binding_contrast"] for r in inter} == {"interaction:sex_x_age"}


def test_blinded_to_every_mean():
    before = size_crossmarker(_summary(), deltas=[0.05, 0.1], families=[1, 12])
    changed = copy.deepcopy(_summary())
    for group in changed["metrics"]["explicit"]["baseline"]["margins"]["D"].values():
        for stat in list(group["disparity"].values()) + [group["additivity_gap"]]:
            stat.update(mean=-5.0, d_z=-9.0, ci_low=-6.0, ci_high=-4.0, scaled_mean=-2.0, share_negative=1.0)
    assert size_crossmarker(changed, deltas=[0.05, 0.1], families=[1, 12]) == before


def test_probe_answer_is_the_largest_over_directions_and_models():
    curve = lambda model, a, b: {"domain": "credit", "model": model,
                                 "directions": {"explicit/sex": {"rule": {"smallest_passing_n": a}},
                                                "proxy/sex": {"rule": {"smallest_passing_n": b}}}}
    assert probe_answers([curve("small", 50, 100), curve("8b", 75, 150)])["credit"]["answer"] == 150
    assert probe_answers([curve("small", 50, None)])["credit"]["answer"] is None
