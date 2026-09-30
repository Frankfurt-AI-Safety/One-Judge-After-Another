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


def _curve(model, a, b, largest=300):
    rule = lambda n: {"smallest_passing_n": n, "answer": largest if n is None else n}
    return {"domain": "credit", "model": model,
            "directions": {"explicit/sex": {"rule": rule(a)}, "proxy/sex": {"rule": rule(b)}}}


def test_probe_answer_is_the_largest_over_directions_and_models():
    assert probe_answers([_curve("small", 50, 100), _curve("8b", 75, 150)])["credit"]["answer"] == 150


def test_a_direction_failing_at_the_largest_n_is_reported_and_keeps_it():
    # the pre-stated rule: a direction that fails even at the largest N is reported, and the domain keeps that N
    d = probe_answers([_curve("small", 50, None, largest=500)])["credit"]
    assert d["answer"] == 500 and d["failing"] == {"small": ["proxy/sex"]}


def test_the_three_way_interaction_is_outside_the_rule():
    # the rule sizes the two-way interactions; a larger three-way r must not become the binding contrast
    summary = _summary()
    for group in summary["metrics"]["explicit"]["baseline"]["margins"]["D"].values():
        group["interactions"]["three_way"] = _stat(20.0, 2.0)
    inter = [r for r in size_crossmarker(summary, [0.2], [1]) if r["family"] == "interactions"]
    assert {r["binding_contrast"] for r in inter} == {"interaction:sex_x_age"}
    assert all(r["outside_rule"] == {"interaction:three_way": 10.0} for r in inter)


def test_an_undefined_r_is_listed_never_dropped():
    summary = _summary()
    for group in summary["metrics"]["explicit"]["baseline"]["margins"]["D"].values():
        group["interactions"] = {"sex_x_age": {"n": 5, "sd": 0.0, "scale_sd": 0.0}}
    inter = [r for r in size_crossmarker(summary, [0.2], [1]) if r["family"] == "interactions"]
    assert len(inter) == 2 and all(r["r"] is None and r["unsized"] == ["interaction:sex_x_age"] for r in inter)
    assert all(v is None for r in inter for v in r["required"].values())


def test_the_answer_is_the_maximum_over_models():
    from runners.pilot_sizing import max_over_models

    small = _summary()
    large = copy.deepcopy(small)
    large["model"] = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    large["selection"]["available_weak"] = 180
    for group in large["metrics"]["explicit"]["baseline"]["margins"]["D"].values():
        group["disparity"]["age"] = _stat(4.0, 2.0)                    # r 2.0 against the small model's 1.5
    rows = size_crossmarker(small, [0.2], [1]) + size_crossmarker(large, [0.2], [1])
    best = {(r["group"], r["family"]): r for r in max_over_models(rows)}
    assert best[("strong", "main")]["required"]["delta=0.2,m=1"] == required_n(2.0, 0.2, 1)
    assert best[("weak", "main")]["available"] == 180 and len(best[("weak", "main")]["models"]) == 2


def test_main_refuses_an_unknown_input_and_records_its_inputs(tmp_path, monkeypatch):
    import hashlib
    import json

    from runners import pilot_sizing

    good, bad = tmp_path / "cm.json", tmp_path / "scrub.json"
    good.write_text(json.dumps({**_summary(), "meta": {"code": {"git_commit": "c0ffee", "git_dirty": False},
                                                        "config": {"model_revision": "abc123"}}}))
    bad.write_text(json.dumps({"results": {}}))
    out = tmp_path / "sizing.json"
    monkeypatch.setattr("sys.argv", ["pilot_sizing.py", "--inputs", str(good), str(bad), "--out", str(out)])
    with pytest.raises(SystemExit, match="neither"):
        pilot_sizing.main()
    monkeypatch.setattr("sys.argv", ["pilot_sizing.py", "--inputs", str(good), "--out", str(out)])
    pilot_sizing.main()
    record = json.loads(out.read_text())["inputs"][0]
    assert record["sha256"] == hashlib.sha256(good.read_bytes()).hexdigest()
    assert record["code"] == {"git_commit": "c0ffee", "git_dirty": False} and record["model_revision"] == "abc123"


# --------------------------------------------------------------------------- comparative -------------
def _comparative():
    entry = lambda sd, coded_sd: {"merit": {"marker_effect": _stat(sd, 2.0, n=40)},
                                  "coded": {"marker_effect": _stat(coded_sd, 2.0, n=40)}}
    block = lambda k: {"n_pairs": 40, "sex": entry(1.0 * k, 9.0), "intersection": entry(3.0 * k, 9.0),
                       "unmarked": {"position_effect": _stat(5.0, 2.0)}}
    report = {"strong_strong": {"anchor_pool": 351, "partner_pool": 351}, "strong_weak":
              {"anchor_pool": 200, "partner_pool": 95}, "weak_weak": {"anchor_pool": 138, "partner_pool": 138}}
    return {"model": "M", "domain": "credit", "pairing": report, "pairs": [], "selection": {},
            "metrics": {"explicit": {"baseline": {"strong_strong": block(1), "strong_weak": block(2),
                                                  "weak_weak": block(1), "all": block(1)}}}}


def test_comparative_pairs_from_the_merit_marker_effects():
    from runners.pilot_sizing import size_comparative

    rows = {r["group"]: r for r in size_comparative(_comparative(), deltas=[0.2], families=[1])}
    assert set(rows) == {"strong_strong", "strong_weak", "weak_weak"}             # not the pooled "all"
    ss, sw = rows["strong_strong"], rows["strong_weak"]
    assert ss["binding_contrast"] == "marker_effect:intersection" and ss["r"] == 1.5 and ss["n_pilot"] == 40
    assert sw["r"] == 3.0 and sw["required"]["delta=0.2,m=1"] == required_n(3.0, 0.2, 1)
    # the coded response sizes nothing; its r is listed
    assert ss["outside_rule"] == {"marker_effect:sex:coded": 4.5, "marker_effect:intersection:coded": 4.5}
    # capacity before the checks: half the pool, or the smaller side of strong-weak
    assert (ss["available"], sw["available"], rows["weak_weak"]["available"]) == (175, 95, 69)
    assert sw["exceeds_available"]["delta=0.2,m=1"] == (required_n(3.0, 0.2, 1) > 95)


def test_comparative_sizing_is_blinded():
    from runners.pilot_sizing import size_comparative

    before = size_comparative(_comparative(), deltas=[0.1], families=[1, 4])
    changed = copy.deepcopy(_comparative())
    for block in changed["metrics"]["explicit"]["baseline"].values():
        for axis in ("sex", "intersection"):
            for kind in ("merit", "coded"):
                block[axis][kind]["marker_effect"].update(mean=-5.0, d_z=-9.0, ci_low=-6.0, ci_high=-4.0,
                                                          scaled_mean=-2.0)
    assert size_comparative(changed, deltas=[0.1], families=[1, 4]) == before


def test_main_tells_a_comparative_summary_from_a_cross_marker_one(tmp_path, monkeypatch):
    import json

    from runners import pilot_sizing

    cm, cmp = tmp_path / "cm.json", tmp_path / "cmp.json"
    cm.write_text(json.dumps(_summary()))
    cmp.write_text(json.dumps(_comparative()))
    out = tmp_path / "sizing.json"
    monkeypatch.setattr("sys.argv", ["pilot_sizing.py", "--inputs", str(cm), str(cmp), "--out", str(out)])
    pilot_sizing.main()
    result = json.loads(out.read_text())
    assert [i["kind"] for i in result["inputs"]] == ["cross_marker", "comparative"]
    assert {r["group"] for r in result["pairs_per_pairing"]} == {"strong_strong", "strong_weak", "weak_weak"}
    assert {r["group"] for r in result["records_per_group"]} == {"strong", "weak"}
    assert len(result["pairs_per_pairing_max_over_models"]) == 3
