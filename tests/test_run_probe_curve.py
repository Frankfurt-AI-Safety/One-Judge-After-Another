"""The pilot's probe-size curve (`runners/run_probe_curve.py`): nested prefixes, one fixed eval set, and the
pre-stated rule that turns a curve into ``probe_records``."""

from __future__ import annotations

import pytest

from probes.probe import build_probe_direction, get_rewards_both
from runners.run_probe_curve import direction_curve, per_record_gaps, probe_rule
from substrates.domains import get_domain
from tests.test_run_cross_marker import _model, _tokenizer, manifest  # noqa: F401  (fixture)

DOM = get_domain("credit")


def _curve(manifest, grid, **kw):
    args = dict(grid=grid, max_eval=None, batch_size=16, device="cpu", max_length=1024, n_boot=50,
                seed=42, split_seed=42)
    args.update(kw)
    return direction_curve(_model(), _tokenizer(), DOM.dataset_cls, str(manifest), "sex", "explicit", **args)


def test_prefixes_are_nested_and_the_eval_set_is_fixed(manifest, monkeypatch):
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    out = _curve(manifest, [4, 8, 12])
    assert [p["n"] for p in out["curve"]] == [4, 8, 12]
    # 16 records: the largest probe set takes 12 (6 strong / 6 weak), the eval set is the other 4
    assert out["eval_records"] == 4
    assert out["curve"][-1]["split"]["probe_strata"] == {"False": 6, "True": 6}
    assert [p["split"]["probe_records"] for p in out["curve"]] == [4, 8, 12]
    assert out["curve"][-1]["cos_to_max"] == pytest.approx(1.0, abs=1e-6)
    assert set(out["rule"]["passes"]) == {4, 8, 12}


def test_the_largest_point_is_the_plain_direction_on_the_fixed_eval_set(manifest, monkeypatch):
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    out = _curve(manifest, [4, 12])
    model, tok = _model(), _tokenizer()
    ds = DOM.dataset_cls(str(manifest), axis="sex", encoding="explicit", split_seed=42, probe_records=12)
    u, _ = build_probe_direction(model, tok, ds.get_probe_pairs(tok), batch_size=16, device="cpu",
                                 max_length=1024)
    examples = ds.get_eval_examples(tok)
    texts = [e.texts["a"] for e in examples] + [e.texts["b"] for e in examples]
    base, nulled = get_rewards_both(model, tok, texts, u, batch_size=16, device="cpu", max_length=1024,
                                    show_progress=False)
    n = len(examples)
    by_record = per_record_gaps(examples, (nulled[:n] - nulled[n:]).tolist())
    expect = sum(sum(abs(x) for x in g) / len(g) for g in by_record.values()) / len(by_record)
    assert out["curve"][-1]["nulled_abs_gap"]["mean"] == pytest.approx(expect, abs=1e-5)
    base_gap = per_record_gaps(examples, (base[:n] - base[n:]).tolist())
    assert out["baseline"]["gap"]["mean"] == pytest.approx(
        sum(sum(g) / len(g) for g in base_gap.values()) / len(base_gap), abs=1e-5)


def test_a_grid_the_records_cannot_fill_raises(manifest, monkeypatch):
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    with pytest.raises(ValueError, match="too few records"):
        _curve(manifest, [4, 16])


# --------------------------------------------------------------------------- the rule ----------------
def _point(n, split_half, gap, ci=(0.9, 1.1)):
    return {"n": n, "split_half": split_half,
            "nulled_abs_gap": {"mean": gap, "ci_low": ci[0], "ci_high": ci[1]}}


def test_rule_needs_stability_and_a_settled_nulled_gap():
    curve = [_point(25, 0.80, 1.00), _point(50, 0.92, 1.30), _point(100, 0.95, 1.05), _point(300, 0.99, 1.00)]
    rule = probe_rule(curve)
    # 25 fails on stability, 50 on the nulled gap (outside [0.9, 1.1]); from 100 on everything passes
    assert rule["passes"] == {25: False, 50: False, 100: True, 300: True}
    assert rule["smallest_passing_n"] == 100


def test_rule_takes_the_n_from_which_every_larger_point_passes():
    curve = [_point(25, 0.95, 1.0), _point(50, 0.85, 1.0), _point(100, 0.95, 1.0), _point(300, 0.97, 1.0)]
    assert probe_rule(curve)["smallest_passing_n"] == 100


def test_rule_reports_none_when_the_largest_point_is_unstable():
    curve = [_point(25, 0.5, 1.0), _point(300, 0.85, 1.0)]
    assert probe_rule(curve)["smallest_passing_n"] is None
    assert probe_rule(curve, threshold=0.8)["smallest_passing_n"] == 300
    assert probe_rule([_point(25, float("nan"), 1.0), _point(300, 0.95, 1.0)])["passes"][25] is False
