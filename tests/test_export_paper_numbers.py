"""`runners/export_paper_numbers.py`: the result names come from the runners, every signed statistic carries its
interval, pre-review files are skipped, the sources are listed, and the write-up breakage is reported. Only
`collect` and the helpers run, on synthetic files (shaped like the reviewed runners' outputs) in a temporary
directory; nothing is written to the write-up."""

from __future__ import annotations

import json
import re

import pytest

from runners.export_paper_numbers import collect, header, result_files, tex_breakage

MODEL = "org/Tiny-RM"


def _iv(est, lo=None, hi=None):
    return {"estimate": est, "ci_low": est - 0.1 if lo is None else lo, "ci_high": est + 0.1 if hi is None else hi}


def _meta(commit="c0ffee", dirty=False):
    return {"code": {"git_commit": commit, "git_dirty": dirty}, "config": {"model_revision": "abc123"},
            "data": {"pairs.jsonl": {"sha256": "f" * 64}}}


def _cell(axis, encoding, base_ai, null_ai, gap, null_gap=0.0, n=200):
    side = lambda ai, g: {"auto_influence": ai, "mean_gap": g, "n_examples": n,
                          "intervals": {"mean_gap": _iv(g), "auto_influence": _iv(ai)}}
    return {"axis": axis, "encoding": encoding, "n_eval": n, "baseline": side(base_ai, gap),
            "nulled": side(null_ai, null_gap),
            "baseline_vs_nulled": {"nulled_minus_baseline": {"mean_gap_change": _iv(null_gap - gap)}}}


def _write(tmp_path, key, data, meta=True):
    name = result_files(MODEL)[key]
    (tmp_path / name).write_text(json.dumps({**({"meta": _meta()} if meta else {}), **data}))
    return name


def test_the_names_are_the_runners():
    files = result_files(MODEL)
    assert files["battery_cv"] == "battery_cv_Tiny-RM.json"
    assert files["battery_credit"] == "battery_credit_Tiny-RM.json"
    assert files["battery_education"] == "battery_education_asap2_Tiny-RM.json"
    assert files["battery_education_stage"] == "battery_education_asap2_stage_Tiny-RM.json"
    assert files["maineffect_implausible"] == "maineffect_edupos_asap2_implausible_Tiny-RM.json"
    assert files["decision_cv"] == "decision_cv_Tiny-RM_explicit.json"
    assert (files["reasoning"], files["reasoning_probe"], files["erasure"]) == (
        "reasoning_cv_Tiny-RM.json", "reasoning_probe_cv_Tiny-RM.json", "erasure_cv_Tiny-RM.json")
    assert files["additivity_credit"] == "additivity_credit_Tiny-RM_explicit.json"


def test_the_direct_arm_with_intervals_and_pre_review_files_are_skipped(tmp_path):
    _write(tmp_path, "battery_credit", {"cells": [
        _cell("sex", "proxy", 0.9, 0.9, 9.0), _cell("sex", "explicit", 1.0, 0.06, -0.25, 0.01),
        _cell("intersection", "explicit", 0.5, 0.1, 0.125)]})
    old = _write(tmp_path, "battery_cv", {"cells": [_cell("sex", "explicit", 1.0, 0.1, 0.5)]}, meta=False)
    m, report = collect(tmp_path, MODEL)
    assert (m["autoInflSexCreditbase"], m["autoInflSexCreditnull"]) == ("1.00", "0.06")
    assert (m["meanGapSexCredit"], m["meanGapSexCreditLo"], m["meanGapSexCreditHi"]) == ("-0.25", "-0.35", "-0.15")
    assert m["meanGapSexCreditnull"] == "+0.01" and m["meanGapChangeSexCredit"] == "+0.26"
    assert report["skipped"] == [old] and "autoInflSexCVbase" not in m and "nEval" not in m
    assert [s["file"] for s in report["sources"]] == ["battery_credit_Tiny-RM.json"]
    assert report["sources"][0]["model_revision"] == "abc123"


def test_the_hiring_battery_gives_the_eval_size_and_the_cells(tmp_path):
    _write(tmp_path, "battery_cv", {"cells": [_cell("sex", "explicit", 1.0, 0.1, 0.5, n=180),
                                              _cell("intersection", "explicit", 1.0, 0.1, 0.5),
                                              _cell("family_status", "proxy", 0.4, 0.1, 0.2)]})
    m, report = collect(tmp_path, MODEL)
    assert (m["nEval"], m["nEvalBattery"], m["modelSmall"]) == ("180", "180", MODEL)
    assert m["batteryFamilyProxyCV"] == "0.40" and m["batteryFamilyProxyCVGapHi"] == "+0.30"
    assert "battery_cv_Tiny-RM.json: ('age', 'explicit')" in report["missing"]


def test_education_reads_the_factorial_and_the_stage_design(tmp_path):
    _write(tmp_path, "battery_education", {"cells": [_cell(a, e, 0.5, 0.1, 0.1) for a in
                                                     ("sex", "ethnicity", "economic_status") for e in ("explicit", "proxy")]})
    _write(tmp_path, "battery_education_stage", {"cells": [_cell("grade_level", e, 0.7, 0.2, -0.3)
                                                           for e in ("explicit", "proxy")]})
    m, _ = collect(tmp_path, MODEL)
    assert m["eduEconomicExplicitAsapTwo"] == "0.50" and m["eduGradeProxyAsapTwo"] == "0.70"
    assert m["eduGradeProxyAsapTwonull"] == "0.20" and m["eduGradeProxyAsapTwoGapChangeLo"] == "+0.20"
    assert m["eduSexExplicitAsapTwonull"] == "0.10" and "eduSexProxyAsapTwonull" not in m


def test_the_other_arms_carry_their_intervals(tmp_path):
    axes = ["pos_sex", "pos_intersection"]
    _write(tmp_path, "maineffect_plausible", {"n_essays": 622, "results": [
        {"axis": a, "intervals": {"identity_gap": _iv(0.13), "main_effect": _iv(0.2)}} for a in axes]})
    _write(tmp_path, "decision_cv", {"results": [{"axis": "sex", "intervals": {"baseline": {
        "discriminatory_win_rate": _iv(0.3), "mean_gap_fair_minus_disc": _iv(1.5),
        "disc_win_rate_vs_neutral_decline": _iv(0.4)}}}]})
    effects = lambda c: {"correctness_effect": _iv(c), "conclusion_effect": _iv(0.1)}
    _write(tmp_path, "reasoning", {"results": [{"premise": p, "intervals": {"baseline": effects(1.0)}}
                                               for p in ("parental_leave", "intersection", "commute")],
                                   "versus_control": {"parental_leave": effects(-0.2), "intersection": effects(-0.7)}})
    acc = {"paired_acc_heldout": _iv(0.8, 0.75, 0.85)}
    _write(tmp_path, "reasoning_probe", {
        "results": [{"premise": p, "directions": {"correctness": acc, "conclusion": acc}}
                    for p in ("parental_leave", "commute")],
        "transfer": {"cosines": {"correctness": 0.94, "conclusion": 0.9},
                     "commute_on_parental_leave": {"correctness": {"intervals": {
                         "baseline": effects(1.0), "nulled": effects(0.2), "nulled_minus_baseline": effects(-0.8)}}}}})
    row = lambda gap: {"linear_acc": 0.5, "mlp_acc": 0.5 + gap, "intervals": {"mlp_above_chance": _iv(gap)}}
    _write(tmp_path, "erasure", {"results": [{"premise": "parental_leave", "concept": "correctness",
                                              "rows": {k: row(0.3) for k in ("none", "diffmean", "leace")},
                                              "lexical_control": {k: row(0.0) for k in ("none", "diffmean", "leace")}}]})
    _write(tmp_path, "additivity_credit", {"threeway": {"cos_intersection_vs_marginal_sum": 0.9994,
                                                        "residual_share_debiased": 0.023, "noise_floor_share": 0.025,
                                                        "sign_flip_p": 0.0001}})
    m, _ = collect(tmp_path, MODEL)
    assert (m["standpointGapSexAsapTwo"], m["standpointGapSexAsapTwoLo"]) == ("+0.13", "+0.03")
    assert m["standpointMainEffectAsapTwoHi"] == "+0.30" and m["nEvalStandpointAsapTwo"] == "622"
    assert (m["decisionDiscWinSex"], m["decisionDiscWinNeutralDeclineSexLo"]) == ("0.30", "0.30")
    assert m["reasonCorrectnessVsCommuteIntersection"] == "-0.70" and m["reasonConclusionCommuteHi"] == "+0.20"
    assert (m["probeCorrAccHeldoutPL"], m["probeCorrAccHeldoutPLLo"]) == ("80\\%", "75\\%")
    assert m["cosCorrectnessPLvsCommute"] == "0.94" and m["transferCommuteToPLchange"] == "-0.80"
    assert m["erasureMlpLeaceCorrPL"] == "80\\%" and m["erasureMlpAboveChanceLeaceCorrPLlexical"] == "+0.00"
    assert (m["additivityCosCredit"], m["additivityResidualCredit"], m["additivityThreeWayPCredit"]) == (
        "0.999", "0.023", "0.0001")
    assert all(re.fullmatch(r"[A-Za-z]+", name) for name in m)


def test_the_header_lists_the_sources_and_warns_on_mixed_commits(tmp_path):
    report = {"sources": [{"file": "a.json", "git_commit": "c1", "git_dirty": False, "model_revision": "r",
                           "data": {"pairs.jsonl": "f" * 64}},
                          {"file": "b.json", "git_commit": "c2", "git_dirty": True, "model_revision": "r", "data": {}}]}
    text = "\n".join(header(tmp_path, report))
    assert "a.json | c1 | False | r | pairs.jsonl ffffffffffff" in text
    assert "WARNING: the sources come from 2 code commit(s) and a dirty tree" in text


def test_the_write_up_breakage_is_reported(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "numbers.tex").write_text("\\newcommand{\\eduSexExplicitPersuade}{0.5}\n\\newcommand{\\keep}{1}\n"
                                        "\\newcommand{\\gone}{2}\n")
    (shared / "numbers_manual.tex").write_text("\\newcommand{\\additivityCosCV}{0.66}\n"
                                               "\\markstale{eduSexExplicitPersuade}{\\dag}\n\\markstale{keep}{\\S}\n")
    (tmp_path / "notes.tex").write_text("value \\eduSexExplicitPersuade{} and \\keep{} and \\goneish{}\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "x.tex").write_text("\\gone")
    b = tex_breakage(shared / "numbers.tex", {"keep": "1", "additivityCosCV": "0.9"}, tmp_path)
    assert b["disappearing"] == {"eduSexExplicitPersuade": ["notes.tex"]} and b["disappearing_unused"] == ["gone"]
    assert b["defined_twice"] == ["additivityCosCV"]
    assert b["stale_marks_without_macro"] == ["eduSexExplicitPersuade"]


def test_main_writes_only_the_out_file(tmp_path, monkeypatch):
    from runners import export_paper_numbers

    results, shared = tmp_path / "results", tmp_path / "writeup" / "shared"
    results.mkdir()
    shared.mkdir(parents=True)
    _write(results, "battery_credit", {"cells": [_cell("sex", "explicit", 1.0, 0.06, -0.25)]})
    monkeypatch.setattr("sys.argv", ["export_paper_numbers.py", "--results-dir", str(results), "--model", MODEL,
                                     "--out", str(shared / "numbers.tex")])
    export_paper_numbers.main()
    text = (shared / "numbers.tex").read_text()
    assert "\\newcommand{\\meanGapSexCreditLo}{-0.35}" in text and "battery_credit_Tiny-RM.json | c0ffee" in text
    assert [p.name for p in tmp_path.rglob("*.tex")] == ["numbers.tex"]


@pytest.mark.parametrize("value, expect", [(float("nan"), "--"), (None, "--"), (0.123, "0.12")])
def test_undefined_values_print_as_a_dash(value, expect):
    from runners.export_paper_numbers import _fmt

    assert _fmt(value) == expect
