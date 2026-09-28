"""`runners/export_paper_numbers.py`: the direct arm's auto-influence macros come from run_battery's explicit
cells. Only `collect` runs, on synthetic files in a temporary directory; nothing is written."""

from __future__ import annotations

import json

from runners.export_paper_numbers import collect


def _cell(axis, encoding, base, null, gap):
    return {"axis": axis, "encoding": encoding, "n_eval": 200, "baseline": {"auto_influence": base, "mean_gap": gap, "n_examples": 200},
            "nulled": {"auto_influence": null, "mean_gap": 0.0, "n_examples": 200}}


def test_auto_influence_macros_read_the_battery_cells(tmp_path):
    (tmp_path / "battery_credit_qwen06.json").write_text(json.dumps({"cells": [
        _cell("sex", "proxy", 0.9, 0.9, 9.0), _cell("sex", "explicit", 1.0, 0.06, -0.25),
        _cell("intersection", "explicit", 0.5, 0.1, 0.125)]}))
    macros, missing = collect(tmp_path)
    assert (macros["autoInflSexCreditbase"], macros["autoInflSexCreditnull"], macros["meanGapSexCredit"]) == (
        "1.00", "0.06", "-0.25")
    assert macros["autoInflIntersectionCreditbase"] == "0.50"
    assert "battery_cv_qwen06.json" in missing and not any("credit" in f for f in missing)


def test_the_eval_size_and_model_come_from_the_cv_battery(tmp_path):
    (tmp_path / "battery_cv_qwen06.json").write_text(json.dumps({"model": "some/rm", "cells": [
        _cell("sex", "explicit", 1.0, 0.1, 0.5), _cell("intersection", "explicit", 1.0, 0.1, 0.5)]}))
    macros, _ = collect(tmp_path)
    assert (macros["nEval"], macros["modelSmall"]) == ("200", "some/rm")
