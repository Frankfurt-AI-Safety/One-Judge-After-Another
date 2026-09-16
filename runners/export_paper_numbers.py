#!/usr/bin/env python3
"""
Single source of truth for paper numbers: read the experiment results JSONs and emit a LaTeX file of
`\\newcommand` macros, so the paper / working notes / progress reports never transcribe a number by hand.

Re-run after any experiment, then recompile the LaTeX:
    python experiments/export_paper_numbers.py \
        --results-dir artifacts/results/demographic \
        --out "/Users/.../greenTeam/OneJudgeAfterAnother/shared/numbers.tex"

Only stats that are serialized to JSON are covered here; anything else lives in shared/numbers_manual.tex
(hand-maintained, clearly marked). A referenced-but-missing macro is a LOUD LaTeX error at compile time —
that is intentional (catches renamed/rerun-needed metrics instead of shipping a wrong value).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _fmt(x: float, dp: int = 2, sign: bool = False, pct: bool = False) -> str:
    if pct:
        return f"{100 * x:.0f}\\%"
    s = f"{x:+.{dp}f}" if sign else f"{x:.{dp}f}"
    return s


def _load(results_dir: Path, name: str):
    p = results_dir / name
    return json.loads(p.read_text()) if p.exists() else None


def collect(results_dir: Path) -> Tuple[Dict[str, str], List[str]]:
    """Return (macros, missing_files). Each macro maps a LaTeX command name (letters only) to a value."""
    m: Dict[str, str] = {}
    missing: List[str] = []

    def need(name: str):
        d = _load(results_dir, name)
        if d is None:
            missing.append(name)
        return d

    # --- auto-influence (run_experiment) — baseline/nulled auto_influence + mean_gap ------------------
    ai = {
        "SexCV": "demographic_cv_sex_qwen06_results.json",
        "IntersectionCV": "demographic_cv_intersection_qwen06_results.json",
        "SexCredit": "demographic_credit_sex_qwen06_results.json",
        "IntersectionCredit": "demographic_credit_intersection_qwen06_results.json",
    }
    for tag, fname in ai.items():
        d = need(fname)
        if d:
            m[f"autoInfl{tag}base"] = _fmt(d["baseline"]["auto_influence"])
            m[f"autoInfl{tag}null"] = _fmt(d["nulled"]["auto_influence"])
            m[f"meanGap{tag}"] = _fmt(d["baseline"]["mean_gap"], sign=True)

    # --- robustness battery (CV): selected explicit/proxy cells --------------------------------------
    d = need("battery_cv_qwen06.json")
    if d:
        cells = {(c["axis"], c["encoding"]): c for c in d["cells"]}
        want = {"batterySexProxyCV": ("sex", "proxy"), "batteryAgeExplicitCV": ("age", "explicit"),
                "batteryAgeProxyCV": ("age", "proxy"), "batteryFamilyExplicitCV": ("family_status", "explicit"),
                "batteryFamilyProxyCV": ("family_status", "proxy")}
        for macro, key in want.items():
            if key in cells:
                m[macro] = _fmt(cells[key]["baseline"]["auto_influence"])
        if d["cells"]:
            m["nEvalBattery"] = str(d["cells"][0]["n_eval"])

    # --- education (A1 header-proxy) battery on the real-essay corpora -------------------------------
    edu_want = {"eduSexExplicit": ("sex", "explicit"), "eduSexProxy": ("sex", "proxy"),
                "eduEthnicityExplicit": ("ethnicity", "explicit"), "eduEthnicityProxy": ("ethnicity", "proxy"),
                "eduGradeExplicit": ("grade_level", "explicit"), "eduGradeProxy": ("grade_level", "proxy")}
    edu_headline = {"eduSexExplicit", "eduGradeProxy"}  # also emit the nulled value for these two
    for tag, fname in (("Asap", "battery_education_asap_qwen06.json"),
                       ("Persuade", "battery_education_persuade_qwen06.json")):
        d = need(fname)
        if d:
            cells = {(c["axis"], c["encoding"]): c for c in d["cells"]}
            for macro, key in edu_want.items():
                if key in cells:
                    m[f"{macro}{tag}"] = _fmt(cells[key]["baseline"]["auto_influence"])
                    if macro in edu_headline:
                        m[f"{macro}{tag}null"] = _fmt(cells[key]["nulled"]["auto_influence"])

    # --- cross-influence: acc_baseline (interpretability) + the n it actually rests on ---------------
    # NOTE: Credit reads crossinf_qwen06.json (n=300), NOT crossinf_credit_qwen06.json (n=50) — the same
    # experiment was run at two sample sizes and the better-powered one is what the prose quotes.
    for tag, fname in (("CV", "crossinf_cv_qwen06.json"),
                       ("Credit", "crossinf_qwen06.json"),
                       ("EduAsap", "crossinf_education_asap_explicit_qwen06.json"),
                       ("EduPersuade", "crossinf_education_persuade_explicit_qwen06.json")):
        d = need(fname)
        if d:
            r0 = d["results"][0]
            m[f"crossAccBaseline{tag}"] = _fmt(r0["baseline"]["acc_baseline"])
            m[f"crossN{tag}"] = str(r0["n_pairs"])

    # --- standpoint credibility (A2): identity gap vs a no-standpoint neutral baseline --------------
    sp_axes = {"Sex": "pos_sex", "Race": "pos_race", "Class": "pos_class", "Origin": "pos_origin",
               "Intersection": "pos_intersection", "Control": "pos_control",
               "Hobby": "pos_ctrl_hobby", "Pet": "pos_ctrl_pet", "Region": "pos_ctrl_region"}
    for tag, fname in (("Persuade", "maineffect_edupos_persuade_qwen06.json"),
                       ("Asap", "maineffect_edupos_asap_qwen06.json")):
        d = need(fname)
        if d:
            by = {r["axis"]: r for r in d["results"]}
            for name, axis in sp_axes.items():
                if axis in by:
                    m[f"standpointGap{name}{tag}"] = _fmt(by[axis]["identity_gap"], sign=True)
            if "pos_intersection" in by:
                m[f"standpointMainEffect{tag}"] = _fmt(by["pos_intersection"]["main_effect"], sign=True)
                m[f"nEvalStandpoint{tag}"] = str(by["pos_intersection"]["n"])

    # --- decision-response (CV): discriminatory win-rate + fair-minus-disc gap -----------------------
    d = need("decision_cv_qwen06.json")
    if d:
        by = {r["axis"]: r["baseline"] for r in d["results"]}
        amap = {"Sex": "sex", "Age": "age", "Family": "family_status", "Intersection": "intersection"}
        for tag, axis in amap.items():
            if axis in by:
                m[f"decisionDiscWin{tag}"] = _fmt(by[axis]["discriminatory_win_rate"])
                m[f"decisionGapFairDisc{tag}"] = _fmt(by[axis]["mean_gap_fair_minus_disc"], sign=True)

    # --- reasoning-flip 2x2 (CV): correctness/conclusion main effects -------------------------------
    d = need("reasoning_cv_qwen06.json")
    if d:
        by = {r["premise"]: r["baseline"] for r in d["results"]}
        pmap = {"PL": "parental_leave", "Intersection": "intersection", "Commute": "commute"}
        for tag, prem in pmap.items():
            if prem in by:
                m[f"reasonCorrectness{tag}"] = _fmt(by[prem]["correctness_effect"], sign=True)
                m[f"reasonConclusion{tag}"] = _fmt(by[prem]["conclusion_effect"], sign=True)

    # --- reasoning probe (held-out) + cross-premise transfer ----------------------------------------
    d = need("reasoning_probe_cv_qwen06.json")
    if d:
        by = {r["premise"]: r for r in d["results"]}
        for tag, prem in (("PL", "parental_leave"), ("Commute", "commute")):
            if prem in by:
                m[f"probeCorrAccHeldout{tag}"] = _fmt(by[prem]["correctness_probe"]["acc_heldout"], pct=True)
                m[f"probeConclAccHeldout{tag}"] = _fmt(by[prem]["conclusion_probe"]["acc_heldout"], pct=True)
        t = d.get("transfer", {})
        if "cos_correctness_parental_vs_commute" in t:
            m["cosCorrectnessPLvsCommute"] = _fmt(t["cos_correctness_parental_vs_commute"])
        tr = t.get("commute_dir__on__parental_leave_eval")
        if tr:
            m["transferCommuteToPLbase"] = _fmt(tr["correctness_effect_base"], sign=True)
            m["transferCommuteToPLnull"] = _fmt(tr["correctness_effect_null"], sign=True)

    # --- LEACE / non-linear-probe erasure (CV): the entanglement headline ---------------------------
    d = need("erasure_cv_qwen06.json")
    if d:
        rows = {(r["premise"], r["concept"]): r["rows"] for r in d["results"]}
        key = ("parental_leave", "correctness")
        if key in rows:
            m["erasureLinearNoneCorrPL"] = _fmt(rows[key]["none"]["linear_acc"], pct=True)
            m["erasureLinearLeaceCorrPL"] = _fmt(rows[key]["leace"]["linear_acc"], pct=True)
            m["erasureMlpLeaceCorrPL"] = _fmt(rows[key]["leace"]["mlp_acc"], pct=True)

    # --- additivity cosine (needs additivity_check.py --out; else see numbers_manual.tex) -----------
    for tag, fname in (("CV", "additivity_cv_qwen06.json"), ("Credit", "additivity_credit_qwen06.json")):
        p = results_dir / fname
        if p.exists():
            d = json.loads(p.read_text())
            m[f"additivityCos{tag}"] = _fmt(d["cos_intersection_vs_marginal_sum"])

    # --- meta -----------------------------------------------------------------------------------------
    any_ai = _load(results_dir, ai["SexCV"])
    if any_ai:
        m["nEval"] = str(any_ai["baseline"]["n_examples"])
        m["modelSmall"] = str(any_ai.get("config", {}).get("model_path", "Skywork-Reward-V2-Qwen3-0.6B"))
    return m, missing


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "artifacts/results/demographic")
    ap.add_argument("--out", type=Path, required=True, help="path to shared/numbers.tex in the paper repo")
    args = ap.parse_args()

    macros, missing = collect(args.results_dir)
    lines = [
        "% AUTO-GENERATED by experiments/export_paper_numbers.py -- DO NOT EDIT BY HAND.",
        f"% source: {args.results_dir}/*.json | generated: {date.today().isoformat()} "
        f"| code commit: {_git_hash()}",
        "% Hand-maintained / not-yet-serialized numbers live in shared/numbers_manual.tex.",
        "",
    ]
    for name in sorted(macros):
        lines.append(f"\\newcommand{{\\{name}}}{{{macros[name]}}}")
    if missing:
        lines += ["", "% missing result files (macros skipped): " + ", ".join(sorted(missing))]
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {len(macros)} macros -> {args.out}")
    if missing:
        print("skipped (missing JSON):", ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
