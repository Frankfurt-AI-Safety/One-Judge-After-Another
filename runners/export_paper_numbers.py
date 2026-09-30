#!/usr/bin/env python3
"""
Single source of truth for paper numbers: read the experiment results JSONs of one model and emit a LaTeX file of
`\\newcommand` macros, so the working notes / progress reports never transcribe a number by hand.

Re-run after any experiment, then recompile the LaTeX:
    python runners/export_paper_numbers.py \
        --results-dir artifacts/results/demographic \
        --model Skywork/Skywork-Reward-V2-Qwen3-0.6B \
        --out "/Users/.../greenTeam/OneJudgeAfterAnother/shared/numbers.tex"

**Inputs.** Each result file's name comes from its runner's own ``default_out`` (one definition), for ``--model``.
A file without ``meta`` predates the reviewed runners (2026-09-28/30) and is skipped and listed, never read. The
header of the written file lists every source with its code commit (and whether the tree was dirty), model commit
and data hashes, and warns when the sources come from different code commits.

**Intervals.** Every exported signed statistic also gets its 95% interval as ``<macro>Lo`` / ``<macro>Hi`` (the
reading rules: a nulled effect is read from the signed interval, e.g. ``mean_gap`` covering 0, and what nulling
removed from the change interval). ``auto_influence`` is folded (positive under noise alone), so it stays a point
value and is never the reading of a nulled effect; nor are cosines (length-type, no bootstrap interval).

Only stats that are serialized to JSON are covered here; anything else lives in shared/numbers_manual.tex
(hand-maintained, clearly marked). A referenced-but-missing macro is a LOUD LaTeX error at compile time —
that is intentional (catches renamed/rerun-needed metrics instead of shipping a wrong value). To see those
errors before compiling, the run compares the old ``--out`` file with the new macros and lists every macro that
disappears with the write-up files still using it, and every macro ``numbers_manual.tex`` also defines (a
"command already defined" error). It changes no ``.tex`` file but ``--out``.

The cross-marker design (the harm evidence) is not exported yet: which of its numbers are headlines is fixed with
the headline family.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
EDU_STAGE_PAIRS = "data/demographic/education/asap2_stage/pairs.jsonl"


def _fmt(x: Optional[float], dp: int = 2, sign: bool = False, pct: bool = False) -> str:
    if x is None or x != x:
        return "--"
    if pct:
        return f"{100 * x:.0f}\\%"
    return f"{x:+.{dp}f}" if sign else f"{x:.{dp}f}"


def result_files(model: str) -> Dict[str, str]:
    """The file name of every exported result for ``model``, from the runners' own ``default_out``."""
    from pairs.positionality import POSITIONED_AXES
    from runners import (
        run_additivity, run_battery, run_decision_response, run_positioned_maineffect, run_reasoning_erasure,
        run_reasoning_flip, run_reasoning_probe,
    )
    from substrates.domains import get_domain

    battery = lambda domain, source: run_battery.default_out(domain, source, model, [], []).name
    positioned = lambda group: run_positioned_maineffect.default_out(
        "asap2", group, model, positions=["conclusion"], stance="endorse", paraphrase="off",
        axes=list(POSITIONED_AXES), n_essays=None).name
    return {
        "battery_cv": battery("cv", get_domain("cv").default_pairs),
        "battery_credit": battery("credit", get_domain("credit").default_pairs),
        "battery_education": battery("education", get_domain("education").default_pairs),
        "battery_education_stage": battery("education", EDU_STAGE_PAIRS),
        "maineffect_plausible": positioned("plausible"),
        "maineffect_implausible": positioned("implausible"),
        "decision_cv": run_decision_response.default_out("cv", model, "explicit").name,
        "reasoning": run_reasoning_flip.default_out(model).name,
        "reasoning_probe": run_reasoning_probe.default_out(model).name,
        "erasure": run_reasoning_erasure.default_out(model).name,
        "additivity_cv": run_additivity.default_out("cv", model, "explicit").name,
        "additivity_credit": run_additivity.default_out("credit", model, "explicit").name,
    }


def source_record(name: str, data: Mapping[str, Any]) -> Dict[str, Any]:
    """What produced a result file: code commit and dirty flag, model commit, and its data files' SHA-256."""
    meta = data["meta"]
    code = meta.get("code") or {}
    return {"file": name, "git_commit": code.get("git_commit"), "git_dirty": code.get("git_dirty"),
            "model_revision": (meta.get("config") or {}).get("model_revision"),
            "data": {k: (v or {}).get("sha256") for k, v in (meta.get("data") or {}).items()}}


def collect(results_dir: Path, model: str = DEFAULT_MODEL) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Return (macros, report). Each macro maps a LaTeX command name (letters only) to a value; the report lists
    the ``missing`` files and keys, the ``skipped`` files (no ``meta``: produced before the reviewed runners) and
    the ``sources`` read."""
    m: Dict[str, str] = {}
    report: Dict[str, Any] = {"missing": [], "skipped": [], "sources": []}
    files = result_files(model)

    def need(key: str) -> Optional[Dict[str, Any]]:
        name = files[key]
        path = results_dir / name
        if not path.exists():
            report["missing"].append(name)
            return None
        data = json.loads(path.read_text())
        if "meta" not in data:
            report["skipped"].append(name)
            return None
        report["sources"].append(source_record(name, data))
        return data

    def put_ci(name: str, entry: Optional[Mapping[str, Any]], dp: int = 2, sign: bool = True,
               pct: bool = False) -> None:
        """``name`` = the interval's estimate, ``nameLo`` / ``nameHi`` its bounds."""
        if entry is None:
            report["missing"].append(f"interval for \\{name}")
            return
        m[name] = _fmt(entry["estimate"], dp, sign, pct)
        m[f"{name}Lo"] = _fmt(entry["ci_low"], dp, sign, pct)
        m[f"{name}Hi"] = _fmt(entry["ci_high"], dp, sign, pct)

    def battery_cells(key: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
        d = need(key)
        return {(c["axis"], c["encoding"]): c for c in d["cells"]} if d else {}

    def direct_cell(macro: str, cell: Mapping[str, Any], with_null: bool) -> None:
        """A direct-arm cell: auto_influence (point), the baseline mean gap with its interval, and with
        ``with_null`` also the nulled auto_influence and mean gap, and nulled − baseline of the mean gap."""
        m[macro] = _fmt(cell["baseline"]["auto_influence"])
        put_ci(f"{macro}Gap", cell["baseline"]["intervals"].get("mean_gap"))
        if with_null:
            m[f"{macro}null"] = _fmt(cell["nulled"]["auto_influence"])
            put_ci(f"{macro}Gapnull", cell["nulled"]["intervals"].get("mean_gap"))
            put_ci(f"{macro}GapChange", cell["baseline_vs_nulled"]["nulled_minus_baseline"].get("mean_gap_change"))

    def want(cells: Mapping, key: Tuple[str, str], source: str) -> Optional[Mapping[str, Any]]:
        if cells and key not in cells:
            report["missing"].append(f"{source}: {key}")
        return cells.get(key)

    # --- the direct arm: explicit sex and intersection (auto-influence macros), hiring and credit -------------
    for tag, key in (("CV", "battery_cv"), ("Credit", "battery_credit")):
        cells = battery_cells(key)
        for axis_tag, axis in (("Sex", "sex"), ("Intersection", "intersection")):
            cell = want(cells, (axis, "explicit"), files[key])
            if cell:
                m[f"autoInfl{axis_tag}{tag}base"] = _fmt(cell["baseline"]["auto_influence"])
                m[f"autoInfl{axis_tag}{tag}null"] = _fmt(cell["nulled"]["auto_influence"])
                put_ci(f"meanGap{axis_tag}{tag}", cell["baseline"]["intervals"].get("mean_gap"))
                put_ci(f"meanGap{axis_tag}{tag}null", cell["nulled"]["intervals"].get("mean_gap"))
                put_ci(f"meanGapChange{axis_tag}{tag}",
                       cell["baseline_vs_nulled"]["nulled_minus_baseline"].get("mean_gap_change"))
        if tag == "CV":
            # --- robustness battery (hiring): selected explicit/proxy cells -----------------------------------
            for macro, cell_key in {"batterySexProxyCV": ("sex", "proxy"), "batteryAgeExplicitCV": ("age", "explicit"),
                                    "batteryAgeProxyCV": ("age", "proxy"),
                                    "batteryFamilyExplicitCV": ("family_status", "explicit"),
                                    "batteryFamilyProxyCV": ("family_status", "proxy")}.items():
                cell = want(cells, cell_key, files[key])
                if cell:
                    direct_cell(macro, cell, with_null=False)
            if cells:
                m["nEvalBattery"] = str(next(iter(cells.values()))["n_eval"])
                sex = cells.get(("sex", "explicit"))
                if sex:
                    m["nEval"] = str(sex["baseline"]["n_examples"])

    # --- education (A1): the factorial (sex, ethnicity, economic status) and the stage design (grade level) ----
    factorial, stage = battery_cells("battery_education"), battery_cells("battery_education_stage")
    edu = {"eduSexExplicit": (factorial, "battery_education", ("sex", "explicit"), True),
           "eduSexProxy": (factorial, "battery_education", ("sex", "proxy"), False),
           "eduEthnicityExplicit": (factorial, "battery_education", ("ethnicity", "explicit"), False),
           "eduEthnicityProxy": (factorial, "battery_education", ("ethnicity", "proxy"), False),
           "eduEconomicExplicit": (factorial, "battery_education", ("economic_status", "explicit"), False),
           "eduEconomicProxy": (factorial, "battery_education", ("economic_status", "proxy"), False),
           "eduGradeExplicit": (stage, "battery_education_stage", ("grade_level", "explicit"), False),
           "eduGradeProxy": (stage, "battery_education_stage", ("grade_level", "proxy"), True)}
    for macro, (cells, key, cell_key, with_null) in edu.items():
        cell = want(cells, cell_key, files[key])
        if cell:
            direct_cell(f"{macro}AsapTwo", cell, with_null)

    # --- standpoint credibility (A2): identity gap vs a no-standpoint neutral baseline --------------------------
    sp_axes = {"Sex": "pos_sex", "Race": "pos_race", "Class": "pos_class", "Origin": "pos_origin",
               "Intersection": "pos_intersection", "Control": "pos_control",
               "Hobby": "pos_ctrl_hobby", "Pet": "pos_ctrl_pet", "Region": "pos_ctrl_region"}
    # ASAP 2.0 in two groups: the prompts where a standpoint is plausible (the A2 result) and the equally large
    # control group from prompts where it is not (`pairs.positionality.STANDPOINT_FIT`)
    for tag, key in (("AsapTwo", "maineffect_plausible"), ("AsapTwoControl", "maineffect_implausible")):
        d = need(key)
        if d:
            by = {r["axis"]: r for r in d["results"]}
            for name, axis in sp_axes.items():
                if axis in by:
                    put_ci(f"standpointGap{name}{tag}", by[axis]["intervals"]["identity_gap"])
            if "pos_intersection" in by:
                put_ci(f"standpointMainEffect{tag}", by["pos_intersection"]["intervals"]["main_effect"])
            m[f"nEvalStandpoint{tag}"] = str(d["n_essays"])        # essays (every axis uses the same)

    # --- decision response (hiring, explicit): the blatant floor ------------------------------------------------
    d = need("decision_cv")
    if d:
        by = {r["axis"]: r["intervals"]["baseline"] for r in d["results"]}
        for tag, axis in {"Sex": "sex", "Age": "age", "Family": "family_status", "Intersection": "intersection"}.items():
            if axis in by:
                put_ci(f"decisionDiscWin{tag}", by[axis]["discriminatory_win_rate"], sign=False)
                put_ci(f"decisionGapFairDisc{tag}", by[axis]["mean_gap_fair_minus_disc"])
                put_ci(f"decisionDiscWinNeutralDecline{tag}", by[axis]["disc_win_rate_vs_neutral_decline"], sign=False)

    # --- reasoning flip 2x2 (hiring): correctness / conclusion effects, and each premise − the commute control -
    d = need("reasoning")
    if d:
        by = {r["premise"]: r["intervals"]["baseline"] for r in d["results"]}
        for tag, prem in {"PL": "parental_leave", "Intersection": "intersection", "Commute": "commute"}.items():
            if prem in by:
                put_ci(f"reasonCorrectness{tag}", by[prem]["correctness_effect"])
                put_ci(f"reasonConclusion{tag}", by[prem]["conclusion_effect"])
        for tag, prem in (("PL", "parental_leave"), ("Intersection", "intersection")):
            if prem in d["versus_control"]:
                put_ci(f"reasonCorrectnessVsCommute{tag}", d["versus_control"][prem]["correctness_effect"])

    # --- reasoning probe (held-out applicants and wording) + cross-premise transfer ----------------------------
    d = need("reasoning_probe")
    if d:
        by = {r["premise"]: r["directions"] for r in d["results"]}
        for tag, prem in (("PL", "parental_leave"), ("Commute", "commute")):
            if prem in by:
                put_ci(f"probeCorrAccHeldout{tag}", by[prem]["correctness"]["paired_acc_heldout"], sign=False, pct=True)
                put_ci(f"probeConclAccHeldout{tag}", by[prem]["conclusion"]["paired_acc_heldout"], sign=False, pct=True)
        t = d["transfer"]
        m["cosCorrectnessPLvsCommute"] = _fmt(t["cosines"]["correctness"])
        iv = t["commute_on_parental_leave"]["correctness"]["intervals"]
        put_ci("transferCommuteToPLbase", iv["baseline"]["correctness_effect"])
        put_ci("transferCommuteToPLnull", iv["nulled"]["correctness_effect"])
        put_ci("transferCommuteToPLchange", iv["nulled_minus_baseline"]["correctness_effect"])

    # --- LEACE / non-linear-probe erasure (hiring), read against the lexical control ---------------------------
    d = need("erasure")
    if d:
        rows = {(r["premise"], r["concept"]): r for r in d["results"]}
        r = rows.get(("parental_leave", "correctness"))
        if r:
            m["erasureLinearNoneCorrPL"] = _fmt(r["rows"]["none"]["linear_acc"], pct=True)
            m["erasureLinearLeaceCorrPL"] = _fmt(r["rows"]["leace"]["linear_acc"], pct=True)
            m["erasureMlpLeaceCorrPL"] = _fmt(r["rows"]["leace"]["mlp_acc"], pct=True)
            put_ci("erasureMlpAboveChanceLeaceCorrPL", r["rows"]["leace"]["intervals"]["mlp_above_chance"])
            put_ci("erasureMlpAboveChanceLeaceCorrPLlexical",
                   r["lexical_control"]["leace"]["intervals"]["mlp_above_chance"])

    # --- additivity (RQ1.1, explicit): the vector-sum cosine sees only the three-way term; read the residual ---
    for tag, key in (("CV", "additivity_cv"), ("Credit", "additivity_credit")):
        d = need(key)
        if d:
            w = d["threeway"]
            m[f"additivityCos{tag}"] = _fmt(w["cos_intersection_vs_marginal_sum"], dp=3)
            m[f"additivityResidual{tag}"] = _fmt(w["residual_share_debiased"], dp=3)
            m[f"additivityResidualFloor{tag}"] = _fmt(w["noise_floor_share"], dp=3)
            m[f"additivityThreeWayP{tag}"] = _fmt(w["sign_flip_p"], dp=4)

    # --- the model ---------------------------------------------------------------------------------------------
    m["modelSmall"] = model
    return m, report


def defined_macros(tex: str) -> List[str]:
    return re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex)


def tex_breakage(old_numbers: Optional[Path], new_macros: Mapping[str, str], writeup_root: Path) -> Dict[str, Any]:
    """What the new export breaks in the write-up: macros of the old ``numbers.tex`` that disappear, with the
    ``.tex`` files still using them; the ``\\markstale`` lines of ``numbers_manual.tex`` for macros the new export
    does not define (they flag nothing any more; delete them); and macros ``numbers_manual.tex`` defines too
    ("already defined")."""
    old = set(defined_macros(old_numbers.read_text())) if old_numbers and old_numbers.exists() else set()
    gone = sorted(old - set(new_macros))
    tex_files = [p for p in writeup_root.rglob("*.tex")
                 if "build" not in p.relative_to(writeup_root).parts and p.name not in ("numbers.tex",)]
    texts = {p: p.read_text(errors="replace") for p in tex_files}
    uses = {name: sorted(str(p.relative_to(writeup_root)) for p, t in texts.items()
                         if re.search(rf"\\{name}(?![A-Za-z])", t))
            for name in gone}
    manual = writeup_root / "shared" / "numbers_manual.tex"
    manual_tex = manual.read_text() if manual.exists() else ""
    manual_defined = set(defined_macros(manual_tex))
    collide = sorted(manual_defined & set(new_macros))
    marked = set(re.findall(r"\\markstale\{([A-Za-z]+)\}", manual_tex))
    return {"disappearing": {k: v for k, v in uses.items() if v},
            "disappearing_unused": [k for k, v in uses.items() if not v], "defined_twice": collide,
            "stale_marks_without_macro": sorted(marked - set(new_macros) - manual_defined)}


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def header(results_dir: Path, report: Mapping[str, Any]) -> List[str]:
    lines = ["% AUTO-GENERATED by runners/export_paper_numbers.py -- DO NOT EDIT BY HAND.",
             f"% source: {results_dir}/*.json | generated: {date.today().isoformat()} | exporter commit: {_git_hash()}",
             "% Hand-maintained / not-yet-serialized numbers live in shared/numbers_manual.tex.",
             "% Sources (file | code commit | dirty | model commit | data SHA-256):"]
    for s in report["sources"]:
        data = ", ".join(f"{k} {str(v)[:12]}" for k, v in s["data"].items())
        lines.append(f"%   {s['file']} | {s['git_commit']} | {s['git_dirty']} | {s['model_revision']} | {data}")
    commits = {s["git_commit"] for s in report["sources"]}
    if len(commits) > 1 or any(s["git_dirty"] for s in report["sources"]):
        lines.append(f"% WARNING: the sources come from {len(commits)} code commit(s)"
                     f"{' and a dirty tree' if any(s['git_dirty'] for s in report['sources']) else ''}.")
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "artifacts/results/demographic")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="The RM whose results are exported (its Hub path)")
    ap.add_argument("--out", type=Path, required=True, help="path to shared/numbers.tex in the write-up repo")
    args = ap.parse_args()

    macros, report = collect(args.results_dir, args.model)
    breakage = tex_breakage(args.out, macros, args.out.resolve().parent.parent)
    lines = header(args.results_dir, report)
    for name in sorted(macros):
        lines.append(f"\\newcommand{{\\{name}}}{{{macros[name]}}}")
    for label, items in (("missing result files / keys (macros skipped)", report["missing"]),
                         ("skipped: no meta, produced before the reviewed runners", report["skipped"])):
        if items:
            lines += ["", f"% {label}: " + ", ".join(sorted(items))]
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {len(macros)} macros -> {args.out}")
    for label, items in (("skipped (missing JSON or key)", report["missing"]),
                         ("skipped (no meta: pre-review result)", report["skipped"])):
        if items:
            print(f"{label}:", ", ".join(sorted(items)))
    if breakage["disappearing"]:
        print("macros that disappear but are still used (LaTeX will fail on these):")
        for name, files in breakage["disappearing"].items():
            print(f"  \\{name}: {', '.join(files)}")
    if breakage["stale_marks_without_macro"]:
        print("\\markstale lines in numbers_manual.tex for macros no longer exported (delete them):",
              ", ".join(breakage["stale_marks_without_macro"]))
    if breakage["defined_twice"]:
        print("defined in numbers_manual.tex as well (\"already defined\"): delete them there:",
              ", ".join(f"\\{n}" for n in breakage["defined_twice"]))


if __name__ == "__main__":
    main()
