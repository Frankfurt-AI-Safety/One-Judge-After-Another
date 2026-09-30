#!/usr/bin/env python3
"""
Pilot sizing — turns the pilot's outputs into the numbers the freeze needs (pilot-then-freeze; the rules
are stated in the working notes, 2026-09-25, before any pilot run). Offline, CPU only.

**Records per group (cross-marker).** For every headline contrast of the decision margin D — each axis's
disparity, the intersection, the additivity gap — and, separately, the two-way interactions, the pilot's
per-record SD in scaled units, r = sd / scale_sd (``scale_sd`` = SD across records of D on the unmarked
control; both from `scoring.cross_marker_metrics.summarize`), gives the records needed to detect a scaled
effect δ with 80% power under a Bonferroni/Holm family of m tests:

    n = ceil(((z_{1 − α/(2m)} + z_{power}) · r / δ)²)

reported for every δ in ``--deltas`` and m in ``--families`` (δ and m are fixed with the headline family),
next to the records available after the probe exclusion, per model and as the **maximum over models** (the
rule's answer). The three-way interaction is not in the rule's families: its r is listed (``outside_rule``) but it
sizes nothing. A contrast whose r is undefined is listed as ``unsized``, never dropped. **Blinded:** only ``n``,
``sd`` and ``scale_sd`` are read; means, CIs and d_z never are, so the choice of n cannot follow the effects.

**Probe records.** Each probe-curve JSON's answer per direction (``rule.answer``: the smallest passing N, or the
largest N for a direction that fails even there, which is reported) and per domain; the domain's answer is the
maximum over directions, encodings and models.

Every input must be a cross-marker summary or a probe curve (anything else is refused); the output lists the
inputs with their SHA-256, code commit and model commit.

Usage:
    python runners/pilot_sizing.py --inputs artifacts/results/demographic/pilot/*.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DELTAS = (0.05, 0.10, 0.20)
DEFAULT_FAMILIES = (1, 4, 12)
MAIN = "main"                 # disparities, intersection, additivity gap
INTERACTIONS = "interactions"  # the two-way interactions
OUTSIDE_RULE = "outside_rule"


def required_n(r: float, delta: float, m: int, alpha: float = 0.05, power: float = 0.80) -> Optional[int]:
    """Records for a paired (per-record) contrast with SD ``r`` (in the unit of ``delta``) to reach
    ``power`` at two-sided ``alpha / m``. None when r is undefined."""
    from scipy.stats import norm

    if r is None or r != r or delta <= 0:
        return None
    z = norm.ppf(1 - alpha / (2 * m)) + norm.ppf(power)
    return int(math.ceil((z * r / delta) ** 2))


def precision(stat: Mapping[str, Any]) -> Dict[str, Any]:
    """The blinded view of one `summarize` result: n and r = sd / scale_sd, nothing about its mean."""
    sd, scale_sd = stat.get("sd"), stat.get("scale_sd")
    ok = sd is not None and scale_sd is not None and sd == sd and scale_sd == scale_sd and scale_sd > 0
    return {"n": stat.get("n"), "r": float(sd) / float(scale_sd) if ok else None}


THREE_WAY = "three_way"


def contrasts(margin_group: Mapping[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """The headline contrasts of one margin and record group, blinded, by family (main / the two-way
    interactions), plus the contrasts outside the rule's families (the three-way interaction)."""
    main = {f"disparity:{axis}": precision(s) for axis, s in margin_group.get("disparity", {}).items()}
    if "additivity_gap" in margin_group:
        main["additivity_gap"] = precision(margin_group["additivity_gap"])
    interactions = margin_group.get("interactions", {})
    inter = {f"interaction:{k}": precision(s) for k, s in interactions.items() if k != THREE_WAY}
    outside = {f"interaction:{k}": precision(s) for k, s in interactions.items() if k == THREE_WAY}
    return {MAIN: main, INTERACTIONS: inter, OUTSIDE_RULE: outside}


def size_crossmarker(summary: Mapping[str, Any], deltas: Sequence[float], families: Sequence[int],
                     alpha: float = 0.05, power: float = 0.80, reward: str = "baseline",
                     margin: str = "D") -> List[Dict[str, Any]]:
    """One row per (encoding, group, family): the binding (largest-r) contrast and the records it needs. A family
    without a defined r still gets its row (``r`` None, nothing required), with its contrasts under ``unsized``."""
    rows: List[Dict[str, Any]] = []
    selection = summary.get("selection", {})
    for encoding, by_reward in summary.get("metrics", {}).items():
        groups = by_reward.get(reward, {}).get("margins", {}).get(margin, {})
        for group, block in groups.items():
            available = selection.get(f"available_{group}")
            sets = contrasts(block)
            for family in (MAIN, INTERACTIONS):
                items = sets[family]
                rs = {name: p["r"] for name, p in items.items() if p["r"] is not None}
                binding = max(rs, key=rs.get) if rs else None
                need = {f"delta={d:g},m={m}": (required_n(rs[binding], d, m, alpha, power) if binding else None)
                        for d in deltas for m in families}
                rows.append({"model": summary.get("model"), "domain": summary.get("domain"),
                             "encoding": encoding, "group": group, "family": family,
                             "n_pilot": items[binding]["n"] if binding else None, "available": available,
                             "binding_contrast": binding, "r": rs.get(binding), "r_by_contrast": rs,
                             "unsized": sorted(set(items) - set(rs)),
                             "outside_rule": {k: v["r"] for k, v in sets[OUTSIDE_RULE].items()},
                             "required": need,
                             "exceeds_available": {k: (available is not None and v is not None and v > available)
                                                   for k, v in need.items()}})
    return rows


def max_over_models(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The rule's answer per (domain, encoding, group, family): the largest required n over the models, next to
    the smallest pool; a model whose family is unsized makes that answer None (it cannot be sized)."""
    by_key: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[(r["domain"], r["encoding"], r["group"], r["family"])].append(r)
    out = []
    for (domain, encoding, group, family), rs in by_key.items():
        available = [r["available"] for r in rs if r["available"] is not None]
        pool = min(available) if available else None
        required = {}
        for k in rs[0]["required"]:
            values = [r["required"][k] for r in rs]
            required[k] = None if None in values else max(values)
        out.append({"domain": domain, "encoding": encoding, "group": group, "family": family,
                    "models": sorted(str(r["model"]) for r in rs), "available": pool, "required": required,
                    "exceeds_available": {k: (pool is not None and v is not None and v > pool)
                                          for k, v in required.items()}})
    return out


def probe_answers(curves: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per domain: each model's per-direction answer (`run_probe_curve.probe_rule`: the smallest passing N, or the
    largest N for a direction that fails even there), the failing directions, and the domain's answer (the maximum
    over directions, encodings and models)."""
    out: Dict[str, Any] = defaultdict(lambda: {"by_model": {}, "failing": {}, "answer": None})
    for c in curves:
        d = out[c["domain"]]
        d["by_model"][c["model"]] = {k: v["rule"]["answer"] for k, v in c["directions"].items()}
        d["failing"][c["model"]] = sorted(k for k, v in c["directions"].items()
                                          if v["rule"]["smallest_passing_n"] is None)
    for d in out.values():
        values = [n for per_model in d["by_model"].values() for n in per_model.values() if n is not None]
        d["answer"] = max(values, default=None)
    return dict(out)


def input_record(path: Path, data: Mapping[str, Any], kind: str) -> Dict[str, Any]:
    """Which pilot file was read: its path and SHA-256, and the code and model commits it came from."""
    from pairs.manifest import file_sha256

    meta = data.get("meta") or {}
    code = meta.get("code") or {}
    return {"path": str(path), "sha256": file_sha256(path), "kind": kind, "model": data.get("model"),
            "code": {k: code.get(k) for k in ("git_commit", "git_dirty")},
            "model_revision": (meta.get("config") or {}).get("model_revision")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True,
                    help="Pilot JSONs: cross-marker summaries and/or probe curves (told apart by content)")
    ap.add_argument("--deltas", default=",".join(map(str, DEFAULT_DELTAS)))
    ap.add_argument("--families", default=",".join(map(str, DEFAULT_FAMILIES)))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--out", type=Path, default=Path("artifacts/results/demographic/pilot/sizing.json"))
    args = ap.parse_args()
    deltas = [float(x) for x in args.deltas.split(",")]
    families = [int(x) for x in args.families.split(",")]

    rows: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    inputs: List[Dict[str, Any]] = []
    for path in args.inputs:
        data = json.loads(path.read_text())
        if "directions" in data and "grid" in data:
            curves.append(data)
            inputs.append(input_record(path, data, "probe_curve"))
        elif "metrics" in data and "selection" in data:
            rows += size_crossmarker(data, deltas, families, args.alpha, args.power)
            inputs.append(input_record(path, data, "cross_marker"))
        else:
            raise SystemExit(f"{path}: neither a cross-marker summary nor a probe curve")
    result = {"inputs": inputs, "deltas": deltas, "families": families, "alpha": args.alpha, "power": args.power,
              "records_per_group": rows, "records_per_group_max_over_models": max_over_models(rows),
              "probe_records": probe_answers(curves)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print_table(result)
    print(f"saved → {args.out}")


def print_table(result: Mapping[str, Any]) -> None:
    deltas, families = result["deltas"], result["families"]
    fmt = lambda r, k: f"{r['required'][k]}{'*' if r['exceeds_available'][k] else ''}"
    cells = lambda r: "  ".join(" " * 4 + "/".join(fmt(r, f"delta={d:g},m={m}") for m in families) for d in deltas)
    print("\n" + "=" * 110)
    print(f"RECORDS PER GROUP — power {result['power']}, alpha {result['alpha']} (Holm/Bonferroni over m); "
          f"'*' = more than available. Means are never read.")
    head = "  ".join(f"δ={d:g}: " + "/".join(f"m{m}" for m in families) for d in deltas)
    print(f"  {'domain / model / enc / group / family':72} {'r':>6}  {head}  avail")
    for r in result["records_per_group"]:
        label = f"{r['domain']} / {Path(str(r['model'])).name} / {r['encoding']} / {r['group']} / {r['family']}"
        rr = "  None" if r["r"] is None else f"{r['r']:6.2f}"
        extra = f"({r['binding_contrast']})" + (f"  unsized: {r['unsized']}" if r["unsized"] else "")
        print(f"  {label:72} {rr}  {cells(r)}  {r['available']}   {extra}")
    print("-" * 110)
    print("THE RULE'S ANSWER — maximum over models")
    for r in result["records_per_group_max_over_models"]:
        label = f"{r['domain']} / {r['encoding']} / {r['group']} / {r['family']} ({len(r['models'])} models)"
        print(f"  {label:79}  {cells(r)}  {r['available']}")
    print("-" * 110)
    print("PROBE RECORDS — per domain (max over directions, encodings, models; a direction failing even at the "
          "largest N answers that N)")
    for domain, d in result["probe_records"].items():
        failing = {Path(m).name: f for m, f in d["failing"].items() if f}
        print(f"  {domain:12} answer {d['answer']}   " +
              "; ".join(f"{Path(m).name}: max {max((v for v in per.values() if v is not None), default=None)}"
                        for m, per in d["by_model"].items()) +
              (f"   failing: {failing}" if failing else ""))
    print("=" * 110)


if __name__ == "__main__":
    main()
