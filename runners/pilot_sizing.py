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
next to the records available after the probe exclusion. **Blinded:** only ``n``, ``sd`` and ``scale_sd``
are read; means, CIs and d_z never are, so the choice of n cannot follow the effects.

**Probe records.** Each probe-curve JSON's ``smallest_passing_n`` per direction and per domain; the
domain's answer is the maximum over directions, encodings and models.

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
INTERACTIONS = "interactions"


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


def contrasts(margin_group: Mapping[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """The headline contrasts of one margin and record group, blinded, by family (main / interactions)."""
    main = {f"disparity:{axis}": precision(s) for axis, s in margin_group.get("disparity", {}).items()}
    if "additivity_gap" in margin_group:
        main["additivity_gap"] = precision(margin_group["additivity_gap"])
    inter = {f"interaction:{k}": precision(s) for k, s in margin_group.get("interactions", {}).items()}
    return {MAIN: main, INTERACTIONS: inter}


def size_crossmarker(summary: Mapping[str, Any], deltas: Sequence[float], families: Sequence[int],
                     alpha: float = 0.05, power: float = 0.80, reward: str = "baseline",
                     margin: str = "D") -> List[Dict[str, Any]]:
    """One row per (encoding, group, family): the binding (largest-r) contrast and the records it needs."""
    rows: List[Dict[str, Any]] = []
    selection = summary.get("selection", {})
    for encoding, by_reward in summary.get("metrics", {}).items():
        groups = by_reward.get(reward, {}).get("margins", {}).get(margin, {})
        for group, block in groups.items():
            available = selection.get(f"available_{group}")
            for family, items in contrasts(block).items():
                rs = {name: p["r"] for name, p in items.items() if p["r"] is not None}
                if not rs:
                    continue
                binding = max(rs, key=rs.get)
                need = {f"delta={d:g},m={m}": required_n(rs[binding], d, m, alpha, power)
                        for d in deltas for m in families}
                rows.append({"model": summary.get("model"), "domain": summary.get("domain"),
                             "encoding": encoding, "group": group, "family": family,
                             "n_pilot": items[binding]["n"], "available": available,
                             "binding_contrast": binding, "r": rs[binding], "r_by_contrast": rs,
                             "required": need,
                             "exceeds_available": {k: (available is not None and v is not None and v > available)
                                                   for k, v in need.items()}})
    return rows


def probe_answers(curves: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per domain: each model's per-direction smallest passing N, and the domain's answer (the maximum;
    None when any direction fails even at the largest N)."""
    out: Dict[str, Any] = defaultdict(lambda: {"by_model": {}, "answer": None})
    for c in curves:
        d = out[c["domain"]]
        d["by_model"][c["model"]] = {k: v["rule"]["smallest_passing_n"] for k, v in c["directions"].items()}
    for d in out.values():
        values = [n for per_model in d["by_model"].values() for n in per_model.values()]
        d["answer"] = None if (not values or None in values) else max(values)
    return dict(out)


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
    for path in args.inputs:
        data = json.loads(path.read_text())
        if "directions" in data and "grid" in data:
            curves.append(data)
        elif "metrics" in data and "selection" in data:
            rows += size_crossmarker(data, deltas, families, args.alpha, args.power)
    result = {"deltas": deltas, "families": families, "alpha": args.alpha, "power": args.power,
              "records_per_group": rows, "probe_records": probe_answers(curves)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print_table(result)
    print(f"saved → {args.out}")


def print_table(result: Mapping[str, Any]) -> None:
    deltas, families = result["deltas"], result["families"]
    print("\n" + "=" * 110)
    print(f"RECORDS PER GROUP — power {result['power']}, alpha {result['alpha']} (Holm/Bonferroni over m); "
          f"'*' = more than available. Means are never read.")
    head = "  ".join(f"δ={d:g}: " + "/".join(f"m{m}" for m in families) for d in deltas)
    print(f"  {'domain / model / enc / group / family':72} {'r':>6}  {head}  avail")
    for r in result["records_per_group"]:
        label = f"{r['domain']} / {Path(str(r['model'])).name} / {r['encoding']} / {r['group']} / {r['family']}"
        cells = "  ".join(
            " " * 4 + "/".join(f"{r['required'][f'delta={d:g},m={m}']}"
                               f"{'*' if r['exceeds_available'][f'delta={d:g},m={m}'] else ''}"
                               for m in families) for d in deltas)
        print(f"  {label:72} {r['r']:6.2f}  {cells}  {r['available']}   ({r['binding_contrast']})")
    print("-" * 110)
    print("PROBE RECORDS — smallest N passing the rule, per domain (max over directions, encodings, models)")
    for domain, d in result["probe_records"].items():
        print(f"  {domain:12} answer {d['answer']}   " +
              "; ".join(f"{Path(m).name}: max {max((v for v in per.values() if v is not None), default=None)}"
                        f"{' (a direction fails at the largest N)' if None in per.values() else ''}"
                        for m, per in d["by_model"].items()))
    print("=" * 110)


if __name__ == "__main__":
    main()
