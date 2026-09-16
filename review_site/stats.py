"""
Descriptive statistics over a generated matched-pair dataset, for the review site.

Reads the artifacts the generators already write --- `manifest.json` for what the run declared,
`pairs.jsonl` for what it actually produced --- and recomputes the realized length deltas so a
reviewer can see the distribution *against the gate threshold* rather than taking the 0% discard
rate on faith.

Two properties of the shipped data this surfaces rather than smooths over:

- `exemplar` is empty for every *explicit* encoding and for the age/family/grade proxies, because
  nothing is sampled there --- the clause is a fixed constant. So "marker distribution" is only
  meaningful for the cells that draw from a pool (sex/proxy, ethnicity/proxy, intersection).
- Where a pool *is* drawn from, the draw is an independent per-pair choice rather than a
  stratified split, so the counts are multinomial noise around uniform. The ethnicity grid's
  `held_sex` balance is the clearest case.

Examples come from `pairs.jsonl`, not `spotcheck.csv`: the spotcheck sample is only 30 rows
spread unevenly across cells (so some cells would have too few), and it lacks `clause_a`,
`exemplar` and `held_fixed`, which the pages need.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class MissingArtifact(FileNotFoundError):
    """Raised with the exact generator command needed to produce the missing dataset."""


_GENERATORS = {
    "credit": "python experiments/generate_credit_data.py",
    "cv": "python experiments/generate_bios_data.py --from-hub",
    "education/persuade": "python experiments/generate_edu_data.py --source persuade",
    "education/asap": "python experiments/generate_edu_data.py --source asap",
    "education_positioned/persuade": "python experiments/generate_positioned_data.py",
}


def _require(path: Path, key: str) -> Path:
    if not path.exists():
        how = _GENERATORS.get(key, "(see experiments/)")
        raise MissingArtifact(
            f"{path} not found. The review site builds from locally generated data "
            f"(it is gitignored). Generate it with:\n    {how}"
        )
    return path


@dataclass(frozen=True)
class CellStats:
    """One (axis, encoding) cell of a dataset."""

    axis: str
    encoding: str
    n: int
    label_a: str
    label_b: str
    templates: Counter
    distinct_sources: int
    char_deltas: List[int]
    token_deltas: List[int]
    held_fixed: List[str]
    exemplar_counts: Dict[str, Counter] = field(default_factory=dict)

    @property
    def samples_a_pool(self) -> bool:
        """True when the marker is drawn from a pool, i.e. a distribution actually exists."""
        return bool(self.exemplar_counts)

    @property
    def max_char_delta(self) -> int:
        return max(self.char_deltas) if self.char_deltas else 0

    @property
    def max_token_delta(self) -> int:
        return max(self.token_deltas) if self.token_deltas else 0

    def delta_histogram(self) -> Counter:
        return Counter(self.char_deltas)


@dataclass(frozen=True)
class DatasetStats:
    key: str
    manifest: Dict[str, Any]
    cells: List[CellStats]

    @property
    def n_records(self) -> int:
        return int(self.manifest.get("n_records", 0))

    @property
    def seed(self) -> int:
        return int(self.manifest.get("seed", 0))

    @property
    def generator_version(self) -> str:
        return str(self.manifest.get("generator_version", "?"))

    @property
    def attribution(self) -> str:
        return str(self.manifest.get("attribution", ""))

    @property
    def templates(self) -> Dict[str, str]:
        return dict(self.manifest.get("templates", {}))

    @property
    def thresholds(self) -> Dict[str, Any]:
        return dict(self.manifest.get("validation_thresholds", {}))

    @property
    def discard_report(self) -> Dict[str, Any]:
        return dict(self.manifest.get("discard_report", {}))

    def threshold_for(self, axis: str) -> int:
        """The char-delta bound actually applied to this axis (intersection gets a relaxed one)."""
        thr = self.thresholds
        if axis == "intersection" and "intersection_char_delta" in thr:
            return int(thr["intersection_char_delta"])
        return int(thr.get("max_char_delta", 12))

    def cell(self, axis: str, encoding: str) -> Optional[CellStats]:
        return next((c for c in self.cells if c.axis == axis and c.encoding == encoding), None)

    @property
    def total_discard_rate(self) -> float:
        rep = self.discard_report.values()
        kept = sum(int(v.get("kept", 0)) for v in rep)
        seen = sum(int(v.get("examined", 0)) for v in rep)
        return 0.0 if not seen else 1.0 - kept / seen


def iter_pairs(root: Path, key: str) -> Iterator[Dict[str, Any]]:
    """Stream `pairs.jsonl` (files run to ~19 MB, so do not slurp)."""
    path = _require(root / key / "pairs.jsonl", key)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_manifest(root: Path, key: str) -> Dict[str, Any]:
    return json.loads(_require(root / key / "manifest.json", key).read_text())


def load(root: Path, key: str) -> DatasetStats:
    """Full descriptive statistics for one dataset."""
    manifest = load_manifest(root, key)

    buckets: Dict[tuple, Dict[str, Any]] = {}
    for rec in iter_pairs(root, key):
        k = (rec["varied_axis"], rec["encoding"])
        b = buckets.setdefault(k, {
            "n": 0, "templates": Counter(), "sources": set(),
            "char_deltas": [], "token_deltas": [], "exemplar": {},
            "label_a": rec.get("label_a", ""), "label_b": rec.get("label_b", ""),
            "held_fixed": list(rec.get("held_fixed", [])),
        })
        b["n"] += 1
        b["templates"][rec.get("template_id", "?")] += 1
        b["sources"].add(rec.get("source_record_id", ""))
        ta, tb = rec.get("text_a", ""), rec.get("text_b", "")
        b["char_deltas"].append(abs(len(ta) - len(tb)))
        # Match validate.py exactly: whitespace split, so these line up with the gate.
        b["token_deltas"].append(abs(len(ta.split()) - len(tb.split())))
        for fname, fval in (rec.get("exemplar") or {}).items():
            if isinstance(fval, (str, int, float, bool)):
                b["exemplar"].setdefault(fname, Counter())[str(fval)] += 1

    cells = [
        CellStats(
            axis=axis, encoding=enc, n=b["n"], label_a=b["label_a"], label_b=b["label_b"],
            templates=b["templates"], distinct_sources=len(b["sources"]),
            char_deltas=b["char_deltas"], token_deltas=b["token_deltas"],
            held_fixed=b["held_fixed"], exemplar_counts=b["exemplar"],
        )
        for (axis, enc), b in sorted(buckets.items())
    ]
    return DatasetStats(key=key, manifest=manifest, cells=cells)


def examples(root: Path, key: str, axis: str, encoding: str, n: int = 4) -> List[Dict[str, Any]]:
    """`n` example pairs for one cell, evenly strided so the choice is deterministic and spread.

    Returns full `pairs.jsonl` records (clause, exemplar and held_fixed included), which the
    30-row `spotcheck.csv` sample cannot provide.
    """
    matching = [r for r in iter_pairs(root, key)
                if r["varied_axis"] == axis and r["encoding"] == encoding]
    if not matching:
        return []
    if len(matching) <= n:
        return matching
    stride = len(matching) // n
    return [matching[i * stride] for i in range(n)]


def available(root: Path) -> List[str]:
    """Dataset keys that are actually generated locally."""
    return [k for k in _GENERATORS if (root / k / "pairs.jsonl").exists()]
