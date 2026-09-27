"""
Manifest schema + writer shared by every pair generator (credit, hiring on Bias-in-Bios, education: the
factorial, the stage design and the positioned-argument design).

Each matched pair becomes one self-describing JSONL row (which axis varied, what was held fixed, template,
encoding, intersectional cell, the A/B texts and clauses, the record's real fields, provenance). The writer
emits three files into the output directory:
- ``pairs.jsonl``   — one row per matched pair (what the loader reads).
- ``manifest.json`` — generator version, the git commit and dirty paths (`code_provenance`), seed, counts
  per axis/encoding/role, template hashes, validation thresholds, discard report, domain and the
  substrate's licence line.
- ``spotcheck.csv`` — a seeded sample for human review, stratified by (axis, encoding).
The factorial generators also write ``cells.jsonl`` (all eight texts per block) next to these, themselves.

Per-RM chat-template formatting is intentionally NOT stored: it is applied at scoring time
(``format_conversation``) with each model's tokenizer, so one manifest serves every RM.
"""

from __future__ import annotations

import csv
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from pairs.factorial import stable_rng
from pairs.markers import GeneratedPair
from pairs.positionality import POSITION_TEMPLATES  # positioned-argument (A2) sentence templates
from substrates.bios_render import BIOS_TEMPLATES
from substrates.credit_render import TEMPLATES
from substrates.education_render import EDU_TEMPLATES


def _merge_templates(*registries: Dict[str, str]) -> Dict[str, str]:
    """One id -> template registry over all domains; an id defined twice raises instead of overwriting."""
    merged: Dict[str, str] = {}
    for reg in registries:
        clash = merged.keys() & reg.keys()
        if clash:
            raise ValueError(f"template ids defined in two registries: {sorted(clash)}")
        merged.update(reg)
    return merged


# Combined registry so provenance hashing works for every domain's template ids.
_ALL_TEMPLATES: Dict[str, str] = _merge_templates(TEMPLATES, BIOS_TEMPLATES, EDU_TEMPLATES, POSITION_TEMPLATES)

# 0.2.0: credit moved to the sex × age × marital-status factorial on the corrected codebook.
# 0.3.0: pairs/ code review (2026-09-26): education proxy names drawn independently per cell, stage pairs'
#        held_fixed/cell metadata, gate reason codes and clause-position check. manifest.json also records the
#        git commit (`code_provenance`), which identifies the code even where this version was not bumped.
GENERATOR_VERSION = "0.3.0"

_REPO_ROOT = Path(__file__).resolve().parents[1]

GERMAN_CREDIT_ATTRIBUTION = (
    "Substrate: UCI Statlog (German Credit Data), CC-BY-4.0, decoded with the corrected code table "
    "of Groemping (2019). Demographic markers (sex x age x marital status, full factorial) are "
    "synthetic and self-licensed."
)

BIOS_ATTRIBUTION = (
    "Substrate: real biographies — Bias in Bios (De-Arteaga et al., 2019), HF mirror "
    "'LabHC/bias_in_bios', MIT licence; sourced from Common Crawl. Bodies are scrubbed (leading name "
    "stripped, gendered pronouns/titles neutralised, contact details replaced; bios still containing a "
    "gendered word or a sex-coded first name dropped) and NOT redistributed — these are biographies of "
    "identifiable real people, so only aggregate metrics are shared. The dataset's own gender label is "
    "retained for validity checks and is never rendered into the text. Quality label is role-match "
    "(profession == target role). Demographic markers are synthetic, injected, self-licensed."
)

EDU_ATTRIBUTION = (
    "Substrate: real essays — PERSUADE 2.0 (CC BY-NC-SA 4.0, research/measurement use; derived essays "
    "NOT redistributed) and/or ASAP 2.0 (CC BY 4.0; only the essays not in PERSUADE). Markers are "
    "synthetic, injected, self-licensed: the sex x ethnicity x economic-status factorial (stated, or a first name for "
    "sex and ethnicity and the school's free-lunch share for economic status), the education-stage design "
    "(a stated age or school stage), and the positioned-argument design (an identity sentence in the essay)."
)


def code_provenance(root: Path = _REPO_ROOT) -> Dict[str, Any]:
    """The git commit the generator ran from and the paths that differed from it (``git status``, ignored
    files excluded, so generated data does not count). All ``None`` where git or the repo is unavailable."""
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"],
                                         text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None, "git_dirty_paths": None}
    paths = sorted(line[3:] for line in status.splitlines() if line.strip())
    return {"git_commit": commit, "git_dirty": bool(paths), "git_dirty_paths": paths}


def _template_hash(template_id: str) -> str:
    return hashlib.sha256(_ALL_TEMPLATES[template_id].encode("utf-8")).hexdigest()[:12]


def _spotcheck_sample(records: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """A seeded sample for human review, stratified by (axis, encoding): ceil(n / strata) rows from each
    stratum (all of a smaller one), in manifest order. A stride over the whole file would keep landing on
    the same rows of each block (credit's 30 rows covered only two of seven strata)."""
    strata: Dict[tuple, List[int]] = {}
    for i, rec in enumerate(records):
        strata.setdefault((rec["varied_axis"], rec["encoding"]), []).append(i)
    if not strata or n <= 0:
        return []
    k = -(-n // len(strata))
    picked: List[int] = []
    for (axis, enc), idx in strata.items():
        picked += idx if len(idx) <= k else stable_rng("spotcheck", seed, axis, enc).sample(idx, k)
    return [records[i] for i in sorted(picked)]


def pair_to_record(
    pair: GeneratedPair,
    item_id: str,
    *,
    role: str = "probe",
    seed: int,
    domain: str,
    real_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize a :class:`GeneratedPair` to a manifest JSONL record.

    ``real_fields`` (the record's real, never-rendered attributes incl. its quality label, as in
    cells.jsonl) is written when given: the probe_records split stratifies on the quality label."""
    row = {
        "id": item_id,
        "domain": domain,
        "role": role,
        "source_record_id": pair.record_id,
        "varied_axis": pair.axis,
        "encoding": pair.encoding,
        "template_id": pair.template_id,
        "held_fixed": pair.held_fixed,
        "intersectional_cell": pair.intersectional_cell,
        "label_a": pair.label_a,
        "label_b": pair.label_b,
        "text_a": pair.text_a,
        "text_b": pair.text_b,
        "clause_a": pair.clause_a,
        "clause_b": pair.clause_b,
        "exemplar": pair.exemplar,
        "provenance": {
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
            "template_hash": _template_hash(pair.template_id),
        },
    }
    header = pair.exemplar.get("header_template")
    if header is not None:  # positioned pairs: the essay shell the sentence sits in, a template of its own
        row["provenance"]["header_template_hash"] = _template_hash(header)
    if real_fields is not None:
        row["real_fields"] = real_fields
    return row


def write_manifest(
    out_dir: Path | str,
    records: List[Dict[str, Any]],
    *,
    seed: int,
    discard_report: Dict[str, Any],
    thresholds: Dict[str, Any],
    spotcheck_n: int = 30,
    domain: str,
    attribution: str,
) -> Dict[str, Path]:
    """Write pairs.jsonl + manifest.json + spotcheck.csv. Returns the written paths.

    ``domain`` and ``attribution`` (the substrate's licence line) have no default: a generator that forgot
    them would otherwise label its manifest as German Credit under CC-BY."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.jsonl"
    manifest_path = out_dir / "manifest.json"
    spotcheck_path = out_dir / "spotcheck.csv"

    with open(pairs_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # counts per (axis, encoding) and per role
    counts: Dict[str, int] = {}
    for rec in records:
        key = f"{rec['varied_axis']}/{rec['encoding']}/{rec['role']}"
        counts[key] = counts.get(key, 0) + 1

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "code": code_provenance(),
        "seed": seed,
        "domain": domain,
        "n_records": len(records),
        "counts_by_axis_encoding_role": counts,
        "templates": {tid: _template_hash(tid) for tid in sorted(
            {r["template_id"] for r in records}
            | {r["exemplar"]["header_template"] for r in records if "header_template" in r.get("exemplar", {})})},
        "validation_thresholds": thresholds,
        "discard_report": discard_report,
        "attribution": attribution,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    sample = _spotcheck_sample(records, spotcheck_n, seed)
    with open(spotcheck_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "varied_axis", "encoding", "template_id", "label_a", "label_b",
                    "text_a", "text_b"])
        for rec in sample:
            w.writerow([rec["id"], rec["varied_axis"], rec["encoding"], rec["template_id"],
                        rec["label_a"], rec["label_b"], rec["text_a"], rec["text_b"]])

    return {"pairs": pairs_path, "manifest": manifest_path, "spotcheck": spotcheck_path}
