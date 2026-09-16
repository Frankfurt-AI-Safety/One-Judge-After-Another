"""
Manifest schema + writer for the demographic credit pipeline.

Each probe pair becomes one self-describing JSONL record (which axis varied, which were held fixed,
template, encoding, intersectional cell, the A/B texts + clauses, and provenance for reproducibility).
The writer emits three artifacts under the output directory:
- ``pairs.jsonl``   — one record per matched pair (the dataset the loader consumes).
- ``manifest.json`` — counts per axis/encoding/cell, generator version/seed, validation thresholds,
  discard report, and source-dataset licence attribution.
- ``spotcheck.csv``  — a deterministic sample for human review.

Per-RM chat-template formatting is intentionally NOT stored — it is applied at scoring time via
``format_conversation`` against each model's tokenizer, so one manifest serves all five RMs.
"""

from __future__ import annotations

import csv
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from pairs.markers import GeneratedPair
from substrates.credit_render import TEMPLATES

try:  # Legacy synthetic-CV templates (retained so pre-2026-08 CV manifests still hash).
    from pairs.cv_render import CV_TEMPLATES
except Exception:  # pragma: no cover - defensive
    CV_TEMPLATES = {}

try:  # Bias-in-Bios (real biography) templates — the current hiring substrate.
    from substrates.bios_render import BIOS_TEMPLATES
except Exception:  # pragma: no cover - defensive
    BIOS_TEMPLATES = {}

try:  # Education (essay) templates; absent on a minimal checkout is fine.
    from substrates.education_render import EDU_TEMPLATES
except Exception:  # pragma: no cover - defensive
    EDU_TEMPLATES = {}

try:  # Positioned-argument (A2) sentence templates.
    from pairs.positionality import POSITION_TEMPLATES
except Exception:  # pragma: no cover - defensive
    POSITION_TEMPLATES = {}

# Combined registry so provenance hashing works for every domain's template ids.
_ALL_TEMPLATES: Dict[str, str] = {
    **TEMPLATES, **CV_TEMPLATES, **BIOS_TEMPLATES, **EDU_TEMPLATES, **POSITION_TEMPLATES,
}

# 0.2.0: credit moved to the sex × age × marital-status factorial on the corrected codebook.
GENERATOR_VERSION = "0.2.0"

GERMAN_CREDIT_ATTRIBUTION = (
    "Substrate: UCI Statlog (German Credit Data), CC-BY-4.0, decoded with the corrected code table "
    "of Groemping (2019). Demographic markers (sex x age x marital status, full factorial) are "
    "synthetic and self-licensed."
)

CV_ATTRIBUTION = (
    "Substrate: synthetic CV records, generated deterministically from seed (no external dataset). "
    "Demographic markers (sex/age/family-status) are synthetic. Self-licensed. "
    "LEGACY: retained to reproduce pre-2026-08 hiring results; the hiring domain now uses BIOS_ATTRIBUTION."
)

BIOS_ATTRIBUTION = (
    "Substrate: real biographies — Bias in Bios (De-Arteaga et al., 2019), HF mirror "
    "'LabHC/bias_in_bios', MIT licence; sourced from Common Crawl. Bodies are scrubbed (leading name "
    "stripped, gendered pronouns/titles neutralised) and NOT redistributed — these are biographies of "
    "identifiable real people, so only aggregate metrics are shared. The dataset's own gender label is "
    "retained for validity checks and is never rendered into the text. Quality label is role-match "
    "(profession == target role). Demographic markers are synthetic, injected, self-licensed."
)

EDU_ATTRIBUTION = (
    "Substrate: real essays — PERSUADE 2.0 (CC BY-NC-SA 4.0, research/measurement use; derived essays "
    "NOT redistributed) and/or ASAP-AES (Kaggle 2012 Hewlett competition terms). Demographic header "
    "markers (name -> sex/ethnicity; grade level -> age) are synthetic, injected, self-licensed."
)


def _template_hash(template_id: str) -> str:
    return hashlib.sha256(_ALL_TEMPLATES[template_id].encode("utf-8")).hexdigest()[:12]


def pair_to_record(
    pair: GeneratedPair,
    item_id: str,
    *,
    role: str = "probe",
    seed: int,
    domain: str = "credit",
) -> Dict[str, Any]:
    """Serialize a :class:`GeneratedPair` to a manifest JSONL record."""
    return {
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


def write_manifest(
    out_dir: Path | str,
    records: List[Dict[str, Any]],
    *,
    seed: int,
    discard_report: Dict[str, Any],
    thresholds: Dict[str, Any],
    spotcheck_n: int = 30,
    domain: str = "credit",
    attribution: str = GERMAN_CREDIT_ATTRIBUTION,
) -> Dict[str, Path]:
    """Write pairs.jsonl + manifest.json + spotcheck.csv. Returns the written paths."""
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
        "seed": seed,
        "domain": domain,
        "n_records": len(records),
        "counts_by_axis_encoding_role": counts,
        "templates": {tid: _template_hash(tid)
                      for tid in sorted({r["template_id"] for r in records})},
        "validation_thresholds": thresholds,
        "discard_report": discard_report,
        "attribution": attribution,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # deterministic spot-check sample (stride sampling over the ordered records)
    n = min(spotcheck_n, len(records))
    stride = max(len(records) // n, 1) if n else 1
    sample = records[::stride][:n]
    with open(spotcheck_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "varied_axis", "encoding", "template_id", "label_a", "label_b",
                    "text_a", "text_b"])
        for rec in sample:
            w.writerow([rec["id"], rec["varied_axis"], rec["encoding"], rec["template_id"],
                        rec["label_a"], rec["label_b"], rec["text_a"], rec["text_b"]])

    return {"pairs": pairs_path, "manifest": manifest_path, "spotcheck": spotcheck_path}
