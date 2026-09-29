"""
Experiment configuration and model loading, shared by every runner.

`ExperimentConfig` is the per-experiment YAML (``configs/*.yaml``; CLI overrides are applied by each runner).
`BiasExperiment.load_model` loads the reward model (`scoring.backend`), verifies that the pipeline reproduces
the model's own score, attaches the embedding cache and records the Hub commit the weights came from.
`add_override_args`/`apply_overrides` are the runners' shared CLI overrides (YAML < CLI); `data_file` and
`run_metadata` are what every result file records about the run that produced it (under ``meta``).

The runners do the rest themselves: the direct arm is `runners/run_battery.py`, the cross-marker design
`runners/run_cross_marker.py`, and so on. An evaluation pipeline used to live here too (``evaluate``/``run``,
``ExperimentResults``, plots and raw-data dumps, a cross-dataset probe source), driven by
``runners/run_experiment.py``; that runner never called it (it exited after building the config, from the
initial commit on), `run_battery.py` computes a superset of its output, and it was removed on 2026-09-28.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for one experiment (one config file)."""

    # Experiment identification
    name: str
    """Experiment name, e.g. 'demographic_credit_sex_qwen06'"""

    bias_type: str
    """The bias family; 'demographic' for every current config"""

    # Model settings
    model_path: str
    """Path or HuggingFace ID of the reward model"""

    trust_remote_code: bool = False
    """Whether to run a checkpoint's own code (``auto_map``) when loading it. None of the eleven models needs it
    (checked 2026-09-28; QRM loads with `scoring/qrm.py`), and with it on, a repo that gains an ``auto_map`` on
    ``main`` would have its code run, and preferred over the transformers class, without notice."""

    model_revision: Optional[str] = None
    """Hub revision (commit) of the model; None = the pinned default in `scoring.backend.PINNED_REVISIONS`,
    else the Hub's main branch. `BiasExperiment.load_model` overwrites it with the commit actually loaded."""

    # Dataset settings
    dataset_source: str = ""
    """Path to the matched-pair manifest (``pairs.jsonl``); "" = the domain's default"""

    probe_records: Optional[int] = None
    """Number of RECORDS for probe training, stratified by quality (required by every dataset; see
    `scoring.dataset_base`). The pair count ``probe_size`` it replaced was removed on 2026-09-28: a record
    contributes several pairs (8 per single axis, 2 for the intersection), so counting pairs fixed the
    records only indirectly."""

    max_test_examples: Optional[int] = None
    """Maximum number of test examples, in pairs (None = use all)"""

    split_seed: int = 42
    """Seed for deterministic train/test split"""

    # Inference settings
    batch_size: int = 8
    """Batch size for inference"""

    max_length: int = 2048
    """Maximum sequence length; a longer input is refused (`probes.probe.InputTooLong`), never truncated"""

    device: str = "auto"
    """Device: 'cuda' or 'auto' (every visible GPU, via accelerate's ``device_map="auto"``), 'cuda:N', 'mps'
    or 'cpu'."""

    embedding_cache_dir: Optional[str] = "artifacts/embedding_cache"
    """Root of the per-model embedding cache (`probes/embedding_cache.py`): each unique text is
    embedded once per model and reused by every probe, reward, α-sweep and offline analysis. None
    disables it; the environment variable ONEJUDGE_EMBED_CACHE overrides (``off`` or a path)."""

    # Additional runner-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)
    """Settings of one runner (e.g. ``domain``, ``cross_marker``)"""

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, source=str(path))

    @classmethod
    def from_dict(cls, data: Dict[str, Any], source: str = "config") -> "ExperimentConfig":
        """Create config from a dictionary. An unknown key is an error: a misspelt key would otherwise fall back to
        its default without notice (``output_dir``, from the upstream configs, is still accepted and ignored)."""
        data = dict(data)
        data.pop("output_dir", None)
        unknown = sorted(set(data) - {f.name for f in dataclasses.fields(cls)})
        if unknown:
            raise ValueError(f"{source}: unknown config key(s) {unknown}; see scoring.experiment.ExperimentConfig")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (every field, in declaration order)."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


def add_override_args(ap: argparse.ArgumentParser) -> None:
    """The CLI flags `apply_overrides` reads (config precedence: YAML < CLI)."""
    ap.add_argument("--model", default=None, help="Overrides the config's model_path (e.g. an 8B RM on a 0.6B config)")
    ap.add_argument("--revision", default=None,
                    help="Overrides the config's model_revision (default: the pinned one, else the Hub's main)")
    ap.add_argument("--batch-size", type=int, default=None, help="Overrides the config's batch_size")
    ap.add_argument("--device", default=None, help="Overrides the config's device (e.g. cpu, mps)")
    ap.add_argument("--probe-records", type=int, default=None, help="Overrides the config's probe_records")


def apply_overrides(cfg: "ExperimentConfig", args: argparse.Namespace) -> "ExperimentConfig":
    """The CLI over the config (config precedence: YAML < CLI), for the flags of `add_override_args`; a flag
    that was not given (None, or absent from ``args``) leaves the config's value."""
    for attr, flag in (("device", "device"), ("model_path", "model"), ("batch_size", "batch_size"),
                       ("probe_records", "probe_records"), ("model_revision", "revision")):
        value = getattr(args, flag, None)
        if value is not None:
            setattr(cfg, attr, value)
    return cfg


def data_file(path: Path | str) -> Dict[str, Any]:
    """Identify a generator output a run reads (``pairs.jsonl``, ``cells.jsonl``): its path and SHA-256, checked
    against the ``files`` entry of the ``manifest.json`` next to it, plus the manifest's generator version, code
    commit and seed. Raises if the manifest is missing, predates ``files`` (generator < 0.4.0) or describes other
    bytes: the file and its manifest then come from different builds. Call it before loading the model."""
    from pairs.manifest import file_sha256

    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{path}: no manifest.json next to it; regenerate the data")
    manifest = json.loads(manifest_path.read_text())
    entry = (manifest.get("files") or {}).get(path.name)
    if entry is None:
        raise ValueError(f"{manifest_path} (generator {manifest.get('generator_version')}) does not list "
                         f"{path.name}; it predates the file hashes (0.4.0): regenerate the data")
    digest = file_sha256(path)
    if digest != entry["sha256"]:
        raise ValueError(f"{path} (SHA-256 {digest[:12]}) is not the file its manifest describes "
                         f"({entry['sha256'][:12]}): they come from different builds; regenerate the data")
    return {"path": str(path), "sha256": digest, "n_rows": entry["n_rows"],
            "generator_version": manifest.get("generator_version"), "generator_code": manifest.get("code"),
            "generator_seed": manifest.get("seed")}


def run_metadata(cfg: "ExperimentConfig", data: Dict[str, Dict[str, Any]], settings: Dict[str, Any]) -> Dict[str, Any]:
    """What produced a result file: the config (after `BiasExperiment.load_model`, so ``model_revision`` is the
    Hub commit the weights came from), the code commit this runner ran from, the data files (`data_file`), the
    runner's settings, the command line and the time. Every runner writes it under ``meta``."""
    from pairs.manifest import code_provenance

    return {"config": cfg.to_dict(), "code": code_provenance(), "data": data, "settings": settings,
            "argv": list(sys.argv), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


class BiasExperiment:
    """One runner's model and tokenizer, loaded for a config."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None

    def load_model(self) -> None:
        """Load the reward model and tokenizer (`scoring.backend.create_backend`), verify the pipeline's score
        path against the model's own score, attach the embedding cache, and record the Hub commit the weights
        came from in ``config.model_revision`` (so every result that carries the config names its weights)."""
        from scoring.backend import create_backend, loaded_revision

        from probes.embedding_cache import attach
        from probes.probe import verify_score_path

        logger.info("Loading model from %s", self.config.model_path)
        self.model, self.tokenizer = create_backend(self.config)
        commit = loaded_revision(self.model)
        if commit:
            logger.info("%s: weights from Hub commit %s", self.config.model_path, commit)
            self.config.model_revision = commit
        # Refuse, loudly, a model whose score the pipeline's reward path does not reproduce (before
        # any state is cached for it).
        gap = verify_score_path(self.model, self.tokenizer, self.config.max_length)
        logger.info("Score path verified: pipeline reward == model score (max |diff| %.4f)", gap)
        attach(self.model, self.tokenizer, self.config.embedding_cache_dir)
