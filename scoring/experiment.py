"""
Experiment configuration and model loading, shared by every runner.

`ExperimentConfig` is the per-experiment YAML (``configs/*.yaml``; CLI overrides are applied by each runner).
`BiasExperiment.load_model` loads the reward model (`scoring.backend`), verifies that the pipeline reproduces
the model's own score, attaches the embedding cache and records the Hub commit the weights came from.

The runners do the rest themselves: the direct arm is `runners/run_battery.py`, the cross-marker design
`runners/run_cross_marker.py`, and so on. An evaluation pipeline used to live here too (``evaluate``/``run``,
``ExperimentResults``, plots and raw-data dumps, a cross-dataset probe source), driven by
``runners/run_experiment.py``; that runner never called it (it exited after building the config, from the
initial commit on), `run_battery.py` computes a superset of its output, and it was removed on 2026-09-28.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
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
