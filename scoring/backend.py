"""
Reward-model loading.

One backend: Hugging Face ``transformers``. The model is returned as the raw HF object,
exactly as the pipeline used it before backends existed.

An MLX (Apple Silicon) backend used to live alongside this, behind a ``ModelBackend``
abstraction, so the pipeline could be prototyped without GPUs. It was removed once the
hessian.AI cluster was validated on CUDA (see cluster/README.md): every reported number now
comes off CUDA, so a second numerical path was pure cost. The abstraction went with it,
because MLX was its only implementation.

``transformers`` is imported lazily inside :func:`create_backend` so importing this module
does not pull in the framework.

One checkpoint family is loaded with our own class instead of ``AutoModelForSequenceClassification``:
QRM (``Gemma2ForQuantileSequenceClassification``, `scoring/qrm.py`), whose remote code no longer imports
under current transformers and whose gate the pipeline needs to see.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from scoring.dataset_base import tokenization_vs_template, use_template_tokens

logger = logging.getLogger(__name__)

__all__ = ["create_backend", "PINNED_REVISIONS", "LOGIT_REWARD_MODELS", "model_revision"]

# Reward models packaged as causal LMs, with the vocabulary token whose logit at the last position is the reward
# (`scoring/logit_reward.py`). Only listed checkpoints are loaded this way; any other causal LM is refused.
LOGIT_REWARD_MODELS = {
    "nvidia/Llama-3.3-Nemotron-70B-Reward": 0,       # model card: generate(...).scores[0][0][0]
}

# Models whose authors score the chat template's own token ids (apply_chat_template(tokenize=True)) rather than
# the tokenizer's defaults on the formatted text; the pipeline then adds no special tokens (scoring.dataset_base).
# Checked against the model cards / evaluation code on 2026-09-26: Nemotron-70B-Reward's card tokenizes with the
# template (no BOS: its template has none); the Skywork cards and RewardBench (RB2) use the tokenizer's defaults,
# the pipeline's default.
TEMPLATE_TOKENIZED = {"nvidia/Llama-3.3-Nemotron-70B-Reward"}

# Checkpoints loaded from a pinned Hub revision rather than main. The AllenAI RB2 reward models publish only
# PyTorch .bin weights on main, which transformers >= 4.50 refuses to load under torch < 2.6 (CVE-2025-32434;
# the cluster image has torch 2.3). Hugging Face's conversion bot (SFconvertbot) opened pull requests with the
# same tensors as safetensors; these are their commits (checked 2026-09-26).
PINNED_REVISIONS = {
    "allenai/Llama-3.1-70B-Instruct-RM-RB2": "9c7bbb8e16516000b85d2ef21f8507b4d4c26952",   # refs/pr/2
    "allenai/Llama-3.1-8B-Instruct-RM-RB2": "792faf3b1621ae11366fcac9ee46f2bab6d08491",    # refs/pr/2
}


def model_revision(config: Any) -> Any:
    """The revision to load: the config's, else the pinned default, else None (the Hub's main)."""
    return getattr(config, "model_revision", None) or PINNED_REVISIONS.get(config.model_path)


def check_placement(model: Any) -> None:
    """Log where ``device_map="auto"`` put the model, and warn loudly when part of it landed on the CPU or
    disk: accelerate offloads silently when the GPUs are too small (a 70B on two 80 GB cards is close), and
    an offloaded model runs orders of magnitude slower without failing."""
    device_map = getattr(model, "hf_device_map", None) or {}
    if not device_map:
        return
    counts: dict = {}
    for device in device_map.values():
        counts[str(device)] = counts.get(str(device), 0) + 1
    logger.info("model placement (modules per device): %s", counts)
    offloaded = sorted(name for name, device in device_map.items() if str(device) in ("cpu", "disk"))
    if offloaded:
        logger.warning("%d module(s) OFFLOADED to CPU/disk (e.g. %s): the GPUs are too small for this model, "
                       "and every forward pass will be very slow", len(offloaded), offloaded[:3])


def _load_transformers(config: Any) -> Tuple[Any, Any]:
    """Load an HF sequence-classification reward model + tokenizer.

    This reproduces the original ``BiasExperiment.load_model`` body exactly so the
    CUDA/CPU paths are byte-for-byte unchanged.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    revision = model_revision(config)
    if revision:
        logger.info("%s at pinned revision %s", config.model_path, revision)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        revision=revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize as the model's authors score it (see TEMPLATE_TOKENIZED); report how that relates to the template.
    if config.model_path in TEMPLATE_TOKENIZED:
        use_template_tokens(tokenizer)
    relation = tokenization_vs_template(tokenizer)
    if relation == "mismatch":
        logger.warning("%s: the pipeline's token ids differ from the chat template's own on a sample conversation "
                       "(beyond a BOS); check how the model's authors tokenize", config.model_path)
    else:
        logger.info("%s: tokenization vs chat template: %s%s", config.model_path, relation,
                    " (template token ids, as its authors score)" if config.model_path in TEMPLATE_TOKENIZED else "")

    # Pin RIGHT padding. Every activation read in probe.py gathers the last real token via
    # `attention_mask.sum(dim=1) - 1`, which is only the last token under right padding; with
    # left padding that index lands mid-sequence on a pad and the scores are silently wrong.
    # The MLX backend and parity_check.py both pinned this explicitly; this path did
    # not, so it inherited each checkpoint's tokenizer default -- fine for Qwen3 ('right'), but
    # several Llama tokenizers default to 'left'. It also keeps the baseline branch, which uses
    # HF's own pad_token_id-based pooling, consistent with the nulled branch's manual gather.
    tokenizer.padding_side = "right"

    # "cuda" / "auto" → let HF shard the model across all visible GPUs.
    # An explicit single-device string (e.g. "cuda:0", "cpu") is passed
    # through unchanged so the caller retains control.
    device_map = "auto" if config.device in ("cuda", "auto") else config.device
    token = LOGIT_REWARD_MODELS.get(config.model_path)
    if token is not None:
        from transformers import AutoModelForCausalLM

        from scoring.logit_reward import LogitRewardModel

        logger.info("%s: causal-LM reward model, reward = logit of token %d at the last position",
                    config.model_path, token)
        causal = AutoModelForCausalLM.from_pretrained(
            config.model_path, revision=revision, dtype=torch.bfloat16, device_map=device_map)
        model = LogitRewardModel(causal, token)
    else:
        model_cls = _own_class(config.model_path, revision) or AutoModelForSequenceClassification
        kwargs = {} if model_cls is not AutoModelForSequenceClassification else {
            "trust_remote_code": config.trust_remote_code}
        model = model_cls.from_pretrained(
            config.model_path,
            revision=revision,
            dtype=torch.bfloat16,
            device_map=device_map,
            **kwargs,
        )
    # .to() intentionally omitted: device_map handles placement.
    check_placement(model)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def _own_class(model_path: str, revision: Any = None) -> Any:
    """Our implementation for a checkpoint whose architecture we carry (`scoring/qrm.py`), else None. Reads
    only the config (no remote code)."""
    from transformers import AutoConfig

    from scoring.qrm import ARCHITECTURE, Gemma2ForQuantileSequenceClassification

    architectures = getattr(AutoConfig.from_pretrained(model_path, revision=revision), "architectures", None) or []
    if any(a.endswith("ForCausalLM") for a in architectures):
        raise ValueError(f"{model_path} is a causal LM ({architectures}): loading it as a sequence classifier would "
                         f"attach a randomly initialised score head. If it is a reward model that reads its reward "
                         f"from one token's logit, add it with that token to scoring.backend.LOGIT_REWARD_MODELS.")
    if ARCHITECTURE in architectures:
        logger.info("%s: %s, loaded with scoring.qrm.Gemma2ForQuantileSequenceClassification (no remote code)",
                    model_path, ARCHITECTURE)
        return Gemma2ForQuantileSequenceClassification
    return None


def create_backend(config: Any) -> Tuple[Any, Any]:
    """``(model, tokenizer)`` for ``config`` — the entry point ``BiasExperiment.load_model`` calls.

    Restored 2026-09-23: removing the MLX backend also removed this dispatcher, while
    ``scoring/experiment.py`` still imported it, so every runner failed at ``load_model()`` with an
    ImportError. No test loaded a model, so it went unnoticed; ``tests/test_backend.py`` now does.
    """
    return _load_transformers(config)
