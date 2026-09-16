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
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)

__all__ = ["create_backend"]


def _load_transformers(config: Any) -> Tuple[Any, Any]:
    """Load an HF sequence-classification reward model + tokenizer.

    This reproduces the original ``BiasExperiment.load_model`` body exactly so the
    CUDA/CPU paths are byte-for-byte unchanged.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    # .to() intentionally omitted: device_map handles placement.

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
