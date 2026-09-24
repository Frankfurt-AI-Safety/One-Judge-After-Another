"""
Probe direction building from contrastive pairs.

Computes the bias direction using difference-of-means:
    probe = mean(positive_embeddings) - mean(negative_embeddings)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scoring.dataset_base import ContrastivePair
from probes.embedding_cache import CACHE_ATTR, text_key

logger = logging.getLogger(__name__)


def tokenize_inputs(
    tokenizer: AutoTokenizer,
    texts: List,
    padding: bool = True,
    truncation: bool = True,
    max_length: int = 2048,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """Tokenize inputs, handling both single strings and (prompt, response) pairs.
    
    Args:
        tokenizer: HuggingFace tokenizer
        texts: List of strings OR list of (prompt, response) tuples
        padding: Whether to pad
        truncation: Whether to truncate
        max_length: Maximum sequence length
        return_tensors: Return format
        
    Returns:
        Tokenized inputs dict
    """
    if not texts:
        raise ValueError("No texts provided")
    
    # Check if inputs are pairs (tuples) or single strings
    first = texts[0]
    if isinstance(first, tuple) and len(first) == 2:
        # Pair format: tokenizer(text_a, text_b)
        texts_a = [t[0] for t in texts]
        texts_b = [t[1] for t in texts]
        return tokenizer(
            texts_a,
            texts_b,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
    else:
        # Single string format
        return tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )


def gram_schmidt(vectors: List[torch.Tensor]) -> torch.Tensor:
    """Orthogonalize vectors using Gram-Schmidt (returns an orthonormal basis).
    
    Args:
        vectors: List of 1D tensors [d]
        
    Returns:
        Orthonormal basis matrix [k, d] (k <= len(vectors))
    """
    if not vectors:
        raise ValueError("No vectors provided to gram_schmidt")
    
    basis: List[torch.Tensor] = []
    for v in vectors:
        v = v.float()
        for b in basis:
            v = v - (v @ b) * b
        norm = v.norm()
        if norm > 1e-8:
            basis.append(v / norm)
    
    # If all were near-zero, return empty basis with correct dim
    if not basis:
        return torch.zeros(0, vectors[0].shape[0])
    
    return torch.stack(basis, dim=0)


def project_to_null_space(hidden: torch.Tensor, basis: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Project hidden states to the null space of a basis (remove subspace components).
    
    Uses Gram-Schmidt to orthonormalize the basis, then projects out each direction
    scaled by ``alpha`` (1.0 = full removal, 0.0 = no change).
    
    Args:
        hidden: [batch, d] hidden states to project
        basis: [d] single vector OR [k, d] multiple vectors (need not be orthonormal)
        alpha: Nullification strength in [0, 1].  1.0 = full projection (default).
        
    Returns:
        [batch, d] with the basis subspace removed.
    """
    # Handle 1D input: [d] -> [1, d]
    if basis.dim() == 1:
        basis = basis.unsqueeze(0)
    
    if basis.shape[0] == 0:
        return hidden
    
    # Orthonormalize using Gram-Schmidt (handles near-colinear vectors)
    ortho_basis = gram_schmidt([v for v in basis])  # [k', d] where k' <= k
    
    if ortho_basis.shape[0] == 0:
        return hidden
    
    # Project onto orthonormal span and remove (scaled by alpha)
    ortho_basis = ortho_basis.to(hidden.device).float()
    coeffs = hidden.float() @ ortho_basis.T  # [batch, k']
    projection = coeffs @ ortho_basis         # [batch, d]
    
    return hidden - (alpha * projection).to(hidden.dtype)


def get_base_model(model: AutoModelForSequenceClassification):
    """Extract base transformer from reward model wrapper.
    
    Args:
        model: Reward model (AutoModelForSequenceClassification)
        
    Returns:
        Base transformer model
        
    Raises:
        ValueError: If base model cannot be found
    """
    for attr in ["model", "transformer", "base_model"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError(f"Cannot find base model in {type(model)}")


def get_score_head(model: AutoModelForSequenceClassification) -> torch.nn.Linear:
    """Extract the score/classification head from reward model.
    
    Args:
        model: Reward model
        
    Returns:
        Linear layer that maps hidden states to scores
        
    Raises:
        ValueError: If score head cannot be found
    """
    for attr in ["score", "classifier", "out_proj", "head"]:
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, torch.nn.Linear):
                return layer
    
    # Fallback: find any Linear with output_features=1
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.out_features == 1:
            return module
    
    raise ValueError("Could not find score head in model")


def project_onto_null(u: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project vector u onto the null space of basis vectors.
    
    Returns u with all components along basis vectors removed.
    
    Args:
        u: Vector to project [d]
        basis: Basis vectors to project out [k, d] or [d] for single vector
        
    Returns:
        Cleaned vector [d]
    """
    if basis.dim() == 1:
        basis = basis.unsqueeze(0)
    
    result = u.clone()
    for v in basis:
        v_norm_sq = torch.dot(v, v)
        if v_norm_sq > 1e-10:
            result = result - (torch.dot(result, v) / v_norm_sq) * v
    return result


def clean_probe(probe: torch.Tensor, nuisance_probes: List[torch.Tensor]) -> torch.Tensor:
    """Clean a probe by removing components along nuisance directions.
    
    Args:
        probe: The probe to clean [d]
        nuisance_probes: List of probes to project out
        
    Returns:
        Cleaned probe (renormalized) [d]
    """
    if not nuisance_probes:
        return probe
    basis = torch.stack([p.to(probe.device) for p in nuisance_probes], dim=0)
    cleaned = project_onto_null(probe, basis)
    return cleaned / (cleaned.norm() + 1e-8)


def clean_probe_with_correctness(
    *,
    probe: torch.Tensor,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    correctness_pairs: List[ContrastivePair],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
    """Clean a probe by projecting out a correctness direction learned from contrastive pairs.
    
    This is a common pattern across multiple experiments:
    1) Learn a "correct vs incorrect" probe direction
    2) Remove that direction from the target bias probe
    
    Args:
        probe: Target probe to clean [d]
        model: Reward model
        tokenizer: Tokenizer
        correctness_pairs: Contrastive pairs defining correctness direction
        batch_size: Batch size for embedding extraction
        device: Device to use
        max_length: Maximum sequence length
        
    Returns:
        Tuple of:
        - cleaned_probe: Cleaned probe (renormalized) [d]
        - metadata: Dict with correctness probe stats and overlap information
        - correctness_probe: The learned correctness direction [d]
    """
    if len(correctness_pairs) == 0:
        raise ValueError("No correctness pairs provided; cannot clean probe.")
    
    correctness_probe, corr_metadata = build_probe_direction(
        model=model,
        tokenizer=tokenizer,
        contrastive_pairs=correctness_pairs,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    
    original_probe = probe.clone()
    cleaned_probe = clean_probe(probe, [correctness_probe])
    
    overlap = torch.dot(
        original_probe, correctness_probe.to(original_probe.device)
    ).abs().item()
    
    metadata = {
        "cleaned_with_correctness": True,
        "correctness_overlap": overlap,
        "correctness_probe_accuracy": corr_metadata.get("probe_accuracy", 0),
    }
    
    return cleaned_probe, metadata, correctness_probe


def clean_probe_basis_with_correctness(
    *,
    probes: List[torch.Tensor],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    correctness_pairs: List[ContrastivePair],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
    """Clean multiple probe directions by projecting out a correctness direction.
    
    Typical use: build several bias directions (e.g. position A-vs-rest, B-vs-rest, ...)
    and remove correctness, then orthonormalize and null the full subspace.
    
    Args:
        probes: List of raw probe directions [d]
        model/tokenizer/correctness_pairs: for learning correctness direction
        
    Returns:
        Tuple of:
        - basis: Orthonormal basis [k, d] after cleaning and Gram-Schmidt
        - metadata: cleaning metadata (includes per-vector overlaps)
        - correctness_probe: learned correctness direction [d]
    """
    if not probes:
        raise ValueError("No probe directions provided; cannot clean basis.")
    
    # Learn correctness direction once
    correctness_probe, corr_metadata = build_probe_direction(
        model=model,
        tokenizer=tokenizer,
        contrastive_pairs=correctness_pairs,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    
    # Clean each probe direction against correctness
    overlaps: List[float] = []
    cleaned_list: List[torch.Tensor] = []
    for v in probes:
        v = v.to(correctness_probe.device)
        overlaps.append(float(torch.dot(v, correctness_probe.to(v.device)).abs().item()))
        cleaned_list.append(clean_probe(v, [correctness_probe]))
    
    # Orthonormalize the cleaned directions
    basis = gram_schmidt(cleaned_list)
    
    metadata: Dict[str, Any] = {
        "cleaned_with_correctness": True,
        "correctness_probe_accuracy": corr_metadata.get("probe_accuracy", 0),
        "correctness_overlap_mean": float(sum(overlaps) / max(len(overlaps), 1)),
        "correctness_overlap_per_direction": overlaps,
        "n_basis_vectors": int(basis.shape[0]),
    }
    
    return basis, metadata, correctness_probe


class ScorePathMismatch(RuntimeError):
    """The pipeline's reward (linear head on the pooled last-token state) is not the model's score."""


_CHECK_TEXTS = ("The applicant has a stable income and repaid every earlier loan on time.",
                "A short essay.")


def verify_score_path(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer,
                      max_length: int = 2048) -> float:
    """Refuse a model whose score the pipeline cannot reproduce. Every reward here is the score head
    applied to the pooled LAST-token state of ``hidden_states[-1]`` (so that the probe subspace can be
    projected out right before the head). That is how Llama/Qwen/Gemma sequence-classification RMs
    score, but not every RM: DeBERTa scores the FIRST token through a ``ContextPooler`` (dense +
    activation) before its classifier, so the pipeline's "rewards" for it were unrelated to the model's
    scores — silently. This compares the pipeline path with the model's own ``logits`` on two texts of
    different length (bf16 tolerance) and raises `ScorePathMismatch` if they disagree. Returns the
    largest absolute difference."""
    model.eval()
    base, _ = get_rewards_both(model, tokenizer, list(_CHECK_TEXTS), None, batch_size=2,
                               max_length=max_length, show_progress=False)
    base_model = get_base_model(model)
    inputs = tokenize_inputs(tokenizer, list(_CHECK_TEXTS), max_length=max_length)
    inputs = {k: v.to(next(base_model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    if logits.dim() > 1 and logits.shape[-1] != 1:
        raise ScorePathMismatch(f"{type(model).__name__} outputs {logits.shape[-1]} scores per text; "
                                f"the pipeline assumes one scalar reward")
    logits = logits.reshape(-1).float().cpu()
    diff = (base.float() - logits).abs()
    if bool((diff > 0.02 + 0.02 * logits.abs()).any()):
        raise ScorePathMismatch(
            f"{type(model).__name__}: the pipeline's reward (score head on the pooled last-token state) "
            f"does not reproduce the model's own score (pipeline {base.float().tolist()}, model "
            f"{logits.tolist()}). This model pools differently (e.g. DeBERTa's first-token ContextPooler) "
            f"or has a custom head; its baseline and nulled rewards would be wrong. It needs its own "
            f"pooling and projection site before it can be run.")
    return float(diff.max())


def _forward_pooled(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    show_progress: bool,
) -> Tuple[torch.Tensor, torch.dtype]:
    """The one forward pass of the pipeline: float32 [n, d] pooled last-token states of
    ``hidden_states[-1]`` (right padding; see `scoring/backend.py`), plus the dtype the model computed
    them in. Tokenizes on the fly per batch to avoid OOM on large datasets."""
    model.eval()
    all_embeddings = []
    state_dtype = next(model.parameters()).dtype
    base_model = get_base_model(model)
    n_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info("Processing %d texts in %d batches (tokenizing on-the-fly)...", len(texts), n_batches)
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting embeddings", total=n_batches)

    with torch.no_grad():
        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            # Tokenize just this batch (handles both strings and pairs)
            inputs = tokenize_inputs(
                tokenizer,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            # With device_map="auto" the first layer may not be on `device`;
            # always send inputs to wherever the first layer actually lives.
            input_device = next(base_model.parameters()).device
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            outputs = base_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
            state_dtype = hidden_states.dtype

            # Get last non-padding token for each example.
            # hidden_states may be on a different GPU than inputs (device_map="auto"),
            # so derive the indexing device from the tensor itself.
            attention_mask = inputs["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1
            hs_device = hidden_states.device
            batch_embeddings = hidden_states[
                torch.arange(hidden_states.size(0), device=hs_device),
                last_token_indices.to(hs_device),
            ]
            all_embeddings.append(batch_embeddings.float().cpu())

    return torch.cat(all_embeddings, dim=0), state_dtype


def _embed(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    show_progress: bool,
) -> Tuple[torch.Tensor, torch.dtype]:
    """`get_embeddings` plus the state dtype, served from the model's embedding cache when one is
    attached (`probes/embedding_cache.py`): only texts never embedded under this `max_length` go
    through the model, each unique text once even if it repeats within the call."""
    cache = getattr(model, CACHE_ATTR, None)
    if cache is None:
        return _forward_pooled(model, tokenizer, texts, batch_size, max_length, show_progress)
    keys = [text_key(t, max_length) for t in texts]
    todo: Dict[str, Any] = {}
    for key, text in zip(keys, texts):
        if key not in cache and key not in todo:
            todo[key] = text
    cache.hits += len(texts) - len(todo)
    cache.misses += len(todo)
    if todo:
        states, dtype = _forward_pooled(model, tokenizer, list(todo.values()), batch_size,
                                        max_length, show_progress)
        cache.put(list(todo), states.to(dtype))  # float32 -> native dtype: exact
        cache.flush()
    logger.info("Embedding cache: %d of %d texts served from cache", len(texts) - len(todo), len(texts))
    return torch.stack([cache.get(k) for k in keys]).float(), cache.state_dtype


def embed_states(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 2048,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.dtype]:
    """Float32 pooled states plus the dtype the score head expects: the input of `rewards_from_hidden`,
    for scoring one embedding pass under several probes or α values (cache-aware, see `_embed`)."""
    return _embed(model, tokenizer, texts, batch_size, max_length, show_progress)


def get_embeddings(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
    show_progress: bool = True,
) -> torch.Tensor:
    """Extract last-token hidden state embeddings from reward model.

    Served from the model's embedding cache when one is attached (see `_embed`).

    Args:
        model: Reward model
        tokenizer: Tokenizer
        texts: List of formatted conversation texts
        batch_size: Batch size for inference
        device: Device to use
        max_length: Maximum sequence length
        show_progress: Whether to show progress bar

    Returns:
        [n_texts, hidden_dim] tensor of embeddings

    Raises:
        ValueError: If texts is empty (no embeddings to extract)
    """
    if not texts:
        raise ValueError(
            "Cannot extract embeddings: no texts provided. "
            "This usually means a dataset position/category has no examples after filtering. "
            "Check that your dataset has enough examples and the parser is working correctly."
        )

    # NOTE: a ModelBackend dispatch used to sit here so the MLX (Apple Silicon) backend
    # could intercept this call. MLX was removed once CUDA was validated on the cluster,
    # and it was the only implementation, so the abstraction went with it. This function
    # now always takes the raw Hugging Face model.
    return _embed(model, tokenizer, texts, batch_size, max_length, show_progress)[0]


def rewards_from_hidden(
    model: AutoModelForSequenceClassification,
    hidden: torch.Tensor,
    state_dtype: torch.dtype,
    probe: Optional[torch.Tensor] = None,
    null_alpha: float = 1.0,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(baseline, nulled) rewards from float32 pooled states: the score head applied to the state, and
    to the state with the probe subspace projected out (scaled by ``null_alpha``). Same arithmetic, in
    the same order, as the reward code has always used: project in float32 on the head's device, cast
    to the model's dtype, apply the head. Cheap — call it once per α instead of re-running the model."""
    score_head = get_score_head(model)
    score_device = score_head.weight.device
    base_out, null_out = [], []
    with torch.no_grad():
        for start in range(0, hidden.shape[0], batch_size):
            h = hidden[start:start + batch_size].to(score_device)
            base = score_head(h.to(state_dtype)).squeeze(-1)
            base_out.append(base.cpu())
            if probe is not None:
                nulled = score_head(project_to_null_space(h, probe, alpha=null_alpha)
                                    .to(state_dtype)).squeeze(-1)
                null_out.append(nulled.cpu())
            else:
                null_out.append(base.cpu())
    return torch.cat(base_out, dim=0), torch.cat(null_out, dim=0)


def build_probe_direction(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    contrastive_pairs: List[ContrastivePair],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build probe direction using difference-of-means.
    
    The probe direction points from negative to positive:
        probe = mean(positive_embeddings) - mean(negative_embeddings)
    
    Args:
        model: Reward model
        tokenizer: Tokenizer
        contrastive_pairs: List of ContrastivePair objects
        batch_size: Batch size for embedding extraction
        device: Device to use
        max_length: Maximum sequence length
        
    Returns:
        Tuple of:
        - probe: [hidden_dim] normalized probe direction tensor
        - metadata: Dictionary with probe statistics
    """
    positive_texts = [pair.positive_text for pair in contrastive_pairs]
    negative_texts = [pair.negative_text for pair in contrastive_pairs]
    
    logger.info("Extracting embeddings for %d positive examples", len(positive_texts))
    positive_emb = get_embeddings(
        model, tokenizer, positive_texts, batch_size, device, max_length
    )
    
    logger.info("Extracting embeddings for %d negative examples", len(negative_texts))
    negative_emb = get_embeddings(
        model, tokenizer, negative_texts, batch_size, device, max_length
    )
    
    # Compute means
    positive_mean = positive_emb.mean(dim=0)
    negative_mean = negative_emb.mean(dim=0)
    
    # Difference of means
    probe = positive_mean - negative_mean
    raw_norm = probe.norm().item()
    probe = probe / (probe.norm() + 1e-8)
    
    # Compute statistics
    positive_proj = positive_emb @ probe
    negative_proj = negative_emb @ probe
    
    # Classification accuracy at optimal threshold
    threshold = (positive_proj.mean() + negative_proj.mean()) / 2
    correct = (positive_proj > threshold).sum() + (negative_proj <= threshold).sum()
    accuracy = float(correct) / (len(positive_proj) + len(negative_proj))
    
    metadata = {
        "n_positive": len(positive_texts),
        "n_negative": len(negative_texts),
        "hidden_dim": int(probe.shape[0]),
        "positive_mean_norm": float(positive_mean.norm()),
        "negative_mean_norm": float(negative_mean.norm()),
        "probe_raw_norm": raw_norm,
        "positive_proj_mean": float(positive_proj.mean()),
        "positive_proj_std": float(positive_proj.std()),
        "negative_proj_mean": float(negative_proj.mean()),
        "negative_proj_std": float(negative_proj.std()),
        "separation": float(positive_proj.mean() - negative_proj.mean()),
        "probe_accuracy": accuracy,
    }
    
    logger.info("Probe statistics:")
    logger.info("  Hidden dim: %d", metadata["hidden_dim"])
    logger.info("  Separation: %.4f", metadata["separation"])
    logger.info("  Probe accuracy: %.2f%%", 100 * metadata["probe_accuracy"])
    
    return probe, metadata


def get_rewards_with_nulling(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    probe: Optional[torch.Tensor] = None,
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
    show_progress: bool = True,
    null_alpha: float = 1.0,
) -> torch.Tensor:
    """Compute reward scores with optional null-space projection.
    
    If probe is provided, projects hidden states onto the null space of the
    probe before computing scores.
    Tokenizes on-the-fly per batch to avoid OOM on large datasets.
    
    Args:
        model: Reward model
        tokenizer: Tokenizer  
        texts: List of formatted conversation texts
        probe: Optional probe direction tensor [hidden_dim] or [k, hidden_dim]
        batch_size: Batch size for inference
        device: Device to use
        max_length: Maximum sequence length
        show_progress: Whether to show progress bar
        null_alpha: Nullification strength (0=no change, 1=full projection).
        
    Returns:
        [n_texts] tensor of reward scores
    """
    # NOTE: a ModelBackend dispatch used to sit here so the MLX (Apple Silicon) backend
    # could intercept this call. MLX was removed once CUDA was validated on the cluster,
    # and it was the only implementation, so the abstraction went with it. This function
    # now always takes the raw Hugging Face model.

    if probe is not None:
        # The nulled branch is exactly get_rewards_both's nulled reward (and shares its cache).
        return get_rewards_both(model, tokenizer, texts, probe, batch_size, device, max_length,
                                show_progress, null_alpha)[1]

    # Without a probe: the model's own forward (HF pooling + head), uncached as before.
    model.eval()
    base_model = get_base_model(model)
    all_scores = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info("Processing %d texts in %d batches (tokenizing on-the-fly)...", len(texts), n_batches)
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing rewards", total=n_batches)
    with torch.no_grad():
        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            inputs = tokenize_inputs(tokenizer, texts[start:end], padding=True, truncation=True,
                                     max_length=max_length, return_tensors="pt")
            input_device = next(base_model.parameters()).device
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
            all_scores.append(model(**inputs).logits.squeeze(-1).cpu())
    return torch.cat(all_scores, dim=0)


def get_rewards_both(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    probe: Optional[torch.Tensor] = None,
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
    show_progress: bool = True,
    null_alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute BOTH baseline and nulled rewards in a single forward pass.
    
    This is 2x faster than calling get_rewards_with_nulling twice.
    Tokenizes on-the-fly per batch to avoid OOM on large datasets.
    
    Args:
        model: Reward model
        tokenizer: Tokenizer  
        texts: List of formatted conversation texts
        probe: Probe direction tensor [hidden_dim] or [k, hidden_dim]
        batch_size: Batch size for inference
        device: Device to use
        max_length: Maximum sequence length
        show_progress: Whether to show progress bar
        null_alpha: Nullification strength (0=no change, 1=full projection).
        
    Returns:
        Tuple of (baseline_rewards, nulled_rewards) tensors, each [n_texts]
    """
    # NOTE: a ModelBackend dispatch used to sit here so the MLX (Apple Silicon) backend
    # could intercept this call. MLX was removed once CUDA was validated on the cluster,
    # and it was the only implementation, so the abstraction went with it. This function
    # now always takes the raw Hugging Face model.
    #
    # One forward pass per text (served from the embedding cache when attached), then the score head
    # twice: the pooled state as-is, and with the probe subspace projected out.
    if not texts:
        empty = torch.zeros(0)
        return empty, empty
    hidden, state_dtype = _embed(model, tokenizer, texts, batch_size, max_length, show_progress)
    return rewards_from_hidden(model, hidden, state_dtype, probe, null_alpha)
