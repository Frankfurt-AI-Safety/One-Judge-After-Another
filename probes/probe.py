"""
The model side of the pipeline: one forward pass per text, the reward from the pooled state, the null-space
projection between the two, and the direct arm's difference-of-means direction.

- `embed_states` / `embed_with_gates` / `get_embeddings` — the pooled last-token state of the final hidden layer
  (right padding, `scoring/backend.py`), served from the embedding cache when one is attached
  (`probes/embedding_cache.py`); for a gated head (QRM) also the per-text gates.
- `project_to_null_space` — ``h − α (h·Bᵀ) B`` for an orthonormal basis B of one or more directions.
- `rewards_from_hidden` / `get_rewards_both` — (baseline, nulled) rewards: the score head (`probes/heads.py`)
  applied to the state and to the projected state. Head weights are never modified.
- `build_probe_direction` — ``mean(h(positive)) − mean(h(negative))``, normalised; the demographic datasets put
  side A (pole A) on the positive side, so the direction points to pole A.
- `verify_score_path` — run at every model load; refuses a model whose score this path does not reproduce.

Every input is tokenized whole: a text longer than ``max_length`` is refused (`InputTooLong`), never truncated.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scoring.dataset_base import ContrastivePair, add_special_tokens, format_conversation
from probes.embedding_cache import CACHE_ATTR, FLUSH_EVERY, text_key
from probes.heads import get_head

logger = logging.getLogger(__name__)


class InputTooLong(ValueError):
    """A text is longer than ``max_length`` tokens (see `tokenize_inputs`)."""


def tokenize_inputs(tokenizer: AutoTokenizer, texts: List, max_length: int = 2048) -> Dict[str, torch.Tensor]:
    """Tokenize chat-formatted strings or (prompt, response) tuples, padded, as PyTorch tensors, **without
    truncation**; special tokens as `scoring.dataset_base.add_special_tokens` says, in both formats.

    A text longer than ``max_length`` tokens raises `InputTooLong` instead of being cut. Truncation would cut
    the end of the conversation (or, under a left truncation side, the header with the marker); and since the
    two sides of a pair differ in length by their markers, it would cut them at different points of the text, so
    the pair would no longer differ by the marker alone and the pooled state would not be the end of the
    conversation. Callers that can meet long inputs select them out first (`pairs.cross_marker.fits_max_length`).
    """
    if not texts:
        raise ValueError("No texts provided")
    kwargs = dict(padding=True, truncation=False, return_tensors="pt",
                  add_special_tokens=add_special_tokens(tokenizer))
    first = texts[0]
    if isinstance(first, tuple) and len(first) == 2:
        inputs = tokenizer([t[0] for t in texts], [t[1] for t in texts], **kwargs)
    else:
        inputs = tokenizer(texts, **kwargs)
    lengths = inputs["attention_mask"].sum(dim=1)
    n_long = int((lengths > max_length).sum())
    if n_long:
        raise InputTooLong(
            f"{n_long} of {len(texts)} inputs exceed max_length={max_length} tokens (longest {int(lengths.max())}). "
            f"Inputs are never truncated: that would cut the two sides of a pair at different points of the text. "
            f"Raise max_length or drop the long records before scoring.")
    return inputs


# A vector whose residual, after removing the basis built so far, is below this fraction of its own norm lies in
# that span already: the residual is rounding noise, and normalising it would add a row far from orthogonal to
# the others (measured: nulling [u, 3u] left 92% of the u component). A genuine near-duplicate (cos 0.99999)
# keeps a residual of 4.5e-3 and is kept.
_GS_RTOL = 1e-5


def gram_schmidt(vectors: List[torch.Tensor]) -> torch.Tensor:
    """Orthonormal basis [k, d] (k <= len(vectors)) of the span of `vectors`, in float32.

    Modified Gram–Schmidt, applied twice per vector (one pass loses orthogonality on nearly dependent inputs). A
    zero vector, or one whose residual is below `_GS_RTOL` of its norm (already in the span), is dropped. A single
    vector is only normalised.
    """
    if not vectors:
        raise ValueError("No vectors provided to gram_schmidt")

    basis: List[torch.Tensor] = []
    for v in vectors:
        v = v.float()
        norm0 = v.norm()
        if norm0 <= 1e-8:
            continue
        for _ in range(2):
            for b in basis:
                v = v - (v @ b) * b
        norm = v.norm()
        if norm > _GS_RTOL * norm0:
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


class ScorePathMismatch(RuntimeError):
    """The pipeline's reward (linear head on the pooled last-token state) is not the model's score."""


# (prompt, response): chat-formatted like every text the pipeline scores (a gated head reads the user turn)
_CHECK_CONVERSATIONS = (("Should this loan be approved?",
                         "The applicant has a stable income and repaid every earlier loan on time."),
                        ("Grade this essay.", "A short essay."))


def verify_score_path(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer,
                      max_length: int = 2048) -> float:
    """Refuse a model whose score the pipeline cannot reproduce. Every reward here is the score head
    applied to the pooled LAST-token state of the final hidden layer (so that the probe subspace can be
    projected out right before the head). That is how Llama/Qwen/Gemma sequence-classification RMs
    score, but not every RM: DeBERTa scores the FIRST token through a ``ContextPooler`` (dense +
    activation) before its classifier, so the pipeline's "rewards" for it were unrelated to the model's
    scores — silently. This compares the pipeline path with the model's own ``logits`` on two
    chat-formatted conversations of different length (bf16 tolerance) and raises `ScorePathMismatch` if
    they disagree; for a gated head (QRM) it also compares the pipeline's gates with the model's. Returns
    the largest absolute difference."""
    model.eval()
    texts = [format_conversation(tokenizer, prompt, response) for prompt, response in _CHECK_CONVERSATIONS]
    states, state_dtype, gates = _forward_pooled(model, tokenizer, texts, 2, max_length, False)
    base, _ = rewards_from_hidden(model, states, state_dtype, None, gates=gates)
    base_model = get_base_model(model)
    inputs = tokenize_inputs(tokenizer, texts, max_length=max_length)
    inputs = {k: v.to(next(base_model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    logits = output.logits
    if gates is not None:
        model_gates = output.gating_output.float().cpu()
        if not torch.allclose(gates, model_gates, atol=1e-3, rtol=1e-3):
            raise ScorePathMismatch(f"{type(model).__name__}: the pipeline's gates {gates.tolist()} differ from "
                                    f"the model's {model_gates.tolist()}")
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
) -> Tuple[torch.Tensor, torch.dtype, Optional[torch.Tensor]]:
    """The one forward pass of the pipeline: float32 [n, d] pooled last-token states of the final hidden
    layer (right padding; see `scoring/backend.py`), the dtype the model computed them in, and — for a gated
    head (QRM, `probes/heads.py`) — the float32 gates [n, objectives] from the same pass (None otherwise).
    Tokenizes on the fly per batch to avoid OOM on large datasets.

    The final layer is read as ``last_hidden_state``, the tensor the models' own heads read (HF
    sequence-classification heads, `scoring/qrm.py`, `scoring/logit_reward.py`); it is the same tensor as
    ``hidden_states[-1]`` (checked on Llama, Qwen3 and Gemma-2), but asking for ``output_hidden_states`` would
    keep every layer's states in GPU memory (~16 GB for the 70B models at batch 8 on 1,500-token texts)."""
    model.eval()
    all_embeddings = []
    all_gates: List[torch.Tensor] = []
    head = get_head(model)
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

            # Tokenize just this batch (handles both strings and pairs; refuses an over-long text)
            inputs = tokenize_inputs(tokenizer, batch_texts, max_length=max_length)
            # With device_map="auto" the first layer may not be on `device`;
            # always send inputs to wherever the first layer actually lives.
            input_device = next(base_model.parameters()).device
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            outputs = base_model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
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
            if head.gated:
                all_gates.append(head.gates_from_hidden(hidden_states, inputs["input_ids"]))

    gates = torch.cat(all_gates, dim=0) if head.gated else None
    return torch.cat(all_embeddings, dim=0), state_dtype, gates


def _embed(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    show_progress: bool,
) -> Tuple[torch.Tensor, torch.dtype, Optional[torch.Tensor]]:
    """`get_embeddings` plus the state dtype and, for a gated head, the gates (else None), served from
    the model's embedding cache when one is attached (`probes/embedding_cache.py`): only texts never
    embedded under this `max_length` go through the model, each unique text once even if it repeats
    within the call. A gated model's cache keeps the gates in its ``gates`` sub-cache, same keys.

    The texts to embed go through the model in chunks of `FLUSH_EVERY` rounded up to whole batches, and each
    chunk is written to disk before the next starts, so a crashed run keeps all but its last chunk. The batches
    are the ones an unchunked pass would form, so the states are the same bits."""
    cache = getattr(model, CACHE_ATTR, None)
    if cache is None:
        return _forward_pooled(model, tokenizer, texts, batch_size, max_length, show_progress)
    gate_cache = cache.gates
    keys = [text_key(t, max_length) for t in texts]
    todo: Dict[str, Any] = {}
    for key, text in zip(keys, texts):
        missing = key not in cache or (gate_cache is not None and key not in gate_cache)
        if missing and key not in todo:
            todo[key] = text
    cache.hits += len(texts) - len(todo)
    cache.misses += len(todo)
    todo_keys, todo_texts = list(todo), list(todo.values())
    chunk = -(-FLUSH_EVERY // batch_size) * batch_size
    for start in range(0, len(todo_keys), chunk):
        if len(todo_keys) > chunk:
            logger.info("Embedding texts %d-%d of %d", start + 1, min(start + chunk, len(todo_keys)),
                        len(todo_keys))
        part = slice(start, start + chunk)
        states, dtype, gates = _forward_pooled(model, tokenizer, todo_texts[part], batch_size,
                                               max_length, show_progress)
        cache.put(todo_keys[part], states.to(dtype))  # float32 -> native dtype: exact
        cache.flush()
        if gate_cache is not None:
            gate_cache.put(todo_keys[part], gates)     # float32, as the head multiplies them
            gate_cache.flush()
    logger.info("Embedding cache: %d of %d texts served from cache", len(texts) - len(todo), len(texts))
    gates_out = None if gate_cache is None else torch.stack([gate_cache.get(k) for k in keys]).float()
    return torch.stack([cache.get(k) for k in keys]).float(), cache.state_dtype, gates_out


def embed_states(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 2048,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.dtype]:
    """Float32 pooled states plus the dtype the score head expects — what directions are fitted on
    (cache-aware, see `_embed`). To score them with `rewards_from_hidden`, use `embed_with_gates`, which
    also returns the gates a gated head (QRM) needs."""
    return _embed(model, tokenizer, texts, batch_size, max_length, show_progress)[:2]


def embed_with_gates(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 2048,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.dtype, Optional[torch.Tensor]]:
    """`embed_states` plus the per-text gates of a gated head (None for a linear head): the full input of
    `rewards_from_hidden`, for scoring one embedding pass under several probes or α values."""
    return _embed(model, tokenizer, texts, batch_size, max_length, show_progress)


def get_embeddings(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 8,
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
        max_length: Maximum sequence length (a longer text raises `InputTooLong`)
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
    return _embed(model, tokenizer, texts, batch_size, max_length, show_progress)[0]


def rewards_from_hidden(
    model: AutoModelForSequenceClassification,
    hidden: torch.Tensor,
    state_dtype: torch.dtype,
    probe: Optional[torch.Tensor] = None,
    null_alpha: float = 1.0,
    batch_size: int = 256,
    gates: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(baseline, nulled) rewards from float32 pooled states: the score head applied to the state, and
    to the state with the probe subspace projected out (scaled by ``null_alpha``). Same arithmetic, in
    the same order, as the reward code has always used: project in float32 on the head's device, cast
    to the model's dtype, apply the head. Cheap — call it once per α instead of re-running the model.

    A gated head (QRM) needs ``gates``, one row per state (`embed_with_gates`); only the state is
    projected, the gate is held fixed (see `probes/heads.py`)."""
    head = get_head(model)
    if head.gated and gates is None:
        raise ValueError(f"{type(model).__name__} has a gated head: pass the states' gates "
                         f"(probes.probe.embed_with_gates) to rewards_from_hidden")
    score_device = head.device
    base_out, null_out = [], []
    with torch.no_grad():
        for start in range(0, hidden.shape[0], batch_size):
            h = hidden[start:start + batch_size].to(score_device)
            g = None if gates is None else gates[start:start + batch_size]
            base = head.score(h.to(state_dtype), g)
            base_out.append(base.cpu())
            if probe is not None:
                nulled = head.score(project_to_null_space(h, probe, alpha=null_alpha).to(state_dtype), g)
                null_out.append(nulled.cpu())
            else:
                null_out.append(base.cpu())
    return torch.cat(base_out, dim=0), torch.cat(null_out, dim=0)


def build_probe_direction(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    contrastive_pairs: List[ContrastivePair],
    batch_size: int = 8,
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
        max_length: Maximum sequence length (a longer text raises `InputTooLong`)

    Returns:
        Tuple of:
        - probe: [hidden_dim] normalized probe direction tensor
        - metadata: Dictionary with probe statistics, all **in-sample** (on the pairs the direction was fitted
          on). ``probe_accuracy`` classifies every state by the midpoint between the two sides' mean projections
          (not an optimised threshold). ``separation`` is the difference of those means, which for a
          difference-of-means direction equals ``probe_raw_norm`` (always >= 0); both are kept for the runners
          that report them.
    """
    positive_texts = [pair.positive_text for pair in contrastive_pairs]
    negative_texts = [pair.negative_text for pair in contrastive_pairs]

    logger.info("Extracting embeddings for %d positive examples", len(positive_texts))
    positive_emb = get_embeddings(
        model, tokenizer, positive_texts, batch_size, max_length
    )

    logger.info("Extracting embeddings for %d negative examples", len(negative_texts))
    negative_emb = get_embeddings(
        model, tokenizer, negative_texts, batch_size, max_length
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

    # In-sample accuracy at the midpoint threshold
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


def get_rewards_both(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    probe: Optional[torch.Tensor] = None,
    batch_size: int = 8,
    max_length: int = 2048,
    show_progress: bool = True,
    null_alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Baseline and nulled rewards from one forward pass per text.

    Tokenizes on-the-fly per batch to avoid OOM on large datasets.

    Args:
        model: Reward model
        tokenizer: Tokenizer
        texts: List of formatted conversation texts
        probe: Probe direction tensor [hidden_dim] or [k, hidden_dim]
        batch_size: Batch size for inference
        max_length: Maximum sequence length (a longer text raises `InputTooLong`)
        show_progress: Whether to show progress bar
        null_alpha: Nullification strength (0=no change, 1=full projection).

    Returns:
        Tuple of (baseline_rewards, nulled_rewards) tensors, each [n_texts]
    """
    # One forward pass per text (served from the embedding cache when attached), then the score head
    # twice: the pooled state as-is, and with the probe subspace projected out.
    if not texts:
        empty = torch.zeros(0)
        return empty, empty
    hidden, state_dtype, gates = _embed(model, tokenizer, texts, batch_size, max_length, show_progress)
    return rewards_from_hidden(model, hidden, state_dtype, probe, null_alpha, gates=gates)
