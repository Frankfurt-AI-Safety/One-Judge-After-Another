"""
Embedding cache: each unique text is embedded ONCE per reward model and environment.

The pipeline's expensive step is the forward pass that produces the pooled last-token hidden state
(`probes.probe.get_embeddings`). Everything after it — the difference-of-means probe, null-space
projection, the score head, the α-sweep, split-seed repeats, bootstrap resamples — is cheap arithmetic
on those states. Without a cache the same text was embedded repeatedly: the battery embeds each axis's
pairs separately although the single-axis pairs of a factorial are all cut from the same 8 cells
(~3.25x), and every α of the sweep re-ran the model. With it, a text costs one forward pass per model and
environment; the runners still load the model, but a repeated text never reaches it.

What is stored is exactly what the score head sees: the pooled last-token state of the final hidden
layer (``last_hidden_state``; right padding, see `scoring/backend.py`) in the model's compute dtype. The
reward code upcasts that bf16 state to float32 before projecting, which is exact, so storing it in bf16
loses nothing.

Keys and isolation. A state is keyed by (text, max_length) — from when inputs were truncated; since 2026-09-27
an over-long input is refused instead, and the key keeps max_length — within a directory named by a
**fingerprint** (`model_fingerprint`) of everything else that decides the state:

- the model: its path, Hub revision (which pins its config files), a few config fields, dtype, and a hash of the
  score head's weights (plus the gating network's for a gated head) — a different checkpoint or a re-download at
  a new revision is never served another model's states;
- the tokenizer: class, name, padding side, and whether special tokens are added;
- the environment (since 2026-09-27): device type and GPU name, attention implementation, torch and
  transformers versions. States computed under another of these differ at least at bf16 noise level (and by
  more under a library bug), so a Mac smoke run, a different GPU or a library upgrade gets its own cache.

Storage. Append-only shard files (``shard-*.pt``) written atomically; no database. The cluster writes to
a shared parallel filesystem (PFSS) where SQLite-style locking is unreliable, and concurrent jobs on the
same model must not corrupt each other: each process only ever creates its own shard files and reads the
shards that existed when it opened the directory. Shards are memory-mapped, so opening a large cache reads
only the states a run uses. `probes.probe._embed` writes a shard every `FLUSH_EVERY` new texts, so a crashed
job keeps all but its last chunk.

Determinism note. A state is reused as first computed. Recomputing the same text in a different batch
(other padding lengths) can differ at bf16 noise level — as the uncached pipeline already did between
runs — so the cache makes results *more* reproducible, not less.

Configure with the experiment config key ``embedding_cache_dir`` (default ``artifacts/embedding_cache``,
gitignored). Environment override ``ONEJUDGE_EMBED_CACHE``: ``off`` disables it, a path replaces the
directory.

Gated heads (QRM, `probes/heads.py`): the score also needs a per-text gate from the same forward pass,
kept in a ``gates/`` sub-cache under the same keys, and the fingerprint adds the gating network's digest.

LIMITATION (pre-existing, not introduced here): the reward path assumes the score is a head on the pooled
LAST-token state, as for the Llama/Qwen/Gemma sequence-classification RMs and QRM. That is not how every
RM computes its score (e.g. DeBERTa scores a pooled FIRST token through a pooler) — see the working notes.

The offline access at the end of the file (states, gates and the head read back without the model) is only
used by the tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

logger = logging.getLogger(__name__)

CACHE_ATTR = "_onejudge_embedding_cache"
ENV_VAR = "ONEJUDGE_EMBED_CACHE"
FORMAT_VERSION = 1
POOLING = "last_hidden_state, last non-pad token (right padding)"
# `probes.probe._embed` sends the texts a call has not cached through the model in chunks of this many
# (rounded up to whole batches) and writes a shard after each, so a crashed job keeps all but its last chunk.
FLUSH_EVERY = 4096

Text = Union[str, Tuple[str, str]]


def text_key(text: Text, max_length: int) -> str:
    """Stable key for one input under one max_length (tuples are (prompt, response) inputs)."""
    payload = json.dumps([max_length, list(text) if isinstance(text, tuple) else text],
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _environment(model: Any) -> Dict[str, Any]:
    """What else decides a state's bits: where and with which kernels and library versions it was computed."""
    import transformers

    device = next(model.parameters()).device
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.machine()          # cpu / mps: the processor architecture
    return {
        "device": device.type,
        "device_name": device_name,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def model_fingerprint(model: Any, tokenizer: Any) -> Dict[str, Any]:
    """Everything that decides the pooled state of a given text, apart from the text itself."""
    from probes.heads import get_head
    from scoring.dataset_base import add_special_tokens

    cfg = model.config
    head = get_head(model)
    fp = {
        "format_version": FORMAT_VERSION,
        "pooling": POOLING,
        "model_path": getattr(cfg, "_name_or_path", "") or "",
        "revision": getattr(cfg, "_commit_hash", None),
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "dtype": str(next(model.parameters()).dtype),
        "tokenizer": type(tokenizer).__name__,
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "add_special_tokens": add_special_tokens(tokenizer),
        "head_digest": head.digest(),
        "environment": _environment(model),
    }
    if head.gated:
        fp["head_kind"] = head.kind
        fp["gate_digest"] = head.gate_digest()
    return fp


def _slug(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.strip("/")).strip("_")[-60:] or "model"


class EmbeddingCache:
    """Pooled last-token states of one model, keyed by `text_key`. See the module docstring."""

    def __init__(self, directory: Union[str, Path], fingerprint: Optional[Dict[str, Any]] = None):
        self.directory = Path(directory)
        self.fingerprint = fingerprint
        self._shards: List[torch.Tensor] = []
        self._index: Dict[str, Tuple[int, int]] = {}
        self._pending_keys: List[str] = []
        self._pending: List[torch.Tensor] = []          # one tensor per put() call
        self._pending_at: Dict[str, Tuple[int, int]] = {}  # key -> (put index, row)
        self.state_dtype: Optional[torch.dtype] = None
        self.hits = 0
        self.misses = 0
        self.gates: Optional["EmbeddingCache"] = None   # a gated head's per-text gates, same keys
        self._load()

    # ------------------------------------------------------------------ disk
    def _load(self) -> None:
        meta_path = self.directory / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if self.fingerprint is not None and meta.get("fingerprint") != self.fingerprint:
                raise ValueError(f"embedding cache {self.directory} belongs to a different model "
                                 f"fingerprint; refusing to mix states")
            self.fingerprint = meta.get("fingerprint")
            if meta.get("state_dtype"):
                self.state_dtype = getattr(torch, meta["state_dtype"].replace("torch.", ""))
        for shard in sorted(self.directory.glob("shard-*.pt")):
            # memory-mapped: a shard's states are read from disk only where a run uses them (shards never change)
            data = torch.load(shard, map_location="cpu", weights_only=True, mmap=True)
            k = len(self._shards)
            self._shards.append(data["states"])
            for row, key in enumerate(data["keys"]):
                self._index.setdefault(key, (k, row))  # first computation wins

    def _write_meta(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        meta_path = self.directory / "meta.json"
        if not meta_path.exists():
            tmp = meta_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps({"fingerprint": self.fingerprint,
                                       "state_dtype": str(self.state_dtype)}, indent=2))
            os.replace(tmp, meta_path)

    def flush(self) -> None:
        """Write pending states as a new shard (atomic rename; unique name per process and call)."""
        if not self._pending_keys:
            return
        self._write_meta()
        states = torch.cat(self._pending, dim=0)
        name = f"shard-{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:8]}.pt"
        path = self.directory / name
        tmp = path.with_suffix(".tmp")
        torch.save({"version": FORMAT_VERSION, "keys": list(self._pending_keys), "states": states}, tmp)
        os.replace(tmp, path)
        k = len(self._shards)
        self._shards.append(states)
        for row, key in enumerate(self._pending_keys):
            self._index.setdefault(key, (k, row))
        self._pending_keys, self._pending, self._pending_at = [], [], {}

    # ------------------------------------------------------------------ access
    def __contains__(self, key: str) -> bool:
        return key in self._index or key in self._pending_at

    def __len__(self) -> int:
        """Distinct keys, stored or pending."""
        return len(self._index) + sum(1 for k in self._pending_at if k not in self._index)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """The stored state (native dtype) or None."""
        loc = self._index.get(key)
        if loc is not None:
            return self._shards[loc[0]][loc[1]]
        loc = self._pending_at.get(key)
        if loc is not None:
            return self._pending[loc[0]][loc[1]]
        return None

    def put(self, keys: Sequence[str], states: torch.Tensor) -> None:
        """Add states (any float dtype; stored in `state_dtype`, which the first put fixes), one row per key.
        A count mismatch would shift every later row of the shard onto another text, so it raises."""
        if len(keys) != states.shape[0]:
            raise ValueError(f"{len(keys)} keys for {states.shape[0]} states")
        if self.state_dtype is None:
            self.state_dtype = states.dtype
        block = len(self._pending)
        for row, key in enumerate(keys):
            self._pending_at.setdefault(key, (block, row))
        self._pending_keys.extend(keys)
        self._pending.append(states.detach().to("cpu", self.state_dtype))
        if len(self._pending_keys) >= FLUSH_EVERY:
            self.flush()


def resolve_directory(configured: Optional[str]) -> Optional[Path]:
    """The cache root after the environment override; None = disabled."""
    env = os.environ.get(ENV_VAR)
    if env is not None:
        return None if env.strip().lower() in ("off", "0", "false", "none", "") else Path(env)
    return Path(configured) if configured else None


def attach(model: Any, tokenizer: Any, root: Optional[str]) -> Optional[EmbeddingCache]:
    """Attach a cache to `model` (so `probes.probe` finds it), in ``root/<model>--<fingerprint>``."""
    directory = resolve_directory(root)
    if directory is None:
        setattr(model, CACHE_ATTR, None)
        logger.info("Embedding cache disabled")
        return None
    fp = model_fingerprint(model, tokenizer)
    digest = hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:12]
    cache = EmbeddingCache(directory / f"{_slug(fp['model_path'])}--{digest}", fp)
    if fp.get("head_kind"):
        cache.gates = EmbeddingCache(cache.directory / "gates", fp)
    save_head(cache, model)
    setattr(model, CACHE_ATTR, cache)
    logger.info("Embedding cache %s: %d states (environment %s)", cache.directory, len(cache), fp["environment"])
    return cache


# ---------------------------------------------------------------------------------------------------------------
# Offline access: states, gates and the score head read back without the model. Only needed for local testing
# (tests/test_embedding_cache.py, tests/test_qrm.py pin that offline rewards equal the online ones); no runner
# uses it, the runners score with the model loaded and serve repeated texts from the cache.
# ---------------------------------------------------------------------------------------------------------------
def save_head(cache: EmbeddingCache, model: Any) -> None:
    """Store the score head once, so rewards can be recomputed without the model (`offline_rewards`)."""
    from probes.heads import get_head

    path = cache.directory / "head.pt"
    if path.exists():
        return
    cache.directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    torch.save(get_head(model).to_saved(), tmp)
    os.replace(tmp, path)


def open_cache(directory: Union[str, Path]) -> EmbeddingCache:
    """Open an existing cache directory without the model."""
    directory = Path(directory)
    if not (directory / "meta.json").exists():
        raise FileNotFoundError(f"no embedding cache at {directory}")
    cache = EmbeddingCache(directory)
    if (directory / "gates" / "meta.json").exists():
        cache.gates = EmbeddingCache(directory / "gates")
    return cache


def lookup(cache: EmbeddingCache, texts: Iterable[Text], max_length: int) -> torch.Tensor:
    """Float32 [n, d] states for `texts`; raises if any was never embedded."""
    rows = []
    for text in texts:
        state = cache.get(text_key(text, max_length))
        if state is None:
            raise KeyError(f"text not in the embedding cache {cache.directory} "
                           f"(max_length={max_length}): {str(text)[:80]!r}")
        rows.append(state)
    return torch.stack(rows).float()


def lookup_gates(cache: EmbeddingCache, texts: Iterable[Text], max_length: int) -> Optional[torch.Tensor]:
    """A gated head's float32 gates [n, objectives]; None for a linear head."""
    return None if cache.gates is None else lookup(cache.gates, texts, max_length)


def load_head(cache: EmbeddingCache) -> Dict[str, Any]:
    """The stored head (`probes.heads` ``to_saved`` format): ``weight``/``bias`` for a linear head,
    ``kind``/``regression_weight``/... for a gated one."""
    return torch.load(cache.directory / "head.pt", map_location="cpu", weights_only=True)


def offline_rewards(cache: EmbeddingCache, states: torch.Tensor, probe: Optional[torch.Tensor] = None,
                    alpha: float = 1.0, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Rewards from cached states and the saved head, on CPU — the same arithmetic as
    `probes.probe.rewards_from_hidden` (project in float32, cast to the state dtype, apply the head). A gated
    head needs the states' ``gates`` (`lookup_gates`)."""
    from probes.heads import score_saved
    from probes.probe import project_to_null_space

    saved = load_head(cache)
    h = states.float()
    if probe is not None:
        h = project_to_null_space(h, probe, alpha=alpha)
    dtype = cache.state_dtype or saved.get("weight", saved.get("regression_weight")).dtype
    # F.linear: the same kernel as the online nn.Linear head, so offline == online exactly.
    return score_saved(saved, h.to(dtype), gates)
