"""
Embedding cache: each unique text is embedded ONCE per reward model; everything downstream is offline.

The pipeline's expensive step is the forward pass that produces the pooled last-token hidden state
(`probes.probe.get_embeddings`). Everything after it — the difference-of-means probe, null-space
projection, the score head, the α-sweep, split-seed repeats, bootstrap resamples — is cheap arithmetic
on those states. Without a cache the same text was embedded repeatedly: the battery embeds each axis's
pairs separately although the single-axis pairs of a factorial are all cut from the same 8 cells
(~3.25x), and every α of the sweep re-ran the model. With it, a text costs one forward pass per model,
ever, and the pilot's probe-size / seed / bootstrap studies need no GPU at all (`open_cache`).

What is stored is exactly what the score head sees: the pooled last-token state of
``hidden_states[-1]`` (right padding; see `scoring/backend.py`) in the model's compute dtype. The
reward code upcasts that bf16 state to float32 before projecting, which is exact, so storing it in bf16
loses nothing. The score head's weights are saved next to the states, so rewards can be recomputed
offline.

Keys and isolation. A state is keyed by (text, max_length) — truncation changes the state — within a
directory named by a **fingerprint of the model**: its path, revision, config, dtype, tokenizer class and
padding side, and a hash of the score head's weights, so a different checkpoint or a re-download at a
new revision can never be served another model's states.

Storage. Append-only shard files (``shard-*.pt``) written atomically; no database. The cluster writes to
a shared parallel filesystem (PFSS) where SQLite-style locking is unreliable, and concurrent jobs on the
same model must not corrupt each other: each process only ever creates its own shard files and reads the
shards that existed when it opened the directory.

Determinism note. A state is reused as first computed. Recomputing the same text in a different batch
(other padding lengths) can differ at bf16 noise level — as the uncached pipeline already did between
runs — so the cache makes results *more* reproducible, not less.

Configure with the experiment config key ``embedding_cache_dir`` (default ``artifacts/embedding_cache``,
gitignored). Environment override ``ONEJUDGE_EMBED_CACHE``: ``off`` disables it, a path replaces the
directory.

Gated heads (QRM, `probes/heads.py`): the score also needs a per-text gate from the same forward pass,
kept in a ``gates/`` sub-cache under the same keys, and the fingerprint adds the gating network's digest.
The fingerprint of a linear-head model is unchanged, so existing caches stay valid.

LIMITATION (pre-existing, not introduced here): the reward path assumes the score is a head on the pooled
LAST-token state, as for the Llama/Qwen/Gemma sequence-classification RMs and QRM. That is not how every
RM computes its score (e.g. DeBERTa scores a pooled FIRST token through a pooler) — see the working notes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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
POOLING = "hidden_states[-1], last non-pad token (right padding)"
# Write a shard every this many new states, so a crashed job keeps most of its work.
FLUSH_EVERY = 4096

Text = Union[str, Tuple[str, str]]


def text_key(text: Text, max_length: int) -> str:
    """Stable key for one input under one truncation length (tuples are (prompt, response) inputs)."""
    payload = json.dumps([max_length, list(text) if isinstance(text, tuple) else text],
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_fingerprint(model: Any, tokenizer: Any) -> Dict[str, Any]:
    """Everything that decides the pooled state of a given text, apart from the text itself."""
    from probes.heads import get_head

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
        "head_digest": head.digest(),
    }
    if head.gated:   # linear heads keep the exact earlier fingerprint, so their caches stay valid
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
            data = torch.load(shard, map_location="cpu", weights_only=True)
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

    def save_head(self, model: Any) -> None:
        """Store the score head once, so rewards can be recomputed without the model (`open_cache`)."""
        from probes.heads import get_head

        path = self.directory / "head.pt"
        if path.exists():
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        torch.save(get_head(model).to_saved(), tmp)
        os.replace(tmp, path)

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
        return len(self._index) + len(self._pending_keys)

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
        """Add states (any float dtype; stored in `state_dtype`, which the first put fixes)."""
        if self.state_dtype is None:
            self.state_dtype = states.dtype
        block = len(self._pending)
        for row, key in enumerate(keys):
            self._pending_at.setdefault(key, (block, row))
        self._pending_keys.extend(keys)
        self._pending.append(states.detach().to("cpu", self.state_dtype))
        if len(self._pending_keys) >= FLUSH_EVERY:
            self.flush()

    def lookup(self, texts: Iterable[Text], max_length: int) -> torch.Tensor:
        """Offline access: float32 [n, d] states for `texts`; raises if any was never embedded."""
        rows = []
        for text in texts:
            state = self.get(text_key(text, max_length))
            if state is None:
                raise KeyError(f"text not in the embedding cache {self.directory} "
                               f"(max_length={max_length}): {str(text)[:80]!r}")
            rows.append(state)
        return torch.stack(rows).float()

    def load_head(self) -> Dict[str, Any]:
        """The stored head (`probes.heads` ``to_saved`` format): ``weight``/``bias`` for a linear head,
        ``kind``/``regression_weight``/... for a gated one."""
        return torch.load(self.directory / "head.pt", map_location="cpu", weights_only=True)

    def lookup_gates(self, texts: Iterable[Text], max_length: int) -> Optional[torch.Tensor]:
        """Offline access to a gated head's float32 gates [n, objectives]; None for a linear head."""
        return None if self.gates is None else self.gates.lookup(texts, max_length)


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
    cache.save_head(model)
    setattr(model, CACHE_ATTR, cache)
    logger.info("Embedding cache %s: %d states", cache.directory, len(cache))
    return cache


def open_cache(directory: Union[str, Path]) -> EmbeddingCache:
    """Open an existing cache directory without the model (offline analysis)."""
    directory = Path(directory)
    if not (directory / "meta.json").exists():
        raise FileNotFoundError(f"no embedding cache at {directory}")
    cache = EmbeddingCache(directory)
    if (directory / "gates" / "meta.json").exists():
        cache.gates = EmbeddingCache(directory / "gates")
    return cache


def offline_rewards(cache: EmbeddingCache, states: torch.Tensor, probe: Optional[torch.Tensor] = None,
                    alpha: float = 1.0, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Rewards from cached states and the saved head, on CPU — the same arithmetic as
    `probes.probe.rewards_from_hidden` (project in float32, cast to the state dtype, apply the head). A gated
    head needs the states' ``gates`` (`EmbeddingCache.lookup_gates`)."""
    from probes.heads import score_saved
    from probes.probe import project_to_null_space

    saved = cache.load_head()
    h = states.float()
    if probe is not None:
        h = project_to_null_space(h, probe, alpha=alpha)
    dtype = cache.state_dtype or saved.get("weight", saved.get("regression_weight")).dtype
    # F.linear: the same kernel as the online nn.Linear head, so offline == online exactly.
    return score_saved(saved, h.to(dtype), gates)
