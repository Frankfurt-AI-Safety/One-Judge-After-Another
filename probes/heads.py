"""
Score heads: how a reward model turns its pooled last-token state ``h`` into the scalar reward. The
null-space projection sits between the two (``h' = h − α(v·h)v``), so the pipeline needs the head as a
separate step. Two kinds:

- `LinearHead` — the sequence-classification RMs (Llama, Qwen, Gemma, ...): ``score = W h + b``, one
  weight vector for every text.
- `QuantileGatedHead` — QRM (`scoring/qrm.py`): ``score = Σ_k gate_k · mean_q (R h)_{k,q}``. The gate is a
  per-text weight vector over the reward objectives, computed by an MLP from the state at the end of the
  USER turn, so it depends on the prompt only. For a fixed gate the score is linear in ``h``, with the
  text's effective head ``gate @ R̄`` (R̄ = the regression rows averaged over quantiles).

**Where the projection acts on a gated head.** Only ``h`` is projected; the gate is computed on the
unprojected forward pass and held fixed. The gate is therefore a second pathway from the prompt to the
reward that nulling ``h`` does not touch — in the direct-scoring arms the gate never sees the marker (it is in
the response); in the cross-marker design it does (it is in the prompt), and the runner reports how much of
each disparity runs through it (``gate_fixed``).

Every ``score`` reproduces the model's own arithmetic kernel for kernel (`nn.Linear` / ``F.linear`` in the
state dtype, the quantile mean in the model dtype, the gate product in float32), so the pipeline's reward
equals the model's score; `probes.probe.verify_score_path` checks that at every load.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def _digest(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in tensors:
        # reshape(-1): 0-dim tensors (BatchNorm's num_batches_tracked) cannot be byte-viewed; same bytes otherwise
        h.update(t.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


class LinearHead:
    """``score = W h + b`` (one output)."""

    gated = False
    kind = "linear"

    def __init__(self, layer: torch.nn.Linear):
        self.layer = layer

    @property
    def device(self) -> torch.device:
        return self.layer.weight.device

    def score(self, h: torch.Tensor, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Rewards [n] from states already cast to the model's dtype and on `device`."""
        return self.layer(h).squeeze(-1)

    def effective_weights(self, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The head vector(s) in float32 on CPU: [1, hidden] (the same for every text)."""
        return self.layer.weight.detach().float().cpu().reshape(1, -1)

    def digest(self) -> str:
        return _digest(self.layer.weight)[:16]

    def to_saved(self) -> Dict[str, Any]:
        return {"weight": self.layer.weight.detach().cpu(),
                "bias": None if self.layer.bias is None else self.layer.bias.detach().cpu()}


class QuantileGatedHead:
    """QRM's gated head (see the module docstring). ``gates`` are [n, objectives], one row per text."""

    gated = True
    kind = "quantile_gated"

    def __init__(self, model: Any):
        self.model = model
        self.regression = model.regression_layer
        self.num_objectives = int(model.num_objectives)
        self.num_quantiles = int(model.num_quantiles)

    @property
    def device(self) -> torch.device:
        return self.regression.weight.device

    def gates_from_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """float32 gates [batch, objectives] from the forward pass's final hidden states."""
        return self.model.gate(hidden, input_ids).float().cpu()

    def score(self, h: torch.Tensor, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
        if gates is None:
            raise ValueError("a gated head (QRM) scores a state only together with its text's gate: embed with "
                             "probes.probe.embed_with_gates and pass gates= to rewards_from_hidden")
        if gates.shape[0] != h.shape[0]:
            raise ValueError(f"{gates.shape[0]} gates for {h.shape[0]} states")
        expected = self.regression(h).reshape(-1, self.num_objectives, self.num_quantiles).mean(dim=2)
        return torch.sum(expected.float() * gates.to(expected.device).float(), dim=-1)

    def mean_rows(self) -> torch.Tensor:
        """R̄: the regression rows averaged over quantiles, float32 [objectives, hidden] on CPU."""
        w = self.regression.weight.detach().float().cpu()
        return w.reshape(self.num_objectives, self.num_quantiles, -1).mean(dim=1)

    def effective_weights(self, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Each text's effective head ``gate @ R̄``, float32 [n, hidden] on CPU (the linear head the
        score is for that text's fixed gate)."""
        if gates is None:
            raise ValueError("a gated head has one effective weight vector per text: pass its gates")
        return gates.float().cpu() @ self.mean_rows()

    def digest(self) -> str:
        return _digest(self.regression.weight)

    def gate_digest(self) -> str:
        return _digest(*(t for _, t in sorted(self.model.gating.state_dict().items())))

    def to_saved(self) -> Dict[str, Any]:
        return {"kind": self.kind, "regression_weight": self.regression.weight.detach().cpu(),
                "num_objectives": self.num_objectives, "num_quantiles": self.num_quantiles}


def is_gated_model(model: Any) -> bool:
    return hasattr(model, "regression_layer") and hasattr(model, "gating") and hasattr(model, "gate")


def get_head(model: Any) -> Any:
    """The model's score head as a `LinearHead` or `QuantileGatedHead`."""
    if is_gated_model(model):
        return QuantileGatedHead(model)
    from probes.probe import get_score_head  # local: probe.py imports this module

    return LinearHead(get_score_head(model))


def score_saved(saved: Dict[str, Any], h: torch.Tensor, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Rewards from a head stored by `to_saved` (offline, CPU; ``h`` already in the model's dtype) — the
    same kernels as the online heads, so offline equals online."""
    if saved.get("kind") == QuantileGatedHead.kind:
        if gates is None:
            raise ValueError("offline scoring of a gated head needs the texts' gates (cache.lookup_gates)")
        w = saved["regression_weight"].to(h.dtype)
        expected = F.linear(h, w).reshape(-1, saved["num_objectives"], saved["num_quantiles"]).mean(dim=2)
        return torch.sum(expected.float() * gates.float(), dim=-1)
    bias = saved.get("bias")
    return F.linear(h, saved["weight"].to(h.dtype), None if bias is None else bias.to(h.dtype)).squeeze(-1)
