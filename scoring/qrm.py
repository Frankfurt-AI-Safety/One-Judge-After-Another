"""
QRM-Gemma-2 (quantile-regression reward model with a gating network) — our own implementation of the
architecture, so the pipeline can load and score it without the checkpoint's remote code.

Model: ``nicolinho/QRM-Gemma-2-27B`` (Dorka 2024, "Quantile Regression for Distributional Reward Models in
RLHF", arXiv:2409.10164; code github.com/Nicolinho/QRM; weights under the Llama 3 licence). Backbone
Skywork-Reward-Gemma-2-27B-v0.2.

Why not ``trust_remote_code``: the checkpoint's ``modeling_custom.py`` imports ``LLAMA_INPUTS_DOCSTRING``
from ``transformers.models.llama.modeling_llama``, which no longer exists (checked 2026-09-26 in 4.57.6, the
cluster's version, and 5.17), so the import fails before anything runs; the constant was dead code there.
Carrying the architecture here also avoids executing unpinned remote code and exposes what the pipeline
needs: where the gate reads and the gate itself.

Scoring, as in the reference implementation (module and parameter names match the checkpoint):

- ``h`` = the final hidden state (after the last norm) at the last real token;
- ``regression_layer`` (no bias) maps ``h`` to ``num_objectives × num_quantiles`` reward quantiles
  (5 HelpSteer2 attributes × 19 quantiles); their mean over quantiles is each objective's expected reward;
- ``gate`` = softmax(MLP(p) / T) · logit_scale over the objectives, where ``p`` is the final hidden state
  at the start of the LAST ``<end_of_turn>\\n<start_of_turn>model\\n`` token pattern — the end of the user
  turn. Under causal attention ``p``, hence the gate, depends on the prompt only;
- ``score`` = Σ_objectives gate · expected reward — for a fixed gate, a linear function of ``h``.

Differences from the reference forward, neither of which changes a score the pipeline computes: the last
token is found from ``attention_mask`` (the pipeline pins right padding; the reference searched for the
first pad id, which is the same position); and the reference's special case for a doubled BOS at batch
size 1 is dropped, because `format_conversation` strips the template's BOS and the tokenizer adds one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Gemma2Config, Gemma2Model, Gemma2PreTrainedModel
from transformers.utils import ModelOutput

# Gemma-2 token ids of "<end_of_turn>\n<start_of_turn>model\n"; the gate reads the first of them
GEMMA2_GATING_PATTERN = (107, 108, 106, 2516, 108)


class GatingNetwork(nn.Module):
    """MLP over the prompt state → softmax weights over the reward objectives, scaled by a learned factor.
    Layer order (and so the checkpoint's parameter indices): n_hidden × [Linear(no bias), ReLU,
    BatchNorm1d, Dropout except after the last], then Linear(bias). BatchNorm and Dropout run in eval mode."""

    def __init__(self, in_features: int, out_features: int, temperature: float = 1.0, hidden_dim: int = 1024,
                 n_hidden: int = 3, dropout: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1))
        layers: list = []
        for i in range(n_hidden):
            layers += [nn.Linear(in_features, hidden_dim, bias=False), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
            if dropout > 0 and i < n_hidden - 1:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=True))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x / self.temperature, dim=-1) * self.logit_scale


def gating_positions(input_ids: torch.Tensor, pattern: Sequence[int]) -> torch.Tensor:
    """Per row, the index where the LAST occurrence of ``pattern`` starts (the end of the user turn)."""
    pattern = list(pattern)
    out = []
    for row in input_ids.tolist():
        for j in range(len(row) - len(pattern), -1, -1):
            if row[j:j + len(pattern)] == pattern:
                out.append(j)
                break
        else:
            raise ValueError(f"QRM gating token pattern {pattern} not found: the input is not a chat-formatted "
                             f"user/assistant conversation")
    return torch.tensor(out, dtype=torch.long)


@dataclass
class QuantileRewardOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None           # = score, [batch, 1]
    score: Optional[torch.FloatTensor] = None
    rewards: Optional[torch.FloatTensor] = None          # expected reward per objective [batch, objectives]
    reward_quantiles: Optional[torch.FloatTensor] = None
    gating_output: Optional[torch.FloatTensor] = None    # [batch, objectives]


class Gemma2ForQuantileSequenceClassification(Gemma2PreTrainedModel):
    """The checkpoint's architecture, under its own name (so a model saved from here declares it too); see
    the module docstring."""

    config_class = Gemma2Config

    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        cfg = config.to_dict()
        self.num_objectives = int(cfg.get("num_objectives", 5))
        self.num_quantiles = int(cfg.get("num_quantiles", 19))
        self.gating_pattern = tuple(cfg.get("gating_token_pattern", GEMMA2_GATING_PATTERN))
        self.model = Gemma2Model(config)
        self.regression_layer = nn.Linear(config.hidden_size, self.num_objectives * self.num_quantiles,
                                          bias=False)
        self.gating = GatingNetwork(config.hidden_size, self.num_objectives,
                                    temperature=cfg.get("gating_temperature", 1),
                                    hidden_dim=cfg.get("gating_hidden_dim", 1024),
                                    n_hidden=cfg.get("gating_n_hidden", 3))
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def gate(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """The gate from the final hidden states [batch, seq, hidden] (native dtype, as the model uses it)."""
        pos = gating_positions(input_ids, self.gating_pattern).to(hidden.device)
        prompt_state = hidden[torch.arange(hidden.shape[0], device=hidden.device), pos]
        return self.gating(prompt_state.to(self.gating.logit_scale.device))

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None,
                **kwargs: Any) -> QuantileRewardOutput:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state
        if attention_mask is None:
            last = torch.full((input_ids.shape[0],), input_ids.shape[1] - 1, dtype=torch.long)
        else:
            last = attention_mask.sum(dim=1) - 1
        h = hidden[torch.arange(hidden.shape[0], device=hidden.device), last.to(hidden.device)]
        quantiles = self.regression_layer(h.to(self.regression_layer.weight.device))
        quantiles = quantiles.reshape(-1, self.num_objectives, self.num_quantiles)
        gate = self.gate(hidden, input_ids).to(quantiles.device)
        expected = quantiles.mean(dim=2)
        score = torch.sum(expected.float() * gate.float(), dim=-1, keepdim=True)
        return QuantileRewardOutput(logits=score, score=score, rewards=expected,
                                    reward_quantiles=torch.mean(quantiles * gate.unsqueeze(-1), dim=1),
                                    gating_output=gate)


ARCHITECTURE = Gemma2ForQuantileSequenceClassification.__name__
