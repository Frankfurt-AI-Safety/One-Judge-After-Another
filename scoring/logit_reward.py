"""
Reward models packaged as causal language models: the reward is the raw logit of one vocabulary token at the
last position of the conversation.

``nvidia/Llama-3.3-Nemotron-70B-Reward`` is a Bradley–Terry scalar RM (HelpSteer3, arXiv:2505.11475) shipped as a
``LlamaForCausalLM``. Its model card scores a conversation with ``generate(max_new_tokens=1, output_scores=True)``
and reads ``scores[0][0][0]``: the logit of token 0 at the first generated position, i.e. at the last input token.
Its generation config sets no sampling or penalty, so that is the unprocessed logit, and the output matrix is not
tied to the embeddings. The reward is therefore

    r = W_lm[k] · h_last          (k = 0, no bias)

— a linear head on the pooled last-token state, the form the pipeline projects before. `LogitRewardModel` exposes
exactly that: ``.model`` (the transformer), ``.score`` (row k of ``lm_head`` as a one-output linear layer, found by
`probes.probe.get_score_head`) and a ``forward`` whose ``logits`` are the model's own reward, computed through the
full ``lm_head`` as the model card does — what `probes.probe.verify_score_path` compares the pipeline against.

A causal-LM checkpoint is only loaded as a reward model when it is listed in `scoring.backend.LOGIT_REWARD_MODELS`
with its token; an unlisted one is refused, because loading it as a sequence classifier would attach a randomly
initialised score head that nothing downstream could detect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers.utils import ModelOutput


@dataclass
class LogitRewardOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None          # the reward, [batch, 1]


class LogitRewardModel(torch.nn.Module):
    """A causal LM whose reward is the logit of vocabulary token ``token_index`` at the last real token."""

    def __init__(self, causal_lm: Any, token_index: int):
        super().__init__()
        self.lm = causal_lm
        self.token_index = int(token_index)
        row = causal_lm.lm_head.weight[self.token_index:self.token_index + 1].detach()
        self.score = torch.nn.Linear(row.shape[1], 1, bias=False, device=row.device, dtype=row.dtype)
        with torch.no_grad():
            self.score.weight.copy_(row)
        self.score.requires_grad_(False)

    @property
    def model(self) -> Any:
        """The transformer (what `probes.probe.get_base_model` reads the pooled state from)."""
        return self.lm.model

    @property
    def config(self) -> Any:
        return self.lm.config

    @property
    def hf_device_map(self) -> Any:
        return getattr(self.lm, "hf_device_map", None)

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None,
                **kwargs: Any) -> LogitRewardOutput:
        logits = self.lm(input_ids=input_ids, attention_mask=attention_mask).logits      # [batch, seq, vocab]
        if attention_mask is None:
            last = torch.full((input_ids.shape[0],), input_ids.shape[1] - 1, dtype=torch.long)
        else:
            last = attention_mask.sum(dim=1) - 1                                         # right padding
        rows = torch.arange(logits.shape[0], device=logits.device)
        return LogitRewardOutput(logits=logits[rows, last.to(logits.device), self.token_index].unsqueeze(-1))
