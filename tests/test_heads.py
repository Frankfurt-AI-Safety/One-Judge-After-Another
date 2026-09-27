"""probes/heads.py on its own: the linear head (with a bias, which no test model has), the offline head
(`score_saved`) against the online one, the digest, and the gate-shape guard of the gated head."""

from __future__ import annotations

import pytest
import torch

from probes.heads import LinearHead, QuantileGatedHead, get_head, score_saved
from tests.test_qrm import _model as qrm_model


class _TinyRM(torch.nn.Module):
    """A body-less stand-in with a biased one-output score head."""

    def __init__(self, hidden=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.score = torch.nn.Linear(hidden, 1, bias=True).to(torch.bfloat16)


def _states(n=5, hidden=8, dtype=torch.bfloat16):
    torch.manual_seed(1)
    return torch.randn(n, hidden).to(dtype)


def test_linear_head_with_bias_offline_equals_online():
    model = _TinyRM()
    head = get_head(model)
    assert isinstance(head, LinearHead) and head.layer is model.score
    h = _states()
    with torch.no_grad():
        online = head.score(h)
        expected = model.score(h).squeeze(-1)
    saved = head.to_saved()
    assert saved["bias"] is not None
    assert online.shape == (5,) and torch.equal(online, expected)
    assert torch.equal(score_saved(saved, h), online)


def test_linear_effective_weights_are_the_head_vector():
    model = _TinyRM()
    w = get_head(model).effective_weights()
    assert w.shape == (1, 8) and w.dtype == torch.float32
    assert torch.equal(w, model.score.weight.detach().float())


def test_digest_is_stable_and_follows_the_weights():
    model = _TinyRM()
    head = get_head(model)
    first = head.digest()
    assert len(first) == 16 and head.digest() == first
    with torch.no_grad():
        model.score.weight[0, 0] += 1
    assert head.digest() != first


def test_gated_head_refuses_gates_of_the_wrong_shape():
    model = qrm_model()
    head = get_head(model)
    assert isinstance(head, QuantileGatedHead)
    h = _states(n=3, hidden=model.config.hidden_size)
    good = torch.softmax(torch.randn(3, head.num_objectives), dim=-1)
    with torch.no_grad():
        online = head.score(h, good)
        assert torch.equal(score_saved(head.to_saved(), h, good), online)
        for bad in (good[:, :1], good[:2], good.T):   # [n, 1] would broadcast silently without the guard
            with pytest.raises(ValueError, match="gates of shape"):
                head.score(h, bad)
            with pytest.raises(ValueError, match="gates of shape"):
                score_saved(head.to_saved(), h, bad)
