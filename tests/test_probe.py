"""probes/probe.py: the orthonormal basis behind every projection, the forward pass's final-layer read, and
tokenization without truncation in both input formats."""

from __future__ import annotations

import pytest
import torch

from probes.probe import (
    InputTooLong, _forward_pooled, get_base_model, gram_schmidt, project_to_null_space, tokenize_inputs,
)
from scoring.dataset_base import ADD_SPECIAL_TOKENS_ATTR
from tests.test_embedding_cache import TEXTS, _model, _tokenizer


def _unit(d=4096, seed=0):
    torch.manual_seed(seed)
    u = torch.randn(d)
    return u / u.norm()


def _states(n=500, d=4096):
    torch.manual_seed(7)
    return torch.randn(n, d) * 5


# --------------------------------------------------------------------------- gram_schmidt
def test_a_single_vector_is_only_normalised():
    v = _unit() * 3.7
    assert torch.equal(gram_schmidt([v])[0], v / v.norm())


@pytest.mark.parametrize("scale", [3.0, 1.0001, -2.0])
def test_a_rescaled_copy_adds_no_row_and_the_direction_is_nulled(scale):
    # One pass with an absolute tolerance turned the copy's rounding residue into a second, non-orthogonal
    # row; nulling [u, 3u] then left 92% of the u component.
    u, h = _unit(), _states()
    assert gram_schmidt([u, scale * u]).shape[0] == 1
    out = project_to_null_space(h, torch.stack([u, scale * u]))
    assert float((out @ u).abs().max()) < 1e-4


def test_a_genuine_near_duplicate_is_kept_and_the_basis_is_orthonormal():
    u, w = _unit(seed=0), _unit(seed=1)
    v = u + 5e-3 * w                          # cos(u, v) ~ 0.99999: a real, if small, second direction
    basis = gram_schmidt([u, v])
    assert basis.shape[0] == 2
    assert torch.allclose(basis @ basis.T, torch.eye(2), atol=1e-5)


@pytest.mark.parametrize("d", [64, 4096, 8192])
def test_nearly_dependent_inputs_give_an_orthonormal_basis(d):
    vs = [_unit(d, seed=s) for s in range(4)]
    vs.append(sum(vs) + 1e-3 * _unit(d, seed=9))   # almost in the span of the first four
    vs.append(2 * vs[1])                            # exactly in it (up to rounding)
    basis = gram_schmidt(vs)
    assert basis.shape[0] == 5
    assert torch.allclose(basis @ basis.T, torch.eye(5), atol=1e-5)
    h = _states(50, d)
    out = project_to_null_space(h, torch.stack(vs))
    assert float((out @ basis.T).abs().max()) < 1e-3


def test_zero_vectors_are_dropped():
    assert gram_schmidt([torch.zeros(8)]).shape == (0, 8)
    assert gram_schmidt([torch.zeros(8), _unit(8)]).shape == (1, 8)


# --------------------------------------------------------------------------- the forward pass
def test_the_pooled_state_is_the_final_layer_without_keeping_every_layer():
    model, tok = _model(), _tokenizer()
    base = get_base_model(model)
    seen = []
    handle = base.register_forward_pre_hook(lambda mod, args, kwargs: seen.append(dict(kwargs)), with_kwargs=True)
    try:
        states, _, _ = _forward_pooled(model, tok, TEXTS, 4, 32, False)
    finally:
        handle.remove()
    assert seen and all(not kw.get("output_hidden_states") for kw in seen)
    # the same state the old read of hidden_states[-1] gave, at the last real token (right padding)
    inputs = tokenize_inputs(tok, TEXTS, max_length=32)
    with torch.no_grad():
        full = base(**inputs, output_hidden_states=True).hidden_states[-1]
    last = inputs["attention_mask"].sum(1) - 1
    assert torch.equal(states, full[torch.arange(len(TEXTS)), last].float())


# --------------------------------------------------------------------------- tokenize_inputs
class _RecordingTokenizer:
    """Records the keyword arguments of every call; token count = whitespace words (+1 per text of a pair)."""

    def __init__(self):
        self.calls = []

    def __call__(self, *texts, **kwargs):
        self.calls.append(kwargs)
        rows = texts[0] if len(texts) == 1 else [f"{a} {b}" for a, b in zip(*texts)]
        lens = [len(r.split()) for r in rows]
        mask = torch.tensor([[1] * n + [0] * (max(lens) - n) for n in lens])
        return {"input_ids": mask.clone(), "attention_mask": mask}


@pytest.mark.parametrize("texts", [["a b c", "a b"], [("a b", "c"), ("a", "b")]])
@pytest.mark.parametrize("special", [True, False])
def test_both_formats_follow_the_special_token_setting_and_never_truncate(texts, special):
    tok = _RecordingTokenizer()
    setattr(tok, ADD_SPECIAL_TOKENS_ATTR, special)
    tokenize_inputs(tok, texts, max_length=3)
    assert tok.calls[-1]["add_special_tokens"] is special and tok.calls[-1]["truncation"] is False
    with pytest.raises(InputTooLong, match="2 of 2 inputs exceed max_length=1"):
        tokenize_inputs(tok, texts, max_length=1)
