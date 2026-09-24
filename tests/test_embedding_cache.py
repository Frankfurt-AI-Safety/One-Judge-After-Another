"""
Tests for the embedding cache (`probes/embedding_cache.py`) and its integration in `probes/probe.py`,
on a tiny randomly initialised Llama reward model (CPU, bf16 — the cluster's dtype).

The cache must be invisible in the numbers: cached and uncached rewards are bit-identical, a text is
embedded once per model and max_length, nothing is ever served across models, and the offline path
(no model loaded) reproduces the online rewards exactly.
"""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForSequenceClassification, PreTrainedTokenizerFast

from probes import embedding_cache as ec
from probes.probe import (
    build_probe_direction, get_base_model, get_embeddings, get_rewards_both, rewards_from_hidden,
    tokenize_inputs,
)
from scoring.dataset_base import ContrastivePair

_WORDS = ("the student essay is good bad a woman man grades high low score of reward text applicant "
          "credit profile strong weak loan").split()
TEXTS = ["the student essay is good", "a woman applicant credit profile", "the essay is bad",
         "a man applicant credit profile strong", "high score of reward text", "low grades weak loan"]


def _tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(_WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")
    hf.padding_side = "right"
    return hf


def _model(seed=0):
    cfg = LlamaConfig(vocab_size=len(_WORDS) + 2, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, pad_token_id=0, num_labels=1,
                      max_position_embeddings=64)
    torch.manual_seed(seed)
    return LlamaForSequenceClassification(cfg).to(torch.bfloat16).eval()


class _Counter:
    """Counts texts pushed through the base model's forward pass."""

    def __init__(self):
        self.n = 0

    @staticmethod
    def attach(model):
        counter = _Counter()

        def pre(module, args, kwargs):
            counter.n += kwargs["input_ids"].shape[0]
        get_base_model(model).register_forward_pre_hook(pre, with_kwargs=True)
        return counter


@pytest.fixture
def cached(tmp_path, monkeypatch):
    monkeypatch.delenv(ec.ENV_VAR, raising=False)
    model, tok = _model(), _tokenizer()
    ec.attach(model, tok, str(tmp_path))
    return model, tok, tmp_path


def _probe(model, tok):
    pairs = [ContrastivePair(positive_text=TEXTS[1], negative_text=TEXTS[3]),
             ContrastivePair(positive_text=TEXTS[0], negative_text=TEXTS[2])]
    return build_probe_direction(model, tok, pairs, batch_size=4, device="cpu", max_length=32)[0]


def test_cached_rewards_are_bit_identical_to_uncached(cached):
    model, tok, _ = cached
    probe = _probe(model, tok)
    plain = _model()                      # same seed -> same weights, no cache attached
    b0, n0 = get_rewards_both(plain, tok, TEXTS, probe, batch_size=4, device="cpu", max_length=32,
                              show_progress=False)
    b1, n1 = get_rewards_both(model, tok, TEXTS, probe, batch_size=4, device="cpu", max_length=32,
                              show_progress=False)
    b2, n2 = get_rewards_both(model, tok, TEXTS, probe, batch_size=4, device="cpu", max_length=32,
                              show_progress=False)  # all hits
    assert torch.equal(b0, b1) and torch.equal(n0, n1)
    assert torch.equal(b1, b2) and torch.equal(n1, n2)
    assert not torch.equal(b1, n1)  # the probe does something


def test_baseline_reward_is_the_models_own_score(cached):
    # Sanity of the pooled-state path: head(last real token) == HF's own sequence-classification logits.
    model, tok, _ = cached
    base, _ = get_rewards_both(model, tok, TEXTS, None, batch_size=4, device="cpu", max_length=32,
                               show_progress=False)
    with torch.no_grad():
        logits = model(**tokenize_inputs(tok, TEXTS, max_length=32)).logits.squeeze(-1)
    assert torch.allclose(base.float(), logits.float(), atol=2e-2)


def test_second_call_runs_no_forward_pass(cached):
    model, tok, _ = cached
    count = _Counter.attach(model)
    get_rewards_both(model, tok, TEXTS, None, batch_size=4, device="cpu", max_length=32, show_progress=False)
    assert count.n == len(TEXTS)
    get_rewards_both(model, tok, TEXTS, None, batch_size=4, device="cpu", max_length=32, show_progress=False)
    get_embeddings(model, tok, TEXTS, batch_size=4, device="cpu", max_length=32, show_progress=False)
    assert count.n == len(TEXTS)


def test_repeated_texts_are_embedded_once(cached):
    model, tok, _ = cached
    count = _Counter.attach(model)
    out = get_embeddings(model, tok, TEXTS[:2] * 5, batch_size=4, device="cpu", max_length=32,
                         show_progress=False)
    assert count.n == 2 and out.shape[0] == 10
    assert torch.equal(out[0], out[2]) and torch.equal(out[1], out[9])


def test_max_length_is_part_of_the_key(cached):
    model, tok, _ = cached
    count = _Counter.attach(model)
    get_embeddings(model, tok, TEXTS, batch_size=4, device="cpu", max_length=32, show_progress=False)
    get_embeddings(model, tok, TEXTS, batch_size=4, device="cpu", max_length=3, show_progress=False)
    assert count.n == 2 * len(TEXTS)  # truncation changes the state, so no reuse


def test_cache_persists_for_a_new_process(cached, monkeypatch):
    model, tok, root = cached
    first = get_embeddings(model, tok, TEXTS, batch_size=4, device="cpu", max_length=32, show_progress=False)
    fresh = _model()                      # "another job": same weights, new object, re-attached
    ec.attach(fresh, tok, str(root))
    count = _Counter.attach(fresh)
    again = get_embeddings(fresh, tok, TEXTS, batch_size=4, device="cpu", max_length=32, show_progress=False)
    assert count.n == 0 and torch.equal(first, again)
    assert not list(root.rglob("*.tmp"))  # atomic writes leave no temp files


def test_alpha_sweep_from_cached_states_equals_a_fresh_pass(cached):
    model, tok, _ = cached
    probe = _probe(model, tok)
    hidden = get_embeddings(model, tok, TEXTS, batch_size=4, device="cpu", max_length=32, show_progress=False)
    dtype = getattr(model, ec.CACHE_ATTR).state_dtype
    for alpha in (0.0, 0.5, 1.0):
        _, swept = rewards_from_hidden(model, hidden, dtype, probe, null_alpha=alpha)
        _, direct = get_rewards_both(_model(), tok, TEXTS, probe, batch_size=4, device="cpu",
                                     max_length=32, show_progress=False, null_alpha=alpha)
        assert torch.equal(swept, direct), alpha


def test_offline_rewards_reproduce_the_online_ones_without_the_model(cached):
    model, tok, root = cached
    probe = _probe(model, tok)
    base, nulled = get_rewards_both(model, tok, TEXTS, probe, batch_size=4, device="cpu", max_length=32,
                                    show_progress=False)
    (directory,) = [d for d in root.iterdir() if d.is_dir()]
    cache = ec.open_cache(directory)
    states = cache.lookup(TEXTS, max_length=32)
    assert cache.state_dtype == torch.bfloat16
    assert torch.equal(ec.offline_rewards(cache, states), base)
    assert torch.equal(ec.offline_rewards(cache, states, probe, alpha=1.0), nulled)
    with pytest.raises(KeyError):
        cache.lookup(["never embedded text"], max_length=32)


def test_a_different_model_never_shares_states(cached):
    model, tok, root = cached
    get_embeddings(model, tok, TEXTS, batch_size=4, device="cpu", max_length=32, show_progress=False)
    other = _model(seed=1)
    ec.attach(other, tok, str(root))
    count = _Counter.attach(other)
    get_embeddings(other, tok, TEXTS, batch_size=4, device="cpu", max_length=32, show_progress=False)
    assert count.n == len(TEXTS)
    assert len([d for d in root.iterdir() if d.is_dir()]) == 2


def test_a_directory_refuses_another_fingerprint(cached):
    model, tok, root = cached
    get_embeddings(model, tok, TEXTS, batch_size=4, device="cpu", max_length=32, show_progress=False)
    (directory,) = [d for d in root.iterdir() if d.is_dir()]
    with pytest.raises(ValueError, match="different model"):
        ec.EmbeddingCache(directory, fingerprint={"model_path": "something else"})


def test_environment_switch_disables_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(ec.ENV_VAR, "off")
    model, tok = _model(), _tokenizer()
    assert ec.attach(model, tok, str(tmp_path)) is None
    get_embeddings(model, tok, TEXTS, batch_size=4, device="cpu", max_length=32, show_progress=False)
    assert not any(tmp_path.iterdir())


def test_experiment_config_carries_the_cache_setting():
    from scoring.experiment import ExperimentConfig

    cfg = ExperimentConfig(name="x", bias_type="demographic", model_path="m")
    assert cfg.embedding_cache_dir == "artifacts/embedding_cache"
    assert ExperimentConfig.from_dict({**cfg.to_dict()}).embedding_cache_dir == "artifacts/embedding_cache"


def test_score_path_check_accepts_a_last_token_model_and_refuses_deberta():
    # Pre-existing silent bug: DeBERTa scores the FIRST token through a ContextPooler, but the pipeline
    # applies the classifier to the pooled LAST-token state -- rewards unrelated to the model's scores.
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

    from probes.probe import ScorePathMismatch, verify_score_path

    assert verify_score_path(_model(), _tokenizer(), max_length=32) < 0.05
    cfg = DebertaV2Config(vocab_size=len(_WORDS) + 2, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_labels=1, pad_token_id=0, max_position_embeddings=64,
                          pooler_hidden_size=32)
    torch.manual_seed(0)
    deberta = DebertaV2ForSequenceClassification(cfg).eval()
    with torch.no_grad():  # scale the head so the random model's scores are not all ~0
        deberta.classifier.weight.mul_(50)
    with pytest.raises(ScorePathMismatch, match="does not reproduce"):
        verify_score_path(deberta, _tokenizer(), max_length=32)
