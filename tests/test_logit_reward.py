"""Reward models packaged as causal LMs (`scoring/logit_reward.py`, Nemotron-70B-Reward): the reward is one
token's logit at the last position — on a tiny random Llama causal LM whose chat template, like Nemotron's, has no
BOS, loaded through the real loader."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import scoring.backend as backend
from probes import embedding_cache as ec
from probes.heads import LinearHead, get_head
from probes.probe import get_rewards_both, verify_score_path
from scoring.dataset_base import add_special_tokens, format_conversation, tokenization_vs_template
from scoring.logit_reward import LogitRewardModel

_WORDS = ("should this loan be approved grade the essay approve decline a short answer yes no . ? "
          "<extra_id_0>System <extra_id_1>User <extra_id_1>Assistant").split()
# Nemotron-style: no BOS in the template; the tokenizer itself prepends one
_TEMPLATE = ("<extra_id_0>System {% for m in messages %}"
             "{{ '<extra_id_1>User' if m['role'] == 'user' else '<extra_id_1>Assistant' }} {{ m['content'] }} "
             "{% endfor %}")


def _tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, "<s>": 2, "</s>": 3, **{w: i + 4 for i, w in enumerate(dict.fromkeys(_WORDS))}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(single="<s> $A", pair="<s> $A $B", special_tokens=[("<s>", 2)])
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", bos_token="<s>",
                                 eos_token="</s>")
    hf.padding_side = "right"
    hf.chat_template = _TEMPLATE
    return hf


def _checkpoint(tmp_path):
    tok = _tokenizer()
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, pad_token_id=0, bos_token_id=2, eos_token_id=3,
                      max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(tmp_path / "rm")
    tok.save_pretrained(tmp_path / "rm")
    return str(tmp_path / "rm")


@pytest.fixture
def loaded(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    monkeypatch.setitem(backend.LOGIT_REWARD_MODELS, path, 0)
    monkeypatch.setattr(backend, "TEMPLATE_TOKENIZED", {path})
    cfg = SimpleNamespace(model_path=path, model_revision=None, trust_remote_code=False, device="cpu")
    model, tok = backend._load_transformers(cfg)
    return path, model.eval(), tok


def test_the_loader_wraps_the_causal_lm_and_the_head_is_row_k(loaded):
    _, model, _ = loaded
    assert isinstance(model, LogitRewardModel) and model.token_index == 0
    head = get_head(model)
    assert isinstance(head, LinearHead)
    assert torch.equal(head.layer.weight, model.lm.lm_head.weight[:1])
    assert model.model is model.lm.model and model.config is model.lm.config


def test_tokenized_as_the_template_without_the_tokenizers_bos(loaded):
    _, _, tok = loaded
    assert add_special_tokens(tok) is False and tokenization_vs_template(tok) == "aligned"
    fresh = _tokenizer()
    assert tokenization_vs_template(fresh) == "extra_bos"          # the default would have added a BOS


def test_pipeline_reward_equals_the_model_cards_generate_readout(loaded):
    _, model, tok = loaded
    assert verify_score_path(model, tok, max_length=128) < 0.02
    conversations = [("should this loan be approved ?", "approve ."), ("grade the essay .", "a short answer no .")]
    texts = [format_conversation(tok, p, r) for p, r in conversations]
    base, _ = get_rewards_both(model, tok, texts, None, batch_size=2, max_length=128, show_progress=False)
    for (prompt, response), reward in zip(conversations, base.tolist()):
        # the model card's procedure, verbatim in substance
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt",
                                      return_dict=True)
        out = model.lm.generate(enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=1,
                                return_dict_in_generate=True, output_scores=True, do_sample=False,
                                pad_token_id=tok.pad_token_id)
        assert reward == pytest.approx(out["scores"][0][0][0].item(), abs=0.02)


def test_the_cache_fingerprint_records_the_tokenization(loaded, tmp_path, monkeypatch):
    monkeypatch.delenv(ec.ENV_VAR, raising=False)
    _, model, tok = loaded
    fp = ec.model_fingerprint(model, tok)
    assert fp["add_special_tokens"] is False
    from tests.test_embedding_cache import _model as linear_model, _tokenizer as linear_tokenizer
    assert "add_special_tokens" not in ec.model_fingerprint(linear_model(), linear_tokenizer())


def test_an_unlisted_causal_lm_is_refused(tmp_path):
    path = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="randomly initialised score head"):
        backend._own_class(path)
