"""QRM (quantile-regression RM with a gating network): our implementation of the architecture
(`scoring/qrm.py`), its gated score head (`probes/heads.py`) and how the pipeline carries the per-text gate —
on a tiny random Gemma-2 QRM with a word-level chat tokenizer."""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Gemma2Config, PreTrainedTokenizerFast

from probes import embedding_cache as ec
from probes.heads import QuantileGatedHead, get_head
from probes.probe import (
    embed_with_gates, get_rewards_both, project_to_null_space, rewards_from_hidden, verify_score_path,
)
from scoring.dataset_base import format_conversation
from scoring.qrm import ARCHITECTURE, Gemma2ForQuantileSequenceClassification, gating_positions
from tests.test_run_cross_marker import manifest  # noqa: F401  (fixture)

# the checkpoint's non-backbone parameters (nicolinho/QRM-Gemma-2-27B, model.safetensors.index.json)
CHECKPOINT_HEAD_KEYS = {
    "gating.layers.0.weight", "gating.layers.4.weight", "gating.layers.8.weight",
    "gating.layers.11.weight", "gating.layers.11.bias", "gating.logit_scale", "regression_layer.weight",
    *(f"gating.layers.{i}.{p}" for i in (2, 6, 10)
      for p in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")),
}

_WORDS = ("the applicant has a stable income and repaid every earlier loan on time should this be approved "
          "grade essay short woman man married single year old credit approve decline . ? ,").split()
_SPECIAL = ["<start_of_turn>", "<end_of_turn>", "user", "model"]
_TEMPLATE = ("{% for m in messages %}<start_of_turn> {{ 'model' if m['role'] == 'assistant' else 'user' }} "
             "{{ m['content'] }} <end_of_turn> {% endfor %}")


def _tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(dict.fromkeys(_SPECIAL + _WORDS))}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()   # keeps "<start_of_turn>" whole
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")
    hf.padding_side = "right"
    hf.chat_template = _TEMPLATE
    return hf


def _pattern(tok):
    return [tok.convert_tokens_to_ids(t) for t in ("<end_of_turn>", "<start_of_turn>", "model")]


def _config(tok, **extra):
    cfg = Gemma2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=1024,
                       pad_token_id=0, query_pre_attn_scalar=8, sliding_window=512, gating_hidden_dim=16,
                       gating_token_pattern=_pattern(tok), **extra)
    cfg._attn_implementation = "eager"
    return cfg


def _model(tok=None, seed=0):
    tok = tok or _tokenizer()
    torch.manual_seed(seed)
    model = Gemma2ForQuantileSequenceClassification(_config(tok))
    with torch.no_grad():   # a gate that varies (not uniform, not saturated), non-trivial BN statistics
        for layer in model.gating.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.normal_(0, 0.3)
            if isinstance(layer, torch.nn.BatchNorm1d):
                layer.running_mean.normal_(0, 0.5)
                layer.running_var.uniform_(0.5, 2.0)
        model.gating.logit_scale.fill_(1.3)
        model.regression_layer.weight.normal_(0, 0.5)
    return model.to(torch.bfloat16).eval()


def _conv(tok, prompt, response):
    return format_conversation(tok, prompt, response)


# --------------------------------------------------------------------------- the architecture ----------
def test_parameter_names_are_the_checkpoints():
    assert {k for k in _model().state_dict() if not k.startswith("model.")} == CHECKPOINT_HEAD_KEYS


def test_score_is_the_gated_mean_over_quantiles():
    tok, model = _tokenizer(), _model()
    inputs = tok([_conv(tok, "should this loan be approved ?", "approve the loan .")], return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        hidden = model.model(**inputs).last_hidden_state
        h = hidden[0, inputs["attention_mask"].sum() - 1]
        q = model.regression_layer(h).reshape(model.num_objectives, model.num_quantiles)
        gate = model.gate(hidden, inputs["input_ids"])[0]
    assert out.logits.shape == (1, 1)
    assert float(out.logits) == pytest.approx(float((q.mean(1).float() * gate.float()).sum()), abs=1e-5)
    assert float(gate.float().sum()) == pytest.approx(1.3, abs=1e-2)          # softmax · logit_scale


def test_gate_reads_the_end_of_the_user_turn_and_depends_on_the_prompt_only():
    tok, model = _tokenizer(), _model()
    texts = [_conv(tok, "should this loan be approved ?", "approve ."),
             _conv(tok, "should this loan be approved ?", "decline the loan , the applicant is single ."),
             _conv(tok, "grade this short essay .", "approve .")]
    inputs = tok(texts, return_tensors="pt", padding=True)
    ids = inputs["input_ids"]
    pos = gating_positions(ids, model.gating_pattern)
    end_of_turn = tok.convert_tokens_to_ids("<end_of_turn>")
    for row, p in zip(ids.tolist(), pos.tolist()):
        assert row[p] == end_of_turn and row.index(end_of_turn) == p          # the FIRST <end_of_turn>: the user's
    with torch.no_grad():
        gates = model(**inputs).gating_output.float()
    assert torch.equal(gates[0], gates[1])                                  # same prompt, other response
    assert not torch.allclose(gates[0], gates[2])                           # other prompt
    with pytest.raises(ValueError, match="pattern"):
        gating_positions(torch.tensor([[2, 3, 4]]), model.gating_pattern)


# --------------------------------------------------------------------------- the pipeline --------------
def test_pipeline_reproduces_the_score_and_the_gate():
    tok, model = _tokenizer(), _model()
    assert verify_score_path(model, tok, max_length=256) < 0.02
    texts = [_conv(tok, "should this loan be approved ?", "approve ."),
             _conv(tok, "grade this short essay .", "the essay has a stable and short argument , approve .")]
    base, _ = get_rewards_both(model, tok, texts, None, batch_size=2, max_length=256, show_progress=False)
    with torch.no_grad():
        logits = model(**tok(texts, return_tensors="pt", padding=True)).logits.reshape(-1).float()
    assert torch.allclose(base.float(), logits, atol=1e-2)


def test_nulling_projects_the_state_and_holds_the_gate():
    tok, model = _tokenizer(), _model()
    texts = [_conv(tok, "should this loan be approved ?", w) for w in ("approve .", "decline .", "approve the loan .")]
    h, dtype, gates = embed_with_gates(model, tok, texts, batch_size=3, max_length=256, show_progress=False)
    torch.manual_seed(1)
    u = torch.randn(h.shape[1])
    u = u / u.norm()
    base, nulled = rewards_from_hidden(model, h, dtype, u, gates=gates)
    head = get_head(model)
    assert isinstance(head, QuantileGatedHead)
    with torch.no_grad():
        expect = head.score(project_to_null_space(h, u).to(dtype), gates)
    assert torch.equal(nulled, expect) and not torch.allclose(base, nulled)
    # for a fixed gate the score is linear in the state: the text's effective head gate @ R̄
    eff = head.effective_weights(gates)
    assert torch.allclose((eff * h).sum(-1), base.float(), atol=0.05)
    with pytest.raises(ValueError, match="gate"):
        rewards_from_hidden(model, h, dtype, u)


def test_embedding_cache_keeps_the_gates(tmp_path, monkeypatch):
    monkeypatch.delenv(ec.ENV_VAR, raising=False)
    tok, model = _tokenizer(), _model()
    cache = ec.attach(model, tok, str(tmp_path))
    assert cache.gates is not None and cache.fingerprint["head_kind"] == "quantile_gated"
    assert "gate_digest" in cache.fingerprint
    texts = [_conv(tok, "should this loan be approved ?", "approve ."), _conv(tok, "grade this essay .", "decline .")]
    h1, dtype, g1 = embed_with_gates(model, tok, texts, batch_size=2, max_length=256, show_progress=False)
    misses = cache.misses
    h2, _, g2 = embed_with_gates(model, tok, texts, batch_size=2, max_length=256, show_progress=False)
    assert cache.misses == misses and torch.equal(h1, h2) and torch.equal(g1, g2)   # served from the cache
    # offline, without the model: the same rewards from the stored head, states and gates
    offline = ec.open_cache(cache.directory)
    states, gates = offline.lookup(texts, 256), offline.lookup_gates(texts, 256)
    u = torch.randn(h1.shape[1])
    base, nulled = rewards_from_hidden(model, h1, dtype, u, gates=g1)
    assert torch.equal(ec.offline_rewards(offline, states, gates=gates), base)
    assert torch.equal(ec.offline_rewards(offline, states, u, gates=gates), nulled)
    with pytest.raises(ValueError, match="gates"):
        ec.offline_rewards(offline, states)


def test_linear_models_keep_their_fingerprint(tmp_path, monkeypatch):
    from tests.test_embedding_cache import _model as linear_model, _tokenizer as linear_tokenizer

    monkeypatch.delenv(ec.ENV_VAR, raising=False)
    fp = ec.model_fingerprint(linear_model(), linear_tokenizer())
    assert "head_kind" not in fp and "gate_digest" not in fp
    cache = ec.attach(linear_model(), linear_tokenizer(), str(tmp_path))
    assert cache.gates is None


def test_the_loader_uses_our_class_for_the_qrm_architecture(tmp_path):
    from scoring.backend import _own_class

    tok, model = _tokenizer(), _model()
    model.config.architectures = [ARCHITECTURE]
    model.save_pretrained(tmp_path / "qrm")
    assert _own_class(str(tmp_path / "qrm")) is Gemma2ForQuantileSequenceClassification
    loaded = Gemma2ForQuantileSequenceClassification.from_pretrained(tmp_path / "qrm", dtype=torch.bfloat16).eval()
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    # any other architecture goes to AutoModelForSequenceClassification (only the config is read)
    plain = _config(tok, architectures=["Gemma2ForSequenceClassification"])
    plain.save_pretrained(tmp_path / "plain")
    assert _own_class(str(tmp_path / "plain")) is None


# --------------------------------------------------------------------------- cross-marker --------------
def test_cross_marker_reports_the_gate_pathway(monkeypatch):
    from runners.run_cross_marker import gate_fixed_column

    monkeypatch.setenv(ec.ENV_VAR, "off")
    tok, model = _tokenizer(), _model()
    prompts = {None: "should this loan be approved ?", "f": "should this loan be approved ? woman",
               "m": "should this loan be approved ? man"}
    rows, convs = [], []
    for cell, prompt in prompts.items():
        for response in ("approve .", "decline ."):
            rows.append({"record_id": "r0", "template_id": "t", "encoding": "explicit", "response": response,
                         "cell": "unmarked" if cell is None else [cell]})
            convs.append(_conv(tok, prompt, response))
    x_idx = list(range(len(rows)))
    h, dtype, g = embed_with_gates(model, tok, convs, batch_size=6, max_length=256, show_progress=False)
    base, _ = rewards_from_hidden(model, h, dtype, None, gates=g)
    for i, r in zip(x_idx, base):
        rows[i]["baseline"] = float(r)
    direct = [{"baseline": 0.5}]
    assert gate_fixed_column(model, rows, x_idx, direct, [0], h, g, dtype)
    # unmarked rows keep their reward; marked rows are rescored with the unmarked prompt's gate
    for i in (0, 1):
        assert rows[i]["gate_fixed"] == pytest.approx(rows[i]["baseline"])
    with torch.no_grad():
        fixed = get_head(model).score(h[2:3].to(dtype), g[0:1])
    assert rows[2]["gate_fixed"] == pytest.approx(float(fixed[0]))
    assert direct[0]["gate_fixed"] == 0.5
    # no unmarked control: no column
    marked = [dict(r) for r in rows[2:]]
    assert not gate_fixed_column(model, marked, list(range(4)), [], [], h[2:], g[2:], dtype)


def test_score_encoding_end_to_end_on_a_gated_model(manifest, monkeypatch):
    from pairs.cross_marker import load_cell_blocks
    from pairs.factorial import CREDIT_DESIGN
    from runners.run_cross_marker import (
        block_fits, build_direct_rows, build_rows, resolve_settings, score_encoding, select_records, token_counter,
    )
    from substrates.domains import get_domain

    monkeypatch.setenv(ec.ENV_VAR, "off")
    dom, tok, model = get_domain("credit"), _tokenizer(), _model()
    settings = resolve_settings({}, {"n_strong": 4, "n_weak": 4, "n_folds": 2, "alphas": [0.0, 1.0],
                                     "directions": ["prompt", "unfair"]})
    fmt = lambda p, r: format_conversation(tok, p, r)
    blocks = load_cell_blocks(manifest.parent / "cells.jsonl", CREDIT_DESIGN)
    selected, _ = select_records(blocks, quality_field="credit_good", encodings=["explicit"],
                                 templates=list(dom.template_ids), exclude=set(), n_strong=4, n_weak=4, seed=42,
                                 fits=block_fits("credit", settings, fmt, token_counter(tok), 1024))
    rows, convs = build_rows(selected, "credit", "credit_good", fmt, settings)
    drows, dconvs = build_direct_rows(selected, CREDIT_DESIGN, "credit_good", dom.assessment_prompt, fmt)
    columns, geometry, _, _ = score_encoding(model, tok, "explicit", CREDIT_DESIGN, rows, convs, drows, dconvs,
                                             {}, settings, batch_size=16, max_length=1024, show_progress=False)
    assert columns[:2] == ["baseline", "gate_fixed"]
    assert all("gate_fixed" in r for r in rows + drows)
    unmarked = [r for r in rows if r["cell"] == "unmarked"]
    assert unmarked and all(r["gate_fixed"] == pytest.approx(r["baseline"]) for r in unmarked)
    assert any(abs(r["gate_fixed"] - r["baseline"]) > 1e-4 for r in rows if r["cell"] != "unmarked")
    assert all(r["gate_fixed"] == r["baseline"] for r in drows)
    assert geometry["head"].startswith("quantile_gated")
