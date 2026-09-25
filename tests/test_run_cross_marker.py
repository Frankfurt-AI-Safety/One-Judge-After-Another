"""
Tests for `runners/run_cross_marker.py`: settings precedence, record selection (probe exclusion, complete
blocks, the length guard), row building, and — on a tiny random Llama RM (CPU) — the direct directions,
scoring and metrics end to end on a generator-style credit manifest.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForSequenceClassification, PreTrainedTokenizerFast

from pairs.cross_marker import RESPONSE_TYPES, block_from_row
from pairs.factorial import CREDIT_DESIGN, build_factorial_rows
from runners.run_cross_marker import (
    DEFAULTS, DIRECTION_SOURCES, PhaseTimer, apply_overrides, block_fits, build_direct_rows, build_parser,
    build_rows, cells_path, direct_directions, print_report, resolve_settings, score_encoding, select_records,
    timing_report, token_counter,
)
from scoring.cross_marker_metrics import cross_marker_metrics, placement_check
from scoring.dataset_base import format_conversation
from substrates.domains import get_domain
from tests.test_cross_marker import _cell_rows, _record

DOM = get_domain("credit")


def _blocks(n_strong=4, n_weak=4, encodings=("explicit", "proxy")):
    recs = [_record("credit", f"s{i}", True) for i in range(n_strong)]
    recs += [_record("credit", f"w{i}", False) for i in range(n_weak)]
    return [block_from_row(r, CREDIT_DESIGN) for r in _cell_rows("credit", recs, encodings=encodings)]


def _select(blocks, **kw):
    args = dict(quality_field="credit_good", encodings=["explicit", "proxy"],
                templates=["credit_v1", "credit_v2"], exclude=set(), n_strong=2, n_weak=2, seed=42)
    args.update(kw)
    return select_records(blocks, **args)


# --------------------------------------------------------------------------- settings ----------------
class TestSettings:
    def test_precedence(self):
        s = resolve_settings({"cross_marker": {"n_strong": 50, "paraphrases": 1}},
                             {"n_strong": 8, "n_weak": None})
        assert s["n_strong"] == 8 and s["paraphrases"] == 1 and s["n_weak"] == DEFAULTS["n_weak"]

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="n_stong"):
            resolve_settings({"cross_marker": {"n_stong": 5}}, {})

    def test_unknown_direction_raises(self):
        with pytest.raises(ValueError, match="directions"):
            resolve_settings({}, {"directions": ["direct", "leace"]})

    def test_cells_next_to_pairs(self):
        assert str(cells_path("data/x/pairs.jsonl", "d/p.jsonl", None)) == "data/x/cells.jsonl"
        assert str(cells_path(None, "d/p.jsonl", None)) == "d/cells.jsonl"
        assert str(cells_path("a/pairs.jsonl", "d/p.jsonl", "c.jsonl")) == "c.jsonl"

    def test_config_files_parse(self):
        from scoring.experiment import ExperimentConfig
        for name in ("credit_crossmarker_qwen06", "cv_crossmarker_qwen06", "edu_crossmarker_persuade_qwen06"):
            cfg = ExperimentConfig.from_yaml(f"configs/demographic_{name}.yaml")
            s = resolve_settings(cfg.extra, {})
            assert s["directions"] == list(DIRECTION_SOURCES) and set(s["encodings"]) == {"explicit", "proxy"}
            assert s["n_folds"] == 5 and s["placement_check"] is True and s["length_bins"] == 5
            assert cfg.dataset_source.endswith("pairs.jsonl")

    def test_cli_overrides_the_config(self):
        from scoring.experiment import ExperimentConfig
        path = "configs/demographic_credit_crossmarker_qwen06.yaml"
        cfg = apply_overrides(ExperimentConfig.from_yaml(path), build_parser().parse_args(
            ["--model", "Skywork/Skywork-Reward-V2-Llama-3.1-8B", "--batch-size", "32", "--probe-records", "10",
             "--device", "cpu"]))
        assert (cfg.model_path, cfg.batch_size, cfg.probe_records, cfg.device) == (
            "Skywork/Skywork-Reward-V2-Llama-3.1-8B", 32, 10, "cpu")
        # no flag: the config's values stand
        plain = ExperimentConfig.from_yaml(path)
        cfg = apply_overrides(ExperimentConfig.from_yaml(path), build_parser().parse_args([]))
        assert (cfg.model_path, cfg.batch_size, cfg.probe_records, cfg.device) == (
            plain.model_path, plain.batch_size, plain.probe_records, plain.device)


# --------------------------------------------------------------------------- timing ------------------
class TestTiming:
    def test_laps_are_booked_to_their_phase_and_summed(self, monkeypatch):
        clock = iter([0.0, 2.0, 5.0, 6.0, 10.0])
        monkeypatch.setattr("runners.run_cross_marker.time.perf_counter", lambda: next(clock))
        timer = PhaseTimer()                                    # start at 0
        timer.lap("load_model")                                 # 0 -> 2
        timer.lap("embed")                                      # 2 -> 5
        timer.lap("load_model")                                 # 5 -> 6, summed
        assert timer.seconds == {"load_model": 3.0, "embed": 3.0}
        assert timer.total() == 10.0

    def test_report(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        timer = PhaseTimer()
        timer.seconds = {"load_model": 12.34, "embed": 50.0}
        rep = timing_report(timer, batch_size=16, texts={"direct_directions": 400, "scoring": 1000})
        assert rep["seconds"] == {"load_model": 12.3, "embed": 50.0} and rep["batch_size"] == 16
        assert rep["scoring_texts_per_s"] == 20.0 and "peak_gpu_memory_gib" not in rep
        # cache off: nothing is known about what went through the model
        rep = timing_report(timer, batch_size=16, texts=None)
        assert rep["texts_through_model"] is None and "scoring_texts_per_s" not in rep

    def test_report_line(self, capsys):
        sel = {"n_strong": 20, "requested_strong": 20, "n_weak": 20, "requested_weak": 20}
        summary = {"domain": "credit", "model": "m", "selection": sel, "metrics": {},
                   "timing": {"seconds": {"load_model": 12.0, "embed": 50.0}, "total_s": 70.0, "batch_size": 16,
                              "texts_through_model": {"direct_directions": 400, "scoring": 1000},
                              "scoring_texts_per_s": 20.0,
                              "peak_gpu_memory_gib": {"allocated": 3.2, "reserved": 3.6}}}
        print_report(summary)
        line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("timing"))
        assert "load_model 12s | embed 50s | total 70s" in line and "400 + 1000 texts" in line
        assert "20 texts/s" in line and "3.2 GiB allocated" in line


# --------------------------------------------------------------------------- selection ---------------
class TestSelection:
    def test_counts_and_labels(self):
        selected, rep = _select(_blocks())
        assert rep["n_strong"] == 2 and rep["n_weak"] == 2
        assert sum(r.startswith("s") for r in selected) == 2
        assert all(len(bs) == 4 for bs in selected.values())          # 2 encodings x 2 templates

    def test_probe_records_are_excluded(self):
        selected, rep = _select(_blocks(), exclude={"s0", "s1", "w3"}, n_strong=4, n_weak=4)
        assert not {"s0", "s1", "w3"} & set(selected)
        assert rep["excluded_probe_records"] == 3 and rep["n_strong"] == 2 and rep["n_weak"] == 3

    def test_incomplete_records_are_skipped(self):
        blocks = [b for b in _blocks() if not (b.record_id == "s2" and b.encoding == "proxy"
                                               and b.template_id == "credit_v2")]
        selected, rep = _select(blocks, n_strong=4)
        assert "s2" not in selected and rep["incomplete_records"] == 1

    def test_length_guard_skips_whole_records(self):
        fits = lambda bs: bs[0].record_id != "s1"
        selected, rep = _select(_blocks(), fits=fits, n_strong=4)
        assert "s1" not in selected and rep["too_long_records"] == 1 and rep["n_strong"] == 3

    def test_deterministic_and_seeded(self):
        a, _ = _select(_blocks(8, 8), n_strong=3, n_weak=3)
        b, _ = _select(_blocks(8, 8), n_strong=3, n_weak=3)
        c, _ = _select(_blocks(8, 8), n_strong=3, n_weak=3, seed=7)
        assert list(a) == list(b) and set(a) != set(c)

    def test_block_fits_uses_the_formatted_conversations(self):
        settings = resolve_settings({}, {})
        count = lambda conv: len(" ".join(conv).split())
        fmt = lambda p, r: (p, r)
        blocks = [b for b in _blocks(1, 0) if b.encoding == "explicit"]
        assert block_fits("credit", settings, fmt, count, 10_000)(blocks)
        assert not block_fits("credit", settings, fmt, count, 50)(blocks)


# --------------------------------------------------------------------------- rows --------------------
def test_rows_carry_ids_not_text():
    selected, _ = _select(_blocks())
    settings = resolve_settings({}, {})
    rows, convs = build_rows(selected, "credit", "credit_good", lambda p, r: (p, r), settings)
    assert len(rows) == len(convs) == 4 * 4 * 9 * len(RESPONSE_TYPES)
    assert set(rows[0]) == {"record_id", "template_id", "encoding", "cell", "response", "paraphrase", "strong"}
    assert {json.dumps(r["cell"]) for r in rows} >= {'"unmarked"', '["female", 30, "married"]'}
    # the conversation's response is the row's response type at the record's paraphrase
    from pairs.cross_marker import DECISION_RESPONSES
    for row, (prompt, text) in zip(rows[:36], convs[:36]):
        assert text == DECISION_RESPONSES["credit"].text(row["response"], row["paraphrase"])


def test_direct_rows_are_the_cells_in_the_response():
    selected, _ = _select(_blocks())
    rows, convs = build_direct_rows(selected, CREDIT_DESIGN, "credit_good", DOM.assessment_prompt,
                                    lambda p, r: (p, r))
    assert len(rows) == len(convs) == 4 * 4 * 8
    block = selected[rows[0]["record_id"]][0]
    assert convs[0] == (DOM.assessment_prompt, block.texts[tuple(rows[0]["cell"])])
    assert set(rows[0]) == {"record_id", "template_id", "encoding", "cell", "strong"}


# --------------------------------------------------------------------------- tiny model ---------------
_WORDS = ("the applicant is a woman man married single year old loan credit approve decline approved "
          "should be i recommend would not profile summary savings checking account 30 50").split()


def _tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(dict.fromkeys(_WORDS))}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")
    hf.padding_side = "right"
    return hf


def _model():
    cfg = LlamaConfig(vocab_size=len(dict.fromkeys(_WORDS)) + 2, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, pad_token_id=0,
                      num_labels=1, max_position_embeddings=1024)
    torch.manual_seed(0)
    return LlamaForSequenceClassification(cfg).to(torch.bfloat16).eval()


@pytest.fixture
def manifest(tmp_path):
    """A credit pairs.jsonl + cells.jsonl as the generator writes them (16 records, both encodings,
    both templates)."""
    recs = [_record("credit", f"s{i}", True) for i in range(8)] + \
           [_record("credit", f"w{i}", False) for i in range(8)]
    pair_rows, cell_rows, _ = build_factorial_rows(
        recs, design=CREDIT_DESIGN, render_fn=DOM.render_fn, id_prefix="credit", domain="credit",
        real_fields=lambda r: {"credit_good": r.credit_good}, axes=DOM.axes, encodings=("explicit", "proxy"),
        templates=DOM.template_ids, seed=42, validate=lambda p: SimpleNamespace(ok=True, reasons=[]),
        content_label="financial_content")
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in pair_rows))
    (tmp_path / "cells.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cell_rows))
    return path


def test_direct_directions_share_the_probe_records(manifest, monkeypatch):
    # Audit item 4.1: with probe_records every direction rests on the same N records, stratified by
    # quality, so the records excluded from the cross-marker evaluation are exactly those N.
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    model, tok = _model(), _tokenizer()
    _, probe_ids, meta = direct_directions(
        model, tok, DOM, str(manifest), ["explicit", "proxy"], probe_size=8, split_seed=42,
        batch_size=16, device="cpu", max_length=1024, probe_records=6)
    assert len(probe_ids) == 6 and sum(r.startswith("s") for r in probe_ids) == 3
    assert {m["n_records"] for m in meta.values()} == {6}
    assert all(m["split"]["probe_strata"] == {"False": 3, "True": 3} for m in meta.values())


def test_end_to_end_on_a_tiny_model(manifest, monkeypatch):
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    from pairs.cross_marker import load_cell_blocks
    from probes.probe import get_rewards_both, rewards_from_hidden, embed_states

    model, tok = _model(), _tokenizer()
    directions, probe_ids, meta = direct_directions(
        model, tok, DOM, str(manifest), ["explicit", "proxy"], probe_size=8, split_seed=42,
        batch_size=8, device="cpu", max_length=1024)
    # every factorial axis with pairs in the encoding: no marital direction under proxy
    assert set(directions["explicit"]) == {"sex", "age", "marital_status", "intersection"}
    assert set(directions["proxy"]) == {"sex", "age", "intersection"}
    assert probe_ids and len(probe_ids) < 16
    assert all(m["n_records"] >= 1 and "split_half_cosine" in m for m in meta.values())

    settings = resolve_settings({}, {"n_strong": 8, "n_weak": 8, "n_folds": 3, "alphas": [0.0, 1.0]})
    fmt = lambda p, r: format_conversation(tok, p, r)
    blocks = load_cell_blocks(manifest.parent / "cells.jsonl", CREDIT_DESIGN)
    selected, rep = select_records(
        blocks, quality_field="credit_good", encodings=["explicit"], templates=list(DOM.template_ids),
        exclude=probe_ids, n_strong=8, n_weak=8, seed=42,
        fits=block_fits("credit", settings, fmt, token_counter(tok), 1024))
    assert not set(selected) & probe_ids and rep["too_long_records"] == 0
    rows, convs = build_rows(selected, "credit", "credit_good", fmt, settings)
    drows, dconvs = build_direct_rows(selected, CREDIT_DESIGN, "credit_good", DOM.assessment_prompt, fmt)
    timer = PhaseTimer()
    columns, geometry, sweep, saved = score_encoding(
        model, tok, "explicit", CREDIT_DESIGN, rows, convs, drows, dconvs, directions["explicit"], settings,
        batch_size=16, max_length=1024, show_progress=False, timer=timer)
    assert set(timer.seconds) == {"embed", "mechanism"} and all(v > 0 for v in timer.seconds.values())
    axes = ("sex", "age", "marital_status", "intersection")
    assert columns == (["baseline"] + [f"null_direct:{a}" for a in axes] + ["null_direct:joint"]
                       + [f"null_prompt:{a}" for a in axes] + [f"null_interaction:{a}" for a in axes]
                       + ["null_unfair"])
    assert all(set(columns) <= set(r) for r in rows + drows) and "_alpha" not in rows[0]

    # the baseline and direct columns are the pipeline's rewards for the same texts
    base, nulled = get_rewards_both(model, tok, convs[:30], directions["explicit"]["sex"], batch_size=16,
                                    device="cpu", max_length=1024, show_progress=False)
    assert [r["baseline"] for r in rows[:30]] == pytest.approx(base.tolist(), abs=1e-6)
    assert [r["null_direct:sex"] for r in rows[:30]] == pytest.approx(nulled.tolist(), abs=1e-6)

    # cross-fitted: a record's column uses the direction fitted without its fold
    folds = saved["folds"]
    rid = rows[0]["record_id"]
    idx = [i for i, r in enumerate(rows) if r["record_id"] == rid][:10]
    h, dtype = embed_states(model, tok, [convs[i] for i in idx], batch_size=16, max_length=1024,
                            show_progress=False)
    u = saved["cross_fitted"]["interaction:sex"][folds[rid]]
    _, expect = rewards_from_hidden(model, h, dtype, u)
    assert [rows[i]["null_interaction:sex"] for i in idx] == pytest.approx(expect.tolist(), abs=1e-6)
    assert not all(torch.equal(u, v) for v in saved["cross_fitted"]["interaction:sex"].values())

    # geometry: a symmetric cosine matrix over every direction, with unit diagonal
    names = geometry["directions"]
    assert {"direct:sex", "prompt:sex", "interaction:sex", "unfair"} <= set(names)
    cos = geometry["cosine"]
    for i in range(len(names)):
        assert cos[i][i] == pytest.approx(1.0, abs=1e-5)
        for j in range(len(names)):
            assert cos[i][j] == pytest.approx(cos[j][i], abs=1e-6)
    assert set(geometry["did_share"]) == set(axes)
    # w·Δ_int from the states reproduces the reward-based disparity of the same (strong) records
    mb = cross_marker_metrics(rows, CREDIT_DESIGN, "explicit", reward_key="baseline", n_boot=1)
    for axis in axes:
        assert geometry["did_from_states"][axis] == pytest.approx(
            mb["margins"]["D"]["strong"]["disparity"][axis]["mean"], abs=0.02)

    # the α-sweep's ends are the baseline and the nulled columns
    m0 = cross_marker_metrics(rows, CREDIT_DESIGN, "explicit", reward_key="baseline", n_boot=1)
    for name, column in (("direct:sex", "null_direct:sex"), ("interaction:sex", "null_interaction:sex")):
        m1 = cross_marker_metrics(rows, CREDIT_DESIGN, "explicit", reward_key=column, n_boot=1)
        g = "strong"
        assert sweep[name]["0.0"]["disparity"] == pytest.approx(
            m0["margins"]["D"][g]["disparity"]["sex"]["mean"], abs=1e-9)
        assert sweep[name]["1.0"]["disparity"] == pytest.approx(
            m1["margins"]["D"][g]["disparity"]["sex"]["mean"], abs=1e-9)

    # document lengths: one per record and template, the unmarked text's tokens
    from runners.run_cross_marker import document_lengths
    lengths = document_lengths(selected, tok)
    assert set(lengths) == {(r, b.template_id) for r, bs in selected.items() for b in bs}
    for row in rows:
        row["doc_tokens"] = lengths[(row["record_id"], row["template_id"])]
    q = cross_marker_metrics(rows, CREDIT_DESIGN, "explicit", reward_key="baseline", n_boot=20)["quality_tracking"]
    assert {"auc_d", "auc_length", "auc_d_within_length_strata", "auc_d_length_residualised"} <= set(q)

    # the placement check runs on both sides of every column
    pc = placement_check(drows, rows, CREDIT_DESIGN, "explicit", reward_key="null_prompt:sex", n_boot=20)
    assert {"direct_gap", "prompt_effect", "did"} <= set(pc["strong"]["axes"]["sex"])
