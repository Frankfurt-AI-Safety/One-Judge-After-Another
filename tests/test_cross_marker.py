"""
Unit tests for the cross-marker decision design's items (`pairs/cross_marker.py`): responses that name
no attribute, blocks read from a generator-style cells.jsonl, 9 prompts x 4 responses per block, prompts
that differ only by the marker clause, one paraphrase per record. No model required.
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from pairs.cross_marker import (
    DECISION_RESPONSES, RESPONSE_TYPES, UNMARKED, BlockMismatch, attribute_words, block_from_row,
    build_block_items, decision_prompt, fits_max_length, load_cell_blocks, paraphrase_index,
    response_violations,
)
from pairs.factorial import DESIGNS, build_factorial_rows
from pairs.verdicts import DECISION_FRAMES
from substrates.bios_ingest import RealCVRecord
from substrates.domains import get_domain

DOMAINS = ("credit", "cv", "education")


def _bio(rid="bios-0001", qualified=True):
    return RealCVRecord(
        source_record_id=rid,
        bio_text=("A data professional with twelve years in industry, specialising in distributed "
                  "systems and applied statistics, who has led teams at two firms."),
        profession="project_manager", target_role="project_manager", role="a project coordinator",
        qualified=qualified, gender=0,
    )


def _record(domain, rid, strong=True):
    if domain == "cv":
        return _bio(rid, strong)
    if domain == "credit":
        from tests.test_credit_pipeline import _fake_record
        return dataclasses.replace(_fake_record(rid), credit_good=strong)
    from tests.test_education_pipeline import _fake_record
    return _fake_record(rid, high_quality=strong)


# The generators' real_fields, reduced to what this design reads.
_REAL_FIELDS = {
    "credit": lambda r: {"credit_good": r.credit_good},
    "cv": lambda r: {"role": r.role, "qualified": r.qualified},
    "education": lambda r: {"high_quality": r.high_quality},
}


def _cell_rows(domain, records, encodings=("explicit", "proxy"), templates=None):
    """cells.jsonl rows exactly as `build_factorial_rows` writes them, after a JSON round trip."""
    dom = get_domain(domain)
    _, rows, _ = build_factorial_rows(
        records, design=DESIGNS[domain], render_fn=dom.render_fn, id_prefix=domain, domain=domain,
        real_fields=_REAL_FIELDS[domain], axes=("sex",), encodings=encodings,
        templates=templates or dom.template_ids, seed=42,
        validate=lambda p: SimpleNamespace(ok=True, reasons=[]), content_label="content",
        subject=DECISION_FRAMES[domain].subject)
    return [json.loads(json.dumps(r)) for r in rows]


def _block(domain, rid="rec-1", strong=True, encoding="explicit"):
    rows = _cell_rows(domain, [_record(domain, rid, strong)], encodings=(encoding,))
    return block_from_row(rows[0], DESIGNS[domain])


# --------------------------------------------------------------------------- responses ---------------
class TestResponses:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_responses_name_no_attribute(self, domain):
        assert response_violations(domain) == {}

    def test_lexicon_covers_values_names_and_categories(self):
        credit = attribute_words("credit")
        assert {"woman", "married", "single", "30", "1996", "gender", "age"} <= credit
        assert "abigail" in credit and "jamal" in credit
        assert {"leave", "employment", "association"} <= attribute_words("cv")
        # the education proxy's school-poverty quantifiers and the explicit clause's income levels
        assert {"most", "few", "income", "black", "white"} <= attribute_words("education")

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_overt_names_every_category_but_no_value(self, domain):
        from pairs.cross_marker import _words, category_words, value_words
        pool = DECISION_RESPONSES[domain]
        axis_words = {"credit": {"sex", "age", "marital"}, "cv": {"sex", "age", "family"},
                      "education": {"sex", "ethnicity"}}[domain]
        for i in range(pool.size):
            words = _words(pool.text("overt", i))
            assert axis_words <= words and not words & value_words(domain)
            assert words & category_words(domain)
        for kind in ("approve", "decline", "coded", "evasive"):
            assert not any(_words(pool.text(kind, i)) & category_words(domain) for i in range(pool.size))

    def test_lexicon_ignores_function_words(self):
        # "on parental leave" / "in continuous employment" differ in "on"/"in", which carry nothing
        assert not {"on", "in", "the", "their"} & attribute_words("cv")

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_pools_are_parallel_and_close_in_length(self, domain):
        pool = DECISION_RESPONSES[domain]
        assert pool.size == 3
        for i in range(pool.size):
            words = [len(pool.text(k, i).split()) for k in ("approve", "decline", "coded", "overt")]
            assert max(words) - min(words) <= 8, (domain, i, words)

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_decisions_are_the_frames_decisions(self, domain):
        frame, pool = DECISION_FRAMES[domain], DECISION_RESPONSES[domain]
        for i in range(pool.size):
            assert pool.text("approve", i).endswith(frame.approve)
            assert pool.text("decline", i).endswith(frame.decline)
            assert pool.text("coded", i).endswith(frame.decline)
            assert pool.text("overt", i).endswith(frame.decline)
        assert pool.text("evasive", 0) == frame.evasive

    def test_unequal_pools_rejected(self):
        from pairs.cross_marker import ResponsePool
        with pytest.raises(ValueError):
            ResponsePool(approve=("a", "b"), decline=("c",), coded=("d",), overt=("o",), evasive=("e",))


# --------------------------------------------------------------------------- blocks ------------------
class TestBlocks:
    @pytest.mark.parametrize("domain", DOMAINS)
    @pytest.mark.parametrize("encoding", ["explicit", "proxy"])
    def test_unmarked_is_the_neutral_render(self, domain, encoding):
        rec = _record(domain, "rec-1")
        dom = get_domain(domain)
        rows = _cell_rows(domain, [rec], encodings=(encoding,))
        for row in rows:
            block = block_from_row(row, DESIGNS[domain])
            assert block.unmarked == dom.render_fn(rec, block.template_id, "")
            assert set(block.texts) == set(DESIGNS[domain].cells)
            for cell, text in block.texts.items():
                assert block.clauses[cell] in text

    def test_cells_keep_their_json_types(self):
        # ages are ints in the design; a JSON round trip must not turn them into strings
        block = _block("credit")
        assert set(block.texts) == set(DESIGNS["credit"].cells)

    def test_tampered_block_is_dropped_and_counted(self, tmp_path):
        rows = _cell_rows("credit", [_record("credit", "a"), _record("credit", "b")],
                          encodings=("explicit",), templates=("credit_v1",))
        rows[1]["cells"][3]["text"] = rows[1]["cells"][3]["text"].replace("Savings", "Savings (verified)")
        path = tmp_path / "cells.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        report = {}
        blocks = load_cell_blocks(path, DESIGNS["credit"], report)
        assert [b.record_id for b in blocks] == ["a"]
        assert report["n_blocks"] == 1 and report["n_dropped_mismatch"] == 1

    def test_missing_clause_raises(self):
        row = _cell_rows("credit", [_record("credit", "a")], encodings=("explicit",))[0]
        row["cells"][0]["text"] = row["cells"][0]["text"].replace(row["cells"][0]["clause"], "")
        with pytest.raises(BlockMismatch):
            block_from_row(row, DESIGNS["credit"])

    def test_missing_file_explains_regeneration(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="regenerate"):
            load_cell_blocks(tmp_path / "cells.jsonl", DESIGNS["credit"])

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_quality_field_is_the_is_strong_label(self, domain):
        dom = get_domain(domain)
        for strong in (True, False):
            rec = _record(domain, "q", strong)
            assert getattr(rec, dom.quality_field) == dom.is_strong(rec) == strong
            assert _block(domain, "q", strong).is_strong(dom.quality_field) == strong


# --------------------------------------------------------------------------- items -------------------
class TestItems:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_nine_prompts_times_every_response(self, domain):
        items = build_block_items(_block(domain), domain)
        assert len(items) == 9 * len(RESPONSE_TYPES)
        assert len({i.cell_key for i in items}) == 9
        assert sum(i.cell is None for i in items) == len(RESPONSE_TYPES)
        assert {i.cell_key for i in items if i.cell is None} == {UNMARKED}
        assert len(build_block_items(_block(domain), domain, include_unmarked=False)) == 8 * len(RESPONSE_TYPES)

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_same_response_text_under_every_cell(self, domain):
        items = build_block_items(_block(domain), domain)
        for kind in RESPONSE_TYPES:
            texts = {i.text for i in items if i.response == kind}
            assert len(texts) == 1, kind
        assert len({i.paraphrase for i in items}) == 1

    @pytest.mark.parametrize("domain", DOMAINS)
    @pytest.mark.parametrize("encoding", ["explicit", "proxy"])
    def test_prompts_differ_only_by_the_clause(self, domain, encoding):
        block = _block(domain, encoding=encoding)
        items = build_block_items(block, domain)
        unmarked_prompt = next(i.prompt for i in items if i.cell is None)
        for item in items:
            if item.cell is not None:
                clause = block.clauses[item.cell]
                assert item.prompt.count(clause) == 1
                assert item.prompt.replace(clause, "", 1) == unmarked_prompt

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_marker_is_in_the_user_turn_only(self, domain):
        block = _block(domain)
        for item in build_block_items(block, domain):
            for clause in block.clauses.values():
                assert clause.strip() not in item.text

    def test_pole_a_cells_are_scored(self):
        # every axis pair of the design (pole A = the penalised level first) has both cells among the
        # items, keyed by the same tuples the metrics will look up
        for domain in DOMAINS:
            design = DESIGNS[domain]
            cells = {i.cell for i in build_block_items(_block(domain), domain)}
            for axis in design.axes + ("intersection",):
                for a, b in design.axis_pairs(axis, "explicit"):
                    assert a in cells and b in cells
                    if axis != "intersection":
                        assert a[design.axes.index(axis)] == design.factors[axis][0]

    def test_paraphrase_is_per_record_and_deterministic(self):
        rec = _record("credit", "same-record")
        rows = _cell_rows("credit", [rec])      # 2 templates x 2 encodings
        idx = {build_block_items(block_from_row(r, DESIGNS["credit"]), "credit")[0].paraphrase
               for r in rows}
        assert len(idx) == 1
        assert idx == {paraphrase_index("same-record", 42, 3)}
        # all three paraphrases are used across records
        assert {paraphrase_index(f"r{i}", 42, 3) for i in range(60)} == {0, 1, 2}
        assert {paraphrase_index(f"r{i}", 42, 1) for i in range(20)} == {0}

    def test_n_paraphrases_bounds(self):
        with pytest.raises(ValueError):
            build_block_items(_block("credit"), "credit", n_paraphrases=4)

    def test_hiring_prompt_names_the_role(self):
        item = build_block_items(_block("cv"), "cv")[0]
        assert item.prompt.startswith("You are screening a candidate for a project coordinator.")
        with pytest.raises(KeyError, match="role"):
            decision_prompt("cv", "some profile", {"qualified": True})

    @pytest.mark.parametrize("domain,question", [("cv", "advance to an interview?"),
                                                 ("credit", "loan be approved?"),
                                                 ("education", "essay pass?")])
    def test_domain_question(self, domain, question):
        assert question in build_block_items(_block(domain), domain)[0].prompt

    def test_education_body_braces_survive(self):
        # the essay body is inserted with str.format as a value, so braces in it are not re-parsed
        from tests.test_education_pipeline import _ESSAY_BODY
        items = build_block_items(_block("education"), "education")
        assert all(_ESSAY_BODY in i.prompt for i in items)


def test_fits_max_length():
    count = lambda text: len(text.split())
    assert fits_max_length(["a b c", "a b"], count, 3)
    assert not fits_max_length(["a b c d", "a b"], count, 3)
