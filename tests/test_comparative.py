"""
Tests for the comparative (two-applicant) design: `pairs/comparative.py` (frames, responses, pairing, items),
`scoring/comparative_metrics.py` (on synthetic reward tables with known effects) and
`probes/comparative_directions.py`.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pytest
import torch

from pairs.comparative import (
    COMPARATIVE_FRAMES, KINDS, ORDERS, PAIRINGS, SIDES, UNMARKED, RecordPair, build_pair_items,
    comparative_violations, context, contrast_axes, draw_pairs, pairing_pool,
)
from pairs.cross_marker import block_from_row, proxy_names
from pairs.factorial import CREDIT_DESIGN, DESIGNS, EDUCATION_DESIGN
from probes.comparative_directions import pair_contrasts
from scoring.comparative_metrics import comparative_metrics, pair_values, summarize_ratio
from tests.test_cross_marker import _cell_rows, _record

DOMAINS = tuple(COMPARATIVE_FRAMES)


def _blocks(domain, rids_strong, encoding="explicit"):
    recs = [_record(domain, rid, strong) for rid, strong in rids_strong]
    rows = _cell_rows(domain, recs, encodings=(encoding,))
    out = {}
    for row in rows:
        b = block_from_row(row, DESIGNS[domain])
        out.setdefault(b.record_id, {})[b.template_id] = b
    return out


# --------------------------------------------------------------------------- frames ------------------
class TestFrames:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_request_and_responses_name_no_attribute(self, domain):
        assert comparative_violations(domain) == {}

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_choices_differ_only_by_the_swapped_labels(self, domain):
        frame = COMPARATIVE_FRAMES[domain]
        swap = lambda t: re.sub(r"\b([AB])\b", lambda m: "B" if m.group(1) == "A" else "A", t)
        for kind in KINDS:
            for i in range(frame.size):
                a, b = frame.response(kind, i, "A"), frame.response(kind, i, "B")
                assert swap(a) == b and a != b
                assert a.count(f"{frame.label} A") >= 1 and a.count(f"{frame.label} B") >= 1

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_merit_and_coded_close_in_characters(self, domain):
        frame = COMPARATIVE_FRAMES[domain]
        for i in range(frame.size):
            assert abs(len(frame.response("merit", i, "A")) - len(frame.response("coded", i, "A"))) <= 20

    @pytest.mark.parametrize("model", ["Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                                       "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
                                       "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"])
    def test_merit_and_coded_token_matched(self, model):
        # within a kind the two choices are the same length by construction; across kinds within 2 tokens
        transformers = pytest.importorskip("transformers")
        try:
            tok = transformers.AutoTokenizer.from_pretrained(model, local_files_only=True)
        except (OSError, ValueError):
            pytest.skip(f"{model} tokenizer not cached")
        n = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
        for domain, frame in COMPARATIVE_FRAMES.items():
            for i in range(frame.size):
                for kind in KINDS:
                    assert n(frame.response(kind, i, "A")) == n(frame.response(kind, i, "B"))
                assert abs(n(frame.response("merit", i, "A")) - n(frame.response("coded", i, "A"))) <= 2, (domain, i)

    def test_bad_arguments(self):
        frame = COMPARATIVE_FRAMES["credit"]
        with pytest.raises(ValueError, match="kind"):
            frame.response("evasive", 0, "A")
        with pytest.raises(ValueError, match="chosen"):
            frame.response("merit", 0, "X")


# --------------------------------------------------------------------------- pairing -----------------
def _candidates(n_strong, n_weak, groups=1):
    c = {f"s{i:03d}": (True, i % groups) for i in range(n_strong)}
    c.update({f"w{i:03d}": (False, i % groups) for i in range(n_weak)})
    return c


class TestPairing:
    def test_pools_are_fixed_per_record(self):
        # a record's pool does not depend on the other records
        assert pairing_pool("s1", True, 42) == pairing_pool("s1", True, 42)
        pools = [pairing_pool(f"r{i}", True, 42) for i in range(3000)]
        assert set(pools) == {"strong_strong", "strong_weak:strong"}
        assert abs(pools.count("strong_strong") / 3000 - 2 / 3) < 0.03
        assert {pairing_pool(f"r{i}", False, 42) for i in range(50)} == {"weak_weak", "strong_weak:weak"}

    def test_each_record_once_and_strata_respected(self):
        cand = _candidates(60, 60, groups=3)
        pairs, rep = draw_pairs(cand, {p: 100 for p in PAIRINGS}, seed=42)
        used = [r for p in pairs for r in (p.x, p.y)]
        assert len(used) == len(set(used))
        for p in pairs:
            assert cand[p.x][1] == cand[p.y][1]                       # same match group
            sx, sy = cand[p.x][0], cand[p.y][0]
            assert (sx, sy) == {"strong_strong": (True, True), "strong_weak": (True, False),
                                "weak_weak": (False, False)}[p.pairing]
        for pairing in PAIRINGS:
            got = [p for p in pairs if p.pairing == pairing]
            assert [p.index for p in got] == list(range(len(got))) and rep[pairing]["n"] == len(got) > 0

    def test_a_smaller_request_is_a_prefix(self):
        cand = _candidates(90, 90, groups=2)
        big, _ = draw_pairs(cand, {p: 1000 for p in PAIRINGS}, seed=7)
        small, rep = draw_pairs(cand, {"strong_strong": 3, "strong_weak": 5, "weak_weak": 0}, seed=7)
        for pairing in PAIRINGS:
            b = [p for p in big if p.pairing == pairing]
            s = [p for p in small if p.pairing == pairing]
            assert s == b[:len(s)]
        assert rep["weak_weak"]["n"] == 0 and rep["strong_weak"]["requested"] == 5

    def test_incompatible_partners_are_skipped_and_counted(self):
        cand = _candidates(30, 30)
        seen = []

        def check(pair):
            seen.append(pair)
            return "shared_name" if pair.y.endswith(("1", "3")) else None

        pairs, rep = draw_pairs(cand, {p: 100 for p in PAIRINGS}, seed=1, compatible=check)
        assert not any(p.y.endswith(("1", "3")) for p in pairs)
        assert sum(rep[p]["skipped"].get("shared_name", 0) for p in PAIRINGS) > 0
        # the check sees the candidate as it will be stored (pairing and index included)
        assert all(isinstance(p, RecordPair) and p.pairing in PAIRINGS for p in seen)

    def test_unknown_pairing_raises(self):
        with pytest.raises(ValueError, match="unknown pairings"):
            draw_pairs(_candidates(4, 4), {"strong": 2}, seed=1)


# --------------------------------------------------------------------------- items -------------------
class TestItems:
    def test_contexts_rotate_and_skip_axes_without_a_marker(self):
        # credit marital status has no proxy; every axis cycles through its 4 settings of the other axes
        assert contrast_axes(CREDIT_DESIGN, "proxy") == ["sex", "age", "intersection"]
        assert contrast_axes(CREDIT_DESIGN, "explicit") == ["sex", "age", "marital_status", "intersection"]
        seen = {context(CREDIT_DESIGN, "sex", "explicit", i) for i in range(8)}
        assert len(seen) == 4 and context(CREDIT_DESIGN, "marital_status", "proxy", 0) is None
        a, b = context(CREDIT_DESIGN, "intersection", "explicit", 3)
        assert a == ("female", 30, "married") and b[0] == "male" and a[1:] != b[1:]

    def test_items_cover_assignments_orders_kinds_and_choices(self):
        blocks = _blocks("credit", [("s1", True), ("s2", True)])
        pair = RecordPair("strong_strong", "s1", "s2", 5)
        x, y = blocks["s1"]["credit_v1"], blocks["s2"]["credit_v1"]
        items = build_pair_items(pair, x, y, "credit", seed=42)
        axes = contrast_axes(CREDIT_DESIGN, "explicit")
        assert len(items) == (len(axes) * 2 + 1) * 2 * len(KINDS) * 2
        assert len({i.paraphrase for i in items}) == 1
        for axis in axes:
            prot, ref = context(CREDIT_DESIGN, axis, "explicit", 5)
            for it in (i for i in items if i.axis == axis):
                first, second = it.order[0], it.order[1]
                docs = {"X": x, "Y": y}
                want_first = docs[first].texts[prot if first == it.protected else ref]
                want_second = docs[second].texts[prot if second == it.protected else ref]
                assert want_first in it.prompt and want_second in it.prompt
                assert it.prompt.index(want_first) < it.prompt.index(want_second)
                letter = "A" if it.chosen == first else "B"
                assert it.text == COMPARATIVE_FRAMES["credit"].response(it.kind, it.paraphrase, letter)
        unmarked = [i for i in items if i.axis == UNMARKED]
        assert len(unmarked) == 2 * len(KINDS) * 2 and all(i.protected is None for i in unmarked)
        assert all(x.unmarked in i.prompt and y.unmarked in i.prompt for i in unmarked)

    def test_hiring_needs_one_role(self):
        blocks = _blocks("cv", [("a", True), ("b", True)])
        x, y = blocks["a"]["bios_v1"], blocks["b"]["bios_v1"]
        pair = RecordPair("strong_strong", "a", "b", 0)
        if x.real_fields["role"] == y.real_fields["role"]:
            items = build_pair_items(pair, x, y, "cv")
            assert all(f"a position as {x.real_fields['role']}" in i.prompt for i in items)
        import dataclasses
        y2 = dataclasses.replace(y, real_fields={**y.real_fields, "role": "a pilot"})
        x2 = dataclasses.replace(x, real_fields={**x.real_fields, "role": "a nurse"})
        with pytest.raises(ValueError, match="different roles"):
            build_pair_items(pair, x2, y2, "cv")

    def test_blocks_must_share_encoding_and_template(self):
        blocks = _blocks("credit", [("s1", True), ("s2", True)])
        with pytest.raises(ValueError, match="encoding/template"):
            build_pair_items(RecordPair("strong_strong", "s1", "s2", 0), blocks["s1"]["credit_v1"],
                             blocks["s2"]["credit_v2"], "credit")

    def test_proxy_names_come_from_the_exemplar(self):
        assert proxy_names({"female_name": "Claire", "male_name": "Hunter"}) == {"Claire", "Hunter"}
        assert proxy_names({"names": {"female_white": "A", "male_black": "B"}}) == {"A", "B"}
        assert proxy_names({"design": "x"}) == frozenset()
        proxy = _blocks("education", [("e1", True)], encoding="proxy")["e1"]
        b = next(iter(proxy.values()))
        assert len(b.names) == 4 and all(n in "".join(b.texts.values()) for n in b.names)
        assert EDUCATION_DESIGN.proxy_axes == ("sex", "ethnicity", "economic_status")


# --------------------------------------------------------------------------- metrics -----------------
def _reward_rows(pairs, reward, axes=("sex",), kinds=KINDS, templates=("t1",), unmarked=True):
    """Rows for ``pairs`` [(pair_id, pairing)] with ``reward(pair, template, axis, protected, order, kind, chosen)``."""
    rows = []
    for (pid, pairing), t in itertools.product(pairs, templates):
        for axis in list(axes) + ([UNMARKED] if unmarked else []):
            for prot in ([None] if axis == UNMARKED else SIDES):
                for order, kind, chosen in itertools.product(ORDERS, kinds, SIDES):
                    rows.append({"pair_id": pid, "pairing": pairing, "template_id": t, "encoding": "explicit",
                                 "axis": axis, "protected": prot, "order": order, "kind": kind, "chosen": chosen,
                                 "r": reward(pid, t, axis, prot, order, kind, chosen)})
    return rows


PAIRS = [(f"strong_weak:{i}", "strong_weak") for i in range(6)] + [(f"strong_strong:{i}", "strong_strong")
                                                                      for i in range(5)]


def _structured(delta=-0.4, quality=1.0, position=0.3, coded_extra=-0.2):
    rng = np.random.default_rng(0)
    level = {p: rng.normal() for p, _ in PAIRS}              # per-pair noise shared by all its texts
    record = {p: rng.normal(size=2) for p, _ in PAIRS}       # per-record appeal, cancels in the marker effect

    def reward(pid, t, axis, prot, order, kind, chosen):
        r = level[pid] + record[pid][SIDES.index(chosen)]
        r += position if chosen == order[0] else 0.0
        if pid.startswith("strong_weak"):
            r += quality if chosen == "X" else 0.0
        if prot is not None and chosen == prot:
            r += delta + (coded_extra if kind == "coded" else 0.0)
        return r
    return reward, record


class TestMetrics:
    def test_marker_effect_position_quality_and_exchange_rate(self):
        reward, record = _structured()
        m = comparative_metrics(_reward_rows(PAIRS, reward), "r", n_boot=200, seed=0)
        sw, ss = m["strong_weak"]["sex"], m["strong_strong"]["sex"]
        assert sw["merit"]["marker_effect"]["mean"] == pytest.approx(-0.4)
        assert ss["merit"]["marker_effect"]["mean"] == pytest.approx(-0.4)
        assert sw["coded"]["marker_effect"]["mean"] == pytest.approx(-0.6)
        assert sw["coded_minus_merit"]["mean"] == pytest.approx(-0.2)
        assert sw["merit"]["position_effect"]["mean"] == pytest.approx(0.3)
        # the quality margin keeps the records' own appeal; the marker effect does not
        q = np.mean([1.0 + record[p][0] - record[p][1] for p, g in PAIRS if g == "strong_weak"])
        assert sw["merit"]["quality_margin"]["mean"] == pytest.approx(q)
        assert sw["merit"]["exchange_rate"]["mean"] == pytest.approx(-0.4 / q)
        assert "exchange_rate" not in ss["merit"] and "quality_margin" not in ss["merit"]
        assert m["all"]["n_pairs"] == len(PAIRS) and m["all"]["sex"]["merit"]["marker_effect"]["mean"] == pytest.approx(-0.4)
        assert m["strong_weak"][UNMARKED]["position_effect"]["mean"] == pytest.approx(0.3)
        assert UNMARKED not in m["all"]

    def test_constant_marker_effect_has_a_degenerate_interval_and_scaled_units(self):
        reward, _ = _structured()
        s = comparative_metrics(_reward_rows(PAIRS, reward), "r", n_boot=200)["strong_strong"]["sex"]["merit"]
        me = s["marker_effect"]
        assert me["ci_low"] == pytest.approx(-0.4) and me["ci_high"] == pytest.approx(-0.4) and me["sd"] < 1e-9
        assert me["scale_sd"] > 0 and me["scaled_mean"] == pytest.approx(-0.4 / me["scale_sd"])

    def test_accuracy_gap_reads_the_overturn(self):
        # the weak applicant wins exactly when the strong one carries the protected cell
        def reward(pid, t, axis, prot, order, kind, chosen):
            base = 1.0 if chosen == "X" else 0.0
            return base - (2.0 if prot == "X" and chosen == "X" else 0.0)
        pairs = [(f"sw{i}", "strong_weak") for i in range(4)]
        s = comparative_metrics(_reward_rows(pairs, reward), "r", n_boot=50)["strong_weak"]["sex"]["merit"]
        assert s["accuracy_strong_protected"]["mean"] == 0.0 and s["accuracy_weak_protected"]["mean"] == 1.0
        # the protected applicant loses under both assignments (strong protected: penalised; weak protected: weaker)
        assert s["accuracy_gap"]["mean"] == -1.0 and s["pref_protected_rate"]["mean"] == 0.0

    def test_ties_count_half(self):
        pairs = [("p", "weak_weak"), ("q", "weak_weak")]
        s = comparative_metrics(_reward_rows(pairs, lambda *a: 0.0), "r", n_boot=20)["weak_weak"]["sex"]["merit"]
        assert s["pref_protected_rate"]["mean"] == 0.5 and s["marker_effect"]["mean"] == 0.0

    def test_templates_average_and_change_is_paired(self):
        reward, _ = _structured()
        rows = _reward_rows(PAIRS, reward, templates=("t1", "t2"))
        for row in rows:
            row["nulled"] = row["r"] - (-0.4 if row["protected"] is not None and row["chosen"] == row["protected"]
                                        else 0.0)
        m = comparative_metrics(rows, "nulled", baseline_key="r", n_boot=100)
        s = m["strong_weak"]["sex"]["merit"]
        assert s["marker_effect"]["mean"] == pytest.approx(0.0, abs=1e-12)
        assert s["marker_effect_change"]["mean"] == pytest.approx(0.4)

    def test_missing_rewards_raise(self):
        reward, _ = _structured()
        rows = _reward_rows(PAIRS[:2], reward)[1:]
        with pytest.raises(ValueError, match="of 8 rewards"):
            pair_values(rows, "r")

    def test_ratio_undefined_without_a_positive_quality_margin(self):
        r = summarize_ratio([1.0, 2.0], [-1.0, -1.0], n_boot=50)
        assert r["mean"] != r["mean"] and r["n_boot_valid"] == 0
        r = summarize_ratio([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], n_boot=50)
        assert r["mean"] == 0.5 and r["ci_low"] == r["ci_high"] == 0.5


# --------------------------------------------------------------------------- directions --------------
def test_pair_contrast_recovers_the_chosen_is_protected_component():
    reward, _ = _structured()
    rows = _reward_rows(PAIRS[:4], reward, axes=("sex", "age"))
    d = 6
    v = torch.zeros(d)
    v[2] = 1.0
    gen = torch.Generator().manual_seed(0)
    pair_noise = {p: torch.randn(d, generator=gen) for p, _ in PAIRS[:4]}
    states = torch.stack([pair_noise[r["pair_id"]] + (v if r["protected"] == r["chosen"] else 0 * v)
                          + (0.5 * v if r["chosen"] == "X" else 0 * v) for r in rows])
    ids, c = pair_contrasts(states, rows, "sex")
    assert ids == sorted(p for p, _ in PAIRS[:4]) and c.shape == (4, d)
    assert torch.allclose(c, v.expand(4, d))
    with pytest.raises(ValueError, match="swapped assignment"):
        # row 1 is (X protected, choose Y), the counterpart of (Y protected, choose Y)
        pair_contrasts(torch.cat([states[:1], states[2:]]), rows[:1] + rows[2:], "sex")
