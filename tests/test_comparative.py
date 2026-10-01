"""
Tests for the comparative (two-applicant) design: `pairs/comparative.py` (frames, responses, pairing, items),
`scoring/comparative_metrics.py` (on synthetic reward tables with known effects) and
`probes/comparative_directions.py`.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from collections import Counter

import numpy as np
import pytest
import torch

from pairs.comparative import (
    COMPARATIVE_FRAMES, KINDS, ORDERS, PAIRINGS, SIDES, UNMARKED, RecordPair, build_pair_items,
    UNUSED, assign_pools, comparative_violations, context, contrast_axes, draw_pairs, pool_split,
)
from pairs.cross_marker import BlockMismatch, block_from_row, proxy_names
from pairs.factorial import CREDIT_DESIGN, DESIGNS, EDUCATION_DESIGN, HIRING_DESIGN
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


def _distinct(block, tag):
    """The fixture records render identical documents; prefix one side's so X's and Y's can be told apart."""
    return dataclasses.replace(block, texts={c: f"{tag} {t}" for c, t in block.texts.items()},
                               unmarked=f"{tag} {block.unmarked}")


# --------------------------------------------------------------------------- frames ------------------
class TestFrames:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_request_and_responses_name_no_attribute(self, domain):
        assert comparative_violations(domain) == {}

    def test_education_is_not_framed_as_one_school(self):
        # the proxy economic-status clause puts the two students in different schools
        assert "school" not in COMPARATIVE_FRAMES["education"].prompt.lower()

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
            assert abs(len(frame.response("merit", i, "A")) - len(frame.response("coded", i, "A"))) <= 15

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
    def test_exact_quotas_from_the_strata(self):
        assert pool_split(100, 100) == {True: pytest.approx(2 / 3), False: pytest.approx(2 / 3)}
        sp = pool_split(564, 239)
        assert sp[False] == pytest.approx(2 / 3) and sp[True] == pytest.approx(2 * 239 / (3 * 564))
        assert pool_split(50, 200)[True] == pytest.approx(2 / 3)
        with pytest.raises(ValueError):
            pool_split(0, 10)
        # credit's strata: exactly round(p·n) to each own pairing, equal strong–weak sides, the strong surplus unused
        strong_of = {f"s{i:03d}": True for i in range(564)} | {f"w{i:03d}": False for i in range(239)}
        sizes = Counter(assign_pools(strong_of, 42).values())
        assert sizes == {"strong_strong": 159, "weak_weak": 159, "strong_weak:strong": 80, "strong_weak:weak": 80,
                         UNUSED: 325}
        assert assign_pools(strong_of, 42) == assign_pools(dict(reversed(list(strong_of.items()))), 42)
        assert assign_pools(strong_of, 42) != assign_pools(strong_of, 43)

    def test_the_pairings_get_equal_capacity(self):
        cand = _candidates(1200, 500)
        _, rep = draw_pairs(cand, {p: 10_000 for p in PAIRINGS}, seed=3)
        assert {p: rep[p]["n"] for p in PAIRINGS} == {"strong_strong": 166, "strong_weak": 167, "weak_weak": 166}
        assert rep["unused"] == 1200 - 333 - 167

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

    def test_another_candidate_set_keeps_the_pairs_before_its_first_change(self):
        # a different probe split removes some records: the per-record order keeps every pair drawn before the
        # first pair that touched a removed record (a pool-wide reshuffle would keep almost none)
        cand = _candidates(300, 300)
        before, _ = draw_pairs(cand, {p: 1000 for p in PAIRINGS}, seed=5)
        removed = set(sorted(cand)[::40])
        after, _ = draw_pairs({k: v for k, v in cand.items() if k not in removed}, {p: 1000 for p in PAIRINGS},
                              seed=5)
        for pairing in PAIRINGS:
            b = [p for p in before if p.pairing == pairing]
            a = [p for p in after if p.pairing == pairing]
            k = next(i for i, p in enumerate(b) if {p.x, p.y} & removed)
            assert k > 0 and a[:k] == b[:k]

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
        stored = {(p.pairing, p.x, p.y): p.index for p in pairs}
        assert all(stored[(c.pairing, c.x, c.y)] == c.index for c in seen if (c.pairing, c.x, c.y) in stored)

    def test_unpaired_anchors_are_counted(self):
        # two match groups of one record each in strong_strong cannot pair
        cand = {"a": (True, 1), "b": (True, 2), "c": (True, 1), "d": (True, 1)}
        _, rep = draw_pairs(cand, {"strong_strong": 5}, seed=0, pools={r: "strong_strong" for r in cand})
        assert rep["strong_strong"]["n"] == 1 and rep["strong_strong"]["unpaired_anchors"] == 2

    def test_unknown_pairing_raises(self):
        with pytest.raises(ValueError, match="unknown pairings"):
            draw_pairs(_candidates(4, 4), {"strong": 2}, seed=1)


# --------------------------------------------------------------------------- items -------------------
class TestItems:
    def test_contexts_rotate_balance_and_skip_axes_without_a_marker(self):
        # credit marital status has no proxy; every axis cycles through its 4 settings of the other axes
        assert contrast_axes(CREDIT_DESIGN, "proxy") == ["sex", "age", "intersection"]
        assert contrast_axes(CREDIT_DESIGN, "explicit") == ["sex", "age", "marital_status", "intersection"]
        assert context(CREDIT_DESIGN, "marital_status", "proxy", 0) is None
        for axis in ("sex", "age", "marital_status"):
            counts = Counter(context(CREDIT_DESIGN, axis, "explicit", i) for i in range(10))
            assert len(counts) == 4 and max(counts.values()) - min(counts.values()) <= 1
            # the same index gives the same context in both encodings (where both have the axis)
            if axis != "marital_status":
                assert all(context(CREDIT_DESIGN, axis, "explicit", i) == context(CREDIT_DESIGN, axis, "proxy", i)
                           for i in range(8))
        a, b = context(CREDIT_DESIGN, "intersection", "explicit", 3)
        assert a == ("female", 30, "married") and b[0] == "male" and a[1:] != b[1:]

    def test_each_side_shows_its_own_document_in_the_right_cell_and_slot(self):
        blocks = _blocks("credit", [("s1", True), ("s2", True)])
        pair = RecordPair("strong_strong", "s1", "s2", 5)
        x, y = _distinct(blocks["s1"]["credit_v1"], "FIRST-RECORD"), _distinct(blocks["s2"]["credit_v1"], "SECOND")
        items = build_pair_items(pair, x, y, "credit", seed=42)
        axes = contrast_axes(CREDIT_DESIGN, "explicit")
        assert len(items) == (len(axes) * 2 + 1) * 2 * len(KINDS) * 2
        assert len({i.paraphrase for i in items}) == 1
        docs = {"X": x, "Y": y}
        for axis in axes:
            prot, ref = context(CREDIT_DESIGN, axis, "explicit", 5)
            for it in (i for i in items if i.axis == axis):
                first, second = it.order[0], it.order[1]
                a = it.prompt.split("Applicant A:\n", 1)[1].split("\n\nApplicant B:\n", 1)
                slot_a, slot_b = a[0], a[1].split("\n\nWhich of the two", 1)[0]
                assert slot_a == docs[first].texts[prot if first == it.protected else ref]
                assert slot_b == docs[second].texts[prot if second == it.protected else ref]
                letter = "A" if it.chosen == first else "B"
                assert it.text == COMPARATIVE_FRAMES["credit"].response(it.kind, it.paraphrase, letter)
        unmarked = [i for i in items if i.axis == UNMARKED]
        assert len(unmarked) == 2 * len(KINDS) * 2 and all(i.protected is None for i in unmarked)
        for it in unmarked:
            first, second = (x, y) if it.order == "XY" else (y, x)
            assert it.prompt.index(first.unmarked) < it.prompt.index(second.unmarked)

    def test_blocks_must_belong_to_the_pair(self):
        blocks = _blocks("credit", [("s1", True), ("s2", True)])
        with pytest.raises(ValueError, match="passed for"):
            build_pair_items(RecordPair("strong_strong", "s1", "s2", 0), blocks["s2"]["credit_v1"],
                             blocks["s1"]["credit_v1"], "credit")
        with pytest.raises(ValueError, match="encoding/template"):
            build_pair_items(RecordPair("strong_strong", "s1", "s2", 0), blocks["s1"]["credit_v1"],
                             blocks["s2"]["credit_v2"], "credit")

    def test_hiring_needs_one_role(self):
        blocks = _blocks("cv", [("a", True), ("b", True)])
        x, y = blocks["a"]["bios_v1"], blocks["b"]["bios_v1"]
        pair = RecordPair("strong_strong", "a", "b", 0)
        x2 = dataclasses.replace(x, real_fields={**x.real_fields, "role": "a nurse"})
        items = build_pair_items(pair, x2, dataclasses.replace(y, real_fields={**y.real_fields, "role": "a nurse"}),
                                 "cv")
        assert all("a position as a nurse" in i.prompt for i in items)
        with pytest.raises(ValueError, match="different roles"):
            build_pair_items(pair, x2, dataclasses.replace(y, real_fields={**y.real_fields, "role": "a pilot"}), "cv")
        with pytest.raises(KeyError, match="role"):
            build_pair_items(pair, dataclasses.replace(x, real_fields={}), y, "cv")

    def test_an_education_proxy_contrast_changes_only_its_axis(self):
        # the name carries sex and ethnicity: a sex contrast keeps both names at one ethnicity
        blocks = _blocks("education", [("e1", True), ("e2", True)], encoding="proxy")
        x, y = (next(iter(blocks[r].values())) for r in ("e1", "e2"))
        for index in range(4):
            prot, ref = context(EDUCATION_DESIGN, "sex", "proxy", index)
            assert prot[1:] == ref[1:] and prot[0] != ref[0]
            items = build_pair_items(RecordPair("strong_strong", "e1", "e2", index), x, y, "education")
            it = next(i for i in items if i.axis == "sex" and i.protected == "X")
            assert x.texts[prot] in it.prompt and y.texts[ref] in it.prompt

    def test_proxy_names_come_from_the_exemplar(self):
        assert proxy_names({"female_name": "Claire", "male_name": "Hunter"}) == {"Claire", "Hunter"}
        assert proxy_names({"names": {"female_white": "A", "male_black": "B"}}) == {"A", "B"}
        assert proxy_names({"design": "x"}) == frozenset()
        proxy = _blocks("education", [("e1", True)], encoding="proxy")["e1"]
        b = next(iter(proxy.values()))
        assert len(b.names) == 4 and all(n in "".join(b.texts.values()) for n in b.names)
        assert EDUCATION_DESIGN.proxy_axes == ("sex", "ethnicity", "economic_status")
        assert HIRING_DESIGN.proxy_axes == ("sex", "age", "family_status")

    def test_a_proxy_block_without_names_is_refused(self):
        row = _cell_rows("credit", [_record("credit", "r", True)], encodings=("proxy",))[0]
        row["exemplar"] = {"design": "credit_factorial_2x2x2"}
        with pytest.raises(BlockMismatch, match="proxy first names"):
            block_from_row(row, CREDIT_DESIGN)


# --------------------------------------------------------------------------- metrics -----------------
def _reward_rows(pairs, reward, axes=("sex",), kinds=KINDS, templates=("t1",)):
    """Rows for ``pairs`` [(pair_id, pairing)] with ``reward(pair, template, axis, protected, order, kind, chosen)``,
    the unmarked prompts included."""
    rows = []
    for (pid, pairing), t in itertools.product(pairs, templates):
        for axis in list(axes) + [UNMARKED]:
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
    def test_marker_effect_position_and_the_unmarked_yardstick(self):
        reward, record = _structured()
        m = comparative_metrics(_reward_rows(PAIRS, reward), "r", n_boot=200, seed=0)
        sw, ss = m["strong_weak"]["sex"], m["strong_strong"]["sex"]
        assert sw["merit"]["marker_effect"]["mean"] == pytest.approx(-0.4)
        assert ss["merit"]["marker_effect"]["mean"] == pytest.approx(-0.4)
        assert sw["coded"]["marker_effect"]["mean"] == pytest.approx(-0.6)
        assert sw["coded_minus_merit"]["mean"] == pytest.approx(-0.2)
        assert sw["merit"]["position_effect"]["mean"] == pytest.approx(0.3)
        # the merit yardstick comes from the unmarked prompts and keeps the records' own appeal
        q = np.mean([1.0 + record[p][0] - record[p][1] for p, g in PAIRS if g == "strong_weak"])
        assert m["strong_weak"][UNMARKED]["quality_margin"]["mean"] == pytest.approx(q)
        assert sw["merit"]["exchange_rate"]["mean"] == pytest.approx(-0.4 / q)
        assert "exchange_rate" not in ss["merit"] and "overturn" not in ss["merit"]
        assert m["all"]["n_pairs"] == len(PAIRS)
        assert m["all"]["sex"]["merit"]["marker_effect"]["mean"] == pytest.approx(-0.4)
        assert m["strong_weak"][UNMARKED]["position_effect"]["mean"] == pytest.approx(0.3)
        assert UNMARKED not in m["all"]

    def test_the_exchange_rate_ignores_a_marker_that_depends_on_who_carries_it(self):
        # the penalty hits only the strong applicant: the marked margin would collapse, the unmarked one does not
        def reward(pid, t, axis, prot, order, kind, chosen):
            return (1.0 if chosen == "X" else 0.0) - (2.0 if prot == "X" and chosen == "X" else 0.0)
        pairs = [(f"sw{i}", "strong_weak") for i in range(4)]
        s = comparative_metrics(_reward_rows(pairs, reward), "r", n_boot=50)["strong_weak"]["sex"]["merit"]
        assert s["marker_effect"]["mean"] == pytest.approx(-1.0)
        assert s["exchange_rate"]["mean"] == pytest.approx(-1.0)

    def test_overturn_and_rescue_against_the_unmarked_accuracy(self):
        # unmarked the strong applicant wins; protected, it loses (overturn 1); a protected weak one changes nothing
        def reward(pid, t, axis, prot, order, kind, chosen):
            return (1.0 if chosen == "X" else 0.0) - (2.0 if prot == "X" and chosen == "X" else 0.0)
        pairs = [(f"sw{i}", "strong_weak") for i in range(4)]
        s = comparative_metrics(_reward_rows(pairs, reward), "r", n_boot=50)["strong_weak"]["sex"]["merit"]
        assert s["accuracy_strong_protected"]["mean"] == 0.0 and s["accuracy_weak_protected"]["mean"] == 1.0
        assert s["overturn"]["mean"] == 1.0 and s["rescue"]["mean"] == 0.0 and s["accuracy_contrast"]["mean"] == 1.0
        # the RM prefers the weak record unmarked; a protected weak applicant hands the strong one the win (rescue),
        # nothing is overturned
        def reward2(pid, t, axis, prot, order, kind, chosen):
            return (-0.2 if chosen == "X" else 0.0) - (0.5 if prot == "Y" and chosen == "Y" else 0.0)
        s = comparative_metrics(_reward_rows(pairs, reward2), "r", n_boot=50)["strong_weak"]["sex"]["merit"]
        assert s["overturn"]["mean"] == 0.0 and s["rescue"]["mean"] == 1.0 and s["accuracy_contrast"]["mean"] == 1.0

    def test_a_marking_effect_moves_overturn_and_rescue_but_not_their_sum(self):
        # any clause costs the strong applicant its win (whoever is protected): overturn +1, rescue −1, contrast 0
        def reward(pid, t, axis, prot, order, kind, chosen):
            return (1.0 if chosen == "X" else 0.0) - (2.0 if prot is not None and chosen == "X" else 0.0)
        pairs = [(f"sw{i}", "strong_weak") for i in range(4)]
        s = comparative_metrics(_reward_rows(pairs, reward), "r", n_boot=50)["strong_weak"]["sex"]["merit"]
        assert (s["overturn"]["mean"], s["rescue"]["mean"], s["accuracy_contrast"]["mean"]) == (1.0, -1.0, 0.0)

    def test_rates_and_levels_are_not_scaled_and_the_pool_has_no_scale(self):
        reward, record = _structured()
        m = comparative_metrics(_reward_rows(PAIRS, reward), "r", n_boot=200)
        ss = m["strong_strong"]["sex"]["merit"]
        me = ss["marker_effect"]
        # the scale is the SD across the group's pairs of the unmarked merit margin r(X) − r(Y)
        margins = [record[p][0] - record[p][1] for p, g in PAIRS if g == "strong_strong"]
        assert me["scale_sd"] == pytest.approx(np.std(margins, ddof=1))
        assert me["ci_low"] == pytest.approx(-0.4) and me["ci_high"] == pytest.approx(-0.4) and me["sd"] < 1e-9
        assert "scaled_mean" not in ss["pref_protected_rate"]
        assert "scaled_mean" not in m["strong_weak"]["sex"]["merit"]["overturn"]
        assert "scaled_mean" not in m["all"]["sex"]["merit"]["marker_effect"]
        assert "scaled_mean" not in m["strong_weak"][UNMARKED]["quality_margin"]

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

    def test_incomplete_or_inconsistent_tables_raise(self):
        reward, _ = _structured()
        rows = _reward_rows(PAIRS[:2], reward)
        with pytest.raises(ValueError, match="of 8 rewards"):
            pair_values(rows[1:], "r")
        with pytest.raises(ValueError, match="duplicate"):
            pair_values(rows + rows[:1], "r")
        other = [dict(r, encoding="proxy") if r["pair_id"] == PAIRS[0][0] else r for r in rows]
        with pytest.raises(ValueError, match="one encoding"):
            pair_values(other, "r")
        lacking = [r for r in rows if not (r["pair_id"] == PAIRS[0][0] and r["kind"] == "coded")]
        with pytest.raises(ValueError, match="lacks"):
            pair_values(lacking, "r")
        with pytest.raises(ValueError, match="unmarked"):
            pair_values([r for r in rows if r["axis"] != UNMARKED], "r")
        # a gap every pair shares: coded missing from the unmarked prompts, or a template from one axis
        with pytest.raises(ValueError, match="lacks"):
            pair_values([r for r in rows if not (r["axis"] == UNMARKED and r["kind"] == "coded")], "r")
        two = _reward_rows(PAIRS[:2], reward, templates=("t1", "t2"))
        with pytest.raises(ValueError, match="lacks"):
            pair_values([r for r in two if not (r["axis"] == UNMARKED and r["template_id"] == "t2")], "r")
        mixed = [dict(r, pairing="weak_weak") if r["axis"] == UNMARKED and r["pair_id"] == PAIRS[0][0] else r
                 for r in rows]
        with pytest.raises(ValueError, match="two pairings"):
            pair_values(mixed, "r")

    def test_the_ratio_interval_needs_a_positive_margin_in_every_replicate(self):
        r = summarize_ratio([1.0, 2.0], [-1.0, -1.0], n_boot=50)
        assert r["mean"] != r["mean"] and r["ci_low"] != r["ci_low"] and r["n_boot_valid"] == 0
        r = summarize_ratio([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], n_boot=50)
        assert r["mean"] == 0.5 and r["ci_low"] == r["ci_high"] == 0.5 and r["n_boot_valid"] == 50
        # a positive mean margin whose replicates reach 0: the estimate stands, the interval is unbounded (NaN)
        r = summarize_ratio([-0.1, -0.2, -0.1, -0.3], [-1.0, 0.5, 0.6, 0.4], n_boot=500)
        assert r["mean"] == r["mean"] and 0 < r["n_boot_valid"] < 500 and r["ci_low"] != r["ci_low"]

    def test_every_statistic_of_a_group_uses_the_same_draws(self):
        # the exchange rate's replicates are the marker effect's draws over the margin's: a constant margin makes
        # its interval the marker effect's interval divided by that margin
        rng = np.random.default_rng(1)
        effect = {p: rng.normal() for p, _ in PAIRS}

        def reward(pid, t, axis, prot, order, kind, chosen):
            return (2.0 if chosen == "X" else 0.0) + (effect[pid] if prot is not None and chosen == prot else 0.0)
        s = comparative_metrics(_reward_rows(PAIRS, reward), "r", n_boot=300, seed=4)["strong_weak"]["sex"]["merit"]
        assert s["exchange_rate"]["ci_low"] == pytest.approx(s["marker_effect"]["ci_low"] / 2)
        assert s["exchange_rate"]["ci_high"] == pytest.approx(s["marker_effect"]["ci_high"] / 2)


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


def test_a_linear_heads_reading_of_the_contrast_is_the_marker_effect():
    # random states per row, a linear head w: w · contrast = the marker effect averaged over the kinds
    rows = _reward_rows(PAIRS, lambda *a: 0.0, templates=("t1", "t2"))
    gen = torch.Generator().manual_seed(2)
    states = torch.randn(len(rows), 8, generator=gen)
    w = torch.randn(8, generator=gen)
    for row, h in zip(rows, states):
        row["r"] = float(h @ w)
    ids, c = pair_contrasts(states, rows, "sex")
    values, _ = pair_values(rows, "r")
    expect = [np.mean([values[("sex", k)][p]["marker_effect"] for k in KINDS]) for p in ids]
    assert np.allclose((c @ w).numpy(), expect, atol=1e-5)


def test_pair_contrast_refuses_incomplete_or_duplicate_rows():
    reward, _ = _structured()
    rows = _reward_rows(PAIRS[:2], reward)
    states = torch.zeros(len(rows), 3)
    keep = lambda drop: ([r for k, r in enumerate(rows) if k != drop], torch.cat([states[:drop], states[drop + 1:]]))
    # row 1 is (X protected, choose Y), the counterpart of (Y protected, choose Y)
    with pytest.raises(ValueError, match="swapped assignment"):
        pair_contrasts(keep(1)[1], keep(1)[0], "sex")
    # row 0 is (X protected, choose X), a "chosen is protected" row: its counterpart is then left over
    with pytest.raises(ValueError, match="lack the swapped"):
        pair_contrasts(keep(0)[1], keep(0)[0], "sex")
    with pytest.raises(ValueError, match="duplicate"):
        pair_contrasts(torch.cat([states, states[:1]]), rows + rows[:1], "sex")
    with pytest.raises(ValueError, match="no rows"):
        pair_contrasts(states, rows, "age")
