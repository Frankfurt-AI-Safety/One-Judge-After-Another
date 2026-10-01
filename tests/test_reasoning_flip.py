"""
Unit tests for the reasoning-flip 2×2 (correctness × conclusion), in every domain of
`pairs.verdicts.REASONING_FRAMES`: the verdict builder + the factorial metric. No model required (the token checks
skip where a tokenizer is not cached).
"""

from __future__ import annotations

import random

import pytest

from pairs.verdicts import (
    PARAPHRASE_FOLDS,
    REASONING_CELLS,
    REASONING_CLAIMS,
    REASONING_FRAMES,
    build_reasoning_item,
    claim_is_favourable,
    reasoning_cells_varied,
    unmarked_reasoning_prompt,
    verdict_length_report,
)
from scoring.demographic_experiment import compute_reasoning_metrics
from tests.test_decision_response import _domain_case

DOMAINS = tuple(REASONING_FRAMES)
_PREMISES = [(d, p) for d, f in REASONING_FRAMES.items() for p in f.premises]
# the flip's premises (a true claim argues for rejecting), which also have the fixed (connective) wording
_FLIP = [(d, p) for d, f in REASONING_FRAMES.items() for p in f.flip_premises]
# the premises of the reasoning probe, transfer and erasure test
_PROBE = [(d, p) for d, f in REASONING_FRAMES.items() for p in (f.primary, f.control, f.favourable)]
# the favourable decision of each frame, as its fixed wording names it
_ADVANCE_WORDS = {"cv": "advancing them to an interview", "credit": "approving the loan",
                  "education": "offering them a place"}


def _item(domain, premise, seed=0, **kw):
    rec, render, tid, _ = _domain_case(domain)
    return build_reasoning_item(rec, premise, render, random.Random(seed), template_id=tid, domain=domain, **kw)


def _stems(domain, premise, truth):
    """The premise's ``truth`` ("true"/"false") stems: from the favourable pool where that claim is the true one."""
    spec = REASONING_FRAMES[domain].premises[premise]
    subject = spec.subject[0].upper() + spec.subject[1:]
    favourable = (truth == "true") == spec.favourable_true
    return [p.format(S=subject) for p in REASONING_CLAIMS[spec.claim]["favourable" if favourable else "unfavourable"]]


class TestBuilder:
    @pytest.mark.parametrize("domain,premise", _FLIP)
    def test_cells_and_premise_clause(self, domain, premise):
        frame, item = REASONING_FRAMES[domain], _item(domain, premise)
        assert set(item["cells"]) == set(REASONING_CELLS)
        assert item["meta"]["premise_clause"] in item["user_prompt"]
        assert item["meta"]["domain"] == domain and item["meta"]["null_axis"] == frame.premises[premise].axis
        # the claim direction is encoded in the text: the fixed wording is entry 0 of the claim's pools
        true_stem, false_stem = _stems(domain, premise, "true")[0], _stems(domain, premise, "false")[0]
        for cell, text in item["cells"].items():
            assert text == f"{true_stem if cell.startswith('true_') else false_stem}, {frame.fixed[cell]}."
        # conclusions: the favourable decision in the advance cells only
        assert _ADVANCE_WORDS[domain] in item["cells"]["false_advance"]
        assert " not " in item["cells"]["true_reject"] and " not " in item["cells"]["false_reject"]
        assert " not " not in item["cells"]["false_advance"] + item["cells"]["true_advance"]

    def test_intersection_premises_are_the_factorial_corners(self):
        # the protected (pole-A) corner of each factorial, explicit, about the subject the generators write
        from pairs.factorial import DESIGNS

        expected = {"cv": " The applicant is a 30-year-old woman currently on parental leave.",
                    "credit": " The applicant is a 30-year-old married woman.",
                    "education": " The student is Black, female, and from a low-income household."}
        subjects = {"cv": "applicant", "credit": "applicant", "education": "student"}   # generate_education: student
        for domain, frame in REASONING_FRAMES.items():
            design = DESIGNS[domain]
            corner = design.axis_pairs("intersection", "explicit")[0][0]
            assert frame.premises["intersection"].clause == expected[domain] == \
                design.clause(corner, "explicit", subject=subjects[domain])
            assert frame.premises["intersection"].axis == "intersection"

    def test_each_demographic_premise_states_its_axis_pole(self):
        # the pole's wording is in the premise and in the factorial's corner clause, so the axis's direct
        # direction (fitted on the factorial's pairs) is about what the premise says
        from pairs.factorial import DESIGNS
        from pairs.markers import AGE_YOUNG

        pole = {("cv", "parental_leave"): "currently on parental leave", ("credit", "age"): f"a {AGE_YOUNG}-year-old",
                ("education", "low_income"): "from a low-income household"}
        for (domain, premise), words in pole.items():
            spec = REASONING_FRAMES[domain].premises[premise]
            assert spec.axis in DESIGNS[domain].axes and spec.demographic
            assert words in spec.clause and words in REASONING_FRAMES[domain].premises["intersection"].clause
            assert REASONING_FRAMES[domain].primary == premise

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_the_control_is_non_demographic(self, domain):
        from pairs.cross_marker import _words, attribute_words

        frame = REASONING_FRAMES[domain]
        spec = frame.premises[frame.control]
        assert spec.axis is None and not spec.demographic
        assert not _words(f"{spec.clause} {spec.subject}") & attribute_words(domain)
        assert [p for p, s in frame.premises.items() if not s.demographic] == [frame.control, frame.favourable]
        # every demographic premise names an attribute (the same lexicon sees them)
        for p, s in frame.premises.items():
            if s.demographic:
                assert _words(f"{s.clause} {s.subject}") & attribute_words(domain), (domain, p)

    def test_the_controls_claims(self):
        # hiring and education: the control makes the demographic premises' claim (a matched predicate); credit's
        # makes another one (income vs credit history; working notes, 2026-10-01)
        claims = {d: {p: s.claim for p, s in f.premises.items()} for d, f in REASONING_FRAMES.items()}
        assert set(claims["cv"].values()) == {"availability"}
        assert set(claims["education"].values()) == {"afford"}
        assert claims["credit"] == {"age": "credit_history", "intersection": "credit_history", "sabbatical": "income",
                                    "pay_raise": "income"}
        assert (REASONING_FRAMES["cv"].control, REASONING_FRAMES["education"].control) == ("abroad", "out_of_district")

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_the_favourable_truth_premise_crosses_truth_and_valence(self, domain):
        # it shares the control's claim and makes the other stem true: the control's true claim is unfavourable, its
        # true claim favourable, so a direction of valence and one of correctness come apart on it
        from pairs.cross_marker import _words, attribute_words

        frame = REASONING_FRAMES[domain]
        fav, ctl = frame.premises[frame.favourable], frame.premises[frame.control]
        assert fav.favourable_true and not ctl.favourable_true and fav.claim == ctl.claim
        assert not fav.demographic and not _words(f"{fav.clause} {fav.subject}") & attribute_words(domain)
        assert frame.favourable not in frame.flip_premises and frame.control in frame.flip_premises
        assert [p for p, s in frame.premises.items() if s.favourable_true] == [frame.favourable]
        assert _stems(domain, frame.favourable, "true") == \
            [x.format(S=fav.subject[0].upper() + fav.subject[1:]) for x in REASONING_CLAIMS[fav.claim]["favourable"]]
        cells = _item(domain, frame.favourable, vary=True)["cells"]
        true_stem = cells["true_advance"].rsplit(". ", 1)[0]
        assert true_stem in _stems(domain, frame.favourable, "true")
        assert true_stem not in _stems(domain, frame.favourable, "false")
        # the valence label: the favourable claim's cells
        assert [c for c in REASONING_CELLS if claim_is_favourable(c, True)] == ["true_reject", "true_advance"]
        assert [c for c in REASONING_CELLS if claim_is_favourable(c, False)] == ["false_advance", "false_reject"]
        # it has no fixed (connective) wording: its true claim argues for advancing
        with pytest.raises(ValueError, match="favourable-truth"):
            _item(domain, frame.favourable)

    def test_each_entry_differs_in_one_antonym_of_its_own(self):
        # so the correctness contrast is one word, and no word that separates true from false at one entry recurs at
        # another entry of any claim, in any domain (a fold fits or holds out an entry index in every claim alike, so
        # the same entry may share a word): held-out wording cannot ride on a fitted word, within a domain or across;
        # nor does a separating word occur in a subject, clause or decision sentence
        import re

        words = lambda s: set(re.findall(r"[a-z']+", s.lower()))
        by_entry = [set() for _ in range(6)]
        for claim, pools in REASONING_CLAIMS.items():
            for k, (u, f) in enumerate(zip(pools["unfavourable"], pools["favourable"])):
                diff = [(a, b) for a, b in zip(u.split(), f.split()) if a != b]
                assert len(u.split()) == len(f.split()) and len(diff) == 1, (claim, u, f)
                by_entry[k] |= words(u) ^ words(f)
        for i in range(6):
            for j in range(i + 1, 6):
                assert not by_entry[i] & by_entry[j], (i, j, by_entry[i] & by_entry[j])
        other = set()
        for frame in REASONING_FRAMES.values():
            for spec in frame.premises.values():
                other |= words(spec.clause) | words(spec.subject)
            for d in frame.advance_decision + frame.reject_decision:
                other |= words(d)
        assert not set().union(*by_entry) & other, set().union(*by_entry) & other

    @pytest.mark.parametrize("domain,premise", _PREMISES)
    def test_cells_length_matched(self, domain, premise):
        # the fixed wording within 3 words; the varied cells' stems equal in words entry by entry, the decisions of an
        # entry within 2 words (cells draw entries independently, so a pair of cells may differ by more)
        if premise in REASONING_FRAMES[domain].flip_premises:
            rep = verdict_length_report(_item(domain, premise)["cells"], keys=REASONING_CELLS)
            assert rep["max_word_delta"] <= 3, (domain, premise, rep)
        frame = REASONING_FRAMES[domain]
        for a, r in zip(frame.advance_decision, frame.reject_decision):
            assert abs(len(a.split()) - len(r.split())) <= 2, (domain, a, r)

    @pytest.mark.parametrize("model", ["Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                                       "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
                                       "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"])
    def test_cells_are_token_matched(self, model):
        # in every domain a true stem is the length of the same entry's false stem, a decision within 2 tokens of the
        # other decision of its entry, and the fixed cells within 2 tokens of each other
        transformers = pytest.importorskip("transformers")
        try:
            tok = transformers.AutoTokenizer.from_pretrained(model, local_files_only=True)
        except (OSError, ValueError):
            pytest.skip(f"{model} tokenizer not cached")
        n = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
        for domain, frame in REASONING_FRAMES.items():
            for a, r in zip(frame.advance_decision, frame.reject_decision):
                assert abs(n(a) - n(r)) <= 2, (model, domain, a, r)
            for premise in frame.premises:
                if premise in frame.flip_premises:
                    fixed = [n(v) for v in _item(domain, premise)["cells"].values()]
                    assert max(fixed) - min(fixed) <= 2, (model, domain, premise)
                for t, f in zip(_stems(domain, premise, "true"), _stems(domain, premise, "false")):
                    assert n(t) == n(f), (model, domain, premise, t)

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_deterministic(self, domain):
        assert _item(domain, "intersection", seed=3, vary=True) == _item(domain, "intersection", seed=3, vary=True)

    def test_unknown_premise_or_domain_raises(self):
        with pytest.raises(ValueError, match="premise"):
            _item("cv", "nope")
        with pytest.raises(ValueError, match="premise"):
            _item("cv", "age")                  # another domain's premise
        with pytest.raises(ValueError, match="no frame"):
            build_reasoning_item(_domain_case("cv")[0], "parental_leave", lambda r, marker="": "x",
                                 random.Random(0), domain="nope")

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_unmarked_prompt_is_the_item_prompt_without_the_premise(self, domain):
        # gate_fixed rescoring takes the gate of this prompt (education's reasoning prompt is not its decision prompt)
        from substrates.domains import get_domain

        rec, render, _, _ = _domain_case(domain)
        for tid in get_domain(domain).template_ids:
            for premise, spec in REASONING_FRAMES[domain].premises.items():
                item = build_reasoning_item(rec, premise, render, random.Random(0), template_id=tid, domain=domain,
                                            vary=True)
                assert item["user_prompt"].count(spec.clause) == 1
                assert unmarked_reasoning_prompt(rec, render, tid, domain) == \
                    item["user_prompt"].replace(spec.clause, "")

    def test_prompts(self):
        from pairs.verdicts import DECISION_FRAMES, DECISION_PROMPT

        assert REASONING_FRAMES["cv"].prompt == DECISION_PROMPT == DECISION_FRAMES["cv"].prompt
        assert REASONING_FRAMES["credit"].prompt == DECISION_FRAMES["credit"].prompt
        # education: a place in a fee-charging program, not the pass/fail grading of the decision arm
        edu = REASONING_FRAMES["education"].prompt
        assert "fee" in edu and "pass" not in edu and edu != DECISION_FRAMES["education"].prompt

    def test_only_credit_drops_records(self):
        # an unpaid sabbatical contradicts a record whose employment or job reads unemployed: credit draws none, for
        # every premise alike
        import dataclasses

        from substrates.credit_ingest import EMPLOYMENT, JOB

        frame = REASONING_FRAMES["credit"]
        rec = _domain_case("credit")[0]
        assert frame.is_eligible(rec) and rec.employment_since != EMPLOYMENT["A71"] and rec.job != JOB["A171"]
        assert not frame.is_eligible(dataclasses.replace(rec, employment_since="none (unemployed)"))
        assert not frame.is_eligible(dataclasses.replace(rec, job="unemployed or unskilled"))
        assert frame.is_eligible(dataclasses.replace(rec, job="unskilled"))
        assert all(REASONING_FRAMES[d].eligible is None for d in ("cv", "education"))
        assert all(REASONING_FRAMES[d].is_eligible(_domain_case(d)[0]) for d in ("cv", "education"))


class TestVariedVerdicts:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("domain,premise", _PREMISES)
    def test_paraphrase_preserves_truth_and_conclusion(self, domain, premise, seed):
        # the decision is a sentence of its own: no "so"/"but" (with them, correctness = XOR(connective, decision))
        frame = REASONING_FRAMES[domain]
        cells = reasoning_cells_varied(premise, random.Random(seed), domain=domain)
        for cell, text in cells.items():
            stem, decision = text[:-1].rsplit(". ", 1)
            assert stem in _stems(domain, premise, "true" if cell.startswith("true_") else "false")
            assert decision in (frame.advance_decision if cell.endswith("_advance") else frame.reject_decision)
            assert not {"so", "but"} & set(text.lower().replace(".", "").replace(",", "").split())
        # the two true cells share one stem, the two false cells another
        stem = lambda c: cells[c].rsplit(". ", 1)[0]
        assert stem("true_reject") == stem("true_advance") and stem("false_advance") == stem("false_reject")

    def test_the_folds_hold_every_entry_out_once(self):
        held = [k for _, h in PARAPHRASE_FOLDS for k in h]
        assert sorted(held) == list(range(6)) and all(len(h) == 2 for _, h in PARAPHRASE_FOLDS)
        assert all(not set(fit) & set(h) and len(fit) == 4 for fit, h in PARAPHRASE_FOLDS)
        assert {len(pool) for f in REASONING_FRAMES.values() for pool in (f.advance_decision, f.reject_decision)} \
            == {6}
        assert {len(pool) for c in REASONING_CLAIMS.values() for pool in c.values()} == {6}

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_no_fitted_decision_sentence_contains_a_held_out_one(self, domain):
        # in a fold, a held-out decision inside a fitted one (or the reverse) would make the held-out wording partly
        # seen
        frame = REASONING_FRAMES[domain]
        pools = (frame.advance_decision, frame.reject_decision)
        for fit, held in PARAPHRASE_FOLDS:
            fitted = [p[k] for p in pools for k in fit]
            for p in pools:
                for k in held:
                    assert not any(p[k] in f or f in p[k] for f in fitted), (domain, p[k])

    def test_the_hiring_decision_sentences(self):
        assert REASONING_FRAMES["cv"].advance_decision[0] == "I recommend advancing them to an interview"

    @pytest.mark.parametrize("domain,premise", _PROBE)
    def test_paraphrases_restrict_every_pool(self, domain, premise):
        # the reasoning probe, transfer and erasure fit on four entries and evaluate on the other two, in every fold:
        # no evaluated string is a fitted one, and the held-out wording varies within the fold
        frame = REASONING_FRAMES[domain]

        def cells(paraphrases, seeds=range(80)):
            return {v for s in seeds for v in _item(domain, premise, seed=s, vary=True,
                                                     paraphrases=paraphrases)["cells"].values()}

        pools = [_stems(domain, premise, "true"), _stems(domain, premise, "false"), frame.advance_decision,
                 frame.reject_decision]
        for fit, held in PARAPHRASE_FOLDS:
            fitted, held_out = cells(fit), cells(held)
            assert len(held_out) == 4 * 2 * 2 and not fitted & held_out        # 2 stems × 2 decisions per cell
            for pool in pools:
                for k in held:  # a held-out entry of every pool appears only in the held-out wording
                    assert any(pool[k] in v for v in held_out) and not any(pool[k] in v for v in fitted)
        with pytest.raises(IndexError):
            cells((6,), seeds=[0])
        with pytest.raises(ValueError, match="vary=True"):
            _item(domain, premise, paraphrases=(0,))


class TestMetric:
    def test_factorial_effects(self):
        # TRUE-claim cells high, FALSE-claim cells low → positive correctness effect, zero conclusion effect.
        scores = {
            "true_reject":  [10.0, 10.0],
            "true_advance": [10.0, 10.0],
            "false_advance": [0.0, 0.0],
            "false_reject":  [0.0, 0.0],
        }
        m = compute_reasoning_metrics(scores)
        assert m["n"] == 2
        assert m["correctness_effect"] == pytest.approx(10.0)
        assert m["conclusion_effect"] == pytest.approx(0.0)
        assert m["interaction"] == pytest.approx(0.0)
        assert m["prefers_correct_over_favorable_rate"] == pytest.approx(1.0)  # true_reject(10) > false_advance(0)
        assert m["gap_correct_minus_favorable"] == pytest.approx(10.0)

    def test_conclusion_effect(self):
        # advance cells high, reject low → positive conclusion effect, zero correctness effect.
        m = compute_reasoning_metrics({
            "true_reject":  [0.0, 0.0],
            "true_advance": [8.0, 8.0],
            "false_advance": [8.0, 8.0],
            "false_reject":  [0.0, 0.0],
        })
        assert m["conclusion_effect"] == pytest.approx(8.0)
        assert m["correctness_effect"] == pytest.approx(0.0)
        # headline: true_reject(0) vs false_advance(8) → RM prefers the wrong-favorable verdict
        assert m["prefers_correct_over_favorable_rate"] == pytest.approx(0.0)

    def test_the_favourable_truth_premise_reads_interaction_as_coherence(self):
        # its coherent cells are true_advance and false_reject: rewarding them is interaction > 0 there too, and the
        # headline pair (correct-but-harmful vs wrong-but-favourable) does not exist
        coherent = {"true_reject": [0.0], "true_advance": [4.0], "false_advance": [0.0], "false_reject": [4.0]}
        assert compute_reasoning_metrics(coherent, favourable_true=True)["interaction"] == pytest.approx(4.0)
        assert compute_reasoning_metrics(coherent)["interaction"] == pytest.approx(-4.0)
        m = compute_reasoning_metrics(coherent, favourable_true=True)
        assert "prefers_correct_over_favorable_rate" not in m and "gap_correct_minus_favorable" not in m

    def test_handles_none(self):
        m = compute_reasoning_metrics({
            "true_reject": [5.0, None], "true_advance": [5.0, 5.0],
            "false_advance": [1.0, 1.0], "false_reject": [1.0, 1.0],
        })
        assert m["n"] == 1  # only one aligned true_reject/false_advance pair


class TestFoldIntervals:
    def test_a_record_is_one_cluster_with_its_items_of_every_fold(self):
        from scoring.demographic_experiment import reasoning_intervals

        cells = ("true_reject", "true_advance", "false_advance", "false_reject")
        fold = lambda k: {c: [float(k + i) for i in range(5)] for c in cells}          # 5 records
        base = [fold(0), fold(1), fold(2)]
        null = [{c: [x + (1.0 if c.startswith("true") else 0.0) for x in f[c]] for c in cells} for f in base]
        iv = reasoning_intervals(base, null, n_boot=50)
        b = iv["baseline"]["correctness_effect"]
        assert (b["n_clusters"], b["n_items"]) == (5, 15)                              # records, not folds × records
        change = iv["nulled_minus_baseline"]["correctness_effect"]
        assert change["estimate"] == pytest.approx(1.0) and change["ci_low"] == pytest.approx(change["ci_high"])
        with pytest.raises(ValueError, match="same folds"):
            reasoning_intervals(base, null[:2], n_boot=10)
