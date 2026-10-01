"""
Unit tests for the reasoning-flip 2×2 (correctness × conclusion), in every domain of
`pairs.verdicts.REASONING_FRAMES`: the verdict builder + the factorial metric. No model required (the token checks
skip where a tokenizer is not cached).
"""

from __future__ import annotations

import random

import pytest

from pairs.verdicts import (
    EVAL_PARAPHRASES,
    FIT_PARAPHRASES,
    REASONING_CELLS,
    REASONING_CLAIMS,
    REASONING_FRAMES,
    build_reasoning_item,
    reasoning_cells_varied,
    unmarked_reasoning_prompt,
    verdict_length_report,
)
from scoring.demographic_experiment import compute_reasoning_metrics
from tests.test_decision_response import _domain_case

DOMAINS = tuple(REASONING_FRAMES)
_PREMISES = [(d, p) for d, f in REASONING_FRAMES.items() for p in f.premises]
# the favourable decision of each frame, as its fixed wording names it
_ADVANCE_WORDS = {"cv": "advancing them to an interview", "credit": "approving the loan",
                  "education": "offering them a place"}


def _item(domain, premise, seed=0, **kw):
    rec, render, tid, _ = _domain_case(domain)
    return build_reasoning_item(rec, premise, render, random.Random(seed), template_id=tid, domain=domain, **kw)


def _stems(domain, premise, truth):
    spec = REASONING_FRAMES[domain].premises[premise]
    subject = spec.subject[0].upper() + spec.subject[1:]
    return [p.format(S=subject) for p in REASONING_CLAIMS[spec.claim][truth]]


class TestBuilder:
    @pytest.mark.parametrize("domain,premise", _PREMISES)
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
        assert [p for p, s in frame.premises.items() if not s.demographic] == [frame.control]
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
        assert claims["credit"] == {"age": "credit_history", "intersection": "credit_history", "sabbatical": "income"}
        assert (REASONING_FRAMES["cv"].control, REASONING_FRAMES["education"].control) == ("abroad", "out_of_district")

    def test_a_true_stem_and_its_false_stem_differ_by_one_antonym(self):
        # so the correctness contrast is one word; hiring's availability pool (from before the port) differs more
        swaps = {("fewer", "more"), ("less", "more"), ("shorter", "longer"), ("reduce", "increase"),
                 ("lower", "raise"), ("harder", "easier"), ("heavier", "lighter"), ("more", "less")}
        for claim in ("credit_history", "income", "afford"):
            for t, f in zip(REASONING_CLAIMS[claim]["true"], REASONING_CLAIMS[claim]["false"]):
                diff = [(a, b) for a, b in zip(t.split(), f.split()) if a != b]
                assert len(t.split()) == len(f.split()) and len(diff) == 1 and diff[0] in swaps, (claim, t, f)

    @pytest.mark.parametrize("domain,premise", _PREMISES)
    def test_cells_length_matched(self, domain, premise):
        # the fixed wording within 3 words; the paraphrases within 5 (hiring's pools, from before the port, within 9)
        rep = verdict_length_report(_item(domain, premise)["cells"], keys=REASONING_CELLS)
        assert rep["max_word_delta"] <= 3, (domain, premise, rep)
        for seed in range(40):
            rep = verdict_length_report(_item(domain, premise, seed=seed, vary=True)["cells"], keys=REASONING_CELLS)
            assert rep["max_word_delta"] <= (9 if domain == "cv" else 5), (domain, premise, rep)

    @pytest.mark.parametrize("model", ["Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                                       "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
                                       "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"])
    def test_credit_and_education_cells_are_token_matched(self, model):
        # the fixed cells within 2 tokens, every decision phrase within 2 of the others, and a true stem the
        # length of the same entry's false stem. Hiring's fixed cells span 4-5 (its true stem says "near-term").
        transformers = pytest.importorskip("transformers")
        try:
            tok = transformers.AutoTokenizer.from_pretrained(model, local_files_only=True)
        except (OSError, ValueError):
            pytest.skip(f"{model} tokenizer not cached")
        n = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
        for domain in ("credit", "education"):
            frame = REASONING_FRAMES[domain]
            phrases = [n(x) for x in frame.advance_conn + frame.reject_conn + frame.advance_balance
                       + frame.reject_balance]
            assert max(phrases) - min(phrases) <= 2, (model, domain)
            for premise in frame.premises:
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
                item = build_reasoning_item(rec, premise, render, random.Random(0), template_id=tid, domain=domain)
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
        frame = REASONING_FRAMES[domain]
        cells = reasoning_cells_varied(premise, random.Random(seed), domain=domain)
        # the stem a cell starts with (a subject may itself contain commas: "Black, female, and …")
        stems = _stems(domain, premise, "true") + _stems(domain, premise, "false")
        true_stem = next(st for st in stems if cells["true_reject"].startswith(st + ", "))
        false_stem = next(st for st in stems if cells["false_advance"].startswith(st + ", "))
        # the two TRUE cells share the true stem; the two FALSE cells share the false stem
        assert cells["true_advance"].startswith(true_stem)
        assert cells["false_reject"].startswith(false_stem)
        # truth value preserved: true stem ∈ true pool, false stem ∈ false pool
        assert true_stem in _stems(domain, premise, "true") and true_stem not in _stems(domain, premise, "false")
        assert false_stem in _stems(domain, premise, "false")
        # conclusion preserved: each cell's connective comes from the right (advance/reject) pool
        conn = lambda cell, stem: cell[len(stem) + 2:-1]  # strip ", " prefix and trailing "."
        assert conn(cells["true_reject"], true_stem) in frame.reject_conn
        assert conn(cells["true_advance"], true_stem) in frame.advance_balance
        assert conn(cells["false_advance"], false_stem) in frame.advance_conn
        assert conn(cells["false_reject"], false_stem) in frame.reject_balance

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_without_a_connective_the_decision_is_its_own_sentence(self, domain, seed):
        # correctness and conclusion are then independent factors: no word encodes their combination
        frame = REASONING_FRAMES[domain]
        cells = reasoning_cells_varied(frame.control, random.Random(seed), connective=False, domain=domain)
        for cell, text in cells.items():
            stem, decision = text[:-1].split(". ")
            assert stem in _stems(domain, frame.control, "true" if cell.startswith("true_") else "false")
            assert decision in (frame.advance_decision if cell.endswith("_advance") else frame.reject_decision)
            assert not {"so", "but"} & set(text.lower().replace(".", "").replace(",", "").split())
        assert all(c.startswith("so ") for c in frame.advance_conn + frame.reject_conn)
        assert all(c.startswith("but ") for c in frame.advance_balance + frame.reject_balance)
        with pytest.raises(ValueError, match="vary=True"):
            _item(domain, frame.control, connective=False)

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_no_fitted_decision_phrase_contains_a_held_out_one(self, domain):
        # an eval phrase without its connective ("I would approve the loan") inside a fitted one ("but on balance
        # I would approve the loan") would make the held-out wording partly seen
        frame = REASONING_FRAMES[domain]
        pools = (frame.advance_conn, frame.reject_conn, frame.advance_balance, frame.reject_balance)
        core = lambda phrase: phrase.split(" ", 1)[1]                    # without "so" / "but"
        fitted = [p[k] for p in pools for k in FIT_PARAPHRASES]
        for p in pools:
            for k in EVAL_PARAPHRASES:
                assert not any(core(p[k]) in f for f in fitted), (domain, p[k])

    def test_the_hiring_decision_sentences(self):
        assert REASONING_FRAMES["cv"].advance_decision[0] == "I recommend advancing them to an interview"

    @pytest.mark.parametrize("domain,premise", [(d, p) for d, f in REASONING_FRAMES.items()
                                                for p in (f.primary, f.control)])
    def test_paraphrases_restrict_every_pool(self, domain, premise):
        # the reasoning probe and erasure fit on some entries and evaluate on the others: no evaluated string is a
        # fitted one (both premises they use)
        frame = REASONING_FRAMES[domain]
        assert not set(FIT_PARAPHRASES) & set(EVAL_PARAPHRASES)

        def cells(paraphrases, connective=True, seeds=range(40)):
            return {v for s in seeds for v in _item(domain, premise, seed=s, vary=True, paraphrases=paraphrases,
                                                     connective=connective)["cells"].values()}

        for connective, pools in ((True, [frame.advance_conn, frame.reject_conn, frame.advance_balance,
                                          frame.reject_balance]),
                                  (False, [frame.advance_decision, frame.reject_decision])):
            fitted, held_out = cells(FIT_PARAPHRASES, connective), cells(EVAL_PARAPHRASES, connective)
            assert len(held_out) == 4 and not fitted & held_out
            for pool in [_stems(domain, premise, "true"), _stems(domain, premise, "false"), *pools]:
                for k in EVAL_PARAPHRASES:  # an eval entry of every pool appears only in the held-out wording
                    assert any(pool[k] in v for v in held_out) and not any(pool[k] in v for v in fitted)
        with pytest.raises(IndexError):
            cells((3,), seeds=[0])
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

    def test_handles_none(self):
        m = compute_reasoning_metrics({
            "true_reject": [5.0, None], "true_advance": [5.0, 5.0],
            "false_advance": [1.0, 1.0], "false_reject": [1.0, 1.0],
        })
        assert m["n"] == 1  # only one aligned true_reject/false_advance pair
