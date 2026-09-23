"""
Tests for the shared demographic-domain registry (`domains.py`) and the cross-influence pairing,
which guard the credit→domain-dispatch refactor of the battery / cross-influence / additivity runners.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from substrates.domains import DOMAINS, get_domain
from substrates.credit_render import TEMPLATES
from substrates.bios_render import BIOS_TEMPLATES
from substrates.education_render import EDU_TEMPLATES


class TestRegistry:
    def test_known_domains(self):
        assert set(DOMAINS) == {"credit", "cv", "education"}
        with pytest.raises(ValueError):
            get_domain("nope")

    def test_template_ids_match_renderers(self):
        assert get_domain("credit").template_ids == tuple(sorted(TEMPLATES))
        assert get_domain("cv").template_ids == tuple(sorted(BIOS_TEMPLATES))
        assert get_domain("education").template_ids == tuple(sorted(EDU_TEMPLATES))

    def test_is_strong_reads_right_field(self):
        cv = get_domain("cv")
        assert cv.is_strong(SimpleNamespace(qualified=True)) is True
        assert cv.is_strong(SimpleNamespace(qualified=False)) is False
        credit = get_domain("credit")
        assert credit.is_strong(SimpleNamespace(credit_good=True)) is True
        assert credit.is_strong(SimpleNamespace(credit_good=False)) is False
        edu = get_domain("education")
        assert edu.is_strong(SimpleNamespace(high_quality=True)) is True
        assert edu.is_strong(SimpleNamespace(high_quality=False)) is False

    def test_load_records_nonempty(self):
        credit_recs = get_domain("credit").load_records()  # uses the downloaded raw file
        # 1000 records minus the credit_clean rules (49 contradictory, 149 with 3+ dependents)
        assert len(credit_recs) == 803 and hasattr(credit_recs[0], "credit_good")

    def test_cv_load_records_needs_corpus(self):
        # The hiring arm now loads real biographies (Bias-in-Bios), user-downloaded like the
        # education corpus — skip gracefully if it isn't present locally rather than failing.
        try:
            recs = get_domain("cv").load_records()
        except FileNotFoundError:
            pytest.skip("Bias-in-Bios corpus not downloaded (data/demographic/cv/raw/)")
        assert len(recs) > 0
        # These attribute names are load-bearing: run_reasoning_*.py filter on
        # getattr(r, "qualified", True) and verdicts.py reads .role — both fail silently if renamed.
        assert hasattr(recs[0], "qualified") and hasattr(recs[0], "role")

    def test_education_load_records_needs_corpus(self):
        # Education loads a user-downloaded corpus; skip gracefully if it isn't present locally.
        try:
            recs = get_domain("education").load_records()
        except FileNotFoundError:
            pytest.skip("PERSUADE corpus not downloaded (data/demographic/education/raw/)")
        assert len(recs) > 0 and hasattr(recs[0], "high_quality")


class TestBuildPairs:
    def test_pairs_strong_with_weak_and_respects_n(self):
        from runners.run_crossinfluence import _build_pairs

        recs = ([SimpleNamespace(source_record_id=f"s{i}", flag=True) for i in range(5)]
                + [SimpleNamespace(source_record_id=f"w{i}", flag=False) for i in range(3)])
        is_strong = lambda r: r.flag
        pairs = _build_pairs(recs, n_pairs=10, seed=0, is_strong=is_strong,
                             template_ids=("t1", "t2"))
        assert len(pairs) == 3  # min(10, 5 strong, 3 weak)
        for strong, weak, tid in pairs:
            assert is_strong(strong) and not is_strong(weak)
            assert tid in ("t1", "t2")
        assert [p[2] for p in pairs] == ["t1", "t2", "t1"]  # template cycling

    def test_deterministic(self):
        from runners.run_crossinfluence import _build_pairs

        recs = ([SimpleNamespace(source_record_id=f"s{i}", flag=True) for i in range(6)]
                + [SimpleNamespace(source_record_id=f"w{i}", flag=False) for i in range(6)])
        kw = dict(n_pairs=4, seed=7, is_strong=lambda r: r.flag, template_ids=("t1", "t2"))
        a = _build_pairs(recs, **kw)
        b = _build_pairs(recs, **kw)
        assert [(s.source_record_id, w.source_record_id, t) for s, w, t in a] == \
               [(s.source_record_id, w.source_record_id, t) for s, w, t in b]


class TestLengthMatchedPairs:
    """Cross-influence pairs must not let text length alone reproduce the quality label (audit
    2026-09-23: education length AUC 0.98, credit 0.72)."""

    @staticmethod
    def _recs(strong_lens, weak_lens, stratum=None):
        s = [SimpleNamespace(source_record_id=f"s{i}", flag=True, n=n, k=(stratum or [None] * 999)[i])
             for i, n in enumerate(strong_lens)]
        w = [SimpleNamespace(source_record_id=f"w{i}", flag=False, n=n,
                             k=(stratum or [None] * 999)[len(strong_lens) + i])
             for i, n in enumerate(weak_lens)]
        return s + w

    @staticmethod
    def _match(recs, n_pairs=100, **kw):
        from runners.run_crossinfluence import _build_length_matched_pairs

        return _build_length_matched_pairs(recs, n_pairs, 0, lambda r: r.flag, ("t1", "t2"),
                                           lambda r, tid: r.n, **kw)

    def test_direction_alternates_strictly_so_length_alone_scores_one_half(self):
        # strong essays are systematically longer, as in education
        recs = self._recs(range(100, 200, 2), range(60, 160, 2))
        pairs = self._match(recs, caliper=0.2)
        assert len(pairs) >= 20 and len(pairs) % 2 == 0
        for k, (s, w, _) in enumerate(pairs):
            assert (w.n > s.n) if k % 2 == 0 else (w.n < s.n)  # strict, never a tie
        longer = sum(s.n > w.n for s, w, _ in pairs)
        assert longer == len(pairs) // 2

    def test_random_pairing_of_the_same_pool_is_length_confounded(self):
        # the problem the matching removes: unmatched pairs let "prefer longer" look like quality
        from runners.run_crossinfluence import _build_pairs

        recs = self._recs(range(100, 200, 2), range(60, 160, 2))
        pairs = _build_pairs(recs, 50, 0, lambda r: r.flag, ("t1", "t2"))
        assert sum(s.n > w.n for s, w, _ in pairs) / len(pairs) > 0.75

    def test_exact_ties_are_never_paired(self):
        recs = self._recs([100] * 10, [100] * 10 + [105] * 5 + [95] * 5)
        pairs = self._match(recs, caliper=0.2)
        assert pairs and all(s.n != w.n for s, w, _ in pairs)

    def test_caliper_is_respected(self):
        recs = self._recs([100, 100, 100, 100], [150, 60, 108, 93])
        pairs = self._match(recs, caliper=0.10)
        assert all(abs(s.n - w.n) <= 0.10 * max(s.n, w.n) for s, w, _ in pairs)
        assert {w.n for _, w, _ in pairs} == {108, 93}

    def test_pairs_stay_within_their_stratum(self):
        strat = ["A"] * 20 + ["B"] * 20 + ["A"] * 20 + ["B"] * 20
        recs = self._recs(range(100, 140, 1), range(90, 130, 1), stratum=strat)
        pairs = self._match(recs, caliper=0.2, stratum=lambda r: r.k)
        assert pairs and all(s.k == w.k for s, w, _ in pairs)

    def test_each_weak_record_used_once_and_n_pairs_respected(self):
        recs = self._recs(range(100, 300), range(90, 290))
        pairs = self._match(recs, n_pairs=40, caliper=0.2)
        assert len(pairs) == 40
        assert len({id(w) for _, w, _ in pairs}) == 40
        assert [t for _, _, t in pairs[:4]] == ["t1", "t2", "t1", "t2"]

    def test_deterministic(self):
        recs = self._recs(range(100, 200, 3), range(80, 180, 3))
        a = [(s.source_record_id, w.source_record_id) for s, w, _ in self._match(recs, caliper=0.2)]
        b = [(s.source_record_id, w.source_record_id) for s, w, _ in self._match(recs, caliper=0.2)]
        assert a == b

    def test_report_states_the_length_only_accuracy(self):
        from runners.run_crossinfluence import _pairing_report

        recs = self._recs(range(100, 200, 2), range(60, 160, 2))
        pairs = self._match(recs, caliper=0.2)
        rep = _pairing_report(pairs, recs, lambda r: r.flag, lambda r, tid: r.n, mode="length_matched",
                              caliper=0.2, requested=100, stratified=False)
        assert rep["length_only_accuracy"] == 0.5
        assert rep["n_pairs"] == len(pairs) and rep["n_requested"] == 100
        assert rep["pool_median_length_strong"] > rep["median_length_strong"]  # drawn from the overlap


def test_education_pairs_within_prompt():
    from substrates.domains import get_domain

    assert get_domain("education").pair_stratum(SimpleNamespace(prompt_id="P")) == "P"
    assert get_domain("credit").pair_stratum is None and get_domain("cv").pair_stratum is None
