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
