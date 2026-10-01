"""
Tests for the shared demographic-domain registry (`domains.py`), which guards the domain dispatch of
the battery, additivity and cross-marker runners. (The direct cross-influence pairing tests went with
`run_crossinfluence.py` on 2026-09-24.)
"""

from __future__ import annotations

from pathlib import Path
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

    def test_corpus_is_the_file_load_records_reads(self, tmp_path, monkeypatch):
        # the reasoning runners hash `corpus` and check it against the manifest's sources: it must be the file
        # `load_records` opens. Run where no corpus exists, every loader names the file it looked for.
        monkeypatch.chdir(tmp_path)
        for name in DOMAINS:
            dom = get_domain(name)
            assert isinstance(dom.corpus, Path) and not dom.corpus.is_absolute()
            with pytest.raises((FileNotFoundError, SystemExit)) as err:
                dom.load_records()
            assert str(dom.corpus) in str(err.value), name

    def test_template_ids_match_renderers(self):
        assert get_domain("credit").template_ids == tuple(sorted(TEMPLATES))
        assert get_domain("cv").template_ids == tuple(sorted(BIOS_TEMPLATES))
        assert get_domain("education").template_ids == tuple(sorted(EDU_TEMPLATES))

    def test_quality_field_names_the_is_strong_label(self):
        for name, field in (("credit", "credit_good"), ("cv", "qualified"), ("education", "high_quality")):
            assert get_domain(name).quality_field == field
            # the probe_records split stratifies on the same label
            assert get_domain(name).dataset_cls.QUALITY_FIELD == field

    def test_every_demographic_config_sizes_its_probe_in_records(self):
        # Audit item 4.1: every demographic config reads a record-grouped manifest, so its probe split
        # is counted in records; probe_size (pairs) would leave the record count to the design.
        from pathlib import Path

        import yaml

        from scoring.experiment import ExperimentConfig
        paths = sorted(Path("configs").glob("demographic_*.yaml"))
        assert paths
        for path in paths:
            assert "probe_size" not in yaml.safe_load(path.read_text()), path.name
            n = ExperimentConfig.from_yaml(path).probe_records
            assert isinstance(n, int) and n > 0, path.name

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
        # the reasoning arm's credit frame drops the records whose employment or job reads unemployed (an unpaid
        # sabbatical contradicts them): 13 of the 564 good-credit ones, the count `pairs/verdicts.py`, the credit
        # reasoning config and the working notes state
        from pairs.verdicts import REASONING_FRAMES

        strong = [r for r in credit_recs if r.credit_good]
        assert len(strong) == 564 and sum(not REASONING_FRAMES["credit"].is_eligible(r) for r in strong) == 13

    def test_cv_load_records_needs_corpus(self):
        # The hiring arm now loads real biographies (Bias-in-Bios), user-downloaded like the
        # education corpus — skip gracefully if it isn't present locally rather than failing.
        try:
            recs = get_domain("cv").load_records()
        except FileNotFoundError:
            pytest.skip("Bias-in-Bios corpus not downloaded (data/demographic/cv/raw/)")
        assert len(recs) > 0
        # These attribute names are load-bearing: the reasoning runners select on `is_strong` (.qualified) and
        # verdicts.py reads .role.
        assert hasattr(recs[0], "qualified") and hasattr(recs[0], "role")

    def test_education_load_records_needs_corpus(self):
        # Education loads a user-downloaded corpus; skip gracefully if it isn't present locally.
        try:
            recs = get_domain("education").load_records()
        except FileNotFoundError:
            pytest.skip("ASAP 2.0 corpus not downloaded (data/demographic/education/raw/)")
        assert len(recs) > 0 and hasattr(recs[0], "high_quality")
