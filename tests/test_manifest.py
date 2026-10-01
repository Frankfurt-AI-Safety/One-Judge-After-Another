"""pairs/manifest.py: the template registry that provenance hashes come from."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import pytest

from pairs.manifest import (
    GENERATOR_VERSION, _ALL_TEMPLATES, _merge_templates, _spotcheck_sample, _template_hash, code_provenance,
    file_sha256, write_manifest,
)
from pairs.positionality import POSITION_TEMPLATES
from substrates.bios_render import BIOS_TEMPLATES
from substrates.credit_render import TEMPLATES
from substrates.education_render import EDU_TEMPLATES


def test_registry_holds_every_domains_templates():
    for reg in (TEMPLATES, BIOS_TEMPLATES, EDU_TEMPLATES, POSITION_TEMPLATES):
        assert reg, "an empty registry would leave its domain's pairs unhashable"
        for tid, text in reg.items():
            assert _ALL_TEMPLATES[tid] == text
            assert len(_template_hash(tid)) == 12
    assert len(_ALL_TEMPLATES) == sum(map(len, (TEMPLATES, BIOS_TEMPLATES, EDU_TEMPLATES, POSITION_TEMPLATES)))


def test_a_template_id_defined_twice_raises():
    with pytest.raises(ValueError, match="edu_v1"):
        _merge_templates({"edu_v1": "a", "x": "b"}, {"edu_v1": "c"})
    assert _merge_templates({"a": "1"}, {"b": "2"}) == {"a": "1", "b": "2"}


def test_an_unknown_template_id_raises():
    with pytest.raises(KeyError):
        _template_hash("no_such_template")


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_code_provenance_names_the_commit_and_what_differs_from_it(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("data/\n")
    (tmp_path / "gen.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "c")
    head = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "pairs.jsonl").write_text("{}\n")  # generated data is ignored: still clean
    assert code_provenance(tmp_path) == {"git_commit": head, "git_dirty": False, "git_dirty_paths": []}

    (tmp_path / "gen.py").write_text("x = 2\n")
    (tmp_path / "new.py").write_text("")
    assert code_provenance(tmp_path) == {"git_commit": head, "git_dirty": True,
                                         "git_dirty_paths": ["gen.py", "new.py"]}


def test_code_provenance_outside_a_repo_is_none(tmp_path):
    assert code_provenance(tmp_path / "not_a_repo") == {
        "git_commit": None, "git_dirty": None, "git_dirty_paths": None}


def test_a_staged_copy_without_git_reports_the_commit_stage_sh_recorded(tmp_path):
    # the cluster's copy holds tracked files only, no .git: cluster/stage.sh writes the commit next to them
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "STAGED_COMMIT").write_text(json.dumps({"git_commit": "abc123", "git_dirty": False,
                                                      "git_dirty_paths": [], "staged_utc": "2026-10-01T20:00:00"}))
    assert code_provenance(staged) == {"git_commit": "abc123", "git_dirty": False, "git_dirty_paths": [],
                                       "source": "staged", "staged_utc": "2026-10-01T20:00:00"}


def test_manifest_records_version_and_code(tmp_path):
    paths = write_manifest(tmp_path, [], seed=1, discard_report={}, thresholds={}, domain="d", attribution="a")
    m = json.loads(paths["manifest"].read_text())
    assert m["generator_version"] == GENERATOR_VERSION
    assert set(m["code"]) == {"git_commit", "git_dirty", "git_dirty_paths"}
    assert (m["domain"], m["attribution"]) == ("d", "a")


def test_domain_and_attribution_have_no_default(tmp_path):
    with pytest.raises(TypeError, match="domain"):
        write_manifest(tmp_path, [], seed=1, discard_report={}, thresholds={}, attribution="a")
    with pytest.raises(TypeError, match="attribution"):
        write_manifest(tmp_path, [], seed=1, discard_report={}, thresholds={}, domain="d")


def _rows(strata):
    # rows in a fixed per-block order, as the generators write them
    return [{"id": f"{a}-{e}-{b}", "varied_axis": a, "encoding": e} for b in range(100) for a, e in strata]


def test_spotcheck_covers_every_axis_and_encoding():
    strata = [("sex", "explicit"), ("age", "explicit"), ("sex", "proxy"), ("age", "proxy"),
              ("marital_status", "explicit"), ("intersection", "explicit"), ("intersection", "proxy")]
    sample = _spotcheck_sample(_rows(strata), 30, seed=42)
    per = {s: sum((r["varied_axis"], r["encoding"]) == s for r in sample) for s in strata}
    assert per == {s: 5 for s in strata}  # ceil(30 / 7)
    rows = _rows(strata)
    assert [rows.index(r) for r in sample] == sorted(rows.index(r) for r in sample)  # manifest order


def test_spotcheck_is_seeded_and_keeps_small_strata_whole():
    rows = _rows([("sex", "explicit")]) + [{"id": "only", "varied_axis": "rare", "encoding": "proxy"}]
    assert _spotcheck_sample(rows, 10, seed=1) == _spotcheck_sample(rows, 10, seed=1)
    assert _spotcheck_sample(rows, 10, seed=1) != _spotcheck_sample(rows, 10, seed=2)
    assert rows[-1] in _spotcheck_sample(rows, 10, seed=1)
    assert _spotcheck_sample([], 10, seed=1) == [] and _spotcheck_sample(rows, 0, seed=1) == []


def test_positioned_pairs_also_hash_their_essay_shell(tmp_path):
    import random

    from pairs.manifest import pair_to_record
    from pairs.positionality import make_positioned_pairs
    from substrates.education_ingest import EssayRecord

    rec = EssayRecord(source_record_id="essay-0", essay_text="One. Two. Three. Four.", holistic_score=5.0,
                      high_quality=True, source_dataset="asap2")
    pair = make_positioned_pairs(rec, "pos_sex", "conclusion", random.Random(0), header_template="edu_v2")[0]
    row = pair_to_record(pair, "p0", seed=1, domain="education")
    assert row["provenance"]["template_hash"] == _template_hash(pair.template_id)
    assert row["provenance"]["header_template_hash"] == _template_hash("edu_v2")
    paths = write_manifest(tmp_path, [row], seed=1, discard_report={}, thresholds={}, domain="education",
                           attribution="a")
    assert set(json.loads(paths["manifest"].read_text())["templates"]) == {pair.template_id, "edu_v2"}


def test_pairs_without_a_shell_carry_no_shell_hash():
    import random

    from pairs.factorial import CREDIT_DESIGN, factorial_pairs
    from pairs.manifest import pair_to_record
    from substrates.credit_render import render_profile
    from tests.test_credit_pipeline import _fake_record

    pair = factorial_pairs(_fake_record(), "credit_v1", "explicit", render_profile, random.Random(0),
                           axes=("sex",), design=CREDIT_DESIGN)[0][0]
    assert "header_template_hash" not in pair_to_record(pair, "c0", seed=1, domain="credit")["provenance"]


def test_rows_carry_no_role_and_the_summary_counts_by_axis_and_encoding(tmp_path):
    """`role` was always "probe" (the split into probe and eval happens in the dataset); removed 2026-09-28."""
    import random

    from pairs.factorial import CREDIT_DESIGN, factorial_pairs
    from pairs.manifest import pair_to_record
    from substrates.credit_render import render_profile
    from tests.test_credit_pipeline import _fake_record

    pair = factorial_pairs(_fake_record(), "credit_v1", "explicit", render_profile, random.Random(0),
                           axes=("sex",), design=CREDIT_DESIGN)[0][0]
    row = pair_to_record(pair, "c0", seed=1, domain="credit")
    assert "role" not in row
    paths = write_manifest(tmp_path, [row], seed=1, discard_report={}, thresholds={}, domain="credit",
                           attribution="a")
    assert json.loads(paths["manifest"].read_text())["counts_by_axis_encoding"] == {"sex/explicit": 1}


# --------------------------------------------------------------------------- the files a manifest names
def _pair_row(i=0):
    return {"id": f"p{i}", "varied_axis": "sex", "encoding": "explicit", "template_id": "credit_v1",
            "label_a": "female", "label_b": "male", "text_a": "a", "text_b": "b"}


def _write(tmp_path, rows, **kw):
    return write_manifest(tmp_path, rows, seed=1, discard_report={}, thresholds={}, domain="credit",
                          attribution="a", **kw)


def test_file_sha256_is_the_digest_of_the_bytes(tmp_path):
    (tmp_path / "f").write_bytes(b"abc" * 1_000_000)  # more than one chunk
    assert file_sha256(tmp_path / "f") == hashlib.sha256(b"abc" * 1_000_000).hexdigest()


def test_manifest_names_its_data_files_by_row_count_and_hash(tmp_path):
    cells = [{"id": "c0", "cells": []}, {"id": "c1", "cells": []}]
    paths = _write(tmp_path, [_pair_row(0), _pair_row(1), _pair_row(2)], cells=cells)
    m = json.loads(paths["manifest"].read_text())
    assert paths["cells"] == tmp_path / "cells.jsonl"
    assert [json.loads(line) for line in paths["cells"].read_text().splitlines()] == cells
    assert m["files"] == {
        "pairs.jsonl": {"n_rows": 3, "sha256": hashlib.sha256(paths["pairs"].read_bytes()).hexdigest()},
        "cells.jsonl": {"n_rows": 2, "sha256": hashlib.sha256(paths["cells"].read_bytes()).hexdigest()},
    }
    assert m["sources"] == {}


def test_manifest_names_the_corpus_files_by_hash(tmp_path):
    src = tmp_path / "german.data"
    src.write_text("A11 6 ...\n")
    m = json.loads(_write(tmp_path / "out", [_pair_row()], sources={"german.data": src})["manifest"].read_text())
    assert m["sources"] == {"german.data": {"path": str(src), "sha256": file_sha256(src)}}


def test_a_build_without_cells_removes_an_earlier_cells_file(tmp_path):
    _write(tmp_path, [_pair_row()], cells=[{"id": "c0"}])
    paths = _write(tmp_path, [_pair_row()])
    assert not (tmp_path / "cells.jsonl").exists() and "cells" not in paths
    assert set(json.loads(paths["manifest"].read_text())["files"]) == {"pairs.jsonl"}


def test_a_build_that_fails_part_way_leaves_no_manifest(tmp_path):
    _write(tmp_path, [_pair_row()])
    with pytest.raises(TypeError):
        _write(tmp_path, [_pair_row()], cells=[{"id": object()}])  # not serialisable: fails after pairs.jsonl
    assert not (tmp_path / "manifest.json").exists()
