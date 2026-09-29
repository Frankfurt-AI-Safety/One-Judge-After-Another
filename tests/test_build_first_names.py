"""`runners/build_first_names.py`: the sex-coded first-name list the Bias-in-Bios scrub drops bios by."""

from __future__ import annotations

import hashlib

import pytest

from runners import build_first_names as bfn


def _csv(rows):
    """baby-names.csv rows (year, name, percent, sex), percent = share of that sex's births."""
    return '"year","name","percent","sex"\n' + "".join(f'{y},"{n}",{p},"{s}"\n' for y, n, p, s in rows)


# One year inside the window is enough: a name passes MIN_SHARE when its summed share / 61 years >= 5e-5.
_ROWS = [
    (1970, "Anna", 0.01, "girl"),                                   # sex-coded
    (1970, "Pat", 0.005, "boy"), (1970, "Pat", 0.005, "girl"),     # neutral: 50%
    (1970, "Rare", 0.0001, "girl"),                                 # too rare
    (1900, "Old", 0.05, "boy"),                                     # outside the birth years
    (1970, "Ninety", 0.009, "boy"), (1970, "Ninety", 0.001, "girl"),       # exactly 90% one sex
    (1970, "Eighty9", 0.0089, "boy"), (1970, "Eighty9", 0.0011, "girl"),   # 89%
    (1970, "Will", 0.01, "boy"), (1970, "May", 0.01, "girl"), (1970, "Austin", 0.01, "boy"),
]


def test_sex_coded_names_applies_window_share_and_sex_threshold():
    assert bfn.sex_coded_names(_csv(_ROWS)) == {"Anna", "Ninety", "Will", "May", "Austin"}


def _parquet(tmp_path, texts):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    path = tmp_path / "bios.parquet"
    pd.DataFrame({"hard_text": texts, "profession": [0] * len(texts), "gender": [0] * len(texts)}).to_parquet(path)
    return path


def test_word_like_is_the_lowercase_to_capitalised_ratio(tmp_path):
    bios = _parquet(tmp_path, ["Will you will? We will. Anna met Anna.", "Hope will help. Grace"])
    # Will: 1 capitalised, 3 lowercase -> word-like; Anna: never lowercase; Grace: 1:0; Hope: 1:0;
    # a name that never occurs capitalised is not word-like (nothing to match)
    assert bfn.word_like({"Will", "Anna", "Grace", "Hope", "Zed"}, bios) == {"Will"}


def test_read_source_refuses_other_data(tmp_path):
    good = tmp_path / "names.csv"
    good.write_bytes(_csv(_ROWS).encode())
    digest = hashlib.sha256(good.read_bytes()).hexdigest()
    assert bfn.read_source(good, expected_sha256=digest) == _csv(_ROWS)
    with pytest.raises(SystemExit, match="refusing"):
        bfn.read_source(good)  # the pinned mirror's hash
    assert bfn.SOURCE_COMMIT in bfn.SOURCE_URL and "master" not in bfn.SOURCE_URL


def test_main_writes_names_and_both_input_hashes(tmp_path, monkeypatch):
    src = tmp_path / "names.csv"
    src.write_bytes(_csv(_ROWS).encode())
    bios = _parquet(tmp_path, ["Will you will? we will. Anna and May and Austin."])
    out = tmp_path / "first_names.txt"
    monkeypatch.setattr(bfn, "SOURCE_SHA256", hashlib.sha256(src.read_bytes()).hexdigest())
    monkeypatch.setattr("sys.argv", ["build_first_names.py", "--source-csv", str(src), "--bios", str(bios),
                                     "--out", str(out)])
    bfn.main()
    lines = out.read_text().splitlines()
    header = [line for line in lines if line.startswith("#")]
    assert [line for line in lines if not line.startswith("#")] == ["Anna", "Ninety"]  # minus word, month, place
    inputs = next(line for line in header if line.startswith("# Inputs (SHA-256):"))
    assert bfn.SOURCE_SHA256 in inputs and hashlib.sha256(bios.read_bytes()).hexdigest() in inputs
    assert "# 5 sex-coded candidates, 1 word-like removed, 2 kept." in header


def test_the_curated_lists_are_complete():
    # A comment typed after "PLACES = {" once swallowed the set's first nine names, and six of them
    # (Alberta, Asia, Austin, Brooklyn, Carolina, Chad) came back into the list; only a rebuild diff showed it.
    assert len(bfn.CALENDAR) == 19 and len(bfn.PLACES) == 36
    assert {"Adelaide", "Africa", "Alberta", "America", "Asia", "Austin", "Brooklyn", "Carolina", "Chad",
            "Virginia", "Christian", "Summer"} <= bfn.PLACES
