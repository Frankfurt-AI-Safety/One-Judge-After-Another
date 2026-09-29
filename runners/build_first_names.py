#!/usr/bin/env python3
"""
Build `substrates/resources/first_names.txt`: the sex-coded first names the Bias-in-Bios loader uses to
drop biographies that still mention a person by first name after scrubbing (a direct sex cue).

Source: U.S. Social Security Administration national baby-name data (public domain), via the mirror
`hadley/data-baby-names` (top 1000 names per sex per year, 1880-2008, as a share of that sex's births),
pinned to its commit ``SOURCE_COMMIT`` and checked against ``SOURCE_SHA256`` (a mismatch, downloaded or
local, is refused). SSA blocks scripted downloads of the original archive. The second input, the Bias-in-Bios
parquet, comes from the HF mirror's moving ``main``; the list's header records the SHA-256 of both inputs.
Rebuilt 2026-09-29 from these inputs: the names are identical to the list built on 2026-09-17.

Selection, in order:
  1. birth years 1940-2000 (the working-age adults the corpus describes);
  2. mean yearly share (boys + girls) >= MIN_SHARE, so rare names do not bloat the list;
  3. sex-coded: at least SEX_SHARE of the name's share is one sex (gender-neutral names such as
     Jordan or Taylor are not sex cues, so they are not listed). The mirror holds only each sex's top
     1000, so a sex counts 0 in the years a name ranks lower there: 201 of the 1,767 listed names have their
     rarer sex in the data only in some years, and look more sex-coded than they are (Cameron 92% boys,
     Hunter 93%). The error drops a few more bios; it never lets a sex cue through;
  4. not an ordinary word: in the Bias-in-Bios corpus, lowercase uses are below WORD_RATIO of the
     capitalised uses (drops Will, Grace, Hope, Guy, ...);
  5. not a calendar word or a name whose corpus uses are dominated by another proper noun — a place or
     institution (Virginia, Austin, Madison, Trinity, ...), a religion (Christian) or a season (Summer)
     (curated lists below). Place contexts of the remaining names ("Santa Barbara", "Cornell University")
     are skipped at match time in `bios_ingest`.

Usage (from the repo root):
    python runners/build_first_names.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pairs.manifest import file_sha256
from substrates.bios_ingest import DEFAULT_BIOS_PATH, FIRST_NAMES_PATH

SOURCE_COMMIT = "b6525579cf7d2816e77fa885299445d15a09c906"  # the mirror's last commit (2009)
SOURCE_URL = f"https://raw.githubusercontent.com/hadley/data-baby-names/{SOURCE_COMMIT}/baby-names.csv"
SOURCE_SHA256 = "1259523fa76e5c18151a4c7612b854b22605d07127c596126940e2941ba15d3c"
YEARS = range(1940, 2001)
MIN_SHARE = 5e-5
SEX_SHARE = 0.9
WORD_RATIO = 0.15
CALENDAR = {"January", "February", "March", "April", "May", "June", "July", "August", "September",
            "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"}
# Places and institutions, plus other proper-noun uses (Christian, Summer).
PLACES = {"Adelaide", "Africa", "Alberta", "America", "Asia", "Austin", "Brooklyn", "Carolina", "Chad",
          "Charlotte", "China", "Christian", "Cleveland", "Dakota", "Dallas", "Denver", "Europe",
          "Florence", "Georgia", "Houston", "India", "Israel", "Jamaica", "Jordan", "Kenya", "Lincoln",
          "Madison", "Orlando", "Paris", "Phoenix", "Savannah", "Summer", "Sydney", "Trinity",
          "Victoria", "Virginia"}


def read_source(source_csv=None, expected_sha256: str | None = None) -> str:
    """The baby-names CSV (downloaded from the pinned ``SOURCE_URL``, or a local copy), refused unless its
    SHA-256 is ``expected_sha256`` (default ``SOURCE_SHA256``)."""
    expected_sha256 = expected_sha256 or SOURCE_SHA256
    if source_csv:
        data = Path(source_csv).read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp:
            data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(f"baby-names CSV has SHA-256 {digest}, expected {expected_sha256} "
                         f"(the mirror at commit {SOURCE_COMMIT}); refusing to build from other data")
    return data.decode("utf-8")


def sex_coded_names(csv_text: str) -> set:
    share = defaultdict(lambda: [0.0, 0.0])
    for row in csv.DictReader(io.StringIO(csv_text)):
        if int(row["year"]) in YEARS:
            share[row["name"]][0 if row["sex"] == "boy" else 1] += float(row["percent"])
    out = set()
    for name, (boy, girl) in share.items():
        total = boy + girl
        if total / len(YEARS) >= MIN_SHARE and max(boy, girl) / total >= SEX_SHARE:
            out.add(name)
    return out


def word_like(names: set, bios_path: Path) -> set:
    import pandas as pd

    lower = {n.lower(): n for n in names}
    cap, low = Counter(), Counter()
    for text in pd.read_parquet(bios_path).hard_text:
        for tok in re.findall(r"[A-Za-z]+", text):
            if tok in names:
                cap[tok] += 1
            elif tok in lower:
                low[lower[tok]] += 1
    return {n for n in names if cap[n] and low[n] / cap[n] >= WORD_RATIO}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-csv", type=Path, default=None, help="Local copy of baby-names.csv")
    ap.add_argument("--bios", type=Path, default=Path(DEFAULT_BIOS_PATH))
    ap.add_argument("--out", type=Path, default=FIRST_NAMES_PATH)
    args = ap.parse_args()

    csv_text = read_source(args.source_csv)
    names = sex_coded_names(csv_text)
    words = word_like(names, args.bios)
    kept = sorted(names - words - CALENDAR - PLACES)
    header = [
        "# Sex-coded U.S. first names, built by runners/build_first_names.py -- do not edit by hand.",
        f"# Source: SSA national baby names (public domain) via {SOURCE_URL}",
        f"# Inputs (SHA-256): baby-names.csv {SOURCE_SHA256}; {Path(args.bios).name} {file_sha256(args.bios)}",
        f"# Birth years {YEARS.start}-{YEARS.stop - 1}; mean share >= {MIN_SHARE}; >= {SEX_SHARE:.0%} one sex;",
        f"# lowercase/capitalised ratio in Bias-in-Bios < {WORD_RATIO}; calendar and place names removed.",
        f"# {len(names)} sex-coded candidates, {len(words)} word-like removed, {len(kept)} kept.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + kept) + "\n")
    print(f"wrote {len(kept)} names -> {args.out}")


if __name__ == "__main__":
    main()
