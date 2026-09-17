#!/usr/bin/env python3
"""
Build `substrates/resources/first_names.txt`: the sex-coded first names the Bias-in-Bios loader uses to
drop biographies that still mention a person by first name after scrubbing (a direct sex cue).

Source: U.S. Social Security Administration national baby-name data (public domain), via the mirror
`hadley/data-baby-names` (top 1000 names per sex per year, 1880-2008, as a share of that sex's births).
SSA blocks scripted downloads of the original archive.

Selection, in order:
  1. birth years 1940-2000 (the working-age adults the corpus describes);
  2. mean yearly share (boys + girls) >= MIN_SHARE, so rare names do not bloat the list;
  3. sex-coded: at least SEX_SHARE of the name's share is one sex (gender-neutral names such as
     Jordan or Taylor are not sex cues, so they are not listed);
  4. not an ordinary word: in the Bias-in-Bios corpus, lowercase uses are below WORD_RATIO of the
     capitalised uses (drops Will, Grace, Hope, Guy, ...);
  5. not a calendar word or a name whose corpus uses are dominated by a place / institution
     (curated lists below: Virginia, Austin, Madison, ...). Place contexts of the remaining names
     ("Santa Barbara", "Cornell University") are skipped at match time in `bios_ingest`.

Usage (from the repo root):
    python runners/build_first_names.py
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from substrates.bios_ingest import DEFAULT_BIOS_PATH, FIRST_NAMES_PATH

SOURCE_URL = "https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv"
YEARS = range(1940, 2001)
MIN_SHARE = 5e-5
SEX_SHARE = 0.9
WORD_RATIO = 0.15
CALENDAR = {"January", "February", "March", "April", "May", "June", "July", "August", "September",
            "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"}
PLACES = {"Adelaide", "Africa", "Alberta", "America", "Asia", "Austin", "Brooklyn", "Carolina", "Chad",
          "Charlotte", "China", "Christian", "Cleveland", "Dakota", "Dallas", "Denver", "Europe",
          "Florence", "Georgia", "Houston", "India", "Israel", "Jamaica", "Jordan", "Kenya", "Lincoln",
          "Madison", "Orlando", "Paris", "Phoenix", "Savannah", "Summer", "Sydney", "Trinity",
          "Victoria", "Virginia"}


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

    if args.source_csv:
        csv_text = args.source_csv.read_text()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp:
            csv_text = resp.read().decode("utf-8")
    names = sex_coded_names(csv_text)
    words = word_like(names, args.bios)
    kept = sorted(names - words - CALENDAR - PLACES)
    header = [
        "# Sex-coded U.S. first names, built by runners/build_first_names.py -- do not edit by hand.",
        f"# Source: SSA national baby names (public domain) via {SOURCE_URL}",
        f"# Birth years {YEARS.start}-{YEARS.stop - 1}; mean share >= {MIN_SHARE}; >= {SEX_SHARE:.0%} one sex;",
        f"# lowercase/capitalised ratio in Bias-in-Bios < {WORD_RATIO}; calendar and place names removed.",
        f"# {len(names)} sex-coded candidates, {len(words)} word-like removed, {len(kept)} kept.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + kept) + "\n")
    print(f"wrote {len(kept)} names -> {args.out}")


if __name__ == "__main__":
    main()
