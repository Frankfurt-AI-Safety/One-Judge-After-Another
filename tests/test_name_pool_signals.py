"""configs/name_pool_signals.yaml must describe exactly the proxy name pools in pairs/markers.py: if a pool
changes, the cohort/class table (and the working-notes numbers computed from it) is stale."""

from pathlib import Path

import yaml

from pairs.markers import BLACK_FEMALE_NAMES, BLACK_MALE_NAMES, FEMALE_NAMES, MALE_NAMES

TABLE = Path(__file__).resolve().parent.parent / "configs" / "name_pool_signals.yaml"
POOLS = {"female_white": FEMALE_NAMES, "male_white": MALE_NAMES,
         "female_black": BLACK_FEMALE_NAMES, "male_black": BLACK_MALE_NAMES}
FIELDS = {"ssa_median_birth_year", "ssa_q25", "ssa_q75", "ssa_births", "ssa_same_sex_share",
          "gaddis_pct_some_college", "gaddis_ny_births", "gaddis_race"}


def _table():
    return yaml.safe_load(TABLE.read_text())


def test_table_lists_exactly_the_pools_in_order():
    table = _table()
    assert set(table) == set(POOLS)
    for pool, names in POOLS.items():
        assert list(table[pool]) == list(names), pool


def test_every_name_has_every_field_with_sane_values():
    for pool, rows in _table().items():
        race = "W" if pool.endswith("white") else "B"
        for name, row in rows.items():
            assert set(row) == FIELDS, name
            assert row["ssa_q25"] <= row["ssa_median_birth_year"] <= row["ssa_q75"], name
            assert 1880 <= row["ssa_q25"] and row["ssa_q75"] <= 2017, name
            assert 0.5 < row["ssa_same_sex_share"] <= 1.0, name
            assert 0.0 <= row["gaddis_pct_some_college"] <= 100.0, name
            assert row["gaddis_race"] == race, name
