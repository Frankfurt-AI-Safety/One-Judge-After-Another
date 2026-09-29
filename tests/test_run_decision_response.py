"""`runners/run_decision_response.py`, the blatant floor: its records (the manifest's population, minus the direction's
probe records), provenance, names, and the gate columns of a gated head — end to end on the tiny models."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
import yaml

from runners import run_decision_response as rdr
from substrates.domains import get_domain
from tests.test_run_cross_marker import _model, _record, _tokenizer, manifest  # noqa: F401  (fixture)

# the fixture manifest's records (tests/test_run_cross_marker.py::manifest)
RECORDS = [_record("credit", f"s{i}", True) for i in range(8)] + [_record("credit", f"w{i}", False) for i in range(8)]


def test_the_hiring_domain_loads_the_manifests_pool(monkeypatch):
    # Without n, the labels were assigned on all 32,774 bios: 147 of the manifest's 12,000 got another label and
    # 5,583 another target role. The domain and the generator must use the same pool size.
    import substrates.domains as domains
    from runners import generate_bios
    from substrates.bios_clean import DEFAULT_N_BIOS

    calls = []
    monkeypatch.setattr(domains, "load_factorial_bios", lambda *a, **kw: calls.append(kw) or [])
    get_domain("cv").load_records()
    assert calls == [{"n": DEFAULT_N_BIOS}] and generate_bios.DEFAULT_N_BIOS is DEFAULT_N_BIOS


def test_items_exclude_the_probe_records_and_follow_a_seeded_order():
    chosen, rep = rdr.select_items(RECORDS, lambda r: r.credit_good, {"s0", "s1", "w0"}, 4, seed=7)
    assert [r.source_record_id for r in chosen] == \
        [r.source_record_id for r in rdr.select_items(RECORDS, lambda r: r.credit_good, {"s0", "s1"}, 4, 7)[0]]
    assert all(r.credit_good and r.source_record_id not in {"s0", "s1"} for r in chosen)
    assert rep == {"strong_records": 8, "excluded_probe_records": 2, "available": 6, "requested": 4, "n_items": 4}


@pytest.fixture
def run(manifest, tmp_path, monkeypatch):
    """`main` on the fixture manifest; ``model_fn``/``tok_fn`` stand in for the loader."""
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(manifest),
        "probe_records": 6, "batch_size": 16, "max_length": 1024, "extra": {"domain": "credit", "n_boot": 50}}))
    monkeypatch.setattr(rdr, "get_domain", lambda name: dataclasses.replace(get_domain(name),
                                                                          load_records=lambda: RECORDS))
    loads = []

    def _run(*extra, model_fn=_model, tok_fn=_tokenizer):
        def load_model(self):
            loads.append(1)
            self.model, self.tokenizer = model_fn(), tok_fn()
            self.config.model_revision = "abc123"

        monkeypatch.setattr(rdr.DemographicBiasExperiment, "load_model", load_model)
        monkeypatch.setattr("sys.argv", ["run_decision_response.py", "--config", str(cfg_path), "--n-items", "100",
                                         *extra])
        rdr.main()
        return json.loads((tmp_path / "artifacts/results/demographic/decision_credit_Tiny-RM_explicit.json").read_text())

    _run.loads = loads
    return _run


def test_end_to_end_on_a_tiny_model(run, manifest):
    result = run()
    assert result["meta"]["config"]["model_revision"] == "abc123"
    assert result["meta"]["data"]["pairs.jsonl"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    # 6 probe records, 3 of them strong: the 5 other strong records are the items
    sel = result["selection"]
    assert (sel["strong_records"], sel["excluded_probe_records"], sel["n_items"]) == (8, 3, 5)
    probe = get_domain("credit").dataset_cls(str(manifest), axis="sex", encoding="explicit", probe_records=6,
                                             split_seed=42).probe_record_ids()
    assert not set(result["records"]) & probe
    assert [r["axis"] for r in result["results"]] == ["sex", "age", "marital_status", "intersection"]
    assert all("gate_fixed" not in r for r in result["results"])        # a linear head has no gate
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


def test_a_gated_head_also_reports_the_gate_fixed_metrics(run):
    from tests.test_qrm import _model as qrm_model, _tokenizer as qrm_tokenizer

    result = run(model_fn=qrm_model, tok_fn=qrm_tokenizer)
    r = result["results"][0]
    assert {"gate_fixed", "nulled_gate_fixed", "gate_fixed_intervals"} <= set(r)
    assert r["gate_fixed"]["n"] == r["nulled_gate_fixed"]["n"] == r["baseline"]["n"] == 5
    # the gate of the unmarked prompt differs from the marked one, so the rescored rewards move
    assert r["gate_fixed"]["mean_gap_fair_minus_disc"] != pytest.approx(r["baseline"]["mean_gap_fair_minus_disc"],
                                                                        abs=1e-9)
