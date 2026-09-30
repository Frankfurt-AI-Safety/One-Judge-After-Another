"""`runners/run_reasoning_erasure.py`: the design that keeps surface features from answering (no connective,
held-out wording; pinned on the lexical control itself), the labels, the refusals, provenance — end to end on the
tiny model and the `hiring` fixture."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random

import pytest
import yaml

from runners import run_reasoning_erasure as rre
from substrates.domains import get_domain
from tests.test_run_cross_marker import _model, _tokenizer


def test_labels_follow_the_cells():
    assert [rre.label(c, "correctness") for c in rre.REASONING_CELLS] == [1, 1, 0, 0]
    assert [rre.label(c, "conclusion") for c in rre.REASONING_CELLS] == [0, 1, 1, 0]


def _lexical_mlp_after_leace(fit, evaluate, connective):
    """MLP accuracy after LEACE on bag-of-words vectors of the verdicts (correctness, 150 + 150 items)."""
    from pairs.verdicts import build_reasoning_item
    from probes.erasure import apply_eraser, bag_of_words, leace_erase, probe_recoverability
    from substrates.bios_render import render_bio
    from tests.test_reasoning_flip import _rec

    def split(offset, paraphrases):
        items = [build_reasoning_item(_rec(f"r{i}"), "commute", render_bio, random.Random(offset + i), vary=True,
                                      paraphrases=paraphrases, connective=connective) for i in range(150)]
        return ([it["cells"][c] for it in items for c in rre.REASONING_CELLS],
                [rre.label(c, "correctness") for _ in items for c in rre.REASONING_CELLS])

    (tr, ytr), (ev, yev) = split(0, fit), split(10_000, evaluate)
    Xtr, Xev = bag_of_words(tr, ev)
    eraser = leace_erase(Xtr, ytr)
    return probe_recoverability(apply_eraser(eraser, Xtr), ytr, apply_eraser(eraser, Xev), yev, n_boot=10)["mlp_acc"]


def test_surface_features_alone_recover_correctness_after_leace_only_in_the_old_design():
    from pairs.verdicts import EVAL_PARAPHRASES, FIT_PARAPHRASES

    # shared pools (with or without "so"/"but"): word identity alone gives the "entangled" reading
    assert _lexical_mlp_after_leace(None, None, connective=True) > 0.9
    assert _lexical_mlp_after_leace(None, None, connective=False) > 0.9
    # the runner's design: held-out wording, no connective — the lexical control sits at chance
    assert _lexical_mlp_after_leace(FIT_PARAPHRASES, EVAL_PARAPHRASES, connective=False) < 0.6


@pytest.fixture
def run(hiring, tmp_path, monkeypatch):
    from substrates.bios_clean import DEFAULT_N_BIOS, load_factorial_bios

    pairs, raw = hiring
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rre, "DEFAULT_BIOS_PATH", str(raw))
    monkeypatch.setattr(rre, "get_domain", lambda name: dataclasses.replace(
        get_domain(name), load_records=lambda: load_factorial_bios(str(raw), n=DEFAULT_N_BIOS)))
    loads = []

    def _run(*extra, domain="cv"):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(pairs),
            "batch_size": 16, "max_length": 1024, "extra": {"domain": domain, "n_boot": 20}}))

        def load_model(self):
            loads.append(1)
            self.model, self.tokenizer = _model(), _tokenizer()
            self.config.model_revision = "abc123"

        monkeypatch.setattr(rre.DemographicBiasExperiment, "load_model", load_model)
        monkeypatch.setattr("sys.argv", ["run_reasoning_erasure.py", "--config", str(cfg_path), "--probe-items", "8",
                                         "--eval-items", "6", *extra])
        rre.main()
        return json.loads((tmp_path / "artifacts/results/demographic/erasure_cv_Tiny-RM.json").read_text())

    _run.loads = loads
    return _run


def test_end_to_end_on_a_tiny_model(run, hiring):
    result = run()
    raw = hiring[1]
    assert result["meta"]["config"]["model_revision"] == "abc123"
    assert result["meta"]["data"][raw.name]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert result["meta"]["settings"]["connective"] is False
    assert not set(result["probe_records"]) & set(result["eval_records"])
    assert [(r["premise"], r["concept"]) for r in result["results"]] == \
        [(p, c) for p in rre.PREMISES for c in rre.CONCEPTS]
    r = result["results"][0]
    assert (r["n_train"], r["n_eval"]) == (32, 24)
    for rows in (r["rows"], r["lexical_control"]):
        assert set(rows) == set(rre.METHODS)
        assert rows["leace"]["intervals"]["mlp_acc"]["n_clusters"] == 6        # applicants, not states
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


def test_bad_inputs_are_refused_before_the_model_loads(run):
    with pytest.raises(SystemExit, match="hiring-only"):
        run(domain="credit")
    with pytest.raises(SystemExit, match="fewer than"):
        run("--eval-items", "100000")
    assert run.loads == []
