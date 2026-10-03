"""`runners/run_rewardbench_guardrail.py` end to end with the tiny model (the edits of two domains: credit and
education fixture manifests; the length exclusion; the readable gate; provenance; the side file aligned with the kept
rows; the refusals) and its reading in isolation (the decision rule, the fixed sequence and the Bonferroni level,
the reproduction gate, the float32 head, the edits' arithmetic)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("sklearn")
pytest.importorskip("concept_erasure")

from runners import run_rewardbench_guardrail as rgr  # noqa: E402
from scoring import rewardbench as rb  # noqa: E402
from tests.test_rewardbench import _parquet  # noqa: E402


@pytest.fixture
def run(manifest, education_corpus, tmp_path, monkeypatch):
    """`main` on the credit and education fixture manifests and a synthetic benchmark (one Math row too long)."""
    from scoring.demographic_experiment import DemographicBiasExperiment
    from tests.test_run_cross_marker import _model, _tokenizer

    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    bench = _parquet(tmp_path, long_row=("Math", 2))
    real_load = rb.load
    monkeypatch.setattr(rgr.rb, "load", lambda revision=None: real_load(revision, path=bench))
    monkeypatch.chdir(tmp_path)
    direct = []
    for name, pairs, domain in (("credit", manifest, "credit"), ("edu", education_corpus[0], "education")):
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump({"name": name, "bias_type": "demographic", "model_path": "org/Tiny-RM",
                                     "dataset_source": str(pairs), "probe_records": 6, "max_length": 1024,
                                     "extra": {"domain": domain}}))
        direct.append(str(p))
    cfg_path = tmp_path / "guardrail.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "g", "bias_type": "demographic", "model_path": "org/Tiny-RM", "batch_size": 16, "max_length": 1024,
        "extra": {"direct_configs": direct, "alphas": [0.5, 1.0], "n_boot": 50, "delta": 0.02,
                  "random_draws": 2}}))
    loads = []

    def load_model(self):
        loads.append(1)
        self.model, self.tokenizer = _model(), _tokenizer()
        self.config.model_revision = "abc123"

    monkeypatch.setattr(DemographicBiasExperiment, "load_model", load_model)

    def go(*argv, config=cfg_path):
        monkeypatch.setattr("sys.argv", ["run_rewardbench_guardrail.py", "--config", str(config), *argv])
        rgr.main()
        return sorted((tmp_path / rgr.RESULTS_DIR).glob("rewardbench_guardrail_*.json"))

    go.loads, go.cfg_path, go.tmp, go.real_load, go.bench = loads, cfg_path, tmp_path, real_load, bench
    return go


def test_end_to_end(run, monkeypatch):
    calls = []
    real = rgr.edited_rewards
    monkeypatch.setattr(rgr, "edited_rewards", lambda saved, H, dtype, gates, edit: (calls.append(edit),
                                                                                       real(saved, H, dtype, gates,
                                                                                            edit))[1])
    (path,) = run()
    assert path.name == "rewardbench_guardrail_Tiny-RM.json"
    result = json.loads(path.read_text())
    assert result["meta"]["config"]["model_revision"] == "abc123"
    assert {"rewardbench2", "credit/pairs.jsonl", "credit/cells.jsonl", "education/pairs.jsonl",
            "education/cells.jsonl"} <= set(result["meta"]["data"])
    assert set(result["meta"]["settings"]["direct_config_settings"]) == {"credit", "education"}
    # the too-long Math row is out, from the baseline and every edit alike
    assert result["excluded"]["Math"] == 1 and sum(result["excluded"].values()) == 1
    assert result["n_rows"] == 5 * 6 + 8 - 1
    edits = result["edits"]
    expected = {f"credit/{e}/{a}/diffmean" for e, axes in (("explicit", ("sex", "age", "marital_status",
                                                                         "intersection")),
                                                          ("proxy", ("sex", "age", "intersection"))) for a in axes}
    expected |= {f"credit/explicit/{a}/leace" for a in ("sex", "age", "marital_status")}
    expected |= {f"credit/proxy/{a}/leace" for a in ("sex", "age")}
    expected |= {f"{d}/joint@{a}" for d in ("credit", "education", "all") for a in ("0.5", "1")}
    assert expected <= set(edits)
    assert {edits[n]["role"] for n in edits if "/joint@" in n} == {"confirmatory"}
    assert {edits[n]["role"] for n in edits if n.endswith(("/diffmean", "/leace"))} == {"exploratory"}
    # exploratory rows carry intervals, never pass flags
    assert all("non_inferior" not in e for e in edits.values() if e["role"] != "confirmatory")
    # the reference rows: random subspaces of the joint basis's rank, per family
    assert {n.split("/")[0] for n in edits if edits[n]["role"] == "reference"} == {"credit", "education", "all"}
    credit_refs = [n for n in edits if n.startswith("credit/random")]
    assert len(credit_refs) == 2 and 1 <= int(credit_refs[0].split("random")[1].split("#")[0]) <= 5   # its rank
    assert result["edit_meta"]["credit"]["joint_vectors"] == 5      # 3 explicit + 2 proxy single-axis directions
    assert set(result["edit_meta"]["credit"]["corner_outside_span"]) == {"explicit", "proxy"}
    # each joint edit's α reaches the projection
    joint_basis = [e for e in calls if e[0] == "project" and e[1].dim() == 2 and e[1].shape[0] == 5]
    assert sorted({e[2] for e in joint_basis if e[2] < 1.0} | {1.0}) == [0.5, 1.0]
    # the tiny model has no published entry: nothing is readable, every flag and claim is None
    assert result["reproduction"]["reproduced"] is None and result["readable"] is False
    assert result["claims"] is None and edits["credit/joint@1"]["non_inferior"] is None
    row = edits["credit/joint@1"]
    assert {"score", "change", "ci_low", "ci_high", "lower_bound_95", "lower_bound"} <= set(row["overall"])
    assert set(row["overall"]["lower_bound"]) == {"0.05", f"{0.05 / 3:g}"}     # three families: Bonferroni
    assert result["selection"]["credit"]["battery_split"]["explicit"]["identical"] is True
    # never replaced without --overwrite
    with pytest.raises(SystemExit, match="exists"):
        run()
    assert len(run.loads) == 1


def test_the_side_file_is_the_kept_rows_own_rewards(run):
    # the review's mutation M1: offsets of all rows instead of the kept ones misaligned rewards with rows silently
    from probes.heads import get_head
    from probes.probe import embed_with_gates
    from scoring.dataset_base import format_conversation
    from tests.test_run_cross_marker import _model, _tokenizer

    (path,) = run()
    side = [json.loads(line) for line in path.with_name(path.stem + "_baseline_scores.jsonl").open()]
    items = {it.row: it for it in run.real_load(path=run.bench)[0]}
    model, tok = _model(), _tokenizer()
    saved = get_head(model).to_saved()
    for line in side[::7]:
        it = items[line["row"]]
        assert line["subset"] == it.subset and line["id"] == it.id
        H, dtype, gates = embed_with_gates(model, tok, [format_conversation(tok, it.prompt, c) for c in it.completions],
                                           batch_size=8, max_length=1024, show_progress=False)
        np.testing.assert_allclose(line["scores"], rgr.head_rewards(saved, H, dtype, gates), rtol=1e-4, atol=1e-4)


def test_a_too_long_ties_row_takes_its_question_out(run, monkeypatch):
    bench = _parquet(run.tmp / "ties_long", long_row=(rb.TIES, 1))
    monkeypatch.setattr(rgr.rb, "load", lambda revision=None: run.real_load(revision, path=bench))
    (path,) = run("--max-items", "3")
    result = json.loads(path.read_text())
    assert result["excluded"][rb.TIES] == 2                          # the question's tied and ref rows
    assert path.name == "rewardbench_guardrail_Tiny-RM__max_items-3.json"
    assert result["readable"] is False and "subsample" in result["reproduction"]["note"]


def test_refusals_before_the_model_loads(run, tmp_path, monkeypatch):
    cfg = yaml.safe_load(run.cfg_path.read_text())
    cfg["extra"]["direct_configs"] = cfg["extra"]["direct_configs"][:1] * 2
    twice = tmp_path / "twice.yaml"
    twice.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="twice"):
        run(config=twice)
    cfg["extra"]["direct_configs"] = []
    none = tmp_path / "none.yaml"
    none.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="no direct config"):
        run(config=none)
    with pytest.raises(SystemExit, match="at least 2"):
        run("--max-items", "1")
    # a malformed published table stops the run here, not after hours of forward pass
    table = yaml.safe_load(rgr.PUBLISHED.read_text())
    del table["scores"]["nicolinho/QRM-Gemma-2-27B"]["Ties"]
    bad = tmp_path / "published.yaml"
    bad.write_text(yaml.safe_dump(table))
    monkeypatch.setattr(rgr, "PUBLISHED", bad)
    with pytest.raises(SystemExit, match="lacks"):
        run()
    assert run.loads == []


# --------------------------------------------------------------------------- the reading ------------------
def _rows(overall_lb, safety_lb, level="0.05"):
    return {"overall": {"lower_bound": {level: overall_lb}}, "Safety": {"lower_bound": {level: safety_lb}}}


def test_the_decision_rule():
    # the review's mutation M10 (δ's sign flipped) and the Safety criterion
    assert rgr.decide(_rows(-0.019, -0.01), "0.05", 0.02, 0.03) is True
    assert rgr.decide(_rows(-0.021, -0.01), "0.05", 0.02, 0.03) is False
    assert rgr.decide(_rows(-0.01, -0.031), "0.05", 0.02, 0.03) is False        # Safety alone fails it
    assert rgr.decide(_rows(+0.01, +0.01), "0.05", 0.02, 0.03) is True


def test_the_fixed_sequence_stops_at_the_first_fail():
    assert rgr.fixed_sequence([(0.25, True), (0.5, True), (0.75, False), (1.0, True)]) == 0.5
    assert rgr.fixed_sequence([(1.0, True), (0.25, True), (0.5, True)]) == 1.0     # tested in order of α
    assert rgr.fixed_sequence([(0.25, False), (0.5, True)]) is None


def test_claims_use_the_bonferroni_level_per_family():
    lv = f"{0.05 / 2:g}"

    def row(ok_95, ok_f):
        return {"non_inferior": ok_95, "overall": {"lower_bound": {"0.05": 0.0 if ok_95 else -1, lv: 0.0 if ok_f else -1}},
                "Safety": {"lower_bound": {"0.05": 0.0, lv: 0.0}}}

    results = {"credit/joint@0.5": row(True, True), "credit/joint@1": row(True, False),
               "all/joint@0.5": row(True, True), "all/joint@1": row(True, True)}
    c = rgr.claims(results, ["credit", "all"], [0.5, 1.0], 0.02, 0.03)
    assert c["family_level"] == pytest.approx(0.025)
    assert c["every_family_at_full_strength"] is True                  # at level 0.05, each part on its own
    assert c["families"]["credit"]["largest_alpha"] == 0.5             # fails at full strength at level 0.025
    assert c["families"]["all"]["largest_alpha"] == 1.0
    results["all/joint@1"]["non_inferior"] = False
    assert rgr.claims(results, ["credit", "all"], [0.5, 1.0], 0.02, 0.03)["every_family_at_full_strength"] is False


def test_the_reproduction_gate():
    table = yaml.safe_load(rgr.PUBLISHED.read_text())
    pub = {"source": table["source"], "revision": table["revision"],
           "scores": table["scores"]["Skywork/Skywork-Reward-V2-Qwen3-0.6B"], "completions": None}
    overall = float(np.mean([pub["scores"][s] for s in rb.SUBSETS]))
    ours = {**pub["scores"], "overall": overall + 0.004}
    assert rgr.reproduction(pub, ours, None, 0.01, 0.99, False)["reproduced"] is True
    assert rgr.reproduction(pub, {**ours, "overall": overall - 0.02}, None, 0.01, 0.99, False)["reproduced"] is False
    # where per-completion scores exist, the correlation must hold too, with every row matched
    good = {"r": 0.9995, "n_completions": 10, "unmatched_rows": []}
    assert rgr.reproduction(pub, ours, good, 0.01, 0.99, False)["reproduced"] is True
    assert rgr.reproduction(pub, ours, {**good, "r": 0.98}, 0.01, 0.99, False)["reproduced"] is False
    assert rgr.reproduction(pub, ours, {**good, "unmatched_rows": [3]}, 0.01, 0.99, False)["reproduced"] is False
    assert rgr.reproduction({**pub, "scores": None}, ours, None, 0.01, 0.99, False)["reproduced"] is None
    assert rgr.reproduction(pub, ours, None, 0.01, 0.99, True)["reproduced"] is None


def test_completion_agreement_matches_rows_by_subset_and_id(tmp_path):
    items, _ = rb.load(path=_parquet(tmp_path))
    offsets = rb.offsets_of(items)
    rewards = np.random.default_rng(0).normal(size=offsets[-1] + len(items[-1].completions))
    published = {(it.subset, it.id): list(rewards[o:o + len(it.completions)] * 2 + 1)
                 for it, o in zip(items, offsets)}
    agree = rb.completion_agreement(items, rewards, offsets, published)
    assert agree["r"] == pytest.approx(1.0) and agree["unmatched_rows"] == []
    del published[(items[0].subset, items[0].id)]
    assert rb.completion_agreement(items, rewards, offsets, published)["unmatched_rows"] == [items[0].row]


def test_the_head_is_float32_on_the_bf16_state():
    from probes.heads import get_head
    from tests.test_run_cross_marker import _model

    model = _model()
    saved = get_head(model).to_saved()
    H = torch.randn(30, model.config.hidden_size)
    r = rgr.head_rewards(saved, H, torch.bfloat16, None)
    w = model.score.weight.detach().float()[0]
    np.testing.assert_allclose(r, (H.to(torch.bfloat16).float() @ w).numpy(), rtol=1e-5, atol=1e-5)
    assert len(set(r.tolist())) == 30                                 # no bf16 rounding of the rewards


def test_edits_on_planted_states():
    # a head reading only its weight direction: projecting it out at strength α scales the reward by 1 − α
    from probes.heads import get_head
    from tests.test_run_cross_marker import _model

    model = _model()
    saved = get_head(model).to_saved()
    head = model.score.weight.detach().float()[0]
    H = torch.randn(20, head.shape[0])
    ref = (H @ head).numpy()
    base = rgr.edited_rewards(saved, H, torch.float32, None, ("none", None, 0.0))
    half = rgr.edited_rewards(saved, H, torch.float32, None, ("project", head, 0.5))
    full = rgr.edited_rewards(saved, H, torch.float32, None, ("project", head, 1.0))
    np.testing.assert_allclose(base, ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(half, ref / 2, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(full, 0, atol=1e-4)
    assert np.allclose(rgr.edited_rewards(saved, H, torch.float32, None, ("erase", lambda X: X * 0, 1.0)), 0)
    with pytest.raises(ValueError, match="unknown edit"):
        rgr.edited_rewards(saved, H, torch.float32, None, ("shift", head, 1.0))


def test_random_bases_and_the_corner_measure():
    qs = rgr.random_bases(16, 3, 2, seed=0)
    assert len(qs) == 2 and qs[0].shape == (3, 16)
    assert torch.allclose(qs[0] @ qs[0].T, torch.eye(3), atol=1e-5) and not torch.allclose(qs[0], qs[1])
    assert torch.equal(rgr.random_bases(16, 3, 2, seed=0)[1], qs[1])
    basis = torch.eye(4)[:2]
    assert rgr.outside_span(torch.tensor([1.0, 1.0, 0, 0]), basis) == pytest.approx(0, abs=1e-6)
    assert rgr.outside_span(torch.tensor([1.0, 0, 1.0, 0]), basis) == pytest.approx(2 ** -0.5)


def test_subsample_keeps_whole_ties_questions(tmp_path):
    items, _ = rb.load(path=_parquet(tmp_path))
    sub = rgr.subsample(items, 2)
    assert sum(1 for it in sub if it.subset == "Math") == 2
    ties = [it for it in sub if it.subset == rb.TIES]
    assert len(ties) == 4 and {it.question for it in ties} == {0, 1}
    assert rgr.subsample(items, None) == list(items)
