"""
Tests for `runners/run_placement_matrix.py`: settings and the result name, the three placements' rows (the
cross-marker arm's texts, one fold per record), the length refusal, the gate references; on synthetic states the
planted-signal worlds (one direction shared by the three placements → the transfers remove the effect; three
orthogonal ones → only the own direction does), the cross-fitting (no item nulled by a direction fitted on its
record) and the gated head's columns; `main` end to end on a generator-style credit manifest, with the comparative
arm's pairs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from pairs.comparative import PAIRINGS, contrast_axes, draw_pairs
from pairs.cross_marker import load_cell_blocks
from pairs.factorial import CREDIT_DESIGN
from probes import cross_marker_directions as cmd
from runners import run_comparative as rc
from runners.run_placement_matrix import (
    MATRIX_ROWS, OWN_ROW, PLACEMENTS, PLACEMENT_DEFAULTS, build_cross_rows, build_direct_rows, build_parser,
    check_lengths, check_placement, default_out, fit_rows, gate_refs, geometry, matrix, null_columns, resolve_all,
    target_values,
)
from scoring.placement_matrix import unit_sums
from scoring.dataset_base import format_conversation
from substrates.domains import get_domain

DOM = get_domain("credit")
TEMPLATES = ["credit_v1", "credit_v2"]
SEX = CREDIT_DESIGN.axes[0]


# --------------------------------------------------------------------------- settings ----------------
class TestSettings:
    def test_pairing_settings_are_the_comparative_ones_without_directions(self):
        s = resolve_all({"comparative": {"n_pairs": 7, "directions": ["own"]}}, {}, {})
        assert s["n_pairs"] == {p: 7 for p in PAIRINGS} and "directions" not in s
        assert s["x_paraphrases"] == PLACEMENT_DEFAULTS["x_paraphrases"] == 3   # the cross-marker arm's default
        assert resolve_all({"placement": {"x_paraphrases": 2}}, {}, {"x_paraphrases": None})["x_paraphrases"] == 2
        assert resolve_all({"placement": {"x_paraphrases": 2}}, {}, {"x_paraphrases": 1})["x_paraphrases"] == 1

    def test_bad_settings_are_refused(self):
        with pytest.raises(KeyError, match="unknown placement"):
            resolve_all({"placement": {"rows": []}}, {}, {})
        with pytest.raises(SystemExit, match="x_paraphrases"):
            check_placement(resolve_all({}, {}, {"x_paraphrases": 9}), "credit")
        with pytest.raises(SystemExit, match="n_folds"):
            check_placement(resolve_all({}, {"n_folds": 1}, {}), "credit")

    def test_default_out(self):
        assert default_out("credit", "data/demographic/credit/pairs.jsonl", "org/RM-8B") == \
            Path("artifacts/results/demographic/placement_credit_RM-8B.json")
        assert default_out("credit", "x/credit2/pairs.jsonl", "org/RM", "__seed-1").name == \
            "placement_credit_credit2_RM__seed-1.json"


# --------------------------------------------------------------------------- rows --------------------
def _tok():
    from tests.test_run_cross_marker import _tokenizer
    return _tokenizer()


def _built(manifest, tok=None, n_folds=2, encoding="explicit"):
    tok = tok or _tok()
    fmt = lambda p, r: format_conversation(tok, p, r)
    settings = resolve_all({}, {"n_pairs": 100, "n_folds": n_folds, "encodings": [encoding], "paraphrases": 1},
                           {"x_paraphrases": 1})
    blocks = load_cell_blocks(manifest.parent / "cells.jsonl", CREDIT_DESIGN)
    by_record, _, cand, _ = rc.candidates_from(blocks, quality_field="credit_good", match_field=None, exclude=set())
    pairs, _ = draw_pairs(cand, settings["n_pairs"], 42)
    rows, convs = {}, {}
    rows["comparative"], convs["comparative"] = rc.build_rows(pairs, by_record, "credit", encoding, TEMPLATES,
                                                              settings, fmt)
    rows["cross"], convs["cross"] = build_cross_rows(pairs, by_record, "credit", encoding, TEMPLATES, "credit_good",
                                                     settings, fmt)
    rows["direct"], convs["direct"] = build_direct_rows(pairs, by_record, CREDIT_DESIGN, encoding, TEMPLATES,
                                                        "credit_good", DOM.assessment_prompt, fmt)
    pairing_of = {p.pairing for p in pairs}
    assert pairing_of == set(PAIRINGS)          # the worlds below need strong records and every pairing
    pairing_of = {p.pair_id: p.pairing for p in pairs}
    folds = cmd.fold_assignment(sorted(pairing_of), pairing_of, n_folds, 42)
    return pairs, by_record, rows, convs, folds, settings


class TestRows:
    def test_counts_ids_and_one_pair_per_record(self, manifest):
        pairs, _, rows, convs, _, _ = _built(manifest)
        n = len(pairs) * 2 * len(TEMPLATES)
        assert len(rows["cross"]) == len(convs["cross"]) == n * 9 * 2       # 8 cells + unmarked, approve / decline
        assert len(rows["direct"]) == len(convs["direct"]) == n * 8
        assert {r["response"] for r in rows["cross"]} == {"approve", "decline"}
        pair_of = {r: p.pair_id for p in pairs for r in (p.x, p.y)}
        for placement in ("cross", "direct"):
            assert all(r["pair_id"] == pair_of[r["record_id"]] for r in rows[placement])
            assert not any("text" in r or "prompt" in r for r in rows[placement])

    def test_cross_texts_are_the_cross_marker_arms(self, manifest):
        # the same record, template and paraphrase setting as run_cross_marker: the same conversations (cache hits)
        from runners import run_cross_marker as rx

        tok = _tok()
        fmt = lambda p, r: format_conversation(tok, p, r)
        pairs, by_record, rows, convs, _, settings = _built(manifest, tok)
        rid = pairs[0].x
        x_settings = rx.resolve_settings({}, {"paraphrases": settings["x_paraphrases"], "seed": settings["seed"]})
        x_rows, x_convs = rx.build_rows({rid: [by_record[rid][("explicit", t)] for t in TEMPLATES]}, "credit",
                                        "credit_good", fmt, x_settings)
        arm = {(r["template_id"], str(r["cell"]), r["response"]): c for r, c in zip(x_rows, x_convs)}
        ours = [(r, c) for r, c in zip(rows["cross"], convs["cross"]) if r["record_id"] == rid]
        assert ours and all(c == arm[(r["template_id"], str(r["cell"]), r["response"])] for r, c in ours)

    def test_direct_texts_are_the_direct_format(self, manifest):
        tok = _tok()
        pairs, by_record, rows, convs, _, _ = _built(manifest, tok)
        r, c = rows["direct"][0], convs["direct"][0]
        block = by_record[r["record_id"]][("explicit", r["template_id"])]
        assert c == format_conversation(tok, DOM.assessment_prompt, block.texts[tuple(r["cell"])])

    def test_too_long_texts_are_refused(self, manifest):
        from runners.run_cross_marker import token_counter

        _, _, _, convs, _, _ = _built(manifest)
        check_lengths({p: convs[p] for p in ("cross", "direct")}, token_counter(_tok()), 10_000)
        with pytest.raises(SystemExit, match="cross texts exceed"):
            check_lengths({"cross": convs["cross"]}, token_counter(_tok()), 20)

    def test_gate_references(self, manifest):
        _, _, rows, _, _, _ = _built(manifest)
        assert gate_refs("direct", rows["direct"]) == list(range(len(rows["direct"])))
        for placement, own in (("cross", lambda r: r["cell"] == "unmarked"),
                               ("comparative", lambda r: r["axis"] == "unmarked")):
            refs = gate_refs(placement, rows[placement])
            for i, r in enumerate(rows[placement]):
                ref = rows[placement][refs[i]]
                assert own(ref) and ref["template_id"] == r["template_id"]
                assert ref.get("record_id", ref["pair_id"]) == r.get("record_id", r["pair_id"])
                assert ref.get("order") == r.get("order")
        no_control = [r for r in rows["cross"] if r["cell"] != "unmarked"]
        with pytest.raises(ValueError, match="no unmarked prompt"):
            gate_refs("cross", no_control)


# --------------------------------------------------------------------------- synthetic worlds --------
def _states(rows, e, amp=20.0, noise=0.05, seed=0, dim=32):
    """Pure-noise states plus a planted sex effect per placement along ``e[placement]``: D ±amp/2 by the document's
    sex, X +amp on (pole A, approve) — a decision disparity —, C +amp where the chosen applicant is protected."""
    g = torch.Generator().manual_seed(seed)
    k, pole = CREDIT_DESIGN.axes.index(SEX), CREDIT_DESIGN.factors[SEX][0]
    out = {}
    for placement in PLACEMENTS:
        h = noise * torch.randn(len(rows[placement]), dim, generator=g)
        for i, r in enumerate(rows[placement]):
            if placement == "direct":
                h[i] += amp * (0.5 if r["cell"][k] == pole else -0.5) * e["direct"]
            elif placement == "cross":
                if r["cell"] != "unmarked" and r["cell"][k] == pole and r["response"] == "approve":
                    h[i] += amp * e["cross"]
            elif r["axis"] == SEX and r["protected"] == r["chosen"]:
                h[i] += amp * e["comparative"]
        out[placement] = h
    return out


def _orthonormal_read_by_the_head(w):
    """Three orthonormal directions, each read by the head alike (w·e = |w|/√3): a random 3-D subspace through w,
    rotated by the normalised Helmert matrix."""
    g = torch.Generator().manual_seed(7)
    q, _ = torch.linalg.qr(torch.stack([w / w.norm(), torch.randn(32, generator=g), torch.randn(32, generator=g)], 1))
    q[:, 0] *= torch.sign(q[:, 0] @ w)
    helmert = torch.tensor([[1 / 3 ** .5] * 3, [1 / 2 ** .5, -1 / 2 ** .5, 0.0], [1 / 6 ** .5, 1 / 6 ** .5, -2 / 6 ** .5]])
    return q @ helmert                              # columns e_0, e_1, e_2


def _model():
    from tests.test_run_cross_marker import _model as tiny
    return tiny().float()


def _world(manifest, e_of, n_folds=2):
    from probes.heads import get_head

    model = _model()
    w = get_head(model).effective_weights().reshape(-1)
    pairs, _, rows, _, folds, _ = _built(manifest, n_folds=n_folds)
    e = e_of(w)
    states = _states(rows, e)
    axes = contrast_axes(CREDIT_DESIGN, "explicit")
    fitted, full, split_half = fit_rows(states, rows, axes, CREDIT_DESIGN, "explicit", folds, 42)
    refs = {p: gate_refs(p, rows[p]) for p in PLACEMENTS}
    columns = null_columns(model, states, {p: None for p in PLACEMENTS}, torch.float32, rows, refs, fitted, folds)
    table = matrix(rows, columns, list(fitted), axes, CREDIT_DESIGN, "explicit", sorted(folds), 200, 42)
    return model, rows, states, fitted, folds, columns, table


def test_a_shared_direction_transfers_across_every_placement(manifest):
    _, _, _, _, _, columns, table = _world(manifest, lambda w: {p: w / w.norm() for p in PLACEMENTS})
    assert columns[0] == "baseline" and "gate_fixed" not in columns
    for target in PLACEMENTS:
        cell = table["own_gates"][target][SEX]
        base = cell["baseline"]["mean"]
        assert abs(base) > 1.0 and cell["own_row"] == OWN_ROW[target]
        for row in MATRIX_ROWS:
            c = cell["rows"][row]
            assert abs(c["nulled"]["mean"]) < 0.05 * abs(base), (target, row)
            assert abs(c["gap"]["mean"]) < 0.05 * abs(base), (target, row)


def test_orthogonal_directions_are_removed_only_in_their_own_placement(manifest):
    def orthogonal(w):
        e = _orthonormal_read_by_the_head(w)
        return {p: e[:, k] for k, p in enumerate(PLACEMENTS)}

    _, _, _, _, _, _, table = _world(manifest, orthogonal)
    for target in PLACEMENTS:
        cell = table["own_gates"][target][SEX]
        base = cell["baseline"]["mean"]
        assert abs(base) > 1.0
        for row in MATRIX_ROWS:
            c = cell["rows"][row]
            if row == OWN_ROW[target]:
                assert abs(c["nulled"]["mean"]) < 0.05 * abs(base), (target, row)
            else:
                assert abs(c["change"]["mean"]) < 0.05 * abs(base), (target, row)
                assert c["gap"]["mean"] == pytest.approx(-base, rel=0.1), (target, row)


def test_no_item_is_nulled_by_a_direction_fitted_on_its_record(manifest):
    from probes.comparative_directions import pair_contrasts
    from probes.probe import rewards_from_hidden

    model, rows, states, fitted, folds, _, _ = _world(manifest, lambda w: {p: w / w.norm() for p in PLACEMENTS},
                                                      n_folds=3)
    assert len(set(folds.values())) == 3
    record_fold = {r["record_id"]: folds[r["pair_id"]] for r in rows["cross"]}
    for row, placement, kind in (("d_shared", "direct", "prompt"), ("x_interaction", "cross", "interaction"),
                                 ("x_prompt", "cross", "prompt"), ("c_own", "comparative", None)):
        if kind is None:
            ids, c = pair_contrasts(states[placement], rows[placement], SEX)
            fold_of = folds
        else:
            ids = sorted({r["record_id"] for r in rows[placement]})
            c = cmd.record_contrasts(states[placement], cmd.state_index(rows[placement]), ids, kind, CREDIT_DESIGN,
                                     "explicit", SEX)
            fold_of = record_fold
        for f in set(folds.values()):
            u = cmd.unit(c[[k for k, i in enumerate(ids) if fold_of[i] != f]].mean(0))
            assert torch.allclose(u, fitted[f"{row}:{SEX}"][f], atol=1e-6), (row, f)
            for target in PLACEMENTS:     # every placement's fold-f items are nulled with exactly this direction
                idx = [i for i, r in enumerate(rows[target]) if folds[r["pair_id"]] == f]
                _, rr = rewards_from_hidden(model, states[target][idx], torch.float32, u)
                got = [rows[target][i][f"null:{row}:{SEX}"] for i in idx]
                assert got == pytest.approx(rr.tolist(), abs=1e-5), (row, target, f)


def test_a_probe_direction_is_the_same_in_every_fold(manifest):
    model = _model()
    _, _, rows, _, folds, _ = _built(manifest)
    states = _states(rows, {p: torch.ones(32) / 32 ** .5 for p in PLACEMENTS})
    axes = contrast_axes(CREDIT_DESIGN, "explicit")
    probe = {a: torch.nn.functional.normalize(torch.randn(32), dim=0) for a in axes}
    fitted, full, _ = fit_rows(states, rows, axes, CREDIT_DESIGN, "explicit", folds, 42, probe)
    for a in axes:
        assert all(torch.equal(u, probe[a]) for u in fitted[f"d_probe:{a}"].values())
        assert set(fitted[f"d_probe:{a}"]) == set(folds.values()) and torch.equal(full[f"d_probe:{a}"], probe[a])
    del model


def test_each_target_counts_the_documented_records(manifest):
    # D: both records of every pair; X: the strong records only (weak–weak pairs count 0); C: one value per pair
    pairs, _, rows, _, _, _ = _built(manifest)
    gen = torch.Generator().manual_seed(2)
    for placement in PLACEMENTS:
        for r, v in zip(rows[placement], torch.randn(len(rows[placement]), generator=gen).tolist()):
            r["reward"] = v
    ids = sorted(p.pair_id for p in pairs)
    strong = {"strong_strong": 2, "strong_weak": 1, "weak_weak": 0}
    expect = {"direct": {p.pair_id: 2 for p in pairs}, "cross": {p.pair_id: strong[p.pairing] for p in pairs},
              "comparative": {p.pair_id: 1 for p in pairs}}
    for placement in PLACEMENTS:
        for axis in contrast_axes(CREDIT_DESIGN, "explicit"):
            _, counts = unit_sums(target_values(placement, rows[placement], "reward", axis, CREDIT_DESIGN,
                                                "explicit"), ids)
            assert dict(zip(ids, counts.tolist())) == expect[placement], (placement, axis)


def test_a_target_without_items_warns_and_reads_nan(manifest, caplog):
    _, rows, _, fitted, folds, columns, _ = _world(manifest, lambda w: {p: w / w.norm() for p in PLACEMENTS})
    for r in rows["cross"]:
        r["strong"] = False
    axes = contrast_axes(CREDIT_DESIGN, "explicit")
    with caplog.at_level("WARNING"):
        table = matrix(rows, columns, list(fitted), axes, CREDIT_DESIGN, "explicit", sorted(folds), 20, 42)
    assert "no item counts" in caplog.text
    assert table["own_gates"]["cross"][SEX]["baseline"]["mean"] != table["own_gates"]["cross"][SEX]["baseline"]["mean"]


def test_the_mean_effective_head_weights_each_placement_once():
    from probes import cross_marker_directions as cmd_
    from probes.heads import get_head
    from tests.test_qrm import _model as qrm_model

    model = qrm_model()
    head = get_head(model)
    d, n_gates = model.config.hidden_size, model.num_objectives
    gen = torch.Generator().manual_seed(6)
    gates = {p: torch.softmax(torch.randn(n, n_gates, generator=gen), -1)
             for p, n in (("direct", 3), ("cross", 40), ("comparative", 400))}
    u = torch.nn.functional.normalize(torch.randn(d, generator=gen), dim=0)
    geo = geometry(model, gates, {f"c_own:{SEX}": u}, {f"c_own:{SEX}": 0.5}, [SEX])
    w = torch.stack([head.effective_weights(g).mean(0) for g in gates.values()]).mean(0)
    assert geo["axes"][SEX]["head_alignment"]["c_own"] == cmd_.head_alignment(w, u)
    assert "each placement weighted equally" in geo["head"]


def test_a_gated_head_gets_gate_fixed_columns_from_the_unmarked_prompts(manifest):
    from probes.heads import get_head
    from probes.probe import embed_with_gates
    from tests.test_qrm import _model as qrm_model
    from tests.test_qrm import _tokenizer as qrm_tokenizer

    qtok = qrm_tokenizer()
    model = qrm_model(qtok)
    _, gate_dtype, g1 = embed_with_gates(model, qtok, [format_conversation(qtok, "a", "b")], show_progress=False)
    _, _, rows, _, folds, _ = _built(manifest)
    d = model.config.hidden_size
    gen = torch.Generator().manual_seed(4)
    states = {p: torch.randn(len(rows[p]), d, generator=gen) for p in PLACEMENTS}
    gates = {p: torch.softmax(torch.randn(len(rows[p]), g1.shape[1], generator=gen), -1).to(g1.dtype)
             for p in PLACEMENTS}
    axes = contrast_axes(CREDIT_DESIGN, "explicit")
    fitted, _, _ = fit_rows(states, rows, axes, CREDIT_DESIGN, "explicit", folds, 42)
    refs = {p: gate_refs(p, rows[p]) for p in PLACEMENTS}
    columns = null_columns(model, states, gates, gate_dtype, rows, refs, fitted, folds)
    assert columns[:2] == ["baseline", "gate_fixed"] and f"null_gf:c_own:{SEX}" in columns
    head = get_head(model)
    for placement in ("cross", "comparative"):
        i = next(k for k, r in enumerate(rows[placement]) if refs[placement][k] != k)
        ref = refs[placement][i]
        with torch.no_grad():
            fixed = float(head.score(states[placement][i:i + 1].to(gate_dtype), gates[placement][ref:ref + 1])[0])
            own = float(head.score(states[placement][i:i + 1].to(gate_dtype), gates[placement][i:i + 1])[0])
        assert rows[placement][i]["gate_fixed"] == pytest.approx(fixed, abs=1e-3) and abs(fixed - own) > 1e-3
    # one prompt for every direct row: its gate is fixed already
    assert all(r["gate_fixed"] == r["baseline"] for r in rows["direct"])
    # the gate-fixed nulled columns: the fold's direction projected out, scored with the reference gate
    from probes.probe import rewards_from_hidden
    for placement in PLACEMENTS:
        for i in (0, len(rows[placement]) // 2, len(rows[placement]) - 1):
            r, ref = rows[placement][i], refs[placement][i]
            u = fitted[f"x_interaction:{SEX}"][folds[r["pair_id"]]]
            _, fixed = rewards_from_hidden(model, states[placement][i:i + 1], gate_dtype, u,
                                           gates=gates[placement][ref:ref + 1])
            _, own_gate = rewards_from_hidden(model, states[placement][i:i + 1], gate_dtype, u,
                                              gates=gates[placement][i:i + 1])
            assert r[f"null_gf:x_interaction:{SEX}"] == pytest.approx(float(fixed[0]), abs=1e-5)
            assert r[f"null:x_interaction:{SEX}"] == pytest.approx(float(own_gate[0]), abs=1e-5)
    table = matrix(rows, columns, list(fitted), axes, CREDIT_DESIGN, "explicit", sorted(folds), 20, 42)
    assert list(table) == ["gate_fixed", "own_gates"]          # the gate-fixed reading first
    from scoring.comparative_metrics import pair_values
    effects = pair_values(rows["comparative"], "gate_fixed")[0][(SEX, "merit")]
    assert table["gate_fixed"]["comparative"][SEX]["baseline"]["mean"] == pytest.approx(
        sum(v["marker_effect"] for v in effects.values()) / len(effects))


def test_a_shared_direction_transfers_in_the_gate_fixed_matrix(manifest):
    # a gated head with a different gate per prompt: once the gate is held fixed, the score is linear in the state,
    # so a shared planted direction is removed by every row, as in the linear worlds
    from probes.heads import get_head
    from probes.probe import embed_with_gates
    from tests.test_qrm import _model as qrm_model
    from tests.test_qrm import _tokenizer as qrm_tokenizer

    qtok = qrm_tokenizer()
    model = qrm_model(qtok)
    _, gate_dtype, g1 = embed_with_gates(model, qtok, [format_conversation(qtok, "a", "b")], show_progress=False)
    _, _, rows, _, folds, _ = _built(manifest)
    gen = torch.Generator().manual_seed(8)
    gates = {p: torch.softmax(torch.randn(len(rows[p]), g1.shape[1], generator=gen), -1).to(g1.dtype)
             for p in PLACEMENTS}
    gates["direct"][:] = gates["direct"][0]                   # one assessment prompt: one gate
    head = get_head(model)
    w = torch.stack([head.effective_weights(g).mean(0) for g in gates.values()]).mean(0)
    states = _states(rows, {p: w / w.norm() for p in PLACEMENTS}, dim=model.config.hidden_size)
    axes = contrast_axes(CREDIT_DESIGN, "explicit")
    fitted, _, _ = fit_rows(states, rows, axes, CREDIT_DESIGN, "explicit", folds, 42)
    refs = {p: gate_refs(p, rows[p]) for p in PLACEMENTS}
    columns = null_columns(model, states, gates, gate_dtype, rows, refs, fitted, folds)
    table = matrix(rows, columns, list(fitted), axes, CREDIT_DESIGN, "explicit", sorted(folds), 50, 42)
    for target in PLACEMENTS:
        cell = table["gate_fixed"][target][SEX]
        base = cell["baseline"]["mean"]
        assert abs(base) > 0.5, target
        for row in MATRIX_ROWS:
            assert abs(cell["rows"][row]["nulled"]["mean"]) < 0.05 * abs(base), (target, row)


# --------------------------------------------------------------------------- main, end to end ---------
@pytest.fixture
def run_main(credit_corpus, tmp_path, monkeypatch):
    """`main` of the matrix and of `run_comparative` on a generated credit manifest (60 records: enough pairs for two
    folds; the 16-record fixture manifest gives one pair per pairing, all in fold 0), the tiny Llama RM standing in."""
    import yaml

    from runners import run_placement_matrix as pm
    from tests.test_run_cross_marker import _model as tiny
    from tests.test_run_cross_marker import _tokenizer

    manifest = credit_corpus[0]
    monkeypatch.setenv("ONEJUDGE_EMBED_CACHE", "off")
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "t", "bias_type": "demographic", "model_path": "org/Tiny-RM", "dataset_source": str(manifest),
        "probe_records": 6, "batch_size": 16, "max_length": 1024,
        "extra": {"domain": "credit", "comparative": {"n_pairs": 100, "n_folds": 2, "n_boot": 20,
                                                       "paraphrases": 1}}}))
    loads = []

    def load_model(self):
        loads.append(1)
        self.model, self.tokenizer = tiny(), _tokenizer()
        self.config.model_revision = "abc123"

    from scoring.demographic_experiment import DemographicBiasExperiment
    monkeypatch.setattr(DemographicBiasExperiment, "load_model", load_model)

    def _run(*extra, runner=pm):
        monkeypatch.setattr("sys.argv", ["runner.py", "--config", str(cfg_path), *extra])
        runner.main()

    _run.loads = loads
    _run.manifest = manifest
    stem = "credit" if manifest.parent.name == "credit" else f"credit_{manifest.parent.name}"
    _run.out = lambda prefix="placement", suffix="": (tmp_path / "artifacts/results/demographic"
                                                      / f"{prefix}_{stem}_Tiny-RM{suffix}.json")
    return _run


def test_main_end_to_end(run_main):
    manifest = run_main.manifest
    run_main()
    out = run_main.out()
    summary = json.loads(out.read_text())
    stem = str(out.with_suffix(""))
    assert all(Path(stem + s).exists() for s in ("_rewards.jsonl", "_directions.pt"))
    meta = summary["meta"]
    assert meta["config"]["model_revision"] == "abc123"
    for name in ("pairs.jsonl", "cells.jsonl"):
        assert meta["data"][name]["sha256"] == hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest()
    assert "directions" not in summary["settings"] and summary["settings"]["x_paraphrases"] == 3
    # d_probe's records are never paired; every pair has a fold, every record its pair's
    probe = set(summary["probe_record_ids"])
    paired = {r for p in summary["pairs"] for r in (p["x"], p["y"])}
    assert len(probe) == 6 and not paired & probe and set(summary["folds"]) == {p["pair_id"] for p in summary["pairs"]}
    assert set(summary["matrix"]) == {"explicit", "proxy"}
    for encoding, modes in summary["matrix"].items():
        assert set(modes) == {"own_gates"}
        axes = contrast_axes(CREDIT_DESIGN, encoding)
        for target in PLACEMENTS:
            assert set(modes["own_gates"][target]) == set(axes)
            assert set(modes["own_gates"][target][SEX]["rows"]) == {"d_shared", "x_interaction", "c_own", "d_probe",
                                                                    "x_prompt"}
        assert summary["geometry"][encoding]["axes"][SEX]["rows"] == ["d_shared", "x_interaction", "c_own",
                                                                      "d_probe", "x_prompt"]
    rows = [json.loads(line) for line in open(stem + "_rewards.jsonl")]
    assert len(rows) == sum(summary["n_texts"].values()) and {r["placement"] for r in rows} == set(PLACEMENTS)
    assert not any("text" in r or "prompt" in r for r in rows)
    with pytest.raises(SystemExit, match="exists"):
        run_main()
    assert len(run_main.loads) == 1


def test_the_pairs_are_the_comparative_arms(run_main):
    run_main()
    run_main(runner=rc)
    ours = json.loads(run_main.out().read_text())
    arm = json.loads(run_main.out("comparative").read_text())
    assert ours["pairs"] == arm["pairs"] and ours["probe_record_ids"] == arm["probe_record_ids"]
    assert ours["folds"] == torch.load(str(run_main.out("comparative").with_suffix("")) + "_directions.pt",
                                       weights_only=False)["explicit"]["folds"]


def test_the_pairs_are_the_comparative_arms_under_cli_overrides(run_main, tmp_path):
    flags = ("--n-pairs", "3", "--seed", "7", "--n-folds", "3", "--encodings", "explicit")
    run_main(*flags)
    run_main(*flags, "--directions", "own", runner=rc)
    out = tmp_path / "artifacts/results/demographic"
    ours = json.loads(next(out.glob("placement_*__*.json")).read_text())
    arm_path = next(out.glob("comparative_*__*.json"))
    arm = json.loads(arm_path.read_text())
    assert ours["pairs"] == arm["pairs"] and len(set(ours["folds"].values())) == 3
    assert ours["folds"] == torch.load(str(arm_path.with_suffix("")) + "_directions.pt",
                                       weights_only=False)["explicit"]["folds"]


def test_a_variant_gets_its_own_name(run_main):
    run_main("--x-paraphrases", "1", "--encodings", "explicit")
    assert run_main.out(suffix="__encodings-explicit__x_paraphrases-1").exists()


def test_bad_requests_stop_before_the_model_loads(run_main):
    with pytest.raises(SystemExit, match="x_paraphrases"):
        run_main("--x-paraphrases", "9")
    with pytest.raises(SystemExit, match="not in cells.jsonl"):
        run_main("--encodings", "explicit,phonetic")
    with pytest.raises(SystemExit, match="n_folds"):
        run_main("--n-folds", "1")
    assert run_main.loads == []


def test_cli_parses():
    args = build_parser().parse_args(["--n-pairs", "4", "--x-paraphrases", "2", "--model", "M"])
    assert args.n_pairs == "4" and args.x_paraphrases == 2 and args.model == "M"
