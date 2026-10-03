#!/usr/bin/env python3
"""
RQ4's accuracy guardrail: does removing the demographic directions (or applying the attributes' LEACE maps) cost the
reward model general accuracy? Out of domain, on RewardBench 2 (`scoring.rewardbench`; design and decisions: working
notes 2026-10-02, review 2026-10-03). In domain, the cross-marker design already tracks the collateral (the AUC of D
after each nulling).

**The benchmark** is embedded once (each RM's own chat template, as RewardBench formats it and as every arm here
does; the final-token states and, for QRM, the gates; in order of length, so batches pad little), at the config's
``max_length``; a row with a longer completion is left out of the baseline and every edit alike (a Ties question with
both its rows), counted per subset. Every reward here is the score head applied in **float32** to the state as the
model holds it (bf16): the bf16 head of the other arms rounds the rewards to few distinct values (Qwen3-0.6B: 1,687 for
8,977 completions), which inflates ties at the top and Ties' zero spreads (decided 2026-10-03; `head_rewards`).

**The edits**, from the direct arm of every domain in ``extra.direct_configs`` (their manifests, probe split and
max_length; the texts are the battery's, so embedding-cache hits after it):
  - exploratory (intervals, no pass flags): ``{domain}/{encoding}/{axis}/diffmean`` — each direct-arm direction, the
    battery's (`run_cross_marker.direct_directions`), projected out; ``{domain}/{encoding}/{axis}/leace`` — the LEACE
    map of each attribute, fitted on the probe records' cell states (`run_demographic_erasure`'s training states).
    Applied to RewardBench states it is the shipped map out of domain, not an erasure of anything there: it moves a
    state along the attribute's direction by a whitened reading centred on the direct-arm mean;
  - **confirmatory**: ``{domain}/joint@α`` — every single-axis direction of the domain, both encodings, projected out
    together (the corner direction is left out: it lies in their span up to its three-way part, whose size
    ``edit_meta[...]["corner_outside_span"]`` records), for every α of ``extra.alphas`` — the edit a deployer would
    ship; and ``all/joint@α`` over every domain;
  - reference: ``{family}/random{k}`` — ``extra.random_draws`` random k-dimensional subspaces (k = the joint basis's
    rank) projected out at α = 1: whether a joint edit's loss comes from its directions or from removing k dimensions.

**The reading.** Nothing is read unless the run is ``readable``: the whole benchmark (no ``--max-items``), and the
unedited model reproduces the leaderboard (``configs/rewardbench2_published.yaml``: the overall within
``extra.reproduction_tolerance``, and, where the leaderboard publishes per-completion scores, a correlation of at least
``extra.completion_r_min`` with them). Otherwise every pass flag and claim is None. Then, per edit, the paired change of
every subset and of the overall (edit − baseline), bootstrapped over rows stratified by subset (the same draws for
every edit). A joint edit is **non-inferior** if the one-sided lower bound of its overall change is above −δ
(``extra.delta``, 2 points) and that of Safety above −δ_safety (``extra.delta_safety``, 3 points; decided
2026-10-03: the overall alone lets one subset lose ~12 points). The base paper's 5 points on the overall is reported as
``non_inferior_reference``. The claims (``claims``; decided 2026-10-03):
  - ``every_family_at_full_strength``: every family's joint edit at the largest α passes at level 0.05 — one
    conjunctive claim, each part on its own (an intersection–union test: no correction);
  - per family, ``largest_alpha``: the α sweep read as a fixed sequence (smallest α first, stopping at the first
    fail), at level 0.05 / (number of families) — Bonferroni across the families, none within one.

Outputs: ``rewardbench_guardrail_{model}{variant}.json`` (``meta``: the config with the loaded model commit, the code
commit, every direct manifest's hashes and config, the benchmark's revision and hash, the settings; never replaced
without ``--overwrite``) and ``…_baseline_scores.jsonl`` (per row: position, id, subset and the unedited completion
rewards, numbers only). No benchmark text is written.

Usage:
    python runners/run_rewardbench_guardrail.py --config configs/rewardbench2_guardrail_qwen06.yaml
    python runners/run_rewardbench_guardrail.py --config configs/rewardbench2_guardrail_qwen06.yaml \\
        --max-items 20    # smoke: not readable
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from pairs.cross_marker import load_cell_blocks
from pairs.factorial import ENCODINGS
from probes.erasure import leace_erase
from probes.heads import get_head, score_saved
from probes.probe import embed_states, embed_with_gates, gram_schmidt, project_to_null_space
from runners.run_comparative import probe_split_ids
from runners.run_cross_marker import cells_path, direct_directions, token_counter
from runners.run_demographic_erasure import (
    States, battery_split, check_names, concept_label, concepts, name_folds, select_records,
)
from scoring import rewardbench as rb
from scoring.dataset_base import format_conversation
from scoring.experiment import (
    ExperimentConfig, add_override_args, apply_overrides, data_file, run_metadata, variant_suffix,
)
from scoring.intervals import DEFAULT_N_BOOT
from substrates.domains import get_domain

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("artifacts/results/demographic")
PUBLISHED = PROJECT_ROOT / "configs" / "rewardbench2_published.yaml"
LEVEL = 0.05                           # the one-sided level of every non-inferiority test
SAFETY = "Safety"


# --------------------------------------------------------------------------- inputs ---------------------
def subsample(items: Sequence[rb.Item], max_items: Optional[int]) -> List[rb.Item]:
    """The first ``max_items`` rows of every best-of-n subset and the first ``max_items`` Ties questions (both
    rows), in benchmark order (a smoke run; None = everything)."""
    if max_items is None:
        return list(items)
    keep_q = sorted({it.question for it in items if it.subset == rb.TIES})[:max_items]
    seen: Dict[str, int] = {}
    out = []
    for it in items:
        if it.subset == rb.TIES:
            if it.question in keep_q:
                out.append(it)
        elif seen.get(it.subset, 0) < max_items:
            seen[it.subset] = seen.get(it.subset, 0) + 1
            out.append(it)
    return out


def published_for(model_path: str) -> Dict[str, Any]:
    """The model's published subset scores and, where the leaderboard has them, its per-completion scores (from the
    HF cache, at the table's revision; `cluster/prefetch_models.py` fetches them). Checked before the model loads: a
    malformed table or a listed but unavailable file stops the run here, not after the forward pass."""
    table = yaml.safe_load(PUBLISHED.read_text())
    for key in ("source", "revision", "scores"):
        if key not in table:
            raise SystemExit(f"{PUBLISHED.name}: no {key!r}")
    for model, scores in table["scores"].items():
        missing = [s for s in rb.SUBSETS if s not in scores]
        if missing:
            raise SystemExit(f"{PUBLISHED.name}: {model} lacks {missing}")
    out: Dict[str, Any] = {"source": table["source"], "revision": table["revision"],
                           "scores": table["scores"].get(model_path), "completions": None}
    if model_path in (table.get("per_completion") or []):
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(table["source"], f"eval-set-scores/{model_path}.json", repo_type="dataset",
                                   revision=table["revision"])
        except Exception as e:  # noqa: BLE001 - offline without the file, or gone: the gate cannot be applied
            raise SystemExit(f"the leaderboard's per-completion scores of {model_path} are listed in "
                             f"{PUBLISHED.name} but could not be read ({type(e).__name__}: {e}); prefetch them")
        out["completions"] = rb.published_completion_scores(Path(path))
    return out


class Domain:
    """One direct config's domain: its spec, config, manifest and the probe records' blocks (checked before the
    model loads)."""

    def __init__(self, path: Path, probe_records: Optional[int]):
        self.cfg = ExperimentConfig.from_yaml(path)
        if probe_records is not None:
            self.cfg.probe_records = probe_records
        self.dom = get_domain(self.cfg.extra.get("domain", "credit"))
        if self.dom.factorial is None:
            raise SystemExit(f"{path}: the guardrail's edits come from a factorial domain, not {self.dom.name!r}")
        self.source = self.cfg.dataset_source or self.dom.default_pairs
        cells = cells_path(self.source, self.dom.default_pairs)
        self.data = {f"{self.dom.name}/pairs.jsonl": data_file(self.source),
                     f"{self.dom.name}/cells.jsonl": data_file(cells)}
        blocks = load_cell_blocks(cells, self.dom.factorial)
        self.encodings = [e for e in ENCODINGS if any(b.encoding == e for b in blocks)]
        probe_ids = probe_split_ids(self.dom, self.source, self.cfg.probe_records, self.cfg.split_seed)
        self.train, _, self.selection = select_records(blocks, self.encodings, probe_ids, 0, self.cfg.split_seed)
        if not self.train:
            raise SystemExit(f"{self.dom.name}: no probe record with a block in {cells}")
        check_names(self.train, {}, self.dom.factorial, name_folds(self.dom.name, 2, self.cfg.split_seed))
        # are the LEACE maps' training records the battery's probe split (as the diffmean directions' are)?
        self.selection["battery_split"] = battery_split(self.dom, self.source, self.cfg, self.encodings, self.train)


# --------------------------------------------------------------------------- the edits --------------------
Edit = Tuple[str, Any, float]          # ("project", basis, alpha), ("erase", map, 1.0) or ("none", None, 0.0)


def outside_span(v: torch.Tensor, basis: torch.Tensor) -> float:
    """The share of unit vector ``v``'s norm outside the span of ``basis``'s rows."""
    q = gram_schmidt([b for b in basis])
    u = v.float() / v.float().norm()
    return float((u - (u @ q.T) @ q).norm())


def domain_edits(exp: Any, cfg: Any, d: Domain, alphas: Sequence[float]) -> Tuple[Dict[str, Edit], List[torch.Tensor],
                                                                                  Dict[str, Any]]:
    """The domain's edits: every direct-arm direction, every attribute's LEACE map, the joint projections; and
    the single-axis directions (for the joint over every domain) and metadata."""
    design = d.dom.factorial
    dirs, _, meta = direct_directions(exp.model, exp.tokenizer, d.dom, d.source, d.encodings,
                                      probe_records=d.cfg.probe_records, split_seed=d.cfg.split_seed,
                                      batch_size=cfg.batch_size, max_length=d.cfg.max_length)
    edits: Dict[str, Edit] = {}
    singles: List[torch.Tensor] = []
    for enc in d.encodings:
        for axis, direction in dirs.get(enc, {}).items():
            edits[f"{d.dom.name}/{enc}/{axis}/diffmean"] = ("project", direction, 1.0)
            if axis != "intersection":
                singles.append(direction)
        states = States.from_blocks(d.train, enc, design, d.dom.dataset_cls.DEFAULT_PROMPT, exp.tokenizer)
        H, _ = embed_states(exp.model, exp.tokenizer, states.texts, batch_size=cfg.batch_size,
                            max_length=d.cfg.max_length, show_progress=False)
        for concept, factors in concepts(design, enc):
            if len(factors) == 1:
                labels = [concept_label(design, cell, factors) for cell in states.cells]
                edits[f"{d.dom.name}/{enc}/{concept}/leace"] = ("erase", leace_erase(H, labels), 1.0)
    basis = torch.stack(singles)
    for a in alphas:
        edits[f"{d.dom.name}/joint@{a:g}"] = ("project", basis, float(a))
    corners = {enc: outside_span(dirs[enc]["intersection"], basis) for enc in d.encodings
               if "intersection" in dirs.get(enc, {})}
    return edits, singles, {"directions": meta, "joint_vectors": len(singles), "corner_outside_span": corners}


def random_bases(d: int, k: int, draws: int, seed: int) -> List[torch.Tensor]:
    """``draws`` random k-dimensional subspaces of R^d (orthonormal rows, seeded)."""
    g = torch.Generator().manual_seed(seed)
    return [torch.linalg.qr(torch.randn(d, k, generator=g))[0].T.contiguous() for _ in range(draws)]


def head_rewards(saved: Dict[str, Any], X: torch.Tensor, dtype: Any, gates: Optional[torch.Tensor]) -> np.ndarray:
    """The score head in float32 on states ``X`` rounded to the model's dtype (the state as the model holds it):
    `probes.heads.score_saved`, the same kernels as the online heads, in the state's dtype."""
    h = X.to(dtype).float()
    return score_saved(saved, h, None if gates is None else gates.float()).reshape(-1).float().numpy()


def edited_rewards(saved: Dict[str, Any], H: torch.Tensor, dtype: Any, gates: Optional[torch.Tensor],
                   edit: Edit) -> np.ndarray:
    """The benchmark's rewards under one edit: the head on the projected (strength α) or mapped states."""
    kind, what, alpha = edit
    if kind == "none":
        X = H.float()
    elif kind == "project":
        X = project_to_null_space(H.float(), what, alpha=alpha)
    elif kind == "erase":
        X = what(H.float())
    else:
        raise ValueError(f"unknown edit kind {kind!r}")
    return head_rewards(saved, X, dtype, gates)


# --------------------------------------------------------------------------- the reading ------------------
def reproduction(published: Dict[str, Any], scores: Dict[str, float], agreement: Optional[Dict[str, Any]],
                 tolerance: float, r_min: float, subsampled: bool) -> Dict[str, Any]:
    """The unedited scores against the published ones: per subset and overall difference, the per-completion
    correlation where the leaderboard has the scores, and ``reproduced`` (None where nothing to compare: no published
    entry, or a subsample)."""
    if subsampled:
        return {"reproduced": None, "note": "a --max-items subsample is not comparable with the published scores"}
    pub = published["scores"]
    if pub is None:
        return {"reproduced": None, "note": "no published RewardBench 2 entry for this model"}
    pub = {**{s: float(pub[s]) for s in rb.SUBSETS}, "overall": float(np.mean([pub[s] for s in rb.SUBSETS]))}
    diff = {s: float(scores[s] - pub[s]) for s in pub if s in scores}
    ok = abs(diff["overall"]) <= tolerance
    out = {"published": pub, "source": published["source"], "revision": published["revision"],
           "ours_minus_published": diff, "tolerance": tolerance, "completions": agreement, "r_min": r_min}
    if agreement is not None:
        ok = ok and agreement["r"] >= r_min and not agreement["unmatched_rows"]
    return {**out, "reproduced": bool(ok)}


def decide(rows: Dict[str, Any], level: str, delta: float, delta_safety: float) -> bool:
    """Non-inferior at one-sided ``level``: the overall's lower bound above −δ and Safety's above −δ_safety."""
    return bool(rows["overall"]["lower_bound"][level] > -delta and rows[SAFETY]["lower_bound"][level] > -delta_safety)


def fixed_sequence(passes: Sequence[Tuple[float, bool]]) -> Optional[float]:
    """The largest α of a sweep tested in order of increasing α, stopping at the first fail (None if the smallest
    fails): a fixed-sequence test, no correction within the sweep."""
    largest = None
    for alpha, ok in sorted(passes):
        if not ok:
            break
        largest = alpha
    return largest


def claims(results: Dict[str, Any], families: Sequence[str], alphas: Sequence[float], delta: float,
           delta_safety: float) -> Dict[str, Any]:
    """The confirmatory claims (module docstring): the conjunctive full-strength claim, and per family the largest
    α of its fixed sequence at the Bonferroni level."""
    level_f = f"{LEVEL / len(families):g}"
    top = max(alphas)
    out: Dict[str, Any] = {
        "every_family_at_full_strength": all(results[f"{f}/joint@{top:g}"]["non_inferior"] for f in families),
        "family_level": float(level_f), "families": {}}
    for f in families:
        seq = [(a, decide(results[f"{f}/joint@{a:g}"], level_f, delta, delta_safety)) for a in alphas]
        out["families"][f] = {"passes": {f"{a:g}": ok for a, ok in sorted(seq)}, "largest_alpha": fixed_sequence(seq)}
    return out


# --------------------------------------------------------------------------- CLI ------------------------
def default_out(model_path: str, variant: str = "") -> Path:
    return RESULTS_DIR / f"rewardbench_guardrail_{Path(model_path).name}{variant}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/rewardbench2_guardrail_qwen06.yaml"))
    ap.add_argument("--max-items", type=int, default=None,
                    help="Rows per best-of-n subset and Ties questions (a smoke run, not readable); default all")
    ap.add_argument("--n-boot", type=int, default=None, help=f"Default extra.n_boot, else {DEFAULT_N_BOOT}")
    ap.add_argument("--seed", type=int, default=42, help="Seeds the bootstrap and the random subspaces")
    ap.add_argument("--dataset-revision", default=None, help="Overrides extra.dataset_revision")
    ap.add_argument("--out", type=Path, default=None, help=f"Default {RESULTS_DIR}/rewardbench_guardrail_{{model}}"
                                                           f"{{variant}}.json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result")
    add_override_args(ap)
    return ap


def _ci(row: Dict[str, float]) -> str:
    return f"{100 * row['change']:+6.2f} [{100 * row['ci_low']:+6.2f}, {100 * row['ci_high']:+6.2f}]"


def _flag(value: Optional[bool]) -> str:
    return "—" if value is None else ("pass" if value else "FAIL")


def print_report(result: Dict[str, Any]) -> None:
    base, rep = result["baseline"], result["reproduction"]
    print("\n" + "=" * 132)
    print(f"REWARDBENCH 2 GUARDRAIL — {result['model']}  ({result['n_rows']} rows; excluded as too long: "
          f"{sum(result['excluded'].values())})")
    print("=" * 132)
    print("baseline:  " + "  ".join(f"{s} {100 * v:.1f}" for s, v in base.items()))
    if rep["reproduced"] is None:
        print(f"reproduction: {rep['note']}")
    else:
        agree = rep["completions"]
        print("published: " + "  ".join(f"{s} {100 * v:.1f}" for s, v in rep["published"].items())
              + (f"   per completion r = {agree['r']:.5f}" if agree else "")
              + f"   ⇒ {'REPRODUCED' if rep['reproduced'] else 'NOT REPRODUCED'}")
    print(f"readable: {result['readable']}" + ("" if result["readable"] else "  — no pass flag or claim is read"))
    print(f"\n{'edit':46} {'role':12} {'overall':>8} {'change, points [95% CI]':>26} {'lower bd':>9} "
          f"{'Safety Δ':>9} {'NI':>5} {'NI@5':>5}  worst subset")
    for name, e in result["edits"].items():
        o = e["overall"]
        worst = min((s for s in rb.SUBSETS if s in e), key=lambda s: e[s]["change"])
        print(f"{name:46} {e['role']:12} {100 * o['score']:>8.2f} {_ci(o):>26} {100 * o['lower_bound_95']:>+9.2f} "
              f"{100 * e[SAFETY]['change']:>+9.2f} {_flag(e.get('non_inferior')):>5} "
              f"{_flag(e.get('non_inferior_reference')):>5}  {worst} {100 * e[worst]['change']:+.2f}")
    c = result["claims"]
    if c is not None:
        print(f"\nevery family at full strength non-inferior: {c['every_family_at_full_strength']}")
        for f, v in c["families"].items():
            print(f"  {f}: largest α passing (fixed sequence, level {c['family_level']:g}): {v['largest_alpha']}")
    print("Non-inferior (confirmatory joint edits only): the one-sided lower bound of the overall change above −δ and "
          "Safety's above −δ_safety; NI@5: the base paper's 5 points on the overall.")


def main() -> None:
    t0 = time.perf_counter()
    ap = build_parser()
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    configured_cfg = ExperimentConfig.from_yaml(args.config)
    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    extra = cfg.extra
    alphas = sorted(float(a) for a in extra.get("alphas", [1.0]))
    delta, delta_ref = float(extra.get("delta", 0.02)), float(extra.get("delta_reference", 0.05))
    delta_safety = float(extra.get("delta_safety", 0.03))
    tolerance, r_min = float(extra.get("reproduction_tolerance", 0.01)), float(extra.get("completion_r_min", 0.99))
    random_draws = int(extra.get("random_draws", 5))
    n_boot = args.n_boot if args.n_boot is not None else int(extra.get("n_boot", DEFAULT_N_BOOT))
    revision = args.dataset_revision or extra.get("dataset_revision")
    if args.max_items is not None and args.max_items < 2:
        raise SystemExit("--max-items must be at least 2")
    keys = ("max_items", "seed")
    configured = {**{k: ap.get_default(k) for k in keys}, "n_boot": int(configured_cfg.extra.get("n_boot",
                                                                                                    DEFAULT_N_BOOT)),
                  "dataset_revision": configured_cfg.extra.get("dataset_revision"),
                  "probe_records": None, "revision": configured_cfg.model_revision}
    used = {**{k: getattr(args, k) for k in keys}, "n_boot": n_boot, "dataset_revision": revision,
            "probe_records": args.probe_records, "revision": cfg.model_revision}
    out = args.out or default_out(cfg.model_path, variant_suffix(configured, used))
    # everything that can fail on the inputs fails here, before the model loads
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or --out")
    direct = extra.get("direct_configs") or []
    if not direct:
        raise SystemExit("extra.direct_configs names no direct config: there would be no edit")
    domains = [Domain(Path(p), args.probe_records) for p in direct]
    if len({d.dom.name for d in domains}) != len(domains):
        raise SystemExit("extra.direct_configs names a domain twice")
    published = published_for(cfg.model_path)
    items, info = rb.load(revision)
    if revision and info["revision"] != revision:
        raise SystemExit(f"asked for RewardBench 2 revision {revision}, got {info['revision']}")
    items = subsample(items, args.max_items)

    from scoring.demographic_experiment import DemographicBiasExperiment

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    saved = get_head(exp.model).to_saved()
    timer = {"load_model": time.perf_counter() - t0}

    # the benchmark: lengths, then one forward pass in order of length
    start = time.perf_counter()
    convs = [[format_conversation(exp.tokenizer, it.prompt, c) for c in it.completions] for it in items]
    count = token_counter(exp.tokenizer)
    lengths = {it.row: count.batch(cs) for it, cs in zip(items, convs)}
    too_long = {row for row, ls in lengths.items() if max(ls) > cfg.max_length}
    kept = rb.drop_questions_with(items, too_long)
    excluded = {s: sum(1 for it in items if it.subset == s) - sum(1 for it in kept if it.subset == s)
                for s in rb.SUBSETS}
    if any(excluded.values()):
        logger.warning("rows left out as longer than max_length %d: %s", cfg.max_length, excluded)
    by_row = {it.row: cs for it, cs in zip(items, convs)}
    texts = [c for it in kept for c in by_row[it.row]]
    order = np.argsort([n for it in kept for n in lengths[it.row]], kind="stable")
    Hs, dtype, Gs = embed_with_gates(exp.model, exp.tokenizer, [texts[i] for i in order], batch_size=cfg.batch_size,
                                     max_length=cfg.max_length, show_progress=False)
    back = torch.as_tensor(np.argsort(order, kind="stable"))
    H, gates = Hs[back], None if Gs is None else Gs[back]
    offsets = rb.offsets_of(kept)
    base_rewards = edited_rewards(saved, H, dtype, gates, ("none", None, 0.0))
    base = rb.Scored(kept, base_rewards, offsets)
    baseline = {s: float(v) for s, v in base.scores().items()}
    timer["benchmark"] = time.perf_counter() - start

    # the edits
    start = time.perf_counter()
    edits: Dict[str, Tuple[str, Edit]] = {}
    families: List[str] = []
    bases: Dict[str, torch.Tensor] = {}
    singles: List[torch.Tensor] = []
    edit_meta: Dict[str, Any] = {}
    for d in domains:
        e, s, m = domain_edits(exp, cfg, d, alphas)
        for name, edit in e.items():
            edits[name] = ("confirmatory" if "/joint@" in name else "exploratory", edit)
        singles += s
        families.append(d.dom.name)
        bases[d.dom.name] = torch.stack(s)
        edit_meta[d.dom.name] = m
    if len(domains) > 1:
        families.append("all")
        bases["all"] = torch.stack(singles)
        for a in alphas:
            edits[f"all/joint@{a:g}"] = ("confirmatory", ("project", bases["all"], float(a)))
    for j, f in enumerate(families):
        k = gram_schmidt([b for b in bases[f]]).shape[0]
        for i, q in enumerate(random_bases(H.shape[1], k, random_draws, args.seed + 1000 * j)):
            edits[f"{f}/random{k}#{i}"] = ("reference", ("project", q, 1.0))
    timer["edits"] = time.perf_counter() - start

    # the reading
    start = time.perf_counter()
    agreement = (None if published["completions"] is None or args.max_items is not None else
                 rb.completion_agreement(kept, base_rewards, offsets, published["completions"]))
    rep = reproduction(published, baseline, agreement, tolerance, r_min, args.max_items is not None)
    readable = bool(rep["reproduced"]) and args.max_items is None
    level_f = f"{LEVEL / len(families):g}"
    draws = rb.draws_for(base, n_boot, args.seed)
    results: Dict[str, Any] = {}
    for name, (role, edit) in edits.items():
        scored = rb.Scored(kept, edited_rewards(saved, H, dtype, gates, edit), offsets)
        rows = rb.paired_bootstrap(base, scored, draws, levels=(LEVEL, float(level_f)))
        results[name] = {"role": role, **rows}
        if role == "confirmatory":
            ni = decide(rows, f"{LEVEL:g}", delta, delta_safety)
            results[name].update({"non_inferior": ni if readable else None,
                                  "non_inferior_reference": (bool(rows["overall"]["lower_bound_95"] > -delta_ref)
                                                             if readable else None)})
    timer["bootstrap"] = time.perf_counter() - start
    confirm = None
    if readable:
        raw = {n: {**r, "non_inferior": decide(r, f"{LEVEL:g}", delta, delta_safety)}
               for n, r in results.items() if r["role"] == "confirmatory"}
        confirm = claims(raw, families, alphas, delta, delta_safety)

    result = {"model": cfg.model_path, "n_rows": len(kept), "excluded": excluded, "baseline": baseline,
              "reproduction": rep, "readable": readable, "delta": delta, "delta_safety": delta_safety,
              "delta_reference": delta_ref, "claims": confirm, "edits": results, "edit_meta": edit_meta,
              "probe_records": {d.dom.name: sorted(d.train) for d in domains},
              "selection": {d.dom.name: d.selection for d in domains}}
    print_report(result)
    timer["total"] = time.perf_counter() - t0
    print("timing: " + " | ".join(f"{k} {v:.0f}s" for k, v in timer.items()))
    data = {"rewardbench2": info}
    for d in domains:
        data.update(d.data)
    settings = {"max_items": args.max_items, "n_boot": n_boot, "seed": args.seed, "alphas": alphas, "delta": delta,
                "delta_safety": delta_safety, "delta_reference": delta_ref, "level": LEVEL,
                "reproduction_tolerance": tolerance, "completion_r_min": r_min, "random_draws": random_draws,
                "head": "float32 on bf16-rounded states", "direct_configs": [str(p) for p in direct],
                "direct_config_settings": {d.dom.name: d.cfg.to_dict() for d in domains}}
    out.parent.mkdir(parents=True, exist_ok=True)
    side = out.with_name(out.stem + "_baseline_scores.jsonl")
    with open(side, "w") as f:
        for it, start_at in zip(kept, offsets):
            f.write(json.dumps({"row": it.row, "id": it.id, "subset": it.subset,
                                "scores": base_rewards[start_at:start_at + len(it.completions)].tolist()}) + "\n")
    out.write_text(json.dumps({"meta": run_metadata(cfg, data, settings), "timing": timer, **result}, indent=2))
    print(f"saved → {out} (+ {side.name})")


if __name__ == "__main__":
    main()
