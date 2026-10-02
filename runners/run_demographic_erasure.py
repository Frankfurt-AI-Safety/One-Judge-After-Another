#!/usr/bin/env python3
"""
LEACE + non-linear-probe test on the **demographic** attributes (RQ3; design and decisions: working notes
2026-10-02): is a protected attribute *low-complexity* in the RM's final-token state, or only *linearly* erasable
while still **non-linearly recoverable** (high-complexity / entangled — the TaCo signature)? And, on the same states,
does the erasure remove the attribute's **reward gap** (the reward-scalar level), so that a row where the reward gap
is gone while an MLP still recovers the attribute shows the reward-vs-representation gap directly?

States: the direct arm's (the marker in the response; the 2026-09-24 decision kept that arm for RQ3). Per encoding,
every block (record × template) of the manifest's ``cells.jsonl`` (next to the config's ``dataset_source``; both
files checked against their manifest before the model loads, `scoring.experiment.data_file`): its 8 factorial cells,
formatted as the direct arm formats them (the domain's assessment prompt, the document as the response). Training:
the battery's own probe records (``probe_records``, the same split: `runners.run_comparative.probe_split_ids`), so
with explicit markers the ``diffmean`` row is exactly the battery's direction. Evaluation: ``--eval-records`` records
outside every direct probe split, with a block for every requested encoding and template, in seeded order.

Concepts per encoding (`concepts`): each axis (proxy encoding: the axes with a proxy), labelled 1 at pole A (the
battery's side A), and each pair of them as an **interaction**, labelled 1 where both factors sit at the same pole
(XNOR; orthogonal to the main effects in the balanced factorial, so a linear probe reading it means the states carry
an interaction component of their own).

Rows, each a held-out **linear vs MLP probe** (`probes.erasure.probe_recoverability`; the MLP a majority vote of three
seeds), standardised on the training states:
  - ``none``        — no erasure (decodable at all?);
  - ``diffmean``    — the concept's difference-of-means direction projected out (our method);
  - ``leace``       — LEACE on the concept alone;
  - ``leace_cells`` — LEACE on all 7 contrasts of the 8 cells jointly (`cell_columns`: the three axes, the three
                      two-way and the three-way products), after which no linear function separates any cell.
Reading (``verdict``, `concept_verdict`, one-sided on the interval of ``mlp_above_chance``), only where the
``none`` row's linear probe is above chance. An axis: not recoverable after ``leace`` ⇒ low-complexity; recoverable
after ``leace_cells`` too ⇒ entangled, high-complexity; recoverable after ``leace`` but not after ``leace_cells`` ⇒ the
MLP used a linear interaction term LEACE on the axis alone leaves in place: intersectional, not high-complexity (where
an interaction of the axis is linearly decodable; else unresolved). An interaction: the ``leace_cells`` row (its own
``leace`` row is not read: with both main effects linear in the states, an MLP rebuilds the XNOR from them, as the
lexical control shows on the words alone). Chance is each fold's majority share (`probes.erasure.pool_folds`).

**Held-out names (proxy).** A class of proxy states is a union of first names (`NAME_POOLS`), and after LEACE a
non-linear probe can map each name it was trained on to its label — the artifact the reasoning test hit with its
phrasings. Proxy rows are therefore cross-fitted over ``--name-folds`` folds (`name_folds`): each fold holds out the
same share of every pool, trains on the training states without those names and evaluates the eval states that carry
them; pooled, every eval state is scored once (`probes.erasure.pool_folds`). Their intervals resample records and
names crossed (`scoring.intervals.crossed_bootstrap`: ~2 held-out names per pool and fold carry the generalisation, so
a record bootstrap alone would treat the names as fixed), and every row reports ``by_name`` accuracies. Explicit
markers are one wording per pole: a single fit, a record bootstrap.

**Lexical control** (``lexical_control``): the same pipeline on bag-of-words vectors of each state's marker clause
(`probes.erasure.bag_of_words`, digits counted as words; the clause is the only text that differs between a block's
cells, and the record's content is constant within a record and balanced over every label). With explicit markers an
axis is at chance after LEACE by construction (one word per pole), an interaction not (an MLP computes it from the
words); with held-out names the eval names are outside its vocabulary. ``lexical_recovers_in_read_row`` flags a
concept whose words alone are recoverable in the row its verdict reads.

**Reward gap** (``reward_gap``): per eval record the mean reward of its label-1 states minus its label-0 states (for an
interaction half the difference-in-differences), for every row: the head on the state as is, after the diffmean
projection and after each LEACE, every erasure fitted once on all training states (`reward_rows`; not per name fold;
a gated head, QRM, with the state's own gate, which every cell of a record shares here since the prompt does not
vary). Each with its record-bootstrap interval and the paired change from ``none`` (``<row>_minus_none``).

Cost: the training states are the battery's probe texts (embedding-cache hits after the battery); most eval texts
are not (the battery evaluates 200 pairs), so expect a forward pass of ≈200 × templates × 8 texts per encoding.

The result ``erasure_demographic_{domain}[_{manifest folder}]_{model}{variant}.json`` (``variant`` names every setting
the CLI changed; never replaced without ``--overwrite``) carries ``meta`` (`scoring.experiment.run_metadata`: the
config with the loaded model commit, the code commit, both data files' SHA-256, the settings), both splits' record ids
and the name folds; no text.

Usage:
    python runners/run_demographic_erasure.py --config configs/demographic_credit_sex_qwen06.yaml
    python runners/run_demographic_erasure.py --config configs/demographic_cv_sex_qwen06.yaml --encodings proxy
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from pairs.cross_marker import CellBlock, load_cell_blocks
from pairs.factorial import ENCODINGS, Cell, FactorialDesign, stable_rng
from pairs.markers import BLACK_FEMALE_NAMES, BLACK_MALE_NAMES, FEMALE_NAMES, MALE_NAMES
from probes.erasure import (
    apply_diffmean, apply_eraser, bag_of_words, diffmean_direction, leace_erase, pool_folds,
    probe_recoverability,
)
from probes.probe import embed_with_gates, rewards_from_hidden
from runners.run_comparative import probe_split_ids
from runners.run_cross_marker import cells_path
from scoring.dataset_base import format_conversation
from scoring.experiment import (
    ExperimentConfig, add_override_args, apply_overrides, data_file, run_metadata, variant_suffix,
)
from scoring.intervals import DEFAULT_N_BOOT, cluster_bootstrap
from substrates.domains import get_domain

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("artifacts/results/demographic")
METHODS = ("none", "diffmean", "leace", "leace_cells")
MLP_SEEDS = 3                    # the MLP's majority vote over three seeds (`probes.erasure.probe_recoverability`)
WORDS = r"[A-Za-z0-9']+"         # the lexical control's words: digits too (ages, birth years)
# The first-name pools a domain's proxy clauses draw from (`pairs.factorial.ProxyNames`): credit and hiring a
# white-coded female/male pair, education one name per sex × ethnicity cell. Every name of a pool is held out in
# exactly one name fold.
NAME_POOLS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "credit": (tuple(FEMALE_NAMES), tuple(MALE_NAMES)),
    "cv": (tuple(FEMALE_NAMES), tuple(MALE_NAMES)),
    "education": (tuple(FEMALE_NAMES), tuple(MALE_NAMES), tuple(BLACK_FEMALE_NAMES), tuple(BLACK_MALE_NAMES)),
}


# --------------------------------------------------------------------------- labels ---------------------
def concepts(design: FactorialDesign, encoding: str) -> List[Tuple[str, Tuple[str, ...]]]:
    """(name, factors) per concept: every axis of the encoding (proxy: the axes with a proxy), then every pair of
    them as an interaction, named ``{a}_x_{b}``."""
    axes = design.axes if encoding == "explicit" else design.proxy_axes
    return [(a, (a,)) for a in axes] + [(f"{a}_x_{b}", (a, b)) for a, b in combinations(axes, 2)]


def concept_label(design: FactorialDesign, cell: Cell, factors: Sequence[str]) -> int:
    """1 where an even number of ``factors`` sit at pole B: one factor at pole A; two at the same pole (XNOR); the
    parity of three. In ±1 coding (pole A = +1) the label is 1 exactly where the product of the factors is +1."""
    at_b = sum(cell[design.axes.index(f)] != design.factors[f][0] for f in factors)
    return int(at_b % 2 == 0)


def cell_columns(design: FactorialDesign, cells: Sequence[Cell]) -> List[List[int]]:
    """The 7 label columns of a 2×2×2 factorial (each axis, each two-way and the three-way product), which with the
    constant span every function of the cell: erased jointly, no linear function separates any cell from another."""
    subsets = [s for k in range(1, len(design.axes) + 1) for s in combinations(design.axes, k)]
    return [[concept_label(design, c, s) for c in cells] for s in subsets]


# --------------------------------------------------------------------------- names ----------------------
def cell_name(block: CellBlock, cell: Cell) -> Optional[str]:
    """The proxy first name in ``cell``'s clause (None for an explicit block). Exactly one of the block's names must
    occur in it, as a whole word."""
    if not block.names:
        return None
    found = [n for n in block.names if re.search(rf"\b{re.escape(n)}\b", block.clauses[cell])]
    if len(found) != 1:
        raise ValueError(f"{block.record_id}/{block.template_id}: {len(found)} proxy names in cell {cell}'s clause")
    return found[0]


def name_folds(domain: str, k: int, seed: int) -> Dict[str, int]:
    """name -> the fold that holds it out: each pool (`NAME_POOLS`) in a seeded order, dealt round-robin over ``k``
    folds, so every fold holds out the same share of every pool (10 names, 5 folds: 2 per pool)."""
    folds: Dict[str, int] = {}
    for p, pool in enumerate(NAME_POOLS[domain]):
        order = sorted(pool)
        stable_rng(seed, "demographic_erasure_name_folds", domain, p).shuffle(order)
        for i, name in enumerate(order):
            if name in folds:
                raise ValueError(f"{name} is in two name pools of {domain}")
            folds[name] = i % k
    return folds


# --------------------------------------------------------------------------- selection ------------------
def select_records(blocks: Sequence[CellBlock], encodings: Sequence[str], probe_ids: Set[str], n_eval: int,
                   seed: int) -> Tuple[Dict[str, List[CellBlock]], Dict[str, List[CellBlock]], Dict[str, Any]]:
    """The training blocks (every block of a probe record in ``encodings``) and ``n_eval`` eval records (outside every
    direct probe split, with a block for every requested encoding and every template of the manifest, in seeded
    order), each as {record: blocks}, and a report."""
    templates = sorted({b.template_id for b in blocks})
    wanted = {(e, t) for e in encodings for t in templates}
    by_record: Dict[str, List[CellBlock]] = {}
    for b in blocks:
        if b.encoding in encodings:
            by_record.setdefault(b.record_id, []).append(b)
    order = lambda bs: sorted(bs, key=lambda b: (b.encoding, b.template_id))
    train = {r: order(bs) for r, bs in sorted(by_record.items()) if r in probe_ids}
    complete = sorted(r for r, bs in by_record.items()
                      if r not in probe_ids and {(b.encoding, b.template_id) for b in bs} == wanted)
    stable_rng(seed, "demographic_erasure_records").shuffle(complete)
    evaluate = {r: order(by_record[r]) for r in complete[:n_eval]}
    report = {"records_in_cells": len(by_record), "probe_records": len(train),
              "probe_records_without_blocks": len(probe_ids - set(train)),
              "eval_candidates": len(complete), "eval_records": len(evaluate),
              "incomplete_records": sum(1 for r, bs in by_record.items() if r not in probe_ids
                                        and {(b.encoding, b.template_id) for b in bs} != wanted),
              "templates": templates}
    return train, evaluate, report


@dataclass
class States:
    """One row per (block, cell) of one encoding: the cell, the record, the formatted conversation, the marker clause
    and the proxy name (None for explicit)."""

    cells: List[Cell] = field(default_factory=list)
    records: List[str] = field(default_factory=list)
    texts: List[Any] = field(default_factory=list)
    clauses: List[str] = field(default_factory=list)
    names: List[Optional[str]] = field(default_factory=list)

    @classmethod
    def from_blocks(cls, records: Dict[str, List[CellBlock]], encoding: str, design: FactorialDesign, prompt: str,
                    tokenizer: Any) -> "States":
        """Every block of ``encoding``, its 8 cells formatted as the direct arm formats them."""
        out = cls()
        for rid, blocks in records.items():
            for block in blocks:
                if block.encoding != encoding:
                    continue
                for cell in design.cells:
                    out.cells.append(cell)
                    out.records.append(rid)
                    out.texts.append(format_conversation(tokenizer, prompt, block.texts[cell]))
                    out.clauses.append(block.clauses[cell])
                    out.names.append(cell_name(block, cell))
        return out

    def __len__(self) -> int:
        return len(self.cells)


# --------------------------------------------------------------------------- the test ---------------------
def fold_indices(train: States, evals: States, folds: Optional[Dict[str, int]], k: int
                 ) -> List[Tuple[List[int], List[int]]]:
    """(training rows, eval rows) per fold: one fit on everything without name folds (``folds`` None, explicit);
    else fold f trains on the training states whose name it does not hold out and evaluates the eval states whose
    name it does, so every eval state is evaluated once and never by a fit that saw its name."""
    if folds is None:
        return [(list(range(len(train))), list(range(len(evals))))]
    for name in set(train.names) | set(evals.names):
        if name not in folds:
            raise ValueError(f"proxy name {name!r} is in no name pool of this domain (NAME_POOLS)")
    return [([i for i, n in enumerate(train.names) if folds[n] != f],
             [i for i, n in enumerate(evals.names) if folds[n] == f]) for f in range(k)]


def fit_transform(method: str, Xtr: torch.Tensor, ytr: List[int], cells: List[List[int]]):
    """The row's map from states to erased states, fitted on the training states (None for ``none``)."""
    if method == "none":
        return None
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "diffmean":
        d = diffmean_direction(Xtr, ytr)
        return lambda X: apply_diffmean(d, X)
    eraser = leace_erase(Xtr, ytr if method == "leace" else cells)
    return lambda X: apply_eraser(eraser, X)


def reward_gap(rewards: Dict[str, np.ndarray], labels: Sequence[int], records: Sequence[str], n_boot: int,
               seed: int) -> Dict[str, Dict[str, float]]:
    """Per record: mean reward of its label-1 states − its label-0 states, per row; the mean over records with its
    record-bootstrap interval, and each row's paired change from ``none``."""
    gaps: Dict[str, Dict[str, float]] = {}
    labels = np.asarray(labels)
    for rid in dict.fromkeys(records):
        idx = np.asarray([i for i, r in enumerate(records) if r == rid])
        one, zero = idx[labels[idx] == 1], idx[labels[idx] == 0]
        if not len(one) or not len(zero):
            raise ValueError(f"record {rid}: eval states of one label only")
        gaps[rid] = {m: float(r[one].mean() - r[zero].mean()) for m, r in rewards.items()}
    mean = lambda key: (lambda s: float(np.mean([g[key] for g in s])))
    change = lambda m: (lambda s: float(np.mean([g[m] - g["none"] for g in s])))
    stats = {**{m: mean(m) for m in rewards}, **{f"{m}_minus_none": change(m) for m in rewards if m != "none"}}
    return cluster_bootstrap([[g] for g in gaps.values()], stats, n_boot=n_boot, seed=seed)


def run_encoding(exp: Any, cfg: Any, design: FactorialDesign, encoding: str, train: States, evals: States,
                 folds: Optional[Dict[str, int]], k: int, seed: int, n_boot: int, timer: Dict[str, float]
                 ) -> List[Dict[str, Any]]:
    """Embed one encoding's states (the training states are cache hits after the battery, most eval states not), then
    `erasure_test` with the model's head."""
    start = time.perf_counter()
    Htr, dtype, _ = embed_with_gates(exp.model, exp.tokenizer, train.texts, batch_size=cfg.batch_size,
                                     max_length=cfg.max_length, show_progress=False)
    Hev, _, Gev = embed_with_gates(exp.model, exp.tokenizer, evals.texts, batch_size=cfg.batch_size,
                                   max_length=cfg.max_length, show_progress=False)
    timer["embed"] += time.perf_counter() - start
    start = time.perf_counter()

    def score(H: torch.Tensor, idx: Sequence[int]) -> np.ndarray:
        gates = None if Gev is None else Gev[list(idx)]
        return rewards_from_hidden(exp.model, H, dtype, gates=gates)[0].reshape(-1).float().numpy()

    out = erasure_test(Htr, Hev, train, evals, design, encoding, folds, k, seed, n_boot, score)
    timer["probes"] += time.perf_counter() - start
    return out


def erasure_test(Htr: torch.Tensor, Hev: torch.Tensor, train: States, evals: States, design: FactorialDesign,
                 encoding: str, folds: Optional[Dict[str, int]], k: int, seed: int, n_boot: int,
                 score: Callable[[torch.Tensor, Sequence[int]], np.ndarray]) -> List[Dict[str, Any]]:
    """Every concept of one encoding on the given states (``Htr``/``Hev``, aligned with ``train``/``evals``): per fold
    and row the probes on the states and on the lexical control; the eval states' rewards (``score(H, idx)``: the
    rewards of states ``H``, which are the eval states ``idx`` erased) under one fit on every training state; then the
    folds pooled per concept and row, and the verdicts."""
    splits = fold_indices(train, evals, folds, k)
    todo = concepts(design, encoding)
    cell_tr = cell_columns(design, train.cells)
    labels_tr = {c: [concept_label(design, cell, f) for cell in train.cells] for c, f in todo}
    labels_ev = {c: [concept_label(design, cell, f) for cell in evals.cells] for c, f in todo}
    rewards = reward_rows(Htr, Hev, labels_tr, cell_tr, score)
    model: Dict[Tuple[str, str], List[Any]] = {(c, m): [] for c, _ in todo for m in METHODS}
    lexical: Dict[Tuple[str, str], List[Any]] = {(c, m): [] for c, _ in todo for m in METHODS}
    keys: List[List[str]] = []
    names: List[List[Optional[str]]] = []
    snapped = {(store, c, m): 0 for store in ("model", "lexical") for c, _ in todo for m in METHODS}
    n_train: List[int] = []
    for tr, ev in splits:
        keys.append([evals.records[i] for i in ev])
        names.append([evals.names[i] for i in ev])
        n_train.append(len(tr))
        Ltr, Lev = bag_of_words([train.clauses[i] for i in tr], [evals.clauses[i] for i in ev], WORDS)
        cells = [[col[i] for i in tr] for col in cell_tr]
        groups = [evals.records[i] for i in ev]
        for store, X, E in ((model, Htr[tr], Hev[ev]), (lexical, Ltr, Lev)):
            if not ev:                           # a name fold without eval states (only at tiny sizes)
                for key in store:
                    store[key].append([])
                continue
            erase_cells = fit_transform("leace_cells", X, [], cells)      # the same for every concept
            erased_cells = (erase_cells(X), erase_cells(E))
            for c, _ in todo:
                ytr = [labels_tr[c][i] for i in tr]
                yev = [labels_ev[c][i] for i in ev]
                for m in METHODS:
                    t = erase_cells if m == "leace_cells" else fit_transform(m, X, ytr, cells)
                    a, b = (X, E) if t is None else erased_cells if m == "leace_cells" else (t(X), t(E))
                    fit = probe_recoverability(a, ytr, b, yev, seed=seed, groups_ev=groups, n_boot=0,
                                               return_items=True, mlp_seeds=MLP_SEEDS, unerased_tr=X)
                    store[(c, m)].append(fit["items"])
                    snapped[("model" if store is model else "lexical", c, m)] += fit["snapped_features"]
    # proxy: the interval resamples records and names crossed (generalising over names, not only over records)
    keys_b = names if folds is not None else None
    pooled = {}
    for c, factors in todo:
        rows = {m: pool_folds(model[(c, m)], keys, n_boot, seed, keys_b) for m in METHODS}
        if folds is not None:
            for m in METHODS:
                rows[m]["by_name"] = by_name(model[(c, m)], names)
        pooled[c] = rows
    out = []
    for c, factors in todo:
        rows = pooled[c]
        verdict = concept_verdict(c, factors, pooled, todo)
        lexical_rows = {m: pool_folds(lexical[(c, m)], keys, n_boot, seed, keys_b) for m in METHODS}
        read = "leace_cells" if len(factors) > 1 else "leace"
        out.append({
            "encoding": encoding, "concept": c, "factors": list(factors),
            "label_1": (f"{factors[0]}={design.factors[factors[0]][0]}" if len(factors) == 1 else
                        f"{' and '.join(factors)} at the same pole"),
            "n_train_by_fold": n_train, "n_eval": len(evals), "folds": len(splits), "rows": rows,
            "lexical_control": lexical_rows,
            "reward_gap": reward_gap(rewards[c], labels_ev[c], evals.records, n_boot, seed),
            # features an erasure left constant up to rounding, summed over the folds (`probe_recoverability`)
            "snapped_features": {kind: {m: snapped[(kind, c, m)] for m in METHODS} for kind in ("model", "lexical")},
            "verdict": verdict, "read_row": read,
            # does the text's surface alone recover the concept in the row the verdict reads?
            "lexical_recovers_in_read_row": bool(lexical_rows[read]["intervals"]["mlp_above_chance"]["ci_low"] > 0)})
    return out


def reward_rows(Htr: torch.Tensor, Hev: torch.Tensor, labels_tr: Dict[str, List[int]], cell_tr: List[List[int]],
                score: Callable[[torch.Tensor, Sequence[int]], np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per concept and row the eval states' rewards, every erasure fitted ONCE on all training states — not per name
    fold: a record's two poles carry different names, so per-fold fits would score its label-1 and label-0 states
    under different erasures, and on uncentered states with a large common mean the fits' differences do not cancel
    (review 2026-10-02). The reward side asks what the erasure does to the head, as the battery's nulling does; the
    held-out names are a question for the probes only."""
    idx = list(range(Hev.shape[0]))
    base = score(Hev, idx)
    erase_cells = fit_transform("leace_cells", Htr, [], cell_tr)
    cells_rewards = score(erase_cells(Hev), idx)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for c, ytr in labels_tr.items():
        out[c] = {"none": base, "leace_cells": cells_rewards}
        for m in ("diffmean", "leace"):
            out[c][m] = score(fit_transform(m, Htr, ytr, cell_tr)(Hev), idx)
    return {c: {m: out[c][m] for m in METHODS} for c in out}


def by_name(folds: List[List[Tuple]], names: List[List[Optional[str]]]) -> Dict[str, Dict[str, float]]:
    """Per proxy first name: linear and MLP accuracy and the number of eval states (pooled over the folds; each name
    is evaluated in one fold)."""
    acc: Dict[str, List[Tuple]] = {}
    for items, ns in zip(folds, names):
        for it, n in zip(items, ns):
            acc.setdefault(str(n), []).append(it)
    return {n: {"linear_acc": float(np.mean([it[0] for it in its])), "mlp_acc": float(np.mean([it[1] for it in its])),
                "label_share": float(np.mean([it[2] for it in its])), "n": len(its)}
            for n, its in sorted(acc.items())}


def _above(row: Dict[str, Any], probe: str = "mlp") -> bool:
    return row["intervals"][f"{probe}_above_chance"]["ci_low"] > 0


def concept_verdict(concept: str, factors: Sequence[str], pooled: Dict[str, Dict[str, Any]],
                    todo: Sequence[Tuple[str, Tuple[str, ...]]]) -> str:
    """The reading of one concept, only where its ``none`` row's linear probe is above chance.

    An axis (decided 2026-10-02 after the review): the MLP after LEACE on the axis alone
      - not above chance ⇒ low-complexity;
      - above chance and still after LEACE on every cell ⇒ non-linear beyond the cell structure: entangled,
        high-complexity (the TaCo signature);
      - above chance but gone after LEACE on every cell ⇒ the MLP rebuilt the axis from the other factors and a
        LINEAR interaction term, which LEACE on the axis alone leaves in place: intersectional, not high-complexity —
        read so only where an interaction concept of this axis is itself linearly decodable (its ``none`` row), else
        unresolved (a 7-column erasure can also just cost power).
    An interaction: its own LEACE row is not read (with both main effects linear in the states an MLP rebuilds the XNOR
    from them; the words alone do) — the row after LEACE on every cell is."""
    rows = pooled[concept]
    if not _above(rows["none"], "linear"):
        return "not decodable without erasure: says nothing"
    if len(factors) > 1:
        if _above(rows["leace_cells"]):
            return "non-linearly recoverable after LEACE on every cell (entangled, high-complexity)"
        return "not recoverable after LEACE on every cell (low-complexity)"
    if not _above(rows["leace"]):
        return "not recoverable after LEACE (low-complexity)"
    if _above(rows["leace_cells"]):
        return "non-linearly recoverable beyond the cell structure (entangled, high-complexity)"
    partners = [c for c, f in todo if len(f) > 1 and factors[0] in f]
    if any(_above(pooled[c]["none"], "linear") for c in partners):
        return "recoverable only through a linear interaction with another factor (intersectional, not high-complexity)"
    return ("recoverable after LEACE but not after LEACE on every cell, and no interaction of it is linearly "
            "decodable: unresolved")


# --------------------------------------------------------------------------- CLI ------------------------
def check_names(train_recs: Dict[str, List[CellBlock]], eval_recs: Dict[str, List[CellBlock]],
                design: FactorialDesign, folds: Dict[str, int]) -> None:
    """Before the model loads: every proxy cell's clause names exactly one of its block's names, and every name is in
    a pool of the domain (a stale `NAME_POOLS` would otherwise fail only after a 70B load)."""
    for recs in (train_recs, eval_recs):
        for blocks in recs.values():
            for block in blocks:
                for cell in design.cells:
                    try:
                        name = cell_name(block, cell)
                    except ValueError as e:
                        raise SystemExit(str(e))
                    if name is not None and name not in folds:
                        raise SystemExit(f"proxy name {name!r} ({block.record_id}) is in no name pool of this domain "
                                         f"(NAME_POOLS)")


def battery_split(dom: Any, source: str, cfg: Any, encodings: Sequence[str],
                  train_recs: Dict[str, List[CellBlock]]) -> Dict[str, Any]:
    """Per encoding: are the training records exactly the battery's probe split (its first axis's dataset; the split
    ignores the axis)? The training set is the union over every axis and encoding the manifest has pairs for, which
    exceeds one dataset's split if gate drops made the encodings' record sets differ; then the explicit diffmean row
    is no longer exactly the battery's direction."""
    out: Dict[str, Any] = {}
    for enc in encodings:
        axis = next(a for a in dom.axes if dom.factorial.axis_pairs(a, enc))
        ids = dom.dataset_cls(source, axis=axis, encoding=enc, probe_records=cfg.probe_records,
                              split_seed=cfg.split_seed).probe_record_ids()
        used = {r for r, blocks in train_recs.items() if any(b.encoding == enc for b in blocks)}
        out[enc] = {"battery_probe_records": len(ids), "training_records": len(used), "identical": used == ids}
        if used != ids:
            logger.warning("%s: the training records (%d) are not the battery's probe split (%d)", enc, len(used),
                           len(ids))
    return out



def default_out(domain: str, source: Path | str, model_path: str, variant: str = "") -> Path:
    """``erasure_demographic_{domain}[_{manifest folder}]_{model}{variant}.json``, as the battery names its results."""
    folder = Path(source).parent.name
    stem = domain if folder == domain else f"{domain}_{folder}"
    return RESULTS_DIR / f"erasure_demographic_{stem}_{Path(model_path).name}{variant}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_credit_sex_qwen06.yaml"))
    ap.add_argument("--encodings", default="explicit,proxy", help="Comma-separated")
    ap.add_argument("--eval-records", type=int, default=200, help="Held-out records evaluated")
    ap.add_argument("--name-folds", type=int, default=5, help="Folds of held-out proxy names")
    ap.add_argument("--seed", type=int, default=42, help="Seeds the eval records, the name folds and the probes")
    ap.add_argument("--n-boot", type=int, default=None, help=f"Bootstrap replicates (default extra.n_boot, else "
                                                              f"{DEFAULT_N_BOOT})")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"Default {RESULTS_DIR}/erasure_demographic_{{domain}}[_{{folder}}]_{{model}}{{variant}}.json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result")
    add_override_args(ap)
    return ap


def _ci(iv: Dict[str, float], fmt: str = "{:+.3f}") -> str:
    return f"{fmt.format(iv['estimate'])} [{fmt.format(iv['ci_low'])}, {fmt.format(iv['ci_high'])}]"


def print_report(results: List[Dict[str, Any]], model_path: str) -> None:
    print("\n" + "=" * 132)
    print(f"DEMOGRAPHIC ERASURE — LEACE + non-linear probe, direct-arm states — {model_path}")
    print("=" * 132)
    print(f"{'enc':8} {'concept':26} {'row':12} {'linear':>7} {'MLP':>7}  {'MLP − chance [95% CI]':>26}  "
          f"{'lexical MLP − chance':>26}  {'reward gap [95% CI]':>26}")
    for r in results:
        for m in METHODS:
            row, lex = r["rows"][m], r["lexical_control"][m]
            gap = r["reward_gap"][m]
            print(f"{r['encoding']:8} {r['concept']:26} {m:12} {row['linear_acc']:>7.3f} {row['mlp_acc']:>7.3f}  "
                  f"{_ci(row['intervals']['mlp_above_chance']):>26}  {_ci(lex['intervals']['mlp_above_chance']):>26}  "
                  f"{_ci(gap, '{:+.4f}'):>26}")
        print("-" * 132)
    for r in results:
        print(f"{r['encoding']:8} {r['concept']:26} ⇒ {r['verdict']}"
              + ("  [the words alone recover it in the row read]" if r["lexical_recovers_in_read_row"] else ""))
    print("Read only where the 'none' row's linear probe is above chance; an axis from its leace row (and leace_cells "
          "for where the recovery lives), an interaction from leace_cells. Reward gap: label-1 − label-0 per record.")


def main() -> None:
    t0 = time.perf_counter()
    ap = build_parser()
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    configured_cfg = ExperimentConfig.from_yaml(args.config)
    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    dom = get_domain(cfg.extra.get("domain", "credit"))
    design = dom.factorial
    if design is None or dom.name not in NAME_POOLS:
        raise SystemExit(f"the demographic erasure test runs on the factorial domains {sorted(NAME_POOLS)}, "
                         f"not {dom.name!r}")
    named = {e.strip() for e in args.encodings.split(",") if e.strip()}
    if not named or named - set(ENCODINGS):
        raise SystemExit(f"--encodings must name some of {list(ENCODINGS)}, got {args.encodings!r}")
    encodings = [e for e in ENCODINGS if e in named]          # canonical order: the result name does not depend on it
    if args.eval_records < 2 or args.name_folds < 2:
        raise SystemExit("--eval-records and --name-folds must be at least 2")
    n_boot = args.n_boot if args.n_boot is not None else int(cfg.extra.get("n_boot", DEFAULT_N_BOOT))
    keys = ("encodings", "eval_records", "name_folds", "seed")
    configured = {**{k: ap.get_default(k) for k in keys}, "n_boot": int(configured_cfg.extra.get("n_boot",
                                                                                                    DEFAULT_N_BOOT)),
                  "probe_records": configured_cfg.probe_records, "revision": configured_cfg.model_revision}
    used = {**{k: getattr(args, k) for k in keys}, "encodings": ",".join(encodings), "n_boot": n_boot,
            "probe_records": cfg.probe_records, "revision": cfg.model_revision}
    source = cfg.dataset_source = cfg.dataset_source or dom.default_pairs
    out = args.out or default_out(dom.name, source, cfg.model_path, variant_suffix(configured, used))
    # everything that can fail on the inputs fails here, before the model loads
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or --out")
    path = cells_path(source, dom.default_pairs)
    data = {"pairs.jsonl": data_file(source), "cells.jsonl": data_file(path)}
    cells_report: Dict[str, Any] = {}
    blocks = load_cell_blocks(path, design, cells_report)
    missing = sorted(set(encodings) - {b.encoding for b in blocks})
    if missing:
        raise SystemExit(f"encoding {missing} not in {path}")
    probe_ids = probe_split_ids(dom, source, cfg.probe_records, cfg.split_seed)
    train_recs, eval_recs, selection = select_records(blocks, encodings, probe_ids, args.eval_records, args.seed)
    selection["cells"] = cells_report
    if len(eval_recs) < 2 or not train_recs:
        raise SystemExit(f"{len(train_recs)} probe records and {len(eval_recs)} eval records with every block: "
                         f"too few ({selection})")
    if len(eval_recs) < args.eval_records:
        logger.warning("%d eval records with every block, fewer than --eval-records %d: took all",
                       len(eval_recs), args.eval_records)
    folds = name_folds(dom.name, args.name_folds, args.seed)
    check_names(train_recs, eval_recs, design, folds)
    selection["battery_split"] = battery_split(dom, source, cfg, encodings, train_recs)

    from scoring.demographic_experiment import DemographicBiasExperiment

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    timer = {"load_model": time.perf_counter() - t0, "embed": 0.0, "probes": 0.0}
    prompt = dom.dataset_cls.DEFAULT_PROMPT            # the battery's format: these texts are its cache hits
    results: List[Dict[str, Any]] = []
    for enc in encodings:
        print(f"[erasure] {dom.name}/{enc} ...", flush=True)
        train = States.from_blocks(train_recs, enc, design, prompt, exp.tokenizer)
        evals = States.from_blocks(eval_recs, enc, design, prompt, exp.tokenizer)
        results += run_encoding(exp, cfg, design, enc, train, evals, folds if enc == "proxy" else None,
                                args.name_folds, args.seed, n_boot, timer)

    print_report(results, cfg.model_path)
    timer["total"] = time.perf_counter() - t0
    print("timing: " + " | ".join(f"{k} {v:.0f}s" for k, v in timer.items()))
    settings = {"encodings": encodings, "eval_records": args.eval_records, "name_folds": args.name_folds,
                "seed": args.seed, "n_boot": n_boot, "mlp_seeds": MLP_SEEDS, "methods": list(METHODS),
                "probe_records": cfg.probe_records}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": run_metadata(cfg, data, settings), "model": cfg.model_path, "domain": dom.name,
         "selection": selection, "probe_records": sorted(train_recs), "eval_records": list(eval_recs),
         "name_folds": {n: f for n, f in sorted(folds.items())}, "timing": timer, "results": results}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
