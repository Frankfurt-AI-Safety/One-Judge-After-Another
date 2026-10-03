#!/usr/bin/env python3
"""
RQ5: do the demographic directions transfer across domains and encodings? One RM, every domain of ``--configs``
(design and decisions: working notes 2026-10-03). Exploratory until the headline family is fixed. Across placements
within a domain, see `runners/run_placement_matrix.py`.

**Sources.** A direction is the direct arm's difference of means for one (domain, encoding, axis), fitted on the
battery's probe records (`probes.transfer_directions.UnitStore`: the mean of the probe blocks' matched-pair state
differences, the battery's direction). Named ``{domain}/{encoding}/{axis}``.

**Targets**, per (domain, encoding, axis), on the cross-marker arm's evaluated records (its selection, length guard and
paraphrases, so its texts are embedding-cache hits; none is a probe record, and domains share no record):
  - ``direct``: the direct gap r(pole A) − r(pole B) of the record's cells with the marker in the response (the
    mechanism reading), every record;
  - ``decision``: the decision disparity D(pole A) − D(pole B), D = r(approve) − r(decline), with the marker in the
    request (the harm measure), strong records. Its own row is the same domain's direct direction: for an explicit
    target the cross-marker arm's ``null_direct`` column; for a proxy target that column is ``own_seen`` (the own row
    is fitted without the pair's names).

**Rows.** Every source is projected out of every target; ``rows[name]["relation"]`` says how it relates to the target,
by what the two sides' pairs flip (`flips`: a single axis flips one factor, an intersection every factor of its domain;
a factor counts as the same across domains only with the same poles: sex everywhere, age in credit and hiring):
  - the same factors: ``own``; ``encoding`` (same domain, the other encoding); ``domain`` and ``domain+encoding``;
  - overlapping but not the same (an intersection against a single axis of its corner, or two domains' corners):
    ``component``, ``component+encoding``, ``component+domain``, ``component+domain+encoding`` — the source should remove
    part of the effect, so these are no specificity controls;
  - nothing shared: ``off_axis`` (same domain: the specificity control) and ``unrelated``.
Where the encodings differ, ``proxy_construct_differs`` says whether a shared factor's proxy measures a related but
different attribute (hiring's family status, education's economic status). ``rows[name]["n_fit"]`` is the range, over
the target's pairs, of the pairs the row's direction rests on. And:
  - ``own_seen`` (proxy targets): the own direction fitted on every probe pair, the battery's;
  - ``template``: the own direction fitted without the target row's document template;
  - ``others``: the unit mean of the other domains' directions of the same axis and encoding (two or more);
  - ``singles_joint`` (intersection targets): the single-axis directions of every factor of the corner projected out
    together — does an edit built from single attributes cover the corner? A factor without a direction in the
    target's encoding takes the explicit one (credit's marital status has no proxy; its proxy corner states it in the
    explicit words). ``singles_joint_axes`` lists the directions used;
  - ``random{k}``: random unit directions, the floor.

**Held-out names.** The domains draw the sex (and ethnicity) proxy from shared pools of first names, so a proxy
direction could remove a proxy gap by recognising names. Every name sits in one of ``--name-folds`` folds, the same in
every domain (`shared_name_folds`); a target pair's rows are nulled with a direction fitted on the pairs that share no
fold with its names. That holds for every row but ``own_seen`` and ``random``, the own row included. The fits are
not equally large: an explicit source carries no name and keeps every pair, a one-name proxy source (age) about 60%
against a two-name target, a two-name one (sex, ethnicity, the corner) about 36% with five folds — so for a proxy
target the ``encoding`` row rests on more pairs than the own row (``rows[name]["n_fit"]``; the worst case per source is
``sources[...]["min_fit_units"]``). The intervals of proxy targets resample records with the names held fixed: they
condition on the name pools.

**Read-outs** per target and row:
  - ``tables[gate mode]`` (`scoring.placement_matrix.cell_table`, records as units, one set of bootstrap draws per
    domain, so differences between rows are paired): baseline, nulled, change, gap (own change − this row's) and the
    shortfall (the gap oriented by the baseline's sign: positive = this row removed less than the own one). Read a gap
    only where the own row's change differs from 0. Gated heads (QRM) get ``gate_fixed`` first (every decision row
    scored with the gate of its record's unmarked prompt; the projection acts on the last-token state only), then
    ``own_gates``; other heads ``own_gates`` only;
  - ``paired_acc`` (direct targets): how often the row's direction orders a target pair's poles, proj(A) > proj(B), a
    tie ½, with a record bootstrap — the representation-level reading, informative where the reward shows no gap;
    ``lexical_paired_acc`` is the same on bag-of-words vectors of the marker clauses (digits as words, the
    vocabulary of the fitted clauses): what shared wording alone transfers. Credit and hiring share most marker words,
    so a transfer between them tests a new document and prompt, not new wording. ``token_paired_acc`` is the same on
    the bag of the model tokenizer's tokens of the clause: whole words tie on a held-out name by construction, but
    names share sub-word tokens across folds (Latonya / Latoya), which only this control shows;
  - ``geometry`` (descriptive, full-data directions): each source's cosine with the target's own direction against its
    ceiling √(rel · rel), its head alignment, and the share of the target's effect it carries by the exact
    decomposition (w·u)(Δ·u) / (w·Δ) (`probes.cross_marker_directions.did_share`).

Inputs: one cross-marker config per domain (their manifests, probe split, ``max_length`` and ``extra.cross_marker``
record settings), all naming one model, cache, device, seed and ``n_boot``. ``--n-strong`` / ``--n-weak`` take one
value for every domain or one per config. What can fail on the inputs fails before the model loads, including a
held-out fit that would be left without pairs; only the length guard needs the tokenizer (an empty selection stops
the run after the load, an over-long direct text when it is embedded). Outputs (no texts):
``demographic_transfer_{model}{variant}.json`` (``meta``, selection, sources, the name folds, one entry per target and
placement; ``variant`` names every setting the CLI changed and the manifest folders where they are not the domains'
defaults) and ``…_directions.pt`` (every source's full-data direction); never replaced without ``--overwrite``.

Usage:
    python runners/run_demographic_transfer.py            # the three cross-marker configs
    python runners/run_demographic_transfer.py --n-strong 4 --n-weak 4 --probe-records 40 --device mps    # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from pairs.cross_marker import CellBlock, build_block_items, load_cell_blocks
from pairs.factorial import Cell, FactorialDesign, stable_rng
from probes import cross_marker_directions as cmd
from probes.erasure import bag_of_words
from probes.heads import get_head
from probes.probe import embed_states, embed_with_gates, rewards_from_hidden
from probes.transfer_directions import Key, UnitStore, mean_unit, random_units
from runners import run_cross_marker as rx
from runners.run_comparative import probe_split_ids
from runners.run_demographic_erasure import NAME_POOLS, WORDS, battery_split, cell_name, check_names
from runners.run_demographic_erasure import select_records as probe_blocks
from runners.run_placement_matrix import gate_refs
from scoring.cross_marker_metrics import RewardIndex, record_axis_effect
from scoring.dataset_base import format_conversation
from scoring.demographic_experiment import DemographicBiasExperiment
from scoring.experiment import (
    ExperimentConfig, add_override_args, apply_overrides, data_file, run_metadata, variant_suffix,
)
from scoring.placement_matrix import cell_table, ratio_summary, shared_draws
from substrates.domains import get_domain

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("artifacts/results/demographic")
DEFAULT_CONFIGS = [Path("configs/demographic_credit_crossmarker_qwen06.yaml"),
                   Path("configs/demographic_cv_crossmarker_qwen06.yaml"),
                   Path("configs/demographic_edu_crossmarker_asap2_qwen06.yaml")]
DEFAULT_DOMAINS = ["credit", "cv", "education"]
# settings every config must share: they are read from the first config only
SHARED = ("embedding_cache_dir", "device", "trust_remote_code")
PER_DOMAIN = ("n_strong", "n_weak")                 # CLI flags that take one value, or one per config
X_RESPONSES = ("approve", "decline")
DIRECT_RESPONSE = "document"       # the direct placement's one "response": the document is the scored text
INTERSECTION = "intersection"
TRANSFER = ("encoding", "domain", "domain+encoding")        # the relations the transfer reading rests on
EXTRA_ROWS = ("own_seen", "template", "others", "singles_joint")
MAX_PAIR_NAMES = 2                                  # a matched pair's two texts carry at most one first name each

Source = Tuple[str, str, str]                       # (domain, encoding, axis)


def source_name(source: Source) -> str:
    return "/".join(source)


# --------------------------------------------------------------------------- names -------------------
def shared_name_folds(k: int, seed: int) -> Dict[str, int]:
    """name -> fold, the same in every domain: each distinct pool of `NAME_POOLS` (the domains share the white-coded
    pools) in a seeded order, dealt round-robin over ``k`` folds, so every fold holds out the same share of every
    pool (10 names, 5 folds: 2 per pool). The pools of every domain enter, whichever domains run."""
    folds: Dict[str, int] = {}
    for pool in sorted({tuple(sorted(pool)) for pools in NAME_POOLS.values() for pool in pools}):
        order = list(pool)
        stable_rng(seed, "demographic_transfer_name_folds", *pool).shuffle(order)
        for i, name in enumerate(order):
            if name in folds:
                raise ValueError(f"{name} is in two name pools")
            folds[name] = i % k
    return folds


def fold_set(names: Iterable[Optional[str]], folds: Mapping[str, int]) -> FrozenSet[int]:
    """The folds of the first names among ``names`` (None: a text without one)."""
    return frozenset(folds[n] for n in names if n is not None)


# --------------------------------------------------------------------------- relations ---------------
def encoding_axes(design: FactorialDesign, encoding: str) -> List[str]:
    """The axes with matched pairs in ``encoding``, then the intersection (the corner pair)."""
    return [a for a in design.axes if design.axis_pairs(a, encoding)] + [INTERSECTION]


def flips(source: Source, designs: Mapping[str, FactorialDesign]) -> FrozenSet[Tuple[str, Tuple[Any, ...]]]:
    """What a source's pairs flip, as (factor, its two poles): one factor for a single axis, every factor of the
    domain for the intersection (credit's proxy corner flips marital status too, in the explicit words)."""
    domain, _, axis = source
    return frozenset((f, tuple(levels)) for f, levels in designs[domain].factors.items()
                     if axis in (INTERSECTION, f))


def same_axis(source: Source, target: Source, designs: Mapping[str, FactorialDesign]) -> bool:
    """Both flip the same factors with the same poles: within a domain the same axis; across domains a single axis
    whose levels agree (sex; age in credit and hiring). The intersection is a different corner in every domain."""
    return flips(source, designs) == flips(target, designs)


def relation(source: Source, target: Source, designs: Mapping[str, FactorialDesign]) -> Dict[str, Any]:
    """How a source direction relates to a target (see the module docstring)."""
    (sd, se, _), (td, te, _) = source, target
    fs, ft = flips(source, designs), flips(target, designs)
    shared = fs & ft
    if not shared:
        return {"relation": "off_axis" if sd == td else "unrelated"}
    scope = "+".join(([] if sd == td else ["domain"]) + ([] if se == te else ["encoding"]))
    out: Dict[str, Any] = {"relation": (scope or "own") if fs == ft else "+".join(filter(None, ["component", scope]))}
    if se != te:
        proxy = designs[sd if se == "proxy" else td]
        out["proxy_construct_differs"] = any(f in proxy.proxy_labels for f, _ in shared)
    return out


def other_domains(target: Source, sources: Iterable[Source], designs: Mapping[str, FactorialDesign]) -> List[Source]:
    """The other domains' sources of the target's axis and encoding."""
    return [s for s in sources if s[0] != target[0] and s[1] == target[1] and same_axis(s, target, designs)]


def single_axes(target: Source, sources: Iterable[Source], designs: Mapping[str, FactorialDesign]) -> List[Source]:
    """The single-axis sources of every factor of the target's domain: in the target's encoding, else the explicit one
    (credit's marital status has no proxy direction, and its proxy corner states it in the explicit words). A factor
    with neither is left out."""
    domain, encoding, _ = target
    have = set(sources)
    out = []
    for factor in designs[domain].axes:
        for e in dict.fromkeys((encoding, "explicit")):
            if (domain, e, factor) in have:
                out.append((domain, e, factor))
                break
    return out


Kind = Tuple[Any, ...]     # ("source", source) | ("own_seen" | "template" | "others" | "singles_joint", target) | ("random", k)


def row_kinds(target: Source, stores: Mapping[Source, UnitStore], designs: Mapping[str, FactorialDesign],
              n_random: int) -> Dict[str, Kind]:
    """row name -> what it projects out, for one target: every source, then the extra rows that exist for it."""
    rows: Dict[str, Kind] = {source_name(s): ("source", s) for s in stores}
    if stores[target].named:
        rows["own_seen"] = ("own_seen", target)
    if len(stores[target].templates) > 1:
        rows["template"] = ("template", target)
    if len(other_domains(target, stores, designs)) > 1:
        rows["others"] = ("others", target)
    if target[2] == INTERSECTION:
        rows["singles_joint"] = ("singles_joint", target)
    rows.update({f"random{k}": ("random", k) for k in range(n_random)})
    return rows


Fit = Tuple[Source, FrozenSet[int], Optional[str]]      # (source, name folds left out, template left out)


def row_fits(kind: Kind, key: Key, stores: Mapping[Source, UnitStore], designs: Mapping[str, FactorialDesign]
             ) -> List[Fit]:
    """The fits row ``kind`` rests on for a target row with hold-out ``key`` (its pair's name folds, its template);
    none for a random row."""
    folds, template = key
    what, arg = kind
    if what == "source":
        return [(arg, folds, None)]
    if what == "own_seen":
        return [(arg, frozenset(), None)]
    if what == "template":
        return [(arg, folds, template)]
    if what == "others":
        return [(s, folds, None) for s in other_domains(arg, stores, designs)]
    if what == "singles_joint":
        return [(s, folds, None) for s in single_axes(arg, stores, designs)]
    if what == "random":
        return []
    raise ValueError(f"unknown row kind {what!r}")


def row_basis(kind: Kind, key: Key, stores: Mapping[Source, UnitStore], designs: Mapping[str, FactorialDesign],
              random: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """What row ``kind`` projects out of a target row with hold-out ``key``: one direction [d], or several [k, d]
    (``singles_joint``). None for a random row without ``random`` (the lexical spaces have none)."""
    directions = [stores[s].direction(folds, template) for s, folds, template in row_fits(kind, key, stores, designs)]
    if kind[0] == "random":
        return None if random is None else random[kind[1]]
    if kind[0] == "others":
        return mean_unit(directions)
    if kind[0] == "singles_joint":
        return torch.stack(directions)
    return directions[0]


def row_fit_units(kind: Kind, keys: Iterable[Key], stores: Mapping[Source, UnitStore],
                  designs: Mapping[str, FactorialDesign]) -> Optional[Dict[str, int]]:
    """The range, over hold-out ``keys``, of the pairs the row's direction rests on (the smallest of its fits where
    it combines several). None for a random row."""
    if kind[0] == "random":
        return None
    n = [min(stores[s].n_fit(folds, template) for s, folds, template in row_fits(kind, key, stores, designs))
         for key in set(keys)]
    return {"min": min(n), "max": max(n)}


def row_relation(kind: Kind, target: Source, designs: Mapping[str, FactorialDesign]) -> Dict[str, Any]:
    return relation(kind[1], target, designs) if kind[0] == "source" else {"relation": kind[0]}


# --------------------------------------------------------------------------- blocks -> units ---------
@dataclass
class BlockTable:
    """One encoding's blocks of a record set, in order, and each cell's proxy first name (None: explicit)."""

    blocks: List[CellBlock]
    names: List[Dict[Cell, Optional[str]]]

    @classmethod
    def of(cls, records: Mapping[str, Sequence[CellBlock]], encoding: str, design: FactorialDesign) -> "BlockTable":
        blocks = [b for bs in records.values() for b in bs if b.encoding == encoding]
        return cls(blocks, [{c: cell_name(b, c) for c in design.cells} for b in blocks])

    def direct_convs(self, design: FactorialDesign, prompt: str, tokenizer: Any) -> List[Any]:
        """Every block's cells as the direct arm formats them (block-major, ``design.cells`` order)."""
        return [format_conversation(tokenizer, prompt, b.texts[c]) for b in self.blocks for c in design.cells]

    def clauses(self, design: FactorialDesign) -> List[str]:
        """The marker clauses, in the order of `direct_convs`."""
        return [b.clauses[c] for b in self.blocks for c in design.cells]


@dataclass
class UnitIndex:
    """The matched pairs of one axis in a `BlockTable`: per pair the rows of its pole-A and pole-B cells (in a
    block-major, ``design.cells``-ordered array), its name folds, template and record."""

    a: List[int]
    b: List[int]
    folds: List[FrozenSet[int]]
    templates: List[str]
    records: List[str]

    @classmethod
    def of(cls, table: BlockTable, design: FactorialDesign, encoding: str, axis: str,
           name_folds: Mapping[str, int]) -> "UnitIndex":
        at = {c: i for i, c in enumerate(design.cells)}
        out = cls([], [], [], [], [])
        for j, block in enumerate(table.blocks):
            for a, b in design.axis_pairs(axis, encoding):
                out.a.append(j * len(at) + at[a])
                out.b.append(j * len(at) + at[b])
                out.folds.append(fold_set((table.names[j][a], table.names[j][b]), name_folds))
                out.templates.append(block.template_id)
                out.records.append(block.record_id)
        return out

    def keys(self) -> List[Key]:
        return list(zip(self.folds, self.templates))

    def diffs(self, vectors: torch.Tensor) -> torch.Tensor:
        """vector(pole A) − vector(pole B) per pair, float32."""
        return vectors[self.a].float() - vectors[self.b].float()

    def store(self, vectors: torch.Tensor) -> UnitStore:
        return UnitStore(self.diffs(vectors), self.folds, self.templates, self.records)


CONTROLS = {"words": "lexical_paired_acc", "tokens": "token_paired_acc"}       # lexical control -> its result key


class ClauseFeatures:
    """The lexical controls' vectors of marker clauses, with the vocabulary of the fitted clauses: ``words`` (binary
    bag of words, digits as words: `probes.erasure.bag_of_words`) and ``tokens`` (binary bag of the model tokenizer's
    token ids of the clause on its own, without special tokens; every clause starts with a space, as in its document,
    so its tokens are the document's up to the two ends)."""

    def __init__(self, tokenizer: Any, fit_clauses: Sequence[str]):
        self.tokenizer, self.fit_clauses = tokenizer, list(fit_clauses)
        self.vocabulary = {t: i for i, t in enumerate(sorted({t for ids in self._ids(self.fit_clauses) for t in ids}))}

    def _ids(self, clauses: Sequence[str]) -> List[List[int]]:
        return self.tokenizer(list(clauses), add_special_tokens=False)["input_ids"]

    def vectors(self, control: str, clauses: Sequence[str]) -> torch.Tensor:
        """One float32 row per clause."""
        if control == "words":
            return bag_of_words(self.fit_clauses, clauses, WORDS)[1]
        if control != "tokens":
            raise ValueError(f"unknown lexical control {control!r}")
        X = torch.zeros(len(clauses), len(self.vocabulary))
        for i, ids in enumerate(self._ids(clauses)):
            X[i, [self.vocabulary[t] for t in set(ids) if t in self.vocabulary]] = 1.0
        return X


def min_fit_units(unit_index: UnitIndex, k: int) -> int:
    """The fewest pairs any held-out fit of these units can rest on: over every set of up to `MAX_PAIR_NAMES` name
    folds a target pair can exclude, with and without a template left out. From the names alone (no states)."""
    templates = sorted(set(unit_index.templates))
    excluded_templates: List[Optional[str]] = [None] + (templates if len(templates) > 1 else [])
    named = any(unit_index.folds)
    fold_sets = [frozenset()] + ([frozenset(c) for n in range(1, MAX_PAIR_NAMES + 1)
                                  for c in combinations(range(k), n)] if named else [])
    return min(sum(1 for f, t in unit_index.keys() if not (f & excluded) and t != template)
               for excluded in fold_sets for template in excluded_templates)


def row_keys(cells: Sequence[Optional[Cell]], block_of: Sequence[int], table: BlockTable, design: FactorialDesign,
             encoding: str, axis: str, name_folds: Mapping[str, int]) -> List[Key]:
    """Each target row's hold-out key for target ``axis``: the name folds of the matched pair its cell belongs to
    (a cell outside every pair of the axis — the intersection uses the two corners only — and the unmarked control
    enter no effect of the axis; they get their own name's fold and none), and the row's template."""
    partner: Dict[Cell, Tuple[Cell, ...]] = {}
    for a, b in design.axis_pairs(axis, encoding):
        partner[a] = partner[b] = (a, b)
    keys: List[Key] = []
    for cell, j in zip(cells, block_of):
        names = () if cell is None else (table.names[j][c] for c in partner.get(cell, (cell,)))
        keys.append((fold_set(names, name_folds), table.blocks[j].template_id))
    return keys


# --------------------------------------------------------------------------- domains -----------------
class Domain:
    """One config's domain: its manifest, cross-marker settings, probe records and their blocks, all checked before
    the model loads."""

    def __init__(self, cfg: ExperimentConfig, overrides: Mapping[str, Any], name_folds: Mapping[str, int], k: int):
        self.cfg = cfg
        self.dom = get_domain(cfg.extra.get("domain", "credit"))
        self.name = self.dom.name
        if self.dom.factorial is None:
            raise SystemExit(f"{cfg.name}: the transfer runs on factorial domains, not {self.name!r}")
        self.design: FactorialDesign = self.dom.factorial
        self.prompt = self.dom.dataset_cls.DEFAULT_PROMPT         # the battery's, for fit and target cells alike
        self.settings = rx.resolve_settings(cfg.extra, dict(overrides))
        self.source = cfg.dataset_source = cfg.dataset_source or self.dom.default_pairs
        cells = rx.cells_path(self.source, self.dom.default_pairs)
        self.data = {f"{self.name}/pairs.jsonl": data_file(self.source), f"{self.name}/cells.jsonl": data_file(cells)}
        self.cells_report: Dict[str, Any] = {}
        self.blocks = load_cell_blocks(cells, self.design, self.cells_report)
        self.encodings: List[str] = list(self.settings["encodings"])
        self.templates = self.settings["templates"] or sorted({b.template_id for b in self.blocks})
        rx.check_requested(self.blocks, self.encodings, self.templates)
        self.probe_ids = probe_split_ids(self.dom, self.source, cfg.probe_records, cfg.split_seed)
        self.train, _, report = probe_blocks(self.blocks, self.encodings, self.probe_ids, 0, cfg.split_seed)
        self.fit_report = {k: report[k] for k in ("records_in_cells", "probe_records", "probe_records_without_blocks",
                                                   "templates")}
        if not self.train:
            raise SystemExit(f"{self.name}: no probe record with a block in {cells}")
        by_record: Dict[str, List[CellBlock]] = defaultdict(list)
        for b in self.blocks:
            if b.encoding in self.encodings:
                by_record[b.record_id].append(b)
        check_names(by_record, {}, self.design, dict(name_folds))
        self.fit_report["battery_split"] = battery_split(self.dom, self.source, cfg, self.encodings, self.train)
        self.fit_tables = {e: BlockTable.of(self.train, e, self.design) for e in self.encodings}
        self.fit_units: Dict[Source, UnitIndex] = {}
        for e in self.encodings:
            for axis in encoding_axes(self.design, e):
                units = UnitIndex.of(self.fit_tables[e], self.design, e, axis, name_folds)
                if min_fit_units(units, k) == 0:
                    raise SystemExit(
                        f"{self.name}/{e}/{axis}: a held-out fit would be left without pairs ({len(units.a)} probe "
                        f"pairs, {k} name folds); raise --probe-records")
                self.fit_units[(self.name, e, axis)] = units

    def select(self, tokenizer: Any) -> None:
        """The cross-marker arm's evaluated records: its seeded choice and its length guard (every response)."""
        format_fn = lambda prompt, response: format_conversation(tokenizer, prompt, response)
        fits = rx.block_fits(self.name, self.settings, format_fn, rx.token_counter(tokenizer), self.cfg.max_length)
        self.selected, self.selection = rx.select_records(
            self.blocks, quality_field=self.dom.quality_field, encodings=self.encodings, templates=self.templates,
            exclude=self.probe_ids, n_strong=self.settings["n_strong"], n_weak=self.settings["n_weak"],
            seed=self.settings["seed"], fits=fits)
        rx.log_selection(self.cells_report, self.selection)
        if not self.selected:
            raise SystemExit(f"{self.name}: no record to evaluate")


# --------------------------------------------------------------------------- placements --------------
@dataclass
class Placement:
    """One (domain, encoding, placement): its rows, states and gates, and what the target effects need."""

    name: str
    rows: List[Dict[str, Any]]
    H: torch.Tensor
    dtype: Any
    gates: Optional[torch.Tensor]
    fixed: Optional[torch.Tensor]              # each row's gate under gate_fixed (None: not gated, or no control)
    index: RewardIndex
    keys: Dict[str, List[Key]]                 # target axis -> each row's hold-out key
    counts: np.ndarray                         # per record of ``index.records``: 1 where it enters the effect
    w: torch.Tensor                            # the (mean effective) head vector over the rows

    def modes(self) -> List[Tuple[str, Optional[torch.Tensor]]]:
        """(gate mode, the gates it scores with): gated heads the gate-fixed reading first."""
        return ([("gate_fixed", self.fixed)] if self.fixed is not None else []) + [("own_gates", self.gates)]

    def effect(self, values: np.ndarray, axis: str) -> np.ndarray:
        """Per record the target effect of rewards ``values`` (row order) on ``axis``, 0 where it does not count."""
        e = record_axis_effect(self.index, np.asarray(values)[self.index.pos], axis,
                               margin=None if self.name == "direct" else "D")
        return np.where(self.counts > 0, e, 0.0)


def direct_rows(table: BlockTable, design: FactorialDesign, quality_field: str
                ) -> Tuple[List[Dict[str, Any]], List[Optional[Cell]], List[int]]:
    """The direct placement's rows (ids and labels, no text) in `BlockTable.direct_convs` order, each row's cell and
    block."""
    rows, cells, block_of = [], [], []
    for j, block in enumerate(table.blocks):
        for cell in design.cells:
            rows.append({"record_id": block.record_id, "template_id": block.template_id, "encoding": block.encoding,
                         "cell": list(cell), "response": DIRECT_RESPONSE, "strong": block.is_strong(quality_field)})
            cells.append(cell)
            block_of.append(j)
    return rows, cells, block_of


def decision_rows(table: BlockTable, domain: str, quality_field: str, settings: Mapping[str, Any],
                  format_fn: Callable[[str, str], Any]
                  ) -> Tuple[List[Dict[str, Any]], List[Any], List[Optional[Cell]], List[int]]:
    """The cross-marker arm's approve and decline items of every block (its paraphrase index, its unmarked control):
    rows, conversations, each row's cell (None: unmarked) and block."""
    rows, convs, cells, block_of = [], [], [], []
    for j, block in enumerate(table.blocks):
        strong = block.is_strong(quality_field)
        for item in build_block_items(block, domain, seed=settings["seed"], n_paraphrases=settings["paraphrases"],
                                      include_unmarked=settings["include_unmarked"], responses=X_RESPONSES):
            rows.append({"record_id": block.record_id, "template_id": block.template_id, "encoding": block.encoding,
                         "cell": "unmarked" if item.cell is None else list(item.cell), "response": item.response,
                         "paraphrase": item.paraphrase, "strong": strong})
            convs.append(format_fn(item.prompt, item.text))
            cells.append(item.cell)
            block_of.append(j)
    return rows, convs, cells, block_of


def build_placement(model: Any, name: str, rows: List[Dict[str, Any]], H: torch.Tensor, dtype: Any,
                    gates: Optional[torch.Tensor], cells: Sequence[Optional[Cell]], block_of: Sequence[int],
                    table: BlockTable, design: FactorialDesign, encoding: str, name_folds: Mapping[str, int],
                    gate_fixed: bool = True) -> Placement:
    """``gate_fixed`` (direct placement of a gated head): whether to list the gate-fixed mode — the decision
    placement of the same target has none without an unmarked control, and the two list the same modes."""
    index = RewardIndex(rows, design, encoding)
    fixed = None
    if gates is not None:
        if name == "direct":                    # one prompt for every row: the gate is fixed already
            fixed = gates if gate_fixed else None
        else:
            try:
                fixed = gates[torch.tensor(gate_refs("cross", rows), dtype=torch.long)]
            except ValueError:
                logger.info("gate_fixed skipped: the decision rows have no unmarked control to take the gate from")
    counts = np.ones(len(index.records), dtype=np.int64) if name == "direct" else index.strong.astype(np.int64)
    keys = {axis: row_keys(cells, block_of, table, design, encoding, axis, name_folds)
            for axis in encoding_axes(design, encoding)}
    return Placement(name, rows, H, dtype, gates, fixed, index, keys, counts,
                     get_head(model).effective_weights(gates).mean(0))


def nulled_rewards(model: Any, pl: Placement, keys: Sequence[Key], gates: Optional[torch.Tensor],
                   basis_of: Callable[[Key], torch.Tensor]) -> np.ndarray:
    """The placement's rewards with each row's basis projected out (rows grouped by hold-out key)."""
    groups: Dict[Key, List[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    out = np.empty(len(keys))
    for key, idx in groups.items():
        sel = torch.tensor(idx, dtype=torch.long)
        _, r = rewards_from_hidden(model, pl.H[sel], pl.dtype, basis_of(key), gates=None if gates is None else gates[sel])
        out[idx] = r.double().numpy()
    return out


def target_tables(model: Any, pl: Placement, target: Source, kinds: Mapping[str, Kind],
                  stores: Mapping[Source, UnitStore], designs: Mapping[str, FactorialDesign], random: torch.Tensor,
                  draws: np.ndarray, cache: Dict[Any, np.ndarray]) -> Dict[str, Any]:
    """``{gate mode: cell_table}`` of one target on one placement. ``cache`` (per placement) keeps each row's
    rewards: a source's are the same for every target axis where the rows carry no names."""
    axis = target[2]
    keys = pl.keys[axis]
    named = any(folds for folds, _ in keys)
    out: Dict[str, Any] = {}
    for mode, gates in pl.modes():
        twin = next((m for m, g in pl.modes() if m in out and g is gates), None)
        if twin is not None:        # the direct placement's gate is fixed already: one table for both modes
            out[mode] = out[twin]
            continue
        if ("baseline", mode) not in cache:
            cache[("baseline", mode)] = rewards_from_hidden(model, pl.H, pl.dtype, None, gates=gates)[0].double().numpy()
        base = pl.effect(cache[("baseline", mode)], axis)
        nulled: Dict[str, np.ndarray] = {}
        for row, kind in kinds.items():
            at = (kind, mode, axis if named else None)
            if at not in cache:
                cache[at] = nulled_rewards(model, pl, keys, gates,
                                           lambda key: row_basis(kind, key, stores, designs, random))
            nulled[row] = pl.effect(cache[at], axis)
        if not pl.counts.any():
            logger.warning("%s/%s: no record counts for this target (no strong record); its cells are NaN",
                           source_name(target), pl.name)
        out[mode] = cell_table((base, pl.counts), nulled, source_name(target), draws)
    return out


def paired_wins(diffs: torch.Tensor, units: UnitIndex, record_at: Mapping[str, int],
                basis_of: Callable[[Key], torch.Tensor], tol: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Per record (``record_at``: record -> position) the sum over its pairs of win(proj(A) − proj(B)) — 1 if the
    direction ``basis_of(key)`` orders the pair's poles, ½ within ``tol`` of a tie — and the number of pairs."""
    groups: Dict[Key, List[int]] = defaultdict(list)
    for i, key in enumerate(units.keys()):
        groups[key].append(i)
    wins = np.empty(len(units.a))
    for key, idx in groups.items():
        proj = (diffs[idx] @ basis_of(key).float()).double().numpy()
        wins[idx] = np.where(proj > tol, 1.0, np.where(proj < -tol, 0.0, 0.5))
    sums, counts = np.zeros(len(record_at)), np.zeros(len(record_at), dtype=np.int64)
    for w, r in zip(wins, units.records):
        sums[record_at[r]] += w
        counts[record_at[r]] += 1
    return sums, counts


def geometry(pl: Placement, target: Source, delta: Optional[torch.Tensor], stores: Mapping[Source, UnitStore],
             reliability: Mapping[Source, float]) -> Dict[str, Any]:
    """Full-data directions (descriptive): per source its cosine with the target's own direction and that cosine's
    ceiling, its head alignment, and the share of the target's mean effect w·Δ it carries (``delta``: the target's
    mean state contrast; None where there is none)."""
    own = stores[target].direction()
    nan = float("nan")
    return {"effect_from_states": nan if delta is None else float(pl.w @ delta),
            "sources": {source_name(s): {
                "cosine_to_own": cmd.cosine(store.direction(), own),
                "cosine_ceiling": cmd.cosine_ceiling(reliability[s], reliability[target]),
                "share": nan if delta is None else cmd.did_share(pl.w, delta, store.direction()),
                **cmd.head_alignment(pl.w, store.direction())} for s, store in stores.items()}}


def decision_delta(pl: Placement, design: FactorialDesign, encoding: str, axis: str) -> Optional[torch.Tensor]:
    """Δ_int of the strong records: the marker × decision interaction of the states, whose reward w·Δ_int is the
    decision disparity (`probes.cross_marker_directions.record_contrasts`). None without a strong record."""
    ids = [r for r, strong in zip(pl.index.records, pl.index.strong) if strong]
    if not ids:
        return None
    return cmd.record_contrasts(pl.H, cmd.state_index(pl.rows), ids, "interaction", design, encoding, axis).mean(0)


def score_domain(model: Any, tokenizer: Any, d: Domain, stores: Mapping[Source, UnitStore],
                 lexical: Mapping[str, Mapping[Source, UnitStore]], features: ClauseFeatures,
                 designs: Mapping[str, FactorialDesign], reliability: Mapping[Source, float],
                 name_folds: Mapping[str, int], random: torch.Tensor, n_random: int) -> List[Dict[str, Any]]:
    """Every target of one domain: embed its evaluated records in both placements, then the tables and read-outs."""
    format_fn = lambda prompt, response: format_conversation(tokenizer, prompt, response)
    embed = lambda convs: embed_with_gates(model, tokenizer, convs, batch_size=d.cfg.batch_size,
                                           max_length=d.cfg.max_length, show_progress=False)
    n_boot, seed = int(d.settings["n_boot"]), int(d.settings["seed"])
    records = list(d.selected)
    record_at = {r: i for i, r in enumerate(records)}
    draws = shared_draws(len(records), n_boot, seed)
    out: List[Dict[str, Any]] = []
    for encoding in d.encodings:
        table = BlockTable.of(d.selected, encoding, d.design)
        rows, convs, cells, block_of = decision_rows(table, d.name, d.dom.quality_field, d.settings, format_fn)
        Hx, dtype, gates = embed(convs)
        decision = build_placement(model, "decision", rows, Hx, dtype, gates, cells, block_of, table, d.design,
                                   encoding, name_folds)
        rows, cells, block_of = direct_rows(table, d.design, d.dom.quality_field)
        H, dtype, gates = embed(table.direct_convs(d.design, d.prompt, tokenizer))
        direct = build_placement(model, "direct", rows, H, dtype, gates, cells, block_of, table, d.design, encoding,
                                 name_folds, gate_fixed=decision.fixed is not None)
        for pl in (direct, decision):
            if pl.index.records != records:
                raise ValueError(f"{d.name}/{encoding}/{pl.name}: the rows' records are not the selection's")
        clause_vectors = {c: features.vectors(c, table.clauses(d.design)) for c in CONTROLS}
        caches: Dict[str, Dict[Any, np.ndarray]] = {"direct": {}, "decision": {}}
        for axis in encoding_axes(d.design, encoding):
            target = (d.name, encoding, axis)
            kinds = row_kinds(target, stores, designs, n_random)
            units = UnitIndex.of(table, d.design, encoding, axis, name_folds)
            relations = {}
            for row, kind in kinds.items():
                relations[row] = row_relation(kind, target, designs)
                n_fit = row_fit_units(kind, units.keys(), stores, designs)
                if n_fit is not None:
                    relations[row]["n_fit"] = n_fit
            diffs = units.diffs(H)
            control_diffs = {c: units.diffs(X) for c, X in clause_vectors.items()}
            paired: Dict[str, Any] = {}
            controls: Dict[str, Dict[str, float]] = {c: {} for c in CONTROLS}
            for row, kind in kinds.items():
                if kind[0] in ("random", "singles_joint"):      # no single fitted direction to read a pair along
                    continue
                paired[row] = ratio_summary(*paired_wins(
                    diffs, units, record_at, lambda key: row_basis(kind, key, stores, designs)), draws)
                for c in CONTROLS:
                    sums, counts = paired_wins(control_diffs[c], units, record_at,
                                               lambda key: row_basis(kind, key, lexical[c], designs), tol=1e-9)
                    controls[c][row] = float(sums.sum() / counts.sum())
            joint = [source_name(s) for s in single_axes(target, stores, designs)] if axis == INTERSECTION else None
            for pl in (direct, decision):
                delta = diffs.mean(0) if pl.name == "direct" else decision_delta(pl, d.design, encoding, axis)
                entry = {"domain": d.name, "encoding": encoding, "axis": axis, "placement": pl.name,
                         "own_row": source_name(target), "rows": relations,
                         "tables": target_tables(model, pl, target, kinds, stores, designs, random, draws,
                                                 caches[pl.name]),
                         "paired_acc": paired if pl.name == "direct" else None,
                         **{key: controls[c] if pl.name == "direct" else None for c, key in CONTROLS.items()},
                         "geometry": geometry(pl, target, delta, stores, reliability)}
                if joint is not None:
                    entry["singles_joint_axes"] = joint
                out.append(entry)
    return out


# --------------------------------------------------------------------------- report ------------------
def _fmt(s: Mapping[str, Any]) -> str:
    v = s.get("mean")
    if v is None or v != v:
        return f"{'—':>24}"
    return f"{v:+.3f} [{s['ci_low']:+.3f},{s['ci_high']:+.3f}]".rjust(24)


def print_report(result: Mapping[str, Any]) -> None:
    print("\n" + "=" * 132)
    print(f"DEMOGRAPHIC TRANSFER across domains and encodings — {result['model']}  ({', '.join(result['domains'])})")
    print("per target: the baseline effect, then per transfer row what nulling changed, the shortfall against the own "
          "row (positive = removed less)\nand, on the direct target, the paired accuracy of the direction against its "
          "bag-of-words and bag-of-tokens controls")
    print("=" * 132)
    for t in result["targets"]:
        mode = next(iter(t["tables"]))
        table = t["tables"][mode]
        own = table["rows"][t["own_row"]]
        print(f"{t['own_row']:34} {t['placement']:9} baseline {_fmt(table['baseline'])}   own change {_fmt(own['change'])}"
              + (f"   [{mode}]" if mode != "own_gates" else ""))
        for row, rel in t["rows"].items():
            if rel["relation"] not in TRANSFER + EXTRA_ROWS:
                continue
            cell = table["rows"][row]
            line = f"    {rel['relation']:16} {row:32} change {_fmt(cell['change'])}  shortfall {_fmt(cell['shortfall'])}"
            if t["paired_acc"] and row in t["paired_acc"]:
                line += (f"  acc {t['paired_acc'][row]['mean']:.2f} (words {t['lexical_paired_acc'][row]:.2f}, "
                         f"tokens {t['token_paired_acc'][row]:.2f})")
            print(line + ("  *proxy construct differs" if rel.get("proxy_construct_differs") else ""))
    print("-" * 132)
    print("Exploratory, uncorrected intervals. Read a shortfall only where the own change differs from 0; component, "
          "off-axis, unrelated and random rows are in the JSON.")
    timing = result.get("timing")
    if timing:      # the line cluster/pilot.sh greps
        phases = " | ".join(f"{k} {v:.0f}s" for k, v in timing["seconds"].items())
        print(f"timing (batch {timing['batch_size']}): {phases} | total {timing['total_s']:.0f}s")
    print("=" * 132)


# --------------------------------------------------------------------------- main --------------------
def default_out(model_path: str, variant: str = "") -> Path:
    return RESULTS_DIR / f"demographic_transfer_{Path(model_path).name}{variant}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", type=Path, nargs="+", default=DEFAULT_CONFIGS,
                    help="One cross-marker config per domain (default: credit, hiring, education)")
    ap.add_argument("--n-strong", type=int, nargs="+", default=None,
                    help="Strong records evaluated: one value, or one per config (default: the configs')")
    ap.add_argument("--n-weak", type=int, nargs="+", default=None, help="Weak records, as --n-strong")
    ap.add_argument("--encodings", default=None, help="Comma-separated, e.g. explicit,proxy")
    ap.add_argument("--paraphrases", type=int, default=None, help="The cross-marker arm's paraphrase setting")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="Seeds the record choice, the name folds and the bootstrap")
    ap.add_argument("--name-folds", type=int, default=5, help="Folds the proxy first names are held out in")
    ap.add_argument("--random-draws", type=int, default=5, help="Random unit directions (the floor rows)")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"Default {RESULTS_DIR}/demographic_transfer_{{model}}{{variant}}.json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result (and its directions file)")
    add_override_args(ap)
    return ap


def domain_overrides(args: argparse.Namespace, n: int) -> List[Dict[str, Any]]:
    """The cross-marker settings the CLI overrides, per config (``None`` = not given)."""
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None
    out = [{"encodings": split(args.encodings), "paraphrases": args.paraphrases, "n_boot": args.n_boot,
            "seed": args.seed} for _ in range(n)]
    for key in PER_DOMAIN:
        values = getattr(args, key)
        if values is None:
            continue
        if len(values) not in (1, n):
            raise SystemExit(f"--{key.replace('_', '-')} takes one value or one per config ({n}), got {len(values)}")
        for i in range(n):
            out[i][key] = values[0] if len(values) == 1 else values[i]
    return out


def _one(values: Sequence[Any]) -> Any:
    """The value all domains share, or the list of their values."""
    return values[0] if all(v == values[0] for v in values) else list(values)


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    timer = rx.PhaseTimer()
    configured = [ExperimentConfig.from_yaml(p) for p in args.configs]
    cfgs = [apply_overrides(ExperimentConfig.from_yaml(p), args) for p in args.configs]
    overrides = domain_overrides(args, len(cfgs))
    # everything that can fail on the inputs fails here, before the model loads
    if len({(c.model_path, c.model_revision) for c in cfgs}) != 1:
        raise SystemExit("the configs must name one model and revision (directions live in one model's states): "
                         f"{sorted({(c.model_path, str(c.model_revision)) for c in cfgs})}; use --model/--revision")
    differing = [k for k in SHARED if len({str(getattr(c, k)) for c in cfgs}) != 1]
    if differing:
        raise SystemExit(f"the configs differ in {differing}, which this runner takes from the first config only")
    if args.name_folds <= MAX_PAIR_NAMES:
        raise SystemExit(f"--name-folds must exceed {MAX_PAIR_NAMES}: a pair's two names can sit in two folds, and "
                         f"a fit needs a fold left")
    if args.random_draws < 0:
        raise SystemExit("--random-draws must not be negative")
    seeds = {int(rx.resolve_settings(c.extra, o)["seed"]) for c, o in zip(cfgs, overrides)}
    if len(seeds) != 1:
        raise SystemExit(f"the configs' cross-marker seeds differ ({sorted(seeds)}); pass --seed")
    seed = seeds.pop()
    name_folds = shared_name_folds(args.name_folds, seed)
    domains = [Domain(c, o, name_folds, args.name_folds) for c, o in zip(cfgs, overrides)]
    names = [d.name for d in domains]
    if len(set(names)) != len(names):
        raise SystemExit(f"--configs needs configs of distinct domains, got {names}")
    if len({int(d.settings["n_boot"]) for d in domains}) != 1:
        raise SystemExit("the configs' n_boot differ; pass --n-boot")
    if any(int(d.settings["n_strong"]) < 1 for d in domains):
        raise SystemExit("n_strong must be at least 1: the decision target reads the strong records")
    used_keys = ("n_strong", "n_weak", "encodings", "paraphrases", "seed", "n_boot")
    plain = [rx.resolve_settings(c.extra, {}) for c in configured]
    variant = variant_suffix(
        {**{k: _one([s[k] for s in plain]) for k in used_keys},
         "probe_records": _one([c.probe_records for c in configured]), "revision": configured[0].model_revision,
         "name_folds": ap.get_default("name_folds"), "random_draws": ap.get_default("random_draws"),
         "domains": DEFAULT_DOMAINS, "manifests": [Path(d.dom.default_pairs).parent.name for d in domains]},
        {**{k: _one([d.settings[k] for d in domains]) for k in used_keys},
         "probe_records": _one([c.probe_records for c in cfgs]), "revision": cfgs[0].model_revision,
         "name_folds": args.name_folds, "random_draws": args.random_draws, "domains": names,
         "manifests": [Path(d.source).parent.name for d in domains]})
    out = args.out or default_out(cfgs[0].model_path, variant)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it (and its directions file), or --out")
    designs = {d.name: d.design for d in domains}
    data = {k: v for d in domains for k, v in d.data.items()}
    timer.lap("setup")

    exp = DemographicBiasExperiment(cfgs[0])
    exp.load_model()
    model, tok = exp.model, exp.tokenizer
    for c in cfgs[1:]:
        c.model_revision = cfgs[0].model_revision            # the commit actually loaded, for every config's record
    cache = getattr(model, "_onejudge_embedding_cache", None)
    timer.lap("load_model")

    # the sources: every domain's directions, and the same contrasts of the marker clauses (the lexical controls)
    features = ClauseFeatures(tok, sorted({c for d in domains for t in d.fit_tables.values()
                                           for c in t.clauses(d.design)}))
    stores: Dict[Source, UnitStore] = {}
    lexical: Dict[str, Dict[Source, UnitStore]] = {c: {} for c in CONTROLS}
    for d in domains:
        for encoding, table in d.fit_tables.items():
            print(f"[transfer] fitting {d.name}/{encoding} ...", flush=True)
            H, _ = embed_states(model, tok, table.direct_convs(d.design, d.prompt, tok), batch_size=d.cfg.batch_size,
                                max_length=d.cfg.max_length, show_progress=False)
            X = {c: features.vectors(c, table.clauses(d.design)) for c in CONTROLS}
            for axis in encoding_axes(d.design, encoding):
                source = (d.name, encoding, axis)
                stores[source] = d.fit_units[source].store(H)
                for c in CONTROLS:
                    lexical[c][source] = d.fit_units[source].store(X[c])
    split_half = {s: cmd.split_half_cosine(store.record_contrasts, seed) for s, store in stores.items()}
    reliability = {s: cmd.full_sample_reliability(v) for s, v in split_half.items()}
    random = random_units(next(iter(stores.values())).sums.shape[1], args.random_draws, seed)
    timer.lap("fit")

    targets: List[Dict[str, Any]] = []
    for d in domains:
        d.select(tok)
        print(f"[transfer] scoring {d.name}: {len(d.selected)} records ...", flush=True)
        targets += score_domain(model, tok, d, stores, lexical, features, designs, reliability, name_folds, random,
                                args.random_draws)
        timer.lap("score")

    settings = {"domains": names, "name_folds": args.name_folds, "random_draws": args.random_draws, "seed": seed,
                "cross_marker": {d.name: {**d.settings, "templates": d.templates} for d in domains},
                "configs": {d.name: d.cfg.to_dict() for d in domains}}
    result = {
        "meta": run_metadata(cfgs[0], data, settings), "model": cfgs[0].model_path, "domains": names,
        "settings": settings, "name_folds": name_folds,
        "selection": {d.name: {"fit": d.fit_report, "evaluated": d.selection} for d in domains},
        "records": {d.name: {
            "strong": [r for r, b in d.selected.items() if b[0].is_strong(d.dom.quality_field)],
            "weak": [r for r, b in d.selected.items() if not b[0].is_strong(d.dom.quality_field)]} for d in domains},
        "sources": {source_name(s): {
            "n_units": store.n_units, "n_records": store.n_records, "templates": store.templates,
            "named": store.named, "separation": store.separation,
            "min_fit_units": min_fit_units(domains[names.index(s[0])].fit_units[s], args.name_folds),
            "split_half_cosine": split_half[s], "reliability": reliability[s]} for s, store in stores.items()},
        "targets": targets,
        "embedding_cache": None if cache is None else {"hits": cache.hits, "misses": cache.misses},
        "timing": rx.timing_report(timer, batch_size=cfgs[0].batch_size, texts=None),
        "caveats": [
            "Exploratory: uncorrected intervals until the headline family is fixed.",
            "A row's bootstrap resamples the target's records; the source direction is held fixed.",
            "Held-out fits rest on fewer pairs than the battery's direction, and on fewer the more names a source's "
            "pairs carry (rows[*].n_fit): for a proxy target the explicit-source rows rest on more pairs than the own "
            "row. own_seen is the battery's.",
            "Proxy targets: the intervals resample records with the first names held fixed (they condition on the "
            "name pools), and names share sub-word tokens across folds (token_paired_acc).",
            "Geometry uses full-data directions and is descriptive; for a gated head it uses the mean effective head.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"full_data": {source_name(s): store.direction() for s, store in stores.items()},
                "name_folds": name_folds}, f"{out.with_suffix('')}_directions.pt")
    out.write_text(json.dumps(result, indent=2))
    print_report(result)
    print(f"saved → {out} (+ _directions.pt)")


if __name__ == "__main__":
    main()
