#!/usr/bin/env python3
"""
Cross-influence (EU-AI-Act reliability harm) on the credit arm, one RM (default Qwen3-0.6B).

Pairs a **good-credit** record (strong) with a **bad-credit** record (weak) from German Credit, then
asks: does attaching the **protected** demographic marker to the *weaker* applicant degrade the RM's
ability to still rank the stronger one first?

Per axis × encoding it renders 5 variants per pair (strong_neutral, weak_neutral, weak_protected,
weak_reference, strong_protected), scores them **baseline and null-space-projected** in one pass
(reusing the axis difference-of-means probe), and reports:
  acc_baseline      = P(strong_neutral > weak_neutral)   [does the RM track creditworthiness at all?]
  cross_influence   = acc_baseline − P(strong_neutral > weak_protected)   [signed reliability harm]
plus reference-pole and marker-on-strong controls — for baseline vs nulled scores.

**Read `protected_vs_reference` as the headline** = P(strong > weak_reference) − P(strong > weak_protected).
`cross_influence` (and `cross_influence_reference`) set a MARKED weak text against an UNMARKED strong one,
so they also move when the RM merely reacts to an extra clause or to any demographic statement; the two
marked weak variants carry length-matched clauses, so their difference isolates the protected pole.

**Pairs are length-matched** (default, since 2026-09-23). Strong records are systematically longer in
every domain but hiring — education essays by so much that length alone predicts the label with AUC
0.98, credit profiles through the checking-account wording (AUC 0.72) — so a reward model that merely
prefers longer text would score a high `acc_baseline` and pass as "tracking quality". Each strong record
is paired with the nearest-length unused weak record within a relative-length caliper, and the
direction alternates STRICTLY: every other pair has the strong text longer, the rest have it shorter
(exact ties are excluded — they sit on one side of the alternation and tilt it). So a length-only scorer
gets `length_only_accuracy` = 0.5 on the pair set by construction; the run reports it next to
`acc_baseline` together with how far the matched pairs sit from their pools (matching draws from the
overlap of the two length distributions, which in education means unusually short strong essays and
long weak ones). `--pairing random` restores the old unmatched pairing for comparison.

Usage:
    python experiments/run_crossinfluence.py --config configs/demographic_credit_sex_qwen06.yaml \
        --encoding explicit --n-pairs 300
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scoring.dataset_base import format_conversation
from substrates.domains import get_domain
from substrates.bios_clean import load_factorial_bios
from substrates.bios_ingest import DEFAULT_BIOS_PATH
from substrates.credit_clean import load_factorial_records
from substrates.education_clean import load_education_essays
from scoring.experiment import ExperimentConfig
from scoring.demographic_experiment import DemographicBiasExperiment, compute_cross_influence
from probes.probe import build_probe_direction, get_rewards_both

EDU_SOURCES = ("persuade", "asap")
VARIANTS = ["strong_neutral", "weak_neutral", "weak_protected", "weak_reference", "strong_protected"]


# Relative length difference allowed within a matched pair: |len_s - len_w| <= CALIPER * max(len_s, len_w).
DEFAULT_CALIPER = 0.10
# Records per class considered for matching (a seeded sample). Keeps tokenising the hiring pool (~60k
# bios) cheap; credit and education fall below it and are used whole.
POOL_CAP_PER_PAIR = 20


def _build_pairs(records, n_pairs, seed, is_strong, template_ids):
    """Random strong/weak pairing (``--pairing random``). NOT length-controlled: see the module docstring."""
    strong = [r for r in records if is_strong(r)]
    weak = [r for r in records if not is_strong(r)]
    rng = random.Random(seed)
    rng.shuffle(strong)
    rng.shuffle(weak)
    n = min(n_pairs, len(strong), len(weak))
    tids = list(template_ids)
    return [(strong[i], weak[i], tids[i % len(tids)]) for i in range(n)]


def _build_length_matched_pairs(
    records: List[Any],
    n_pairs: int,
    seed: int,
    is_strong: Callable[[Any], bool],
    template_ids,
    length: Callable[[Any, str], int],
    *,
    stratum: Optional[Callable[[Any], Any]] = None,
    caliper: float = DEFAULT_CALIPER,
    pool_cap: Optional[int] = None,
) -> List[Tuple[Any, Any, str]]:
    """Length-matched, direction-balanced strong/weak pairs (see the module docstring).

    Strong records are visited in seeded random order. Pair *k* needs a weak partner that is strictly
    LONGER than the strong record for even *k* and strictly SHORTER for odd *k*; of the eligible unused
    weak records in the same ``stratum`` the nearest in length is taken, if it lies within ``caliper``.
    A strong record with no such partner is skipped and the direction stays, so the alternation is
    exact. Lengths are measured under the first template (``length(record, template_id)``); templates are
    then assigned round-robin, both members sharing one, as before. ``pool_cap`` limits each class to a
    seeded sample before any length is computed.
    """
    rng = random.Random(seed)
    strong = [r for r in records if is_strong(r)]
    weak = [r for r in records if not is_strong(r)]
    rng.shuffle(strong)
    rng.shuffle(weak)
    if pool_cap:
        strong, weak = strong[:pool_cap], weak[:pool_cap]
    key = stratum or (lambda r: None)
    ref = template_ids[0]
    # per stratum: weak records sorted by length (ties in seeded order), with a parallel length list
    avail: Dict[Any, Tuple[List[int], List[Any]]] = {}
    for w in sorted(weak, key=lambda r: length(r, ref)):
        lens, recs = avail.setdefault(key(w), ([], []))
        lens.append(length(w, ref))
        recs.append(w)
    tids = list(template_ids)
    pairs: List[Tuple[Any, Any, str]] = []
    for s in strong:
        if len(pairs) >= n_pairs:
            break
        if key(s) not in avail:
            continue
        lens, recs = avail[key(s)]
        ls = length(s, ref)
        if len(pairs) % 2 == 0:              # weak strictly longer: shortest length > ls
            j = bisect.bisect_right(lens, ls)
        else:                                 # weak strictly shorter: longest length < ls
            j = bisect.bisect_left(lens, ls) - 1
        if not 0 <= j < len(lens) or abs(lens[j] - ls) > caliper * max(ls, lens[j]):
            continue
        w = recs.pop(j)
        lens.pop(j)
        pairs.append((s, w, tids[len(pairs) % len(tids)]))
    return pairs


def _pairing_report(pairs, records, is_strong, length, *, mode: str, caliper: Optional[float],
                    requested: int, stratified: bool) -> Dict[str, Any]:
    """How much length can explain in the pair set actually scored (``length_only_accuracy`` and the
    gap use each pair's own template, i.e. the texts the model sees), and how the matched records
    compare with the pools they were drawn from (all medians under ONE template, so templates of
    different length do not masquerade as a selection effect)."""
    ls = [length(s, t) for s, _, t in pairs]
    lw = [length(w, t) for _, w, t in pairs]
    n = len(pairs)
    longer = sum(a > b for a, b in zip(ls, lw)) + 0.5 * sum(a == b for a, b in zip(ls, lw))
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    ref = pairs[0][2] if pairs else None
    pool_s = [length(r, ref) for r in records if is_strong(r)] if ref else []
    pool_w = [length(r, ref) for r in records if not is_strong(r)] if ref else []
    pair_s = [length(s, ref) for s, _, _ in pairs]
    pair_w = [length(w, ref) for _, w, _ in pairs]
    return {
        "mode": mode, "caliper": caliper, "stratified": stratified,
        "n_pairs": n, "n_requested": requested,
        # what a reward model that only prefers the longer text would score as acc_baseline
        "length_only_accuracy": longer / n if n else float("nan"),
        "median_relative_length_gap": med([abs(a - b) / max(a, b) for a, b in zip(ls, lw)]),
        "median_length_strong": med(pair_s), "median_length_weak": med(pair_w),
        "pool_median_length_strong": med(pool_s), "pool_median_length_weak": med(pool_w),
    }


def run_axis(exp, cfg, dom, axis, encoding, pairs, rng, subject="applicant") -> Dict[str, Any]:
    tok = exp.tokenizer
    fmt = lambda txt: format_conversation(tok, dom.assessment_prompt, txt)
    render = dom.render_fn
    texts: Dict[str, List[Any]] = {v: [] for v in VARIANTS}
    for good, bad, tid in pairs:
        spec = dom.make_marker(axis, encoding, rng, subject)  # clause_a = protected pole, clause_b = reference
        texts["strong_neutral"].append(fmt(render(good, tid, "")))
        texts["weak_neutral"].append(fmt(render(bad, tid, "")))
        texts["weak_protected"].append(fmt(render(bad, tid, spec.clause_a)))
        texts["weak_reference"].append(fmt(render(bad, tid, spec.clause_b)))
        texts["strong_protected"].append(fmt(render(good, tid, spec.clause_a)))

    # axis difference-of-means probe (from the existing matched-pair manifest)
    ds = dom.dataset_cls(cfg.dataset_source, axis=axis, encoding=encoding,
                         probe_size=cfg.probe_size, split_seed=cfg.split_seed)
    probe, _ = build_probe_direction(exp.model, tok, ds.get_probe_pairs(tok),
                                     batch_size=cfg.batch_size, device=cfg.device,
                                     max_length=cfg.max_length)

    # score all variants at once → baseline + nulled
    n = len(pairs)
    flat = [t for v in VARIANTS for t in texts[v]]
    base, nulled = get_rewards_both(exp.model, tok, flat, probe, batch_size=cfg.batch_size,
                                    device=cfg.device, max_length=cfg.max_length,
                                    null_alpha=1.0, show_progress=False)
    base_by = {v: base[i * n:(i + 1) * n].tolist() for i, v in enumerate(VARIANTS)}
    null_by = {v: nulled[i * n:(i + 1) * n].tolist() for i, v in enumerate(VARIANTS)}
    return {"axis": axis, "encoding": encoding, "n_pairs": n,
            "baseline": compute_cross_influence(base_by),
            "nulled": compute_cross_influence(null_by)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_credit_sex_qwen06.yaml"))
    ap.add_argument("--encoding", default="explicit", choices=["explicit", "proxy"])
    ap.add_argument("--axes", default=None, help="Comma-separated axes; default is domain-appropriate.")
    ap.add_argument("--dataset-source", default=None, help="Override the matched-pair manifest (probe pairs).")
    ap.add_argument("--source", default="persuade", choices=EDU_SOURCES,
                    help="Education only: which corpus to load strong/weak records from.")
    ap.add_argument("--raw-path", default=None,
                    help="Override the corpus file path (credit: german.data; cv/education: the corpus).")
    ap.add_argument("--n-pairs", type=int, default=300)
    ap.add_argument("--pairing", choices=["length_matched", "random"], default="length_matched",
                    help="length_matched (default): nearest-length, direction-balanced pairs, so a "
                         "length-only scorer gets 0.5. random: the old unmatched pairing.")
    ap.add_argument("--caliper", type=float, default=DEFAULT_CALIPER,
                    help="Max relative length difference within a length-matched pair.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="Defaults to artifacts/results/demographic/crossinf_{domain}_qwen06.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.dataset_source:
        cfg.dataset_source = args.dataset_source
    dom = get_domain(cfg.extra.get("domain", "credit"))
    subject = "student" if dom.name == "education" else "applicant"
    axes = [a.strip() for a in args.axes.split(",")] if args.axes else list(dom.axes)
    out = args.out or Path(f"artifacts/results/demographic/crossinf_{dom.name}_qwen06.json")
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    # Every domain honours --raw-path (and education also --source); the registry loader is zero-arg,
    # so the override has to happen here.
    if dom.name == "education":
        # The shared education pool's essays. Length-matched pairs are drawn within prompt, which
        # already makes the label uninformative about the prompt, so they use the pool before its
        # within-prompt class balancing (which discards most weak essays, and with them most of the
        # long weak essays a length match needs: ~58 matched pairs from the balanced pool vs ~154).
        records = load_education_essays(args.raw_path, source=args.source,
                                        balance=args.pairing == "random")
    elif dom.name == "cv":
        records = load_factorial_bios(args.raw_path or DEFAULT_BIOS_PATH)
    elif dom.name == "credit":
        # same population as the registry loader (both consistency rule sets), from the given file
        records = load_factorial_records(args.raw_path)
    else:
        records = dom.load_records()
    # Length as the reward model sees it: tokens of the rendered neutral text (the assessment prompt
    # around it is the same for both members). Cached: the report re-reads it.
    cache: Dict[Tuple[int, str], int] = {}

    def length(record: Any, tid: str) -> int:
        k = (id(record), tid)
        if k not in cache:
            text = dom.render_fn(record, tid, "")
            cache[k] = len(exp.tokenizer(text, add_special_tokens=False)["input_ids"])
        return cache[k]

    if args.pairing == "length_matched":
        pairs = _build_length_matched_pairs(
            records, args.n_pairs, args.seed, dom.is_strong, dom.template_ids, length,
            stratum=dom.pair_stratum, caliper=args.caliper,
            pool_cap=POOL_CAP_PER_PAIR * args.n_pairs)
    else:
        pairs = _build_pairs(records, args.n_pairs, args.seed, dom.is_strong, dom.template_ids)
    # pool medians from a seeded sample, so the ~60k-bio hiring pool is not tokenised in full
    pool_sample = random.Random(args.seed).sample(records, min(len(records), 4000))
    pairing_report = _pairing_report(
        pairs, pool_sample, dom.is_strong, length,
        mode=args.pairing, caliper=args.caliper if args.pairing == "length_matched" else None,
        requested=args.n_pairs, stratified=dom.pair_stratum is not None)
    print(f"[cross-influence] pairing: {pairing_report}", flush=True)
    if len(pairs) < args.n_pairs:
        print(f"[cross-influence] WARNING: only {len(pairs)} of {args.n_pairs} pairs could be "
              f"length-matched at caliper {args.caliper}", flush=True)
    pairing = [{"strong": g.source_record_id, "weak": b.source_record_id, "template_id": t}
               for g, b, t in pairs]
    rng = random.Random(args.seed)

    results = []
    for axis in axes:
        print(f"[cross-influence] {dom.name}/{axis}/{args.encoding} ...", flush=True)
        results.append(run_axis(exp, cfg, dom, axis, args.encoding, pairs, rng, subject))

    print("\n" + "=" * 96)
    print(f"CROSS-INFLUENCE [{dom.name}] — {cfg.model_path}  "
          f"(encoding={args.encoding}, n_pairs={len(pairs)}, pairing={args.pairing}, "
          f"length-only acc={pairing_report['length_only_accuracy']:.3f})")
    print("=" * 96)
    print(f"{'axis':14} {'acc_base':>9} {'tracks?':>8} | {'PvR_base':>8} {'PvR_null':>8} | "
          f"{'CI_base':>8} {'CI_null':>8} {'CI_ref':>8} {'mark_on_strong':>15}")
    for r in results:
        b, nl = r["baseline"], r["nulled"]
        print(f"{r['axis']:14} {b['acc_baseline']:>9.3f} {str(b['baseline_tracks_quality']):>8} | "
              f"{b.get('protected_vs_reference', float('nan')):>8.3f} "
              f"{nl.get('protected_vs_reference', float('nan')):>8.3f} | "
              f"{b['cross_influence']:>8.3f} {nl['cross_influence']:>8.3f} "
              f"{b.get('cross_influence_reference', float('nan')):>8.3f} "
              f"{b.get('marker_on_strong_effect', float('nan')):>15.3f}")
    print("=" * 96)
    print("acc_base≈0.5 ⇒ RM doesn't track applicant quality ⇒ cross-influence not interpretable.")
    print("Read acc_base against the length-only accuracy above: only the excess over it is quality.")
    print("PvR = protected_vs_reference, the headline: CI/CI_ref also move for any clause on the weak side.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": cfg.model_path, "domain": dom.name, "encoding": args.encoding, "seed": args.seed,
         "pairing_report": pairing_report, "pairing": pairing, "results": results}, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
