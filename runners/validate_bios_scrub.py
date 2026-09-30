#!/usr/bin/env python3
"""
Did the Bias-in-Bios scrub remove the sex signal from the biography body, beyond what the occupation carries?

This exists because **the Tier-1 validation gate cannot answer that question**. The gate only checks
that the two poles of a matched pair differ by exactly the injected clause; a leaked "she" sits on
*both* sides, so it cancels in the diff and passes silently. Yet residual sex signal in the body is
precisely what would confound the sex axis — the injected marker would no longer be the only cue.

So we ask the question empirically, against the dataset's own **real** gender label, on the **manifest's pool**
(`load_factorial_bios`, the same bios, rules, profession cap and labels as `generate_bios` wrote; refused unless
the parquet and the pool's labels are the manifest's):

    extract activations for scrubbed bios -> train a linear AND a non-linear (MLP) probe to predict
    the real gender -> held-out accuracy against the majority-class chance rate, and against occupation.

Occupation predicts gender (that is what Bias-in-Bios is about), and no scrub can remove it without destroying the
substrate, so a scrubbed body is never at chance. The question is what the probe recovers **beyond the occupation**:
``linear_minus_reference`` / ``mlp_minus_reference``, the probe's accuracy minus the no-model occupation rule's
(majority gender per profession on the probe split) on the same eval bios, with its bootstrap interval over bios.
Reading: an interval above 0 ⇒ the body carries sex signal beyond the occupation (inspect the scrub); covering 0 ⇒
no evidence of more than the occupational prior. Each accuracy − chance has its interval too (`probes.erasure`). No
verdict is printed: the intervals are the result.

As a reference point the same probes are run on the **unscrubbed** bodies, which should be strongly decodable — if
they are not, the probe setup itself is broken and the scrubbed result means nothing (``--no-control`` skips it).
The target role is a design check: it is randomised for the mismatched half of the bios, so its rule should sit at
chance; if it does not, the role-match header leaks.

The result ``scrubcheck_bios_{model}.json`` (never replaced without ``--overwrite``) carries ``meta`` (the config
with the loaded model commit, the code commit, the manifest's and the parquet's SHA-256; `scoring.experiment`) and
both splits' bio ids.

Usage:
    python runners/validate_bios_scrub.py --config configs/demographic_cv_sex_qwen06.yaml \
        --probe-items 1000 --eval-items 1000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scoring.dataset_base import format_conversation
from scoring.bios_dataset import BIOS_ASSESSMENT_PROMPT
from substrates.bios_clean import DEFAULT_N_BIOS, load_factorial_bios
from substrates.bios_ingest import DEFAULT_BIOS_PATH
from substrates.bios_render import render_bio
from scoring.experiment import ExperimentConfig, add_override_args, apply_overrides, data_file, run_metadata
from scoring.demographic_experiment import DemographicBiasExperiment
from scoring.intervals import DEFAULT_N_BOOT, cluster_bootstrap
from probes.erasure import probe_recoverability
from probes.probe import get_embeddings
from runners.run_reasoning_flip import bios_source

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("artifacts/results/demographic")
LABEL_FIELDS = ("gender", "profession", "target_role", "qualified")


def check_pool(records: Sequence[Any], cells_path: Path | str) -> None:
    """Refuse unless ``records`` are the manifest's bios with the manifest's labels (its ``cells.jsonl``)."""
    manifest: Dict[str, tuple] = {}
    with open(cells_path) as f:
        for line in f:
            row = json.loads(line)
            manifest[row["source_record_id"]] = tuple(row["real_fields"][k] for k in LABEL_FIELDS)
    pool = {r.source_record_id: tuple(getattr(r, k) for k in LABEL_FIELDS) for r in records}
    if pool != manifest:
        differ = sum(pool.get(k) != v for k, v in manifest.items()) + len(set(pool) - set(manifest))
        raise SystemExit(f"the loaded pool is not the manifest's ({len(pool)} bios loaded, {len(manifest)} in "
                         f"{cells_path}, {differ} differ); regenerate the data")


def field_rule(train: Sequence[Any], evalr: Sequence[Any], key: str) -> List[bool]:
    """Per eval bio: does the no-model rule "the majority gender of this ``key`` value on the probe split" get its
    gender right? A value the probe split never saw gets the probe split's overall majority."""
    by_value: Dict[Any, Counter] = defaultdict(Counter)
    for r in train:
        by_value[getattr(r, key)][r.gender] += 1
    overall = Counter(r.gender for r in train).most_common(1)[0][0]
    rule = {k: c.most_common(1)[0][0] for k, c in by_value.items()}
    return [rule.get(getattr(r, key), overall) == r.gender for r in evalr]


def field_baselines(train: Sequence[Any], evalr: Sequence[Any], n_boot: int, seed: int) -> Dict[str, Any]:
    """The occupation rule (the irreducible prior) and the target-role rule (a design check: should sit at chance),
    each accuracy and accuracy − chance with its bootstrap interval over eval bios."""
    occupation, role = field_rule(train, evalr, "profession"), field_rule(train, evalr, "target_role")
    items = [[(o, t, r.gender)] for o, t, r in zip(occupation, role, evalr)]

    def chance(s) -> float:
        share = sum(x[2] for x in s) / len(s)
        return max(share, 1.0 - share)

    acc = lambda k: lambda s: sum(x[k] for x in s) / len(s)
    return cluster_bootstrap(items, {"chance": chance, "occupation_acc": acc(0), "target_role_acc": acc(1),
                                     "occupation_above_chance": lambda s: acc(0)(s) - chance(s),
                                     "target_role_above_chance": lambda s: acc(1)(s) - chance(s)}, n_boot, seed)


def texts(tokenizer, records: Sequence[Any], *, scrubbed: bool) -> List[str]:
    """The rendered profiles (direct-arm format); ``scrubbed=False`` restores the raw body as a control."""
    out = []
    for r in records:
        rec = r if scrubbed else dataclasses.replace(r, bio_text=str(r.extra["raw_bio"]))
        out.append(format_conversation(tokenizer, BIOS_ASSESSMENT_PROMPT, render_bio(rec, "bios_v1")))
    return out


def default_out(model_path: str) -> Path:
    return RESULTS_DIR / f"scrubcheck_bios_{Path(model_path).name}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_cv_sex_qwen06.yaml"))
    ap.add_argument("--raw-path", default=DEFAULT_BIOS_PATH)
    ap.add_argument("--probe-items", type=int, default=1000)
    ap.add_argument("--eval-items", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-control", action="store_true",
                    help="Skip the unscrubbed reference arm (faster, but loses the sanity check).")
    ap.add_argument("--out", type=Path, default=None, help=f"Default {RESULTS_DIR}/scrubcheck_bios_{{model}}.json")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing result")
    add_override_args(ap)
    return ap


def _ci(entry: Dict[str, float]) -> str:
    return f"{entry['estimate']:+.3f} [{entry['ci_low']:+.3f}, {entry['ci_high']:+.3f}]"


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    if cfg.extra.get("domain") != "cv":
        raise SystemExit(f"the scrub check reads the hiring manifest: set extra.domain: cv "
                         f"(the config has {cfg.extra.get('domain')!r})")
    # everything that can fail on the inputs fails here, before the model loads
    out = args.out or default_out(cfg.model_path)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or --out")
    manifest_dir = Path(cfg.dataset_source).parent
    data = {"pairs.jsonl": data_file(cfg.dataset_source), "cells.jsonl": data_file(manifest_dir / "cells.jsonl"),
            Path(args.raw_path).name: bios_source(cfg.dataset_source, args.raw_path)}
    records = load_factorial_bios(args.raw_path, n=DEFAULT_N_BIOS, seed=data["pairs.jsonl"]["generator_seed"],
                                  keep_raw=True)
    check_pool(records, manifest_dir / "cells.jsonl")
    need = args.probe_items + args.eval_items
    if len(records) < need:
        raise SystemExit(f"{len(records)} bios in the manifest's pool, fewer than --probe-items + --eval-items = {need}")
    random.Random(args.seed).shuffle(records)
    probe_recs, eval_recs = records[:args.probe_items], records[args.probe_items:need]
    n_boot = int(cfg.extra.get("n_boot", DEFAULT_N_BOOT))
    baselines = field_baselines(probe_recs, eval_recs, n_boot, args.seed)
    occupation_ok = field_rule(probe_recs, eval_recs, "profession")

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()

    y_tr, y_ev = [r.gender for r in probe_recs], [r.gender for r in eval_recs]
    results: Dict[str, Any] = {}
    for arm in (["scrubbed"] if args.no_control else ["scrubbed", "unscrubbed"]):
        print(f"[scrubcheck] embedding {arm} ...", flush=True)
        embed = lambda recs: get_embeddings(exp.model, exp.tokenizer, texts(exp.tokenizer, recs,
                                                                             scrubbed=arm == "scrubbed"),
                                            batch_size=cfg.batch_size, max_length=cfg.max_length)
        Xtr: torch.Tensor = embed(probe_recs)
        results[arm] = probe_recoverability(Xtr, y_tr, embed(eval_recs), y_ev, seed=args.seed, n_boot=n_boot,
                                            reference_ok=occupation_ok)

    print("\n" + "=" * 100)
    print(f"BIAS-IN-BIOS SCRUB CHECK — real gender recoverability — {cfg.model_path}  "
          f"(probe={len(probe_recs)}, eval={len(eval_recs)} bios of the manifest's pool)")
    print("=" * 100)
    print(f"{'arm':12} {'linear':>7} {'MLP':>7} {'chance':>7}  {'linear − occupation [95%]':>27}  "
          f"{'MLP − occupation [95%]':>27}")
    for arm, r in results.items():
        iv = r["intervals"]
        print(f"{arm:12} {r['linear_acc']:>7.3f} {r['mlp_acc']:>7.3f} {r['chance']:>7.3f}  "
              f"{_ci(iv['linear_minus_reference']):>27}  {_ci(iv['mlp_minus_reference']):>27}")
    print("-" * 100)
    print("No-model baselines — how much of gender is predictable from a single field (above chance):")
    print(f"  occupation   {_ci(baselines['occupation_above_chance'])}   <- irreducible: what Bias-in-Bios is about")
    print(f"  target role  {_ci(baselines['target_role_above_chance'])}   <- design check: should cover 0")
    print("=" * 100)
    print("Scrubbed: probe − occupation above 0 ⇒ sex signal beyond the occupation (inspect the scrub); covering 0 ⇒ "
          "no evidence of more than the occupational prior. Unscrubbed: should be far above chance.")

    settings = {"probe_items": args.probe_items, "eval_items": args.eval_items, "seed": args.seed, "n_boot": n_boot,
                "control": not args.no_control, "pool": {"n": DEFAULT_N_BIOS,
                                                         "seed": data["pairs.jsonl"]["generator_seed"]}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": run_metadata(cfg, data, settings), "model": cfg.model_path, "seed": args.seed,
         "probe_records": [r.source_record_id for r in probe_recs],
         "eval_records": [r.source_record_id for r in eval_recs],
         "results": results, "field_baselines": baselines}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
