#!/usr/bin/env python3
"""
First real load of each reward model on one GPU, and the batch sizes it fits at our input lengths — before any
main run commits GPU hours to it. Per model:

- the load exactly as every runner does it (`scoring.experiment.DemographicBiasExperiment.load_model`): the pinned
  revision, the attention implementation (Gemma-2: eager, which keeps the soft-cap but needs more memory), the refusal
  of a model offloaded to CPU, and `verify_score_path` (the pipeline's reward must equal the model's own score; for
  QRM the gate too);
- for each input length (default 2,048 and 4,096 tokens: the configs' max_length, education's comparative prompts
  holding two essays) and batch size 1, 2, 4, 8, ... until the first out-of-memory: the peak GPU memory and the
  forward-pass throughput on synthetic texts of exactly that length (memory and time depend on the length, not the
  words). The embedding cache is off, so every text goes through the model.

    python cluster/load_check.py                                   # the mid tier and the two untested 8B models
    python cluster/load_check.py --models nicolinho/QRM-Gemma-2-27B --lengths 2048,4096 --max-batch 16

Writes ``artifacts/results/load_check/<model>.json`` (numbers only; ``meta.code`` names the staged commit, and
``meta.script_sha256`` this file, which may run as a copy outside the staged tree) and prints one line per
measurement. A model that fails to load is recorded with its error and the next one runs.

While pilot lanes run, do not re-stage (it would change STAGED_COMMIT under them); copy this file instead and run
it from the staged repo root:  cd $PFSS/OneBiasAfterAnotherFork && python $PFSS/tools/load_check.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# the repo this file sits in, or (a copy outside any repo, e.g. $PFSS/tools/ while pilot lanes run on the staged
# tree) the working directory, which must then be the repo root
_HERE = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _HERE if (_HERE / "runners").is_dir() else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["ONEJUDGE_EMBED_CACHE"] = "off"      # every text through the model: the point is to time it

DEFAULT_MODELS = [
    "nicolinho/QRM-Gemma-2-27B",                 # gated head: only ever run on a tiny random model
    "Skywork/Skywork-Reward-Gemma-2-27B",        # eager attention's memory
    "nvidia/Qwen-2.5-Nemotron-32B-Reward",       # stored in fp32, loaded in bf16
    "nvidia/Qwen-3-Nemotron-32B-Reward",
    "Skywork/Skywork-Reward-V2-Qwen3-8B",
    "allenai/Llama-3.1-8B-Instruct-RM-RB2",      # pinned safetensors revision
]
BASE_CONFIG = "configs/demographic_edu_comparative_asap2_qwen06.yaml"   # max_length 4096; the model is replaced
FILLER = ("The applicant has held a steady position for several years and describes the responsibilities of the "
          "role in some detail. ")


def text_of_length(tokenizer: Any, format_fn: Any, count: Any, target: int, salt: int) -> Any:
    """A formatted conversation of exactly ``target`` tokens (as the forward pass counts them): a short prompt and
    a filler response trimmed token by token. ``salt`` makes texts distinct (no deduplication anywhere)."""
    prompt = f"Assess the following application (case {salt})."
    words = (FILLER * (target // 8 + 10)).split()
    lo, hi = 0, len(words)
    while lo < hi:                               # the most words whose conversation stays within target
        mid = (lo + hi + 1) // 2
        if count(format_fn(prompt, " ".join(words[:mid]))) <= target:
            lo = mid
        else:
            hi = mid - 1
    return format_fn(prompt, " ".join(words[:lo]))


def measure(model: Any, tokenizer: Any, texts: List[Any], batch: int, max_length: int) -> Dict[str, Any]:
    """Peak memory and throughput of embedding ``texts`` (two batches after a one-batch warm-up)."""
    import torch

    from probes.probe import embed_with_gates

    devices = range(torch.cuda.device_count())
    embed_with_gates(model, tokenizer, texts[:batch], batch_size=batch, max_length=max_length, show_progress=False)
    for d in devices:
        torch.cuda.reset_peak_memory_stats(d)
    torch.cuda.synchronize()
    start = time.perf_counter()
    embed_with_gates(model, tokenizer, texts[batch:3 * batch], batch_size=batch, max_length=max_length,
                     show_progress=False)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return {"peak_gib": round(sum(torch.cuda.max_memory_allocated(d) for d in devices) / 2**30, 2),
            "reserved_gib": round(sum(torch.cuda.max_memory_reserved(d) for d in devices) / 2**30, 2),
            "texts_per_s": round(2 * batch / seconds, 2)}


def check_model(model_path: str, lengths: List[int], max_batch: int) -> Dict[str, Any]:
    import torch

    from runners.run_cross_marker import token_counter
    from scoring.dataset_base import format_conversation
    from scoring.demographic_experiment import DemographicBiasExperiment
    from scoring.experiment import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(PROJECT_ROOT / BASE_CONFIG)
    cfg.model_path, cfg.model_revision, cfg.device = model_path, None, "cuda"
    cfg.max_length = max(lengths)
    out: Dict[str, Any] = {"model": model_path, "lengths": lengths}
    exp = DemographicBiasExperiment(cfg)
    start = time.perf_counter()
    try:
        exp.load_model()                          # pinned revision, attention, offload refusal, verify_score_path
    except Exception as e:  # noqa: BLE001 - a failed load is a result: record it, go on with the next model
        out["load_error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc(limit=5)
        return out
    model, tok = exp.model, exp.tokenizer
    out["load_s"] = round(time.perf_counter() - start, 1)
    out["model_revision"] = cfg.model_revision
    out["dtype"] = str(next(model.parameters()).dtype)
    out["attention"] = getattr(model.config, "_attn_implementation", None)
    out["weights_gib"] = round(sum(torch.cuda.memory_allocated(d) for d in range(torch.cuda.device_count())) / 2**30, 2)
    format_fn = lambda p, r: format_conversation(tok, p, r)
    count = token_counter(tok)
    out["runs"] = []
    for length in lengths:
        texts = [text_of_length(tok, format_fn, count, length, salt) for salt in range(3 * max_batch)]
        batch = 1
        while batch <= max_batch:
            row: Dict[str, Any] = {"length": length, "batch": batch}
            try:
                row.update(measure(model, tok, texts, batch, length))
            except torch.cuda.OutOfMemoryError:
                row["oom"] = True
            torch.cuda.empty_cache()
            out["runs"].append(row)
            print(f"{model_path}  length {length}  batch {batch}: " +
                  ("OUT OF MEMORY" if row.get("oom") else
                   f"peak {row['peak_gib']} GiB, {row['texts_per_s']} texts/s"), flush=True)
            if row.get("oom"):
                break
            batch *= 2
    del exp, model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    import torch

    from pairs.manifest import code_provenance, file_sha256

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--lengths", default="2048,4096", help="Input lengths in tokens")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/results/load_check"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    lengths = [int(x) for x in args.lengths.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gpus = [torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]
    for model_path in args.models:
        result = check_model(model_path, lengths, args.max_batch)
        result["meta"] = {"code": code_provenance(), "gpus": gpus, "torch": torch.__version__,
                          "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "argv": sys.argv, "script_sha256": file_sha256(__file__)}
        path = args.out_dir / f"{Path(model_path).name}.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"{model_path}: " + (f"LOAD FAILED: {result['load_error']}" if "load_error" in result
                                  else f"loaded in {result['load_s']} s, weights {result['weights_gib']} GiB")
              + f" -> {path}", flush=True)


if __name__ == "__main__":
    main()
