#!/usr/bin/env python3
"""
Pre-download reward-model checkpoints into the PFSS Hugging Face cache.

Run this from a **slots=0** shell before any GPU job. Downloading ~566 GB while holding an
A100 wastes the allocation on I/O, and a job that dies mid-download leaves a half-populated
cache that later fails confusingly.

It also answers the question we cannot answer from outside: **do the compute nodes have
outbound internet?** If they do not, this fails immediately and cleanly, and the checkpoints
have to be pushed from a machine that does (see cluster/README.md).

    export HF_HOME=/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias/hf_cache
    python cluster/prefetch_models.py --check          # connectivity + disk only
    python cluster/prefetch_models.py --tier small     # 0.6B + DeBERTa, ~2 GB
    python cluster/prefetch_models.py --tier 8b        # the three 8B models
    python cluster/prefetch_models.py --tier all
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

# (hf_id, params_b, tier, confirmed)
#
# `confirmed=False` marks the NVIDIA Nemotron entries: working_notes.tex records that their
# exact Hugging Face paths were never verified against the current releases. They are listed
# so the ladder is complete, but are skipped unless --include-unconfirmed is passed, because
# guessing an id here would silently fetch the wrong model or fail deep in a scaling run.
MODELS = [
    ("Skywork/Skywork-Reward-V2-Qwen3-0.6B",      0.6, "small", True),
    ("OpenAssistant/reward-model-deberta-v3-large-v2", 0.4, "small", True),
    ("Skywork/Skywork-Reward-V2-Llama-3.1-8B",      8, "8b",   True),
    ("Skywork/Skywork-Reward-V2-Qwen3-8B",          8, "8b",   True),
    ("allenai/Llama-3.1-8B-Instruct-RM-RB2",        8, "8b",   True),
    ("Skywork/Skywork-Reward-Gemma-2-27B",         27, "mid",  True),
    ("nicolinho/QRM-Gemma-2-27B",                  27, "mid",  True),
    ("nvidia/Qwen-2.5-Nemotron-32B-Reward",        32, "mid",  False),
    ("nvidia/Qwen-3-Nemotron-32B-Reward",          32, "mid",  False),
    ("allenai/Llama-3.1-70B-Instruct-RM-RB2",      70, "70b",  True),
    ("nvidia/Llama-3.3-Nemotron-70B-Reward",       70, "70b",  False),
]

TIERS = ["small", "8b", "mid", "70b"]


def preflight() -> bool:
    """Report the two things that make a download fail hours in rather than immediately."""
    home = os.environ.get("HF_HOME")
    print(f"HF_HOME = {home or '(unset -- will cache to ~/.cache, NOT PFSS)'}")
    if not home:
        print("  ! set HF_HOME to the PFSS cache first, or the node-local disk will fill up")
    elif os.path.isdir(home) or os.path.isdir(os.path.dirname(home)):
        free = shutil.disk_usage(home if os.path.isdir(home) else os.path.dirname(home)).free
        print(f"  free space: {free / 1e12:.2f} TB  (need ~0.6 TB for the full ladder)")

    print("\nchecking outbound access to huggingface.co ...")
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co/api/models/gpt2", timeout=15).read(64)
        print("  OK -- compute node can reach the Hub, prefetch here.")
        return True
    except Exception as e:  # noqa: BLE001 - any failure means the same thing operationally
        print(f"  NO ({type(e).__name__}: {e})")
        print("  -> the nodes are offline. Download on a machine with access and rsync the")
        print("     cache directory across, or ask hessian.AI for a proxy. See cluster/README.md.")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=TIERS + ["all"], default="small")
    ap.add_argument("--check", action="store_true", help="preflight only, download nothing")
    ap.add_argument("--include-unconfirmed", action="store_true",
                    help="also fetch the Nemotron ids that were never verified")
    args = ap.parse_args()

    online = preflight()
    if args.check:
        return
    if not online:
        sys.exit(1)

    from huggingface_hub import snapshot_download

    wanted = [m for m in MODELS if args.tier == "all" or m[2] == args.tier]
    skipped = [m for m in wanted if not m[3] and not args.include_unconfirmed]
    todo = [m for m in wanted if m[3] or args.include_unconfirmed]

    total = sum(b * 2 for _, b, _, _ in todo)
    print(f"\n{len(todo)} checkpoint(s), ~{total:.0f} GB in bf16\n")

    for hf_id, params_b, _, confirmed in todo:
        flag = "" if confirmed else "  [UNCONFIRMED ID]"
        print(f"--> {hf_id} (~{params_b * 2:.0f} GB){flag}", flush=True)
        try:
            snapshot_download(hf_id, resume_download=True)
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}")
            if not confirmed:
                print("    (expected -- this id was never verified; find the real path first)")

    for hf_id, _, _, _ in skipped:
        print(f"skipped (unverified id): {hf_id}")
    if skipped:
        print("Confirm these against the current NVIDIA releases, then re-run with "
              "--include-unconfirmed.")


if __name__ == "__main__":
    main()
