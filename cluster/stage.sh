#!/usr/bin/env bash
# Stage code + data onto PFSS. Run from the repo root, with the TU VPN connected and a
# Determined shell already running (it is the only SSH route onto the cluster).
#
#   det shell start -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=0
#   det shell show-ssh-command <shell-id>      # -> ProxyCommand, key and user for an SSH alias
#   ./cluster/stage.sh det-stage               # the alias's name in ~/.ssh/config
#
# The target must be an SSH host alias: a shell is only reachable through Determined's ProxyCommand with
# its own key, and both ssh and rsync below need that. Setup: cluster/README.md §2, step 3.
# slots=0 matters: staging needs no GPU, and an idle GPU-holding shell burns the allocation.
set -euo pipefail

REMOTE="${1:?usage: stage.sh <ssh-alias> (see cluster/README.md §2)}"
PFSS="/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias"
REPO="$PFSS/OneBiasAfterAnotherFork"

echo "==> creating PFSS layout"
ssh "$REMOTE" "mkdir -p '$REPO' '$PFSS/hf_cache' '$PFSS/artifacts/results/demographic'"

echo "==> code (tracked files only; no venvs, no caches)"
git ls-files -z | rsync -av --files-from=- --from0 ./ "$REMOTE:$REPO/"

# Raw corpora ~665 MB (+ ~1.9 GB generated data below). Gitignored, so it is not covered by the git ls-files pass above.
#
# INODES: send the corpora as single large files and let the generators rebuild pairs.jsonl
# on the cluster. The onboarding deck asks for <2M inodes per workspace; a handful of big
# files costs almost nothing, and HF's cache of 11 checkpoints is the only real consumer.
echo "==> raw corpora (~665 MB)"
rsync -av --progress \
  --include='*/' --include='raw/***' --exclude='*' \
  data/demographic/ "$REMOTE:$REPO/data/demographic/"

# cells.jsonl (all 8 factorial texts per block) is what the cross-marker design reads; it must come from
# the SAME generator run as pairs.jsonl (its cells are byte-identical to the ones the pairs were cut from),
# so the two always travel together. Local embedding caches (artifacts/, gitignored) are never staged.
# ~1.9 GB since 2026-09-24 (hiring 0.6 GB, PERSUADE 0.65 GB, ASAP 0.35 GB, credit 0.1 GB): every generator
# emits the full factorial. Over a slow VPN it can be quicker to skip this block and rerun the generators on
# the cluster from the staged raw corpora (about a minute each, deterministic from the seed).
echo "==> generated matched pairs + factorial cells (~1.9 GB; regenerable, but skip the rebuild)"
rsync -av --progress \
  --include='*/' --include='pairs.jsonl' --include='cells.jsonl' --include='manifest.json' --exclude='*' \
  data/demographic/ "$REMOTE:$REPO/data/demographic/"

echo
echo "staged -> $REPO"
echo "next: pre-download the checkpoints so the first GPU job does not spend its slot on I/O:"
echo "  ssh $REMOTE"
echo "  export HF_HOME=$PFSS/hf_cache"
echo "  python cluster/prefetch_models.py --tier small"
