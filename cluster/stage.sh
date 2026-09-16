#!/usr/bin/env bash
# Stage code + data onto PFSS. Run from the repo root, with the TU VPN connected and a
# Determined shell already running (it is the only SSH route onto the cluster).
#
#   det shell start -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=0
#   det shell show-ssh-command <shell-id>      # -> gives host/port/key for scp and rsync
#   ./cluster/stage.sh <ssh-host-from-that-command>
#
# slots=0 matters: staging needs no GPU, and an idle GPU-holding shell burns the allocation.
set -euo pipefail

REMOTE="${1:?usage: stage.sh <ssh-target-from-det-shell-show-ssh-command>}"
PFSS="/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias"
REPO="$PFSS/OneBiasAfterAnotherFork"

echo "==> creating PFSS layout"
ssh "$REMOTE" "mkdir -p '$REPO' '$PFSS/hf_cache' '$PFSS/artifacts/results/demographic'"

echo "==> code (tracked files only; no venvs, no caches)"
git ls-files -z | rsync -av --files-from=- --from0 ./ "$REMOTE:$REPO/"

# ~730 MB total. Gitignored, so it is not covered by the git ls-files pass above.
#
# INODES: send the corpora as single large files and let the generators rebuild pairs.jsonl
# on the cluster. The onboarding deck asks for <2M inodes per workspace; a handful of big
# files costs almost nothing, and HF's cache of 11 checkpoints is the only real consumer.
echo "==> raw corpora (~665 MB)"
rsync -av --progress \
  --include='*/' --include='raw/***' --exclude='*' \
  data/demographic/ "$REMOTE:$REPO/data/demographic/"

echo "==> generated matched pairs (~60 MB; regenerable, but skip the rebuild)"
rsync -av --progress \
  --include='*/' --include='pairs.jsonl' --include='manifest.json' --exclude='*' \
  data/demographic/ "$REMOTE:$REPO/data/demographic/"

echo
echo "staged -> $REPO"
echo "next: pre-download the checkpoints so the first GPU job does not spend its slot on I/O:"
echo "  ssh $REMOTE"
echo "  export HF_HOME=$PFSS/hf_cache"
echo "  python cluster/prefetch_models.py --tier small"
