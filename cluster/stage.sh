#!/usr/bin/env bash
# Stage code + raw corpora onto PFSS. Run from the repo root, with the TU VPN connected and a
# Determined shell already running (it is the only SSH route onto the cluster).
#
#   det shell start -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=0
#   det shell show-ssh-command <shell-id>      # -> ProxyCommand, key and user for an SSH alias
#   ./cluster/stage.sh det-stage               # the alias's name in ~/.ssh/config
#
# The target must be an SSH host alias: a shell is only reachable through Determined's ProxyCommand with
# its own key, and both ssh and rsync below need that. Setup: cluster/README.md §2, step 3.
# slots=0 matters: staging needs no GPU, and an idle GPU-holding shell burns the allocation.
#
# Only a COMMITTED tree is staged: the copy holds the tracked files (no .git), and STAGED_COMMIT records their
# commit, which every result's `meta.code` then names (pairs.manifest.code_provenance). Uncommitted changes or
# untracked files would make the staged code differ from that commit, so the script refuses them
# (STAGE_ALLOW_DIRTY=1 overrides; the results are then marked dirty).
#
# Generated data is NOT staged: cluster/prepare_data.sh regenerates every manifest on the cluster from the raw
# corpora (deterministic, a few minutes), so the data always comes from the staged code.
set -euo pipefail

REMOTE="${1:?usage: stage.sh <ssh-alias> (see cluster/README.md §2)}"
PFSS="/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias"
REPO="$PFSS/OneBiasAfterAnotherFork"

DIRTY="$(git status --porcelain)"
if [ -n "$DIRTY" ] && [ "${STAGE_ALLOW_DIRTY:-0}" != 1 ]; then
  echo "the working tree differs from HEAD; commit first (or STAGE_ALLOW_DIRTY=1):"
  echo "$DIRTY"
  exit 1
fi
COMMIT="$(git rev-parse HEAD)"

echo "==> creating PFSS layout"
ssh "$REMOTE" "mkdir -p '$REPO' '$PFSS/hf_cache' '$PFSS/artifacts/results/demographic'"

echo "==> code at $COMMIT (tracked files only; no venvs, no caches)"
git ls-files -z | rsync -av --files-from=- --from0 ./ "$REMOTE:$REPO/"
python3 - "$COMMIT" "$DIRTY" <<'EOF' | ssh "$REMOTE" "cat > '$REPO/STAGED_COMMIT'"
import json, sys
from datetime import datetime, timezone
paths = sorted(line[3:] for line in sys.argv[2].splitlines() if line.strip())
print(json.dumps({"git_commit": sys.argv[1], "git_dirty": bool(paths), "git_dirty_paths": paths,
                  "staged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
EOF

# Raw corpora ~105 MB (German Credit, Bias-in-Bios parquet, ASAP 2.0). Gitignored, so not covered by the git
# ls-files pass above; rsync skips what an earlier staging (or README §2 step 2's copy) already put there.
echo "==> raw corpora"
rsync -av --progress \
  --include='*/' --include='raw/***' --exclude='*' \
  data/demographic/ "$REMOTE:$REPO/data/demographic/"

echo
echo "staged $COMMIT -> $REPO"
echo "next (cluster terminal): bash cluster/prepare_data.sh   # regenerates every manifest, then checks them"
echo "checkpoints, if not on PFSS yet: python cluster/prefetch_models.py --tier small (then 8b, 70b)"
