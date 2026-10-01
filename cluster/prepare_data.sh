#!/usr/bin/env bash
# Regenerate every manifest on PFSS from the staged raw corpora (generator 0.4.0), once after each staging and
# before any lane of cluster/pilot.sh. CPU only, a few minutes in all; run it in the cluster terminal of a slots=0
# shell (cluster/README.md §2):
#
#   cd $PFSS/OneBiasAfterAnotherFork && bash cluster/prepare_data.sh
#
# The manifests name every data and corpus file by SHA-256 and every runner refuses a file its manifest does not
# describe, so data generated elsewhere (or by an older generator) is never mixed in: stage.sh no longer copies
# generated data. Hiring is fixed at 12,000 bios (substrates.bios_clean.DEFAULT_N_BIOS) — never change it after
# the pilot: another size relabels the same bios. Everything is deterministic from the default seed (42).
#
# Do not rerun it while a lane is running: the runners check the data's hashes before loading the model, but a
# file replaced mid-run would no longer be the one a running step's manifest check saw.
set -euo pipefail

PFSS="${PFSS:?PFSS is not set -- cluster/config.yaml sets it in every task}"
cd "$PFSS/OneBiasAfterAnotherFork"
[ -f STAGED_COMMIT ] || { echo "no STAGED_COMMIT: stage with cluster/stage.sh first"; exit 1; }

run() { echo "$(date +%T) $*"; python "$@"; }

run runners/generate_credit.py                                      # sex x age x marital factorial
run runners/generate_bios.py                                        # hiring, 12,000 bios
run runners/generate_education.py                                   # ASAP 2.0 factorial
run runners/generate_education.py --design stage --include-ladder   # the grade-level stage design
run runners/generate_positioned.py --standpoint-fit plausible       # A2, the result
run runners/generate_positioned.py --standpoint-fit implausible     # A2, its control

python cluster/check_data.py
