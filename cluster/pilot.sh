#!/usr/bin/env bash
# The pilot (pilot-then-freeze), one GPU lane per model. Submitted from the Mac as an unattended
# Determined command, which runs without an SSH connection and frees the GPU when the lane ends:
#
#   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config idle_timeout=24h \
#     --config description=pilot_small bash cluster/pilot.sh small
#   (the same with 8b; the workspace allows 2 GPUs, so the two lanes are all that can run)
#
# Per lane: the probe-size curve for credit, hiring and PERSUADE, then the cross-marker design on the full
# credit and education pools and 600 + 600 hiring records. Each step writes
# artifacts/results/demographic/pilot/<step>_<lane>.json and logs to $PFSS/pilot_logs/<step>_<lane>.log.
# A step whose JSON exists is skipped, so resubmitting a lane resumes it; a failing step is logged and the
# lane moves on. The rules that turn these outputs into probe_records and n are in the working notes
# (2026-09-25); runners/pilot_sizing.py applies them.
set -u

LANE="${1:?usage: pilot.sh small|8b}"
case "$LANE" in
  small) MODEL=Skywork/Skywork-Reward-V2-Qwen3-0.6B ;;
  8b)    MODEL=Skywork/Skywork-Reward-V2-Llama-3.1-8B ;;
  *)     echo "unknown lane: $LANE (small|8b)"; exit 2 ;;
esac
PFSS="${PFSS:?PFSS is not set -- cluster/config.yaml sets it in every task}"
cd "$PFSS/OneBiasAfterAnotherFork" || exit 1

# The persistent cache: the main runs of the same model reuse these states.
export ONEJUDGE_EMBED_CACHE="$PFSS/embedding_cache"
OUT=artifacts/results/demographic/pilot
LOGS="$PFSS/pilot_logs"
mkdir -p "$OUT" "$LOGS"

CREDIT=configs/demographic_credit_crossmarker_qwen06.yaml
HIRING=configs/demographic_cv_crossmarker_qwen06.yaml
EDUCATION=configs/demographic_edu_crossmarker_persuade_qwen06.yaml

step() {  # step <name> <runner> <args...>
  local name=$1 runner=$2
  shift 2
  local out="$OUT/${name}_${LANE}.json" log="$LOGS/${name}_${LANE}.log"
  if [ -f "$out" ]; then
    echo "$(date +%T) skip   $name (output exists)"
    return
  fi
  echo "$(date +%T) start  $name"
  python "runners/$runner" --model "$MODEL" --out "$out" "$@" > "$log" 2>&1
  local status=$?
  if [ "$status" -eq 0 ]; then
    echo "$(date +%T) done   $name  $(grep -E '^timing' "$log" | tail -1)"
  else
    echo "$(date +%T) FAILED $name (exit $status) -- see $log"
  fi
}

echo "$(date +%T) pilot lane $LANE: $MODEL"
step probecurve_credit     run_probe_curve.py  --config "$CREDIT"
step probecurve_hiring     run_probe_curve.py  --config "$HIRING"
step probecurve_education  run_probe_curve.py  --config "$EDUCATION"
# 100000 = every available record (select_records takes the seeded order up to what the pool holds)
step crossmarker_credit    run_cross_marker.py --config "$CREDIT"    --n-strong 100000 --n-weak 100000
step crossmarker_hiring    run_cross_marker.py --config "$HIRING"    --n-strong 600    --n-weak 600
step crossmarker_education run_cross_marker.py --config "$EDUCATION" --n-strong 100000 --n-weak 100000
echo "$(date +%T) pilot lane $LANE finished"
