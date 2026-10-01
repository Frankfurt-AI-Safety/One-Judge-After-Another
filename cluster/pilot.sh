#!/usr/bin/env bash
# The pilot (pilot-then-freeze) and the test run of every planned experiment: one lane per model, submitted from the
# Mac as unattended Determined commands, which run without an SSH connection and free their GPUs when the lane ends.
# The workspace has 4 A100s since 2026-10-01, so the three lanes run in parallel (1 + 1 + 2 GPUs):
#
#   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config idle_timeout=48h \
#     --config description=pilot_small bash cluster/pilot.sh small
#   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config idle_timeout=48h \
#     --config description=pilot_8b bash cluster/pilot.sh 8b
#   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=2 \
#     --config idle_timeout=48h --config description=pilot_70b bash cluster/pilot.sh 70b
#
# Before: stage (cluster/stage.sh, a committed tree) and regenerate the data (cluster/prepare_data.sh); a lane
# refuses to start when cluster/check_data.py finds a manifest missing, stale or generated from other code. Then
# lane `smoke` (Qwen3-0.6B, every step at a tiny size, ~1 h on one GPU) checks the whole pipeline before the three
# big lanes take the GPUs:
#
#   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config idle_timeout=6h \
#     --config description=pilot_smoke bash cluster/pilot.sh smoke
#
# Part 1, SIZING — the steps the pre-stated rules read (working notes 2026-09-25, amendments (1) and (2) of
# 2026-09-26; the comparative rule of 2026-09-30): the probe-size curve to 500 probe records, the cross-marker design
# (the records rule) and the comparative design (the pairs rule) for credit, hiring and education. Every lane, the
# 70B one at the sizes of amendment (1) (cross-marker 100 + 100 / 50 + 50; comparative 50 / 25 pairs per pairing).
# Part 2, TEST RUNS — every other planned experiment once: the placement matrix, the reasoning arm (flip, probe,
# erasure per domain, then the cross-domain transfer), the direct arm (battery per domain, the grade-level stage
# design, A2 on both positioned manifests), A2's main-effect decomposition, the blatant decision floor, additivity,
# the real-field check and the Bias-in-Bios scrub check. Lane `small` runs them at the configured (main-run)
# sizes — the full-scale dress rehearsal —, lanes `8b` and `70b` at reduced sizes, for timings and plumbing.
# Nothing in part 2 sizes anything; no decision is read from it before the headline family is fixed.
#
# Each step writes artifacts/results/demographic/pilot/<step>_<lane>.json (+ the runner's side files) and logs to
# $PFSS/pilot_logs/<step>_<lane>.log. A step whose JSON exists is skipped, so resubmitting a lane resumes it; a
# failing step is logged and the lane moves on. The lane ends with a list of the failed steps.
# (no `set -u`: macOS's bash 3.2, which runs the local dry run, treats an empty array as unset)

LANE="${1:?usage: pilot.sh smoke|small|8b|70b}"
case "$LANE" in
  smoke|small) MODEL=Skywork/Skywork-Reward-V2-Qwen3-0.6B ;;
  8b)    MODEL=Skywork/Skywork-Reward-V2-Llama-3.1-8B ;;
  70b)   MODEL=allenai/Llama-3.1-70B-Instruct-RM-RB2 ;;
  *)     echo "unknown lane: $LANE (smoke|small|8b|70b)"; exit 2 ;;
esac
PFSS="${PFSS:?PFSS is not set -- cluster/config.yaml sets it in every task}"
cd "${PILOT_REPO:-$PFSS/OneBiasAfterAnotherFork}" || exit 1

# The persistent cache: the main runs of the same model reuse these states.
export ONEJUDGE_EMBED_CACHE="${ONEJUDGE_EMBED_CACHE:-$PFSS/embedding_cache}"
OUT="${PILOT_OUT:-artifacts/results/demographic/pilot}"
LOGS="${PILOT_LOGS:-$PFSS/pilot_logs}"
mkdir -p "$OUT" "$LOGS"
# PILOT_EXTRA: flags appended to every step (the local dry run passes --device mps; empty on the cluster)
EXTRA=(${PILOT_EXTRA:-})

python cluster/check_data.py || exit 1

CFG=configs
CREDIT_X=$CFG/demographic_credit_crossmarker_qwen06.yaml
HIRING_X=$CFG/demographic_cv_crossmarker_qwen06.yaml
EDU_X=$CFG/demographic_edu_crossmarker_asap2_qwen06.yaml
CREDIT_C=$CFG/demographic_credit_comparative_qwen06.yaml
HIRING_C=$CFG/demographic_cv_comparative_qwen06.yaml
EDU_C=$CFG/demographic_edu_comparative_asap2_qwen06.yaml
CREDIT_R=$CFG/demographic_credit_reasoning_qwen06.yaml
HIRING_R=$CFG/demographic_cv_reasoning_qwen06.yaml
EDU_R=$CFG/demographic_edu_reasoning_asap2_qwen06.yaml
CREDIT_D=$CFG/demographic_credit_sex_qwen06.yaml
HIRING_D=$CFG/demographic_cv_sex_qwen06.yaml
EDU_D=$CFG/demographic_edu_sex_asap2_qwen06.yaml
STAGE_D=$CFG/demographic_edu_grade_level_asap2_qwen06.yaml
EDUPOS=$CFG/demographic_edupos_qwen06.yaml
IMPLAUSIBLE=data/demographic/education_positioned/asap2_implausible/pairs.jsonl

FAILED=()
step() {  # step <name> <runner> <args...>
  local name=$1 runner=$2
  shift 2
  local out="$OUT/${name}_${LANE}.json" log="$LOGS/${name}_${LANE}.log"
  if [ -f "$out" ]; then
    echo "$(date +%T) skip   $name (output exists)"
    return
  fi
  echo "$(date +%T) start  $name"
  python "runners/$runner" --model "$MODEL" --out "$out" "$@" "${EXTRA[@]}" > "$log" 2>&1
  local status=$?
  if [ "$status" -eq 0 ]; then
    echo "$(date +%T) done   $name  $(grep -E '^timing' "$log" | tail -1)"
  else
    echo "$(date +%T) FAILED $name (exit $status) -- see $log"
    FAILED+=("$name")
  fi
}

# Per-lane sizes. small: the configured sizes in part 2; 8b: part 2 reduced where it is costly; 70b: amendment (1)
# in part 1 and everything reduced in part 2; smoke: every step tiny (plumbing only, never read). Education texts
# are long: the 70B runs them at a small batch (its 2026-09-26 pilot fitted next to 140 GB of weights only at batch
# 2; comparative prompts hold two essays).
PR=(); CURVE=(); EDU_BATCH=(); EDU_PAIR_BATCH=()
X_CREDIT=(--n-strong 100000 --n-weak 100000)          # 100000 = every available record
X_HIRING=(--n-strong 600 --n-weak 600)
X_EDU=(--n-strong 100000 --n-weak 100000)
C_CREDIT=(); C_HIRING=(); C_EDU=()                     # the configured 150 per pairing (credit: its capacity)
M_PAIRS=(); M_EDU_PAIRS=()                             # the matrix on the comparative arm's configured pairs
R_FLIP=(); R_ITEMS=()                                  # reasoning: the runners' defaults (200 + 200)
A2=(); SCRUB=(); DECISION=()
case "$LANE" in
  8b)
    M_PAIRS=(--n-pairs 30); M_EDU_PAIRS=(--n-pairs 30)
    ;;
  70b)
    X_CREDIT=(--n-strong 100 --n-weak 100); X_HIRING=(--n-strong 100 --n-weak 100); X_EDU=(--n-strong 50 --n-weak 50)
    C_CREDIT=(--n-pairs 50); C_HIRING=(--n-pairs 50); C_EDU=(--n-pairs 25)
    EDU_BATCH=(--batch-size 2); EDU_PAIR_BATCH=(--batch-size 1)
    M_PAIRS=(--n-pairs 15); M_EDU_PAIRS=(--n-pairs 8)
    R_FLIP=(--n-items 100); R_ITEMS=(--probe-items 100 --eval-items 100)
    A2=(--n-essays 150); SCRUB=(--probe-items 300 --eval-items 300)
    ;;
  smoke)
    PR=(--probe-records 20); CURVE=(--grid 10,20 --max-eval 40 --n-boot 50)
    X_CREDIT=(--n-strong 4 --n-weak 4 --n-boot 50); X_HIRING=("${X_CREDIT[@]}"); X_EDU=("${X_CREDIT[@]}")
    C_CREDIT=(--n-pairs 2 --n-folds 2 --n-boot 50); C_HIRING=("${C_CREDIT[@]}"); C_EDU=("${C_CREDIT[@]}")
    M_PAIRS=("${C_CREDIT[@]}"); M_EDU_PAIRS=("${C_CREDIT[@]}")
    R_FLIP=(--n-items 8); R_ITEMS=(--probe-items 8 --eval-items 8)
    A2=(--n-essays 6); SCRUB=(--probe-items 20 --eval-items 20); DECISION=(--n-items 8)
    ;;
esac

echo "$(date +%T) pilot lane $LANE: $MODEL"

# ---- part 1: sizing --------------------------------------------------------------------------------------------
step probecurve_credit      run_probe_curve.py   --config "$CREDIT_X" "${CURVE[@]}"
step probecurve_hiring      run_probe_curve.py   --config "$HIRING_X" "${CURVE[@]}"
step probecurve_education   run_probe_curve.py   --config "$EDU_X" "${CURVE[@]}" "${EDU_BATCH[@]}"
step crossmarker_credit     run_cross_marker.py  --config "$CREDIT_X" "${X_CREDIT[@]}" "${PR[@]}"
step crossmarker_hiring     run_cross_marker.py  --config "$HIRING_X" "${X_HIRING[@]}" "${PR[@]}"
step crossmarker_education  run_cross_marker.py  --config "$EDU_X"    "${X_EDU[@]}" "${PR[@]}" "${EDU_BATCH[@]}"
step comparative_credit     run_comparative.py   --config "$CREDIT_C" "${C_CREDIT[@]}" "${PR[@]}"
step comparative_hiring     run_comparative.py   --config "$HIRING_C" "${C_HIRING[@]}" "${PR[@]}"
step comparative_education  run_comparative.py   --config "$EDU_C"    "${C_EDU[@]}" "${PR[@]}" "${EDU_PAIR_BATCH[@]}"

# ---- part 2: test runs of every other planned experiment ------------------------------------------------------
# the placement matrix right after the comparative design: its comparative texts are cache hits
step placement_credit       run_placement_matrix.py --config "$CREDIT_C" "${M_PAIRS[@]}" "${PR[@]}"
step placement_hiring       run_placement_matrix.py --config "$HIRING_C" "${M_PAIRS[@]}" "${PR[@]}"
step placement_education    run_placement_matrix.py --config "$EDU_C"    "${M_EDU_PAIRS[@]}" "${PR[@]}" "${EDU_PAIR_BATCH[@]}"
# the reasoning arm; the transfer reuses the probe runs' states (same items: same sizes)
for dom in credit hiring education; do
  case $dom in credit) cfg=$CREDIT_R ;; hiring) cfg=$HIRING_R ;; education) cfg=$EDU_R ;; esac
  batch=(); [ "$dom" = education ] && batch=("${EDU_BATCH[@]}")
  step "reasoning_flip_$dom"    run_reasoning_flip.py    --config "$cfg" "${R_FLIP[@]}"  "${PR[@]}" "${batch[@]}"
  step "reasoning_probe_$dom"   run_reasoning_probe.py   --config "$cfg" "${R_ITEMS[@]}" "${PR[@]}" "${batch[@]}"
  step "reasoning_erasure_$dom" run_reasoning_erasure.py --config "$cfg" "${R_ITEMS[@]}" "${PR[@]}" "${batch[@]}"
done
step reasoning_transfer     run_reasoning_transfer.py --configs "$CREDIT_R" "$HIRING_R" "$EDU_R" "${R_ITEMS[@]}" \
  "${PR[@]}" "${EDU_BATCH[@]}"
# the direct arm (the mechanism layer): every axis and encoding the manifest holds, 200 eval pairs each
step battery_credit         run_battery.py --config "$CREDIT_D" "${PR[@]}"
step battery_hiring         run_battery.py --config "$HIRING_D" "${PR[@]}"
step battery_education      run_battery.py --config "$EDU_D"   "${PR[@]}" "${EDU_BATCH[@]}"
step battery_grade_level    run_battery.py --config "$STAGE_D" "${PR[@]}" "${EDU_BATCH[@]}"
step battery_a2_plausible   run_battery.py --config "$EDUPOS"  "${PR[@]}" "${EDU_BATCH[@]}"
step battery_a2_implausible run_battery.py --config "$EDUPOS"  --dataset-source "$IMPLAUSIBLE" "${PR[@]}" "${EDU_BATCH[@]}"
# A2's main-effect decomposition, the result group and its control
step a2_maineffect_plausible   run_positioned_maineffect.py --config "$EDUPOS" --standpoint-fit plausible \
  "${A2[@]}" "${EDU_BATCH[@]}"
step a2_maineffect_implausible run_positioned_maineffect.py --config "$EDUPOS" --standpoint-fit implausible \
  "${A2[@]}" "${EDU_BATCH[@]}"
# the blatant decision-response floor, both encodings
for dom in credit hiring education; do
  case $dom in
    credit)    cfg=$CFG/demographic_credit_decision_qwen06.yaml; batch=() ;;
    hiring)    cfg=$CFG/demographic_cv_decision_qwen06.yaml; batch=() ;;
    education) cfg=$CFG/demographic_edu_decision_asap2_qwen06.yaml; batch=("${EDU_BATCH[@]}") ;;
  esac
  for enc in explicit proxy; do
    step "decision_${dom}_$enc" run_decision_response.py --config "$cfg" --encoding "$enc" "${DECISION[@]}" "${PR[@]}" "${batch[@]}"
  done
done
# additivity (RQ1.1), the real-field marital check, the scrub check
step additivity_credit      run_additivity.py --domain credit    --encoding explicit "${PR[@]}"
step additivity_hiring      run_additivity.py --domain cv        --encoding explicit "${PR[@]}"
step additivity_education   run_additivity.py --domain education --encoding explicit "${PR[@]}" "${EDU_BATCH[@]}"
step realfield_credit       run_realfield.py  --config "$CREDIT_D" "${PR[@]}"
step scrub_hiring           validate_bios_scrub.py --config "$HIRING_D" "${SCRUB[@]}" "${PR[@]}"

echo "$(date +%T) pilot lane $LANE finished: ${#FAILED[@]} failed step(s)${FAILED[*]:+: ${FAILED[*]}}"
