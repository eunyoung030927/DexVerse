#!/bin/bash
# Retarget recorded demos of one task onto another floating hand: stages 1-3 per episode, then stage 4 once.
# One Isaac Sim process at a time (each stage is its own process; finished stages are skipped on a rerun).
#
#   scripts/retarget/run_episodes.sh <task> <demos.pkl> <target robot_type> <episode> [<episode> ...]
#   scripts/retarget/run_episodes.sh Dexverse-GraspCup-v0 demos/.../Dexverse-GraspCup-v0/demos.pkl floating_allegro_right 0 1 2
#
# Extra stage-3 arguments: SPIDER_ARGS="--num_samples 2048". Python: $PYTHON (default: python on PATH, e.g. after
# sourcing the DexVerse env). Logs: $DEXVERSE_RETARGET_RUNS/logs (default <repo>/results/retarget/logs).
set -o pipefail

TASK=${1:?task}; DEMOS=${2:?demos.pkl}; TARGET=${3:?target robot_type}; shift 3
[ $# -gt 0 ] || { echo "give at least one episode index" >&2; exit 1; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
export PYTHONPATH="${REPO}/source/dexverse${PYTHONPATH:+:${PYTHONPATH}}"
LOGS=${DEXVERSE_RETARGET_RUNS:-${REPO}/results/retarget}/logs
mkdir -p "${LOGS}"
cd "${REPO}"

stage() {   # stage <name> <log> <args...>
    local name=$1 log=$2; shift 2
    echo "[retarget] ${name} ... log ${log}"
    timeout -k 30 "${STAGE_TIMEOUT:-14400}" "${PY}" -u "$@" --headless --device "${DEVICE}" > "${log}" 2>&1
    local rc=$?
    grep -E "^RESULT|^skip:" "${log}" | tail -1
    [ ${rc} -eq 0 ] || echo "[retarget] ${name} exited ${rc}"
    return ${rc}
}

for EP in "$@"; do
    tag="$(basename "${TASK}")_ep$(printf %03d "${EP}")"
    stage "stage1 ep${EP}" "${LOGS}/${tag}_stage1.log" scripts/retarget/stage1_reference.py \
        --task "${TASK}" --demos "${DEMOS}" --episode "${EP}" || continue
    stage "stage2 ep${EP} -> ${TARGET}" "${LOGS}/${tag}_${TARGET}_stage2.log" scripts/retarget/stage2_kinematic.py \
        --task "${TASK}" --demos "${DEMOS}" --episode "${EP}" --target "${TARGET}" || continue
    # shellcheck disable=SC2086
    stage "stage3 ep${EP} -> ${TARGET}" "${LOGS}/${tag}_${TARGET}_stage3.log" scripts/retarget/stage3_spider.py \
        --task "${TASK}" --demos "${DEMOS}" --episode "${EP}" --target "${TARGET}" ${SPIDER_ARGS} || continue
done
"${PY}" scripts/retarget/stage4_record.py --task "${TASK}" --demos "${DEMOS}" --target "${TARGET}"
