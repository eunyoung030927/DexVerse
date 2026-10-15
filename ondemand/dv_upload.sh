#!/bin/bash
# Upload this job's successful episodes to the HF dataset Dexverse-CSI/dv-demos in the dv-demos layout
# (v1/retargeting/<category>/<task>/<robot_type>/epNNN.pkl + index.json), merged per episode with what the central PC
# uploads for the servers. Started BY retarget_worker.sbatch in the background (every DV_UPLOAD_S s, default 1800) and
# run once by finalize.sbatch at the end of the job.
#   dv_upload.sh [INTERVAL]    INTERVAL 0 = one round (with the arm_joint_meta backfill: no queue is running then)
# One round = ondemand/dv_upload_round.py in the Isaac container (export, upload). The periodic rounds skip the
# backfill, the only GPU user, so the queue's sessions stay the only ones on the GPUs.
INTERVAL="${1:-${DV_UPLOAD_S:-1800}}"
# shellcheck source=/dev/null
source "$HOME/dexverse/DexVerse/ondemand/env.sh"
export APPTAINERENV_TASKS="$TASKS" APPTAINERENV_HANDS="${HANDS:-allegro inspire leap sharpa wuji}"
export APPTAINERENV_DEMOS_ROOT="$DEMOS_ROOT" APPTAINERENV_PY="$PY" APPTAINERENV_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
while true; do
  [ "$INTERVAL" = 0 ] || sleep "$INTERVAL"
  echo "[dv-upload $(date +%T)] round start"
  NB=""; [ "$INTERVAL" = 0 ] || NB="--no_backfill"
  "${ISAAC[@]}" "$PY" ondemand/dv_upload_round.py $NB 2>&1 | grep -vE '^\s*$|Warning|INFO' | tail -60 | sed 's/^/[dv-upload] /'
  echo "[dv-upload $(date +%T)] round done (exit ${PIPESTATUS[0]})"
  [ "$INTERVAL" = 0 ] && break
done
