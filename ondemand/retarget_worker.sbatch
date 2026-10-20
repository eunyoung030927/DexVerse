#!/bin/bash
# Two pro6000 GPUs, CUDA MPS on, one queue over all TASKS: a new session starts whenever a slot is free (up to PER_GPU
# per GPU) and that GPU still has MIN_FREE_GB of VRAM; items are offered in TASKS order (task 1 first).
# Run by ondemand/main_job.sbatch inside its own allocation (bash, not sbatch); can also be submitted on its own to
# resume a task whose demos are already downloaded (sbatch --export=ALL,TASKS=Dexverse-GraspBleach-v1 ...).
#
# Runs scripts/retarget/run_queue.py inside the Isaac Sim container: (episode, target hand) items of $TASKS,
# episodes 0..EPISODES-1, onto every hand of $HANDS; each item is scripts/retarget/run_target.py (stage 2, SPIDER with
# the attempt schedule). Stage 1 runs once per episode in a free slot. Finished items are skipped, failed ones get
# new attempts if MAX_ATTEMPTS grew, so resubmitting resumes. No rendering, no upload (finalize.sbatch decides).
#
# Progress: $HOME/dexverse/dvr_status.md (OOD Files app), rewritten every 5 min by ondemand/monitor.sh.
#
#SBATCH --job-name=dvr-retarget
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-dvr-retarget-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
# shellcheck source=/dev/null
source "$HOME/dexverse/DexVerse/ondemand/env.sh"
sync_repo
cd "$REPO_DIR"
EPISODES="${EPISODES:-50}"
HANDS="${HANDS:-allegro inspire leap sharpa wuji}"
PER_GPU="${PER_GPU:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
SESSION_GB="${SESSION_GB:-6}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
echo "=== dvr-retarget: [$TASKS] eps 0..$(( EPISODES - 1 )) hands [$HANDS] on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?} $(date) ==="
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

# ------------------------------------------------------------------ MPS
MPS_DIR="$NODE/mps-${SLURM_JOB_ID:-$$}"
export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
MPS=0
if [ "${MPS_ON:-1}" = "1" ] && command -v nvidia-cuda-mps-control >/dev/null; then
  nvidia-cuda-mps-control -d && MPS=1
  export APPTAINERENV_CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" APPTAINERENV_CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_LOG_DIRECTORY"
  echo "MPS control daemon started ($MPS_DIR)"
else
  echo "WARNING: MPS not started (MPS_ON=${MPS_ON:-1}, nvidia-cuda-mps-control $(command -v nvidia-cuda-mps-control || echo missing)); running without MPS"
fi
# stop the monitor and MPS whatever way this script ends
trap 'kill ${MON:-} ${DVU:-} 2>/dev/null; [ "$MPS" = 1 ] && echo quit | nvidia-cuda-mps-control; rm -rf "$MPS_DIR"' EXIT

# ------------------------------------------------------------------ GPU ids as the sessions see them
# A session pins its GPU with CUDA_VISIBLE_DEVICES=<id>. Depending on cgroups / MPS that id is the job-relative
# ordinal (0 1 ...) or the node id SLURM gave; take the first form that reaches N distinct GPUs, N = the GPUs this
# job holds (--gres=gpu:pro6000:N; p1 has 4).
probe() { "${ISAAC[@]}" "$PY" -c 'import torch; print(torch.cuda.get_device_properties(0).uuid)' 2>/dev/null | tail -1; }
NGPU="${SLURM_GPUS_ON_NODE:-$(echo "${CUDA_VISIBLE_DEVICES:-0,1}" | tr ',' '\n' | grep -c .)}"
GPUS=""
for cand in "$(seq -s ' ' 0 $(( NGPU - 1 )))" "$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' ' ')"; do
  u=()
  for g in $cand; do u+=("$(APPTAINERENV_CUDA_VISIBLE_DEVICES=$g probe)"); done
  n_found="$(printf '%s\n' "${u[@]}" | grep -c .)"; n_unique="$(printf '%s\n' "${u[@]}" | grep . | sort -u | wc -l)"
  if [ "$n_found" -eq "$NGPU" ] && [ "$n_unique" -eq "$NGPU" ]; then GPUS="$cand"; break; fi
done
[ -n "$GPUS" ] || { echo "FATAL: could not address $NGPU distinct GPUs"; exit 1; }
echo "sessions use CUDA_VISIBLE_DEVICES in [$GPUS], up to $PER_GPU per GPU while >= $MIN_FREE_GB GB free, MPS=$MPS"

# ------------------------------------------------------------------ queue
bash "$CODE_DIR/outputs_note.sh" || true     # OUTPUTS.md in the job folder: where the results go
MYLOG="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'StdOut=[^ ]*' | cut -d= -f2)"
bash "$CODE_DIR/monitor.sh" "${MYLOG:-/dev/null}" "${MONITOR_S:-300}" &
MON=$!
if [ "${UPLOAD:-1}" = "1" ]; then           # successes -> HF Dexverse-CSI/dv-demos every DV_UPLOAD_S s
  # arm_joint_meta for older results now, while the GPUs are free; the periodic rounds then use no GPU
  APPTAINERENV_TASKS="$TASKS" APPTAINERENV_HANDS="$HANDS" APPTAINERENV_DEMOS_ROOT="$DEMOS_ROOT" APPTAINERENV_PY="$PY" \
    "${ISAAC[@]}" "$PY" ondemand/dv_upload_round.py --backfill_only 2>&1 | grep -E 'RESULT|Error|Traceback' || true
  bash "$CODE_DIR/dv_upload.sh" "${DV_UPLOAD_S:-1800}" &
  DVU=$!
fi
echo "live status: $HOME/dexverse/dvr_status.md (every ${MONITOR_S:-300} s)"
"${ISAAC[@]}" env PY="$PY" "$PY" scripts/retarget/run_queue.py --tasks $TASKS --demos_root "$DEMOS_ROOT" \
  --episodes $(seq 0 $(( EPISODES - 1 ))) --hands $HANDS \
  --gpus $GPUS --per_gpu "$PER_GPU" --min_free_gb "$MIN_FREE_GB" --session_gb "$SESSION_GB" --max_attempts "$MAX_ATTEMPTS" --spider_args "${SPIDER_ARGS:-}" --retry_failed

kill "$MON" ${DVU:-} 2>/dev/null || true
timeout 20 bash "$CODE_DIR/monitor.sh" "${MYLOG:-/dev/null}" 99999 || true     # final snapshot
echo "=== retarget done $(date): the final dv-demos upload runs in finalize.sbatch ==="
