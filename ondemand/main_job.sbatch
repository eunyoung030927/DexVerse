#!/bin/bash
# DexVerse cross-hand SPIDER retargeting -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit. Nothing else is submitted by hand.
#
# In order:
#   1. preflight  ~/.hf_token must exist (upload; the Hub also rate-limits anonymous downloads)
#   2. git sync   clone or fast-forward $HOME/dexverse/DexVerse, the fork eunyoung030927/DexVerse, branch
#                 retarget/shadow-cross-hand (this pipeline is its ondemand/ folder)
#   3. setup      Isaac Sim 5.1 SIF (Bench2Dex's if present), Isaac Lab v2.3.2, DexVerse + deps, robot/object assets
#   4. fetch      the released v1 demos of every task in TASKS (dexverse/DexVerse_release; skips what is there)
#   5. retarget   IN THIS JOB, on the 2 pro6000 GPUs it holds, ONE queue over all TASKS: items are (task, episode,
#                 target hand) in TASKS order, so free GPU capacity is filled with task 1's items first, then task 2's,
#                 ... without waiting for a task to finish. A new session starts whenever a slot is free (PER_GPU per
#                 GPU) and that GPU has MIN_FREE_GB of VRAM left; CUDA MPS. Each item: SPIDER (v2 settings, 60 Hz)
#                 until the task's own success term holds 10 steps, up to MAX_ATTEMPTS
#                 (ondemand/retarget_worker.sbatch -> scripts/retarget/run_queue.py -> run_target.py; no rendering)
#   6. upload     DURING the queue every DV_UPLOAD_S s (ondemand/dv_upload.sh) and once more at the end
#                 (ondemand/finalize.sbatch): each successful episode as v1/retargeting/<category>/<task>/<robot_type>/
#                 epNNN.pkl + index.json to the HF dataset Dexverse-CSI/dv-demos, merged per episode with the repo
#
# Walltime: at its start the job queues a continuation of itself (--dependency=afterany on this job). If this job
# finishes every item, it cancels the continuation; if it hits the walltime, the continuation resumes the queue
# (finished items are skipped). At most MAX_CHAIN jobs in a row. To stop: cancel the running job AND the queued one.
#
# Live status: ~/dexverse/dvr_status.md (OOD Files app), rewritten every 5 min; summary lines in this log.
#
# Everything after step 2 is read from the pulled fork, so edits to scripts/retarget/, source/ and ondemand/*.sh
# only need a push. Only an edit to THIS file needs re-pasting. Resubmitting is always safe.
#
# Knobs (sbatch --export=ALL,KNOB=value, or edit the defaults below):
#   TASKS="Dexverse-GraspBleach-v1 ..." (order = fill order)  HANDS="allegro wuji"  EPISODES=50  MAX_ATTEMPTS=3
#   PER_GPU=10  MIN_FREE_GB=10  SESSION_GB=6  MPS_ON=1  SPIDER_ARGS="" (v2)  UPLOAD=1  HF_NAMESPACE  MAX_CHAIN=10
#
#SBATCH --job-name=dvr-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:2        # held from the start through download, retargeting and upload
#SBATCH --cpus-per-task=32          # 21 sessions (PER_GPU x 2 + stage 1) x ~1.4 cores (Bench2Dex measurement)
#SBATCH --mem=200G                  # 21 sessions x ~6.6 GB + headroom
# p1 = 4x pro6000 96 GB, 64 threads, 500000M (sysinfo job 1534, Bench2Dex experiments/ENVIRONMENT.md). The whole
# node: --gres=gpu:pro6000:4 --cpus-per-task=64 --mem=440G with PER_GPU=11 (64 threads / ~1.4 cores per session);
# the worker addresses however many GPUs the job holds.
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-dvr-main-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== dvr-main start: $(date) on $(hostname) ==="

REPO_URL="${REPO_URL:-https://github.com/eunyoung030927/DexVerse.git}"
export REPO_BRANCH="${REPO_BRANCH:-retarget/shadow-cross-hand}"
# First the (task, hand) pairs whose ep0 already succeeded on a4/amp7 (2026-10-16): GraspKettle (ep0 failed for every
# hand) and leap (ep0 succeeded only in GraspPan) come later, as do the bimanual tasks, each in its own job:
#   TASKS="Dexverse-GraspKettle-v1"  HANDS="allegro inspire leap sharpa wuji"
#   TASKS="Dexverse-GraspPan-v1 Dexverse-GraspBleach-v1 Dexverse-FunctionalPourMug-v1 Dexverse-FunctionalHammerStrike-v1"  HANDS="leap"
#   TASKS="Dexverse-BimanualLiftTray-v1 Dexverse-BimanualLiftCarton-v1 Dexverse-SlideUtilityKnife-v1 Dexverse-OpenFlatFolder-v1"
# Fill order: the tasks with the fewest results so far first.
export TASKS="${TASKS:-Dexverse-GraspPan-v1 Dexverse-FunctionalHammerStrike-v1 Dexverse-FunctionalPourMug-v1 Dexverse-GraspBleach-v1}"
export HANDS="${HANDS:-allegro inspire sharpa wuji}"   # target hand families; unsupported ones are skipped with a log line
export EPISODES="${EPISODES:-50}"
export PER_GPU="${PER_GPU:-10}" MIN_FREE_GB="${MIN_FREE_GB:-10}" SESSION_GB="${SESSION_GB:-6}"
CHAIN_N="${CHAIN_N:-1}"; MAX_CHAIN="${MAX_CHAIN:-10}"
echo "tasks (fill order): $TASKS"
echo "hands: $HANDS · episodes 0..$(( EPISODES - 1 )) · up to $PER_GPU sessions/GPU while >= ${MIN_FREE_GB} GB VRAM free · job $CHAIN_N/$MAX_CHAIN of the chain"

# One dvr job at a time: a job that finds another RUNNING one requeues itself behind it.
SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'Command=[^ ]*' | cut -d= -f2)"
[ -r "$SELF" ] || SELF="$0"                      # this very file (the pasted one)
BUSY="$(squeue -h -u "$USER" -t R -o '%i %j' 2>/dev/null | awk -v me="${SLURM_JOB_ID:-0}" '$1 != me && $2 ~ /^dvr-(main|retarget)/ {print $1}' | head -1)"
if [ -n "$BUSY" ]; then
  NEXT="$(sbatch --parsable --chdir="${SLURM_SUBMIT_DIR:-$PWD}" --dependency="afterany:$BUSY" --export=ALL,CHAIN_N="$CHAIN_N" "$SELF")"
  echo "dvr job $BUSY is running: requeued as job $NEXT after it; exiting"
  exit 0
fi

echo "=== [1/6] preflight ==="
[ -s "$HOME/.hf_token" ] || { echo "FATAL: ~/.hf_token missing or empty. Create it in the OOD Files app."; exit 1; }
echo "token ok"

echo "=== [2/6] git sync ==="
WS="$HOME/dexverse"
D="$WS/DexVerse"
mkdir -p "$WS"
export APPTAINER_CACHEDIR="/tmp/dexverse-$USER/apptainer-cache"
mkdir -p "$APPTAINER_CACHEDIR"
LG() { apptainer exec --bind "$WS" docker://hyeonseoke/lerobot:v1 git "$@"; }
if [ -d "$D/.git" ]; then
  LG -C "$D" fetch -q origin "$REPO_BRANCH" || { echo "FATAL: git fetch failed"; exit 1; }
  if [ "$(LG -C "$D" rev-parse --abbrev-ref HEAD)" != "$REPO_BRANCH" ]; then
    LG -C "$D" checkout -q -B "$REPO_BRANCH" "origin/$REPO_BRANCH"
  fi
  LG -C "$D" merge -q --ff-only "origin/$REPO_BRANCH" || { echo "FATAL: checkout diverged from origin/$REPO_BRANCH. Fix it by hand."; exit 1; }
else
  LG clone -q --branch "$REPO_BRANCH" "$REPO_URL" "$D"
fi
# shellcheck source=/dev/null
source "$D/ondemand/env.sh"
echo "DexVerse fork @ $(GIT -C "$REPO_DIR" rev-parse --short HEAD) ($REPO_BRANCH)"
CONT=""
if [ "$CHAIN_N" -lt "$MAX_CHAIN" ]; then     # continuation in case this job hits the walltime (from the pulled copy)
  CONT="$(sbatch --parsable --chdir="${SLURM_SUBMIT_DIR:-$PWD}" --dependency="afterany:${SLURM_JOB_ID}" \
          --export=ALL,CHAIN_N="$(( CHAIN_N + 1 ))" "$CODE_DIR/main_job.sbatch")"
  echo "continuation queued as job $CONT (cancelled below if this job finishes the queue)"
fi

echo "=== [3/6] setup ==="
# shellcheck source=/dev/null
source "$CODE_DIR/setup.sh"

echo "=== [4/6] fetch: demos of [$TASKS] ==="
export APPTAINERENV_TASKS="$TASKS"
"${ISAAC[@]}" "$PY" ondemand/fetch_data.py

echo "=== [5/6] retarget (this job, 2 GPUs, one queue over all tasks) ==="
bash "$CODE_DIR/retarget_worker.sbatch" || echo "[warn] retargeting ended with an error (see above); deciding the upload anyway"

echo "=== [6/6] upload (last dv-demos round) ==="
if [ "${UPLOAD:-1}" = "1" ]; then
  bash "$CODE_DIR/finalize.sbatch" || echo "[warn] upload step failed"
fi

# Queue finished? (no item left that is neither recorded nor out of attempts) -> no continuation needed.
LEFT="$("${ISAAC[@]}" "$PY" scripts/retarget/run_queue.py --tasks $TASKS --demos_root "$DEMOS_ROOT" \
        --episodes $(seq 0 $(( EPISODES - 1 ))) --hands $HANDS --dry_run 2>/dev/null | grep -c "would run" || true)"
if [ -n "$CONT" ] && [ "${LEFT:-1}" = "0" ]; then
  scancel "$CONT" && echo "queue finished: continuation job $CONT cancelled"
else
  echo "items left: ${LEFT:-?}${CONT:+ -> continuation job $CONT resumes them}"
fi
echo "=== dvr-main done: $(date) ==="
