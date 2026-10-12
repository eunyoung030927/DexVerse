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
#   4. fetch      the task's released v1 demos (dexverse/DexVerse_release; skips what is there)
#   5. retarget   IN THIS JOB, on the 2 pro6000 GPUs it holds: CUDA MPS, PER_GPU (10) sessions per GPU = 20 at once,
#                 through every (episode, target hand): SPIDER (v2 settings, 60 Hz) until the task's own success term
#                 holds 10 steps, up to MAX_ATTEMPTS (ondemand/retarget_worker.sbatch -> scripts/retarget/run_queue.py
#                 -> run_target.py; no rendering on the cluster)
#   6. upload     IN THIS JOB (ondemand/finalize.sbatch): per target hand, the successful runs become one trajectory
#                 pickle (record_demos format) pushed to <ns>/dexverse-<task>-<hand>-retargeting ONLY if all
#                 EPISODES source episodes succeeded
#
# Live status: ~/dexverse/dvr_status_<task>.md (OOD Files app), rewritten every 5 min; summary lines in this log.
#
# Everything after step 2 is read from the pulled fork, so edits to scripts/retarget/, source/ and ondemand/*.sh
# only need a push. Only an edit to THIS file needs re-pasting.
#
# Intended use: submit ONCE; the tasks of TASK_QUEUE run one after another, one job each, all 50 source episodes.
# Resubmitting is always safe: finished stages and successful episodes are skipped; a job that starts while another
# job on the same task runs requeues itself behind it. To stop the chain, cancel the running job AND the queued one.
#
# Knobs (sbatch --export=ALL,KNOB=value, or edit the defaults below):
#   TASK_QUEUE="Dexverse-GraspBleach-v1 ..."  (TASKS=<task> runs one task alone)  HANDS="allegro wuji"  EPISODES=50
#   MAX_ATTEMPTS=3  PER_GPU=10  MPS_ON=1  SPIDER_ARGS="" (v2)  PARALLEL=0  UPLOAD=1  HF_NAMESPACE (default: token's user)
#
#SBATCH --job-name=dvr-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:2        # held from the start through download, retargeting and upload
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-dvr-main-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== dvr-main start: $(date) on $(hostname) ==="

REPO_URL="${REPO_URL:-https://github.com/eunyoung030927/DexVerse.git}"
export REPO_BRANCH="${REPO_BRANCH:-retarget/shadow-cross-hand}"
# Tasks run one per job, in this order: this job takes the first, and right at its start queues the next job for
# the rest (--dependency=afterany on this one), so the chain goes on even if a task fails or hits the walltime.
ALONE=0; [ -n "${TASKS:-}" ] && [ -z "${TASK_QUEUE:-}" ] && ALONE=1      # TASKS=<task> alone: no chain
TASK_QUEUE="${TASK_QUEUE:-Dexverse-GraspBleach-v1 Dexverse-GraspPan-v1 Dexverse-GraspKettle-v1 Dexverse-FunctionalPourMug-v1 Dexverse-FunctionalHammerStrike-v1 Dexverse-BimanualLiftTray-v1 Dexverse-BimanualLiftCarton-v1 Dexverse-SlideUtilityKnife-v1 Dexverse-OpenFlatFolder-v1}"
export TASKS="${TASKS:-${TASK_QUEUE%% *}}"
export HANDS="${HANDS:-allegro inspire leap sharpa wuji}"   # target hand families; unsupported ones are skipped with a log line
REST="$(echo " $TASK_QUEUE " | sed "s/ $TASKS / /" | xargs)"
[ "$ALONE" = 1 ] && REST=""
export EPISODES="${EPISODES:-50}"
echo "task $TASKS now (hands: $HANDS); queued after it: [${REST:-none}]"

# Jobs for different tasks may run side by side; a job that finds another RUNNING job on the same task requeues
# itself behind it. PARALLEL=0 (default): one dvr job at a time, so the tasks run strictly one after another.
LOCK="$HOME/dexverse/locks/${TASKS}.jobid"; mkdir -p "$(dirname "$LOCK")"
if [ "${PARALLEL:-0}" = "1" ]; then
  BUSY="$(cat "$LOCK" 2>/dev/null || true)"
  [ -n "$BUSY" ] && [ "$BUSY" != "${SLURM_JOB_ID:-0}" ] && squeue -h -j "$BUSY" -t R 2>/dev/null | grep -q . || BUSY=""
else
  BUSY="$(squeue -h -u "$USER" -t R -o '%i %j' 2>/dev/null | awk -v me="${SLURM_JOB_ID:-0}" '$1 != me && $2 ~ /^dvr-(main|retarget)/ {print $1}' | head -1)"
fi
if [ -n "$BUSY" ]; then
  SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'Command=[^ ]*' | cut -d= -f2)"
  [ -r "$SELF" ] || SELF="$0"
  NEXT="$(sbatch --parsable --chdir="${SLURM_SUBMIT_DIR:-$PWD}" --dependency="afterany:$BUSY" \
          --export=ALL,TASK_QUEUE="$TASK_QUEUE",TASKS="$TASKS" "$SELF")"
  echo "dvr job $BUSY is running on $TASKS: requeued as job $NEXT after it; exiting"
  exit 0
fi
echo "${SLURM_JOB_ID:-$$}" > "$LOCK"

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
if [ -n "$REST" ]; then      # queue the next task now (from the freshly pulled copy of this file)
  NEXT="$(sbatch --parsable --chdir="${SLURM_SUBMIT_DIR:-$PWD}" --dependency="afterany:${SLURM_JOB_ID}" \
          --export=ALL,TASK_QUEUE="$REST",TASKS="${REST%% *}" "$CODE_DIR/main_job.sbatch")"
  echo "next task ${REST%% *} queued as job $NEXT (runs after this one); then [$(echo " $REST " | sed "s/ ${REST%% *} / /" | xargs)]"
fi

echo "=== [3/6] setup ==="
# shellcheck source=/dev/null
source "$CODE_DIR/setup.sh"

echo "=== [4/6] fetch: [$TASKS] demos ==="
export APPTAINERENV_TASKS="$TASKS"
"${ISAAC[@]}" "$PY" ondemand/fetch_data.py

echo "=== [5/6] retarget (this job, 2 GPUs) ==="
bash "$CODE_DIR/retarget_worker.sbatch" || echo "[warn] retargeting ended with an error (see above); deciding the upload anyway"

echo "=== [6/6] upload (all-or-nothing per target hand) ==="
if [ "${UPLOAD:-1}" = "1" ]; then
  bash "$CODE_DIR/finalize.sbatch" || echo "[warn] upload step failed"
fi
echo "=== dvr-main done: $(date) ==="
