#!/bin/bash
# After the retargeting: one last dv-demos upload round (ondemand/dv_upload.sh 0): every successful episode of TASKS as
# v1/retargeting/<category>/<task>/<robot_type>/epNNN.pkl + index.json to the HF dataset Dexverse-CSI/dv-demos, merged
# per episode with the repo. During the job the worker already uploads every DV_UPLOAD_S s.
# (The old per-task repos <ns>/dexverse-<task>-<hand>-retargeting, scripts/retarget/upload_task.py, are not used.)
# Run by ondemand/main_job.sbatch at its end (bash, not sbatch); can also be submitted on its own.
#
#SBATCH --job-name=dvr-finalize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1        # arm_joint_meta backfill (short Isaac launches)
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-dvr-finalize-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
# shellcheck source=/dev/null
source "$HOME/dexverse/DexVerse/ondemand/env.sh"
sync_repo
export TASKS="${TASKS:?}"
bash "$CODE_DIR/dv_upload.sh" 0
bash "$CODE_DIR/outputs_note.sh" || true
