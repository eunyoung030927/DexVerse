#!/bin/bash
# After a task's retargeting: per target hand, collect the successful runs of all EPISODES source episodes into one
# trajectory pickle and push it to <ns>/dexverse-<task>-<hand>-retargeting (+ dataset card) -- ONLY if every one of
# them succeeded; otherwise nothing is pushed and the missing episodes are printed (scripts/retarget/upload_task.py).
# Resubmit main_job.sbatch to retry the missing ones; the upload is decided again at its end.
# Run by ondemand/main_job.sbatch at its end (bash, not sbatch); can also be submitted on its own.
#
#SBATCH --job-name=dvr-finalize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1        # unused; the gpu partition does not schedule 0-GPU jobs
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-dvr-finalize-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
# shellcheck source=/dev/null
source "$HOME/dexverse/DexVerse/ondemand/env.sh"
sync_repo
export APPTAINERENV_HF_NAMESPACE="${HF_NAMESPACE:-}"
for t in ${TASKS:?}; do
  "${ISAAC[@]}" "$PY" scripts/retarget/upload_task.py --task "$t" --demos_root "$DEMOS_ROOT" --episodes "${EPISODES:-50}" \
    --hands "${HANDS:-allegro inspire leap sharpa wuji}" || echo "[info] task $t: not every target complete, see above"
done
bash "$CODE_DIR/outputs_note.sh" || true
