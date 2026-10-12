#!/bin/bash
# Upload a finished task's retargeted pickles to Hugging Face NOW, even if some episodes failed (they are listed in
# the repo's failures.json and can be added later: upload_task.py merges with the pickle already on the Hub, so a
# later run that retargeted only the missing ones completes the repo to 50/50).
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> edit TASKS below if needed -> Submit.
#
# Knobs: TASKS="Dexverse-GraspBleach-v1 ..."  EPISODES=50  HANDS="allegro inspire leap sharpa wuji"
#        ALLOW_PARTIAL=1 (0 = only targets with all EPISODES succeeded)  HF_NAMESPACE (default: token's user)
#
#SBATCH --job-name=dvr-upload
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1        # unused; the gpu partition does not schedule 0-GPU jobs
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-dvr-upload-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
TASKS="${TASKS:-Dexverse-GraspBleach-v1}"
EPISODES="${EPISODES:-50}"
HANDS="${HANDS:-allegro inspire leap sharpa wuji}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-1}"
# shellcheck source=/dev/null
source "$HOME/dexverse/DexVerse/ondemand/env.sh"
sync_repo
export APPTAINERENV_HF_NAMESPACE="${HF_NAMESPACE:-}"
P=""; [ "$ALLOW_PARTIAL" = "1" ] && P="--allow_partial"
for t in $TASKS; do
  "${ISAAC[@]}" "$PY" scripts/retarget/upload_task.py --task "$t" --demos_root "$DEMOS_ROOT" --episodes "$EPISODES" \
    --hands "$HANDS" $P || echo "[info] task $t: some targets incomplete (see above)"
done
