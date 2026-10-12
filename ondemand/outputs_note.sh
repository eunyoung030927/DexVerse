#!/bin/bash
# Write OUTPUTS.md (where this job's results are on the cluster, with counts) into the job's folder.
# Called by retarget_worker.sbatch at its start and by finalize.sbatch at the end; TASKS from the job.
#   outputs_note.sh [DIR]   (default: $SLURM_SUBMIT_DIR, the OOD Job Composer folder of the job)
DIR="${1:-${SLURM_SUBMIT_DIR:-.}}"
R="$HOME/dexverse/runs/retarget"
{
  echo "# DexVerse retargeting outputs on the cluster"
  echo
  echo "job ${SLURM_JOB_ID:-?} · written $(date '+%F %T') · tasks [${TASKS:-?}]"
  echo
  echo "| what | path |"
  echo "|---|---|"
  echo "| per (episode, target): status.json, run.log, spider_aK.json / _traj.pkl.gz, spider.* -> the successful one | \`$R/<task>/demos/epNNN/<target>/\` |"
  echo "| stage-1 reference per episode | \`$R/<task>/demos/epNNN/reference.npz\` |"
  echo "| session logs | \`$R/logs/\` |"
  echo "| progress report | \`$HOME/dexverse/runs/report/STATUS.md\` · live: \`$HOME/dexverse/dvr_status.md\` |"
  echo "| source demos | \`$HOME/dexverse/demos/demonstrations/v1/<category>/<task>/demos.pkl\` |"
  echo "| uploaded pickle | HF \`<ns>/dexverse-<task>-<hand>-retargeting\`: \`demonstrations/v1/<category>/<task>/demos.pkl\` |"
  echo
  echo "## successful episodes so far"
  echo
  found=0
  for t in ${TASKS:-}; do
    for g in "$R/$t"/demos/ep000/*/; do
      [ -d "$g" ] || continue
      found=1
      tgt="$(basename "$g")"
      n=$(ls "$R/$t"/demos/ep*/"$tgt"/spider.json 2>/dev/null | wc -l)
      echo "- $t -> $tgt: $n"
    done
  done
  [ "$found" = 1 ] || echo "(none yet)"
} > "$DIR/OUTPUTS.md.tmp" && mv "$DIR/OUTPUTS.md.tmp" "$DIR/OUTPUTS.md"
echo "outputs note: $DIR/OUTPUTS.md"
