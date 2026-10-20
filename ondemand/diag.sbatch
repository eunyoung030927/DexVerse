#!/bin/bash
# Diagnose a running dvr-main job without a shell: OOD Job Composer -> New Job -> paste THIS FILE -> Submit.
# Read-only (changes nothing). Output: slurm-dvr-diag-<job>.out in the job folder (paste it back).
#   1. squeue           the user's jobs
#   2. GPU / processes  on the dvr-main node, via srun --overlap into that job (nvidia-smi, the retarget processes:
#                       elapsed time, CPU %, memory)
#   3. open items       every status.json not finished: state, attempts, minutes since run.log last changed,
#                       last lines of run.log (a stuck item shows a long silence)
#   4. errors           the last Tracebacks in the session logs, with context
#SBATCH --job-name=dvr-diag
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-dvr-diag-%j.out

export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
RUNS="$HOME/dexverse/runs/retarget"
echo "=== dvr-diag $(date '+%F %T') on $(hostname -s) ==="

echo; echo "## 1. squeue"
squeue -u "$USER" -o '%.8i %.12j %.3t %.12M %.12l %R'

MAIN="${MAIN_JOB:-$(squeue -h -u "$USER" -t R -o '%i %j' | awk '$2 ~ /^dvr-(main|retarget)/ {print $1; exit}')}"
echo; echo "## 2. node of job ${MAIN:-none}"
if [ -n "$MAIN" ]; then
  unset SLURM_JOB_ID SLURM_JOBID                          # srun must attach to MAIN, not to this job
  timeout 120 srun --jobid="$MAIN" --overlap --ntasks=1 --cpus-per-task=1 --mem=0 bash -c '
    nvidia-smi
    echo; echo "uptime: $(uptime)"
    echo; echo "PID ELAPSED %CPU RSS(MB) COMMAND"
    ps -u "$USER" -o pid=,etime=,pcpu=,rss=,args= --sort=-etime \
      | grep -E "stage[1234]_|run_target|run_queue" | grep -v grep \
      | awk "{\$4 = int(\$4/1024); print}" | cut -c1-230
  ' 2>&1 || echo "(srun into job $MAIN failed, rc=$?)"
fi

echo; echo "## 3. open items (status.json not recorded / failed_all_attempts / stage2_failed)"
now=$(date +%s)
for st in "$RUNS"/*/demos/ep*/*/status.json; do
  [ -f "$st" ] || continue
  state=$(grep -o '"state": *"[^"]*"' "$st" | cut -d'"' -f4)
  case "$state" in recorded|failed_all_attempts|stage2_failed) continue ;; esac
  d=$(dirname "$st"); log="$d/run.log"
  age=$(( (now - $(stat -c %Y "$log" 2>/dev/null || echo "$now")) / 60 ))
  tags=$(grep -o '"tag": *"[^"]*"' "$st" | cut -d'"' -f4 | tr '\n' ' ')
  echo "--- ${d#$RUNS/}  state=$state  attempts=[${tags}]  run.log idle ${age} min"
  tail -n 3 "$log" 2>/dev/null | cut -c1-200 | sed 's/^/    /'
done

echo; echo "## 4. last Tracebacks (session logs)"
for f in $(grep -l Traceback "$RUNS"/logs/*.log 2>/dev/null | xargs -r ls -t | head -3); do
  echo "--- $f ($(stat -c %y "$f" | cut -c1-19))"
  grep -n -A14 Traceback "$f" | tail -32 | cut -c1-220
done
for f in $(grep -l Traceback "$RUNS"/*/demos/ep*/*/run.log 2>/dev/null | xargs -r ls -t | head -3); do
  echo "--- ${f#$RUNS/} ($(stat -c %y "$f" | cut -c1-19))"
  grep -n -A14 Traceback "$f" | tail -32 | cut -c1-220
done
echo; echo "=== done $(date '+%F %T') ==="
