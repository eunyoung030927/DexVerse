#!/bin/bash
# Live status of a running retargeting worker -- started BY the worker in the background (nothing to type on the
# cluster). Every INTERVAL seconds (default 300) it rewrites
#   $HOME/dexverse/dvr_status.md   (open it in the OOD Files app)
# and prints a one-line summary into the worker's SLURM log.
#   monitor.sh WORKER_LOG [INTERVAL]
# Shows: progress per (task, target) (runs/report/STATUS.md: status_report.py, refreshed by run_queue.py), success on the first
# attempt vs a retry, the queue's latest lines and errors, GPU memory / utilisation / sessions, MPS server, node /tmp.
LOG="$1"; INTERVAL="${2:-300}"
RUNS="$HOME/dexverse/runs/retarget"
REPORT="$HOME/dexverse/runs/report/STATUS.md"
OUT="$HOME/dexverse/dvr_status.md"            # one dvr job at a time
LINE='\] \S+ ep[0-9]+ -> \S+: '
while true; do
  fin=$(grep -cE "$LINE" "$LOG" 2>/dev/null || true)
  ok=$(grep -cE "${LINE}recorded" "$LOG" 2>/dev/null || true)
  bad=$(grep -cE "${LINE}(failed_all_attempts|stage2_failed)" "$LOG" 2>/dev/null || true)
  ses=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  mps=$(pgrep -u "$(id -u)" nvidia-cuda-mps >/dev/null && echo running || echo "not running")
  tmp=$(df -h /tmp | tail -1 | awk '{print $4" free ("$5" used)"}')
  {
    echo "# DexVerse retargeting status"
    echo
    echo "updated $(date '+%F %T')  ·  job ${SLURM_JOB_ID:-?} on $(hostname -s)  ·  log \`$LOG\`"
    echo
    echo "results: \`$RUNS/<task>/demos/epNNN/<target>/\` (status.json, spider_aK.json, trajectories)"
    echo
    echo "## items finished: ${fin:-0} (recorded ${ok:-0}, failed ${bad:-0})"
    echo
    if [ -f "$REPORT" ]; then grep -E '^Total|^\||^updated' "$REPORT"; else echo "(report not written yet)"; fi
    echo
    echo "## success by attempt (first try vs retry)"
    echo
    echo "| task | target | 1st try | retry |"
    echo "|---|---|---|---|"
    for t in $TASKS; do
      for d in "$RUNS/$t"/demos/ep*/*/; do
        [ -f "$d/spider.json" ] || continue
        echo "$(basename "$d") $(readlink "$d/spider.json")"
      done | awk -v t="$t" '{ if ($2 == "spider_a0.json") a[$1]++; else b[$1]++; k[$1] = 1 }
                            END { for (g in k) printf "| %s | %s | %d | %d |\n", t, g, a[g], b[g] }'
    done
    echo
    echo "## node"
    echo '```'
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null
    echo "sessions on GPUs: $ses   MPS server: $mps   /tmp: $tmp"
    echo '```'
    echo
    echo "## latest queue lines"
    echo '```'
    grep -E '\] (start|stage 1|skip|\S+ ep[0-9]+ -> )' "$LOG" 2>/dev/null | tail -15
    echo '```'
    echo
    echo "## errors (last 5)"
    echo '```'
    grep -hiE 'Traceback|out of memory|PhysX state|FATAL' "$LOG" "$RUNS"/logs/*.log 2>/dev/null | tail -5
    echo '```'
  } > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
  echo "[monitor $(date +%T)] finished ${fin:-0} (ok ${ok:-0}, failed ${bad:-0}) | sessions $ses | MPS $mps | /tmp $tmp"
  sleep "$INTERVAL"
done
