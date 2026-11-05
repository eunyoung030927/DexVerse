#!/bin/bash
# One GPU session: trains the teleop sets one after another (240x320 input), each with snapshot selection on pickle
# episodes 0-9 and final 50 rollouts of best on all 50 pickle episodes (last is kept for resuming, not evaluated: 2026-11-02).
# The evaluation of a set runs in the background while the next set trains (2026-11-02 15:30 KST; one eval job per GPU at a time).
#   RID=<YYYYMMDD-HHMMSS> bash /workspace/codes/dexverse/DexVerse/baselines/lerobot_dp/run_sets.sh <gpu> <Task>:<hand> ...   (same RID resumes)
# Code lives in the DexVerse repo (baselines/lerobot_dp, 2026-11-05); runs are written under RUNS (default: this server's
# /workspace/experiments/<server>/20261031_teleop_single/teleop_dp_image). Task may carry its version (SlideUtilityKnife-v1), default -v0.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)  # before cd: the script dir (eval_lerobot.py, finalize_run.py live here)
SERVER=${SERVER:-$(ls /workspace/experiments | grep -v -E '^(_|\.)|\.(py|txt)$' | head -1)}  # the one experiments/<server> dir of this container
source /workspace/codes/_setup/dexverse_env.sh DexVerse; cd $DEXVERSE_ROOT  # dev checkout (the DexVerse-develop worktree was retired 2026-11-05)
unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
CK=/workspace/checkpoints/20261031_teleop_single
LP=/workspace/envs/dexverse/lerobot/bin/python; GPU=$1; shift; EPOCHS=${EPOCHS:-300}
NUM_WORKERS=${NUM_WORKERS:-12}  # data loader workers per training (vast: 24, its GPUs outrun 12)
SNAPSHOT_EPOCHS=${SNAPSHOT_EPOCHS:-50}; MAX_STEPS=${MAX_STEPS:-3000}; RUNS=${RUNS:-/workspace/experiments/$SERVER/20261031_teleop_single/teleop_dp_image}; CK=${CK_ROOT:-$CK}  # overrides: pipeline tests only
RID=${RID:-$(date -u -d '+9 hours' +%Y%m%d-%H%M%S)}
W=/workspace/codes/_setup/gpu_watch.sh; bash $W ensure  # GPU fault watchdog attaches to every session (GPU_FAULT_MONITOR.md)
vram_wait() { while [ "$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits)" -lt ${1:-8000} ]; do sleep 60; done; }

rollout() {  # run_dir ckpt out episodes [--no-video]   (two attempts; a rollout whose log is silent for 60 min is killed: Isaac Sim can hang at start-up; eval prints a heartbeat every 500 steps)
  local log=$1/logs/eval_$(basename $3).log attempt pid
  for attempt in 1 2; do
    [ $attempt = 2 ] && { echo "[run] $(basename $1) $(basename $3): retry"; rm -rf $3; }
    bash $W wait $GPU 2>/dev/null  # GPU fault (MPS stuck / lost): wait for the watchdog's recovery instead of failing
    exec 8>>$RUNS/.vram_lock_gpu$GPU; flock 8  # one rollout starts at a time, and only when the GPU has room for it (Isaac Sim + policy server ~6.5 GB)
    vram_wait ${VRAM_FREE_MIB:-8000}
    CUDA_VISIBLE_DEVICES=$GPU python $HERE/eval_lerobot.py --demos $PK --ckpt $2 --output_dir $3 --episodes $4 --max_steps $MAX_STEPS ${5:-} --headless > $log 2>&1 & pid=$!
    sleep 120; flock -u 8; exec 8>&-
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      [ $(( $(date +%s) - $(stat -c %Y $log) )) -gt ${STALL_SEC:-3600} ] && { echo "[run] $(basename $1) $(basename $3): stalled, killed"; pkill -P $pid; kill $pid; sleep 5; pkill -9 -P $pid; kill -9 $pid 2>/dev/null; }
    done 2>/dev/null; wait $pid 2>/dev/null
    [ -f $3/metrics.json ] && [ "$(python3 -c "import json; print(json.load(open('$3/metrics.json'))['num_episodes'])")" = "$(echo $4 | tr , '\n' | wc -l)" ] && return 0
  done
  return 1
}
select_snapshots() {  # run_dir ck final_step
  local r=$1 c=$2/checkpoints final=$3 s o ok best pid
  mkdir -p $r/eval/select
  while true; do
    for s in $(ls $c 2>/dev/null | grep -E '^[0-9]+$' | sort); do
      [ $((10#$s)) = $final ] && continue
      [ -f $c/$s/training_state/training_step.json ] || continue
      o=$r/eval/select/$s
      pid=$(pgrep -f "eval_lerobot.py .* --output_dir $o ")  # left by a session relaunched meanwhile: let it finish
      [ -n "$pid" ] && while kill -0 $pid 2>/dev/null; do sleep 30; done
      if [ "$(python3 -c "import json; print(json.load(open('$o/metrics.json'))['num_episodes'])" 2>/dev/null)" != 10 ]; then
        rm -rf $o; rollout $r $c/$s/pretrained_model $o $(seq -s, 0 9) --no-video || { echo "[run] $(basename $r) $s: SELECT FAILED"; continue; }
      fi
      if ! grep -q "^$s " $r/logs/select.log 2>/dev/null; then  # also a rollout finished by a session that was relaunched meanwhile
        ok=$(python3 -c "import json; print(sum(e['success'] for e in json.load(open('$o/metrics.json'))['episodes']))")
        best=$(cut -d' ' -f2 $r/eval/select/best.txt 2>/dev/null)
        echo "$s $ok/10" >> $r/logs/select.log
        if [ -z "$best" ] || [ $ok -ge ${best%/10} ]; then  # ties go to the later snapshot
          rm -rf $2/best.tmp; cp -r $c/$s/pretrained_model $2/best.tmp; rm -rf $2/best; mv $2/best.tmp $2/best; echo "$s $ok/10" > $r/eval/select/best.txt
        fi
        echo "[run] $(basename $r) $s: $ok/10 (best $(cat $r/eval/select/best.txt))"
      fi
      [ "$(readlink $c/last)" = $s ] || rm -rf $c/$s
    done
    [ -f $r/logs/trained ] && [ -z "$(for s in $(ls $c | grep -E '^[0-9]+$'); do [ $((10#$s)) = $final ] || [ "$(readlink $c/last)" = $s ] || echo x; done)" ] && return 0
    sleep 120
  done
}
eval_set() {  # run r c final sel_pid: remaining snapshot selection, then the 50 best rollouts and the run files.
  # Runs as a background job so the next set trains meanwhile; the eval jobs of one GPU run one at a time (flock).
  local run=$1 r=$2 c=$3 final=$4 sel=$5 a=$(seq -s, 0 24) b=$(seq -s, 25 49) part e evals
  while kill -0 $sel 2>/dev/null; do sleep 30; done
  evals=$(pgrep -f "eval_lerobot.py .* --output_dir $r/eval/best")  # relaunched during the final rollouts: let them finish
  if [ -n "$evals" ]; then echo "[run] $run: attached to running evaluation (pids $(echo $evals | tr '\n' ' '))"; for pid in $evals; do while kill -0 $pid 2>/dev/null; do sleep 60; done; done; fi
  local round; for round in 1 2 3; do  # each round = 2 rollout attempts per part; rounds wait for the GPU watchdog's recovery
    for part in 0 1; do e=$a; [ $part = 1 ] && e=$b
      [ -f $r/eval/best_$part/metrics.json ] && [ "$(python3 -c "import json; print(json.load(open('$r/eval/best_$part/metrics.json'))['num_episodes'])")" = 25 ] && continue
      { rm -rf $r/eval/best_$part; rollout $r $c/best $r/eval/best_$part $e || echo "[run] $run best part$part: EVAL FAILED (round $round)"; } &
    done; wait
    [ "$(python3 -c "import json,glob; print(sum(json.load(open(f))['num_episodes'] for f in glob.glob('$r/eval/best_[01]/metrics.json')))" 2>/dev/null)" = 50 ] && break
    sleep 300; bash $W wait $GPU 2>/dev/null
  done
  if [ "$(python3 -c "import json,glob; print(sum(json.load(open(f))['num_episodes'] for f in glob.glob('$r/eval/best_[01]/metrics.json')))" 2>/dev/null)" = 50 ]; then
    python3 $HERE/finalize_run.py end $r done
    touch $r/logs/finished; echo "[run] $run: finished $(grep "^$run," $RUNS/INDEX.csv | tail -1)"
  else
    python3 $HERE/finalize_run.py end $r failed; echo "[run] $run: EVAL FAILED after 3 rounds, left unfinished (relaunch the session to retry)"
  fi
}
one_set() {  # Task hand: trains (or attaches to / resumes a training), then hands the evaluation to a background job
  local T=$1 H=$2 TV=$1 B D slug run r c; [[ $TV == *-v[0-9] ]] || TV=$TV-v0  # Task may carry its version: SlideUtilityKnife-v1 (default -v0)
  B=/workspace/datasets/dexverse/teleop_lerobot/Dexverse-$TV/$2; D=$(ls -d $B/*-v[0-9]-floating_* | grep -v spool | head -1)
  PK=$(python3 -c "import json; print([json.loads(l) for l in open('$B/lerobot_conversions.jsonl')][-1]['pickle'])")
  slug=${T,,}; slug=${slug#functional}; slug=${slug#bimanual}; slug=$slug-${H%_*}
  run=${RID}_${SERVER}_$slug; r=$RUNS/$run; c=$CK/$run
  [ -f $r/logs/finished ] && return 0
  mkdir -p $r/logs $r/eval $r/artifacts $CK
  python3 $HERE/finalize_run.py start $r "$TV" "$H" 320 "$PK" "$D" "$c" "$EPOCHS" "RID=$RID NUM_WORKERS=$NUM_WORKERS run_sets.sh $GPU $T:$H" "$SERVER" "$NUM_WORKERS"
  local final=$(python3 -c "import json, math; print($EPOCHS * math.ceil(json.load(open('$D/meta/info.json'))['total_frames'] / 64))")
  select_snapshots $r $c $final & local sel=$!
  if [ ! -f $r/logs/trained ]; then  # a session relaunched with the same RID attaches to a training it finds running
    local pid=$(pgrep -f "train_lerobot.py --dataset_root $D --output_dir $c ")
    if [ -n "$pid" ]; then
      echo "[run] $run: attached to running training (pid $pid)"
      while kill -0 $pid 2>/dev/null; do sleep 60; done
    fi
    [ "$(python3 -c "import json; print(json.load(open('$c/checkpoints/last/training_state/training_step.json'))['step'])" 2>/dev/null)" = $final ] && touch $r/logs/trained
  fi
  local attempt=0 resume
  while [ ! -f $r/logs/trained ]; do
    attempt=$((attempt+1)); bash $W wait $GPU 2>/dev/null; vram_wait ${TRAIN_VRAM_MIB:-10000}
    resume=; [ -e $c/checkpoints/last ] && resume=--resume
    echo "[run] $run: training on gpu $GPU ($(date -u -d '+9 hours' '+%F %T') KST)${resume:+ $resume}$([ $attempt -gt 1 ] && echo " attempt $attempt")"
    env -u LD_PRELOAD -u PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU $LP $HERE/train_lerobot.py --dataset_root $D --output_dir $c --epochs $EPOCHS --snapshot_epochs $SNAPSHOT_EPOCHS --num_workers $NUM_WORKERS --resize $resume >> $r/logs/train.log 2>&1 \
      && touch $r/logs/trained || { echo "[run] $run: TRAIN FAILED (attempt $attempt)"
           [ $attempt -ge 5 ] && { kill $sel; python3 $HERE/finalize_run.py end $r failed; return 1; }; sleep 300; }
  done
  echo "[run] $run: trained, evaluation continues in the background ($(date -u -d '+9 hours' '+%F %T') KST)"
  ( flock 9; eval_set $run $r $c $final $sel ) 9>>$RUNS/.eval_lock_gpu$GPU &
}
mkdir -p $RUNS; echo "[run] start $(date -u -d '+9 hours' '+%F %T') KST RID=$RID gpu $GPU: $* (code $HERE @ $(git -C $HERE rev-parse --short HEAD 2>/dev/null), runs $RUNS)"
for item in "$@"; do one_set ${item%%:*} ${item##*:}; done
wait  # evaluation jobs of the last sets
echo "[run] ALL DONE $(date -u -d '+9 hours' '+%F %T') KST"
