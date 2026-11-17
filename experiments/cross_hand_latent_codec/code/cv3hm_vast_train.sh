#!/bin/bash
# CV3HM training on vast 2x5090 (261117): 3-hand (shadow + wuji + allegro) nohot FunctionalHammerStrike image DP, one arm per
# invocation. Copy of cv3dp_vast_train.sh (unchanged original) with only: RUNS/DS = cv3hm, markers [cv3hm_vast], and a
# stricter START GATE (coordinator 261117: the cv3dp D arms train on vast first):
#   - /workspace/xhand/runs/cv3dpD/TRAINED_ALL exists (written by the D agent), AND
#   - dataset carries .cv2dp_staged, AND
#   - GPU <gpu> has >= NEED_MIB free (default 28000) and no compute process on it, AND
#   - / has >= NEED_DISK_GB free (default 22; one arm keeps 6 x 1.1 GB pretrained_model + <= 2 x 2.1 GB training_state).
# Recipe unchanged: train_lerobot.py --resize (LeRobot 0.4.2 image DP), 300 epochs, batch 64, seed 42 (train_lerobot
# default), snapshot every 50 epochs, num_workers 12, --resume retries, max 5 attempts; pruner of older training_state.
# Dataset /workspace/xhand/datasets/cv3hm/<ds> (default wsa_<arm>) -> /workspace/xhand/runs/cv3hm/<arm>/train.
# Markers (UTC) in /workspace/xhand/runs/cv3hm/logs/vast_queue.log:
#   [cv3hm_vast] <arm>: WAITING ... | GATE OPEN gpu N | training ... | TRAINED | TRAIN FAILED | GIVING UP | pruned ...
# usage: nohup bash /workspace/xhand/cv3hm_vast_train.sh <arm> <gpu> [dataset_dir_name] > /dev/null 2>&1 &
arm=$1; gpu=$2; dsn=${3:-wsa_$1}
R=/workspace/xhand/lerobot_dp_pretrained
CODE=$R/code/lerobot_dp
DS=/workspace/xhand/datasets/cv3hm
RUNS=/workspace/xhand/runs/cv3hm
LOG=$RUNS/logs; mkdir -p $LOG $RUNS/$arm
Q=$LOG/vast_queue.log
PY=${PY:-/opt/conda/bin/python}
NEED_MIB=${NEED_MIB:-28000}
NEED_DISK_GB=${NEED_DISK_GB:-22}
D_DONE=${D_DONE:-/workspace/xhand/runs/cv3dpD/TRAINED_ALL}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 TZ=UTC
export EXPERIMENTS_ROOT=$R/experiments HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ACCELERATE_MIXED_PRECISION=no
EPOCHS=${EPOCHS:-300}; NUM_WORKERS=${NUM_WORKERS:-12}
ts() { date -u '+%F %T UTC'; }
echo $$ > $LOG/launcher_$arm.pid
echo "[cv3hm_vast] $arm: start $(ts) pid $$ gpu $gpu dataset $dsn" | tee -a $Q

gate() {
  [ -f $D_DONE ] || { GATE_WHY="$D_DONE missing (cv3dp D arms first)"; return 1; }
  [ -f $DS/$dsn/.cv2dp_staged ] || { GATE_WHY="dataset $dsn not staged"; return 1; }
  local disk; disk=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  [ "${disk:-0}" -ge $NEED_DISK_GB ] || { GATE_WHY="/ has < $NEED_DISK_GB GB free"; return 1; }
  local uuid procs; uuid=$(nvidia-smi -i $gpu --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  procs=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | grep -c "$uuid")
  [ "${procs:-0}" -eq 0 ] || { GATE_WHY="GPU $gpu has a compute process"; return 1; }
  local free; free=$(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "${free:-0}" -ge $NEED_MIB ]; then GPU=$gpu; return 0; fi
  GATE_WHY="GPU $gpu has < $NEED_MIB MiB free"; return 1
}

prune() {  # delete training_state of every numbered snapshot older than the newest one with a complete training_state
  local C=$RUNS/$arm/train/checkpoints newest="" d
  for d in $(ls -d $C/[0-9]* 2>/dev/null | sort); do
    [ -f $d/training_state/training_step.json ] && [ -f $d/pretrained_model/model.safetensors ] \
      && [ $(( $(date +%s) - $(stat -c %Y $d/training_state) )) -ge 300 ] && newest=$d  # >= 5 min old: save finished
  done
  [ -n "$newest" ] || return
  for d in $(ls -d $C/[0-9]* 2>/dev/null | sort); do
    [ "$d" \< "$newest" ] && [ -d $d/training_state ] || continue
    rm -rf $d/training_state && echo "[cv3hm_vast] $arm: pruned $(basename $d)/training_state $(ts) (newest $(basename $newest))" | tee -a $Q
  done
}

last=""
until gate; do
  [ "$GATE_WHY" != "$last" ] && echo "[cv3hm_vast] $arm: WAITING $(ts): $GATE_WHY" | tee -a $Q
  last=$GATE_WHY
  sleep 300
done
echo "[cv3hm_vast] $arm: GATE OPEN $(ts) gpu $GPU" | tee -a $Q

D=$DS/$dsn; C=$RUNS/$arm/train
final=$($PY -c "import json, math; print($EPOCHS * math.ceil(json.load(open('$D/meta/info.json'))['total_frames'] / 64))")
attempt=0
while [ ! -f $RUNS/$arm/trained ]; do
  attempt=$((attempt+1))
  resume=; [ -e $C/checkpoints/last ] && resume=--resume
  echo "[cv3hm_vast] $arm: training gpu $GPU $(ts) final step $final ${resume} attempt $attempt" | tee -a $Q
  CUDA_VISIBLE_DEVICES=$GPU $PY $CODE/train_lerobot.py --dataset_root $D --output_dir $C \
      --epochs $EPOCHS --snapshot_epochs 50 --num_workers $NUM_WORKERS --resize $resume >> $LOG/train_$arm.log 2>&1 &
  tp=$!; echo $tp > $LOG/train_$arm.pid
  while kill -0 $tp 2>/dev/null; do sleep 120; prune; done
  wait $tp
  prune
  if [ "$($PY -c "import json; print(json.load(open('$C/checkpoints/last/training_state/training_step.json'))['step'])" 2>/dev/null)" = $final ]; then
    touch $RUNS/$arm/trained; echo "[cv3hm_vast] $arm: TRAINED $(ts)" | tee -a $Q; break
  fi
  echo "[cv3hm_vast] $arm: TRAIN FAILED attempt $attempt $(ts)" | tee -a $Q
  [ $attempt -ge 5 ] && { echo "[cv3hm_vast] $arm: GIVING UP after 5 attempts" | tee -a $Q; break; }
  sleep 300
done
