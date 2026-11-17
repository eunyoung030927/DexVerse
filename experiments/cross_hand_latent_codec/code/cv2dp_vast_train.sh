#!/bin/bash
# CV2DP training on vast 2x5090 (261115): 2-hand (shadow + wuji) nohot GraspKettle image DP, one arm per invocation.
# Recipe = ho_vast_train.sh / p3h_vast_train.sh unchanged: train_lerobot.py --resize (LeRobot 0.4.2 image DP), 300 epochs,
# batch 64, seed 42 (train_lerobot default), snapshot every 50 epochs, num_workers 12, --resume retries, max 5 attempts.
# Dataset /workspace/xhand/datasets/cv2dp/<ds> (default ws_<arm>) -> /workspace/xhand/runs/cv2dp/<arm>/train.
# START GATE (polled every 5 min; nothing under runs/pi05_nohot is touched):
#   1. "[pi05_vast] ALL DONE" in runs/pi05_nohot/logs/queue.log, OR ("[pi05_vast] D: TRAINED" and no pi05_train.py alive),
#   2. dataset carries .cv2dp_staged (written after the sha256 check of the staging),
#   3. GPU <gpu> (or, for gpu=auto, any GPU) has >= NEED_MIB free (default 28000).
# Markers (UTC) in /workspace/xhand/runs/cv2dp/logs/vast_queue.log:
#   [cv2dp_vast] <arm>: WAITING ... | GATE OPEN gpu N | training ... | TRAINED | TRAIN FAILED | GIVING UP
# usage: nohup bash /workspace/xhand/cv2dp_vast_train.sh <arm> <gpu|auto> [dataset_dir_name] > /dev/null 2>&1 &
arm=$1; gpu=$2; dsn=${3:-ws_$1}
R=/workspace/xhand/lerobot_dp_pretrained
CODE=$R/code/lerobot_dp
DS=${DS:-/workspace/xhand/datasets/cv2dp}  # Zv2m: DS=/workspace/xhand/datasets/handmask
RUNS=/workspace/xhand/runs/cv2dp
LOG=$RUNS/logs; mkdir -p $LOG $RUNS/$arm
Q=$LOG/vast_queue.log
PIQ=/workspace/xhand/runs/pi05_nohot/logs/queue.log
PY=${PY:-/opt/conda/bin/python}
NEED_MIB=${NEED_MIB:-28000}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 TZ=UTC
export EXPERIMENTS_ROOT=$R/experiments HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ACCELERATE_MIXED_PRECISION=no
EPOCHS=${EPOCHS:-300}; NUM_WORKERS=${NUM_WORKERS:-12}
ts() { date -u '+%F %T UTC'; }
echo $$ > $LOG/launcher_$arm.pid
echo "[cv2dp_vast] $arm: start $(ts) pid $$ gpu $gpu dataset $dsn" | tee -a $Q

gate() {
  if grep -q "\[pi05_vast\] ALL DONE" $PIQ 2>/dev/null; then :
  elif grep -q "\[pi05_vast\] D: TRAINED" $PIQ 2>/dev/null && ! pgrep -f pi05_train.py > /dev/null; then :
  else GATE_WHY="pi0.5 queue not done"; return 1; fi
  if pgrep -f pi05_train.py > /dev/null; then GATE_WHY="a pi05_train.py is still alive"; return 1; fi
  [ -f $DS/$dsn/.cv2dp_staged ] || { GATE_WHY="dataset $dsn not staged"; return 1; }
  local g cand; [ "$gpu" = auto ] && cand="0 1" || cand=$gpu
  for g in $cand; do
    local free; free=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [ "${free:-0}" -ge $NEED_MIB ]; then GPU=$g; return 0; fi
  done
  GATE_WHY="no GPU in [$cand] with >= $NEED_MIB MiB free"; return 1
}

last=""
until gate; do
  [ "$GATE_WHY" != "$last" ] && echo "[cv2dp_vast] $arm: WAITING $(ts): $GATE_WHY" | tee -a $Q
  last=$GATE_WHY
  sleep 300
done
echo "[cv2dp_vast] $arm: GATE OPEN $(ts) gpu $GPU" | tee -a $Q

D=$DS/$dsn; C=$RUNS/$arm/train
final=$($PY -c "import json, math; print($EPOCHS * math.ceil(json.load(open('$D/meta/info.json'))['total_frames'] / 64))")
attempt=0
while [ ! -f $RUNS/$arm/trained ]; do
  attempt=$((attempt+1))
  resume=; [ -e $C/checkpoints/last ] && resume=--resume
  echo "[cv2dp_vast] $arm: training gpu $GPU $(ts) final step $final ${resume} attempt $attempt" | tee -a $Q
  CUDA_VISIBLE_DEVICES=$GPU $PY $CODE/train_lerobot.py --dataset_root $D --output_dir $C \
      --epochs $EPOCHS --snapshot_epochs 50 --num_workers $NUM_WORKERS --resize $resume >> $LOG/train_$arm.log 2>&1 &
  echo $! > $LOG/train_$arm.pid; wait $!
  if [ "$($PY -c "import json; print(json.load(open('$C/checkpoints/last/training_state/training_step.json'))['step'])" 2>/dev/null)" = $final ]; then
    touch $RUNS/$arm/trained; echo "[cv2dp_vast] $arm: TRAINED $(ts)" | tee -a $Q; break
  fi
  echo "[cv2dp_vast] $arm: TRAIN FAILED attempt $attempt $(ts)" | tee -a $Q
  [ $attempt -ge 5 ] && { echo "[cv2dp_vast] $arm: GIVING UP after 5 attempts" | tee -a $Q; break; }
  sleep 300
done
