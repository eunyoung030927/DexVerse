#!/bin/bash
# CV3DP A3 (261117): run Dw3 (GPU0) and De3 (GPU1) in parallel with cv3dpD_vast_train.sh; when both launchers have exited,
# write /workspace/xhand/runs/cv3dpD/TRAINED_ALL if both arms are trained (else TRAIN_INCOMPLETE). The cv3hm launchers wait
# on TRAINED_ALL before using the vast GPUs.   usage: nohup bash /workspace/xhand/cv3dpD_all.sh > /dev/null 2>&1 &
R=/workspace/xhand/runs/cv3dpD; mkdir -p $R/logs
echo $$ > $R/logs/all.pid
nohup bash /workspace/xhand/cv3dpD_vast_train.sh Dw3 0 > /dev/null 2>&1 &
p1=$!
nohup bash /workspace/xhand/cv3dpD_vast_train.sh De3 1 > /dev/null 2>&1 &
p2=$!
echo "[cv3dpD_all] start $(date -u '+%F %T UTC') launchers Dw3=$p1 De3=$p2" >> $R/logs/vast_queue.log
while kill -0 $p1 2>/dev/null || kill -0 $p2 2>/dev/null; do sleep 120; done
st() { python -c "import json; print(json.load(open('$R/$1/train/checkpoints/last/training_state/training_step.json'))['step'])" 2>/dev/null; }
if [ -f $R/Dw3/trained ] && [ -f $R/De3/trained ]; then
  echo "both trained $(date -u '+%F %T UTC') Dw3 step $(st Dw3) De3 step $(st De3)" > $R/TRAINED_ALL
  echo "[cv3dpD_all] TRAINED_ALL $(date -u '+%F %T UTC')" >> $R/logs/vast_queue.log
else
  echo "launchers exited $(date -u '+%F %T UTC'); Dw3 trained=$([ -f $R/Dw3/trained ] && echo y || echo n) De3 trained=$([ -f $R/De3/trained ] && echo y || echo n)" > $R/TRAIN_INCOMPLETE
  echo "[cv3dpD_all] TRAIN_INCOMPLETE $(date -u '+%F %T UTC')" >> $R/logs/vast_queue.log
fi
