#!/bin/bash
# cv3 training launcher. usage: cv3_launch.sh <id> <memory_fraction|cpu> [cv3_train args...]
ID=$1; MF=$2; shift 2
H=/home/eun/xhand/latent_codec_v1/artifacts/codec_v3
mkdir -p $H/logs
if [ "$MF" = cpu ]; then DEV="--device cpu --threads 16"; else DEV="--memory-fraction $MF"; fi
nohup docker exec -w /workspace/xhand/latent_codec_v1 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=16 xhand \
  /workspace/isaaclab/_isaac_sim/python.sh -m xhand_latent.cv3_train --output artifacts/codec_v3/runs/$ID $DEV "$@" \
  > $H/logs/train_$ID.log 2>&1 &
echo "launched $ID pid $!"
