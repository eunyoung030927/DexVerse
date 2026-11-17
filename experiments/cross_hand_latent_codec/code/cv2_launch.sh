#!/bin/bash
# cv2 training launcher (work1). usage: cv2_launch.sh <id> <lam_cyc> <lam_pair> [extra cv2_train args]
ID=$1; LC=$2; LP=$3
H=/home/eun/xhand/latent_codec_v1/artifacts/codec_v2
mkdir -p $H/logs
nohup docker exec -w /workspace/xhand/latent_codec_v1 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 xhand \
  /workspace/isaaclab/_isaac_sim/python.sh -m xhand_latent.cv2_train --output artifacts/codec_v2/runs/$ID \
  --lam-cyc $LC --lam-pair $LP --memory-fraction 0.12 ${@:4} > $H/logs/train_$ID.log 2>&1 &
echo "launched $ID pid $!"
