#!/bin/bash
# codec v3 cross-hand Kettle replay for ONE target hand: init settle (arm_init_settle, GraspKettle reference env, same rule as
# 11/11 run_initfix.sh, applied to every condition) then replay_sim with the cohort flags (arm_latent_replay_queue.command).
# usage: cv3_xreplay_run.sh <phase> <hand> <cond1,cond2,...>     (manifest written by cv3_xreplay manifest beforehand)
set -u
cd /home/eun/xhand/latent_codec_v1
PH=$1; H=$2; CC=$3
X=artifacts/codec_v3/xreplay; T=Dexverse-GraspKettle-v0
mkdir -p $X/inputs_initfix/$T $X/$PH/logs $X/settle
P=
for c in ${CC//,/ }; do
  [ -f $X/inputs_initfix/$T/${H}_$c.npz ] && continue
  P=$P${P:+,}$X/inputs/$T/${H}_$c.npz:$X/inputs_initfix/$T/${H}_$c.npz
done
if [ -n "$P" ]; then
  echo "$(date +%F_%T) SETTLE $H $P" >> $X/$PH/logs/driver.log
  flock -s /tmp/isaac_gpu.lock docker exec -w /workspace/xhand/latent_codec_v1 --env OMP_NUM_THREADS=4 xhand /workspace/isaaclab/_isaac_sim/python.sh \
    -m xhand_latent.arm_init_settle --task $T --hand $H --pairs $P --report $X/settle/${PH}_${H}.json \
    --headless --enable_cameras --device cpu > $X/settle/${PH}_${H}.log 2>&1
fi
OUT=$X/$PH/replay/${T}__${H}.jsonl
echo "$(date +%F_%T) REPLAY $H" >> $X/$PH/logs/driver.log
flock -s /tmp/isaac_gpu.lock docker exec -w /workspace/xhand/latent_codec_v1 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
  --env OPENBLAS_NUM_THREADS=4 xhand /workspace/isaaclab/_isaac_sim/python.sh -m xhand_latent.replay_sim \
  --task $T --hand $H --sources sharpa --decoded artifacts/decoded/all_five_v1 --data data/replay \
  --diagnostic-manifest $X/$PH/manifests/${T}__${H}.json --output $OUT --episodes 10 --episode-indices 0,1,2,3,4,5,6,7,8,9 \
  --native-episodes 0 --native-reference $X/native_reference --scene-trace --no-auto-reset \
  --resolve-scene-entities --headless --enable_cameras --device cpu > $X/$PH/logs/replay_${H}.log 2>&1
echo "$(date +%F_%T) DONE $H rows=$(grep -c . $OUT 2>/dev/null)" >> $X/$PH/logs/driver.log
