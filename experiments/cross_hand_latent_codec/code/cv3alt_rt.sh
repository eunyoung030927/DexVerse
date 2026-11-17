#!/bin/bash
# cv3alt native round-trip replay (261116): copy of cv2_replay.sh (P0b flags) with the output root moved to artifacts/codec_v3alt/rt.
#   usage: cv3alt_rt.sh <tag> <checkpoint> [hands=sharpa,shadow,wuji] [parallel=1]
# Step 1 (CPU): p3h_roundtrip --arms Z --checkpoint <ckpt> --tag <tag> --out $B ; native reference rows copied from codec_v2 (P0b).
set -u
TAG=$1; CK=$2; HANDS=${3:-sharpa,shadow,wuji}; P=${4:-1}
cd /home/eun/xhand/latent_codec_v1
B=artifacts/codec_v3alt/rt; T=Dexverse-GraspKettle-v0; ARM=Z_$TAG
mkdir -p $B/out $B/logs $B/native_reference
cp -n artifacts/codec_v2/replay/native_reference/* $B/native_reference/ 2>/dev/null
docker exec -w /workspace/xhand/latent_codec_v1 --env CUDA_VISIBLE_DEVICES= xhand /workspace/isaaclab/_isaac_sim/python.sh \
  -m xhand_latent.p3h_roundtrip --arms Z --checkpoint $CK --tag $TAG --out $B --hands $HANDS > $B/logs/prep_${TAG}.log 2>&1
run_one() {
  h=$1
  out=$B/out/${T}__${h}__${ARM}.jsonl
  for attempt in 1 2; do
    if [ -f ${out%.jsonl}.done.json ] && [ "$(grep -c . $out)" -ge 20 ]; then
      echo "$(date +%F_%T) DONE $ARM $h" >> $B/logs/rt_driver.log; return 0; fi
    echo "$(date +%F_%T) START $ARM $h attempt $attempt" >> $B/logs/rt_driver.log
    flock -s /tmp/isaac_gpu.lock docker exec -w /workspace/xhand/latent_codec_v1 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
      --env OPENBLAS_NUM_THREADS=4 xhand /workspace/isaaclab/_isaac_sim/python.sh -m xhand_latent.replay_sim \
      --task $T --hand $h --sources $h --decoded $B/decoded/$ARM --data data/replay --output $out \
      --episodes 10 --native-reference $B/native_reference --no-auto-reset --resolve-scene-entities \
      --headless --enable_cameras --device cpu > $B/logs/replay_${h}_${ARM}.log.$attempt 2>&1
  done
  if [ -f ${out%.jsonl}.done.json ] && [ "$(grep -c . $out)" -ge 20 ]; then
    echo "$(date +%F_%T) DONE $ARM $h" >> $B/logs/rt_driver.log
  else echo "$(date +%F_%T) FAIL $ARM $h" >> $B/logs/rt_driver.log; fi
}
export -f run_one; export B T ARM
for h in ${HANDS//,/ }; do echo "$h"; done | xargs -P $P -L 1 bash -c 'run_one $0'
echo "$(date +%F_%T) rt driver finished $ARM" >> $B/logs/rt_driver.log
