#!/bin/bash
# cv2 fidelity replay (261115): copy of p3h_replay.sh (P0b) with the output root moved to artifacts/codec_v2/replay.
# Inputs from: python.sh -m xhand_latent.p3h_roundtrip --arms Z --checkpoint <ckpt> --tag <tag> --out artifacts/codec_v2/replay
#   -> decoded/Z_<tag>/<task>/<hand>/<hand>_{native,codec}.npz ; native rows reused from native_reference (copied from p0b).
# Unmodified xhand_latent.replay_sim, same flags as P0b. Completion = done.json + >=20 rows (exit codes not trusted).
#   usage: cv2_replay.sh <tag> [hands=sharpa,shadow,wuji] [parallel=3]
set -u
TAG=$1; HANDS=${2:-sharpa,shadow,wuji}; P=${3:-3}
cd /home/eun/xhand/latent_codec_v1
B=artifacts/codec_v2/replay; T=Dexverse-GraspKettle-v0; ARM=Z_$TAG
mkdir -p $B/out $B/logs $B/native_reference
run_one() {
  h=$1
  out=$B/out/${T}__${h}__${ARM}.jsonl
  for attempt in 1 2; do
    if [ -f ${out%.jsonl}.done.json ] && [ "$(grep -c . $out)" -ge 20 ]; then
      echo "$(date +%F_%T) DONE $ARM $h" >> $B/logs/replay_driver.log; return 0; fi
    echo "$(date +%F_%T) START $ARM $h attempt $attempt" >> $B/logs/replay_driver.log
    flock -s /tmp/isaac_gpu.lock docker exec -w /workspace/xhand/latent_codec_v1 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
      --env OPENBLAS_NUM_THREADS=4 xhand /workspace/isaaclab/_isaac_sim/python.sh -m xhand_latent.replay_sim \
      --task $T --hand $h --sources $h --decoded $B/decoded/$ARM --data data/replay --output $out \
      --episodes 10 --native-reference $B/native_reference --no-auto-reset --resolve-scene-entities \
      --headless --enable_cameras --device cpu > $B/logs/replay_${h}_${ARM}.log.$attempt 2>&1
  done
  if [ -f ${out%.jsonl}.done.json ] && [ "$(grep -c . $out)" -ge 20 ]; then
    echo "$(date +%F_%T) DONE $ARM $h" >> $B/logs/replay_driver.log
  else echo "$(date +%F_%T) FAIL $ARM $h" >> $B/logs/replay_driver.log; fi
}
export -f run_one; export B T ARM
for h in ${HANDS//,/ }; do echo "$h"; done | xargs -P $P -L 1 bash -c 'run_one $0'
echo "$(date +%F_%T) replay driver finished $ARM" >> $B/logs/replay_driver.log
