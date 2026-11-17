#!/bin/bash
# cv3alt cross-hand retarget replay (261116): Sharpa eps 0-9 -> target hand, Kettle, conditions prepared by cv3alt_prepare.
# Same tools/flags as the arm_latent_v1 cohort (p3_leapfix): arm_init_settle (unmodified) on every target's initial rows,
# then unmodified replay_sim with a diagnostic manifest (scripts/arm_latent_replay_queue.py command line).
#   usage: cv3alt_xreplay.sh <tag> <cond1,cond2,...> [targets=allegro,leap,shadow,wuji] [parallel=1]
set -u
TAG=$1; CONDS=$2; TARGETS=${3:-allegro,leap,shadow,wuji}; P=${4:-1}
cd /home/eun/xhand/latent_codec_v1
B=artifacts/codec_v3alt/replay; T=Dexverse-GraspKettle-v0
mkdir -p $B/inputs_initfix/$T $B/manifests $B/out $B/logs $B/settle $B/native_reference
export B T TAG CONDS
run_one() {
  h=$1
  out=$B/out/${T}__${h}__${TAG}.jsonl
  if [ -f ${out%.jsonl}.done.json ]; then echo "$(date +%F_%T) SKIP done $TAG $h" >> $B/logs/xreplay_driver.log; return 0; fi
  P=
  for c in ${CONDS//,/ }; do [ -f $B/inputs_initfix/$T/${h}_$c.npz ] || P=$P${P:+,}$B/inputs/$T/${h}_$c.npz:$B/inputs_initfix/$T/${h}_$c.npz; done
  echo "$(date +%F_%T) SETTLE $TAG $h ${P:+todo}" >> $B/logs/xreplay_driver.log
  [ -n "$P" ] && flock -s /tmp/isaac_gpu.lock docker exec -w /workspace/xhand/latent_codec_v1 --env OMP_NUM_THREADS=4 xhand /workspace/isaaclab/_isaac_sim/python.sh \
    -m xhand_latent.arm_init_settle --task $T --hand $h --pairs $P --report $B/settle/${TAG}_${h}.json \
    --headless --enable_cameras --device cpu > $B/logs/settle_${TAG}_${h}.log 2>&1
  python3 - "$h" <<'EOF'
import json, os, sys
h = sys.argv[1]; B, T, TAG, CONDS = (os.environ[k] for k in ("B", "T", "TAG", "CONDS"))
spec = json.load(open(f"metadata/hands/{h}.json"))
jobs = []
for c in CONDS.split(","):
    f = f"{B}/inputs_initfix/{T}/{h}_{c}.npz"
    if not os.path.exists(f):
        raise SystemExit(f"missing settled input {f}")
    jobs.append({"condition": f"arm_{c}", "file": f, "finger_reference": f, "finger_mode": "target_decoded", "target_hand": h,
                 "role": c, "EE_definition": "pad-surface correspondence wrist (arm_latent_v1)",
                 "initial_state_protocol": "Source Sharpa scene; target root from target episode; codec-mapped robot initial state"})
json.dump({"protocol_id": f"cv3alt_xreplay_{TAG}", "task": T, "source": "sharpa", "replay_hand": h, "run": "cv3alt",
           "wrist_joints": spec["wrist_joints"], "jobs": jobs}, open(f"{B}/manifests/{T}__{h}__{TAG}.json", "w"), indent=2)
EOF
  echo "$(date +%F_%T) REPLAY $TAG $h" >> $B/logs/xreplay_driver.log
  flock -s /tmp/isaac_gpu.lock docker exec -w /workspace/xhand/latent_codec_v1 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    xhand /workspace/isaaclab/_isaac_sim/python.sh -m xhand_latent.replay_sim \
    --task $T --hand $h --sources sharpa --decoded artifacts/decoded/all_five_v1 --data data/replay \
    --diagnostic-manifest $B/manifests/${T}__${h}__${TAG}.json --output $out --episodes 10 \
    --episode-indices 0,1,2,3,4,5,6,7,8,9 --native-episodes 0 --native-reference $B/native_reference \
    --scene-trace --no-auto-reset --resolve-scene-entities --headless --enable_cameras --device cpu > $B/logs/replay_${TAG}_${h}.log 2>&1
  echo "$(date +%F_%T) END $TAG $h rows=$(grep -c . $out 2>/dev/null)" >> $B/logs/xreplay_driver.log
}
export -f run_one
for h in ${TARGETS//,/ }; do echo $h; done | xargs -P $P -L 1 bash -c 'run_one $0'
echo "$(date +%F_%T) driver finished $TAG" >> $B/logs/xreplay_driver.log
