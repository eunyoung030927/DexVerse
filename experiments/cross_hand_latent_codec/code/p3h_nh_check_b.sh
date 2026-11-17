#!/bin/bash
# P3NH check (b) (copy of p3h_p3_check_b.sh, 261114): nohot J dataset action rows -> P3HNoHotCodec.decode_action inside the
# real eval loop (p3h_nh_eval_pooled.py --replay_npz), pkl eps 0-2, J x 3 hands. Rows: p3nh/check/p3h_nh_dsrows.npz
# (p3h_nh_check_codec). SHARED lock on /tmp/isaac_gpu.lock. usage: p3h_nh_check_b.sh [P=3]
P=${1:-3}
B=/workspace/xhand/latent_codec_v1/artifacts/pooled3hand; H=/home/eun/xhand/latent_codec_v1/artifacts/pooled3hand
mkdir -p $H/p3nh/check/b
one() {
  arm=$1; h=$2
  flock -s /tmp/isaac_gpu.lock docker exec xhand bash -c "export DEXVERSE_ROOT=/workspace/DexVerse OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Seoul; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS DEXVERSE_OBJECT_MATERIAL DEXVERSE_DUMMY_LINK_MASS; cd /workspace/DexVerse; /workspace/isaaclab/_isaac_sim/python.sh /workspace/xhand/latent_codec_v1/xhand_latent/p3h_nh_eval_pooled.py --arm $arm --demos $B/p0b/demos/Dexverse-GraspKettle-v0_floating_${h}_right.pkl --replay_npz $B/p3nh/check/p3h_nh_dsrows.npz --output_dir $B/p3nh/check/b/${arm}_${h} --episodes 0,1,2 --max_steps 3000 --no-video --headless" > $H/p3nh/check/b/${arm}_${h}.log 2>&1
  echo "$(date +%F_%T) $arm $h $(grep -c 'success=yes' $H/p3nh/check/b/${arm}_${h}.log)/3" >> $H/p3nh/check/b/summary.txt
}
export -f one; export B H
for h in sharpa shadow wuji; do echo "J $h"; done | xargs -P $P -L 1 bash -c 'one $0 $1'
echo "$(date +%F_%T) CHECK_B DONE" >> $H/p3nh/check/b/summary.txt
