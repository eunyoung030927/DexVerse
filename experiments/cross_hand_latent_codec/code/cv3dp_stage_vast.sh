#!/bin/bash
# CV3DP (261116, work2): stage a cv3dp LeRobot set to vast /workspace/xhand/datasets/cv3dp/<name>. Unlike cv2dp_stage_vast.sh
# (unchanged) only data/ + meta/ (~22 MB) are sent; the mp4s are hard-linked ON VAST from the identical files of the already
# staged sets (pooled3hand_nohot/kettle3_Z, ho2hand/sa_Z) per meta/p3h_pooled.json. Then the sha256 of EVERY file
# (videos included) is compared work2 vs vast, and .cv2dp_staged is written on vast (training start gate).
# Key ~/.ssh/ho_vast_ed25519 (work2's own).   usage: bash cv3dp_stage_vast.sh <name>   (log: codec_v3/cv3dp/logs/stage_vast.log)
n=$1
SRC=/home/eun/xhand/datasets/cv3dp/$n
DST=/workspace/xhand/datasets/cv3dp
VAST="ssh -i $HOME/.ssh/ho_vast_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -p <VAST_PORT> root@<VAST_HOST>"
t0=$(date +%s)
links=$(python3 - "$SRC" <<'EOF'
import json, sys
root = sys.argv[1]
m = json.load(open(f"{root}/meta/p3h_pooled.json"))
for e in m["episodes"]:
    for vk in ("observation.images.third_person", "observation.images.eye_in_hand"):
        print(f"ln {e['source_dataset']}/videos/{vk}/chunk-000/file-{e['source_episode_index']:03d}.mp4 "
              f"videos/{vk}/chunk-000/file-{e['episode_index']:03d}.mp4 || exit 1")
EOF
)
[ -n "$links" ] || { echo "$n link list empty"; exit 1; }
$VAST "mkdir -p $DST && rm -rf $DST/$n $DST/.$n.tmp && mkdir -p $DST/.$n.tmp" 2>/dev/null
tar -C $SRC -cf - data meta | $VAST "tar -xf - -C $DST/.$n.tmp" 2>/dev/null || { echo "$n tar FAILED"; exit 1; }
echo "$links" | $VAST "cd $DST/.$n.tmp && mkdir -p videos/observation.images.third_person/chunk-000 videos/observation.images.eye_in_hand/chunk-000 && bash -e" 2>/dev/null || { echo "$n vast hardlinks FAILED"; exit 1; }
l=$(cd $SRC && find . -type f ! -name .cv2dp_staged -print0 | sort -z | xargs -0 sha256sum | sha256sum)
r=$($VAST "cd $DST/.$n.tmp && find . -type f ! -name .cv2dp_staged -print0 | sort -z | xargs -0 sha256sum | sha256sum" 2>/dev/null)
nf=$(cd $SRC && find . -type f | wc -l)
if [ "$l" = "$r" ] && [ -n "$r" ]; then
  $VAST "mv $DST/.$n.tmp $DST/$n && echo 'staged $(date '+%F %T') from work2 (data/meta sent, videos hard-linked on vast) sha256 ok' > $DST/$n/.cv2dp_staged" 2>/dev/null \
    || { echo "$n STAGE final mv FAILED (vast ssh); data left in $DST/.$n.tmp"; exit 1; }
  echo "$n STAGE_OK $(( $(date +%s) - t0 ))s $nf files"
else
  echo "$n STAGE sha256 MISMATCH"; exit 1
fi
