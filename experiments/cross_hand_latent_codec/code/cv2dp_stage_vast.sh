#!/bin/bash
# CV2DP (261115, work2): stage a cv2dp LeRobot set to vast (/workspace/xhand/datasets/cv2dp/<name>), sha256 of every file
# compared, then .cv2dp_staged written on vast (training start gate). Key ~/.ssh/ho_vast_ed25519 (work2's own).
# usage: bash cv2dp_stage_vast.sh <name> [cv2dp|handmask]   (log: artifacts/cv2dp/logs/stage_vast.log)
n=$1; SUB=${2:-cv2dp}  # dataset group dir: cv2dp, or handmask for the masked Zv2m copy
SRC=/home/eun/xhand/datasets/$SUB/$n
DST=/workspace/xhand/datasets/$SUB
VAST="ssh -i $HOME/.ssh/ho_vast_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -p <VAST_PORT> root@<VAST_HOST>"
t0=$(date +%s)
$VAST "mkdir -p $DST && rm -rf $DST/$n $DST/.$n.tmp && mkdir -p $DST/.$n.tmp" 2>/dev/null
tar -C $SRC -cf - . | $VAST "tar -xf - -C $DST/.$n.tmp" 2>/dev/null || { echo "$n tar FAILED"; exit 1; }
l=$(cd $SRC && find . -type f ! -name .cv2dp_staged -print0 | sort -z | xargs -0 sha256sum | sha256sum)
r=$($VAST "cd $DST/.$n.tmp && find . -type f ! -name .cv2dp_staged -print0 | sort -z | xargs -0 sha256sum | sha256sum" 2>/dev/null)
nf=$(cd $SRC && find . -type f | wc -l)
if [ "$l" = "$r" ] && [ -n "$r" ]; then
  $VAST "mv $DST/.$n.tmp $DST/$n && echo 'staged $(date '+%F %T') from work2 sha256 ok' > $DST/$n/.cv2dp_staged" 2>/dev/null     || { echo "$n STAGE final mv FAILED (vast ssh); data left in $DST/.$n.tmp"; exit 1; }
  echo "$n STAGE_OK $(( $(date +%s) - t0 ))s $nf files $(du -sh $SRC | cut -f1)"
else
  echo "$n STAGE sha256 MISMATCH"; exit 1
fi
