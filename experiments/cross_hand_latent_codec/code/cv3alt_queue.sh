#!/bin/bash
# cv3alt training queue (261116): runs the given jobs sequentially on this host's GPU (one at a time = GPU budget).
#   usage: cv3alt_queue.sh <memfrac> "<id>|<pairs csv>|<extra args>" ...
# Each job waits until its pair files exist. Completion = runs/<id>/done.json.
set -u
cd /home/eun/xhand/latent_codec_v1
MF=$1; shift
D=artifacts/codec_v3alt
for job in "$@"; do
  IFS='|' read -r ID PAIRS EXTRA <<< "$job"
  [ -f $D/runs/$ID/done.json ] && { echo "$(date +%F_%T) skip $ID" >> $D/logs/queue_$(hostname).log; continue; }
  for p in ${PAIRS//,/ }; do until [ -f $p ]; do sleep 30; done; done
  echo "$(date +%F_%T) start $ID ($PAIRS $EXTRA)" >> $D/logs/queue_$(hostname).log
  docker exec -w /workspace/xhand/latent_codec_v1 xhand /workspace/isaaclab/_isaac_sim/python.sh -m xhand_latent.cv3alt_train \
    --pairs $PAIRS --output $D/runs/$ID --memory-fraction $MF --lean-contacts $EXTRA > $D/logs/train_$ID.log 2>&1
  echo "$(date +%F_%T) end $ID $(cat $D/runs/$ID/done.json 2>/dev/null)" >> $D/logs/queue_$(hostname).log
done
echo "$(date +%F_%T) queue finished" >> $D/logs/queue_$(hostname).log
