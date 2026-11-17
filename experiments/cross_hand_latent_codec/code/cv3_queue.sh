#!/bin/bash
# cv3 training queue: cv3_queue.sh <slots> <memfrac> <queue_file>; queue lines: "<id> <cv3_train args...>"; lines starting # skipped.
# A line is launched when fewer than <slots> of this host's cv3_train processes run. Re-reads the file each loop (append-able).
SL=$1; MF=$2; Q=$3
cd /home/eun/xhand/latent_codec_v1
while true; do
  n=$(pgrep -af "xhand_latent.cv3_train --output" | grep -v -- "--device cpu" | grep -c python3)  # GPU runs only (one kit python3 per run)
  nxt=$(grep -v '^#' $Q | while read id rest; do [ -n "$id" ] && [ ! -f artifacts/codec_v3/runs/$id/config.json ] && echo "$id $rest" && break; done)
  if [ -z "$nxt" ]; then [ $n -eq 0 ] && break; sleep 60; continue; fi
  if [ $n -lt $SL ]; then
    set -- $nxt; id=$1; shift
    bash xhand_latent/cv3_launch.sh $id $MF "$@"; echo "$(date +%T) launched $id" ; sleep 90
  else sleep 60; fi
done
echo "$(date +%T) queue empty"
