#!/bin/bash
# CV2DP (261115, work2): archive copy of every finished cv2dp DP snapshot (pretrained_model, >= 3 min old) from vast
# runs/cv2dp/<arm>/train/checkpoints/<step> -> work2 artifacts/cv2dp/ckpt_archive/<arm>/<step> (tar stream, sha256 of
# every file compared, .complete). Every 10 min; exits when ctl/STOP_PULL exists. Key ~/.ssh/ho_vast_ed25519.
# usage: nohup bash cv2dp_pull_w2.sh >> artifacts/cv2dp/logs/pull_w2.log 2>&1 &
CV=/home/eun/xhand/latent_codec_v1/artifacts/cv2dp
R=/workspace/xhand/runs/cv2dp
VAST="ssh -i $HOME/.ssh/ho_vast_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -p <VAST_PORT> root@<VAST_HOST>"
echo $$ > $CV/logs/pull_w2.pid
while [ ! -f $CV/ctl/STOP_PULL ]; do
  list=$($VAST "for f in $R/*/train/checkpoints/[0-9]*/pretrained_model/model.safetensors; do d=\$(dirname \$f); test -f \$d/train_config.json || continue; a=\$(( \$(date +%s) - \$(stat -c %Y \$f) )); [ \$a -ge 180 ] && echo \$f; done" 2>/dev/null)
  for f in $list; do
    arm=$(echo $f | awk -F/ '{print $(NF-5)}'); step=$(echo $f | awk -F/ '{print $(NF-2)}')
    d=$CV/ckpt_archive/$arm/$step
    [ -f $d/.complete ] && continue
    rm -rf $d.tmp; mkdir -p $d.tmp
    t0=$(date +%s)
    $VAST "tar -C $R/$arm/train/checkpoints/$step -cf - pretrained_model" 2>/dev/null | tar -xf - -C $d.tmp || { echo "$(date '+%F %T') $arm $step tar failed"; continue; }
    rs=$($VAST "cd $R/$arm/train/checkpoints/$step/pretrained_model && sha256sum * | sort" 2>/dev/null)
    ls_=$(cd $d.tmp/pretrained_model && sha256sum * | sort)
    if [ -n "$rs" ] && [ "$rs" = "$ls_" ]; then
      rm -rf $d; mv $d.tmp $d; echo "pulled $(date '+%F %T') sha256 ok" > $d/.complete
      echo "$(date '+%F %T') $arm $step OK $(( $(date +%s) - t0 ))s"
    else
      echo "$(date '+%F %T') $arm $step sha256 mismatch"; rm -rf $d.tmp
    fi
  done
  sleep 600
done
echo "$(date '+%F %T') STOP_PULL -> exit"
