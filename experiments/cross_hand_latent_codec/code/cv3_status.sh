#!/bin/bash
cd /home/eun/xhand/latent_codec_v1/artifacts/codec_v3
for f in runs/*/train.jsonl; do d=$(dirname $f); s=$(tail -1 $f | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['step'], round(r['wall_s']), round(r['peak_MiB'] or 0), 'cyc',round(r['cyc'],3), 'pair',round(r['pair'],3), 'rec',round(r['rec'],5), 'pc1',round(r['pc1_share'],3), 'pr',round(r['participation_ratio'],2))"); echo "$d $s $( [ -f $d/done.json ] && echo DONE)"; done
grep -l -i "error\|Traceback" logs/train_*.log 2>/dev/null
