#!/bin/bash
# work2: v2-comparable cv2_measure + cv3_measure for runs (artifacts/codec_v3/runs/<id>/best.pt). usage: cv3_measure_runs.sh id1 id2 ...
cd /home/eun/xhand/latent_codec_v1
M=artifacts/codec_v3/measure; mkdir -p $M/cv2
TAX=artifacts/codec_v3/data/tax_pairs.npz; TA=; [ -f $TAX ] && TA="--tax $TAX"
C=; for id in "$@"; do C="$C --codec $id=artifacts/codec_v3/runs/$id/best.pt"; done
PY="docker exec -w /workspace/xhand/latent_codec_v1 --env CUDA_VISIBLE_DEVICES= --env PYTHONDONTWRITEBYTECODE=1 xhand nice -n 10 /workspace/isaaclab/_isaac_sim/python.sh"
$PY -m xhand_latent.cv2_measure $C --out $M/cv2 > artifacts/codec_v3/logs/measure_cv2_$1.log 2>&1
$PY -m xhand_latent.cv3_measure $C $TA > artifacts/codec_v3/logs/measure_cv3_$1.log 2>&1
docker exec xhand chown -R $(id -u):$(id -g) /workspace/xhand/latent_codec_v1/artifacts/codec_v3
python3 xhand_latent/cv3_report.py > $M/gates.md
echo "measured $*"
