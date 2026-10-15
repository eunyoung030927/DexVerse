#!/usr/bin/env bash
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Stage 2 of the Diffusion Policy baseline: train on the dataset that
# replay.sh produced.
#
#   TASK=Dexverse-GraspKettle-v0 bash scripts/diffusion/train.sh
#   GPU=1 TASK=Dexverse-OpenFaucet-v0 bash scripts/diffusion/train.sh

set -euo pipefail

# Training is pure PyTorch, so CUDA_VISIBLE_DEVICES is the right way to pin a GPU.
export CUDA_VISIBLE_DEVICES=${GPU:-0}
TASK=${TASK:?set TASK, e.g. TASK=Dexverse-GraspKettle-v0}

DATASET=datasets/diffusion/${TASK}.h5
if [[ ! -f "${DATASET}" ]]; then
    echo "${DATASET} not found. Run: TASK=${TASK} bash scripts/diffusion/replay.sh" >&2
    exit 1
fi

echo "[train] task=${TASK} gpu=${CUDA_VISIBLE_DEVICES} dataset=${DATASET}"

python scripts/diffusion/train.py \
    --task "${TASK}" \
    --dataset_file "${DATASET}" \
    --output_dir "runs/dp_${TASK}" \
    --epochs 300 \
    --batch_size 256 \
    --obs_horizon 2 \
    --action_horizon 16 \
    --num_workers 4 \
    --early_stopping_patience 40
