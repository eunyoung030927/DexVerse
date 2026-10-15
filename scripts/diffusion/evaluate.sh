#!/usr/bin/env bash
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Stage 3 of the Diffusion Policy baseline: run the trained checkpoint
# closed-loop and report success rate.
#
#   TASK=Dexverse-GraspKettle-v0 bash scripts/diffusion/evaluate.sh
#   GPU=1 EPISODES=10 TASK=Dexverse-OpenFaucet-v0 bash scripts/diffusion/evaluate.sh
#
# --observation_preset state matches the --obs-groups state used during replay.
# Without it the live env publishes observation groups that were absent from the
# training data and the state vector no longer matches the checkpoint.

set -euo pipefail

# Isaac Sim selects GPUs through NVML, so pin with --device, not CUDA_VISIBLE_DEVICES.
unset CUDA_VISIBLE_DEVICES
GPU=${GPU:-0}
TASK=${TASK:?set TASK, e.g. TASK=Dexverse-GraspKettle-v0}
EPISODES=${EPISODES:-50}

CKPT=runs/dp_${TASK}/best.pt
if [[ ! -f "${CKPT}" ]]; then
    echo "${CKPT} not found. Run: TASK=${TASK} bash scripts/diffusion/train.sh" >&2
    exit 1
fi

echo "[evaluate] task=${TASK} gpu=${GPU} ckpt=${CKPT} episodes=${EPISODES}"

python scripts/diffusion/eval_online.py \
    --task "${TASK}" \
    --ckpt "${CKPT}" \
    --output_dir "runs/dp_${TASK}/eval" \
    --num_episodes "${EPISODES}" \
    --max_steps 1000 \
    --observation_preset state \
    --device "cuda:${GPU}" \
    --headless
