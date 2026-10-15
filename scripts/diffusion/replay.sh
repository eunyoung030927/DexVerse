#!/usr/bin/env bash
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Stage 1 of the Diffusion Policy baseline: replay a demonstration pickle in
# simulation and convert the result into a training dataset.
#
#   TASK=Dexverse-GraspKettle-v0 bash scripts/diffusion/replay.sh
#   GPU=1 TASK=Dexverse-OpenFaucet-v0 bash scripts/diffusion/replay.sh
#   PKL=path/to/demos.pkl TASK=Dexverse-PushT-v0 bash scripts/diffusion/replay.sh
#
# Replay steps the recorded actions through physics instead of restoring the
# recorded scene states. That matters: set-state replay produces observations no
# closed-loop policy will ever see, and a policy trained on them drifts
# immediately at evaluation time.

set -euo pipefail

# Isaac Sim selects GPUs through NVML, so pin with --device, not CUDA_VISIBLE_DEVICES.
unset CUDA_VISIBLE_DEVICES
GPU=${GPU:-0}
TASK=${TASK:?set TASK, e.g. TASK=Dexverse-GraspKettle-v0}

DEMO_ROOT=source/dexverse/demonstrations
REPLAY_DIR=outputs/replay/${TASK}
DATASET=datasets/diffusion/${TASK}.h5

# Baseline demos live at <demo-root>/<category>/<task>/<file>.pkl
# (scripts/demo_tools/download_demos.py --baseline).
if [[ -z "${PKL:-}" ]]; then
    PKL=$(ls "${DEMO_ROOT}"/*/"${TASK}"/*.pkl 2>/dev/null | head -n1 || true)
fi
if [[ ! -f "${PKL:-}" ]]; then
    echo "No demonstration pickle found at ${DEMO_ROOT}/*/${TASK}/*.pkl." >&2
    echo "Fetch it with: python scripts/demo_tools/download_demos.py --baseline" >&2
    echo "Or point at one explicitly with PKL=<path>." >&2
    exit 1
fi

# The replay script mirrors the pickle's path relative to --demos-root under
# --output-dir, and names the file after the observation preset.
REPLAY_H5="${REPLAY_DIR}/$(realpath --relative-to="${DEMO_ROOT}" "$(dirname "${PKL}")")/$(basename "${PKL}" .pkl).state.seq.demo.h5"

echo "[replay] task=${TASK} gpu=${GPU}"
echo "[replay]   pickle  = ${PKL}"
echo "[replay]   replay  = ${REPLAY_H5}"
echo "[replay]   dataset = ${DATASET}"

python scripts/demo_tools/create_demo_files_sequential.py \
    --file "${PKL}" \
    --obs-groups state \
    --output-dir "${REPLAY_DIR}" \
    --device "cuda:${GPU}" \
    --overwrite \
    --headless

# simulation_app.close() can os._exit(0) and swallow a traceback, so `set -e`
# alone does not catch a failed replay. Check that the writer produced output.
if [[ ! -f "${REPLAY_H5}" ]]; then
    echo "Replay exited without writing ${REPLAY_H5} -- Isaac Sim most likely failed during" >&2
    echo "scene construction and the traceback was swallowed by app shutdown." >&2
    exit 1
fi

python scripts/diffusion/build_dataset.py --in_file "${REPLAY_H5}" --out_file "${DATASET}"
echo "[replay] dataset ready: ${DATASET}"
