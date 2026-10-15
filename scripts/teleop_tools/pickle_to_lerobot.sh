#!/bin/bash
# Build LeRobot datasets from recorded trajectory pickles by replaying them headless
# (see scripts/data_tools/pickle_to_lerobot.py):
#
#   scripts/teleop_tools/pickle_to_lerobot.sh /workspace/local/datasets/trajectories
#   scripts/teleop_tools/pickle_to_lerobot.sh <file.pkl> [...] [--dry_run] [--force] [--datasets_dir DIR]
set -eo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
unset PYTHONPATH
exec "${ISAACLAB_PATH:-/workspace/isaaclab}/_isaac_sim/python.sh" "${REPO_ROOT}/scripts/data_tools/pickle_to_lerobot.py" "$@"
