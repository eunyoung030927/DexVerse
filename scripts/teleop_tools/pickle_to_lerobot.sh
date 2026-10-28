#!/bin/bash
# Build LeRobot datasets from recorded trajectory pickles by replaying them headless
# (see scripts/data_tools/pickle_to_lerobot.py):
#
#   scripts/teleop_tools/pickle_to_lerobot.sh /workspace/local/datasets/trajectories
#   scripts/teleop_tools/pickle_to_lerobot.sh <file.pkl> [...] [--dry_run] [--force] [--datasets_dir DIR]
set -eo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
unset PYTHONPATH
# Binary Isaac Sim ships _isaac_sim/python.sh; the pip-installed one (conda env, e.g. envs/dexverse/main) does not:
# there the active interpreter is Isaac Sim's python (pickle_to_lerobot.py falls back the same way for its workers).
PY="${ISAACLAB_PATH:-/workspace/isaaclab}/_isaac_sim/python.sh"
[ -x "${PY}" ] || PY=python
exec "${PY}" "${REPO_ROOT}/scripts/data_tools/pickle_to_lerobot.py" "$@"
