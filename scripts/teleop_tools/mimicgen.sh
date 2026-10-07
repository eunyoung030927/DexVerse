#!/bin/bash
# MimicGen / DexMimicGen for DexVerse (see scripts/data_tools/mimicgen.py):
#
#   scripts/teleop_tools/mimicgen.sh /workspace/local/datasets/trajectories/functional/Dexverse-GraspCup-v0 --num_demos 30
#
# annotate (replay the source pickles) -> generate --num_demos successful demos -> LeRobot dataset
# <datasets_dir>/<task>-<robot_type>-mimicgen. Run it on the machine the demos were recorded on.
set -eo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
unset PYTHONPATH
exec "${ISAACLAB_PATH:-/workspace/isaaclab}/_isaac_sim/python.sh" "${REPO_ROOT}/scripts/data_tools/mimicgen.py" "$@"
