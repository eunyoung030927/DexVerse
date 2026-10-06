#!/bin/bash
# Run a DexVerse teleop / recording script against the running CloudXR runtime.
#
#   scripts/teleop_tools/run_teleop.sh teleop_agent --task Dexverse-PickCube-v0 --robot_type floating_allegro_right
#   scripts/teleop_tools/run_teleop.sh record_demos --task Dexverse-PickUpStick-v0 --dataset_dir grasping --num_demos 50
#
# Added unless you pass them yourself: --teleop_device handtracking --enable_pinocchio --headless
# (use --gui to drop --headless and watch the scene over VNC; the AR session still starts automatically),
# plus --xr_stream_log 120 for teleop_agent.
#
# Runs Isaac Sim's bundled python directly: with the conda base env active (the default in interactive
# shells of this image) "isaaclab.sh -p" would pick the conda interpreter instead.
set -eo pipefail

SCRIPT=${1:?usage: run_teleop.sh <teleop_agent|record_demos|path/to/script.py> [args...]}
shift
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
case "${SCRIPT}" in
    teleop_agent|record_demos) SCRIPT_PATH="${REPO_ROOT}/scripts/${SCRIPT}.py" ;;
    *) SCRIPT_PATH="${SCRIPT}" ;;
esac

CXR_ENV=${CXR_ENV:-/root/.cloudxr/run/cloudxr.env}
# cloudxr.env is written before the EULA check, so its presence does not mean the runtime is up:
# also require the runtime's WSS port to be listening (bash /dev/tcp; the image has no ss/netstat).
if [ ! -f "${CXR_ENV}" ] || ! (exec 3<>"/dev/tcp/127.0.0.1/${CXR_WSS_PORT:-48322}") 2>/dev/null; then
    echo "[teleop] CloudXR runtime is not running: start scripts/teleop_tools/start_cloudxr_runtime.sh first." >&2
    exit 1
fi
set -a
source "${CXR_ENV}"
set +a
unset PYTHONPATH

ARGS=("$@")
has_arg() {
    local a
    for a in "${ARGS[@]}"; do
        [[ "$a" == "$1" || "$a" == "$1="* ]] && return 0
    done
    return 1
}
headless=1
if has_arg --gui; then
    headless=0
    filtered=()
    for a in "${ARGS[@]}"; do [[ "$a" != "--gui" ]] && filtered+=("$a"); done
    ARGS=("${filtered[@]}")
fi
has_arg --teleop_device || ARGS+=(--teleop_device handtracking)
has_arg --enable_pinocchio || ARGS+=(--enable_pinocchio)
if [[ ${headless} == 1 ]] && ! has_arg --headless; then ARGS+=(--headless); fi
if [[ "$(basename "${SCRIPT_PATH}")" == "teleop_agent.py" ]] && ! has_arg --xr_stream_log; then
    ARGS+=(--xr_stream_log 120)
fi

PYTHON=${ISAACLAB_PATH:-/workspace/isaaclab}/_isaac_sim/python.sh
cd "${REPO_ROOT}"
echo "[teleop] ${PYTHON} ${SCRIPT_PATH} ${ARGS[*]}"
exec "${PYTHON}" "${SCRIPT_PATH}" "${ARGS[@]}"
