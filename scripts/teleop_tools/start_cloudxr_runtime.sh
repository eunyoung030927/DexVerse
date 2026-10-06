#!/bin/bash
# Start the CloudXR 6 runtime (bundled with the isaacteleop pip package) for Quest / WebXR hand tracking.
# Keep this terminal open; Ctrl+C stops the runtime. Then, in another terminal, run run_teleop.sh.
#
# Pitfalls handled here (see docs/teleop_quest_cloudxr6.md):
#   * NV_CXR_ENABLE_PUSH_DEVICES=0 selects optical (headset) hand tracking. Without it the runtime picks the
#     "Push Hand Tracker" and every hand joint stays at 0/26.
#   * Isaac Sim setup_conda_env.sh puts websockets 12 and a broken pydantic first on PYTHONPATH, which kills
#     the runtime with "websockets >= 14". PYTHONPATH is cleared before activating the conda env.
#   * Only one CloudXR runtime (WSS port 48322) can run per host, because containers use --network host.
set -eo pipefail

CONDA_ENV=${ISAACTELEOP_CONDA_ENV:-isaacteleop}
ENV_CONFIG=${CXR_ENV_CONFIG:-/root/cxr_optical.env}
WSS_PORT=48322

# The image ships neither ss nor netstat; probe the port with bash's /dev/tcp instead.
if (exec 3<>"/dev/tcp/127.0.0.1/${WSS_PORT}") 2>/dev/null; then
    echo "[cloudxr] port ${WSS_PORT} is already in use on this host (another CloudXR runtime, e.g. in the vls container)." >&2
    echo "[cloudxr] stop it first; only one runtime can run per host." >&2
    exit 1
fi

echo "NV_CXR_ENABLE_PUSH_DEVICES=0" > "${ENV_CONFIG}"

unset PYTHONPATH
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "[cloudxr] starting runtime (env config: ${ENV_CONFIG}). The first run asks you to accept the NVIDIA EULA."
exec python -m isaacteleop.cloudxr --cloudxr-env-config="${ENV_CONFIG}" "$@"
