#!/bin/bash
# Start the CloudXR 6 runtime (bundled with the isaacteleop pip package) for Quest / WebXR hand tracking.
# Keep this terminal open; Ctrl+C stops the runtime. Then, in another terminal, run run_teleop.sh.
#
# Pitfalls handled here (see docs/teleop_quest_cloudxr6.md):
#   * NV_CXR_ENABLE_PUSH_DEVICES=0 selects optical (headset) hand tracking. Without it the runtime picks the
#     "Push Hand Tracker" and every hand joint stays at 0/26.
#   * Isaac Sim setup_conda_env.sh puts websockets 12 and a broken pydantic first on PYTHONPATH, which kills
#     the runtime with "websockets >= 14". PYTHONPATH is cleared before activating the conda env.
#   * Only one CloudXR runtime (WSS port 48322) can run per network namespace (per host with --network host).
#   * The media stream goes to the address the runtime advertises, chosen among adapters that have a MAC, so
#     tailscale0 is never picked. With --network host that is the host's public IP (fine). In a container with
#     its own network namespace (n1) it would be the unreachable docker bridge IP and the headset disconnects
#     ~15 s after signing in, so there the Tailscale IP is advertised via NV_CXR_ENDPOINT_IP. Override with
#     CXR_ENDPOINT_IP=<ip>.
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

ENDPOINT_IP=${CXR_ENDPOINT_IP:-}
if [ -z "${ENDPOINT_IP}" ] && command -v tailscale >/dev/null 2>&1; then
    ts_ip=$(tailscale ip -4 2>/dev/null | head -n 1 || true)
    # Globally routable IPv4 on a non-Tailscale adapter (present with --network host).
    public_ip=$(ip -4 -o addr show scope global 2>/dev/null | awk '$2 != "tailscale0" {split($4, a, "/"); print a[1]}' \
        | grep -vE '^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)' || true)
    if [ -n "${ts_ip}" ] && [ -z "${public_ip}" ]; then
        ENDPOINT_IP=${ts_ip}
    fi
fi
if [ -n "${ENDPOINT_IP}" ]; then
    echo "NV_CXR_ENDPOINT_IP=${ENDPOINT_IP}" >> "${ENV_CONFIG}"
    echo "[cloudxr] media endpoint advertised to the headset: ${ENDPOINT_IP}"
fi

unset PYTHONPATH
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "[cloudxr] starting runtime (env config: ${ENV_CONFIG}). The first run asks you to accept the NVIDIA EULA."
exec python -m isaacteleop.cloudxr --cloudxr-env-config="${ENV_CONFIG}" "$@"
