# Shared environment for every DexVerse ondemand (cluster) job. Sourced, never submitted (no #SBATCH).
#
# Layout: $HOME/dexverse is bound to /workspace/dexverse inside the Isaac Sim container, the same tree as the dev box.
#
#   $HOME/dexverse/
#     DexVerse/            the fork eunyoung030927/DexVerse, branch retarget/shadow-cross-hand (pulled by every job);
#                          this pipeline is its ondemand/ folder, the retargeting its scripts/retarget/
#     IsaacLab/            v2.3.2
#     demos/demonstrations/v1/<category>/<task>/demos.pkl     source demos (dexverse/DexVerse_release)
#     runs/retarget/       outputs (DEXVERSE_RETARGET_RUNS)    .pyuser/  Isaac Lab + DexVerse + deps
#   Isaac Sim 5.1 SIF: the Bench2Dex one ($HOME/b2d/sif) if it exists, else $HOME/dexverse/sif (same image).

export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"

WS="$HOME/dexverse"
REPO_DIR="$WS/DexVerse"
CODE_DIR="$REPO_DIR/ondemand"
REPO_BRANCH="${REPO_BRANCH:-retarget/shadow-cross-hand}"
if [ -z "${SIF:-}" ]; then
  SIF="$WS/sif/isaac-sim_5.1.0.sif"
  [ -s "$HOME/b2d/sif/isaac-sim_5.1.0.sif" ] && SIF="$HOME/b2d/sif/isaac-sim_5.1.0.sif"
fi
ISAAC_IMAGE="docker://nvcr.io/nvidia/isaac-sim:5.1.0"
GIT_IMAGE="docker://hyeonseoke/lerobot:v1"      # carries git; the bare compute node may not
ISAACLAB_TAG="v2.3.2"
RUNS="$WS/runs/retarget"
mkdir -p "$WS/sif" "$WS/.pyuser" "$WS/.cache/pip" "$WS/.cache/huggingface" "$RUNS"

# Node-local scratch: apptainer layers and the container $HOME (Kit's shader/texture caches).
NODE="/tmp/dexverse-$USER"
NODE_HOME="$NODE/home"
export APPTAINER_CACHEDIR="$NODE/apptainer-cache"
export APPTAINER_TMPDIR="$NODE/apptainer-tmp"
mkdir -p "$NODE_HOME" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Headless Vulkan through EGL: the GLX ICD fails without a display.
cat > "$NODE_HOME/nvidia_icd.json" <<'EOF'
{ "file_format_version" : "1.0.1", "ICD": { "library_path": "libEGL_nvidia.so.0", "api_version" : "1.4.312" } }
EOF
cat > "$NODE_HOME/10_nvidia.json" <<'EOF'
{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }
EOF

export APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES
export APPTAINERENV_ACCEPT_EULA=Y
export APPTAINERENV_PRIVACY_CONSENT=Y
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONUSERBASE=/workspace/dexverse/.pyuser
export APPTAINERENV_PIP_CACHE_DIR=/workspace/dexverse/.cache/pip
export APPTAINERENV_HF_HOME=/workspace/dexverse/.cache/huggingface
export APPTAINERENV_VK_ICD_FILENAMES="$NODE_HOME/nvidia_icd.json"
export APPTAINERENV___EGL_VENDOR_LIBRARY_FILENAMES="$NODE_HOME/10_nvidia.json"
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINERENV_DEXVERSE_RETARGET_RUNS=/workspace/dexverse/runs/retarget
export APPTAINERENV_HF_HUB_DISABLE_PROGRESS_BARS=1         # no tqdm bars in SLURM logs
if [ -s "$HOME/.hf_token" ]; then
  export APPTAINERENV_HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")"
fi

GPU_FLAG=(--nv)
if [ "${NVCCLI:-0}" = "1" ]; then
  export NVIDIA_DRIVER_CAPABILITIES=all
  GPU_FLAG=(--nv --nvccli)
fi
# An array, not a function, so `timeout` can run it. The SIF is bound read-only next to the workspace.
ISAAC=(apptainer exec "${GPU_FLAG[@]}" --writable-tmpfs
       --home "$NODE_HOME" --bind "$WS:/workspace/dexverse" --pwd /workspace/dexverse/DexVerse "$SIF")
PY=/isaac-sim/python.sh
DEMOS_ROOT=/workspace/dexverse/demos/demonstrations/v1      # container path
GIT() { apptainer exec --bind "$WS" "$GIT_IMAGE" git "$@"; }

# Fast-forward the checkout of REPO_BRANCH; flock so concurrent jobs do not race on .git.
sync_repo() {
  flock "$REPO_DIR/.git" apptainer exec --bind "$WS" "$GIT_IMAGE" git -C "$REPO_DIR" pull --ff-only -q origin "$REPO_BRANCH" \
    || { echo "FATAL: git pull of $REPO_DIR failed (network, or the checkout diverged). Fix it by hand."; return 1; }
  echo "$REPO_DIR @ $(GIT -C "$REPO_DIR" rev-parse --short HEAD) ($REPO_BRANCH)"
}
