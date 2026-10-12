# One-time environment setup, sourced by main_job.sbatch after env.sh. Every step is idempotent and stamped,
# so after the first run (pip + assets; plus ~30 min for the SIF if Bench2Dex has not built it) it takes seconds.

echo "=== [setup 1/4] Isaac Sim image -> $SIF ==="
if [ -s "$HOME/.ngc_key" ]; then        # nvcr.io/nvidia/isaac-sim is public; a key is only a fallback
  export APPTAINER_DOCKER_USERNAME='$oauthtoken'
  export APPTAINER_DOCKER_PASSWORD="$(tr -d '[:space:]' < "$HOME/.ngc_key")"
fi
(
  flock 9
  if [ -s "$SIF" ]; then
    echo "SIF present ($(du -h "$SIF" | cut -f1))"
  else
    echo "building SIF from $ISAAC_IMAGE (one-off, ~20-40 min) ..."
    apptainer build "$SIF.partial" "$ISAAC_IMAGE" && mv "$SIF.partial" "$SIF"
  fi
) 9>"$SIF.lock"
[ -s "$SIF" ] || { echo "FATAL: SIF build failed"; exit 1; }
# The build leaves ~15 GB of OCI layers in this node's /tmp, never needed again.
if [ -d "$APPTAINER_CACHEDIR" ] && [ "$(du -s "$APPTAINER_CACHEDIR" | cut -f1)" -gt 2000000 ]; then
  echo "dropping $(du -sh "$APPTAINER_CACHEDIR" | cut -f1) of node-local OCI cache"
  rm -rf "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
  mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
fi

echo "=== [setup 2/4] IsaacLab $ISAACLAB_TAG ==="
[ -d "$WS/IsaacLab/.git" ] || GIT clone -q --branch "$ISAACLAB_TAG" --depth 1 https://github.com/isaac-sim/IsaacLab.git "$WS/IsaacLab"

echo "=== [setup 3/4] python env -> $WS/.pyuser ==="
STAMP="$WS/.pyuser/.dexverse_setup_v1.ok"
if [ -f "$STAMP" ]; then
  echo "already set up ($(cat "$STAMP"))"
else
  "${ISAAC[@]}" bash -c '
    set -euo pipefail
    P=/isaac-sim/python.sh
    L=/workspace/dexverse/IsaacLab/source
    $P -c "import site; assert site.ENABLE_USER_SITE, \"user site disabled\""
    printf "numpy==1.26.4\ntorch==2.7.0\ntorchvision==0.22.0\npillow==11.3.0\n" > /workspace/dexverse/.pyuser/constraints.txt
    # Blackwell (sm_120) needs a CUDA 12.8 build; a 2.7.0+cu126 torch imports fine but has no sm_120 kernels.
    $P -c "import torch, sys; sys.exit(0 if torch.__version__.startswith(\"2.7\") and torch.version.cuda == \"12.8\" else 1)" \
      || $P -m pip install --user torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    # Isaac Lab v2.3.2 pins flatdict==4.0.1, whose setup.py needs pkg_resources (IsaacLab issue #4576).
    $P -m pip install --user "setuptools==65.0.0"
    $P -m pip install --user --no-build-isolation "flatdict==4.0.1"
    # DexVerse as the dev box installs it (isaaclab.sh --install + the DexVerse README): Isaac Lab packages,
    # the dexverse package from this checkout, and what its tasks and the retargeting import.
    $P -m pip install --user -c /workspace/dexverse/.pyuser/constraints.txt \
      -e $L/isaaclab -e $L/isaaclab_assets -e $L/isaaclab_tasks -e $L/isaaclab_rl \
      -e /workspace/dexverse/DexVerse/source/dexverse \
      toml psutil h5py PyYAML scipy trimesh matplotlib "opencv-python<4.12" imageio imageio-ffmpeg huggingface_hub
    $P -c "import isaaclab, dexverse, dex_retargeting, trimesh, scipy; print(\"imports ok\")"
  ' || { echo "FATAL: pip setup failed"; exit 1; }
  echo "isaacsim-5.1.0 isaaclab-$ISAACLAB_TAG $(date +%F)" > "$STAMP"
fi

echo "=== [setup 4/4] DexVerse robot + object assets (into the checkout, git-ignored) ==="
# Retargeting does not render: the six hand bundles, core_assets (task objects/scenes) and the ManiTwin object pool
# (grasping tasks); no HDRIs (per-reset backgrounds are off in the baselines) and no long-horizon meshes.
ASTAMP="$REPO_DIR/source/dexverse/dexverse/assets/.dexverse_assets_v1.ok"
if [ -f "$ASTAMP" ]; then
  echo "assets present ($(cat "$ASTAMP"))"
else
  "${ISAAC[@]}" bash -c '
    set -euo pipefail
    P=/isaac-sim/python.sh
    $P scripts/asset_tools/download_robot_agents.py --all
    $P scripts/asset_tools/download_assets.py --core --mani-twin
  ' || { echo "FATAL: asset download failed"; exit 1; }
  date +%F > "$ASTAMP"
fi
