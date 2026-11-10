#!/bin/bash
# Wrapper: the Manus + Quest teleop session script lives in the cross_dex package (cross_dex/dexverse/adapters/
# dexverse_session.sh, installed with `pip install --no-deps -e /workspace/cross_dex`). Same arguments:
#   scripts/teleop_tools/dexverse_session.sh record --task Dexverse-GraspCup-v1 --robot_type floating_shadow_right --num_demos 50
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DEXVERSE_ROOT=${DEXVERSE_ROOT:-$(cd "${HERE}/../.." && pwd)}
PKG=$(python -c "import cross_dex, pathlib; print(pathlib.Path(cross_dex.__file__).resolve().parent)" 2>/dev/null) \
  || { echo "[session] cross_dex is not installed in this environment: pip install --no-deps -e /workspace/cross_dex" >&2; exit 1; }
exec bash "${PKG}/dexverse/adapters/dexverse_session.sh" "$@"
