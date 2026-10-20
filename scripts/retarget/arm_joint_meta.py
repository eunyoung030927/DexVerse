# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write <target dir>/arm_joint_meta.json (arm joints + action map, dexverse.retarget.env.arm_joint_meta) for stage-3
results saved before stage 3 stored it in the trajectory, so stage 4 can fill record_demos' ``arm_joint_actions``.

  python scripts/retarget/arm_joint_meta.py --task Dexverse-GraspBleach-v1 --target floating_wuji_right \
      --out <runs>/<task>/demos/ep000/floating_wuji_right [--out ...] --headless
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--out", action="append", required=True, help="target dir(s) to write arm_joint_meta.json into")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

from dexverse.retarget import env as renv_mod  # noqa: E402


def main():
    renv, _ = renv_mod.build(args.task, args.target, 1, args.device)
    meta = renv_mod.arm_joint_meta(renv)
    for d in args.out:
        with open(os.path.join(d, "arm_joint_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
    print(f"RESULT arm_joint_meta {args.task} {args.target}: {meta['arm_joint_names']} -> {len(args.out)} dirs", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
