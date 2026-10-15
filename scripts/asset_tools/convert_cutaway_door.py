"""Convert the source-only cutaway door URDF using Isaac Sim's importer.

Run: python scripts/run_local.py scripts/asset_tools/convert_cutaway_door.py --headless
No downloads. Output is an ignored generated asset; original door is untouched.
"""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "source/dexverse/dexverse/assets/core_assets/dexverse_authored/cutaway_door"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=OUTPUT)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

try:
    cfg = UrdfConverterCfg(
        asset_path=str(ROOT / "scripts/asset_tools/urdf/cutaway_door.urdf"),
        usd_dir=str(args.output_dir.resolve()),
        usd_file_name="cutaway_door.usd",
        fix_base=True,
        merge_fixed_joints=False,
        self_collision=True,
        make_instanceable=False,
        collision_from_visuals=False,
        collider_type="convex_hull",  # Separate panel boxes and one convex bevel; never hull the whole panel.
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={"door_hinge": 0.20, "latch_slide": 12.0},
                damping={"door_hinge": 0.12, "latch_slide": 1.0},
            ),
        ),
    )
    converter = UrdfConverter(cfg)
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(converter.usd_path)
    joint = stage.GetPrimAtPath("/cutaway_door/joints/door_hinge")
    # URDF has no portable spring preload; preserve it in the standalone USD
    # as well as in the task's reset-time targets. USD angles use degrees.
    UsdPhysics.DriveAPI.Get(joint, "angular").GetTargetPositionAttr().Set(math.degrees(-0.20))
    stage.GetRootLayer().Save()
    print(f"CUTAWAY_DOOR_USD: {converter.usd_path}", flush=True)
finally:
    app.close()
