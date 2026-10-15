"""Check the 100 original registrations and 20 isolated v1 configurations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/dexverse"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, default=Path("outputs/task_versions/pairs.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import dexverse.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402
from dexverse.benchmark import BASELINE_PAIRS, BASELINE_V1_TASKS, LEGACY_UPGRADE_TASKS, task_identity  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from dexverse.tasks.config.floating_teleop import apply_teleop_retargeter_mode  # noqa: E402
from dexverse.devices.retargeters.simple_relative_retargeting import SimpleRelativeRetargeter  # noqa: E402
from dexverse.robot_agents.shadow import floating as shadow  # noqa: E402
from dexverse.baseline_v1.control_profile import SHADOW_WRIST_LIMITS  # noqa: E402


def main():
    shared_layouts = {name: deepcopy(value) for name, value in vars(shadow).items()
                      if name.endswith("_SIMPLE_RELATIVE_RETARGETER_LAYOUT")}
    specs = {s.id: s for s in gym.registry.values() if s.id.startswith("Dexverse-")}
    assert len(specs) == 120
    assert {task for task in specs if task.endswith("-v1")} == set(BASELINE_V1_TASKS)
    assert len([task for task in specs if task.endswith("-v0")]) == 100
    assert not (set(LEGACY_UPGRADE_TASKS) & specs.keys())
    for task, spec in specs.items():
        assert spec.kwargs["env_cfg_entry_point"].startswith("dexverse.baseline_v1.") == task.endswith("-v1"), task
    report = {"registrations": 120, "pairs": {}}
    for old, new in BASELINE_PAIRS.items():
        configs = [parse_env_cfg(task, device=args.device, num_envs=1) for task in (old, new)]
        a, b = configs
        assert type(a) is not type(b)
        for version, cfg in enumerate(configs):
            assert cfg.is_finite_horizon is True
            wrist_event = getattr(cfg.events, "set_shadow_wrist_joint_limits", None)
            assert (wrist_event is not None) == bool(version)
            if version:
                assert (wrist_event.params["lower"], wrist_event.params["upper"]) == SHADOW_WRIST_LIMITS
            for device in cfg.teleop_devices.devices.values():
                for retargeter in device.retargeters:
                    assert retargeter.task_version == version
                    instance = SimpleRelativeRetargeter.__new__(SimpleRelativeRetargeter)
                    instance.cfg = retargeter
                    layout = instance._get_action_layout(cfg.robot_type)
                    assert all(("wrist_euler_limits" in hand) == bool(version) for hand in layout["hands"].values())
                    assert instance._get_robot_module() is shadow
                    for side, hand in instance._get_dex_retargeting_spec()["hands"].items():
                        assert "robot_agents/shadow/retarget/" in hand["config_paths"]["vector"]
                        for scheme, path in hand["config_paths"].items():
                            retargeter.retargeting_scheme = scheme
                            instance._dex_config_temp_paths = []
                            try:
                                patched = instance._create_dex_config_with_urdf(path, hand["urdf_path"])
                                original = yaml.safe_load(Path(path).read_text())
                                expected = deepcopy(original)
                                expected["retargeting"]["urdf_path"] = hand["urdf_path"]
                                if version and scheme == "vector":
                                    expected["retargeting"].update(low_pass_alpha=0.2, scaling_factor=1.125)
                                assert yaml.safe_load(Path(patched).read_text()) == expected
                                assert yaml.safe_load(Path(path).read_text()) == original
                            finally:
                                for temp in instance._dex_config_temp_paths:
                                    Path(temp).unlink()
            apply_teleop_retargeter_mode(cfg.teleop_devices, "absolute")
            for device in cfg.teleop_devices.devices.values():
                assert all(r.task_version == version for r in device.retargeters)
        if "OpenLaptop" in old:
            assert not hasattr(a.terminations, "base_displaced")
            assert b.terminations.base_displaced.params["max_upward_z"] == 0.005
        if "GraspCup" in old:
            assert not hasattr(a.scene, "dispenser") and hasattr(b.scene, "dispenser")
        if "BimanualLiftTray" in old:
            assert not hasattr(a.scene, "rack") and hasattr(b.scene, "rack")
        report["pairs"][old] = {
            "v1": new,
            "v0_identity": task_identity(old),
            "v1_identity": task_identity(new),
            "status": "passed",
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    assert all(getattr(shadow, name) == value for name, value in shared_layouts.items())
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("VERSION_CHECK: 100 original v0 IDs, 20 isolated v1 pairs; robot/retargeter settings passed.", flush=True)


try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
