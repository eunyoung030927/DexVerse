"""CPU tests for v1 control overrides without duplicating/mutating shared robots."""

import ast
from copy import deepcopy
from importlib import import_module
import math
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/dexverse/dexverse"
PROFILE = SOURCE / "baseline_v1/control_profile.py"
profile = runpy.run_path(str(PROFILE))


def test_layout_adds_only_limits_and_is_independent():
    layout = {"output_dim": 56, "hands": {side: {"finger_indices": [1, 2], "wrist_rot_order": "yaw_pitch_roll"}
                                         for side in ("left", "right")}}
    before = deepcopy(layout)
    result = profile["action_layout"](layout)
    expected = deepcopy(before)
    for hand in expected["hands"].values():
        hand["wrist_euler_limits"] = ((-2 * math.pi, 2 * math.pi),) * 3
    assert result == expected
    result["hands"]["left"]["finger_indices"].append(3)
    assert layout == before


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("scheme", ["vector", "dexpilot"])
def test_yaml_overrides_only_intended_values(side, scheme):
    path = SOURCE / f"robot_agents/shadow/retarget/{side}_{scheme}.yml"
    original = yaml.safe_load(path.read_text())
    before = deepcopy(original)
    expected = deepcopy(original)
    if scheme == "vector":
        expected["retargeting"].update(low_pass_alpha=0.2, scaling_factor=1.125)
    result = profile["retargeting_config"](original, scheme)
    assert result == expected and original == before
    result["retargeting"]["target_joint_names"] = []
    assert original == before


def _retargeter_methods():
    tree = ast.parse((SOURCE / "devices/retargeters/simple_relative_retargeting.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SimpleRelativeRetargeter")
    names = {"_get_action_layout", "_get_robot_module", "_create_dex_config_with_urdf"}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(Any=object, yaml=yaml, tempfile=tempfile, import_module=import_module,
                     SIMPLE_RETARGETER_LAYOUT_SOURCES={"floating_shadow_bimanual": ("dexverse.robot_agents.shadow.floating", "LAYOUT"),
                                                      "other_robot": ("test_other_robot", "LAYOUT")},
                     HAND_NAME_TO_TARGET={"left": "left", "right": "right"})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<control-profile-retargeter>", "exec"), namespace)
    return type("TestRetargeter", (), {name: namespace[name] for name in names})


@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("robot", ["floating_shadow_bimanual", "other_robot"])
def test_real_retargeter_routes_profile_without_mutating_sources(tmp_path, monkeypatch, version, robot):
    cls = _retargeter_methods()
    shared = SimpleNamespace(__name__="dexverse.robot_agents.shadow.floating", LAYOUT={"output_dim": 56, "hands": {"left": {}, "right": {}}})
    other = SimpleNamespace(__name__="test_other_robot", LAYOUT=deepcopy(shared.LAYOUT))
    monkeypatch.setitem(sys.modules, shared.__name__, shared)
    monkeypatch.setitem(sys.modules, other.__name__, other)
    monkeypatch.setitem(sys.modules, "dexverse.baseline_v1.control_profile", SimpleNamespace(**profile))
    instance = cls()
    instance.cfg = SimpleNamespace(task_version=version, robot_type=robot, retargeting_scheme="vector")
    instance._dex_config_temp_paths = []
    layout = instance._get_action_layout(robot)
    enabled = version == 1 and robot == "floating_shadow_bimanual"
    assert all(("wrist_euler_limits" in hand) == enabled for hand in layout["hands"].values())
    assert shared.LAYOUT["hands"] == {"left": {}, "right": {}}
    path = SOURCE / "robot_agents/shadow/retarget/left_vector.yml"
    before = path.read_bytes()
    try:
        patched = instance._create_dex_config_with_urdf(str(path), "/resolved/robot.urdf")
        values = yaml.safe_load(Path(patched).read_text())["retargeting"]
        assert values["urdf_path"] == "/resolved/robot.urdf"
        assert values["low_pass_alpha"] == (0.2 if enabled else 0.8)
        assert values["scaling_factor"] == (1.125 if enabled else 1.3)
        assert path.read_bytes() == before
    finally:
        for temporary in instance._dex_config_temp_paths:
            Path(temporary).unlink()


def test_shared_setup_is_copied_before_task_overrides():
    tree = ast.parse((SOURCE / "baseline_v1/dexverse_base_env_cfg.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DexVerseBaseEnvCfg")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_configure_robot_from_type")
    shared = SimpleNamespace(robot_config_kwargs={"fingertip_body_names": ["tip"]}, scene_robot={"actuator": {"stiffness": 10}})
    namespace = dict(deepcopy=deepcopy, _get_tabletop_robot_setup_builders=lambda: {"floating_shadow_right": lambda: shared})
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<robot-setup>", "exec"), namespace)
    received = []
    instance = SimpleNamespace(robot_type="floating_shadow_right", _apply_robot_setup=received.append)
    namespace[method.name](instance)
    received[0].scene_robot["actuator"]["stiffness"] = 999
    received[0].robot_config_kwargs["fingertip_body_names"].append("another")
    assert shared.scene_robot["actuator"]["stiffness"] == 10
    assert shared.robot_config_kwargs["fingertip_body_names"] == ["tip"]
