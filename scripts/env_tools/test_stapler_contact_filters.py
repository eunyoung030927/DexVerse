"""CPU regressions for explicit bimanual stapler-release contact filters."""

from __future__ import annotations

import __future__
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2] / "source/dexverse/dexverse"
CONFIG = ROOT / "baseline_v1/config/articulation/reload_stapler_cfg.py"
ROBOT_USD = ROOT / "robot_agents/shadow/floating_shadow_bimanual/floating_shadow_bimanual.usd"


def _compile(nodes, namespace):
    # Avoid launching Isaac Sim just to inspect the actual config expressions.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CONFIG), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)


def _resolver():
    tree = ast.parse(CONFIG.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "_robot_contact_filter_paths")
    namespace = {}
    _compile([node], namespace)
    return namespace[node.name]


def test_real_bimanual_usd_resolves_each_body_without_joint_or_material_scopes():
    paths = _resolver()(str(ROBOT_USD), "{ENV_REGEX_NS}/Robot")
    stage = Usd.Stage.Open(str(ROBOT_USD))
    root = stage.GetDefaultPrim()
    assert len(root.GetChildren()) == 74  # Reproduces the erroneous wildcard count.
    assert len(paths) == len(set(paths)) == 71
    for side in ("lh", "rh"):
        assert f"{{ENV_REGEX_NS}}/Robot/{side}_palm" in paths
        assert f"{{ENV_REGEX_NS}}/Robot/{side}_ffmiddle" in paths
        assert f"{{ENV_REGEX_NS}}/Robot/{side}_thtip" in paths
    for path in paths:
        relative = path.removeprefix("{ENV_REGEX_NS}/Robot/")
        assert "*" not in relative and "(" not in relative
        assert stage.GetPrimAtPath(root.GetPath().AppendPath(relative)).HasAPI(UsdPhysics.RigidBodyAPI)
    assert not any(path.endswith(("/Looks", "/joints", "/root_joint")) for path in paths)


def test_resolver_handles_nested_and_root_bodies_and_rejects_empty_robot(tmp_path):
    path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()
    resolve = _resolver()
    with pytest.raises(ValueError, match="No robot rigid bodies"):
        resolve(str(path), "{ENV_REGEX_NS}/SelectedRobot")
    UsdPhysics.RigidBodyAPI.Apply(root)
    body = UsdGeom.Xform.Define(stage, "/Root/assembly/finger").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    stage.GetRootLayer().Save()
    assert resolve(str(path), "{ENV_REGEX_NS}/SelectedRobot") == [
        "{ENV_REGEX_NS}/SelectedRobot", "{ENV_REGEX_NS}/SelectedRobot/assembly/finger",
    ]


def test_all_three_release_sensors_are_wired_to_explicit_robot_filters():
    tree = ast.parse(CONFIG.read_text())
    namespace = {"_robot_contact_filter_paths": _resolver(), "ContactSensorCfg": SimpleNamespace}
    _compile([node for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in {"STAPLER_LINKS", "TARGET_TABLE_HEIGHTS", "TARGET_TABLE_PRIM_PATHS"}], namespace)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    post = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    start = next(i for i, node in enumerate(post.body) if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]) == "target_filter_paths")
    end = next(i for i, node in enumerate(post.body) if isinstance(node, ast.For)
               and ast.unparse(node.iter) == "STAPLER_LINKS")
    scene = SimpleNamespace(robot=SimpleNamespace(
        spawn=SimpleNamespace(usd_path=str(ROBOT_USD)), prim_path="{ENV_REGEX_NS}/Robot",
    ))
    namespace["self"] = SimpleNamespace(scene=scene)
    _compile(post.body[start:end + 1], namespace)
    names = namespace["robot_contact_sensor_names"]
    assert len(names) == 3
    for link, name in zip(namespace["STAPLER_LINKS"], names, strict=True):
        sensor = getattr(scene, name)
        assert sensor.prim_path == f"{{ENV_REGEX_NS}}/Articulation/{link}"
        assert sensor.filter_prim_paths_expr == namespace["robot_filter_paths"]
        assert getattr(scene, f"{link}_target_tables_s").filter_prim_paths_expr == list(
            namespace["TARGET_TABLE_PRIM_PATHS"])
    success = next(node for node in post.body if isinstance(node, ast.Assign)
                   and ast.unparse(node.targets[0]) == "self.terminations.success.params")
    params = {ast.literal_eval(key): ast.unparse(value) for key, value in zip(success.value.keys, success.value.values)}
    assert params["robot_sensor_names"] == "robot_contact_sensor_names"
    assert params["robot_force_threshold"] == "self.release_contact_force_threshold"
    assert params["hold_steps"] == "self.success_hold_steps"


def test_release_force_reduction_checks_both_hands_and_every_sensor():
    tree = ast.parse((ROOT / "baseline_v1/mdp/terminations.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "joint_pose_contact_release_hold_success")
    helper = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_sensors_exceed_force")
    helper.decorator_list = []
    namespace = {"torch": torch}
    _compile([helper], namespace)
    evaluate = namespace[helper.name]
    paths = _resolver()(str(ROBOT_USD), "{ENV_REGEX_NS}/Robot")
    sensors = {f"sensor_{i}": SimpleNamespace(data=SimpleNamespace(
        force_matrix_w=torch.zeros(2, 1, len(paths), 3))) for i in range(3)}
    env = SimpleNamespace(num_envs=2, device="cpu", scene=SimpleNamespace(sensors=sensors))
    assert not evaluate(env, list(sensors), 0.1).any()
    for sensor in sensors.values():
        for side in ("lh", "rh"):
            column = paths.index(f"{{ENV_REGEX_NS}}/Robot/{side}_palm")
            sensor.data.force_matrix_w[1, 0, column, 0] = 0.11
            assert evaluate(env, list(sensors), 0.1).tolist() == [False, True]
            sensor.data.force_matrix_w.zero_()
