"""CPU regressions for dispenser cup geometry and its task-local scale change."""

from __future__ import annotations

import __future__
import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "source/dexverse/dexverse"
sys.path.insert(0, str(ROOT.parent))
CONFIG = ROOT / "baseline_v1/config/functional/place_cup_under_dispenser_cfg.py"


def _compile(nodes, namespace):
    # Use the actual config expressions/functions without simulator imports.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CONFIG), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)


def _config():
    tree = ast.parse(CONFIG.read_text())
    names = {"CUP_SCALE", "CUP_RADIUS", "CUP_HEIGHT", "CUP_UP_AXIS_LOCAL", "_S45",
             "CUP_REST_ORIENTATIONS", "CUP_REST_ORDER", "STAGE_KEY", "CUP_SPAWN_BUFFER"}
    namespace = {"math": math, "ForbiddenZone": SimpleNamespace}
    _compile([node for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names], namespace)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    _compile([node for node in cls.body if isinstance(node, ast.AnnAssign)], namespace)
    cfg = SimpleNamespace(**{node.target.id: namespace[node.target.id]
                             for node in cls.body if isinstance(node, ast.AnnAssign)})
    post = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    return cfg, namespace, post


def test_scale_reset_heights_and_rim_zone_are_consistent_and_v0_unchanged():
    cfg, values, _ = _config()
    assert cfg.object_scale == (0.81, 0.81, 0.81)
    assert values["CUP_RADIUS"] == pytest.approx(0.034425)
    assert values["CUP_HEIGHT"] == pytest.approx(0.09558)
    orientations = values["CUP_REST_ORIENTATIONS"]
    assert orientations["side"][1] == pytest.approx(values["CUP_RADIUS"] + 0.003)
    assert orientations["upside_down"][1] == pytest.approx(values["CUP_HEIGHT"] + 0.003)
    assert orientations["upright"][1] == 0.003
    rim, = cfg.forbidden_zones
    assert rim.center == pytest.approx((0, 0, 0.11 * 0.81))
    assert rim.radius == pytest.approx(0.035 * 0.81)
    assert rim.half_height == pytest.approx(0.005 * 0.81)
    assert rim.radius < values["CUP_RADIUS"]
    assert rim.center[2] + rim.half_height < values["CUP_HEIGHT"]
    assert cfg.pour_goal_object_local_offset == pytest.approx((0, 0, 0.081))
    for path in ("tasks/config/functional/grasp_cup_cfg.py", "baseline_v1/config/functional/grasp_cup_cfg.py"):
        tree = ast.parse((ROOT / path).read_text())
        fields = {node.target.id: node.value for node in ast.walk(tree) if isinstance(node, ast.AnnAssign)
                  and isinstance(node.target, ast.Name)}
        assert ast.literal_eval(fields["object_scale"]) == (0.9, 0.9, 0.9)
        assert ast.literal_eval(fields["object_mass"]) == 0.2
        assert ast.literal_eval(fields["object_static_friction"]) == 2.5


def _placed_stage():
    cfg, namespace, post = _config()
    cfg.events = SimpleNamespace()
    cfg.forbidden_zone_asset_cfg = lambda: SimpleNamespace(name="robot", body_ids=[0])
    namespace.update(self=cfg, table_top_z=0.6, EventTerm=SimpleNamespace,
                     SceneEntityCfg=lambda name: SimpleNamespace(name=name), SPOT_LOCAL=(0.0, 0.0, 0.012))
    graphs = {}
    namespace["mdp"] = SimpleNamespace(
        reset_root_pose_uniform_orientations=object(), object_axis_upright=object(), object_placed_at_spot=object(),
        StageGraphSpec=SimpleNamespace, StageSpec=SimpleNamespace,
        register_stage_graph=lambda key, graph, **kwargs: graphs.update({key: graph}),
    )
    event = next(node for node in post.body if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]) == "self.events.reset_cup")
    register = next(node for node in post.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "mdp.register_stage_graph")
    _compile([event, register], namespace)
    assert cfg.events.reset_cup.params["orientations"] == list(namespace["CUP_REST_ORIENTATIONS"].values())
    stage = graphs[namespace["STAGE_KEY"]].stages[1]
    assert stage.params["object_radius"] == namespace["CUP_RADIUS"]
    assert stage.params["object_axis_length"] == namespace["CUP_HEIGHT"]
    return stage.params


def test_smaller_release_geometry_keeps_30_step_hold_and_placement_tolerances():
    params = _placed_stage()
    assert params["hold_steps"] == 30
    assert params["xy_tol"] == 0.03 and params["z_tol"] == 0.02
    assert params["max_lin_speed"] == 0.05 and params["release_clearance"] == 0.02
    namespace = {"torch": torch, "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
                 "quat_apply": lambda quat, vector: vector}  # All test orientations are identity.
    tree = ast.parse((ROOT / "baseline_v1/mdp/terminations.py").read_text())
    _compile([node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name in {"_points_to_segment_distance", "object_placed_at_spot"}], namespace)
    evaluate = namespace["object_placed_at_spot"]
    # Hand is beyond the new release radius but within the old one: ensures
    # the stage actually consumes the new cup radius, not a stale 0.9 scale.
    obj = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0., 0., 0.612]]),
                          root_quat_w=torch.tensor([[1., 0., 0., 0.]]), root_lin_vel_w=torch.zeros(1, 3)))
    spot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0., 0., 0.6]]),
                           root_quat_w=obj.data.root_quat_w))
    robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=torch.tensor([[[0.056, 0., 0.65]]])))
    env = SimpleNamespace(scene={"object": obj, "dispenser": spot, "robot": robot}, num_envs=1,
                          device="cpu", episode_length_buf=torch.tensor([1]))
    for step in range(1, 31):
        env.episode_length_buf.fill_(step)
        assert evaluate(env, **params).item() == (step == 30)
    robot.data.body_pos_w[0, 0, 0] = 0.04
    assert not evaluate(env, **params).item()
    assert env._placed_hold_state[params["cache_key"]]["count"].item() == 0


def test_dispenser_revision_requires_review_for_earlier_cup_recordings():
    from dexverse.benchmark import (
        BENCHMARK_REVISION, DISPENSER_SMALL_CUP_REVISION, task_identity, validate_replay_identity,
    )
    task = "Dexverse-GraspCup-v1"
    identity = task_identity(task)
    assert identity["benchmark_revision"] == DISPENSER_SMALL_CUP_REVISION
    validate_replay_identity({"task": task, **identity}, task)
    for revision in (None, BENCHMARK_REVISION):
        payload = {"task": task, "benchmark_revision": revision}
        with pytest.raises(ValueError, match="smaller cup"):
            validate_replay_identity(payload, task)
        validate_replay_identity(payload, task, explicit_override=True)
    baseline = "Dexverse-GraspCup-v0"
    validate_replay_identity({"task": baseline, **task_identity(baseline)}, baseline)
    assert task_identity("Dexverse-GraspBleach-v1")["benchmark_revision"] == BENCHMARK_REVISION
