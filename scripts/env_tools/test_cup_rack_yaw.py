"""CPU regressions for the v1 rack's hand-relative yaw and cup reset transform."""

from __future__ import annotations

import __future__
import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2] / "source/dexverse/dexverse/tasks"
BASELINE_V1 = ROOT.parent / "baseline_v1"
CONFIG = "config/functional/remove_cup_from_rack_cfg.py"
sys.path.insert(0, str(ROOT.parents[1]))


def _compile(nodes, namespace):
    # Importing task modules would launch simulator dependencies. Compile the
    # actual reset functions/defaults and event assignment, as other CPU tests do.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<rack-test>", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)


def _defaults(version="v1"):
    path = BASELINE_V1 / CONFIG if version == "v1" else ROOT / CONFIG
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {"math": math, "FLOATING_SHADOW_RIGHT_WRIST_POSITION_OFFSET": (-0.25, 0.0, 0.8)}
    names = {"cup_holder_reset_x_range", "cup_holder_reset_y_range", "cup_holder_reset_yaw_range",
             "cup_holder_face_point", "cup_offset_from_holder", "object_scale", "cup_holder_scale", "object_mass",
             "target_x_range", "target_y_range", "object_half_height", "place_clearance",
             "success_position_threshold", "success_max_tilt_rad", "destination_table_size",
             "cup_holder_init_x_offset", "cup_holder_init_y_offset"}
    _compile([node for node in cls.body if isinstance(node, ast.AnnAssign) and node.target.id in names], namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names if name in namespace}), cls


def _event():
    cfg, cls = _defaults()
    cfg.events = SimpleNamespace()
    post = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    event = next(node for node in post.body if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]) == "self.events.reset_cup_holder")
    _compile([event], {"self": cfg, "mdp": SimpleNamespace(reset_root_pose_uniform_facing=_reset),
                       "EventTerm": SimpleNamespace, "SceneEntityCfg": lambda name: SimpleNamespace(name=name)})
    return cfg.events.reset_cup_holder.params


def _quat_from_euler_xyz(roll, pitch, yaw):
    values = torch.stack((roll, pitch, yaw), dim=-1).numpy()
    return torch.tensor(Rotation.from_euler("xyz", values).as_quat(scalar_first=True), dtype=yaw.dtype)


def _quat_apply(quat, vector):
    return torch.tensor(Rotation.from_quat(quat.numpy(), scalar_first=True).apply(vector.numpy()), dtype=vector.dtype)


def _quat_mul(left, right):
    result = Rotation.from_quat(left.numpy(), scalar_first=True) * Rotation.from_quat(right.numpy(), scalar_first=True)
    return torch.tensor(result.as_quat(scalar_first=True), dtype=left.dtype)


_namespace = {"torch": torch, "math": math, "quat_from_euler_xyz": _quat_from_euler_xyz,
              "quat_apply": _quat_apply, "quat_mul": _quat_mul,
              "SceneEntityCfg": lambda name: SimpleNamespace(name=name)}
for _file, _names in (("mdp/utils.py", {"resolve_env_ids"}),
                       ("mdp/resets.py", {"reset_root_pose_uniform_facing", "sync_object"})):
    _tree = ast.parse((BASELINE_V1 / _file).read_text())
    _compile([node for node in _tree.body if isinstance(node, ast.FunctionDef) and node.name in _names], _namespace)
_reset = _namespace["reset_root_pose_uniform_facing"]


class _Asset:
    def __init__(self, count):
        state = torch.zeros(count, 13)
        state[:, :3] = torch.tensor([0.0, 0.2, 0.5125])
        state[:, 3] = 1.0
        self.data = SimpleNamespace(default_root_state=state, root_pos_w=state[:, :3].clone(),
                                    root_quat_w=state[:, 3:7].clone())
        self.cfg = SimpleNamespace(spawn=SimpleNamespace(rigid_props=SimpleNamespace(kinematic_enabled=True)))

    def write_root_pose_to_sim(self, pose, env_ids):
        self.data.root_pos_w[env_ids] = pose[:, :3]
        self.data.root_quat_w[env_ids] = pose[:, 3:7]

    def write_root_velocity_to_sim(self, *args, **kwargs):
        pytest.fail("Kinematic rack must not receive velocity writes")


def _env(count):
    class Scene(dict):
        pass

    scene = Scene(cup_holder=_Asset(count), object=_Asset(count))
    scene.env_origins = torch.arange(count).unsqueeze(-1) * torch.tensor([[2.0, -3.0, 0.0]])
    return SimpleNamespace(scene=scene, num_envs=count, device="cpu")


def _relative_yaw(env):
    rack = env.scene["cup_holder"].data
    front = _quat_apply(rack.root_quat_w, torch.tensor([0.16, -0.01, 0.0]).expand(env.num_envs, -1))
    target = torch.tensor([-0.25, 0.0]) + env.scene.env_origins[:, :2]
    to_hand = target - rack.root_pos_w[:, :2]
    delta = torch.atan2(front[:, 1], front[:, 0]) - torch.atan2(to_hand[:, 1], to_hand[:, 0])
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def test_only_v1_uses_wide_hand_relative_yaw():
    cfg, _ = _defaults()
    baseline, _ = _defaults("v0")
    assert cfg.cup_holder_reset_yaw_range == pytest.approx(tuple(map(math.radians, (-115, 115))))
    assert baseline.cup_holder_reset_yaw_range == (-0.5, 0.5)
    params = _event()
    assert params["face_asset_cfg"] is None
    assert params["face_point"] == (-0.25, 0.0)
    assert params["open_dir_local"] == cfg.cup_offset_from_holder[:2]
    assert params["yaw_jitter"] == cfg.cup_holder_reset_yaw_range


@pytest.mark.parametrize("degrees", [-115, 0, 115])
def test_heading_and_cup_seating_at_center_and_endpoints(degrees):
    env = _env(8)
    angle = math.radians(degrees)
    params = _event() | {"yaw_jitter": (angle, angle)}
    _reset(env, None, **params)
    torch.testing.assert_close(_relative_yaw(env), torch.full((8,), angle), atol=2e-6, rtol=0)
    # Cup orientation must stay rack-local even at the widest yaw endpoints.
    cfg, _ = _defaults()
    local_rot = Rotation.from_euler("z", -90, degrees=True).inv() * Rotation.from_euler("x", -105, degrees=True)
    _namespace["sync_object"](
        env, None, target_cfg=SimpleNamespace(name="object"), source_cfg=SimpleNamespace(name="cup_holder"),
        source_local_offset=cfg.cup_offset_from_holder, quat_local=tuple(local_rot.as_quat(scalar_first=True)),
    )
    rack, cup = env.scene["cup_holder"].data, env.scene["object"].data
    inverse = rack.root_quat_w.clone()
    inverse[:, 1:] *= -1
    torch.testing.assert_close(_quat_apply(inverse, cup.root_pos_w - rack.root_pos_w),
                               torch.tensor(cfg.cup_offset_from_holder).expand(8, -1), atol=2e-6, rtol=0)
    actual = _quat_mul(inverse, cup.root_quat_w)
    expected = torch.tensor(local_rot.as_quat(scalar_first=True), dtype=actual.dtype).expand(8, -1)
    assert torch.all(torch.abs((actual * expected).sum(-1)) > 1 - 1e-6)


def test_seeded_distribution_covers_arc_and_subset_reset_leaves_others_untouched():
    env = _env(2048)
    # Avoid large float32 origins masking small heading differences in the histogram.
    env.scene.env_origins.zero_()
    torch.manual_seed(1003)
    _reset(env, None, **_event())
    rack = env.scene["cup_holder"].data
    positions, rotations = rack.root_pos_w.clone(), rack.root_quat_w.clone()
    degrees = torch.rad2deg(_relative_yaw(env))
    assert degrees.min() >= -115.001 and degrees.max() <= 115.001
    assert degrees.min() < -114 and degrees.max() > 114
    assert torch.all(torch.histc(degrees, bins=10, min=-115, max=115) > 140)
    torch.manual_seed(4200)
    _reset(env, torch.tensor([1, 3]), **_event())
    torch.testing.assert_close(rack.root_pos_w[[0, 2]], positions[[0, 2]])
    torch.testing.assert_close(rack.root_quat_w[[0, 2]], rotations[[0, 2]])
    torch.manual_seed(1003)
    _reset(env, None, **_event())
    torch.testing.assert_close(rack.root_pos_w, positions)
    torch.testing.assert_close(rack.root_quat_w, rotations)


def test_small_destination_geometry_and_goal(tmp_path):
    from dexverse.assets.small_table_builder import ensure_small_table_asset
    from pxr import Usd, UsdGeom, UsdPhysics

    cfg, cls = _defaults()
    post = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    assert "self._configure_destination_table(table_pos=table_pos, table_top_z=table_top_z)" in ast.unparse(post)
    helper = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_configure_destination_table")
    class RigidCfg(SimpleNamespace):
        InitialStateCfg = SimpleNamespace

    cfg.scene = SimpleNamespace()
    cfg.commands = SimpleNamespace(object_pose=SimpleNamespace(ranges=SimpleNamespace()))
    namespace = {
        "ensure_small_table_asset": lambda name, **params: ensure_small_table_asset(name, out_dir=tmp_path, **params),
        "RigidObjectCfg": RigidCfg,
        "sim_utils": SimpleNamespace(UsdFileCfg=SimpleNamespace, RigidBodyPropertiesCfg=SimpleNamespace),
        "dexverse_base_env": SimpleNamespace(spawn_usd_with_rigid_properties=object()),
    }
    _compile([helper], namespace)
    namespace["_configure_destination_table"](cfg, table_pos=(0, 0, 0.575), table_top_z=0.6)
    stand = cfg.scene.cup_destination_table
    assert stand.init_state.pos == pytest.approx((0.1, -0.325, 0.6))
    assert stand.spawn.rigid_props.kinematic_enabled
    assert stand.spawn.collision_props is None
    assert cfg.destination_table_size == (0.15, 0.15, 0.15)
    stage = Usd.Stage.Open(stand.spawn.usd_path)
    cubes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Cube)]
    assert len(cubes) == 5
    assert all(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() for prim in cubes)
    top = next(prim for prim in cubes if prim.GetName() == "top")
    assert tuple(top.GetAttribute("xformOp:scale").Get()) == pytest.approx((0.15, 0.15, 0.012))
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        stage.GetDefaultPrim()).ComputeAlignedRange()
    assert tuple(bounds.GetSize()) == pytest.approx((0.15, 0.15, 0.15))
    assert cfg.commands.object_pose.ranges.pos_z == pytest.approx((0.775, 0.775))
    baseline, _ = _defaults("v0")
    assert baseline.target_x_range == (0.05, 0.15)
    assert baseline.target_y_range == (-0.4, -0.25)
    assert not hasattr(baseline, "destination_table_size")
    assert cfg.success_position_threshold == baseline.success_position_threshold
    assert cfg.success_max_tilt_rad == baseline.success_max_tilt_rad


def test_cup_shrinks_without_changing_rack_or_v0():
    cfg, _ = _defaults()
    baseline, _ = _defaults("v0")
    assert cfg.object_scale == (0.9, 0.9, 0.9)
    assert baseline.object_scale == (1.0, 1.0, 1.0)
    assert cfg.cup_holder_scale == baseline.cup_holder_scale == (1.5, 1.5, 1.5)
    assert cfg.cup_offset_from_holder == baseline.cup_offset_from_holder
    assert cfg.object_mass == baseline.object_mass == 0.2
    # The shared object builder sends scale to the USD spawner, which scales
    # both rendered geometry and authored colliders (not just a visual marker).
    base_tree = ast.parse((BASELINE_V1 / "config/functional/base_cfg.py").read_text())
    builder = next(node for node in base_tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "build_object_cfg_from_usd")
    spawn_call = next(node for node in ast.walk(builder) if isinstance(node, ast.Call)
                      and ast.unparse(node.func) == "sim_utils.UsdFileCfg")
    assert any(kw.arg == "scale" and ast.unparse(kw.value) == "scale" for kw in spawn_call.keywords)


def test_main_table_legs_and_bounds_remain_original_size():
    cfg, cls = _defaults()
    tree = ast.parse((BASELINE_V1 / "dexverse_base_env_cfg.py").read_text())
    constants = {"DEFAULT_TABLE_THICKNESS", "DEFAULT_TABLE_TOP_HEIGHT", "DEFAULT_TABLE_SIZE",
                 "DEFAULT_TABLE_INIT_POS", "DEFAULT_TABLE_INIT_ROT", "DEFAULT_TABLE_LEG_THICKNESS",
                 "DEFAULT_TABLE_LEG_INSET"}
    namespace = {}
    _compile([node for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants], namespace)
    _compile([node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name in {"_compute_table_leg_size_and_pos", "_sync_table_legs_to_table"}], namespace)
    cfg.scene = SimpleNamespace(table=SimpleNamespace(
        spawn=SimpleNamespace(size=namespace["DEFAULT_TABLE_SIZE"]),
        init_state=SimpleNamespace(pos=namespace["DEFAULT_TABLE_INIT_POS"], rot=namespace["DEFAULT_TABLE_INIT_ROT"]),
    ))
    signs = {"front_left": (1, 1), "front_right": (1, -1), "back_left": (-1, 1), "back_right": (-1, -1)}
    for name in signs:
        setattr(cfg.scene, f"table_leg_{name}", SimpleNamespace(spawn=SimpleNamespace(), init_state=SimpleNamespace()))
    post = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    super_index = next(i for i, node in enumerate(post.body) if ast.unparse(node) == "super().__post_init__()")
    _compile(post.body[:super_index], {"self": cfg})
    assert cfg.scene.table.spawn.size == (1.5, 1.5, 0.05)
    assert namespace["DEFAULT_TABLE_SIZE"] == (1.5, 1.5, 0.05)
    assert "self.scene.table.spawn.size =" not in ast.unparse(cls)
    assert cfg.scene.table.init_state.pos == pytest.approx((0, 0, 0.575))
    assert "_sync_table_legs_to_table(self.scene)" in ast.unparse(tree)
    namespace["_sync_table_legs_to_table"](cfg.scene)
    for name, (sx, sy) in signs.items():
        leg = getattr(cfg.scene, f"table_leg_{name}")
        assert leg.spawn.size == pytest.approx((0.08, 0.08, 0.55))
        assert leg.init_state.pos == pytest.approx((sx * 0.63, sy * 0.63, 0.275))
        assert leg.init_state.rot == cfg.scene.table.init_state.rot

    # Execute the inherited footprint-bound setup against the original scene.
    base_tree = ast.parse((BASELINE_V1 / "config/functional/base_cfg.py").read_text())
    base_cls = next(node for node in base_tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "FunctionalGraspingEnvCfg")
    base_post = next(node for node in base_cls.body if isinstance(node, ast.FunctionDef)
                     and node.name == "__post_init__")
    bounds = next(node for node in base_post.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "self.terminations.object_out_of_bound is not None")
    cfg.terminations = SimpleNamespace(object_out_of_bound=SimpleNamespace(params={}))
    _compile([bounds], {"self": cfg, "BOUND_Z_MIN": -0.2, "BOUND_Z_MAX": 1.5})
    assert cfg.terminations.object_out_of_bound.params["in_bound_range"] == {
        "x": (-0.75, 0.75), "y": (-0.75, 0.75), "z": (-0.2, 1.5),
    }


def test_main_table_fits_rack_and_small_destination_fits_goal_cup():
    cfg, _ = _defaults()
    # Conservative asset bounds (metres before spawn scaling), rounded up
    # from the USDs. Radial envelopes cover every yaw, not just sampled ones.
    rack_radius = math.hypot(0.084 * cfg.cup_holder_scale[0], 0.189 * cfg.cup_holder_scale[1])
    cup_radius = math.sqrt(0.044**2 + 0.043**2 + 0.119**2) * max(cfg.object_scale)
    seated_radius = math.hypot(*cfg.cup_offset_from_holder[:2]) + cup_radius
    for axis in range(2):
        initial = (cfg.cup_holder_init_x_offset, cfg.cup_holder_init_y_offset)[axis]
        reset = (cfg.cup_holder_reset_x_range, cfg.cup_holder_reset_y_range)[axis]
        max_center = max(abs(initial + offset) for offset in reset)
        assert max_center + max(rack_radius, seated_radius) < 0.75
        goals = (cfg.target_x_range, cfg.target_y_range)[axis]
        stand_half = cfg.destination_table_size[axis] / 2
        assert abs(sum(goals) / 2) + stand_half < 0.75
        # An upright cup fits at every sampled target, at any cup yaw.
        upright_radius = math.hypot(0.044, 0.043) * max(cfg.object_scale)
        assert (goals[1] - goals[0]) / 2 + upright_radius < stand_half


def test_small_destination_revision_rejects_previous_scene_recordings():
    from dexverse.benchmark import (
        BENCHMARK_REVISION, CUP_RACK_SMALL_DESTINATION_REVISION, task_identity, validate_replay_identity,
    )

    task = "Dexverse-RemoveCupFromRack-v1"
    identity = task_identity(task)
    assert identity["benchmark_revision"] == CUP_RACK_SMALL_DESTINATION_REVISION
    validate_replay_identity({"task": task, **identity}, task)
    for old_revision in (None, BENCHMARK_REVISION, "cup-rack-raised-destination-v1-2026-09-09",
                         "cup-rack-small-cup-v1-2026-09-09", "cup-rack-compact-table-v1-2026-09-09"):
        payload = {"task": task, "benchmark_revision": old_revision}
        with pytest.raises(ValueError, match="15 x 15 cm raised destination"):
            validate_replay_identity(payload, task)
        validate_replay_identity(payload, task, explicit_override=True)
    assert task_identity("Dexverse-RemoveCupFromRack-v0")["task_version"] == 0
    assert task_identity("Dexverse-GraspBleach-v1")["benchmark_revision"] == BENCHMARK_REVISION
