"""CPU checks for faucet v1 reach and whole-assembly tabletop clearance."""

from __future__ import annotations

import __future__
import ast
import itertools
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2] / "source/dexverse/dexverse"
sys.path.insert(0, str(ROOT.parent))
CONFIG = ROOT / "baseline_v1/config/articulation/openfaucet_cfg.py"


def _compile(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CONFIG), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)


def _defaults():
    tree = ast.parse(CONFIG.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    names = {"articulation_init_pos", "articulation_init_rot", "articulation_scale",
             "articulation_half_height_est", "articulation_reset_pose_range", "sink_scale",
             "sink_init_rot", "sink_offset_from_faucet", "support_footprint", "success_threshold"}
    namespace = {"math": math, "R": R}
    _compile([node for node in cls.body if isinstance(node, ast.AnnAssign) and node.target.id in names], namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def test_faucet_is_closer_without_changing_yaw_size_or_success():
    cfg = _defaults()
    assert cfg.articulation_init_pos == (0.15, 0, 0)
    ranges = cfg.articulation_reset_pose_range
    assert np.array(ranges["x"]) + cfg.articulation_init_pos[0] == pytest.approx((0.1, 0.2))
    assert ranges["y"] == [-0.1, 0.1]
    assert ranges["yaw"] == [-3.14159, 3.14159]
    assert all(ranges[axis] == [0, 0] for axis in ("z", "roll", "pitch"))
    assert cfg.articulation_scale == (0.75, 0.75, 0.75)
    assert cfg.sink_scale == (0.83, 0.83, 0.83)
    assert cfg.articulation_half_height_est == 0.1
    assert cfg.success_threshold == math.radians(80)
    # Entire new root range is closer than the closest old root position.
    assert math.hypot(0.2 - (-0.25), 0.1) < 0.3 - (-0.25)


def test_full_assembly_fits_table_at_every_yaw():
    from dexverse.assets import SYNTHESIS_DIR
    from pxr import Usd, UsdGeom

    def corners(relative_path):
        stage = Usd.Stage.Open(str(SYNTHESIS_DIR / relative_path))
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            stage.GetDefaultPrim()).ComputeAlignedRange()
        return np.array(list(itertools.product(*zip(box.GetMin(), box.GetMax()))))

    cfg = _defaults()
    faucet_rot = R.from_quat(cfg.articulation_init_rot, scalar_first=True)
    sink_rot = R.from_quat(cfg.sink_init_rot, scalar_first=True)
    sink_local = (faucet_rot.inv() * sink_rot).apply(corners("sink/sink.usd") * cfg.sink_scale)
    sink_local += cfg.sink_offset_from_faucet
    faucet_local = corners("faucet001/model_faucet_3.usd") * cfg.articulation_scale
    radius = max(np.linalg.norm(sink_local[:, :2], axis=1).max(),
                 np.linalg.norm(faucet_local[:, :2], axis=1).max(),
                 np.linalg.norm(np.array(cfg.support_footprint) / 2))
    assert radius < 0.48
    # Read actual shared table dimensions: do not assume a test-only table size.
    namespace = {}
    tree = ast.parse((ROOT / "baseline_v1/dexverse_base_env_cfg.py").read_text())
    _compile([node for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in {"DEFAULT_TABLE_THICKNESS", "DEFAULT_TABLE_SIZE"}], namespace)
    for axis, key in enumerate(("x", "y")):
        endpoints = np.array(cfg.articulation_reset_pose_range[key]) + cfg.articulation_init_pos[axis]
        # A radial bound contains all rotated box corners for continuous yaw.
        assert namespace["DEFAULT_TABLE_SIZE"][axis] / 2 - max(abs(endpoints)) - radius >= 0.07
    assert 0.5 + radius > 0.75  # The old root range could overhang.


def test_seeded_reset_uses_new_ranges_with_env_origins_and_subset_ids():
    cfg = _defaults()

    def quat_from_euler_xyz(roll, pitch, yaw):
        values = torch.stack((roll, pitch, yaw), dim=-1).numpy()
        return torch.tensor(R.from_euler("xyz", values).as_quat(scalar_first=True), dtype=yaw.dtype)

    def quat_mul(left, right):
        result = R.from_quat(left.numpy(), scalar_first=True) * R.from_quat(right.numpy(), scalar_first=True)
        return torch.tensor(result.as_quat(scalar_first=True), dtype=left.dtype)

    namespace = {"torch": torch, "quat_from_euler_xyz": quat_from_euler_xyz, "quat_mul": quat_mul}
    for file, names in (("utils.py", {"resolve_env_ids"}), ("resets.py", {"reset_root_pose_uniform"})):
        tree = ast.parse((ROOT / "baseline_v1/mdp" / file).read_text())
        _compile([node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], namespace)
    count = 256
    states = torch.zeros(count, 13)
    states[:, :3] = torch.tensor((0.15, 0.0, 0.6 + cfg.articulation_half_height_est))
    states[:, 3:7] = torch.tensor(cfg.articulation_init_rot)
    poses = torch.zeros(count, 7)
    asset = SimpleNamespace(
        data=SimpleNamespace(default_root_state=states), cfg=SimpleNamespace(),
        write_root_pose_to_sim=lambda pose, env_ids: poses.__setitem__(env_ids, pose),
        write_root_velocity_to_sim=lambda velocity, env_ids: None,
    )

    class Scene(dict):
        pass

    scene = Scene(articulation=asset)
    scene.env_origins = torch.arange(count).unsqueeze(-1) * torch.tensor([[2.0, -3.0, 0.0]])
    env = SimpleNamespace(scene=scene, num_envs=count, device="cpu")
    params = dict(asset_cfg=SimpleNamespace(name="articulation"), pose_range=cfg.articulation_reset_pose_range)
    reset = namespace["reset_root_pose_uniform"]
    torch.manual_seed(42)
    reset(env, None, **params)
    first = poses.clone()
    local = poses[:, :3] - scene.env_origins
    assert local[:, 0].min() >= 0.1 - 1e-4 and local[:, 0].max() <= 0.2 + 1e-4
    assert local[:, 1].abs().max() <= 0.1 + 1e-4
    assert local[:, 0].std() > 0.025 and local[:, 1].std() > 0.05
    reset(env, torch.tensor([1, 3]), **params)
    torch.testing.assert_close(poses[[0, 2, 4]], first[[0, 2, 4]])
    torch.manual_seed(42)
    reset(env, None, **params)
    torch.testing.assert_close(poses, first)
