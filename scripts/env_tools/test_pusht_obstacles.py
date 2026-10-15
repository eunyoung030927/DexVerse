"""CPU regressions for goal-local Push-T rods and swept XY/yaw reachability."""

import ast
import importlib.util
import itertools
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/dexverse"))
PATH = ROOT / "source/dexverse/dexverse/baseline_v1/mdp/pusht_obstacles.py"
spec = importlib.util.spec_from_file_location("pusht_layout", PATH)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


def test_t_bound_matches_actual_asset():
    from pxr import Usd, UsdGeom
    from dexverse.assets import DEXVERSE_AUTHORED_ASSETS_DIR
    stage = Usd.Stage.Open(str(DEXVERSE_AUTHORED_ASSETS_DIR / "push_t/tee.usda"))
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
    np.testing.assert_allclose(np.array(layout.TEE_XY_BOUNDS), np.stack((lo[:2], hi[:2]), axis=-1), atol=1e-7)
    corners = np.array(list(itertools.product(*layout.TEE_XY_BOUNDS)))
    assert np.linalg.norm(corners, axis=1).max() == pytest.approx(layout.TEE_ROOT_RADIUS)
    for i, name in enumerate(("bar_horizontal", "bar_vertical")):
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            stage.GetPrimAtPath(f"/tee/{name}")).ComputeAlignedRange()
        lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
        np.testing.assert_allclose((lo[:2] + hi[:2]) / 2, layout.TEE_RECT_CENTERS[i], atol=1e-7)
        np.testing.assert_allclose((hi[:2] - lo[:2]) / 2, layout.TEE_RECT_HALF_SIZES[i], atol=1e-7)
    assert layout.ROD_MIN_SPACING - 2 * layout.ROD_RADIUS == pytest.approx(.15)


def independently_check_path(path, rods, *, table_center=(0., 0.)):
    """Dense world-polygon checks independent of the planner's local boxes."""
    for start, end in zip(path[:-1], path[1:]):
        delta = end - start
        delta[2] = (delta[2] + math.pi) % (2 * math.pi) - math.pi
        travel = np.linalg.norm(delta[:2]) + layout.TEE_ROOT_RADIUS * abs(delta[2])
        n = max(1, int(np.ceil(travel / .002)))
        poses = start + np.linspace(0., 1., n + 1)[:, None] * delta
        for center, half in zip(layout.TEE_RECT_CENTERS, layout.TEE_RECT_HALF_SIZES):
            local = center + half * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
            c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
            rotation = np.stack((c, -s, s, c), axis=-1).reshape(-1, 2, 2)
            vertices = np.einsum("nij,kj->nki", rotation, local) + poses[:, None, :2]
            assert (np.abs(vertices - table_center) <= .75 - layout.PATH_MARGIN + 1e-7).all()
            edge = np.roll(vertices, -1, axis=1) - vertices
            relative = rods[None, None, :, :] - vertices[:, :, None, :]
            t = np.clip((relative * edge[:, :, None, :]).sum(-1) / (edge * edge).sum(-1)[:, :, None], 0, 1)
            distance = np.linalg.norm(relative - t[..., None] * edge[:, :, None, :], axis=-1)
            assert distance.min() >= layout.ROD_RADIUS + layout.PATH_MARGIN - 1e-7
            cross = edge[:, :, None, 0] * relative[..., 1] - edge[:, :, None, 1] * relative[..., 0]
            assert not (cross >= 0).all(axis=1).any(), "Rod inside T bar"


def test_dense_uniform_proposals_block_direct_route_but_have_clear_detour():
    torch.manual_seed(42)
    samples = layout.sample_layouts(64)
    rods = samples[:, 6:].reshape(-1, layout.ROD_COUNT, 2)
    r = layout.TEE_CLEARANCE_RADIUS
    assert layout.ROD_COUNT == 10 and layout.ROD_HEIGHT == .04
    assert (rods.abs() + layout.ROD_RADIUS < .75).all()
    distance = torch.cdist(rods, rods) + torch.eye(layout.ROD_COUNT).unsqueeze(0) * 10
    assert distance.min() >= layout.ROD_MIN_SPACING - 1e-6
    assert distance.min() < .20, "Sampling must actually use the tighter spacing"
    radius = torch.linalg.vector_norm(rods - samples[:, None, 3:5], dim=-1)
    assert radius.min() >= layout.GOAL_ROD_RADII[0] - 1e-6
    assert radius.max() <= layout.GOAL_ROD_RADII[1] + 1e-6
    for point in (samples[:, :2], samples[:, 3:5]):
        assert (point.abs() < .75 - r).all()
    assert torch.linalg.vector_norm(samples[:, :2] - samples[:, 3:5], dim=-1).min() >= layout.START_GOAL_MIN_DISTANCE - 1e-6
    assert ((samples[:, [2, 5]] >= 0) & (samples[:, [2, 5]] < 2 * math.pi)).all()
    # Sequential proposals are diverse, not a fixed grid or identical ring.
    assert (rods.std(0) > .10).all()
    for i in range(layout.ROD_COUNT):
        assert torch.unique(rods[:, i, 0]).numel() == len(samples)
    assert layout.direct_path_blocked(samples[:, :2].numpy(), samples[:, 3:5].numpy(), rods.numpy()).all()
    for row, obstacles in zip(samples.numpy(), rods.numpy()):
        bearing = np.sort(np.arctan2(obstacles[:, 1] - row[4], obstacles[:, 0] - row[3]))
        assert np.diff(np.r_[bearing, bearing[0] + 2 * math.pi]).max() <= layout.MAX_ROD_ANGULAR_GAP + 1e-6
        path = layout.find_clear_path(row[:3], row[3:6], obstacles)
        assert path is not None
        np.testing.assert_allclose(path[0], row[:3])
        np.testing.assert_allclose(path[-1], row[3:6])
        independently_check_path(path, obstacles)
        assert np.linalg.norm(np.diff(path[:, :2], axis=0), axis=-1).sum() > np.linalg.norm(row[:2] - row[3:5])


def test_seed_repeatability_and_partial_batch():
    torch.manual_seed(42)
    first = layout.sample_layouts(5)
    torch.manual_seed(43)
    different = layout.sample_layouts(5)
    torch.manual_seed(42)
    torch.testing.assert_close(layout.sample_layouts(5), first)
    assert not torch.equal(first, different)
    assert layout.sample_layouts(0).shape == (0, 6 + 2 * layout.ROD_COUNT)


def test_no_unsafe_fallback_for_impossible_settings():
    with pytest.raises(ValueError, match="Table too small"):
        layout.sample_layouts(1, table_size=(.5, .5))
    with pytest.raises(RuntimeError, match="within budget"):
        layout.sample_layouts(1, spawn_x=(0., .001), spawn_y=(0., .001), max_rounds=2)


def test_planner_rejects_wall_and_handles_shifted_table():
    start, goal = np.array([-.4, 0., 0.]), np.array([.4, 0., math.pi])
    wall = np.stack((np.zeros(51), np.linspace(-.75, .75, 51)), axis=-1)
    assert layout.direct_path_blocked(start[:2], goal[:2], wall)
    assert layout.find_clear_path(start, goal, wall) is None
    obstacle = np.array([[0., 0.]])
    path = layout.find_clear_path(start, goal, obstacle)
    assert path is not None
    shift = np.array([2., -3., 0.])
    shifted = layout.find_clear_path(start + shift, goal + shift, obstacle + shift[:2], table_center=shift[:2])
    assert shifted is not None
    np.testing.assert_allclose(shifted[[0, -1]] - shift, [start, goal])


def test_actual_shape_allows_clear_notches_and_checks_swept_rotation():
    pose = np.zeros(3)
    rod = np.array([[.08, .10]])  # Inside bounding box, outside both actual bars.
    assert layout.pose_clearance(pose, rod) > layout.PATH_MARGIN
    assert np.linalg.norm(rod[0]) < layout.TEE_CLEARANCE_RADIUS + layout.ROD_RADIUS
    assert layout.pose_clearance(np.array([0., 0., math.pi]), rod) > layout.PATH_MARGIN
    assert not layout.motion_is_clear(pose, np.array([0., 0., math.pi]), rod)
    # Seam connection must take the short turn, not an almost-full revolution.
    assert layout.motion_is_clear(np.array([0., 0., .01]), np.array([0., 0., 2 * math.pi - .01]), rod)
    assert layout.pose_clearance(pose, np.array([[0., 0.]])) < 0
    assert layout.pose_clearance(np.array([.72, 0., 0.]), np.empty((0, 2))) < 0


def test_planner_turns_through_passage_that_rejects_rotation_envelope():
    # Synthetic channel narrower than the old 37 cm rotation envelope. The
    # starting yaw cannot translate through it; a changed yaw fits the real T.
    rods = np.array([(x, y) for x in np.linspace(-.21, .21, 15) for y in (-.15, .15)])
    start, goal = np.array([-.45, 0., math.pi / 4]), np.array([.45, 0., math.pi / 4])
    assert not layout.motion_is_clear(start, goal, rods)
    path = layout.find_clear_path(start, goal, rods)
    assert path is not None
    assert np.abs(path[:, 1]).max() < .04  # Through the channel, not around it.
    assert np.abs(layout.angle_delta(start[2], path[:, 2])).max() > .20
    independently_check_path(path, rods)


def test_sampler_supports_translated_table():
    torch.manual_seed(123)
    first = layout.sample_layouts(2)
    torch.manual_seed(123)
    shifted = layout.sample_layouts(2, table_center=(2., -3.))
    first[:, :2] += torch.tensor([2., -3.])
    first[:, 3:5] += torch.tensor([2., -3.])
    first[:, 6:] += torch.tensor([2., -3.] * layout.ROD_COUNT)
    torch.testing.assert_close(shifted, first)


def test_all_ten_collision_rods_declared():
    path = ROOT / "source/dexverse/dexverse/baseline_v1/config/non_prehensile/pusht_cfg.py"
    tree = ast.parse(path.read_text())
    scene = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PushTSceneCfg")
    rods = {n.target.id for n in scene.body if isinstance(n, ast.AnnAssign) and n.target.id.startswith("push_t_rod_")}
    assert rods == set(layout.ROD_NAMES)


class Asset:
    def __init__(self, count, z, dynamic=False):
        self.dynamic = dynamic
        default = torch.zeros(count, 13)
        default[:, 2], default[:, 3] = z, 1
        self.data = SimpleNamespace(default_root_state=default, root_pos_w=torch.zeros(count, 3))
        self.pose = default[:, :7].clone()
        self.velocity = torch.ones(count, 6)

    def write_root_pose_to_sim(self, pose, env_ids):
        self.pose[env_ids] = pose
        self.data.root_pos_w[env_ids] = pose[:, :3]

    def write_root_velocity_to_sim(self, values, env_ids):
        assert self.dynamic, "Must not write kinematic rod/goal velocities"
        self.velocity[env_ids] = values


def test_reset_subset_origins_and_kinematic_pose_only():
    class Scene(dict):
        pass
    scene = Scene(object=Asset(4, .621, True), goal_tee=Asset(4, .603), table=Asset(4, .575))
    scene.update({name: Asset(4, .62) for name in layout.ROD_NAMES})
    scene.env_origins = torch.tensor([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.], [3., 3., 0.]])
    env = SimpleNamespace(scene=scene, num_envs=4, device="cpu")
    ids = torch.tensor([1, 3])
    torch.manual_seed(42)
    layout.reset_push_t_with_rods(env, ids)
    first = scene['object'].pose.clone()
    assert scene['object'].pose[[0, 2], :2].count_nonzero() == 0
    torch.testing.assert_close(scene['object'].velocity[ids], torch.zeros(2, 6))
    assert scene['object'].pose[ids, 0].min() > 2.5
    assert scene['push_t_rod_0'].pose[ids, 2].tolist() == pytest.approx([.62, .62])
    torch.manual_seed(42)
    layout.reset_push_t_with_rods(env, ids)
    torch.testing.assert_close(scene['object'].pose, first)
    assert layout.rod_positions_in_table(env).shape == (4, 30)


def test_only_v1_has_rods_looser_success_and_revision_guard():
    v1 = ROOT / "source/dexverse/dexverse/baseline_v1/config/non_prehensile/pusht_cfg.py"
    v0 = ROOT / "source/dexverse/dexverse/tasks/config/non_prehensile/pusht_cfg.py"
    def threshold(path):
        tree = ast.parse(path.read_text())
        return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == "OVERLAP_SUCCESS_THRESHOLD")
    assert threshold(v1) == .70 and threshold(v0) == .90
    assert "push_t_rod_0" in v1.read_text() and "push_t_rod" not in v0.read_text()
    from dexverse.benchmark import task_identity, validate_replay_identity, BENCHMARK_REVISION
    identity = task_identity("Dexverse-PushT-v1")
    validate_replay_identity({"task": "Dexverse-PushT-v1", **identity}, "Dexverse-PushT-v1")
    for old in (None, BENCHMARK_REVISION, "push-t-orange-rods-v1-2026-09-10",
                "push-t-ten-low-rods-detour-v1-2026-09-10"):
        with pytest.raises(ValueError, match="orange rods"):
            validate_replay_identity({"task": "Dexverse-PushT-v1", "benchmark_revision": old}, "Dexverse-PushT-v1")
