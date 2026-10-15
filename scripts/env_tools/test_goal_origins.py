"""CPU regressions for the real v1 pose sampler across translated scenes."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def command_classes():
    path = ROOT / "source/dexverse/dexverse/baseline_v1/mdp/commands/pose_commands.py"
    tree = ast.parse(path.read_text())
    # Load the actual classes without starting Kit; only quaternion math and
    # the unused manager constructor need stand-ins for these CPU cases.
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    namespace = {"CommandTerm": object, "torch": torch,
                 "quat_from_euler_xyz": lambda r, p, y: torch.tensor([1., 0., 0., 0.]).repeat(len(r), 1),
                 "quat_unique": lambda q: q, "quat_apply": lambda q, v: v}
    import __future__

    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def make_command(tracking=False, world=True):
    classes = command_classes()
    cls = classes["ObjectAssetTrackingPoseCommand" if tracking else "ObjectUniformPoseCommand"]
    cmd = cls.__new__(cls)
    # CommandTerm exposes device via the environment; the CPU stand-in does not.
    cmd.device = "cpu"
    cmd.num_envs = 3
    origins = torch.tensor([[0., 0., 0.], [1.5, -3., .5], [-1.5, 3., 1.]])
    cmd._env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins))
    cmd.cfg = SimpleNamespace(use_world_frame=world, make_quat_unique=True,
                             local_offset=(.1, .2, .3),
                             ranges=SimpleNamespace(pos_x=(.4, .4), pos_y=(.1, .1), pos_z=(.8, .8),
                                                    roll=(0., 0.), pitch=(0., 0.), yaw=(0., 0.)))
    cmd.pose_command_b = torch.zeros(3, 7)
    cmd.pose_command_w = torch.zeros(3, 7)
    cmd._local_offset = torch.tensor(cmd.cfg.local_offset)
    return cmd, origins


@pytest.mark.parametrize("world", [True, False])
def test_partial_resets_keep_goals_in_their_own_frame_without_accumulation(world):
    cmd, origins = make_command(world=world)
    cmd._resample_command([0, 1, 2])
    expected = torch.tensor([[.4, .1, .8]]).repeat(3, 1)
    if world:
        expected += origins
    torch.testing.assert_close(cmd.pose_command_b[:, :3], expected)
    saved = cmd.pose_command_b[0].clone()
    for _ in range(3):
        cmd._resample_command(torch.tensor([1, 2]))
        torch.testing.assert_close(cmd.pose_command_b[:, :3], expected)
        torch.testing.assert_close(cmd.pose_command_b[0], saved)
    if world:
        torch.testing.assert_close(cmd.pose_command_w, cmd.pose_command_b)


def test_asset_tracking_does_not_add_origin_twice():
    cmd, origins = make_command(tracking=True)
    cmd.target = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=origins + torch.tensor([.2, .3, .4]),
        root_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(3, 1)))
    cmd._resample_command([0, 1, 2])
    expected = cmd.target.data.root_pos_w + torch.tensor(cmd.cfg.local_offset)
    torch.testing.assert_close(cmd.pose_command_b[:, :3], expected)
    cmd.target.data.root_pos_w += .1
    cmd._update_command()
    torch.testing.assert_close(cmd.pose_command_w[:, :3], expected + .1)
