"""Offline regressions for kinematic-safe replay; no simulator launch."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "source/dexverse/dexverse/replay_rigid_object.py"


class FakeRigidObject:
    def __init__(self, flags):
        self.device = "cpu"
        self.flags = flags
        self.root_physx_view = SimpleNamespace(prim_paths=list(range(len(flags))))
        self.calls = []

    def _initialize_impl(self):
        pass

    def write_root_com_velocity_to_sim(self, velocity, env_ids=None):
        self.calls.append(("com", velocity, env_ids))

    def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
        self.calls.append(("link", velocity, env_ids))

    def write_root_velocity_to_sim(self, velocity, env_ids=None):
        self.write_root_com_velocity_to_sim(velocity, env_ids=env_ids)

    def write_root_state_to_sim(self, state, env_ids=None):
        self.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(state[:, 7:], env_ids=env_ids)

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.calls.append(("pose", pose, env_ids))


class FakeCfg:
    def __init__(self, class_type=FakeRigidObject):
        self.class_type = class_type


def load_adapter(flags):
    # Execute the actual class/helper against a recording base class. Importing
    # isaaclab.assets directly requires an AppLauncher and is not an offline test.
    stage = SimpleNamespace(GetPrimAtPath=lambda index: flags[index])
    physics = SimpleNamespace(RigidBodyAPI=lambda flag: SimpleNamespace(
        GetKinematicEnabledAttr=lambda: SimpleNamespace(Get=lambda: flag)))
    namespace = dict(torch=torch, RigidObject=FakeRigidObject, RigidObjectCfg=FakeCfg,
                     UsdPhysics=physics, sim_utils=SimpleNamespace(get_current_stage=lambda: stage))
    nodes = [node for node in ast.parse(MODULE.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODULE), "exec"), namespace)
    asset = namespace["ReplayRigidObject"](flags)
    asset._initialize_impl()
    return asset, namespace


@pytest.mark.parametrize("method", ["write_root_velocity_to_sim", "write_root_com_velocity_to_sim",
                                   "write_root_link_velocity_to_sim"])
@pytest.mark.parametrize("ids", [None, [0], torch.tensor([0]), slice(None)])
def test_kinematic_writes_never_reach_physx(method, ids):
    asset, _ = load_adapter([True])
    getattr(asset, method)(torch.ones(1, 6), env_ids=ids)
    assert asset.calls == []


@pytest.mark.parametrize("method", ["write_root_velocity_to_sim", "write_root_com_velocity_to_sim",
                                   "write_root_link_velocity_to_sim"])
def test_dynamic_velocity_arguments_unchanged(method):
    asset, _ = load_adapter([False, False])
    velocity, ids = torch.arange(12).reshape(2, 6), torch.tensor([1, 0])
    getattr(asset, method)(velocity, env_ids=ids)
    assert asset.calls[0][1] is velocity
    assert asset.calls[0][2] is ids


@pytest.mark.parametrize("ids,expected", [(None, [0, 2]), ([3, 2, 0], [2, 0]),
                                         (torch.tensor([2, 1]), [2]),
                                         (slice(1, 4), [2]), ([1, 3], []), ([], [])])
@pytest.mark.parametrize("method", ["write_root_com_velocity_to_sim", "write_root_link_velocity_to_sim"])
def test_mixed_instances_filter_selected_rows_not_global_rows(ids, expected, method):
    asset, _ = load_adapter([False, True, False, True])
    all_velocity = torch.arange(24).reshape(4, 6)
    velocity = all_velocity if ids is None else all_velocity[ids]
    getattr(asset, method)(velocity, env_ids=ids)
    if not expected:
        assert asset.calls == []
    else:
        assert torch.equal(asset.calls[0][1], all_velocity[expected])
        assert asset.calls[0][2].tolist() == expected


def test_kinematic_pose_still_restored_and_source_unchanged():
    asset, _ = load_adapter([True])
    state = torch.arange(13).reshape(1, 13).float()
    before = state.clone()
    asset.write_root_state_to_sim(state)
    assert [call[0] for call in asset.calls] == ["pose"]
    assert torch.equal(asset.calls[0][1], state[:, :7])
    assert torch.equal(state, before)


def test_dynamic_pose_and_velocity_both_restored():
    asset, _ = load_adapter([False])
    asset.write_root_state_to_sim(torch.zeros(1, 13))
    assert [call[0] for call in asset.calls] == ["pose", "com"]


def test_reinitialize_refreshes_spawned_flags():
    flags = [True]
    asset, _ = load_adapter(flags)
    asset.write_root_velocity_to_sim(torch.ones(1, 6))
    assert not asset.calls
    flags[0] = False
    asset._initialize_impl()
    asset.write_root_velocity_to_sim(torch.ones(1, 6))
    assert len(asset.calls) == 1


def test_configuration_is_local_idempotent_and_preserves_custom_classes():
    _, namespace = load_adapter([])
    custom_class = type("CustomRigidObject", (FakeRigidObject,), {})
    ordinary, custom = FakeCfg(), FakeCfg(custom_class)
    articulation = SimpleNamespace(class_type=object)
    untouched = FakeCfg()
    scene = SimpleNamespace(object=ordinary, custom=custom, robot=articulation, absent=None)
    configure = namespace["configure_replay_rigid_objects"]
    configure(scene)
    configure(scene)
    assert ordinary.class_type is namespace["ReplayRigidObject"]
    assert custom.class_type is custom_class
    assert articulation.class_type is object
    assert untouched.class_type is FakeRigidObject


@pytest.mark.parametrize("relative,fn", [
    ("scripts/demo_tools/create_demo_files_sequential.py", "_build_env_for_pickle"),
    ("source/dexverse/dexverse/IL/diffusion/online_eval.py", "_make_env"),
])
def test_adapter_configured_before_scene_creation(relative, fn):
    tree = ast.parse((ROOT / relative).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == fn)
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    configure = next(node for node in calls if ast.unparse(node.func) == "configure_replay_rigid_objects")
    make = next(node for node in calls if ast.unparse(node.func) == "gym.make")
    assert configure.lineno < make.lineno
