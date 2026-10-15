"""Regressions for the four replayable, depth-visible carton corner spheres."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / "source/dexverse/dexverse"
CONFIG = SOURCE / "baseline_v1/config/bimanual/carton_regrasp_cfg.py"
sys.path.insert(0, str(SOURCE.parent))
from dexverse.teleop_utils.episode_task_state import capture_episode_task_state, restore_episode_task_state


def _geometry():
    nodes = []
    for node in ast.parse(CONFIG.read_text()).body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "GOAL_FRAME_CFG" for target in node.targets):
                break
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef):
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CONFIG), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("face", range(6))
def test_four_spheres_form_selected_face_rectangle(face):
    values = _geometry()
    axes, half, origin = values["_AXES"], values["_HALF"], values["_CENTER"]
    axis, sign = values["_FACES"][face]
    points = values["FACE_CORNERS_PER_FACE"][face]
    assert values["FACE_CORNER_RADIUS"] == 0.02
    assert len(set(points)) == 4
    normal = axes.index(axis)
    for point in points:
        assert point[normal] == pytest.approx(origin[axis] + sign * half[axis])
    for index, name in enumerate(axes):
        if index != normal:
            assert sorted(set(p[index] for p in points)) == pytest.approx([origin[name] - half[name], origin[name] + half[name]])


def test_config_uses_red_geometry_and_preserves_recorded_buffer():
    tree = ast.parse(CONFIG.read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "add_scene_vis_term"
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == "graspable_face")
    keywords = {k.arg: k.value for k in call.args[1].keywords}
    assert ast.unparse(keywords["func"]) == "CartonFaceCornerCue"
    params = {k.value: v for k, v in zip(keywords["params"].keys, keywords["params"].values)}
    assert ast.literal_eval(params["color"]) == (0.9, 0.03, 0.03)
    assert ast.unparse(params["buffer_name"]) == "GRASP_FACE_BUFFER"
    assert _geometry()["GRASP_FACE_BUFFER"] == "carton_grasp_face"
    assert "ALLOWED_PATCHES_PER_FACE" not in CONFIG.read_text()
    code = (SOURCE / "baseline_v1/config/bimanual/carton_face_cue.py").read_text()
    assert "VisualizationMarkers" not in code
    assert "invisibleToSecondaryRays" not in code
    assert "use_fabric" not in code


def test_green_spheres_mark_only_the_goal_frame():
    tree = ast.parse(CONFIG.read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "add_scene_vis_term"
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == "goal_box_corners")
    params = next(kw.value for kw in call.args[1].keywords if kw.arg == "params")
    fields = {key.value: value for key, value in zip(params.keys, params.values)}
    assert ast.unparse(fields["asset_cfg"]) == "SceneEntityCfg('goal_frame')"
    assert ast.literal_eval(fields["color"]) == (0.15, 0.75, 0.25)


def test_record_restore_updates_four_local_offsets_and_skips_unchanged_choice(monkeypatch):
    values = _geometry()
    cls = next(n for n in ast.parse((CONFIG.parent / "carton_face_cue.py").read_text()).body
               if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    namespace = {"torch": torch, "SceneEntityCfg": object}
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(Gf=SimpleNamespace(Vec3d=lambda *p: p)))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(CONFIG), "exec"), namespace)
    writes = [[] for _ in range(4)]
    cue = SimpleNamespace(_buffer_name="carton_grasp_face", _points=values["FACE_CORNERS_PER_FACE"],
                          _translations=[[SimpleNamespace(Set=items.append) for items in writes]], _last_choice=[-1])
    env = SimpleNamespace(num_envs=1, device="cpu", carton_grasp_face=torch.tensor([4]),
                          cfg=SimpleNamespace(events=SimpleNamespace(sample_grasp_face=SimpleNamespace(
                              params={"buffer_name": "carton_grasp_face"}))))
    params = dict(points_per_choice=cue._points, buffer_name=cue._buffer_name, object_cfg=None,
                  radius=0.02, color=(0.9, 0.03, 0.03))
    snapshot = capture_episode_task_state(env)
    assert snapshot["env_buffers"]["carton_grasp_face"].item() == 4
    env.carton_grasp_face.fill_(1)
    namespace["__call__"](cue, env, **params)
    assert [items[-1] for items in writes] == cue._points[1]
    assert restore_episode_task_state(env, snapshot) == []
    namespace["__call__"](cue, env, **params)
    assert [items[-1] for items in writes] == cue._points[4]
    namespace["__call__"](cue, env, **params)
    assert [len(items) for items in writes] == [2] * 4


def test_carton_and_success_dimensions_unchanged():
    values = _geometry()
    assert values["CARTON_SCALE"] == 0.65
    hx, hy, hz = values["CARTON_HALF_X"], values["CARTON_HALF_Y"], values["CARTON_HEIGHT"] / 2
    assert values["cue_half_extents_per_face"](0.65) == [(hz, hy, hx)] * 2 + [(hx, hz, hy)] * 2 + [(hx, hy, hz)] * 2
