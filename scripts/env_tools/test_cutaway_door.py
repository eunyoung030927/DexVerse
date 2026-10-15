"""CPU asset topology and task-sequence checks for the replacement door v1."""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/dexverse"))
from dexverse.teleop_utils.door_latch_progress import new_door_latch_state, update_door_latch_state  # noqa: E402
from dexverse.benchmark import task_identity, validate_replay_identity, V1_CONFIGS  # noqa: E402


def step(state, index, angle=0.0, height=0.0, touch=False, support=False, clear=False, speed=0.0, latch_speed=0.0):
    return update_door_latch_state(
        state,
        torch.tensor([index]),
        torch.tensor([angle]),
        torch.tensor([speed]),
        torch.tensor([height]),
        torch.tensor([latch_speed]),
        *[torch.tensor([x]) for x in (touch, support, clear)],
        hold_steps=3,
    )


def test_closed_start_and_reclosure_after_opening():
    state = new_door_latch_state(1, "cpu")
    assert not step(state, 0)[1].item()
    step(state, 1, angle=0.4, touch=True)
    assert step(state, 2, angle=0.05)[1].item()
    assert step(state, 3, angle=1.57, support=True, clear=True)[1].item()
    assert not step(state, 0)[1].item()  # Per-environment counter rollback clears history.


def test_angle_alone_or_no_prior_hand_contact_do_not_succeed():
    state = new_door_latch_state(1, "cpu")
    for i in range(10):
        assert not step(state, i, angle=1.57, support=True, clear=True)[0].item()
    step(state, 10, angle=1.57, touch=True)
    for i in range(11, 20):
        assert not step(state, i, angle=1.57, clear=True)[0].item()


def test_auto_latch_needs_release_dwell_without_manual_lift_or_prescribed_grasp():
    state = new_door_latch_state(1, "cpu")
    step(state, 0, angle=0.2, touch=True)
    step(state, 1, angle=0.9, touch=True)
    step(state, 2, angle=1.3, height=0.04, touch=True)
    for i in range(3, 7):
        assert not step(state, i, angle=1.54, support=True, clear=False)[0].item()
    assert not step(state, 7, angle=1.54, support=True, clear=True)[0].item()
    assert not step(state, 7, angle=1.54, support=True, clear=True)[0].item()  # No double-counting.
    assert not step(state, 8, angle=1.54, support=True, clear=True)[0].item()
    assert step(state, 9, angle=1.54, support=True, clear=True)[0].item()
    assert not step(state, 10, angle=1.54, support=False, clear=True)[0].item()


@pytest.mark.parametrize("interruption", [
    {"touch": True}, {"clear": False}, {"support": False},
    {"angle": 1.2}, {"height": 0.035}, {"speed": 0.1}, {"latch_speed": 0.05},
])
def test_release_dwell_restarts_on_recontact_or_loss_of_stable_latch(interruption):
    state = new_door_latch_state(1, "cpu")
    step(state, 0, angle=1.54, touch=True)
    seated = {"angle": 1.54, "support": True, "clear": True}
    assert not step(state, 1, **seated)[0].item()
    assert not step(state, 2, **seated)[0].item()
    assert not step(state, 3, **(seated | interruption))[0].item()
    assert state[0, 4] == 0
    assert not step(state, 4, **seated)[0].item()
    assert not step(state, 5, **seated)[0].item()
    assert step(state, 6, **seated)[0].item()


def test_skipped_evaluation_does_not_count_as_continuous_release():
    state = new_door_latch_state(1, "cpu")
    step(state, 0, angle=1.54, touch=True)
    for index in (1, 2, 5, 6):
        assert not step(state, index, angle=1.54, support=True, clear=True)[0].item()
    assert step(state, 7, angle=1.54, support=True, clear=True)[0].item()


def test_urdf_has_real_cutout_and_separate_physical_catch():
    root = ET.parse(ROOT / "scripts/asset_tools/urdf/cutaway_door.urdf").getroot()
    assert {link.attrib["name"] for link in root.findall("link")} == {"frame", "door", "latch"}
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    assert set(joints) == {"door_hinge", "latch_slide"}
    assert joints["latch_slide"].find("parent").attrib["link"] == "frame"
    assert float(joints["door_hinge"].find("limit").attrib["upper"]) > 1.57
    # Test the centre of the window against all four solid collision boxes.
    panel = root.find("link[@name='door']")
    assert len(panel.findall("collision")) == 4
    point = (0.25, 0.0, 0.25)
    for collider in panel.findall("collision"):
        center = list(map(float, collider.find("origin").attrib["xyz"].split()))
        size = list(map(float, collider.find("geometry/box").attrib["size"].split()))
        assert not all(abs(p - c) < s / 2 for p, c, s in zip(point, center, size))


def test_door_revision_rejects_predecessor_without_affecting_other_tasks():
    task = "Dexverse-OpenDoor-v1"
    assert "open_cutaway_door_cfg" in V1_CONFIGS[task][0]
    validate_replay_identity({"task": task, **task_identity(task)}, task)
    for revision in (None, "baseline-v1-2026-09-08", "door-cutaway-latch-v1-2026-09-08",
                     "door-cutaway-latch-framed-v1-2026-09-08"):
        with pytest.raises(ValueError, match="provenance"):
            validate_replay_identity({"task": task, "benchmark_revision": revision}, task)
    assert task_identity("Dexverse-GraspBleach-v1")["benchmark_revision"] == "baseline-v1-2026-09-08"


def test_progress_resets_only_environment_whose_step_counter_rolls_back():
    state = new_door_latch_state(2, "cpu")
    state[:, :4] = 1
    state[:, 4] = 2
    state[:, 5] = 10
    update_door_latch_state(
        state,
        torch.tensor([0, 11]),
        torch.tensor([0.0, 1.54]),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([False, True]),
        torch.tensor([False, True]),
        hold_steps=3,
    )
    assert not state[0, :5].any()
    assert state[1, 6] == 1


def test_narrow_window_and_matching_beveled_latch():
    directory = ROOT / "scripts/asset_tools/urdf"
    root = ET.parse(directory / "cutaway_door.urdf").getroot()
    boxes = [_box(collider) for collider in root.find("link[@name='door']").findall("collision")]

    def solid(point):
        return any(all(abs(p - c) < h for p, c, h in zip(point, center, half)) for center, half in boxes)

    assert solid((0.25, 0, 0.205)) and solid((0.25, 0, 0.295))
    assert not solid((0.25, 0, 0.215)) and not solid((0.25, 0, 0.285))
    mesh = root.find("link[@name='latch']/collision[@name='latch_bevel']/geometry/mesh")
    vertices = [tuple(map(float, line.split()[1:])) for line in
                (directory / mesh.attrib["filename"]).read_text().splitlines() if line.startswith("v ")]
    assert len(vertices) == 6
    sizes = [max(v[i] for v in vertices) - min(v[i] for v in vertices) for i in range(3)]
    assert sizes == pytest.approx([0.05, 0.12, 0.06])
    assert (0, 0, 0) in vertices and (0.05, 0, 0.06) in vertices  # Camming slope.
    # Outward-facing, watertight triangles for both rendering and collision.
    faces = [tuple(int(x) - 1 for x in line.split()[1:]) for line in
             (directory / mesh.attrib["filename"]).read_text().splitlines() if line.startswith("f ")]
    edges = [(face[i], face[(i + 1) % 3]) for face in faces for i in range(3)]
    assert all(edges.count((b, a)) == 1 for a, b in edges)
    volume = 0.0
    for face in faces:
        a, b, c = (torch.tensor(vertices[i], dtype=torch.float64) for i in face)
        volume += float(torch.dot(a, torch.linalg.cross(b, c))) / 6
    assert volume == pytest.approx(0.5 * 0.05 * 0.06 * 0.12)


def _box(element):
    center = tuple(map(float, element.find("origin").attrib["xyz"].split()))
    half = tuple(float(x) / 2 for x in element.find("geometry/box").attrib["size"].split())
    return center, half


def test_full_frame_blocks_initial_edge_hooks_but_leaves_window_and_later_regrasp_clear():
    root = ET.parse(ROOT / "scripts/asset_tools/urdf/cutaway_door.urdf").getroot()
    frame = root.find("link[@name='frame']")
    names = {"frame_header", "frame_sill", "frame_free_jamb", "frame_hinge_jamb", "frame_hinge_rebate"}
    for name in names:
        assert _box(frame.find(f"collision[@name='{name}']")) == _box(frame.find(f"visual[@name='{name}']"))
    boxes = [_box(collider) for collider in frame.findall("collision")]

    def blocked(point, radius=0.008):
        # An 8 mm radius probe represents a finger trying to wrap an edge;
        # this geometry check is not a full articulated-hand adversarial test.
        return any(
            sum(max(abs(p - c) - h, 0) ** 2 for p, c, h in zip(point, center, half)) < radius**2
            for center, half in boxes
        )

    for point in ((0.26, 0, 0.428), (0.348, 0, 0.25), (0.26, 0, 0.012), (-0.004, -0.009, 0.22)):
        assert blocked(point)
    assert not blocked((0.25, 0, 0.25))  # Initial window access.
    for degrees in (40, 60, 90):
        angle = math.radians(degrees)
        for height in (0.02, 0.42):
            assert not blocked((0.26 * math.cos(angle), 0.26 * math.sin(angle), height))


def test_full_frame_does_not_intersect_panel_through_entire_hinge_sweep():
    root = ET.parse(ROOT / "scripts/asset_tools/urdf/cutaway_door.urdf").getroot()
    frame = [_box(collider) for collider in root.find("link[@name='frame']").findall("collision")]
    panel = [_box(collider) for collider in root.find("link[@name='door']").findall("collision")]
    # Separating-axis test for rotated boxes, including the two bodies' 2 mm
    # contact offsets. Frame geometry must not rely on parent collision filtering.
    for quarter_degree in range(385):
        angle = math.radians(quarter_degree / 4)
        u, v = (math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))
        for center, half in panel:
            rotated = (center[0] * u[0] + center[1] * v[0], center[0] * u[1] + center[1] * v[1])
            for fixed, fixed_half in frame:
                if abs(center[2] - fixed[2]) >= half[2] + fixed_half[2] + 0.004:
                    continue
                delta = (rotated[0] - fixed[0], rotated[1] - fixed[1])
                separated = False
                for axis in ((1, 0), (0, 1), u, v):
                    distance = abs(sum(d * a for d, a in zip(delta, axis)))
                    extent = (
                        half[0] * abs(sum(x * a for x, a in zip(u, axis)))
                        + half[1] * abs(sum(x * a for x, a in zip(v, axis)))
                        + sum(h * abs(a) for h, a in zip(fixed_half, axis))
                    )
                    separated |= distance >= extent + 0.004
                assert separated, f"Frame intersects panel at {quarter_degree / 4} degrees"
