# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacSL (Isaac Lab VisuoTactileSensor) smoke test on the Sharpa Wave hand's finger elastomers (experiment).

Standalone scene (not a DexVerse task): floating Sharpa right (or bimanual) hand, whose ``*_elastomer`` finger
sleeves get a COMPLIANT contact material (``--compliant_stiffness``; 0 = leave them rigid), and a kinematic box
whose collision is an SDF mesh (TacSL's force field needs an SDF on the contact object). One
VisuoTactileSensor per instrumented elastomer, plus a plain ContactSensor on the same body for comparison.

Press test (``--press``): the box is placed just off the index pad (TacSL pad side = the elastomer's dome side
along its thinnest axis), then the index flexion targets (MCP_FE, PIP, DIP) are moved level by level in the
direction that drives the pad into the box (measured first); per level the TacSL force-field sum / max /
contact area / max penetration and the ContactSensor force are logged.

Camera mode (``--camera``): a pinhole camera is spawned INSIDE every instrumented elastomer, ``--cam_dist`` behind
the pad apex and looking out through the pad; the elastomer and distal-phalanx visuals of those fingers are
hidden and the far clip is the pad's tangent plane (depth_clipping_behavior="max"), so the depth image is flat
without contact and shows the object where it pushes past the pad surface: the height map TacSL renders with
the GELSIGHT_R15 calibration.

    /workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/tacsl_sharpa_smoke.py --headless --press \
        --fingers index --out /root/tactile_test/tacsl/ff_k2000 [--camera --enable_cameras]
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hand", choices=["right", "bimanual"], default="right")
parser.add_argument("--fingers", default="index", help="'index', 'all', 'none' or a comma list of thumb,index,...")
parser.add_argument("--compliant_stiffness", type=float, default=2000.0,
                    help="elastomer compliant contact stiffness (N/m); 0 = rigid elastomer (no compliant material)")
parser.add_argument("--compliant_damping", type=float, default=20.0)
parser.add_argument("--tactile_k", type=float, default=1.0, help="TacSL normal_contact_stiffness (per m of depth)")
parser.add_argument("--array", type=int, nargs=2, default=[12, 16], help="TacSL tactile_array_size")
parser.add_argument("--press", action="store_true", help="run the index press test")
parser.add_argument("--levels", type=int, default=10)
parser.add_argument("--delta", type=float, default=0.04, help="rad added to each index flexion target per level")
parser.add_argument("--steps_per_level", type=int, default=40)
parser.add_argument("--gap", type=float, default=0.0005, help="box face distance from the pad apex at placement (m)")
parser.add_argument("--camera", action="store_true", help="enable TacSL camera tactile (needs --enable_cameras)")
parser.add_argument("--no_force_field", action="store_true")
parser.add_argument("--cam_dist", type=float, default=0.006)
parser.add_argument("--pad_sign", choices=["tacsl", "+1", "-1"], default="tacsl",
                    help="pad side along the elastomer's thinnest axis: TacSL's own dome heuristic, or forced (the "
                         "tactile points are then regenerated on that side after the sensor initialised)")
parser.add_argument("--timing_steps", type=int, default=0, help="extra steps for timing (after the press test)")
parser.add_argument("--object", choices=["box", "sphere"], default="box", help="kinematic SDF contact object")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0")
args = parser.parse_args()
app = AppLauncher(vars(args)).app


def _exit_on_error(exc_type, exc, tb):  # an uncaught error otherwise leaves Kit hanging until the timeout
    import traceback  # noqa: PLC0415

    traceback.print_exception(exc_type, exc, tb)
    os._exit(1)


sys.excepthook = _exit_on_error

import cv2  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from isaaclab_assets.sensors import GELSIGHT_R15_CFG  # noqa: E402
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg  # noqa: E402
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import compute_tactile_shear_image  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

import dexverse  # noqa: E402
from dexverse.robot_agents.sharpa import floating as sharpa  # noqa: E402

print("DEXVERSE_FILE", dexverse.__file__)
os.makedirs(args.out, exist_ok=True)
ALL = ["thumb", "index", "middle", "ring", "pinky"]
if args.fingers == "all":
    fingers = ALL
elif args.fingers == "none":
    fingers = []
else:
    fingers = args.fingers.split(",")
sides = ["right"] if args.hand == "right" else ["right", "left"]
pads = [f"{s}_{f}_elastomer" for s in sides for f in fingers]
base_cfg = sharpa.FLOATING_SHARPA_RIGHT_CFG if args.hand == "right" else sharpa.FLOATING_SHARPA_BIMANUAL_CFG
usd_path = base_cfg.spawn.usd_path
use_ff = not args.no_force_field
cam_w, cam_h, mmpp = GELSIGHT_R15_CFG.image_width, GELSIGHT_R15_CFG.image_height, GELSIGHT_R15_CFG.mm_per_pixel


# ---------------------------------------------------------------- elastomer geometry (same rules as TacSL)
def pad_geometry(body: str) -> dict:
    """Elastomer visual mesh (body frame): thinnest axis, dome sign and apex along it (as TacSL decides)."""
    stage = Usd.Stage.Open(usd_path)
    mesh_prim = None
    for p in stage.Traverse():
        if p.GetName() == body:
            for d in Usd.PrimRange(p):
                if d.IsA(UsdGeom.Mesh) and not d.HasAPI(UsdPhysics.CollisionAPI):
                    mesh_prim = d
                    break
            break
    m = UsdGeom.Mesh(mesh_prim)
    pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
    faces = np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    tm = trimesh.Trimesh(vertices=pts, faces=faces)
    lo, hi = pts.min(0), pts.max(0)
    slim = int(np.argmin(hi - lo))
    sign = 1.0 if tm.center_mass[slim] > (lo[slim] + hi[slim]) / 2 else -1.0
    tacsl_sign = sign
    if args.pad_sign != "tacsl":
        sign = float(args.pad_sign)
    apex = hi[slim] if sign > 0 else lo[slim]
    MESHES[body] = tm
    return {"slim": slim, "sign": sign, "tacsl_sign": tacsl_sign, "apex": float(apex),
            "center": ((lo + hi) / 2).tolist(), "size": (hi - lo).tolist(), "mesh": str(mesh_prim.GetPath())}


MESHES = {}
geo = {p: pad_geometry(p) for p in pads}
print("PAD_GEOMETRY", json.dumps(geo))


def regen_tactile_points(sensor, body: str):
    """TacSL's own grid + ray cast (``_generate_tactile_points``), but on the pad side ``geo[body]['sign']``."""
    g, tm = geo[body], MESHES[body]
    slim, side = g["slim"], g["sign"]
    lo, hi = tm.vertices.min(0), tm.vertices.max(0)
    dims, center = hi - lo, (lo + hi) / 2
    num_divs, margin = list(args.array), 0.001
    others = [i for i in range(3) if i != slim]
    dx = min((dims[others] - 2 * margin) / (np.array(num_divs) + 1))
    planar, k = [], 0
    for ax in range(3):
        if ax == slim:
            planar.append([side])
        else:
            line = np.linspace(center[ax] - dx * (num_divs[k] + 1) / 2, center[ax] + dx * (num_divs[k] + 1) / 2,
                               num_divs[k] + 2)
            planar.append(line[1:-1])
            k += 1
    origins = np.array(list(itertools.product(*planar)))
    ray = np.zeros(3)
    ray[slim] = -side
    _tri, idx_ray, loc = trimesh.ray.ray_triangle.RayMeshIntersector(tm).intersects_id(
        origins, np.tile([ray], (len(origins), 1)), return_locations=True, multiple_hits=False)
    pts = loc[idx_ray.argsort()]
    if len(pts) != num_divs[0] * num_divs[1]:
        raise RuntimeError(f"regenerated {len(pts)} tactile points on {body}, expected {num_divs[0] * num_divs[1]}")
    dev = sensor._tactile_pos_local.device
    sensor._tactile_pos_local = torch.tensor(pts, dtype=torch.float32, device=dev)
    sensor._tactile_pos_expanded = sensor._tactile_pos_local.unsqueeze(0).expand(sensor._num_envs, -1, -1)


def cam_pose_local(g: dict) -> tuple[tuple, tuple]:
    """Camera (USD/opengl: looks along -Z) inside the elastomer, cam_dist behind the apex, looking out of the pad."""
    slim, sign = g["slim"], g["sign"]
    pos = np.array(g["center"])
    pos[slim] = g["apex"] - sign * args.cam_dist
    e = np.eye(3)
    z_cam = -sign * e[slim]                         # camera -Z = outward pad normal
    others = [i for i in range(3) if i != slim]
    x_cam = e[others[0]]                            # image width along the first non-slim axis
    y_cam = np.cross(z_cam, x_cam)
    rot = torch.tensor(np.stack([x_cam, y_cam, z_cam], axis=1), dtype=torch.float32)
    quat = math_utils.quat_from_matrix(rot).tolist()
    return tuple(pos.tolist()), tuple(quat)


# ---------------------------------------------------------------- scene
scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, lazy_sensor_update=True)
scene_cfg.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
sp = base_cfg.spawn
compliant = args.compliant_stiffness > 0
scene_cfg.robot = base_cfg.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileWithCompliantContactCfg(
        usd_path=sp.usd_path,
        activate_contact_sensors=True,
        rigid_props=sp.rigid_props,
        articulation_props=sp.articulation_props,
        joint_drive_props=sp.joint_drive_props,
        compliant_contact_stiffness=args.compliant_stiffness if compliant else None,
        compliant_contact_damping=args.compliant_damping if compliant else None,
        physics_material_prim_path=list(pads) if (compliant and pads) else None,
    ),
)
BOX = (0.04, 0.04, 0.012)
SPHERE_R = 0.008
HALF = BOX[2] / 2 if args.object == "box" else SPHERE_R   # object centre -> face along the pad normal
_obj_kw = {"size": BOX} if args.object == "box" else {"radius": SPHERE_R}
scene_cfg.contact_object = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/contact_object",
    spawn=(sim_utils.MeshCuboidCfg if args.object == "box" else sim_utils.MeshSphereCfg)(
        **_obj_kw,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.35, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.6, 0.3)),
)
for p in pads:
    g = geo[p]
    cam_cfg = None
    if args.camera:
        pos, quat = cam_pose_local(g)
        h_ap = 1.0 * (cam_w * mmpp / 1000.0) / args.cam_dist
        cam_cfg = TiledCameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{p}/tactile_cam",
            offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=quat, convention="opengl"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=1.0, horizontal_aperture=h_ap,
                                             clipping_range=(1.0e-4, args.cam_dist)),
            data_types=["distance_to_image_plane"],
            height=cam_h,
            width=cam_w,
            depth_clipping_behavior="max",
        )
    setattr(scene_cfg, f"tac_{p}", VisuoTactileSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{p}/tactile_sensor",
        history_length=0,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=args.camera,
        enable_force_field=use_ff,
        tactile_array_size=tuple(args.array),
        tactile_margin=0.001,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
        normal_contact_stiffness=args.tactile_k,
        camera_cfg=cam_cfg,
    ))
    setattr(scene_cfg, f"cs_{p}", ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{p}", filter_prim_paths_expr=["{ENV_REGEX_NS}/contact_object"],
        track_friction_forces=True, max_contact_data_count_per_prim=64))

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    dt=1.0 / 120.0, device=args.device, physx=sim_utils.PhysxCfg(gpu_collision_stack_size=2**28)))
scene = InteractiveScene(scene_cfg)
stage = get_current_stage()

# SDF collision on the box mesh (TacSL force field queries the contact object's SDF)
box_mesh = None
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/contact_object")):
    if prim.IsA(UsdGeom.Mesh):
        box_mesh = prim
        break
UsdPhysics.CollisionAPI.Apply(box_mesh)
UsdPhysics.MeshCollisionAPI.Apply(box_mesh).CreateApproximationAttr().Set("sdf")
PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(box_mesh).CreateSdfResolutionAttr().Set(256)
print("BOX_MESH", box_mesh.GetPath(), "approx", UsdPhysics.MeshCollisionAPI(box_mesh).GetApproximationAttr().Get())

# which physics material does each elastomer collider really use?
mat_report = {}
for p in pads:
    col = None
    for d in Usd.PrimRange(stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{p}")):
        if d.HasAPI(UsdPhysics.CollisionAPI):
            col = d
            break
    mat, _rel = UsdShade.MaterialBindingAPI(col).ComputeBoundMaterial("physics")
    stiff = None
    if mat:
        a = mat.GetPrim().GetAttribute("physxMaterial:compliantContactStiffness")
        stiff = a.Get() if a and a.IsValid() else None
    mat_report[p] = {"collider": str(col.GetPath()), "material": str(mat.GetPath()) if mat else None,
                     "compliantContactStiffness": stiff}
print("ELASTOMER_MATERIALS", json.dumps(mat_report))

# hide the instrumented fingers' elastomer + distal-phalanx visuals from the in-pad cameras
if args.camera:
    for p in pads:
        for body in (p, p.replace("_elastomer", "_DP")):
            vis = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{body}/visuals")
            if vis.IsValid():
                UsdGeom.Imageable(vis).MakeInvisible()

sim.reset()
robot = scene["robot"]
box = scene["contact_object"]
dt = sim.get_physics_dt()
print("[smoke] joints:", robot.joint_names)


def step(n: int = 1, read: bool = False):
    for _ in range(n):
        scene.write_data_to_sim()
        sim.step(render=args.camera)
        scene.update(dt)
        if read:
            for p in pads:
                _ = scene[f"tac_{p}"].data


targets = robot.data.default_joint_pos.clone()
robot.set_joint_position_target(targets)
for _ in range(10):  # settle: the hand starts away from its home targets
    step(30)
    vmax = float(robot.data.joint_vel[0].abs().max())
    if vmax < 1e-3:
        break
print(f"[smoke] settled: max |joint vel| {vmax:.2e}")

tac = {p: scene[f"tac_{p}"] for p in pads}
if use_ff:
    for p in pads:
        if geo[p]["sign"] != geo[p]["tacsl_sign"]:
            regen_tactile_points(tac[p], p)
            print(f"[smoke] {p}: tactile points regenerated on side {geo[p]['sign']:+.0f} "
                  f"(TacSL chose {geo[p]['tacsl_sign']:+.0f})")
if args.camera:
    for _ in range(3):
        sim.render()
    for p in pads:
        nominal = tac[p].get_initial_render()
        d = nominal["distance_to_image_plane"]
        print(f"[smoke] nominal depth {p}: min {d.min().item():.5f} max {d.max().item():.5f}")

results = {"args": vars(args), "pad_geometry": geo, "materials": mat_report, "levels": []}
tp = tac[pads[0]] if pads else None
if tp is not None and use_ff:
    loc = tp._tactile_pos_local.cpu().numpy()
    dists = np.linalg.norm(loc[:, None] - loc[None], axis=-1) + np.eye(len(loc))
    cell = float(np.median(dists.min(1)))
    results["tactile_points"] = {"n": int(len(loc)), "cell_m": cell, "local_min": loc.min(0).tolist(),
                                 "local_max": loc.max(0).tolist()}
    print("[smoke] tactile points:", results["tactile_points"])


def pad_frame(p: str):
    bid = robot.find_bodies(p)[0][0]
    pos = robot.data.body_link_pos_w[0, bid]
    quat = robot.data.body_link_quat_w[0, bid]
    g = geo[p]
    e = torch.zeros(3, device=pos.device)
    e[g["slim"]] = g["sign"]
    apex_local = torch.tensor(g["center"], device=pos.device, dtype=torch.float32)
    apex_local[g["slim"]] = g["apex"]
    n_w = math_utils.quat_apply(quat, e)
    apex_w = pos + math_utils.quat_apply(quat, apex_local)
    return apex_w, n_w


def measure(p: str) -> dict:
    out = {}
    s = tac[p]
    if use_ff:
        d = s.data
        depth = d.penetration_depth[0]
        fn = d.tactile_normal_force[0].abs()   # sign depends on the pad side vs TacSL's fixed tactile frame
        fs = d.tactile_shear_force[0]
        n_in = int((depth > 0).sum())
        out.update({"ff_normal_sum": float(fn.sum()), "ff_normal_max": float(fn.max()),
                    "ff_kdepth_sum": float(depth.sum()) * args.tactile_k,
                    "ff_shear_sum": float(fs.norm(dim=-1).sum()), "ff_points_in_contact": n_in,
                    "ff_area_mm2": n_in * results.get("tactile_points", {}).get("cell_m", 0.0) ** 2 * 1e6,
                    "ff_depth_max_mm": float(depth.max()) * 1e3})
    if args.camera:
        d = s.data
        diff = s._nominal_tactile["distance_to_image_plane"][0, ..., 0] - d.tactile_depth_image[0, ..., 0]
        out.update({"cam_indent_max_mm": float(diff.max()) * 1e3, "cam_pixels_indented": int((diff > 1e-5).sum())})
    cs = scene[f"cs_{p}"].data
    nrm = cs.force_matrix_w[0].sum(dim=(0, 1))
    fr = torch.nan_to_num(cs.friction_forces_w[0]).sum(dim=(0, 1))
    out.update({"cs_normal_N": float(nrm.norm()), "cs_total_N": float((nrm + fr).norm())})
    return out


def save_images(tag: str, p: str):
    s = tac[p]
    if use_ff:
        r, c = args.array
        fn = s.data.tactile_normal_force[0].abs().view(r, c).cpu().numpy()
        fs = s.data.tactile_shear_force[0].view(r, c, 2).cpu().numpy()
        scale = max(float(fn.max()), 1e-9)
        img = compute_tactile_shear_image(fn / scale * 8e-5, fs / scale * 8e-5)
        cv2.imwrite(os.path.join(args.out, f"ff_{p}_{tag}.png"), (img * 255).astype(np.uint8))
    if args.camera:
        rgb = s.data.tactile_rgb_image[0].cpu().numpy()
        rgb = rgb.astype(np.uint8) if rgb.max() > 1.0 else (rgb * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out, f"rgb_{p}_{tag}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        diff = (s._nominal_tactile["distance_to_image_plane"][0, ..., 0] - s.data.tactile_depth_image[0, ..., 0])
        dimg = np.clip(diff.cpu().numpy() / 0.002, 0, 1) * 255
        cv2.imwrite(os.path.join(args.out, f"depth_{p}_{tag}.png"), dimg.astype(np.uint8))


step_times = []
if args.press and "right_index_elastomer" in pads:
    p = "right_index_elastomer"
    flex = [robot.joint_names.index(j) for j in ("right_index_MCP_FE", "right_index_PIP", "right_index_DIP")]
    home = targets.clone()
    # direction that drives the pad outward along its normal
    a0, n0 = pad_frame(p)
    palm = robot.data.body_link_pos_w[0, robot.find_bodies("right_hand_C_MC")[0][0]]
    to_palm = torch.nn.functional.normalize(palm - a0, dim=0)
    print(f"[smoke] pad normal (side {geo[p]['sign']:+.0f}) . direction to palm centre = {float(torch.dot(n0, to_palm)):+.3f}")
    t = home.clone()
    t[0, flex] += 0.15
    robot.set_joint_position_target(t)
    step(120)
    a1, _ = pad_frame(p)
    print(f"[smoke] probe: apex home {a0.tolist()} flexed {a1.tolist()} base_q {robot.data.joint_pos[0, :6].tolist()}")
    proj = float(torch.dot(a1 - a0, n0))
    direction = 1.0 if proj > 0 else -1.0
    print(f"[smoke] +0.15 rad flexion moves the pad apex {proj * 1e3:.2f} mm along the pad normal -> direction {direction}")
    robot.set_joint_position_target(home)
    step(150)
    apex_w, n_w = pad_frame(p)
    print(f"[smoke] back home: apex {apex_w.tolist()} (probe start {a0.tolist()})")
    center = apex_w + n_w * (args.gap + HALF)
    # box local +z -> n_w
    z = torch.tensor([0.0, 0.0, 1.0], device=n_w.device)
    axis = torch.linalg.cross(z, n_w)
    s_ = axis.norm()
    ang = torch.atan2(s_, torch.dot(z, n_w))
    q = math_utils.quat_from_angle_axis(ang.unsqueeze(0), (axis / s_.clamp(min=1e-9)).unsqueeze(0))[0]
    box.write_root_pose_to_sim(torch.cat([center, q]).unsqueeze(0))
    step(20)
    print(f"[smoke] box placed: wanted {center.tolist()} got {box.data.root_pos_w[0].tolist()}; apex {apex_w.tolist()} "
          f"normal {n_w.tolist()}")
    results["press"] = {"direction": direction, "flex_probe_mm": proj * 1e3, "apex_w": apex_w.tolist(),
                        "normal_dot_to_palm": float(torch.dot(n0, to_palm)),
                        "normal_w": n_w.tolist()}
    for lvl in range(args.levels + 1):
        t = home.clone()
        t[0, flex] += direction * lvl * args.delta
        robot.set_joint_position_target(t)
        acc = []
        for k in range(args.steps_per_level):
            t0 = time.perf_counter()
            step(1, read=True)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            if k >= args.steps_per_level - 10:
                acc.append(measure(p))
        row = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        row["level"] = lvl
        row["target_offset_rad"] = direction * lvl * args.delta
        row["q_err_rad"] = float((robot.data.joint_pos[0, flex] - t[0, flex]).abs().mean())
        ap, _n = pad_frame(p)
        row["apex_w"] = [round(v, 5) for v in ap.tolist()]
        row["base_q"] = [round(v, 4) for v in robot.data.joint_pos[0, :6].tolist()]
        row["flex_q"] = [round(v, 4) for v in robot.data.joint_pos[0, flex].tolist()]
        row["apex_past_box_face_mm"] = float(torch.dot(ap - (box.data.root_pos_w[0] - n_w * HALF), n_w)) * 1e3
        results["levels"].append(row)
        print("[level]", json.dumps(row))
        if lvl in (0, args.levels // 2, args.levels):
            save_images(f"L{lvl:02d}", p)

for _ in range(args.timing_steps):
    t0 = time.perf_counter()
    step(1, read=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    step_times.append(time.perf_counter() - t0)

if step_times:
    st = np.array(step_times[5:]) if len(step_times) > 10 else np.array(step_times)
    results["timing"] = {"steps": int(len(st)), "mean_ms": float(st.mean() * 1e3), "median_ms": float(np.median(st) * 1e3),
                         "p95_ms": float(np.percentile(st, 95) * 1e3)}
if args.device.startswith("cuda"):
    results["torch_max_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
try:
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20).stdout
    mine = [ln for ln in q.splitlines() if ln.split(",")[0].strip() == str(os.getpid())]
    results["nvidia_smi_used_MB"] = float(mine[0].split(",")[1]) if mine else None
except Exception as exc:  # noqa: BLE001
    results["nvidia_smi_used_MB"] = repr(exc)
print("RESULT", json.dumps({k: v for k, v in results.items() if k in ("timing", "torch_max_mem_MB", "nvidia_smi_used_MB")}))
with open(os.path.join(args.out, "result.json"), "w") as f:
    json.dump(results, f, indent=1)
print("TACSL_SMOKE_DONE", args.out, flush=True)
os._exit(0)  # app.close() can hang after a standalone scene; everything is written already
