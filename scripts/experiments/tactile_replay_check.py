# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics checks of the --tactile contact-force logging on a replayed trajectory pickle (experiment, not a tool).

Replays one episode of a pickle (record_demos --replay_demos style: reset, restore the initial state, feed the
recorded actions) with the pad sensors of ``dexverse.data_collection.tactile`` plus two diagnostic sensors:

* ``chk_all``: every robot body, filtered against the target object (normal + friction) -> the total
  hand->object contact force, compared with the object's Newton balance m (a + g z) to check the SIGN of the
  normal and friction parts (forces on the hand must sum to -m (a + g z) while only the hand touches the object);
* the task's own ``<tip>_object_s`` sensors (dummy-tip check: Shadow / Inspire / Sharpa ``*tip`` bodies have no
  collision, so they should read zero while the pad bodies do not).

    /workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/tactile_replay_check.py \
        --task Dexverse-PickCube-v0 --robot_type floating_allegro_right --pickle demo.pkl --out check.npz \
        --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--robot_type", required=True)
parser.add_argument("--pickle", required=True)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--tactile_targets", nargs="+", default=None)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", rendering_mode="performance")
args = parser.parse_args()
app = AppLauncher(vars(args)).app

import pickle  # noqa: E402

import dexverse  # noqa: E402
import dexverse.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dexverse.data_collection.tactile import TactileReader, add_tactile_sensors  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from dexverse.teleop_utils.episode_task_state import reset_managers_after_state_restore  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402

print("DEXVERSE_FILE", dexverse.__file__)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg = type(cfg)(robot_type=args.robot_type)
cfg.sim.device = args.device
cfg.scene.num_envs = 1
cfg.terminations.success = None
cfg.terminations.time_out = None
cfg.recorders = {}
tac = add_tactile_sensors(cfg, args.tactile_targets)
cfg.scene.chk_all = ContactSensorCfg(prim_path=f"{cfg.scene.robot.prim_path}/.*", filter_prim_paths_expr=tac["targets"],
                                     track_friction_forces=True, max_contact_data_count_per_prim=32)
tip_sensors = [f"{b}_object_s" for b in cfg.robot_config.fingertip_body_names
               if getattr(cfg.scene, f"{b}_object_s", None) is not None]

env = gym.make(args.task, cfg=cfg).unwrapped
reader = TactileReader(env, tac["bodies"], tac["targets"])
chk = env.scene.sensors["chk_all"]
print("chk_all bodies:", chk.num_bodies, "filter count", chk.contact_physx_view.filter_count)
obj = env.scene["object"] if "object" in env.scene.keys() else None

with open(args.pickle, "rb") as f:
    ep = [e for e in pickle.load(f)["episodes"] if len(e.get("actions", []))][args.episode]


def to_t(x):
    if isinstance(x, dict):
        return {k: to_t(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=env.device)
    return x


env.sim.reset()
env.reset()
env.reset_to(to_t(ep["initial_state"]), env_ids=None, is_relative=True)
reset_managers_after_state_restore(env)
mass = float(obj.root_physx_view.get_masses().sum()) if obj is not None else float("nan")
print("object mass", mass)
log = {k: [] for k in ("pad_normal_w", "pad_fric_w", "all_normal_w", "all_fric_w", "tip_force_w", "obj_pos", "obj_vel",
                       "pad_force_link", "net_normal_link")}
dt = env.step_dt
with torch.inference_mode():
    for t, a in enumerate(ep["actions"]):
        act = torch.as_tensor(np.asarray(a, dtype=np.float32).reshape(1, -1), device=env.device)
        env.step(act)
        r = reader.read_torch()
        log["pad_normal_w"].append(r["normal_w"].cpu().numpy())
        log["pad_fric_w"].append(r["friction_w"].cpu().numpy())
        log["pad_force_link"].append(r["force"].cpu().numpy())
        log["net_normal_link"].append(r["net_normal"].cpu().numpy())
        log["all_normal_w"].append(chk.data.force_matrix_w[0].sum(dim=(0, 1)).cpu().numpy())
        log["all_fric_w"].append(torch.nan_to_num(chk.data.friction_forces_w[0]).sum(dim=(0, 1)).cpu().numpy())
        log["tip_force_w"].append(np.stack([env.scene.sensors[s].data.force_matrix_w[0].sum(dim=(0, 1)).cpu().numpy()
                                            for s in tip_sensors]) if tip_sensors else np.zeros((0, 3)))
        if obj is not None:
            log["obj_pos"].append(obj.data.root_pos_w[0].cpu().numpy())
            log["obj_vel"].append(obj.data.root_lin_vel_w[0].cpu().numpy())
arr = {k: np.asarray(v) for k, v in log.items() if len(v)}
np.savez(args.out, mass=mass, dt=dt, bodies=np.array(tac["bodies"]), tip_sensors=np.array(tip_sensors), **arr)

# ---- report
pn, pf = arr["pad_normal_w"], arr["pad_fric_w"]
an, af = arr["all_normal_w"], arr["all_fric_w"]
pad_mag = np.linalg.norm(pn + pf, axis=-1)
print("pad bodies:", tac["bodies"])
print("pad |F| max per body:", pad_mag.max(0).round(3), " contact frames:", (pad_mag > 1e-3).sum(0))
if tip_sensors:
    tip_mag = np.linalg.norm(arr["tip_force_w"], axis=-1)
    print("tip sensors:", tip_sensors)
    print("tip |F| max per body:", tip_mag.max(0).round(3), " contact frames:", (tip_mag > 1e-3).sum(0))
if obj is not None:
    p, v = arr["obj_pos"], arr["obj_vel"]
    acc = np.gradient(v, dt, axis=0)
    expected = -mass * (acc + np.array([0.0, 0.0, 9.81]))      # total force ON the hand if only the hand holds it
    lifted = p[:, 2] > p[0, 2] + 0.03
    print(f"object z0 {p[0, 2]:.3f}, max z {p[:, 2].max():.3f}, lifted frames {lifted.sum()}")
    if lifted.any():
        idx = np.where(lifted)[0]
        for name, tot in (("normal+friction", an + af), ("normal-friction", an - af), ("normal only", an)):
            err = np.linalg.norm(tot[idx] - expected[idx], axis=-1)
            print(f"  hand->object balance using {name:16s}: median |sum - expected| {np.median(err):7.3f} N "
                  f"(median sum z {np.median(tot[idx, 2]):7.3f}, expected z {np.median(expected[idx, 2]):7.3f})")
        print(f"  pads share of total |F| (lifted frames): {np.median(np.linalg.norm((pn + pf).sum(1)[idx], axis=-1)) :.3f} "
              f"vs all-body {np.median(np.linalg.norm((an + af)[idx], axis=-1)):.3f}")
print("TACTILE_CHECK_DONE")
env.close()
app.close()
