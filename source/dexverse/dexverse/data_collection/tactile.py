# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-finger contact-force ("tactile") logging for the floating dexterous hands.

No scene sensor is added and no env config is changed: after the env is built, :class:`RawContactView` creates its
own PhysX tensor-API rigid-contact view (``create_rigid_contact_view``) over the hand bodies, filtered against the
task's target object(s). Read-only: the view only queries PhysX contact reports (every hand body already carries
``PhysxContactReportAPI`` because the hand spawn configs set ``activate_contact_sensors=True``), so it does not
change the simulation. The task's own contact sensors and observations are untouched.

Pad body per finger (the body carrying the fingertip COLLISION geometry): :func:`tactile_body_names` -- the
fingertip marker of ``robot_config.fingertip_body_names`` mapped by :data:`TACTILE_BODY_RULES` (Shadow ``*tip`` ->
``*distal``, Inspire ``*_tip`` -> ``thumb_distal`` / ``*_intermediate``, Sharpa ``*_fingertip`` -> ``*_elastomer``;
other hands: the fingertip body itself).

Per finger and frame :class:`TactileReader` yields one 3-vector per quantity -- the same "one resultant force per
pad" shape as the RH5DG2 hardware (8 pads x 3 axes), with no spatial (taxel) resolution:

* ``force``       normal + friction force exerted ON the pad body BY the target object(s), in the pad body's
                  link frame (N). Sum over all contact points and all matched target bodies.
* ``normal``      the normal part of ``force`` only (link frame, N).
* ``force_w``     ``force`` in the world frame (N).
* ``net_normal``  normal force from ALL contacts of the pad body (object, table, other fingers, ...), link frame,
                  N; friction is not available unfiltered in PhysX.

PhysX reports contacts per physics substep (``force = impulse / physics_dt``). The LeRobot recorder
(``dexverse.data_collection.tactile_logging``) averages them over the substeps of the control step by default.
"""

from __future__ import annotations

import re

import numpy as np

TACTILE_QUANTITIES = ("force", "normal", "force_w", "net_normal")
TACTILE_FEATURE_PREFIX = "observation.tactile."

TACTILE_DEFINITIONS = {
    "observation.tactile.force": "per pad body [fx,fy,fz]: normal + friction contact force exerted ON the pad body "
                                 "BY the target object(s), summed over contact points and target bodies, expressed "
                                 "in the pad body's link frame (USD body frame), newtons",
    "observation.tactile.normal": "per pad body [fx,fy,fz]: normal part of observation.tactile.force (link frame, N)",
    "observation.tactile.force_w": "per pad body [fx,fy,fz]: observation.tactile.force in the world frame (N)",
    "observation.tactile.net_normal": "per pad body [fx,fy,fz]: normal contact force from ALL contacts of the pad "
                                      "body (target object, table, other links, ...), link frame, N",
    "timing": "PhysX contact solves of the physics substeps of the control step that led to the frame (force = "
              "impulse / physics_dt); reduced over those substeps as given by tactile_reduce (mean = average of "
              "the substeps, last = last substep only); captured with observation.state before the action of the "
              "same frame",
    "first_frame": "frame 0 of every episode (and any frame right after a reset / state restore) is all zeros: no "
                   "contact has been solved since the reset, PhysX would report the pre-reset contacts",
    "sign": "force acting on the pad (reaction of the object on the finger): pressing a pad against an object gives "
            "a force pointing from the object into the pad",
    "source": "PhysX tensor-API rigid contact view created by the recorder (no scene sensor): contact force matrix "
              "(normal), friction rows summed per pad / target (friction), net contact forces (net normal); link "
              "orientation = robot.data.body_link_quat_w",
}

# fingertip marker -> pad body (collision-bearing): (regex, replacement) applied to robot_config.fingertip_body_names
TACTILE_BODY_RULES = {
    "shadow": [(r"(th|ff|mf|rf|lf)tip$", r"\1distal")],
    "inspire": [(r"thumb_tip$", "thumb_distal"), (r"(index|middle|ring|pinky)_tip$", r"\1_intermediate")],
    "sharpa": [(r"_fingertip$", "_elastomer")],
}


def tactile_feature_names(bodies: list[str]) -> list[str]:
    return [f"{b}.{a}" for b in bodies for a in ("fx", "fy", "fz")]


def tactile_body_names(robot_type: str, fingertip_body_names: list[str], body_names: list[str]) -> list[str]:
    """Pad body per finger (same order as ``fingertip_body_names``) by :data:`TACTILE_BODY_RULES`."""
    rules = next((r for k, r in TACTILE_BODY_RULES.items() if k in str(robot_type).lower()), [])
    out = []
    for n in fingertip_body_names:
        m = n
        for pat, rep in rules:
            m = re.sub(pat, rep, m)
        if m not in body_names:
            raise RuntimeError(f"[tactile] pad body {m!r} (from fingertip {n!r}, {robot_type}) is not a robot body")
        out.append(m)
    return out


def articulation_link_prims(art_cfg) -> list[str]:
    """One prim path per rigid link of an articulation spawned from a USD file (``<prim_path>/<link path>``).

    PhysX contact filters need ONE pattern per filtered body (a pattern matching several bodies is rejected with
    "did not match the correct number of entries"), so an articulation is expanded link by link from its USD.
    """
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    usd_path = getattr(getattr(art_cfg, "spawn", None), "usd_path", None)
    if not usd_path:
        return []
    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    if not root:
        return []
    root_path = root.GetPath()
    links = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            links.append(f"{art_cfg.prim_path}/{prim.GetPath().MakeRelativePath(root_path).pathString}")
    return links


def default_target_prims(env_cfg) -> list[str]:
    """The task's manipulated object: ``scene.object``'s prim path (grasping / functional tasks), else every rigid
    link of ``scene.articulation`` (articulation tasks, one path per link, see :func:`articulation_link_prims`)."""
    obj = getattr(env_cfg.scene, "object", None)
    path = getattr(obj, "prim_path", None) if obj is not None else None
    if path:
        return [path]
    art = getattr(env_cfg.scene, "articulation", None)
    return articulation_link_prims(art) if art is not None else []


def resolve_targets(env_cfg, target_prims: list[str] | None = None) -> list[str]:
    """Contact partners: the given expressions (bare names get ``{ENV_REGEX_NS}/``) or :func:`default_target_prims`."""
    targets = list(target_prims or default_target_prims(env_cfg))
    if not targets:
        raise RuntimeError("[tactile] the task has no scene.object / scene.articulation (no contact target)")
    return [t if (t.startswith("{ENV_REGEX_NS}") or t.startswith("/")) else "{ENV_REGEX_NS}/" + t for t in targets]


def add_tactile_sensors(env_cfg, target_prims: list[str] | None = None) -> dict:
    """Compatibility helper for the experiment scripts: adds NOTHING to ``env_cfg`` (the readers open their own
    contact views after the env is built); returns ``{"bodies", "targets"}`` for :class:`TactileReader`."""
    rc = env_cfg.robot_config
    rules = next((r for k, r in TACTILE_BODY_RULES.items() if k in str(getattr(env_cfg, "robot_type", "")).lower()), [])
    bodies = []
    for n in rc.fingertip_body_names:
        for pat, rep in rules:
            n = re.sub(pat, rep, n)
        bodies.append(n)
    return {"bodies": bodies, "targets": resolve_targets(env_cfg, target_prims)}


def _glob(expr: str) -> str:
    return expr.replace("{ENV_REGEX_NS}", "/World/envs/env_.*").replace(".*", "*")


def _np(x):
    return np.array(x.detach().to("cpu").numpy() if hasattr(x, "detach") else x, dtype=np.float64, copy=True)


class RawContactView:
    """PhysX rigid-contact view over robot bodies, filtered against target bodies, created AFTER the env is built
    (no scene sensor, no config change). Env 0. Recreated automatically when PhysX invalidates it (sim reset)."""

    def __init__(self, env, bodies: list[str], targets: list[str], max_contacts_per_body: int = 64):
        self.env = env
        self.bodies = list(bodies)
        self.targets = list(targets)
        self.max_contacts = int(max_contacts_per_body) * max(len(self.bodies), 1)
        robot_prim = env.scene["robot"].cfg.prim_path
        self.pattern = _glob(f"{robot_prim}/(" + "|".join(self.bodies) + ")")
        self.filters = [_glob(t) for t in self.targets]
        self.dt = float(env.physics_dt)
        self.n_created = 0
        self._create()

    def _create(self):
        from isaacsim.core.simulation_manager import SimulationManager  # noqa: PLC0415

        sv = SimulationManager.get_physics_sim_view()
        self.view = sv.create_rigid_contact_view(self.pattern, filter_patterns=self.filters,
                                                 max_contact_data_count=self.max_contacts)
        names = [p.rsplit("/", 1)[-1] for p in self.view.sensor_paths]
        if sorted(names) != sorted(self.bodies) or self.view.sensor_count != len(self.bodies):
            raise RuntimeError(f"[tactile] contact view resolved {names} for bodies {self.bodies} ({self.pattern})")
        if self.view.filter_count != len(self.filters):
            raise RuntimeError(f"[tactile] contact view: {self.view.filter_count} filter bodies for the "
                               f"{len(self.filters)} target expressions {self.targets}; each must match exactly one "
                               "rigid body (see the PhysX 'did not match the correct number of entries' error)")
        self.names = names                                       # row order of every buffer
        self.n_created += 1

    def ensure(self):
        if not self.view.check():
            self._create()

    def read(self, raw: bool = False) -> dict:
        """See :meth:`_read`; a view left over from before a simulation reset (stop / play) is re-created once."""
        try:
            return self._read(raw)
        except Exception:  # noqa: BLE001 -- PhysX raises a bare Exception for a stale view
            self._create()
            return self._read(raw)

    def _read(self, raw: bool = False) -> dict:
        """Deep copies (float64) of: net (B, 3), fmat (B, F, 3) normal force matrix, fric (B, F, 3) summed friction
        rows; with ``raw`` also the contact rows (get_contact_data) and friction rows (get_friction_data). The two
        row calls share the view's count / start buffers, so each result is copied right after its call."""
        self.ensure()
        v, dt = self.view, self.dt
        B, F = len(self.names), max(v.filter_count, 1)
        out = {"net": _np(v.get_net_contact_forces(dt)).reshape(B, 3),
               "fmat": _np(v.get_contact_force_matrix(dt)).reshape(B, F, 3)}
        cdat = tuple(_np(x) for x in v.get_contact_data(dt))     # forces, points, normals, separations, count, start
        fdat = tuple(_np(x) for x in v.get_friction_data(dt))    # forces, points, count, start
        ff, fc, fs = fdat[0].reshape(-1, 3), fdat[2].reshape(B, F).astype(int), fdat[3].reshape(B, F).astype(int)
        fric = np.zeros((B, F, 3))
        for b in range(B):
            for m in range(F):
                if fc[b, m]:
                    fric[b, m] = np.nan_to_num(ff[fs[b, m]:fs[b, m] + fc[b, m]]).sum(0)
        out["fric"] = fric
        if raw:
            out["contact"], out["friction"] = cdat, fdat
        return out


class TactileReader:
    """Per-pad-body forces (see module docstring) from a :class:`RawContactView` over the pad bodies, env 0."""

    def __init__(self, env, bodies: list[str], targets: list[str]):
        self.bodies = list(bodies)
        self.targets = list(targets)
        self.robot = env.scene["robot"]
        self.view = RawContactView(env, self.bodies, self.targets)
        self.body_ids = [self.robot.body_names.index(b) for b in self.bodies]

    @property
    def num_bodies(self) -> int:
        return len(self.bodies)

    def zeros(self) -> dict[str, np.ndarray]:
        return {q: np.zeros((self.num_bodies, 3), dtype=np.float32) for q in TACTILE_QUANTITIES}

    def read64(self) -> dict[str, np.ndarray]:
        from dexverse.data_collection.tactile_pads import _qmat  # noqa: PLC0415

        d = self.view.read()
        row = [self.view.names.index(b) for b in self.bodies]      # view row of each pad body (after any re-create)
        normal_w = d["fmat"][row].sum(1)
        fric_w = d["fric"][row].sum(1)
        net_w = d["net"][row]
        R = _qmat(_np(self.robot.data.body_link_quat_w[0, self.body_ids]))      # (n, 3, 3) link -> world
        force_w = normal_w + fric_w
        return {"force": np.einsum("nji,nj->ni", R, force_w), "normal": np.einsum("nji,nj->ni", R, normal_w),
                "force_w": force_w, "net_normal": np.einsum("nji,nj->ni", R, net_w)}

    def read(self) -> dict[str, np.ndarray]:
        return {q: v.astype(np.float32) for q, v in self.read64().items()}
