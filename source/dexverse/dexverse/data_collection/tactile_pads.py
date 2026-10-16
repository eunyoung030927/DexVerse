# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Virtual fingerprint pads: split each distal segment's contact force into pad / non-pad by contact LOCATION.

Physics is untouched: no collider, USD or env-config change. One read-only PhysX rigid-contact view
(:class:`~dexverse.data_collection.tactile.RawContactView`, created after the env is built) covers every link of
every finger's "distal bundle" (the distal phalanx plus the links fixed to it, e.g. Allegro ``link_N_0`` +
``link_N_0_tip``, Sharpa ``*_DP`` + ``*_elastomer``) and the palm bundle, filtered against the task's target
object(s).

The pad region of each hand is a file, ``tactile_pads/<robot_type>.npz`` (written by
``scripts/data_tools/make_tactile_pads.py``): for every bundle link a set of surface samples of its collision
shapes in the LINK frame with a pad / non-pad label. Definition (agreed with the user): the pad is the whole
palmar "fingerprint" face of the distal segment, from the base of the finger pad to the tip, on the bundle surface;
back and sides are non-pad.

Per physics substep (the LeRobot recorder averages the substeps of a control step by default, see
``tactile_logging``; ``read()`` is one substep):

1. the raw PhysX contact rows (``get_contact_data``) and friction rows (``get_friction_data``) are read and
   deep-copied right after each call -- both calls write the SAME count / start-index buffers of the view;
2. rows carrying no force (speculative contacts) are dropped;
3. contact points are produced BEFORE the substep integrates the bodies, so each point is mapped into its link
   frame with the link poses of the end of the PREVIOUS substep: the reader caches the link poses after every
   ``scene.update`` (read-only wrapper); without a cached pose (first step after a reset with decimation 1) the
   current pose is integrated one physics step backwards (``x - v dt``, ``exp(-w dt) R`` of the link COM);
4. a normal-force row is pad if its point's nearest surface sample of that link is a pad sample (nearest-surface
   rule); friction rows (PhysX friction-patch anchor points, not at the current contact) are split by the pad
   fraction of the normal force of their sensor-filter pair;
5. per finger: bundle total, pad and non-pad sums of the rows (summed in a canonical row order: PhysX's buffer order
   varies between identical runs), force-weighted pad contact centre; pad + non-pad is checked every frame against
   the aggregated sensor (force matrix + friction) of the bundle.

PALM ZONES (definition files written by ``scripts/data_tools/make_tactile_palm_pads.py``): the palmar face of each
palm bundle (palm body + the links fixed to it that have colliders; surface samples whose outward normal is within
60 deg of the palm's palmar normal) is cut into three equal-width zones along the thumb <-> little-finger axis like
the RH5DG2 palm sensor: ``palm_l`` = thumb side, ``palm_c`` = centre, ``palm_r`` = little-finger side (anatomical:
the same for a left hand, i.e. mirrored). Palm contact rows are labelled by the nearest palm sample (zone or
non-pad: back / sides / non-palmar faces) exactly like the finger pads; friction rows follow the zone shares of
their pair's normal force.

REAL-HAND FORMAT (``observation.touch``, only when the definition has palm zones): per hand 8 pads x (fn, ftx, fty)
in the RH5DG2 driver order little, ring, middle, index, thumb, palm_r, palm_c, palm_l (N, float32; bimanual: right
hand then left hand). fn = compressive normal force (positive = pressed, = -F.n with n the pad's outward normal),
(ftx, fty) = tangential force in the pad plane with hand-common axes: ftx = distal (toward the fingertip / the
fingers for the palm), fty = toward the thumb side (thumb pad: see the definition json, ``touch.fty_rule``). F is
the force ON the pad BY the object. Missing fingers (4-finger hands) are zero with ``observation.touch_valid`` = 0;
``observation.touch_contact`` = fn > ``TOUCH_CONTACT_N``. Simulation has no tare: raw = tared.
"""

from __future__ import annotations

import os

import numpy as np

PAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tactile_pads")
PAD_FEATURES = ("total", "pad", "nonpad", "pad_centroid", "pad_w", "nonpad_w")
PAD_FEATURE_PREFIX = "observation.tactile.pad."
PALM_FEATURE = "observation.tactile.palm"
TOUCH_FEATURE = "observation.touch"
TOUCH_VALID_FEATURE = "observation.touch_valid"
TOUCH_CONTACT_FEATURE = "observation.touch_contact"
TOUCH_PADS = ("little", "ring", "middle", "index", "thumb", "palm_r", "palm_c", "palm_l")   # RH5DG2 driver order
TOUCH_COMPONENTS = ("fn", "ftx", "fty")
PALM_ZONES = ("palm_l", "palm_c", "palm_r")            # zone ids 1, 2, 3 of the definition file (0 = non-pad)
TOUCH_CONTACT_N = 0.1

PAD_DEFINITIONS = {
    "observation.tactile.pad.total": "per finger [fx,fy,fz]: normal + friction force ON the finger's distal bundle "
                                     "(distal phalanx + links fixed to it) BY the target, in the DISTAL PHALANX link "
                                     "frame (N); sum of all force-carrying rows (= aggregated sensor value, checked)",
    "observation.tactile.pad.pad": "per finger: part of the bundle force whose contact points lie on the virtual "
                                   "fingerprint pad (distal phalanx link frame, N)",
    "observation.tactile.pad.nonpad": "per finger: the rest (back / sides / joint side), distal phalanx frame, N; "
                                      "pad + nonpad = total up to float error (checked per frame, see episode meta)",
    "observation.tactile.pad.pad_centroid": "per finger [x,y,z]: |normal force|-weighted centre of the pad contact "
                                            "points in the distal phalanx link frame (m); zeros when the pad has no "
                                            "contact",
    "observation.tactile.pad.pad_w": "per finger: observation.tactile.pad.pad in the world frame (N)",
    "observation.tactile.pad.nonpad_w": "per finger: observation.tactile.pad.nonpad in the world frame (N)",
    "observation.tactile.palm": "per palm body [fx,fy,fz]: normal + friction force ON the palm link BY the target, "
                                "palm link frame (N)",
    "classification": "normal-force contact point -> nearest surface sample of its own link (collision shapes, link "
                      "frame); pad if that sample is a pad sample; friction rows (patch anchor points, not at the "
                      "contact) are split by the pad fraction of their pair's normal force. Points are placed with the link pose of the end of the previous physics "
                      "substep (contacts are generated before the substep integrates); rows without force dropped",
    "frames": "force vectors use the link orientation at the end of the substep (as observation.tactile.force); "
              "the pad centre uses the pose before the substep (where the contact was generated)",
    "reduce": "per frame the substeps of the control step that led to it are averaged (tactile_reduce = mean; "
              "pad_centroid = force-weighted over all those substeps) or the last substep is kept (last)",
    "first_frame": "zeros after a reset / state restore (no contact solved yet)",
}

TOUCH_DEFINITIONS = {
    "observation.touch": "per hand 8 pads x (fn, ftx, fty), pad order little, ring, middle, index, thumb, palm_r, "
                         "palm_c, palm_l (= RH5DG2 driver TOUCH_PADS order), float32, N; bimanual: right hand block "
                         "then left hand block",
    "fn": "compressive normal force on the pad, positive when pressed: -F . n, n = pad outward normal (finger: the "
          "distal frame's palmar direction h; palm: the palm's palmar normal); not clipped",
    "ftx": "tangential force along the hand-common DISTAL axis: finger pad = along the finger toward its tip; palm "
           "zone = from the wrist toward the fingers",
    "fty": "tangential force along the hand-common THUMB-SIDE axis (perpendicular to n and ftx): non-thumb fingers "
           "and palm zones = toward the thumb side (little -> index); thumb pad = see touch.fty_rule in the "
           "definition json (the thumb has no thumb side)",
    "F": "force ON the pad BY the object (normal + friction rows of the pad region), the pad's link orientation at "
         "the end of the control step (same as observation.tactile.pad.pad)",
    "finger_pads": "virtual fingerprint pad of observation.tactile.pad.pad (distal bundle, 60 deg palmar)",
    "palm_zones": "palm_l = thumb side, palm_c = centre, palm_r = little-finger side, equal thirds of the palmar face "
                  "width along the thumb-little axis; anatomical for a left hand (mirrored); the real RH5DG2 "
                  "left/right zone naming is NOT verified on hardware (assumption, from the robosynapse touch_layout "
                  "palm_l next to the thumb in the facing view)",
    "instant": "contacts of the physics substeps of the env.step that led to the frame, averaged (tactile_reduce "
               "mean, default; the pad forces are averaged and touch is computed from them) or the last substep "
               "(last); same as observation.tactile.*, aligned with the frame's images / state",
    "tare": "simulation has no sensor offset: raw = tared values (the real export tares before writing)",
    "observation.touch_valid": "uint8 per pad slot: 1 = the hand has this pad, 0 = slot filled with zeros "
                               "(4-finger hands: little; no palm zones defined)",
    "observation.touch_contact": "uint8 per pad slot: fn > TOUCH_CONTACT_N (see touch.contact_threshold_N)",
    "real_correspondence": "real RH5DG2 export (observation.touch [24], no names) stores per pad normal + tangential "
                           "magnitude + direction from the driver's raw registers (0.01 N); each pad has its own "
                           "raw direction convention -- fingertip pads: 0 deg = distal, counter-clockwise toward the "
                           "thumb side; palm_l / palm_c: 0 deg = thumb side, clockwise toward distal; palm_r: a "
                           "mirror image with only the distal-proximal axis flipped (0 deg = thumb side, 90 deg = "
                           "proximal), NOT a 180 deg rotation (robosynapse profile.py touch_layout dir_off / "
                           "dir_mirror; back-computed from a display calibration, to be re-verified on the "
                           "hardware) -- so the REAL export must map its (tangential, direction) into these "
                           "hand-common axes before sim and real can be compared component-wise",
}



# ----------------------------------------------------------------------------------- definition file
def pad_definition_path(robot_type: str) -> str:
    return os.path.join(PAD_DIR, f"{robot_type}.npz")


def load_pad_definition(robot_type: str, variant: str = "pad") -> dict:
    """Pad definition of a robot type. ``variant`` selects the sample mask (``pad`` = fingerprint pad; a file may
    carry extra masks, e.g. ``elastomer`` for Sharpa = the elastomer link surface)."""
    path = pad_definition_path(robot_type)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[tactile_pads] no pad definition for {robot_type!r} ({path}); create it with "
                                "scripts/data_tools/make_tactile_pads.py")
    z = np.load(path, allow_pickle=False)
    fingers = [str(x) for x in z["fingers"]]
    distal = [str(x) for x in z["distal"]]
    links = [str(x) for x in z["links"]]
    link_finger = z["link_finger"].astype(int)
    palms = [str(x) for x in z["palms"]]
    samples, masks = {}, {}
    for i, ln in enumerate(links):
        samples[ln] = z[f"samples_{i}"].astype(np.float64)
        key = f"mask_{variant}_{i}"
        if key not in z.files:
            raise KeyError(f"[tactile_pads] {path} has no '{variant}' mask (have "
                           f"{sorted({k.split('_')[1] for k in z.files if k.startswith('mask_')})})")
        masks[ln] = z[key].astype(bool)
    d = {"robot_type": robot_type, "variant": variant, "fingers": fingers, "distal": distal, "links": links,
         "link_finger": link_finger, "palms": palms, "samples": samples, "masks": masks, "path": path,
         "palm_links": list(palms), "palm_link_palm": list(range(len(palms))), "palm_samples": {}, "palm_zone": {},
         "touch": None}
    if "palm_links" in z.files:       # palm zones + real-hand touch layout (make_tactile_palm_pads.py)
        d["palm_links"] = [str(x) for x in z["palm_links"]]
        d["palm_link_palm"] = [int(x) for x in z["palm_link_palm"]]
        for j, ln in enumerate(d["palm_links"]):
            d["palm_samples"][ln] = z[f"palm_samples_{j}"].astype(np.float64)
            d["palm_zone"][ln] = z[f"palm_zone_{j}"].astype(np.int64)
        d["touch"] = {"finger_canon": [str(x) for x in z["finger_canon"]],
                      "finger_hand": [int(x) for x in z["finger_hand"]],
                      "hand_names": [str(x) for x in z["hand_names"]],
                      "finger_axes": [z[f"touch_axes_{i}"].astype(np.float64) for i in range(len(fingers))],
                      "palm_axes": [z[f"palm_frame_{p}"].astype(np.float64) for p in range(len(palms))]}
    return d


def add_tactile_pad_sensor(env_cfg, targets: list[str], robot_type: str, **_kw) -> dict:
    """Compatibility helper for the experiment scripts: adds NOTHING to ``env_cfg`` (:class:`TactilePadReader`
    opens its own contact view after the env is built)."""
    load_pad_definition(robot_type)
    return {"robot_type": robot_type, "targets": list(targets)}


# ----------------------------------------------------------------------------------- small rotation helpers
def _qmat(q):
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _rotvec_mat(r):
    """Rodrigues: rotation matrices of rotation vectors (..., 3)."""
    th = np.linalg.norm(r, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, r / np.maximum(th, 1e-12), 0.0)
    K = np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _np(x):
    return np.array(x.detach().to("cpu").numpy() if hasattr(x, "detach") else x, dtype=np.float64, copy=True)


class TactilePadReader:
    """Pad / palm-zone split of the bundle contacts for env 0 (see module docstring)."""

    def __init__(self, env, robot_type: str, targets: list[str], variant: str = "pad",
                 cache_substep_poses: bool = True):
        from dexverse.data_collection.tactile import RawContactView  # noqa: PLC0415

        self.env = env
        self.d = load_pad_definition(robot_type, variant)
        self.targets = list(targets)
        self.robot = env.scene["robot"]
        self.dt = float(env.physics_dt)
        self.view = RawContactView(env, list(self.d["links"]) + list(self.d["palm_links"]), self.targets)
        self.fingers = self.d["fingers"]
        nf = len(self.fingers)
        self.link_finger = {ln: int(f) for ln, f in zip(self.d["links"], self.d["link_finger"])}
        self._plp = dict(zip(self.d["palm_links"], self.d["palm_link_palm"]))
        self._map_rows()
        self.distal_rob = np.array([self.robot.body_names.index(n) for n in self.d["distal"]])
        self.palm_rob = np.array([self.robot.body_names.index(n) for n in self.d["palms"]])
        self.nf, self.np_ = nf, len(self.d["palms"])
        self.zones_on = bool(self.d["palm_zone"])
        self.touch_on = self.d["touch"] is not None
        if self.touch_on:
            self._init_touch()
        self.last_diag = {}
        self.last_raw = {}
        self._fric_anchor_mm = []        # distance friction anchor -> nearest normal contact point (diagnostic)
        self._cache = []
        if cache_substep_poses:
            self._install_pose_cache()

    def _map_rows(self):
        """Buffer row order of the contact view -> robot body / finger / palm (again after a view re-create)."""
        names = list(self.view.names)
        self.sens_names = names
        self.rob_idx = np.array([self.robot.body_names.index(n) for n in names])
        self.sens_finger = np.array([self.link_finger.get(n, -1) for n in names])
        self.sens_palm = np.array([self._plp.get(n, -1) for n in names])
        self._rows_version = self.view.n_created

    # ------------------------------------------------------------------ substep pose cache
    def _install_pose_cache(self):
        scene = self.env.scene
        if getattr(scene, "_tactile_pad_pose_hooks", None) is None:
            orig = scene.update
            scene._tactile_pad_pose_hooks = []

            def update(dt, _orig=orig, _scene=scene):
                _orig(dt)
                for fn in _scene._tactile_pad_pose_hooks:
                    fn()

            scene.update = update
        scene._tactile_pad_pose_hooks.append(self._on_substep)

    def _on_substep(self):
        rd = self.robot.data
        self._cache.append((_np(rd.body_link_pos_w[0]), _qmat(_np(rd.body_link_quat_w[0]))))
        if len(self._cache) > 2:
            self._cache.pop(0)

    def mark_reset(self):
        """Call after anything that moves the scene outside env.step (reset, state restore)."""
        self._cache = []

    def zeros(self) -> dict:
        out = {q: np.zeros((self.nf, 3), np.float32) for q in PAD_FEATURES}
        out["palm"] = np.zeros((self.np_, 3), np.float32)
        if self.zones_on:
            out["palm_zone"] = np.zeros((self.np_, 3, 3), np.float32)
            out["palm_nonpad"] = np.zeros((self.np_, 3), np.float32)
        if self.touch_on:
            out["touch"] = np.zeros(self.n_hands * 24, np.float32)
            out["touch_valid"] = self.touch_valid.copy()
            out["touch_contact"] = np.zeros(self.n_hands * 8, np.uint8)
        return out

    # ------------------------------------------------------------------ real-hand touch layout
    def _init_touch(self):
        t = self.d["touch"]
        self.n_hands = len(t["hand_names"])
        self.touch_valid = np.zeros(self.n_hands * 8, np.uint8)
        self.touch_finger_slot = {}
        for i, (c, h) in enumerate(zip(t["finger_canon"], t["finger_hand"])):
            slot = h * 8 + TOUCH_PADS.index(c)
            if self.touch_valid[slot]:
                raise RuntimeError(f"[tactile_pads] two fingers map to touch slot {c} of hand {h}")
            self.touch_finger_slot[i] = slot
            self.touch_valid[slot] = 1
        for p in range(self.np_):
            hand = p if self.n_hands > 1 else 0
            for zi, zn in enumerate(PALM_ZONES):
                if any((self.d["palm_zone"][ln] == zi + 1).any() for ln, pp in
                       zip(self.d["palm_links"], self.d["palm_link_palm"]) if pp == p):
                    self.touch_valid[hand * 8 + TOUCH_PADS.index(zn)] = 1

    def touch_names(self) -> list[str]:
        """``observation.touch`` column names: [<hand>.]<pad>.<fn|ftx|fty> (hand prefix only when bimanual)."""
        hn = self.d["touch"]["hand_names"]
        pre = [f"{h}." for h in hn] if len(hn) > 1 else [""]
        return [f"{p}{pad}.{c}" for p in pre for pad in TOUCH_PADS for c in TOUCH_COMPONENTS]

    def touch_slot_names(self) -> list[str]:
        hn = self.d["touch"]["hand_names"]
        pre = [f"{h}." for h in hn] if len(hn) > 1 else [""]
        return [f"{p}{pad}" for p in pre for pad in TOUCH_PADS]

    def _touch(self, pad_link: np.ndarray, palm_zone):
        """finger pad forces (distal frames) + palm zone forces (palm frames) -> (touch, contact)."""
        t = self.d["touch"]
        vec = np.zeros((self.n_hands * 8, 3))
        for i, slot in self.touch_finger_slot.items():
            u, h, w = t["finger_axes"][i]
            F = pad_link[i]
            vec[slot] = (-F @ h, F @ u, F @ w)
        if palm_zone is not None:
            for p in range(self.np_):
                hand = p if self.n_hands > 1 else 0
                _c, u, h, w = t["palm_axes"][p]
                for zi, zn in enumerate(PALM_ZONES):
                    F = palm_zone[p, zi]
                    vec[hand * 8 + TOUCH_PADS.index(zn)] = (-F @ h, F @ u, F @ w)
        vec[self.touch_valid == 0] = 0.0
        contact = ((vec[:, 0] > TOUCH_CONTACT_N) & (self.touch_valid > 0)).astype(np.uint8)
        return vec.reshape(-1).astype(np.float32), contact

    # ------------------------------------------------------------------ geometry
    def _prev_link_poses(self):
        """Link poses one physics step back (backward semi-implicit Euler of the COM state)."""
        rd = self.robot.data
        lp, lq = _np(rd.body_link_pos_w[0]), _np(rd.body_link_quat_w[0])
        cp = _np(rd.body_com_pos_w[0])
        v, w = _np(rd.body_com_lin_vel_w[0]), _np(rd.body_com_ang_vel_w[0])
        R = _qmat(lq)
        c_b = np.einsum("bji,bj->bi", R, cp - lp)                      # COM offset in the link frame
        R_prev = _rotvec_mat(-w * self.dt) @ R
        cp_prev = cp - v * self.dt
        lp_prev = cp_prev - np.einsum("bij,bj->bi", R_prev, c_b)
        return lp, R, lp_prev, R_prev

    def classify(self, link: str, x_local: np.ndarray):
        """pad label + distance to the nearest pad / non-pad sample (m) for points in the link frame."""
        S, Mk = self.d["samples"][link], self.d["masks"][link]
        d2 = ((x_local[:, None, :] - S[None]) ** 2).sum(-1)
        big = 1e9
        dp = np.sqrt(np.where(Mk[None], d2, big).min(1)) if Mk.any() else np.full(len(x_local), np.inf)
        dn = np.sqrt(np.where(~Mk[None], d2, big).min(1)) if (~Mk).any() else np.full(len(x_local), np.inf)
        return dp <= dn, dp, dn

    def classify_palm(self, link: str, x_local: np.ndarray):
        """palm zone id (0 = non-pad, 1/2/3 = palm_l/c/r) of the nearest palm surface sample, points in the link frame."""
        S, Z = self.d["palm_samples"][link], self.d["palm_zone"][link]
        d2 = ((x_local[:, None, :] - S[None]) ** 2).sum(-1)
        return Z[d2.argmin(1)]

    # ------------------------------------------------------------------ read
    def read(self, poses: tuple | None = None, keep_rows: bool = False, pose_source: str = "cache") -> dict:
        """``poses``: optional (link_pos_prev, R_prev) override for the point -> link mapping (validation);
        ``pose_source``: "cache" (pose after the previous substep, fallback backward), "backward" or "current"."""
        raw = self.view.read(raw=True)
        if self._rows_version != self.view.n_created:
            self._map_rows()
        cf, cpnt, cn, _sep, cc, cs = raw["contact"]
        ff, fp, fc, fs = raw["friction"]
        B, M = len(self.sens_names), max(self.view.view.filter_count, 1)
        cf, cn, cpnt = cf.reshape(-1), cn.reshape(-1, 3), cpnt.reshape(-1, 3)
        ff, fp = ff.reshape(-1, 3), fp.reshape(-1, 3)
        cc, cs = cc.reshape(B, M).astype(int), cs.reshape(B, M).astype(int)
        fc, fs = fc.reshape(B, M).astype(int), fs.reshape(B, M).astype(int)
        fm, ffm = raw["fmat"], raw["fric"]
        lp, R, lp_prev, R_prev = self._prev_link_poses()
        used = "backward"
        if poses is not None:
            lp_prev, R_prev = poses
            used = "override"
        elif pose_source == "current":
            lp_prev, R_prev = lp, R
            used = "current"
        elif pose_source == "cache" and len(self._cache) == 2 and np.array_equal(self._cache[-1][0], lp):
            lp_prev, R_prev = self._cache[0]
            used = "cache"
        nf = self.nf
        pad_w, non_w, tot_w = np.zeros((nf, 3)), np.zeros((nf, 3)), np.zeros((nf, 3))
        cen_num, cen_w = np.zeros((nf, 3)), np.zeros(nf)
        palm_w = np.zeros((self.np_, 3))
        zone_w = np.zeros((self.np_, 4, 3))      # [non-pad, palm_l, palm_c, palm_r], world frame
        palm_agg = np.zeros((self.np_, 3))
        n_rows = n_boundary = 0
        f_boundary = f_rows = 0.0
        rows, prow = [], []
        agg_w = np.zeros((nf, 3))
        for b in range(B):
            agg = (fm[b] + ffm[b]).sum(0)
            fi, pi = self.sens_finger[b], self.sens_palm[b]
            if fi < 0 and pi < 0:
                continue
            if fi >= 0:
                agg_w[fi] += agg
            else:
                palm_agg[pi] += agg
            link = self.sens_names[b]
            rb = self.rob_idx[b]
            pts, vec, isn, pair = [], [], [], []
            for m in range(M):
                for i in range(cs[b, m], cs[b, m] + cc[b, m]):
                    if cf[i] > 0:
                        pts.append(cpnt[i]); vec.append(cf[i] * cn[i]); isn.append(True); pair.append(m)
                for i in range(fs[b, m], fs[b, m] + fc[b, m]):
                    if np.any(ff[i] != 0):
                        pts.append(fp[i]); vec.append(ff[i]); isn.append(False); pair.append(m)
            if not pts:
                continue
            pts, vec, isn, pair = np.asarray(pts), np.asarray(vec), np.asarray(isn), np.asarray(pair)
            # PhysX row order within the buffers can differ between identical runs: sum in a canonical order so the
            # outputs are bit-reproducible
            o_ = np.lexsort(np.c_[isn, pair, pts, vec].T[::-1])
            pts, vec, isn, pair = pts[o_], vec[o_], isn[o_], pair[o_]
            if pi >= 0:
                palm_w[pi] += vec.sum(0)
                if self.zones_on:
                    xl = (pts - lp_prev[rb]) @ R_prev[rb]
                    zl = self.classify_palm(link, xl)
                    mag = np.linalg.norm(vec, axis=1)
                    share = np.zeros((len(vec), 4))
                    share[np.arange(len(vec)), zl] = 1.0
                    for m in np.unique(pair[~isn]):      # friction rows: zone shares of the pair's normal force
                        fi_rows = np.where(~isn & (pair == m))[0]
                        ni_rows = np.where(isn & (pair == m))[0]
                        if len(ni_rows):
                            sh = np.bincount(zl[ni_rows], weights=mag[ni_rows], minlength=4)
                            share[fi_rows] = sh / max(sh.sum(), 1e-12)
                    zone_w[pi] += share.T @ vec
                    if keep_rows:
                        prow.append((link, xl[isn], vec[isn], zl[isn]))
                continue
            tot_w[fi] += vec.sum(0)
            xl = (pts - lp_prev[rb]) @ R_prev[rb]                       # world -> link frame (pre-integration)
            lab, dp, dn = self.classify(link, xl)
            mag = np.linalg.norm(vec, axis=1)
            # pad fraction per row: normal rows by location; friction rows (PhysX friction-patch ANCHOR points, which
            # can sit millimetres away from the current contact points) get the pad fraction of the normal force of
            # their sensor-filter pair (Coulomb: friction follows normal load); own location only if the pair has no
            # normal row
            frac = lab.astype(np.float64)
            for m in np.unique(pair[~isn]):
                fi_rows = np.where(~isn & (pair == m))[0]
                ni_rows = np.where(isn & (pair == m))[0]
                if len(ni_rows):
                    frac[fi_rows] = mag[ni_rows][lab[ni_rows]].sum() / max(mag[ni_rows].sum(), 1e-12)
                    dd = np.linalg.norm(xl[fi_rows, None] - xl[None, ni_rows], axis=-1)
                    self._fric_anchor_mm.extend((dd.min(1) * 1e3).tolist())
                    del self._fric_anchor_mm[:-20000]
            pad_w[fi] += (frac[:, None] * vec).sum(0)
            non_w[fi] += ((1.0 - frac)[:, None] * vec).sum(0)
            sel = lab & isn
            if sel.any():
                rd_ = self.distal_rob[fi]
                xd = (pts[sel] - lp_prev[rd_]) @ R_prev[rd_]             # pad points in the distal frame
                cen_num[fi] += (mag[sel, None] * xd).sum(0)
                cen_w[fi] += mag[sel].sum()
            bnd = (np.abs(dp - dn) < 1e-3) & isn
            n_rows += int(isn.sum()); n_boundary += int(bnd.sum())
            f_rows += float(mag[isn].sum()); f_boundary += float(mag[bnd].sum())
            if keep_rows:
                rows.append((link, xl[isn], vec[isn], isn[isn], lab[isn], dp[isn], dn[isn]))
        Rd = R[self.distal_rob]
        Rp = R[self.palm_rob]
        out = {
            "total": np.einsum("fji,fj->fi", Rd, tot_w),
            "pad": np.einsum("fji,fj->fi", Rd, pad_w),
            "nonpad": np.einsum("fji,fj->fi", Rd, non_w),
            "pad_centroid": np.where(cen_w[:, None] > 0, cen_num / np.maximum(cen_w, 1e-12)[:, None], 0.0),
            "pad_w": pad_w, "nonpad_w": non_w,
            "palm": np.einsum("pji,pj->pi", Rp, palm_w),
        }
        palm_zone = None
        if self.zones_on:
            palm_zone = np.einsum("pji,pkj->pki", Rp, zone_w[:, 1:])
            out["palm_zone"] = palm_zone
            out["palm_nonpad"] = np.einsum("pji,pj->pi", Rp, zone_w[:, 0])
        # check: row-wise pad + non-pad against the AGGREGATED sensor (force matrix + friction) of the bundle
        resid = np.linalg.norm(pad_w + non_w - agg_w, axis=1)
        self.last_diag = {"sum_resid_max_N": float(resid.max()) if nf else 0.0,
                          "sum_resid_rel_max": float((resid / np.maximum(np.linalg.norm(agg_w, axis=1), 1e-3)).max())
                          if nf else 0.0,
                          "rows": n_rows, "boundary_rows": n_boundary, "force_rows": f_rows, "force_boundary": f_boundary,
                          "tot_w": tot_w, "pose_source": used, "cen_num": cen_num, "cen_w": cen_w,
                          "fric_link": ffm.sum(1)}
        if self.zones_on:
            # palm zones + palm non-pad against the AGGREGATED sensor (force matrix + friction) of the palm bundle
            presid = np.linalg.norm(zone_w.sum(1) - palm_agg, axis=1)
            self.last_diag["palm_resid_max_N"] = float(presid.max()) if self.np_ else 0.0
            self.last_diag["palm_zone_w"] = zone_w
        if keep_rows:
            self.last_diag["rows_detail"] = rows
            self.last_diag["palm_rows_detail"] = prow
            self.last_diag["prev_pose"] = (lp_prev, R_prev)
        self.last_raw = out                                             # float64 (substep averaging)
        res = {k: v.astype(np.float32) for k, v in out.items()}
        if self.touch_on:
            res["touch"], res["touch_contact"] = self._touch(out["pad"], palm_zone)
            res["touch_valid"] = self.touch_valid.copy()
        return res


def pad_feature_names(fingers: list[str]) -> list[str]:
    return [f"{f}.{a}" for f in fingers for a in ("x", "y", "z")]
