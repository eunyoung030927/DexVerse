# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default-on tactile logging of the LeRobot recorder: per-pad forces (``tactile``), virtual fingerprint pads and
palm zones (``tactile_pads``) and the real-hand ``observation.touch`` layout, reduced over the physics substeps.

Created by :class:`~dexverse.data_collection.lerobot_spool.LeRobotSpoolRecorder` AFTER the env is built; nothing
is added to the env config: both readers open their own PhysX rigid-contact views
(:class:`~dexverse.data_collection.tactile.RawContactView`). Environment variables:

* ``DEXVERSE_TACTILE=0``            no tactile logging (the recorder behaves as without this module).
* ``DEXVERSE_TACTILE_PROBE=<file>`` (diagnostic, off by default) also keep, per physics substep, each pad body's
  contact force and the smallest contact separation (negative = penetration depth, m) against the target, and whether
  the substep is the first one after a state restore / reset; saved to ``<file>`` (.npz) at exit.
* ``DEXVERSE_TACTILE_REDUCE=mean``  (default) every physics substep of a control step is read from a read-only hook
  after ``scene.update`` (the substep's contacts are placed with the link poses before that substep) and the frame
  stores the AVERAGE over the substeps of the env.step that led to it (force-weighted pad centre over all those
  substeps; ``touch`` from the averaged forces; ``touch_contact`` = averaged fn > threshold); the frame right after
  a reset is zeros. ``last`` = only the last substep, read at capture time (the previous behaviour).
"""

from __future__ import annotations

import os

import numpy as np

REDUCE_ENV = "DEXVERSE_TACTILE_REDUCE"
PROBE_ENV = "DEXVERSE_TACTILE_PROBE"
ENABLE_ENV = "DEXVERSE_TACTILE"
TOUCH_FN_MAX_FEATURE = "observation.touch_fn_substep_max"

_PAD_SUM_KEYS = ("total", "pad", "nonpad", "pad_w", "nonpad_w", "palm", "palm_zone", "palm_nonpad")


def tactile_enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "1").strip().lower() not in ("0", "false", "off", "no")


def reduce_mode() -> str:
    m = os.environ.get(REDUCE_ENV, "mean").strip().lower()
    if m not in ("mean", "last"):
        raise ValueError(f"[tactile] {REDUCE_ENV}={m!r}: expected 'mean' or 'last'")
    return m


class TactileLogger:
    """``TactileReader`` + ``TactilePadReader`` with substep reduction (module docstring). Env 0."""

    def __init__(self, env, robot_type: str, target_prims: list[str] | None = None, reduce: str | None = None):
        from dexverse.data_collection.tactile import TactileReader, resolve_targets, tactile_body_names  # noqa: PLC0415
        from dexverse.data_collection.tactile_pads import TactilePadReader, pad_definition_path  # noqa: PLC0415

        if not os.path.exists(pad_definition_path(robot_type)):
            raise FileNotFoundError(f"no pad definition for {robot_type!r} ({pad_definition_path(robot_type)})")
        self.env = env
        self.robot_type = robot_type
        self.reduce = reduce or reduce_mode()
        rc = env.cfg.robot_config
        robot = env.scene["robot"]
        self.targets = resolve_targets(env.cfg, target_prims)
        bodies = tactile_body_names(robot_type, list(rc.fingertip_body_names), list(robot.body_names))
        self.tactile = TactileReader(env, bodies, self.targets)
        self.pads = TactilePadReader(env, robot_type, self.targets)   # installs the substep pose cache hook
        self.decimation = int(env.cfg.decimation)
        self.physics_dt = float(env.physics_dt)
        self.debug_substeps = None        # set to a list to keep every substep's readings (validation)
        self._reset_acc()
        self.substep_counts = []
        self.probe_path = os.environ.get(PROBE_ENV, "").strip() or None
        if self.probe_path:
            import atexit  # noqa: PLC0415

            self._probe_rows, self._probe_first = [], True
            atexit.register(self._probe_save)
            print(f"[tactile] probe ON -> {self.probe_path} (pads {self.tactile.bodies})")
        if self.reduce == "mean":
            # after the pad reader's pose-cache hook (same scene.update wrapper, called in list order)
            env.scene._tactile_pad_pose_hooks.append(self._on_substep)

    # ------------------------------------------------------------------ accumulation
    def _reset_acc(self):
        self._n = 0
        self._tac = {}
        self._pad = {}
        self._cen_num = None
        self._cen_w = None
        self._fn_max = None
        self._diag = {"sum_resid_max_N": 0.0, "palm_resid_max_N": 0.0, "rows": 0, "boundary_rows": 0,
                      "force_rows": 0.0, "force_boundary": 0.0, "pose_source": {}}

    def _on_substep(self):
        tac = self.tactile.read64()
        self.pads.read()
        pad, dg = self.pads.last_raw, self.pads.last_diag
        for k, v in tac.items():
            self._tac[k] = self._tac.get(k, 0.0) + v
        for k in _PAD_SUM_KEYS:
            if k in pad:
                self._pad[k] = self._pad.get(k, 0.0) + pad[k]
        self._cen_num = dg["cen_num"] if self._cen_num is None else self._cen_num + dg["cen_num"]
        self._cen_w = dg["cen_w"] if self._cen_w is None else self._cen_w + dg["cen_w"]
        if self.pads.touch_on:
            fn = self.pads._touch(pad["pad"], pad.get("palm_zone"))[0].reshape(-1, 3)[:, 0].astype(np.float64)
            self._fn_max = fn if self._fn_max is None else np.maximum(self._fn_max, fn)
        d = self._diag
        d["sum_resid_max_N"] = max(d["sum_resid_max_N"], dg["sum_resid_max_N"])
        d["palm_resid_max_N"] = max(d["palm_resid_max_N"], dg.get("palm_resid_max_N", 0.0))
        for k in ("rows", "boundary_rows", "force_rows", "force_boundary"):
            d[k] += dg[k]
        d["pose_source"][dg["pose_source"]] = d["pose_source"].get(dg["pose_source"], 0) + 1
        if self.probe_path:
            self._probe(tac)
        self._n += 1
        if self.debug_substeps is not None:
            self.debug_substeps.append({"tactile": {k: v.copy() for k, v in tac.items()},
                                        "pads": {k: np.array(v, copy=True) for k, v in pad.items()},
                                        "cen_num": dg["cen_num"].copy(), "cen_w": dg["cen_w"].copy()})

    def mark_reset(self):
        """After anything that moves the scene outside env.step (reset / state restore)."""
        self.pads.mark_reset()
        self._reset_acc()
        self.note_state_moved()

    def note_state_moved(self):
        """Probe only: the next substep starts from a state set outside env.step (reset / state restore)."""
        if self.probe_path:
            self._probe_first = True

    # ------------------------------------------------------------------ probe (DEXVERSE_TACTILE_PROBE)
    def _probe(self, tac):
        """One row per substep: frame, substep, first-after-restore, per pad |force| (N), min separation (m), contacts."""
        v = self.tactile.view
        d = v._read(raw=True)
        _f, _p, _n, sep, cnt, start = d["contact"]
        sep = sep.reshape(-1)
        B, F = len(v.names), max(v.view.filter_count, 1)
        cnt, start = cnt.reshape(B, F).astype(int), start.reshape(B, F).astype(int)
        row = [v.names.index(b) for b in self.tactile.bodies]
        min_sep = np.full(len(row), np.nan)
        n_c = np.zeros(len(row), int)
        for i, b in enumerate(row):
            s = [sep[start[b, m]:start[b, m] + cnt[b, m]] for m in range(F) if cnt[b, m]]
            if s:
                s = np.concatenate(s)
                min_sep[i], n_c[i] = float(s.min()), s.size
        self._probe_rows.append((len(self.substep_counts), self._n, self._probe_first,
                                 np.linalg.norm(tac["force_w"], axis=1), min_sep, n_c))
        self._probe_first = False
        if len(self._probe_rows) % 600 == 0:   # Isaac Sim may exit without running atexit
            self._probe_save()

    def _probe_save(self):
        if not self._probe_rows:
            return
        r = list(zip(*self._probe_rows))
        np.savez(self.probe_path, frame=np.array(r[0]), substep=np.array(r[1]), first_after_restore=np.array(r[2]),
                 force=np.stack(r[3]), min_sep=np.stack(r[4]), contacts=np.stack(r[5]),
                 pads=np.array(self.tactile.bodies), physics_dt=self.physics_dt)
        print(f"[tactile] probe: {len(self._probe_rows)} substeps -> {self.probe_path}")

    # ------------------------------------------------------------------ frame
    def zeros(self):
        z = self.pads.zeros()
        if self.pads.touch_on:
            z["touch_fn_substep_max"] = np.zeros(self.pads.n_hands * 8, np.float32)
        return self.tactile.zeros(), z, None

    def capture(self, force_zero: bool):
        """(tactile dict, pads dict, diag) of the frame; zeros (and a reset of the accumulator) if ``force_zero``."""
        if force_zero:
            self.mark_reset()
            return self.zeros()
        if self.reduce == "last":
            tac = self.tactile.read()
            pad = self.pads.read()
            if self.pads.touch_on:
                pad["touch_fn_substep_max"] = pad["touch"].reshape(-1, 3)[:, 0].copy()
            dg = dict(self.pads.last_diag)
            dg["substeps"] = 1
            return tac, pad, dg
        n = self._n
        self.substep_counts.append(n)
        if n == 0:
            out = self.zeros()
            self._reset_acc()
            return out[0], out[1], {"substeps": 0}
        tac = {k: (v / n).astype(np.float32) for k, v in self._tac.items()}
        pad64 = {k: v / n for k, v in self._pad.items()}
        cen = np.where(self._cen_w[:, None] > 0, self._cen_num / np.maximum(self._cen_w, 1e-12)[:, None], 0.0)
        pad = {k: v.astype(np.float32) for k, v in pad64.items()}
        pad["pad_centroid"] = cen.astype(np.float32)
        if self.pads.touch_on:
            pad["touch"], pad["touch_contact"] = self.pads._touch(pad64["pad"], pad64.get("palm_zone"))
            pad["touch_valid"] = self.pads.touch_valid.copy()
            pad["touch_fn_substep_max"] = self._fn_max.astype(np.float32)
        dg = dict(self._diag)
        dg["substeps"] = n
        self._reset_acc()
        return tac, pad, dg

    def info(self) -> dict:
        return {"reduce": self.reduce, "substeps_per_frame": self.decimation, "physics_dt": self.physics_dt,
                "control_dt": self.physics_dt * self.decimation, "pad_bodies": self.tactile.bodies,
                "targets": self.targets,
                "definition": ("mean: every tactile value of frame t is the average over the physics substeps of the "
                               "env.step that led to frame t (substep contacts read from a read-only hook after "
                               "scene.update, placed with the link poses before that substep); pad_centroid is "
                               "force-weighted over those substeps; observation.touch is computed from the averaged "
                               "forces and touch_contact = averaged fn > threshold; observation.touch_fn_substep_max = "
                               "the largest per-substep fn of the pad (diagnostic); frame 0 / after a reset = zeros"
                               if self.reduce == "mean" else
                               "last: the contact solve of the last physics substep of the env.step that led to frame "
                               "t (read at capture time); touch_fn_substep_max = that substep's fn"),
                "switches": {ENABLE_ENV: "0 disables tactile logging", REDUCE_ENV: "mean (default) | last"}}
