# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cut-progress machinery for the "cut" tasks (utility knife seam, scissors strip).

Cutting is represented by geometric predicates while the workpiece stays
kinematic. Individual tasks decide whether material/blade collision is enabled:
the utility-knife tape is filtered, while the scissors paper collides normally.

Two mechanisms live here:

- **Frontier sweep** (utility knife): a per-env scalar *frontier* -- how far
  along the workpiece's seam the cut has advanced -- that only moves while all
  gates hold at once (blade extended, tip pressed into the slot within lateral
  tolerance, blade aligned with the seam) and only *contiguously*: the tip must
  be at the current frontier (small backlash) and the frontier advances at a
  capped per-step rate. Teleporting the tip, tapping the far end, or waving
  above the tape accrues nothing.

- **Blade contact** (scissors): each blade's inward-facing cutting edge carries
  a continuous rectangular contact zone in the blade link's frame. Success
  requires both functional blade regions to touch the colliding paper.

Shared-state architecture copies :mod:`.stage_machine`: a module registry keyed
by ``task_key`` plus per-env runtime state cached on the env object, with one
``evaluate_*`` entry point memoized per environment step so termination,
reward, observation and visualization terms all read a single consistent
result (and state advances at most once per step). Per-env resets are detected
from ``episode_length_buf`` exactly like the stage machine does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .terminations import joint_relative_move
from .utils import asset_axis_w

# =====================================================================================
# Frontier sweep (utility knife)
# =====================================================================================


@dataclass(frozen=True)
class CutSweepSpec:
    """Static definition of one seam-sweep cut, registered per task key.

    Assets are referenced by scene-entity *name* (ids resolve lazily at first
    evaluation). The seam is a segment in the workpiece's root frame; the tool
    tip is a fixed offset in ``tip_body_name``'s link frame (scaled asset
    coordinates).
    """

    workpiece_asset_name: str
    seam_start_local: tuple[float, float, float]
    seam_end_local: tuple[float, float, float]
    tool_asset_name: str
    tip_body_name: str
    tip_local_offset: tuple[float, float, float]
    ext_joint_name: str
    # Blade-extension gate, in joint_relative_move "progress" units (scale-invariant).
    ext_threshold: float = 0.4
    # Tip must be within this horizontal distance of the seam line (m).
    lateral_tol: float = 0.012
    # Seam surface height in the workpiece frame; the tip must dip below
    # ``z_surface_local - min_dip`` (i.e. through the tape, into the slot).
    z_surface_local: float = 0.045
    min_dip: float = 0.003
    # Optional undirected *heading* gate: the horizontal projection (in the
    # workpiece frame) of this tool-root axis must align with the seam
    # direction within the cosine threshold. Only yaw counts -- the natural
    # tip-down cutting pitch is free. None disables the gate.
    align_axis_tool_local: tuple[float, float, float] | None = (0.0, 1.0, 0.0)
    align_threshold_cos: float = 0.90
    # Contiguity: the tip may lead the frontier by at most ``backlash`` (m) for
    # cutting to count, and the frontier advances at most ``max_step_advance``
    # (m) per env step.
    backlash: float = 0.02
    max_step_advance: float = 0.01
    # Frontier fraction at which the cut counts as complete.
    complete_frac: float = 0.95


_CUT_SWEEP_REGISTRY: dict[str, CutSweepSpec] = {}
# Set DEXVERSE_CUT_DEBUG=1 to print a per-gate trace for env 0 every
# DEXVERSE_CUT_DEBUG_EVERY steps (default 10).
_CUT_DEBUG = os.environ.get("DEXVERSE_CUT_DEBUG", "0") not in ("", "0", "false", "False")
_CUT_DEBUG_EVERY = max(int(os.environ.get("DEXVERSE_CUT_DEBUG_EVERY", "10")), 1)


def register_cut_sweep(task_key: str, spec: CutSweepSpec, *, override: bool = False) -> None:
    """Register (or replace, with ``override=True``) a sweep spec for a task key."""
    if not override and task_key in _CUT_SWEEP_REGISTRY and _CUT_SWEEP_REGISTRY[task_key] != spec:
        raise ValueError(f"Cut sweep '{task_key}' already registered with a different spec; pass override=True.")
    _CUT_SWEEP_REGISTRY[task_key] = spec


def get_cut_sweep_spec(task_key: str) -> CutSweepSpec:
    if task_key not in _CUT_SWEEP_REGISTRY:
        raise KeyError(f"Cut sweep '{task_key}' is not registered. Available: {sorted(_CUT_SWEEP_REGISTRY)}")
    return _CUT_SWEEP_REGISTRY[task_key]


def _get_episode_length_buf(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if isinstance(episode_length_buf, torch.Tensor) and episode_length_buf.ndim == 1:
        return episode_length_buf
    return None


def _compute_reset_mask(
    episode_length_buf: torch.Tensor | None,
    previous_step_buf: torch.Tensor | None,
) -> torch.Tensor | None:
    if episode_length_buf is None:
        return None
    if isinstance(previous_step_buf, torch.Tensor) and previous_step_buf.shape == episode_length_buf.shape:
        # A counter that went *backwards* is an unambiguous new episode. Relying on
        # ``== 0`` alone only works if this state happens to be evaluated at the
        # very first step -- lazy consumers (a strict stage graph only evaluates
        # its first pending stage's predicate; obs presets can drop the terms
        # that would touch it) otherwise inherit the previous episode's progress
        # (Aug 2026: the cut frontier stayed at 1.0 into the next replayed demo,
        # so success fired as soon as the blade extended).
        return ((episode_length_buf == 0) & (previous_step_buf != 0)) | (episode_length_buf < previous_step_buf)
    return episode_length_buf == 0


class _CutSweepState:
    """Per-task mutable runtime state cached on the env object."""

    def __init__(self):
        self.frontier_m: torch.Tensor | None = None
        self.last_step_buf: torch.Tensor | None = None
        self.memo_step_buf: torch.Tensor | None = None
        self.memo_out: dict[str, torch.Tensor] | None = None
        self.tip_body_id: int | None = None
        self.ext_joint_cfg: SceneEntityCfg | None = None


def _get_sweep_state(env: ManagerBasedRLEnv, task_key: str) -> _CutSweepState:
    cache = getattr(env, "_cut_sweep_runtime_cache", None)
    if cache is None:
        cache = {}
        env._cut_sweep_runtime_cache = cache
    if task_key not in cache:
        cache[task_key] = _CutSweepState()
    return cache[task_key]


def _advance_frontier(
    frontier_m: torch.Tensor,
    s_tip: torch.Tensor,
    gates_ok: torch.Tensor,
    *,
    backlash: float,
    max_step_advance: float,
    seam_length: float,
    reset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure frontier-advance rule (unit-testable without a simulator).

    The frontier advances toward ``s_tip`` only when ``gates_ok`` holds and the
    tip's *lead* over the frontier is at most ``backlash`` (a lagging tip
    re-cuts finished material: no advance, no penalty), by at most
    ``max_step_advance`` per call. Monotone within an episode; ``reset_mask``
    zeroes it first.
    """
    if reset_mask is not None:
        frontier_m = torch.where(reset_mask, torch.zeros_like(frontier_m), frontier_m)
    lead = s_tip - frontier_m
    ok = gates_ok & (lead <= backlash)
    advance = torch.clamp(lead, min=0.0, max=max_step_advance) * ok.to(frontier_m.dtype)
    return torch.clamp(frontier_m + advance, min=0.0, max=seam_length)


def evaluate_cut_sweep(env: ManagerBasedRLEnv, task_key: str) -> dict[str, torch.Tensor]:
    """Advance (at most once per env step) and return the sweep signals.

    Returns a dict with ``frontier_frac`` (E,), ``frontier_m`` (E,),
    ``tip_w`` (E, 3), ``frontier_point_w`` (E, 3), ``ext_ok`` (E,) bool and
    ``gates_ok`` (E,) bool. Repeated calls within one step return the memoized
    result, so termination / reward / obs / vis terms stay consistent.
    """
    spec = get_cut_sweep_spec(task_key)
    state = _get_sweep_state(env, task_key)
    device = env.device

    episode_length_buf = _get_episode_length_buf(env)
    if (
        state.memo_out is not None
        and episode_length_buf is not None
        and isinstance(state.memo_step_buf, torch.Tensor)
        and torch.equal(state.memo_step_buf, episode_length_buf)
    ):
        return state.memo_out

    workpiece = env.scene[spec.workpiece_asset_name]
    tool = env.scene[spec.tool_asset_name]
    if state.tip_body_id is None:
        body_ids, _ = tool.find_bodies(spec.tip_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Cut sweep '{task_key}': tip body '{spec.tip_body_name}' matched {len(body_ids)} bodies.")
        state.tip_body_id = body_ids[0]
    if state.ext_joint_cfg is None:
        state.ext_joint_cfg = SceneEntityCfg(spec.tool_asset_name, joint_names=[spec.ext_joint_name])

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Blade-tip world point from the (non-root) blade link pose.
    tip_pos_w = tool.data.body_pos_w[:, state.tip_body_id]
    tip_quat_w = tool.data.body_quat_w[:, state.tip_body_id]
    tip_offset = torch.tensor(spec.tip_local_offset, device=device, dtype=tip_pos_w.dtype)
    tip_w = tip_pos_w + quat_apply(tip_quat_w, tip_offset.unsqueeze(0).expand(tip_pos_w.shape[0], -1))

    # Tip in the workpiece root frame, decomposed against the seam segment.
    wp_pos_w = workpiece.data.root_pos_w
    wp_quat_w = workpiece.data.root_quat_w
    tip_local = quat_apply_inverse(wp_quat_w, tip_w - wp_pos_w)
    seam_start = torch.tensor(spec.seam_start_local, device=device, dtype=tip_w.dtype)
    seam_end = torch.tensor(spec.seam_end_local, device=device, dtype=tip_w.dtype)
    seam_vec = seam_end - seam_start
    seam_length = float(torch.linalg.norm(seam_vec))
    seam_dir = seam_vec / max(seam_length, 1e-9)
    rel = tip_local - seam_start.unsqueeze(0)
    s_tip = torch.clamp(rel @ seam_dir, min=0.0, max=seam_length)
    perp = rel - s_tip.unsqueeze(-1) * seam_dir.unsqueeze(0)
    lateral = torch.linalg.norm(perp[:, :2], dim=1)
    z_tip = tip_local[:, 2]

    # Gates.
    ext_ok = joint_relative_move(
        env, threshold=spec.ext_threshold, asset_cfg=state.ext_joint_cfg, mode="progress", op=">=", reduce="any"
    )
    in_slot = (lateral <= spec.lateral_tol) & (z_tip <= spec.z_surface_local - spec.min_dip)
    gates_ok = ext_ok & in_slot
    aligned = None
    if spec.align_axis_tool_local is not None:
        # Heading-only alignment: project the tool axis into the workpiece
        # frame, drop z, and compare its direction with the (horizontal) seam
        # direction. A near-vertical tool has no meaningful heading -> not aligned.
        tool_axis_w = asset_axis_w(env, asset_cfg=SceneEntityCfg(spec.tool_asset_name), axis_local=spec.align_axis_tool_local)
        tool_axis_wp = quat_apply_inverse(wp_quat_w, tool_axis_w)
        tool_axis_h = tool_axis_wp.clone()
        tool_axis_h[:, 2] = 0.0
        seam_dir_h = seam_dir.clone()
        seam_dir_h[2] = 0.0
        seam_dir_h = seam_dir_h / torch.clamp(torch.linalg.norm(seam_dir_h), min=1e-9)
        h_norm = torch.linalg.norm(tool_axis_h, dim=1)
        cos = torch.abs(tool_axis_h @ seam_dir_h) / torch.clamp(h_norm, min=1e-9)
        aligned = (h_norm > 0.3) & (cos >= spec.align_threshold_cos)
        gates_ok = gates_ok & aligned

    # Frontier update with per-env reset detection.
    if state.frontier_m is None or state.frontier_m.shape[0] != env.num_envs:
        state.frontier_m = torch.zeros(env.num_envs, device=device, dtype=tip_w.dtype)
        state.last_step_buf = None
    reset_mask = _compute_reset_mask(episode_length_buf, state.last_step_buf)
    if episode_length_buf is not None:
        state.last_step_buf = episode_length_buf.clone()
    state.frontier_m = _advance_frontier(
        state.frontier_m,
        s_tip,
        gates_ok,
        backlash=spec.backlash,
        max_step_advance=spec.max_step_advance,
        seam_length=seam_length,
        reset_mask=reset_mask,
    )

    frontier_point_local = seam_start.unsqueeze(0) + state.frontier_m.unsqueeze(-1) * seam_dir.unsqueeze(0)
    frontier_point_w = wp_pos_w + quat_apply(wp_quat_w, frontier_point_local)

    out = {
        "frontier_m": state.frontier_m,
        "frontier_frac": state.frontier_m / max(seam_length, 1e-9),
        "tip_w": tip_w,
        "frontier_point_w": frontier_point_w,
        "ext_ok": ext_ok,
        "gates_ok": gates_ok,
    }
    if _CUT_DEBUG:
        # Throttled per-gate trace for env 0 (set DEXVERSE_CUT_DEBUG=1), e.g. to
        # see which gate stalls the frontier during a teleop drag.
        state.debug_tick = getattr(state, "debug_tick", 0) + 1
        if state.debug_tick % _CUT_DEBUG_EVERY == 0:
            lead = float(s_tip[0] - state.frontier_m[0])
            print(
                f"[cut_sweep:{task_key}] frontier={float(out['frontier_frac'][0]):.3f} "
                f"s_tip={float(s_tip[0]):.3f} lead={lead:+.3f} (backlash {spec.backlash}) "
                f"lateral={float(lateral[0]):.4f}/{spec.lateral_tol} "
                f"z_tip={float(z_tip[0]):.4f} (need<={spec.z_surface_local - spec.min_dip:.4f}) "
                f"ext_ok={bool(ext_ok[0])} in_slot={bool(in_slot[0])} "
                f"aligned={bool(aligned[0]) if spec.align_axis_tool_local is not None else 'n/a'} "
                f"gates_ok={bool(gates_ok[0])}"
            )
    state.memo_out = out
    state.memo_step_buf = episode_length_buf.clone() if episode_length_buf is not None else None
    return out


def cut_sweep_complete(env: ManagerBasedRLEnv, task_key: str, threshold: float | None = None) -> torch.Tensor:
    """Per-env bool: the sweep frontier passed ``threshold`` (spec default) of the seam."""
    out = evaluate_cut_sweep(env, task_key)
    thr = get_cut_sweep_spec(task_key).complete_frac if threshold is None else threshold
    return out["frontier_frac"] >= thr


def cut_sweep_frontier(env: ManagerBasedRLEnv, task_key: str) -> torch.Tensor:
    """Observation: sweep frontier fraction, shape (E, 1)."""
    return evaluate_cut_sweep(env, task_key)["frontier_frac"].unsqueeze(-1)


def cut_sweep_progress_reward(env: ManagerBasedRLEnv, task_key: str) -> torch.Tensor:
    """Dense reward: frontier fraction in [0, 1] (monotone within an episode)."""
    return evaluate_cut_sweep(env, task_key)["frontier_frac"]


def cut_sweep_tip_tracking_reward(
    env: ManagerBasedRLEnv,
    task_key: str,
    distance_gain: float = 10.0,
    require_ext: bool = True,
) -> torch.Tensor:
    """Dense reward: ``exp(-gain * |tip - frontier_point|)``, optionally gated
    on the blade-extension condition so tracking only pays with the blade out."""
    out = evaluate_cut_sweep(env, task_key)
    dist = torch.linalg.norm(out["tip_w"] - out["frontier_point_w"], dim=1)
    reward = torch.exp(-distance_gain * dist)
    if require_ext:
        reward = reward * out["ext_ok"].to(reward.dtype)
    return reward


class cut_frontier_vis(ManagerTermBase):
    """Sphere marker riding the cut frontier, red -> green with progress.

    Visualization-only side-effect term (wire into ``observations.scene_vis``).
    Unlike the physical tape/seam target, this is a debug progress cue and
    follows the shared v1 camera-visibility policy.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self._task_key: str = cfg.params["task_key"]
        radius = float(cfg.params.get("radius", 0.02))
        opacity = float(cfg.params.get("opacity", 0.8))
        self._visible = bool(cfg.params.get("visible", True))
        num_steps = max(int(cfg.params.get("num_color_steps", 11)), 2)
        prim_path_prefix = cfg.params.get("prim_path_prefix", "/Visuals/CutFrontierMarker")
        self._num_steps = num_steps

        markers = {}
        for i in range(num_steps):
            t = i / (num_steps - 1)
            markers[f"step_{i}"] = sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0 - t, t, 0.0), opacity=opacity),
            )
        self._visualizer = VisualizationMarkers(VisualizationMarkersCfg(prim_path=prim_path_prefix, markers=markers))
        from dexverse.baseline_v1.visual_purpose import hide_marker_from_cameras

        hide_marker_from_cameras(self._visualizer)
        self._visualizer.set_visibility(self._visible)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        task_key: str | None = None,
        radius: float | None = None,
        opacity: float | None = None,
        visible: bool | None = None,
        num_color_steps: int | None = None,
        prim_path_prefix: str | None = None,
    ) -> torch.Tensor:
        if visible is not None and bool(visible) != self._visible:
            self._visible = bool(visible)
            self._visualizer.set_visibility(self._visible)
        if not self._visible:
            return torch.zeros(env.num_envs, 0, device=env.device)

        out = evaluate_cut_sweep(env, task_key or self._task_key)
        progress = out["frontier_frac"]
        indices = (progress * (self._num_steps - 1)).round().clamp(0, self._num_steps - 1).to(torch.int32)
        self._visualizer.visualize(translations=out["frontier_point_w"], marker_indices=indices)
        return torch.zeros(env.num_envs, 0, device=env.device)


# =====================================================================================
# Blade contact (scissors)
# =====================================================================================


def point_in_asset_zone(
    env: ManagerBasedRLEnv,
    point_asset_cfg: SceneEntityCfg,
    point_local_offset: tuple[float, float, float],
    zone_asset_cfg: SceneEntityCfg,
    zone_center_local: tuple[float, float, float],
    zone_half_extents: tuple[float, float, float],
) -> torch.Tensor:
    """Per-env bool: a fixed local point on one asset lies inside an
    axis-aligned box in another asset's root frame."""
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    point_asset = env.scene[point_asset_cfg.name]
    zone_asset = env.scene[zone_asset_cfg.name]
    dtype = point_asset.data.root_pos_w.dtype
    offset = torch.tensor(point_local_offset, device=env.device, dtype=dtype)
    point_w = point_asset.data.root_pos_w + quat_apply(
        point_asset.data.root_quat_w, offset.unsqueeze(0).expand(env.num_envs, -1)
    )
    point_zone = quat_apply_inverse(zone_asset.data.root_quat_w, point_w - zone_asset.data.root_pos_w)
    center = torch.tensor(zone_center_local, device=env.device, dtype=dtype)
    half = torch.tensor(zone_half_extents, device=env.device, dtype=dtype)
    return (torch.abs(point_zone - center) <= half).all(dim=1)


def scissors_inner_blades_touch_strip(
    env: ManagerBasedRLEnv,
    scissors_cfg: SceneEntityCfg,
    blade_body_names: tuple[str, str],
    blade_zone_center_local: tuple[float, float, float],
    blade_zone_half_extents: tuple[float, float, float],
    strip_cfg: SceneEntityCfg,
    strip_box_center_local: tuple[float, float, float],
    strip_box_half_extents: tuple[float, float, float],
    contact_inflation: tuple[float, float, float] = (0.004, 0.004, 0.006),
    strip_edge_margin_xy: tuple[float, float] = (0.0, 0.0),
    hold_steps: int = 1,
) -> torch.Tensor:
    """Per-env bool: both inward-facing blade regions touch the paper strip.

    Each blade region is an oriented box in that blade link's frame. Contact is
    the exact box-vs-box overlap test (15-axis separating-axis theorem) against
    the paper box, inflated by ``contact_inflation`` to include resting contact
    without requiring penetration. This continuous rectangle avoids the dead
    gaps created by discrete edge samples. Requiring both blade regions and
    their centres to lie on opposite sides of the paper plane excludes handles,
    fingers, a single blade resting on the paper, and scissors laid flat on
    top. The midpoint between those centres must also pass through an inset
    region of the paper's top face, so touching a vertical side edge does not
    count. The valid configuration must persist for ``hold_steps`` consecutive
    environment steps; losing contact resets that counter. The stage graph
    separately requires that the scissors were opened earlier.
    """
    from isaaclab.utils.math import quat_apply

    scissors = env.scene[scissors_cfg.name]
    strip = env.scene[strip_cfg.name]
    dtype = strip.data.root_pos_w.dtype
    zone_center = torch.tensor(blade_zone_center_local, device=env.device, dtype=dtype)
    zone_half = torch.tensor(blade_zone_half_extents, device=env.device, dtype=dtype)
    strip_center_local = torch.tensor(strip_box_center_local, device=env.device, dtype=dtype)
    strip_half = torch.tensor(strip_box_half_extents, device=env.device, dtype=dtype) + torch.tensor(
        contact_inflation, device=env.device, dtype=dtype
    )
    strip_pos = strip.data.root_pos_w
    strip_quat = strip.data.root_quat_w

    basis = torch.eye(3, device=env.device, dtype=dtype)

    def _box_axes(quat: torch.Tensor) -> torch.Tensor:
        E = quat.shape[0]
        return quat_apply(
            quat.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4),
            basis.unsqueeze(0).expand(E, -1, -1).reshape(-1, 3),
        ).reshape(E, 3, 3)

    def _obb_overlap(
        center_a: torch.Tensor,
        axes_a: torch.Tensor,
        half_a: torch.Tensor,
        center_b: torch.Tensor,
        axes_b: torch.Tensor,
        half_b: torch.Tensor,
    ) -> torch.Tensor:
        """Batched 3-D OBB overlap using all 15 separating axes."""
        # R[i,j] = A_i dot B_j; translation expressed in A's frame.
        R = torch.einsum("eik,ejk->eij", axes_a, axes_b)
        abs_R = R.abs() + 1.0e-6  # stabilize nearly parallel cross axes
        delta = center_b - center_a
        t = torch.einsum("eik,ek->ei", axes_a, delta)

        overlap = (t.abs() <= half_a + torch.einsum("eij,j->ei", abs_R, half_b)).all(dim=1)
        t_b = torch.einsum("ei,eij->ej", t, R)
        overlap &= (t_b.abs() <= half_b + torch.einsum("eij,i->ej", abs_R, half_a)).all(dim=1)

        for i in range(3):
            i1, i2 = (i + 1) % 3, (i + 2) % 3
            for j in range(3):
                j1, j2 = (j + 1) % 3, (j + 2) % 3
                lhs = (t[:, i2] * R[:, i1, j] - t[:, i1] * R[:, i2, j]).abs()
                rhs = (
                    half_a[i1] * abs_R[:, i2, j]
                    + half_a[i2] * abs_R[:, i1, j]
                    + half_b[j1] * abs_R[:, i, j2]
                    + half_b[j2] * abs_R[:, i, j1]
                )
                overlap &= lhs <= rhs
        return overlap

    strip_center = strip_pos + quat_apply(
        strip_quat, strip_center_local.unsqueeze(0).expand(env.num_envs, -1)
    )
    strip_axes = _box_axes(strip_quat)

    blade_contact = []
    blade_side = []
    blade_center_in_strip = []
    for body_name in blade_body_names:
        body_ids, _ = scissors.find_bodies(body_name)
        bid = body_ids[0]
        pos = scissors.data.body_pos_w[:, bid]  # (E, 3)
        quat = scissors.data.body_quat_w[:, bid]
        blade_center = pos + quat_apply(quat, zone_center.unsqueeze(0).expand(env.num_envs, -1))
        # Signed distance along the paper normal. A real snip places one inner
        # edge on either side; broad-side contact puts both on the same side.
        blade_center_strip = torch.einsum("eik,ek->ei", strip_axes, blade_center - strip_center)
        blade_side.append(blade_center_strip[:, 2])
        blade_center_in_strip.append(blade_center_strip)
        blade_contact.append(
            _obb_overlap(blade_center, _box_axes(quat), zone_half, strip_center, strip_axes, strip_half)
        )
    opposite_sides = blade_side[0] * blade_side[1] <= 0.0
    cut_midpoint = 0.5 * (blade_center_in_strip[0] + blade_center_in_strip[1])
    edge_margin = torch.tensor(strip_edge_margin_xy, device=env.device, dtype=dtype)
    cut_half_xy = torch.clamp(
        torch.tensor(strip_box_half_extents[:2], device=env.device, dtype=dtype) - edge_margin,
        min=0.0,
    )
    through_top_face = (cut_midpoint[:, :2].abs() <= cut_half_xy).all(dim=1)
    valid_contact = blade_contact[0] & blade_contact[1] & opposite_sides & through_top_face

    hold_steps = max(int(hold_steps), 1)
    if hold_steps == 1:
        return valid_contact

    # Keep this small temporal latch outside the scene: the predicate may be
    # read by termination, reward, and observations in one simulation step, so
    # memoize by episode_length_buf to increment at most once per step.
    cache = getattr(env, "_scissors_strip_contact_hold_cache", None)
    if cache is None:
        cache = {}
        env._scissors_strip_contact_hold_cache = cache
    cache_key = (scissors_cfg.name, strip_cfg.name, tuple(strip_edge_margin_xy), hold_steps)
    state = cache.get(cache_key)
    current_step = env.episode_length_buf.to(dtype=torch.long)
    if state is None or state["count"].shape != current_step.shape:
        state = {
            "count": torch.zeros_like(current_step),
            "last_step": None,
            "memo": None,
        }
        cache[cache_key] = state

    last_step = state["last_step"]
    reset_rows = current_step == 0
    if isinstance(last_step, torch.Tensor):
        reset_rows &= last_step != 0
        if torch.equal(last_step, current_step):
            return state["memo"]
    state["count"][reset_rows] = 0
    state["count"] = torch.where(valid_contact, state["count"] + 1, torch.zeros_like(state["count"]))
    state["memo"] = state["count"] >= hold_steps
    state["last_step"] = current_step.clone()
    return state["memo"]
