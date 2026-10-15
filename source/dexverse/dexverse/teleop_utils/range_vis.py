# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw the environment's randomization ranges as wireframe boxes.

A teleoperator sees one sample from each randomization range per episode and has
no way to know how wide those ranges actually are. This module reads the ranges
straight out of the resolved env config and outlines them in the scene, so the
operator can see the whole spawn/goal envelope while demonstrating.

Two pieces:

* :func:`collect_range_boxes` -- introspects ``env.cfg.events`` (every reset term
  carrying a ``pose_range``) and ``env.cfg.commands`` (goal-command sampling
  ranges) and returns frame-resolved :class:`RangeBox` specs.
* :class:`randomization_ranges_vis` -- a side-effect-only observation term (the
  same ``ManagerTermBase`` pattern as ``mdp.forbidden_zones_vis``) that renders
  those boxes as true wireframes: 12 thin cuboid edge bars per box, drawn from a
  single instanced marker prototype via per-instance scales.

Attach it with :func:`attach_range_vis` before ``gym.make``; the teleop scripts
expose it as ``--show_ranges``.

Frame semantics (these differ per reset function -- getting them wrong silently
draws the box in the wrong place):

* ``reset_root_state_uniform`` / ``reset_root_pose_uniform`` /
  ``reset_root_pose_uniform_excluding`` treat ``pose_range`` as an offset from
  the asset's ``default_root_state``, so the box is
  ``default_root_state[:, :3] + env_origins + range``.
* ``reset_object_from_place_annotations`` anchors on the annotation instead:
  ``env_origins + per_env_positions`` with ``z`` replaced by ``support_z``, and
  ``pose_range`` is extra jitter on top.
* Goal commands sample ``ranges.pos_*`` directly into world coordinates when
  ``use_world_frame`` is set -- ``env_origins`` is *not* added (see
  ``mdp/commands/pose_commands.py::_resample_command``). Drawing them with an
  env-origin offset would show where the goal ought to be, not where it is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

logger = logging.getLogger(__name__)

# Reset functions whose ``pose_range`` is an offset from ``default_root_state``.
_DEFAULT_ROOT_ANCHORED = (
    "reset_root_state_uniform",
    "reset_root_pose_uniform",
    "reset_root_pose_uniform_excluding",
    "reset_articulation_with_supports_uniform",
)
# Reset function anchored on per-object placement annotations instead.
_ANNOTATION_ANCHORED = ("reset_object_from_place_annotations",)

#: Entity/term name fragments that mark a range as a *goal* region even though
#: it is randomized by a plain reset event. Several tasks implement the goal as
#: a scene asset (``success_marker``, ``goal_tee``) rather than a command term,
#: and the operator cares about the spawn/goal distinction, not the mechanism.
_GOAL_NAME_HINTS = ("goal", "success_marker", "target")


def _kind_for(term_name: str, entity: str) -> str:
    haystack = f"{term_name} {entity}".lower()
    return "goal" if any(hint in haystack for hint in _GOAL_NAME_HINTS) else "spawn"

#: Colors keyed by range kind, chosen to stay distinct from the existing
#: red forbidden-zone / green-blue goal markers already in the scene.
DEFAULT_COLORS: dict[str, tuple[float, float, float]] = {
    "spawn": (0.15, 0.75, 1.00),  # cyan  -- where an entity may be spawned
    "goal": (1.00, 0.80, 0.10),  # amber -- where the goal may be sampled
}


@dataclass
class RangeBox:
    """One axis-aligned randomization envelope, in world coordinates."""

    name: str
    kind: str  # "spawn" | "goal"
    entity: str
    center: tuple[float, float, float]
    half_extent: tuple[float, float, float]
    #: Set when the range samples orientation too -- recorded for the readout,
    #: not drawn (a yaw fan would clutter the box it belongs to).
    rot_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: True when the box is degenerate in all three axes (a fixed pose). These
    #: are kept for reporting but skipped by the renderer.
    degenerate: bool = False


def _as_pair(value) -> tuple[float, float]:
    if value is None:
        return (0.0, 0.0)
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, IndexError, ValueError):
        return (0.0, 0.0)
    return (lo, hi) if lo <= hi else (hi, lo)


def _pos_ranges(pose_range: dict) -> tuple[tuple[float, float], ...]:
    return tuple(_as_pair(pose_range.get(axis)) for axis in ("x", "y", "z"))


def _rot_ranges(pose_range: dict) -> dict[str, tuple[float, float]]:
    out = {}
    for key in ("roll", "pitch", "yaw"):
        lo, hi = _as_pair(pose_range.get(key))
        if lo != hi:
            out[key] = (lo, hi)
    return out


def _entity_name(params: dict) -> str | None:
    for key in ("asset_cfg", "object_cfg", "reference_asset_cfg"):
        value = params.get(key)
        if isinstance(value, SceneEntityCfg):
            return value.name
    return None


def _box_from_bounds(
    name: str,
    kind: str,
    entity: str,
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
    rot_range: dict,
) -> RangeBox:
    center = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    half = tuple((hi[i] - lo[i]) * 0.5 for i in range(3))
    return RangeBox(
        name=name,
        kind=kind,
        entity=entity,
        center=center,  # type: ignore[arg-type]
        half_extent=half,  # type: ignore[arg-type]
        rot_range=rot_range,
        degenerate=all(h <= 1.0e-9 for h in half),
    )


def collect_range_boxes(env, env_index: int = 0) -> list[RangeBox]:
    """Resolve every randomization envelope in ``env`` to world-frame boxes.

    ``env_index`` selects which environment instance the boxes describe; teleop
    runs with ``num_envs=1`` so the default is the only one that exists there.
    Unknown reset functions are skipped with a warning rather than guessed at,
    since a box drawn under the wrong frame convention is worse than no box.
    """
    boxes: list[RangeBox] = []
    origin = env.scene.env_origins[env_index].detach().cpu().numpy()

    events_cfg = getattr(env.cfg, "events", None)
    for term_name, term in vars(events_cfg).items() if events_cfg is not None else []:
        params = getattr(term, "params", None)
        func = getattr(term, "func", None)
        if not isinstance(params, dict) or func is None:
            continue
        pose_range = params.get("pose_range")
        if not isinstance(pose_range, dict):
            continue
        entity = _entity_name(params)
        if entity is None or entity not in env.scene.keys():
            continue

        func_name = getattr(func, "__name__", str(func))
        rot_range = _rot_ranges(pose_range)
        (xr, yr, zr) = _pos_ranges(pose_range)

        if func_name in _ANNOTATION_ANCHORED:
            positions = params.get("per_env_positions") or []
            if not positions:
                continue
            anchor = positions[min(env_index, len(positions) - 1)]
            support_z = params.get("support_z")
            base = (
                origin[0] + float(anchor[0]),
                origin[1] + float(anchor[1]),
                float(support_z) if support_z is not None else origin[2] + float(anchor[2]),
            )
        elif func_name in _DEFAULT_ROOT_ANCHORED:
            asset = env.scene[entity]
            default_pos = asset.data.default_root_state[env_index, :3].detach().cpu().numpy()
            base = (
                float(default_pos[0]) + origin[0],
                float(default_pos[1]) + origin[1],
                float(default_pos[2]) + origin[2],
            )
        else:
            logger.warning(
                "range_vis: skipping event '%s' -- unrecognized reset function '%s'; "
                "its pose_range frame convention is unknown.",
                term_name,
                func_name,
            )
            continue

        boxes.append(
            _box_from_bounds(
                name=term_name,
                kind=_kind_for(term_name, entity),
                entity=entity,
                lo=(base[0] + xr[0], base[1] + yr[0], base[2] + zr[0]),
                hi=(base[0] + xr[1], base[1] + yr[1], base[2] + zr[1]),
                rot_range=rot_range,
            )
        )

    commands_cfg = getattr(env.cfg, "commands", None)
    for cmd_name, cmd in vars(commands_cfg).items() if commands_cfg is not None else []:
        ranges = getattr(cmd, "ranges", None)
        if ranges is None:
            continue
        px, py, pz = (_as_pair(getattr(ranges, f"pos_{a}", None)) for a in ("x", "y", "z"))
        rot_range = {
            key: (lo, hi)
            for key in ("roll", "pitch", "yaw")
            for lo, hi in [_as_pair(getattr(ranges, key, None))]
            if lo != hi
        }
        # Sampled straight into world coordinates; env_origins is deliberately
        # not added (see module docstring).
        offset = (0.0, 0.0, 0.0) if getattr(cmd, "use_world_frame", False) else tuple(origin)
        boxes.append(
            _box_from_bounds(
                name=cmd_name,
                kind="goal",
                entity=getattr(cmd, "asset_name", "") or "",
                lo=(offset[0] + px[0], offset[1] + py[0], offset[2] + pz[0]),
                hi=(offset[0] + px[1], offset[1] + py[1], offset[2] + pz[1]),
                rot_range=rot_range,
            )
        )

    return boxes


def describe_boxes(boxes: list[RangeBox]) -> str:
    """Human-readable digest of the collected ranges, for the console."""
    if not boxes:
        return "range_vis: no randomization ranges found for this task."
    lines = [f"range_vis: {len(boxes)} randomization range(s):"]
    for b in boxes:
        span = tuple(round(2.0 * h, 4) for h in b.half_extent)
        centre = tuple(round(c, 4) for c in b.center)
        flag = "  [fixed pose]" if b.degenerate else ""
        rot = f"  rot={ {k: tuple(round(v, 3) for v in vs) for k, vs in b.rot_range.items()} }" if b.rot_range else ""
        lines.append(f"  - {b.kind:5s} {b.name} ({b.entity}): centre={centre} span={span}{rot}{flag}")
    return "\n".join(lines)


def _edge_instances(box: RangeBox, thickness: float):
    """Twelve edge bars for one box: (translation, scale) per edge."""
    cx, cy, cz = box.center
    hx, hy, hz = box.half_extent
    half = (hx, hy, hz)
    out = []
    for axis in range(3):
        u, v = [i for i in range(3) if i != axis]
        # Full length along `axis`; thickness padded so corners meet cleanly.
        scale = [thickness, thickness, thickness]
        scale[axis] = max(2.0 * half[axis] + thickness, thickness)
        for su in (-1.0, 1.0):
            for sv in (-1.0, 1.0):
                pos = [cx, cy, cz]
                pos[u] += su * half[u]
                pos[v] += sv * half[v]
                out.append((pos, list(scale)))
    return out


class randomization_ranges_vis(ManagerTermBase):
    """Outline every randomization range as a wireframe box.

    Side-effect-only observation term (returns a zero-width tensor), matching
    the ``forbidden_zones_vis`` convention already used in this repo. One
    instanced ``VisualizationMarkers`` per range kind holds all edge bars, so
    the whole overlay costs two prototypes regardless of how many ranges a task
    declares.

    The boxes are static, so geometry is computed once on the first call --
    ``default_root_state`` is only meaningful after the scene is initialized,
    which is why this is lazy rather than done in ``__init__``.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._thickness = float(cfg.params.get("thickness", 0.004))
        self._colors = dict(DEFAULT_COLORS)
        self._colors.update(cfg.params.get("colors", {}) or {})
        self._opacity = float(cfg.params.get("opacity", 1.0))
        self._prim_path_prefix = cfg.params.get("prim_path_prefix", "/Visuals/RandomizationRange")
        self._verbose = bool(cfg.params.get("verbose", True))
        self._hide_from_cameras = bool(cfg.params.get("hide_from_cameras", True))
        self._visualizers: dict[str, object] = {}
        self._instances: dict[str, tuple] = {}
        self._built = False
        self.boxes: list[RangeBox] = []

    def _build(self, env) -> None:
        import isaaclab.sim as sim_utils
        if type(env.cfg).__module__.startswith("dexverse.baseline_v1."):
            from dexverse.baseline_v1.visual_purpose import hide_marker_from_cameras
        else:
            from dexverse.visual_purpose import hide_marker_from_cameras
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self._built = True
        self.boxes = collect_range_boxes(env)
        if self._verbose:
            print(describe_boxes(self.boxes))

        drawable = [b for b in self.boxes if not b.degenerate]
        for kind in sorted({b.kind for b in drawable}):
            translations, scales = [], []
            for box in (b for b in drawable if b.kind == kind):
                for pos, scale in _edge_instances(box, self._thickness):
                    translations.append(pos)
                    scales.append(scale)
            if not translations:
                continue
            color = self._colors.get(kind, (1.0, 1.0, 1.0))
            marker_cfg = VisualizationMarkersCfg(
                prim_path=f"{self._prim_path_prefix}/{kind}",
                markers={
                    "edge": sim_utils.CuboidCfg(
                        # Unit prototype: per-instance `scales` gives each edge
                        # bar its own dimensions from this single prototype.
                        size=(1.0, 1.0, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=tuple(color),
                            emissive_color=tuple(color),
                            opacity=self._opacity,
                        ),
                    )
                },
            )
            visualizer = VisualizationMarkers(marker_cfg)
            if self._hide_from_cameras:
                # Operator-only geometry: keep it out of every recorded RGB/depth
                # observation, exactly as the zone markers do. Turning this off is
                # only useful for capturing a screenshot of the overlay itself.
                hide_marker_from_cameras(visualizer)
            self._visualizers[kind] = visualizer
            self._instances[kind] = (
                torch.tensor(translations, device=env.device, dtype=torch.float32),
                torch.tensor(scales, device=env.device, dtype=torch.float32),
            )

    def __call__(
        self,
        env,
        thickness: float | None = None,
        colors: dict | None = None,
        opacity: float | None = None,
        prim_path_prefix: str | None = None,
        verbose: bool | None = None,
        hide_from_cameras: bool | None = None,
    ) -> torch.Tensor:
        # Parameters are consumed in __init__ (the markers are built once);
        # they are declared here because ObservationManager validates that every
        # cfg param maps to a named argument and rejects **kwargs outright.
        if not self._built:
            try:
                self._build(env)
            except Exception as exc:  # never let an overlay break a teleop session
                logger.warning("range_vis: disabled after build failure: %s", exc)
                self._built = True
        for kind, visualizer in self._visualizers.items():
            translations, scales = self._instances[kind]
            visualizer.visualize(
                translations=translations,
                scales=scales,
                marker_indices=torch.zeros(translations.shape[0], dtype=torch.int32, device=env.device),
            )
        return torch.zeros(env.num_envs, 0, device=env.device)


def attach_range_vis(env_cfg, **params) -> bool:
    """Register :class:`randomization_ranges_vis` on ``env_cfg``.

    Call before ``gym.make``. Returns whether the term was added. The term goes
    into ``observations.scene_vis``, which observation presets never null out,
    so the overlay survives whichever preset the task selects.
    """
    from isaaclab.managers import ObservationTermCfg as ObsTerm

    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        logger.warning("range_vis: env cfg exposes no observations group; overlay not attached.")
        return False

    if getattr(observations, "scene_vis", None) is None:
        import dexverse.tasks.dexverse_base_env_cfg as dexverse_base_env

        observations.scene_vis = dexverse_base_env.ObservationsCfg.SceneVisObsCfg()

    observations.scene_vis.randomization_ranges_vis = ObsTerm(
        func=randomization_ranges_vis, params=dict(params)
    )
    return True
