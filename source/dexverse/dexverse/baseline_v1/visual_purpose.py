# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tag prims as USD ``purpose='guide'``.

Cameras (the RTX render product behind ``TiledCamera``) ignore ``guide``
prims by default, while the Kit viewport shows them. Use this to hide
visualization-only assets (camera-body cuboids, goal markers, fingertip
spheres, frame markers, forbidden-zone shells, …) from policy/recorded
RGB without hiding them from the human teleoperator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.markers import VisualizationMarkers


#: Process-wide switch. When ``False``, :func:`hide_marker_from_cameras` /
#: :func:`set_prim_purpose_guide` become no-ops, so operator cues also show up
#: in camera RGB. Escape hatch for teleop set-ups whose XR render path does not
#: display ``guide`` prims (``scripts/teleop_agent.py --cues_in_rgb``).
CAMERA_HIDING_ENABLED: bool = True


def set_camera_hiding_enabled(enabled: bool) -> None:
    """Enable/disable camera hiding of visualization markers process-wide."""
    global CAMERA_HIDING_ENABLED
    CAMERA_HIDING_ENABLED = bool(enabled)


def set_prim_purpose_guide(prim_path: str) -> None:
    """Set ``purpose=guide`` on the prim at ``prim_path`` **and every imageable
    descendant** (no-op if missing).

    Purpose is inheritable in USD namespace, but the RTX render products behind
    ``TiledCamera`` read the *authored* purpose of the prims they draw --
    ``VisualizationMarkers`` are ``PointInstancer`` prims whose prototypes are
    child Xforms/meshes, and tagging only the instancer root left the instances
    visible in recorded RGB (verified Aug 2026: 440 marker pixels with the root
    tag alone, 0 with descendants tagged). Authoring the tag down the subtree
    fixes that for every marker style.
    """
    if not CAMERA_HIDING_ENABLED:
        return
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)


def hide_marker_from_cameras(marker: VisualizationMarkers) -> None:
    """Tag a ``VisualizationMarkers`` so its instances are camera-invisible."""
    set_prim_purpose_guide(marker.prim_path)
