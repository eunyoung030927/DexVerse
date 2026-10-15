# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural props for the "cut" tasks (utility knife seam, scissors strip).

Two generators, both one kinematic rigid body of box prims:

- :func:`build_taped_seam_workpiece` -- a "mimicked taped box" for the
  utility-knife cut task: two cuboid blocks separated by a thin open slot
  (the seam), covered by a thin ``tape`` prim. The tape prim is authored as a
  normal collider; at env setup the task collision-filters it against the
  knife's *blade* link only, so the extended blade phantoms through the tape
  into the slot (as if cutting) while the knife body, fingers and table still
  collide with it. The slot walls are unfiltered and physically guide the
  blade along the seam. The slot has no floor -- standing on the table, the
  tabletop bounds the blade's dip.

- :func:`build_strip_stand` -- a stand (base + post + clamp) holding a
  cantilevered paper ``strip`` for the scissors cut task. The strip is its own
  colliding prim, so the complete scissors and the paper exchange contact
  forces; the cut point on the strip is exported analytically.

Local frames: origin at the centre of the footprint on the *bottom* face
(z = 0 where the prop stands); +z up. The seam runs along local **+y**
(matching the knife's long-axis convention); the strip cantilevers along
local **-x** (the "open" side to present to the robot).

Task configs call the ``ensure_*`` helpers at construction time, which build
the files on first use (the assets tree is not tracked by git; this module
is).
"""

from __future__ import annotations

import json
from pathlib import Path

_AUTHORED_DIR = Path(__file__).resolve().parent / "core_assets" / "dexverse_authored"
CUT_WORKPIECE_ASSET_DIR = _AUTHORED_DIR / "cut_workpiece"
STRIP_STAND_ASSET_DIR = _AUTHORED_DIR / "strip_stand"


def _new_stage(usd_path: Path, name: str, mass: float):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    root.GetPrim().CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    UsdGeom.Scope.Define(stage, f"/{name}/Looks")
    return stage


def _make_material(stage, name: str, mname: str, rgb: tuple[float, float, float]):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, f"/{name}/Looks/{mname}")
    sh = UsdShade.Shader.Define(stage, f"/{name}/Looks/{mname}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _add_box(stage, name: str, prim_name: str, center, size, mat):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, f"/{name}/{prim_name}")
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)


def build_taped_seam_workpiece(
    out_dir: Path,
    name: str = "taped_seam_workpiece",
    seam_length: float = 0.15,
    slot_width: float = 0.012,
    block_width: float = 0.07,
    block_height: float = 0.045,
    tape_thickness: float = 0.0025,
    tape_width: float = 0.05,
    block_color: tuple[float, float, float] = (0.72, 0.55, 0.34),
    tape_color: tuple[float, float, float] = (0.93, 0.78, 0.18),
) -> tuple[Path, Path]:
    """Write ``<name>.usda`` + ``<name>_manifest.json`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    usd_path = out_dir / f"{name}.usda"
    manifest_path = out_dir / f"{name}_manifest.json"

    stage = _new_stage(usd_path, name, mass=4.0)
    mat_block = _make_material(stage, name, "BlockMat", block_color)
    mat_tape = _make_material(stage, name, "TapeMat", tape_color)

    bx = (slot_width + block_width) / 2.0
    _add_box(stage, name, "block_left", (bx, 0.0, block_height / 2.0), (block_width, seam_length, block_height), mat_block)
    _add_box(stage, name, "block_right", (-bx, 0.0, block_height / 2.0), (block_width, seam_length, block_height), mat_block)
    _add_box(
        stage,
        name,
        "tape",
        (0.0, 0.0, block_height + tape_thickness / 2.0),
        (tape_width, seam_length, tape_thickness),
        mat_tape,
    )
    stage.GetRootLayer().Save()

    x_half = slot_width / 2.0 + block_width
    manifest = {
        "name": name,
        "usd": usd_path.name,
        "frame": "origin at footprint centre on the bottom face; seam along +y, +z up",
        "params": {
            "seam_length": seam_length,
            "slot_width": slot_width,
            "block_width": block_width,
            "block_height": block_height,
            "tape_thickness": tape_thickness,
            "tape_width": tape_width,
            "block_color": list(block_color),
            "tape_color": list(tape_color),
        },
        "footprint": {"x_half": x_half, "y_half": seam_length / 2.0, "height": block_height + tape_thickness},
        "seam": {
            "start": [0.0, -seam_length / 2.0, block_height],
            "end": [0.0, seam_length / 2.0, block_height],
            "axis_local": [0.0, 1.0, 0.0],
            "length": seam_length,
        },
        "slot": {
            "center": [0.0, 0.0, block_height / 2.0],
            "half_extents": [slot_width / 2.0, seam_length / 2.0, block_height / 2.0],
        },
        "z_block_top": block_height,
        "z_tape_top": block_height + tape_thickness,
        "tape_prim": "tape",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return usd_path, manifest_path


def build_strip_stand(
    out_dir: Path,
    name: str = "strip_stand",
    base_size: float = 0.12,
    base_height: float = 0.02,
    post_size: float = 0.03,
    post_height: float = 0.115,
    clamp_size: tuple[float, float, float] = (0.04, 0.035, 0.015),
    strip_length: float = 0.14,
    strip_width: float = 0.03,
    strip_thickness: float = 0.0015,
    cut_point_inset: float = 0.025,
    stand_color: tuple[float, float, float] = (0.45, 0.45, 0.48),
    strip_color: tuple[float, float, float] = (0.95, 0.95, 0.92),
) -> tuple[Path, Path]:
    """Write ``<name>.usda`` + ``<name>_manifest.json`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    usd_path = out_dir / f"{name}.usda"
    manifest_path = out_dir / f"{name}_manifest.json"

    stage = _new_stage(usd_path, name, mass=2.0)
    mat_stand = _make_material(stage, name, "StandMat", stand_color)
    mat_strip = _make_material(stage, name, "StripMat", strip_color)

    post_top = base_height + post_height
    clamp_cx, clamp_cy, clamp_cz = clamp_size
    z_strip = post_top + clamp_cz / 2.0  # strip is clamped at the clamp's mid-height
    # Strip runs from inside the clamp footprint out along -x.
    strip_x0 = clamp_cx / 2.0 - 0.01  # 1 cm of the strip inside the clamp
    strip_center_x = strip_x0 - strip_length / 2.0
    free_edge_x = strip_x0 - strip_length

    _add_box(stage, name, "base", (0.0, 0.0, base_height / 2.0), (base_size, base_size, base_height), mat_stand)
    _add_box(stage, name, "post", (0.0, 0.0, base_height + post_height / 2.0), (post_size, post_size, post_height), mat_stand)
    _add_box(stage, name, "clamp", (0.0, 0.0, post_top + clamp_cz / 2.0), clamp_size, mat_stand)
    _add_box(
        stage,
        name,
        "strip",
        (strip_center_x, 0.0, z_strip),
        (strip_length, strip_width, strip_thickness),
        mat_strip,
    )
    stage.GetRootLayer().Save()

    manifest = {
        "name": name,
        "usd": usd_path.name,
        "frame": "origin at footprint centre on the bottom face; strip cantilevers along -x, +z up",
        "params": {
            "base_size": base_size,
            "base_height": base_height,
            "post_size": post_size,
            "post_height": post_height,
            "clamp_size": list(clamp_size),
            "strip_length": strip_length,
            "strip_width": strip_width,
            "strip_thickness": strip_thickness,
            "cut_point_inset": cut_point_inset,
            "stand_color": list(stand_color),
            "strip_color": list(strip_color),
        },
        "footprint": {"x_half": base_size / 2.0, "y_half": base_size / 2.0, "height": post_top + clamp_cz},
        "z_strip": z_strip,
        "strip_axis_local": [-1.0, 0.0, 0.0],
        "cut_point": [free_edge_x + cut_point_inset, 0.0, z_strip],
        "free_edge_local": [free_edge_x, 0.0, z_strip],
        "strip_prim": "strip",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return usd_path, manifest_path


def _ensure_asset(build_fn, default_dir: Path, name: str, out_dir: Path | None, params: dict) -> tuple[Path, dict]:
    out_dir = Path(out_dir) if out_dir is not None else default_dir
    usd_path = out_dir / f"{name}.usda"
    manifest_path = out_dir / f"{name}_manifest.json"
    if usd_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        stored = manifest.get("params", {})

        def _norm(v):
            # JSON round-trips tuples as lists; compare like-for-like.
            return list(v) if isinstance(v, tuple) else v

        if all(k in stored and stored[k] == _norm(v) for k, v in params.items()):
            return usd_path, manifest
    usd_path, manifest_path = build_fn(out_dir, name=name, **params)
    return usd_path, json.loads(manifest_path.read_text())


def ensure_taped_seam_workpiece_asset(
    name: str = "taped_seam_workpiece", out_dir: Path | None = None, **params
) -> tuple[Path, dict]:
    """Return ``(usd_path, manifest)`` for a taped-seam workpiece, generating it
    if missing or if the stored parameters differ from ``params``."""
    return _ensure_asset(build_taped_seam_workpiece, CUT_WORKPIECE_ASSET_DIR, name, out_dir, params)


def ensure_strip_stand_asset(name: str = "strip_stand", out_dir: Path | None = None, **params) -> tuple[Path, dict]:
    """Return ``(usd_path, manifest)`` for a strip-on-a-stand prop, generating
    it if missing or if the stored parameters differ from ``params``."""
    return _ensure_asset(build_strip_stand, STRIP_STAND_ASSET_DIR, name, out_dir, params)
