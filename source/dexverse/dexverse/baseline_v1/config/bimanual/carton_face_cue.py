# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Task geometry for the carton's recorded target face, visible in RGB/depth."""

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


class CartonFaceCornerCue(ManagerTermBase):
    """Four ordinary, non-colliding spheres parented to the carton rigid body.

    Parenting follows both physics and set-state replay without per-step world
    transforms or Fabric changes. Only a changed/restored face choice updates
    local offsets. That choice is already recorded as an episode task buffer.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        import isaaclab.sim as sim_utils
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        self._buffer_name = cfg.params["buffer_name"]
        self._points = cfg.params["points_per_choice"]
        obj = env.scene[cfg.params["object_cfg"].name]
        stage = omni.usd.get_context().get_stage()
        sphere_cfg = sim_utils.SphereCfg(
            radius=cfg.params["radius"],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg.params["color"]),
        )
        self._translations = []
        for body_path in obj.root_physx_view.prim_paths:
            # Local geometry constants are in metres after carton scaling.
            # Remove inherited scale for the cue, retaining rigid-body pose.
            body = stage.GetPrimAtPath(body_path)
            transform = Gf.Transform(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            scale = tuple(float(value) for value in transform.GetScale())
            root = UsdGeom.Xform.Define(stage, f"{body_path}/TargetFaceCorners")
            root.AddScaleOp().Set(Gf.Vec3f(*(1.0 / value for value in scale)))
            ops = []
            for index in range(4):
                prim = sphere_cfg.func(f"{root.GetPath()}/Corner_{index}", sphere_cfg)
                ops.append(next(op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate))
            self._translations.append(ops)
        self._last_choice = [-1] * env.num_envs
        self(env, **cfg.params)

    def __call__(self, env, points_per_choice: list, buffer_name: str,
                 object_cfg: SceneEntityCfg, radius: float, color: tuple) -> torch.Tensor:
        from pxr import Gf

        choice = getattr(env, self._buffer_name, None)
        choices = [0] * env.num_envs if choice is None else choice.detach().cpu().tolist()
        for index, face in enumerate(choices):
            if face != self._last_choice[index]:
                for op, point in zip(self._translations[index], self._points[face], strict=True):
                    op.Set(Gf.Vec3d(*point))
                self._last_choice[index] = face
        return torch.zeros(env.num_envs, 0, device=env.device)
