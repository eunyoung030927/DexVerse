# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mass of the robot's dummy links (no mass in the USD).

PhysX gives a rigid body without mass properties 1 kg and an inertia of 1e-4 kg m^2. Some hand USDs have such
dummy links on the fingers: LEAP's ``*_virtual_link`` (1 cm visual-only spheres) and ``*_tip_head`` and Inspire's
``*_tip`` markers (thumb..pinky). With 1 kg each the LEAP hand weighs 23.7 kg and the Inspire hand 13.2 kg in the
simulator, the finger joints carry ~100x their real inertia and the tactile pads feel the dummy masses.

``RobotConfig.dummy_mass_body_names`` (regex list) names those links; the base env cfg then adds the startup event
:func:`set_link_mass` (0.01 g, inertia scaled with the mass): the official URDFs (LEAP_Hand_Sim's source of these
links, dex-urdf, Unitree / ManiSkill Inspire) give them no ``<inertial>`` and attach them with fixed joints, i.e.
no mass; PhysX needs a positive one. ``DEXVERSE_DUMMY_LINK_MASS=legacy`` keeps the 1 kg.
"""

from __future__ import annotations

import os

import torch

DUMMY_LINK_MASS_KG = 1.0e-5


def dummy_link_mass_enabled() -> bool:
    return os.environ.get("DEXVERSE_DUMMY_LINK_MASS", "").strip().lower() != "legacy"


def set_link_mass(env, env_ids, asset_cfg, mass: float = DUMMY_LINK_MASS_KG):
    """Startup event: set the mass of ``asset_cfg.body_ids`` to ``mass``, inertia scaled by the same factor."""
    asset = env.scene[asset_cfg.name]
    view = asset.root_physx_view
    ids = asset_cfg.body_ids
    masses, inertias = view.get_masses().clone(), view.get_inertias().clone()
    scale = mass / masses[:, ids].clamp_min(1e-9)
    masses[:, ids] = mass
    inertias[:, ids] *= scale[..., None]
    idx = torch.arange(masses.shape[0])
    view.set_masses(masses, idx)
    view.set_inertias(inertias, idx)
