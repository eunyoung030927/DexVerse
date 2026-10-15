# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State-based Diffusion Policy baseline.

Modules, in pipeline order:

* :mod:`obs_state` -- the state vector's layout, shared by data conversion and
  evaluation so the two cannot drift apart.
* :mod:`dataset` -- sliding (observation history, action chunk) windows over
  recorded demonstrations.
* :mod:`model` -- the conditional 1-D UNet denoiser.
* :mod:`train` -- the offline DDPM training loop.
* :mod:`runner` -- receding-horizon closed-loop control from a checkpoint.
* :mod:`online_eval` -- success-rate evaluation in simulation.

Entry points live in ``scripts/diffusion/``. Import submodules directly; nothing
is re-exported here so that ``obs_state`` stays importable without ``diffusers``.
"""
