# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

import gymnasium as gym
from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp", "_archive"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

# Keep the established task-discovery entry point registering both versions.
# Only baseline upgrades live in this sibling package; original tasks stay here.
from dexverse import baseline_v1  # noqa: E402, F401
