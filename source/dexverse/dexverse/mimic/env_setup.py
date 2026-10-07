# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a DexVerse task as an Isaac Lab Mimic env (post-app): config, subtask spec and recorders.

The task config is built like ``record_demos.py`` builds it (``parse_env_cfg`` + ``robot_type`` rebuild, CPU
physics by default) so replays and generated demos stay bit-exact with the recording machine.

Subtask spec (per hand): by default two object-centric subtasks on the task object ``object`` --
(1) approach and grasp, ending when the object is lifted (``object_lifted``), (2) the rest (lift / pour / place)
re-anchored on the object pose at that moment. ``--subtasks 1`` uses a single subtask for the whole demo.
"""

from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import DataGenConfig, MimicEnvCfg, SubTaskConfig


def build_env_cfg(task: str, robot_type: str | None, *, device: str = "cpu", num_envs: int = 1,
                  seed: int | None = None, keep_cameras: bool = True):
    """Task config as record_demos builds it, cameras stripped; returns ``(env_cfg, success_term)``."""
    from dexverse.tasks.utils import prune_stale_obs_refs, strip_camera_cfgs  # noqa: PLC0415
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: PLC0415

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    if robot_type:
        env_cfg = type(env_cfg)(robot_type=robot_type)
        env_cfg.sim.device = device
        env_cfg.scene.num_envs = num_envs
    env_cfg.env_name = task.split(":")[-1]
    if seed is not None:
        env_cfg.seed = seed
    if ("TopDownGrasp" in task or "Lift" in task) and hasattr(getattr(env_cfg, "commands", None), "object_pose"):
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is None:
        raise RuntimeError(f"{task} has no success termination; Mimic needs one to keep only successful demos")
    env_cfg.terminations = None
    if keep_cameras:
        # Rendering steps must match the recording/replay path (record_demos renders the task cameras every step);
        # keep the cameras but tiny and RGB-only, and drop depth / point-cloud observations.
        for name in ("third_person_camera", "wrist_camera", "right_wrist_camera", "left_wrist_camera"):
            cam = getattr(env_cfg.scene, name, None)
            if cam is not None:
                cam.height, cam.width, cam.data_types = 64, 64, ["rgb"]
        for group in ("depth", "pointcloud"):
            if getattr(env_cfg.observations, group, None) is not None:
                setattr(env_cfg.observations, group, None)
    else:
        env_cfg = strip_camera_cfgs(env_cfg)
        env_cfg = prune_stale_obs_refs(env_cfg)
    env_cfg.observations.policy.concatenate_terms = False
    return env_cfg, success_term


def attach_mimic_cfg(env_cfg, sides, *, num_subtasks: int = 2, num_trials: int = 10, seed: int = 1,
                     action_noise: float = 0.002, max_num_failures: int = 1000, nn_k: int = 3,
                     second_ref: str = "object"):
    """Give ``env_cfg`` the MimicEnvCfg fields the Isaac Lab datagen reads."""
    dg = DataGenConfig()
    dg.name = env_cfg.env_name
    dg.generation_guarantee = True          # stop after num_trials SUCCESSES
    dg.generation_keep_failed = False
    dg.generation_num_trials = int(num_trials)
    dg.generation_select_src_per_subtask = False
    dg.generation_select_src_per_arm = False
    dg.generation_transform_first_robot_pose = False
    dg.generation_interpolate_from_last_target_pose = True
    dg.max_num_failures = int(max_num_failures)
    dg.seed = int(seed)

    def subtask(term_signal, interp, ref="object"):
        return SubTaskConfig(
            object_ref=ref,
            subtask_term_signal=term_signal,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": nn_k},
            action_noise=action_noise,
            num_interpolation_steps=interp,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

    def specs():
        if num_subtasks == 2:
            return [subtask("object_lifted", 10), subtask(None, 5, ref=second_ref)]
        return [subtask(None, 10)]

    env_cfg.datagen_config = dg
    env_cfg.subtask_configs = {side: specs() for side in sides}
    env_cfg.task_constraint_configs = []
    env_cfg.mimic_recorder_config = None
    # The datagen asserts isinstance(cfg, MimicEnvCfg): re-class the task cfg instance as task cfg + MimicEnvCfg
    # (the same mix-in Isaac Lab's own *MimicEnvCfg classes use), keeping every field already set.
    cls = type(env_cfg)
    if not isinstance(env_cfg, MimicEnvCfg):
        env_cfg.__class__ = type(f"{cls.__name__}Mimic", (cls, MimicEnvCfg), {})
    return env_cfg


def layout_sides(robot_type: str) -> list[str]:
    from importlib import import_module  # noqa: PLC0415

    from dexverse.devices.retargeters.simple_relative_retargeting import (  # noqa: PLC0415
        SIMPLE_RETARGETER_LAYOUT_SOURCES,
    )

    module_name, attr = SIMPLE_RETARGETER_LAYOUT_SOURCES[robot_type]
    return list(getattr(import_module(module_name), attr)["hands"].keys())


def make_env(cfg):
    from dexverse.mimic.floating_hand_mimic_env import FloatingHandMimicEnv  # noqa: PLC0415

    return FloatingHandMimicEnv(cfg=cfg)


def resolve_success_term(env, term):
    """Instantiate a class-based success term (``ManagerTermBase``, e.g. ``lift_and_tilt_with_contact_zones``) so
    ``term.func(env, **term.params)`` works, as the Isaac Lab datagen calls it (same as record_demos)."""
    import inspect  # noqa: PLC0415

    from isaaclab.managers import ManagerTermBase  # noqa: PLC0415

    if inspect.isclass(term.func) and issubclass(term.func, ManagerTermBase):
        term.func = term.func(term, env)
    return term
