# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mimic annotation recorders (same terms as Isaac Lab's ``annotate_demos.py``, which keeps them in the script)."""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Per step: object poses, eef (palm) poses and the target eef poses of the action being applied."""

    def record_pre_step(self):
        eef_pose = {name: self._env.get_robot_eef_pose(eef_name=name) for name in self._env.cfg.subtask_configs}
        return "obs/datagen_info", {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepSubtaskTermsRecorder


@configclass
class MimicAnnotationRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsRecorderCfg()
