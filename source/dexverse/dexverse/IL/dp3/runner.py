# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DP3 inference runner for closed-loop evaluation in DexVerse environments."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .config import DP3PolicyConfig, PointcloudEncoderConfig
from .dp3_core.dp3_policy import DP3
from .dp3_core.normalizer import LinearNormalizer
from .model import build_dp3_policy, load_dp3_checkpoint


def _fps_downsample_torch(points: torch.Tensor, num_points: int) -> torch.Tensor:
    """Farthest-point-sample a (N, 3) tensor down to (num_points, 3)."""
    N = points.shape[0]
    if N <= num_points:
        if N == 0:
            return torch.zeros(num_points, 3, device=points.device, dtype=points.dtype)
        pad_idx = torch.randint(N, (num_points - N,), device=points.device)
        return torch.cat([points, points[pad_idx]], dim=0)

    try:
        import pytorch3d.ops as torch3d_ops

        sampled, _ = torch3d_ops.sample_farthest_points(points.unsqueeze(0).float(), K=num_points)
        return sampled.squeeze(0).to(dtype=points.dtype)
    except ImportError:
        pass

    selected = torch.zeros(num_points, dtype=torch.long, device=points.device)
    dists = torch.full((N,), float("inf"), device=points.device)
    selected[0] = torch.randint(N, (1,), device=points.device)
    for i in range(1, num_points):
        last = points[selected[i - 1]]
        d = ((points - last) ** 2).sum(dim=-1)
        dists = torch.minimum(dists, d)
        selected[i] = dists.argmax()
    return points[selected]


class DexVerseDP3Runner:
    """Closed-loop DP3 inference for DexVerse environments.

    Reads point clouds from the ``pointcloud`` obs group and proprioception
    from the ``proprio`` obs group (as produced by the ``pointcloud`` or
    ``3view_pointcloud`` observation preset), maintains a short obs history
    window, and returns actions with temporal chunking.
    """

    def __init__(
        self,
        *,
        policy: DP3,
        device: torch.device,
        num_points: int = 512,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        replan_interval: int = 8,
    ):
        self.policy = policy
        self.device = device
        self.num_points = num_points
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.replan_interval = replan_interval

        self.pcd_history: deque[torch.Tensor] = deque(maxlen=n_obs_steps)
        self.proprio_history: deque[torch.Tensor] = deque(maxlen=n_obs_steps)
        self.action_queue: deque[torch.Tensor] = deque()
        self.replan_tick: int = 0

    def reset(self) -> None:
        """Clear state between episodes."""
        self.pcd_history.clear()
        self.proprio_history.clear()
        self.action_queue.clear()
        self.replan_tick = 0

    @torch.no_grad()
    def act(self, obs: dict) -> torch.Tensor:
        """Compute an action from one environment observation dict.

        Expects:
          ``obs["pointcloud"]`` — concatenated point cloud for the active term
              (``camera_point_cloud_w`` or ``merged_point_cloud_w``). Shape is
              ``(num_envs, num_points * 3)`` (flat, from ``concatenate_terms``),
              ``(num_envs, num_points, 3)``, or a dict mapping term-name → tensor.
          ``obs["proprio"]`` — flat concatenated proprio tensor of shape
              ``(num_envs, proprio_dim)`` (history is already flattened in by
              the observation preset).

        Returns an action tensor of shape ``(action_dim,)`` for env_id=0.
        """
        pcd = self._extract_point_cloud(obs)
        pcd_down = _fps_downsample_torch(pcd, self.num_points)
        self.pcd_history.append(pcd_down.cpu())

        proprio = self._extract_proprio(obs)
        self.proprio_history.append(proprio.cpu())

        while len(self.pcd_history) < self.n_obs_steps:
            self.pcd_history.appendleft(self.pcd_history[0])
        while len(self.proprio_history) < self.n_obs_steps:
            self.proprio_history.appendleft(self.proprio_history[0])

        should_replan = len(self.action_queue) == 0 or self.replan_tick >= self.replan_interval
        if should_replan:
            pcd_window = torch.stack(list(self.pcd_history), dim=0)  # (To, N, 3)
            proprio_window = torch.stack(list(self.proprio_history), dim=0)  # (To, D)
            obs_dict = {
                "point_cloud": pcd_window.unsqueeze(0).to(self.device),
                "agent_pos": proprio_window.unsqueeze(0).to(self.device),
            }
            result = self.policy.predict_action(obs_dict)
            actions = result["action"][0]  # (n_action_steps, action_dim)
            self.action_queue = deque(actions.cpu().unbind(dim=0))
            self.replan_tick = 0

        self.replan_tick += 1
        return self.action_queue.popleft()

    def _extract_point_cloud(self, obs: dict) -> torch.Tensor:
        """Return the env_0 point cloud as ``(N, 3)`` on CPU.

        The ``pointcloud`` obs group is configured with ``concatenate_terms=True``
        and one active term (single- or multi-view), so the runtime tensor is
        ``(num_envs, num_points * 3)``. We also accept the un-concatenated and
        nested-dict shapes for robustness.
        """
        raw = obs.get("pointcloud")
        if raw is None:
            raise ValueError(
                "No 'pointcloud' group in observation dict. "
                "Apply the 'pointcloud' or '3view_pointcloud' observation preset "
                "to the env cfg before gym.make()."
            )
        if isinstance(raw, dict):
            for term in ("camera_point_cloud_w", "merged_point_cloud_w"):
                if term in raw:
                    raw = raw[term]
                    break
            else:
                raise ValueError(
                    "pointcloud dict has none of (camera_point_cloud_w, "
                    f"merged_point_cloud_w); got keys {list(raw.keys())}"
                )
        pcd = torch.as_tensor(raw, dtype=torch.float32)
        if pcd.ndim == 3 and pcd.shape[-1] == 3:
            pcd = pcd[0]
        elif pcd.ndim == 2:
            pcd = pcd[0].reshape(-1, 3)
        elif pcd.ndim == 1:
            pcd = pcd.reshape(-1, 3)
        else:
            raise ValueError(f"Unexpected pointcloud tensor shape: {tuple(pcd.shape)}")
        return pcd.cpu()

    def _extract_proprio(self, obs: dict) -> torch.Tensor:
        """Return the env_0 proprio vector as ``(D,)`` on CPU.

        With ``concatenate_terms=True`` (and history flattening from the obs
        preset) the runtime tensor is already flat ``(num_envs, D)``. Nested
        dicts and per-env tensors are also accepted.
        """
        proprio = obs.get("proprio")
        if proprio is None:
            raise ValueError(
                "No 'proprio' group in observation dict. The obs preset must "
                "keep 'proprio' enabled (the default presets all do)."
            )
        if isinstance(proprio, dict):
            parts = []
            for key in sorted(proprio.keys()):
                t = torch.as_tensor(proprio[key], dtype=torch.float32)
                if t.ndim >= 2:
                    t = t[0]
                parts.append(t.flatten())
            return torch.cat(parts, dim=0).cpu()
        t = torch.as_tensor(proprio, dtype=torch.float32)
        if t.ndim >= 2:
            t = t[0]
        return t.flatten().cpu()


def load_runner_from_checkpoint(
    ckpt_path: str,
    *,
    device: str = "cuda",
    replan_interval: int = 8,
) -> tuple[DexVerseDP3Runner, dict]:
    """Load a trained DP3 policy and create an inference runner."""
    ckpt = load_dp3_checkpoint(ckpt_path, device=device)

    raw_cfg = ckpt["policy_cfg"]
    if isinstance(raw_cfg.get("pointcloud_encoder_cfg"), dict):
        raw_cfg = dict(raw_cfg)
        raw_cfg["pointcloud_encoder_cfg"] = PointcloudEncoderConfig(**raw_cfg["pointcloud_encoder_cfg"])
    if isinstance(raw_cfg.get("down_dims"), list):
        raw_cfg = dict(raw_cfg)
        raw_cfg["down_dims"] = tuple(raw_cfg["down_dims"])
    policy_cfg = DP3PolicyConfig(**raw_cfg)
    policy = build_dp3_policy(policy_cfg)

    state_key = "ema_policy_state" if "ema_policy_state" in ckpt else "policy_state"
    policy.load_state_dict(ckpt[state_key])
    policy.to(device)
    policy.eval()

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(ckpt["normalizer_state"])
    policy.set_normalizer(normalizer)

    dev = torch.device(device)
    runner = DexVerseDP3Runner(
        policy=policy,
        device=dev,
        num_points=policy_cfg.num_points,
        n_obs_steps=policy_cfg.n_obs_steps,
        n_action_steps=policy_cfg.n_action_steps,
        replan_interval=replan_interval,
    )
    return runner, ckpt
