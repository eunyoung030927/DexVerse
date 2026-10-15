# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online evaluation of a trained DP3 policy in a DexVerse simulator."""
from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from isaaclab.managers.manager_base import ManagerTermBase

from .config import DP3OnlineEvalConfig
from .runner import load_runner_from_checkpoint


def _write_episode_video(frames: list[np.ndarray], output_path: str | Path, fps: int = 30) -> None:
    """Write a list of ``(H, W, 3)`` uint8 RGB frames to an MP4 file."""
    import cv2

    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _read_camera_rgb(env, camera_name: str) -> np.ndarray:
    """Read one ``(H, W, 3)`` uint8 RGB frame from a named scene sensor."""
    cam = env.unwrapped.scene.sensors[camera_name]
    rgb = cam.data.output["rgb"][0].cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).clip(0, 255)
        rgb = rgb.clip(0, 255).astype(np.uint8)
    return rgb


class ManualSuccessEvaluator:
    """Evaluate episode success by directly calling the env's success term.

    A termination term's ``func`` is either a plain callable or a
    :class:`ManagerTermBase` subclass. Class-based terms must be instantiated
    once (``func(cfg=term_cfg, env=env)``) and then invoked per step, mirroring
    what IsaacLab's ``TerminationManager`` does internally. Calling the class
    directly as ``func(env, **params)`` raises a ``TypeError``.
    """

    def __init__(self, term_cfg: Any):
        self.term_cfg = term_cfg
        self._instance: Any = None

    def __call__(self, env: Any) -> bool:
        params = dict(getattr(self.term_cfg, "params", {}) or {})
        func = self.term_cfg.func
        if inspect.isclass(func) and issubclass(func, ManagerTermBase):
            if self._instance is None:
                self._instance = func(cfg=self.term_cfg, env=env)
            value = self._instance(env, **params)
        else:
            value = func(env, **params)
        if isinstance(value, torch.Tensor):
            return bool(value.reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return bool(value.reshape(-1)[0])
        if isinstance(value, (list, tuple)):
            return bool(value[0])
        return bool(value)


def _setup_env(
    task: str,
    *,
    device: str,
    observation_preset: str,
    json_path: str | None = None,
):
    """Create an env with the requested obs preset; extract a success evaluator.

    Pulls ``success`` out of the env cfg so we can score the rollout manually
    without time-out / failure terms ending the episode early. Other
    terminations (``time_out``, etc.) are left in place so the rollout still
    truncates instead of running forever.
    """
    import dexverse.tasks  # noqa: F401 — registers all tasks
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from dexverse.tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task, device=device, num_envs=1, json_path=json_path)

    # Apply the observation preset BEFORE building the env. The preset must
    # match what the demos were recorded with so train-time and eval-time obs
    # shapes line up.
    if observation_preset and hasattr(env_cfg, "_apply_observation_preset"):
        env_cfg.observation_preset = observation_preset
        env_cfg._apply_observation_preset(observation_preset)

    success_evaluator = None
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None and hasattr(terminations, "success"):
        success_cfg = terminations.success
        if success_cfg is not None:
            success_evaluator = ManualSuccessEvaluator(success_cfg)
            terminations.success = None

    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = {}

    env = gym.make(task, cfg=env_cfg)
    return env, success_evaluator


def eval_main(cfg: DP3OnlineEvalConfig) -> dict[str, Any]:
    """Run closed-loop evaluation of a DP3 policy."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    device_str = cfg.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"

    runner, ckpt = load_runner_from_checkpoint(
        cfg.ckpt,
        device=device_str,
        replan_interval=cfg.replan_interval,
    )
    metadata = ckpt.get("metadata", {})
    print(f"[dp3][eval] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"[dp3][eval] Task: {metadata.get('task', cfg.task)}")
    print(f"[dp3][eval] observation_preset={cfg.observation_preset!r}")

    json_path = getattr(cfg, "json_path", None)
    env, success_evaluator = _setup_env(
        cfg.task,
        device=device_str,
        observation_preset=cfg.observation_preset,
        json_path=json_path,
    )

    returns: list[float] = []
    lengths: list[int] = []
    latencies_ms: list[float] = []
    successes = 0

    video_dir = None
    if cfg.record_video:
        video_dir = os.path.join(cfg.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        print(f"[dp3][eval] Recording video from sensor '{cfg.video_camera}' @ {cfg.video_fps} fps -> {video_dir}")

    for ep in range(cfg.num_episodes):
        obs, _ = env.reset()
        runner.reset()
        ep_return = 0.0
        ep_success = False
        frames: list[np.ndarray] = []

        if cfg.record_video:
            frames.append(_read_camera_rgb(env, cfg.video_camera))

        for step in range(cfg.max_steps):
            start = time.perf_counter()
            action = runner.act(obs).to(env.unwrapped.device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

            obs, reward, terminated, truncated, _info = env.step(action.unsqueeze(0))

            if cfg.record_video:
                frames.append(_read_camera_rgb(env, cfg.video_camera))

            if isinstance(reward, torch.Tensor):
                ep_return += float(reward.reshape(-1)[0].item())
            else:
                ep_return += float(np.asarray(reward).reshape(-1)[0])

            if success_evaluator is not None:
                ep_success = ep_success or bool(success_evaluator(env.unwrapped))

            terminated_flag = bool(torch.as_tensor(terminated).reshape(-1)[0].item())
            truncated_flag = bool(torch.as_tensor(truncated).reshape(-1)[0].item())
            done = terminated_flag or truncated_flag or ep_success
            if done:
                lengths.append(step + 1)
                break
        else:
            lengths.append(cfg.max_steps)

        if ep_success:
            successes += 1
        returns.append(ep_return)

        status = "SUCCESS" if ep_success else "fail"
        print(f"[dp3][eval] Episode {ep + 1}/{cfg.num_episodes}: return={ep_return:.2f}  len={lengths[-1]}  {status}")

        if cfg.record_video and frames:
            vid_path = os.path.join(video_dir, f"ep{ep:03d}_{status.lower()}.mp4")
            _write_episode_video(frames, vid_path, fps=cfg.video_fps)
            print(f"  -> {vid_path}")

    env.close()

    metrics: dict[str, Any] = {
        "num_episodes": cfg.num_episodes,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_ep_len": float(np.mean(lengths)) if lengths else float(cfg.max_steps),
        "inference_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "success_rate": successes / max(1, cfg.num_episodes) if success_evaluator is not None else None,
        "successes": successes,
    }

    results_path = os.path.join(cfg.output_dir, "metrics.json")
    with open(results_path, "w", encoding="utf-8") as fp:
        json.dump({"config": asdict(cfg), "metrics": metrics}, fp, ensure_ascii=False, indent=2)

    print(f"\n[dp3][eval] Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {results_path}")
    return metrics
