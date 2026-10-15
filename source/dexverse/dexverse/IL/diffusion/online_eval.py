# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop evaluation of a trained diffusion policy in simulation.

Must run under a live Isaac Sim app, so the caller (``scripts/diffusion/
eval_online.py``) launches ``AppLauncher`` before importing this module.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .config import EvalConfig
from .runner import DiffusionPolicyRunner, load_policy


def evaluate(cfg: EvalConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_device = cfg.policy_device or cfg.device
    device = torch.device(policy_device if torch.cuda.is_available() else "cpu")

    policy, stats = load_policy(cfg.ckpt, task=cfg.task, device=device)
    env, success_probe = _make_env(cfg)
    runner = DiffusionPolicyRunner(
        policy,
        stats,
        device=device,
        num_envs=env.unwrapped.num_envs,
        sample_steps=cfg.sample_steps,
        replan_interval=cfg.replan_interval,
    )

    initial_states = _load_initial_states(cfg.reset_from_dataset) if cfg.reset_from_dataset else None
    if initial_states is not None:
        print(f"[diffusion] pinning episode starts to {len(initial_states)} recorded initial states", flush=True)

    successes = 0
    returns: list[float] = []
    lengths: list[int] = []
    latencies_ms: list[float] = []
    started = time.perf_counter()

    for episode in range(cfg.num_episodes):
        obs, _ = env.reset()
        if initial_states is not None:
            state = _to_torch(initial_states[episode % len(initial_states)], env.unwrapped.device)
            obs, _ = env.unwrapped.reset_to(state, torch.tensor([0], device=env.unwrapped.device), is_relative=True)
        runner.reset()

        episode_return = 0.0
        succeeded = False
        steps = cfg.max_steps
        for step in range(cfg.max_steps):
            action_start = time.perf_counter()
            action = runner.act(obs).to(env.unwrapped.device)
            latencies_ms.append((time.perf_counter() - action_start) * 1000.0)

            obs, reward, terminated, truncated, _ = env.step(action.unsqueeze(0))
            episode_return += float(torch.as_tensor(reward).reshape(-1)[0])
            succeeded = succeeded or (success_probe is not None and success_probe(env.unwrapped))
            if succeeded or _flag(terminated) or _flag(truncated):
                steps = step + 1
                break

        successes += int(succeeded)
        returns.append(episode_return)
        lengths.append(steps)
        print(
            f"[diffusion] episode {episode + 1:>3}/{cfg.num_episodes} success={'yes' if succeeded else 'no '} "
            f"steps={steps:>4} return={episode_return:+.3f} running_success={successes}/{episode + 1}",
            flush=True,
        )

    env.close()
    metrics = {
        "num_episodes": cfg.num_episodes,
        "success_rate": successes / max(1, cfg.num_episodes) if success_probe is not None else None,
        "avg_return": float(np.mean(returns)),
        "avg_ep_len": float(np.mean(lengths)),
        "inference_latency_ms": float(np.mean(latencies_ms)),
        "elapsed_s": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps({"config": asdict(cfg), "metrics": metrics}, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def _make_env(cfg: EvalConfig):
    """Build the task env, and detach its success termination into a probe."""
    import gymnasium as gym

    import dexverse.tasks  # noqa: F401  (registers the Dexverse-* task ids)
    from dexverse.tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(cfg.task, device=cfg.device, num_envs=1, json_path=cfg.json_path)
    if cfg.robot_type is not None and cfg.robot_type != getattr(env_cfg, "robot_type", None):
        # Same rebuild as create_demo_files_sequential.py: robot_type is a constructor input of the task cfg.
        env_cfg = type(env_cfg)(robot_type=cfg.robot_type)
        env_cfg.sim.device = cfg.device
        env_cfg.scene.num_envs = 1
    if cfg.seed is not None:
        env_cfg.seed = cfg.seed
    # The recorded demos were replayed under an observation preset; the live env
    # has to publish the same groups or the state vector no longer lines up.
    if cfg.observation_preset is not None:
        env_cfg.observation_preset = cfg.observation_preset
        env_cfg._apply_observation_preset(cfg.observation_preset)

    # Success normally terminates the episode, which makes "terminated" ambiguous
    # between success and failure. Evaluate the same term manually instead.
    success_probe = None
    success_cfg = getattr(getattr(env_cfg, "terminations", None), "success", None)
    if success_cfg is not None:
        success_probe = _SuccessProbe(success_cfg)
        env_cfg.terminations.success = None

    from dexverse.replay_rigid_object import configure_replay_rigid_objects

    configure_replay_rigid_objects(env_cfg.scene)
    env = gym.make(cfg.task, cfg=env_cfg, render_mode="rgb_array" if cfg.video else None)
    if cfg.video:
        video_dir = Path(cfg.output_dir) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: True,
            video_length=cfg.video_length or cfg.max_steps,
            disable_logger=True,
        )
    return env, success_probe


class _SuccessProbe:
    """Evaluates a termination term out-of-band, without terminating the episode.

    Isaac Lab termination terms are either plain ``func(env, **params)`` callables
    or ``ManagerTermBase`` subclasses that must be constructed once with
    ``(cfg, env)`` so they can pre-build per-term tensors from their params.
    """

    def __init__(self, term_cfg: Any):
        self.term_cfg = term_cfg
        self.term = None if inspect.isclass(term_cfg.func) else term_cfg.func

    def __call__(self, env: Any) -> bool:
        if self.term is None:
            self.term = self.term_cfg.func(cfg=self.term_cfg, env=env)
        return _flag(self.term(env, **(self.term_cfg.params or {})))


def _load_initial_states(dataset_file: str) -> list[dict[str, Any]]:
    with h5py.File(dataset_file, "r") as handle:
        states = [
            _read_h5_tree(handle["data"][name]["initial_state"])
            for name in sorted(handle["data"])
            if "initial_state" in handle["data"][name]
        ]
    if not states:
        raise ValueError(f"No 'initial_state' entries in {dataset_file}; re-run build_dataset.py with the source pickle.")
    return states


def _read_h5_tree(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Dataset):
        return node[...]
    return {key: _read_h5_tree(node[key]) for key in node}


def _to_torch(tree: Any, device: torch.device | str) -> Any:
    if isinstance(tree, dict):
        return {key: _to_torch(value, device) for key, value in tree.items()}
    return torch.as_tensor(tree, device=device)


def _flag(value: Any) -> bool:
    """First element of a possibly-batched termination/success signal."""
    return bool(torch.as_tensor(value).reshape(-1)[0].item())
