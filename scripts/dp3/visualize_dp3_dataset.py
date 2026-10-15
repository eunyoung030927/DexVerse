#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize point clouds stored in a DP3 HDF5 dataset.

Provides several visualization modes:
  - Single frame: show point cloud at a specific timestep
  - Animation: play through all frames of an episode
  - Multi-frame: show multiple timesteps side-by-side
  - Episode overview: show first/mid/last frames of an episode

Usage:
  # Show frame 0 of episode 0
  python scripts/dp3/visualize_dp3_dataset.py --hdf5_file outputs/dp3_demo.hdf5

  # Animate episode 2
  python scripts/dp3/visualize_dp3_dataset.py --hdf5_file outputs/dp3_demo.hdf5 \\
      --episode 2 --mode animate --interval 100

  # Show 4 evenly-spaced frames from episode 0
  python scripts/dp3/visualize_dp3_dataset.py --hdf5_file outputs/dp3_demo.hdf5 \\
      --mode multi --num_frames 4

  # Show overview of all episodes (first frame each)
  python scripts/dp3/visualize_dp3_dataset.py --hdf5_file outputs/dp3_demo.hdf5 \\
      --mode overview

  # Save frame as image instead of showing interactively
  python scripts/dp3/visualize_dp3_dataset.py --hdf5_file outputs/dp3_demo.hdf5 \\
      --save_path outputs/pcd_vis.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _load_episode(hdf5_path: str, episode_idx: int) -> dict:
    """Load a single episode from the DP3 HDF5 file."""
    with h5py.File(hdf5_path, "r") as hf:
        ep_name = f"episode_{episode_idx}"
        if ep_name not in hf["data"]:
            available = list(hf["data"].keys())
            raise KeyError(f"Episode '{ep_name}' not found. Available: {available}")

        ep = hf["data"][ep_name]
        data = {
            "point_cloud": ep["obs/point_cloud"][...],  # (T, N, 3)
            "agent_pos": ep["obs/agent_pos"][...],  # (T, D)
            "action": ep["action"][...],  # (T, Da)
        }

        # Load dataset-level metadata
        data["bbox_min"] = hf.attrs.get("bbox_min", None)
        data["bbox_max"] = hf.attrs.get("bbox_max", None)
        data["task"] = hf.attrs.get("task", "unknown")
        data["num_points"] = hf.attrs.get("num_points", data["point_cloud"].shape[1])
        data["num_episodes"] = hf.attrs.get("num_episodes", 0)

        return data


def _load_metadata(hdf5_path: str) -> dict:
    """Load dataset-level metadata."""
    with h5py.File(hdf5_path, "r") as hf:
        meta = {
            "task": hf.attrs.get("task", "unknown"),
            "num_episodes": hf.attrs.get("num_episodes", 0),
            "num_points": hf.attrs.get("num_points", 0),
            "bbox_min": hf.attrs.get("bbox_min", None),
            "bbox_max": hf.attrs.get("bbox_max", None),
        }
        # Per-episode info
        episodes = []
        for ep_name in sorted(hf["data"].keys()):
            ep = hf["data"][ep_name]
            episodes.append({
                "name": ep_name,
                "num_steps": ep["obs/point_cloud"].shape[0],
                "pcd_shape": ep["obs/point_cloud"].shape,
                "agent_pos_dim": ep["obs/agent_pos"].shape[-1],
                "action_dim": ep["action"].shape[-1],
            })
        meta["episodes"] = episodes
        return meta


def _plot_single_frame(
    pcd: np.ndarray,
    title: str = "",
    bbox_min=None,
    bbox_max=None,
    ax=None,
    color_by: str = "z",
    point_size: float = 1.0,
):
    """Plot a single (N, 3) point cloud on a 3D axis."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    # Color by coordinate
    if color_by == "z":
        colors = z
    elif color_by == "x":
        colors = x
    elif color_by == "y":
        colors = y
    else:
        colors = z

    ax.scatter(x, y, z, c=colors, cmap="viridis", s=point_size, alpha=0.6)

    # Draw bbox if provided
    if bbox_min is not None and bbox_max is not None:
        _draw_bbox(ax, bbox_min, bbox_max)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return ax


def _draw_bbox(ax, bbox_min, bbox_max):
    """Draw a wireframe bounding box on a 3D axis."""
    bmin = np.asarray(bbox_min)
    bmax = np.asarray(bbox_max)
    # 12 edges of a box
    corners = np.array([
        [bmin[0], bmin[1], bmin[2]],
        [bmax[0], bmin[1], bmin[2]],
        [bmax[0], bmax[1], bmin[2]],
        [bmin[0], bmax[1], bmin[2]],
        [bmin[0], bmin[1], bmax[2]],
        [bmax[0], bmin[1], bmax[2]],
        [bmax[0], bmax[1], bmax[2]],
        [bmin[0], bmax[1], bmax[2]],
    ])
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # vertical
    ]
    for i, j in edges:
        ax.plot3D(*zip(corners[i], corners[j]), color="red", alpha=0.3, linewidth=0.8)


def visualize_single(
    hdf5_path: str, episode_idx: int, frame_idx: int, save_path: str | None = None, show_bbox: bool = True
):
    """Visualize a single frame from a DP3 dataset."""
    import matplotlib.pyplot as plt

    data = _load_episode(hdf5_path, episode_idx)
    pcd = data["point_cloud"]  # (T, N, 3)
    T = pcd.shape[0]
    if frame_idx >= T:
        raise ValueError(f"Frame {frame_idx} out of range (episode has {T} steps)")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    title = f"{data['task']} | Episode {episode_idx} | Frame {frame_idx}/{T-1} | {data['num_points']} pts"
    bbox_min = data["bbox_min"] if show_bbox else None
    bbox_max = data["bbox_max"] if show_bbox else None
    _plot_single_frame(pcd[frame_idx], title=title, bbox_min=bbox_min, bbox_max=bbox_max, ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def visualize_animate(
    hdf5_path: str, episode_idx: int, interval: int = 100, save_path: str | None = None, show_bbox: bool = True
):
    """Animate through all frames of an episode."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    data = _load_episode(hdf5_path, episode_idx)
    pcd = data["point_cloud"]  # (T, N, 3)
    T = pcd.shape[0]
    bbox_min = data["bbox_min"] if show_bbox else None
    bbox_max = data["bbox_max"] if show_bbox else None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx):
        ax.cla()
        title = f"{data['task']} | Episode {episode_idx} | Frame {frame_idx}/{T-1} | {data['num_points']} pts"
        _plot_single_frame(pcd[frame_idx], title=title, bbox_min=bbox_min, bbox_max=bbox_max, ax=ax, point_size=1.5)

    anim = FuncAnimation(fig, update, frames=T, interval=interval, repeat=True)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        else:
            anim.save(save_path, fps=1000 // interval)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()


def visualize_multi(
    hdf5_path: str, episode_idx: int, num_frames: int = 4, save_path: str | None = None, show_bbox: bool = True
):
    """Show multiple evenly-spaced frames side by side."""
    import matplotlib.pyplot as plt

    data = _load_episode(hdf5_path, episode_idx)
    pcd = data["point_cloud"]  # (T, N, 3)
    T = pcd.shape[0]
    bbox_min = data["bbox_min"] if show_bbox else None
    bbox_max = data["bbox_max"] if show_bbox else None

    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

    fig = plt.figure(figsize=(5 * num_frames, 5))
    fig.suptitle(f"{data['task']} | Episode {episode_idx} | {data['num_points']} pts", fontsize=14)

    for i, fi in enumerate(frame_indices):
        ax = fig.add_subplot(1, num_frames, i + 1, projection="3d")
        _plot_single_frame(pcd[fi], title=f"Frame {fi}", bbox_min=bbox_min, bbox_max=bbox_max, ax=ax, point_size=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def visualize_overview(hdf5_path: str, save_path: str | None = None, max_episodes: int = 16):
    """Show first frame of each episode in a grid."""
    import matplotlib.pyplot as plt

    meta = _load_metadata(hdf5_path)
    num_eps = min(len(meta["episodes"]), max_episodes)

    cols = min(4, num_eps)
    rows = (num_eps + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    fig.suptitle(f"{meta['task']} | {meta['num_episodes']} episodes | {meta['num_points']} pts", fontsize=14)

    for i in range(num_eps):
        data = _load_episode(hdf5_path, i)
        pcd = data["point_cloud"][0]  # first frame
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _plot_single_frame(pcd, title=f"Ep {i} (T={data['point_cloud'].shape[0]})", ax=ax, point_size=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def print_dataset_info(hdf5_path: str):
    """Print dataset metadata and per-episode statistics."""
    meta = _load_metadata(hdf5_path)
    print(f"Dataset: {hdf5_path}")
    print(f"  Task: {meta['task']}")
    print(f"  Num episodes: {meta['num_episodes']}")
    print(f"  Points per frame: {meta['num_points']}")
    if meta["bbox_min"] is not None:
        print(f"  Workspace bbox min: {meta['bbox_min']}")
        print(f"  Workspace bbox max: {meta['bbox_max']}")
    print(f"\n  {'Episode':<12} {'Steps':<8} {'PCD shape':<20} {'AgentPos':<12} {'Action':<10}")
    print(f"  {'-'*62}")
    for ep in meta["episodes"]:
        print(
            f"  {ep['name']:<12} {ep['num_steps']:<8} {str(ep['pcd_shape']):<20} "
            f"{ep['agent_pos_dim']:<12} {ep['action_dim']:<10}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize point clouds from a DP3 HDF5 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hdf5_file", type=str, required=True, help="Path to the DP3 HDF5 dataset")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize (default: 0)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index for 'single' mode (default: 0)")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "animate", "multi", "overview", "info"],
        help="Visualization mode (default: single)",
    )
    parser.add_argument("--num_frames", type=int, default=4, help="Number of frames for 'multi' mode (default: 4)")
    parser.add_argument(
        "--interval", type=int, default=100, help="Animation interval in ms for 'animate' mode (default: 100)"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Save visualization to file instead of showing interactively"
    )
    parser.add_argument(
        "--no_bbox", action="store_true", default=False, help="Hide the workspace bounding box wireframe"
    )
    parser.add_argument(
        "--max_episodes", type=int, default=16, help="Max episodes to show in 'overview' mode (default: 16)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    show_bbox = not args.no_bbox

    if args.mode == "info":
        print_dataset_info(args.hdf5_file)
    elif args.mode == "single":
        visualize_single(args.hdf5_file, args.episode, args.frame, save_path=args.save_path, show_bbox=show_bbox)
    elif args.mode == "animate":
        visualize_animate(
            args.hdf5_file, args.episode, interval=args.interval, save_path=args.save_path, show_bbox=show_bbox
        )
    elif args.mode == "multi":
        visualize_multi(
            args.hdf5_file, args.episode, num_frames=args.num_frames, save_path=args.save_path, show_bbox=show_bbox
        )
    elif args.mode == "overview":
        visualize_overview(args.hdf5_file, save_path=args.save_path, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
