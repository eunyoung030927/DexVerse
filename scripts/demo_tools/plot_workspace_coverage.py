#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Workspace-coverage figures: where can objects/goals start (the env's reset
distribution) vs where the demos actually started.

Inputs (simulator-free):
  * ``outputs/reset_audit/<Task>/samples.npz`` from
    ``scripts/env_tools/sample_reset_distributions.py`` (N sampled resets;
    ``rigid_object/<name>/root_pose`` / ``articulation/<name>/root_pose`` (N,7)),
  * the session pickles' ``initial_state`` (same layout) for the demo starts.

For every scene entity that actually moves between resets, one top-down panel:
reset density (2-D histogram, sequential blue) with demo starts overlaid
(orange), a yaw histogram when the entity's yaw is randomized, and a coverage
number: the fraction of the reset distribution's occupied area that lies within
``--radius`` metres of at least one demo start. Discrete per-episode goals found
in ``task_state`` (e.g. the carton's target face) get a bar panel.

    python scripts/demo_tools/plot_workspace_coverage.py \\
        --task Dexverse-FunctionalPourMug-v0:functional/Dexverse-FunctionalPourMug-v0/0823_zhuxiu \\
        --out-dir outputs/demo_audit/coverage
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Reference palette (dataviz skill): categorical slot 1/2, sequential blue ramp, chrome.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)
# Human-readable class labels for known discrete goal buffers.
GOAL_LABELS = {"carton_grasp_face": ["+x", "-x", "+y", "-y", "+z", "-z"]}


def yaw_deg(q):  # (N,4) wxyz
    w, x, y, z = q.T
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def load_samples(task):
    p = os.path.join(REPO, "outputs", "reset_audit", task, "samples.npz")
    with np.load(p) as d:
        return {k: d[k] for k in d.files if k.endswith("root_pose") and not k.startswith("settled/")}


def load_demo_starts(rel):
    eps = [e for p in sorted(glob.glob(os.path.join(REPO, "source/dexverse/demonstrations", rel, "*.pkl")))
           for e in pickle.load(open(p, "rb"))["episodes"]]
    poses, goals = {}, {}
    for e in eps:
        st = e["initial_state"]
        for kind in ("rigid_object", "articulation"):
            for name, entry in st.get(kind, {}).items():
                poses.setdefault(f"{kind}/{name}/root_pose", []).append(np.asarray(entry["root_pose"]).reshape(-1)[:7])
        for b, v in ((e.get("task_state") or {}).get("env_buffers") or {}).items():
            v = np.asarray(v)
            if v.size == 1:
                goals.setdefault(b, []).append(int(v))
    return {k: np.stack(v) for k, v in poses.items()}, goals, len(eps)


def coverage(samples_xy, demo_xy, radius, cell=0.02):
    """Fraction of occupied reset cells (cell m grid) within `radius` of a demo start."""
    cells = np.unique(np.floor(samples_xy / cell).astype(int), axis=0)
    centers = (cells + 0.5) * cell
    d = np.sqrt(((centers[:, None, :] - demo_xy[None, :, :]) ** 2).sum(-1)).min(1)
    return float((d <= radius).mean()), float(d.max())


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_task(task, rel, out_dir, radius):
    samples = load_samples(task)
    demos, goals, n_demos = load_demo_starts(rel)
    ents = []
    for k in samples:
        name = k.split("/")[1]
        if "table" in name or name == "robot" or k not in demos:
            continue
        xy_s = samples[k][:, :2]
        if xy_s.std(0).max() < 1e-3 and demos[k][:, :2].std(0).max() < 1e-3:
            continue  # static entity
        ents.append(k)
    n = len(ents) + (1 if goals else 0)
    yaw_rows = [np.ptp(yaw_deg(samples[k][:, 3:7])) > 5 for k in ents]
    fig_h = 5.0 + (1.8 if any(yaw_rows) else 0)
    fig, axes = plt.subplots(2 if any(yaw_rows) else 1, max(n, 1), figsize=(4.4 * max(n, 1) + 0.8, fig_h), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]} if any(yaw_rows) else None)
    fig.patch.set_facecolor(SURFACE)
    stats = {}
    # table footprint (from the sampled leg positions); every top-down panel uses it as its frame
    legs = [samples[k][:, :2] for k in samples if "table_leg" in k]
    ext = np.abs(np.concatenate(legs)).max(0) + 0.02 if legs else np.array([0.65, 0.65])
    vmax_all = 1
    for i, k in enumerate(ents):
        ax = axes[0, i]; style(ax)
        s_xy, d_xy = samples[k][:, :2], demos[k][:, :2]
        lo, hi = -ext, ext
        bins = [np.arange(lo[0], hi[0] + 0.02, 0.02), np.arange(lo[1], hi[1] + 0.02, 0.02)]
        h, xe, ye = np.histogram2d(s_xy[:, 0], s_xy[:, 1], bins=bins)
        vmax = max(1, np.percentile(h[h > 0], 95)); vmax_all = max(vmax_all, vmax)
        mesh = ax.pcolormesh(xe, ye, h.T, cmap=CMAP, vmin=0, vmax=vmax, shading="flat")
        ax.scatter(d_xy[:, 0], d_xy[:, 1], s=14, c=ORANGE, edgecolors=SURFACE, linewidths=0.7, zorder=3)
        ax.add_patch(plt.Rectangle((-ext[0] + 0.02, -ext[1] + 0.02), 2 * ext[0] - 0.04, 2 * ext[1] - 0.04, fill=False, ec=AXIS, lw=0.9, ls="--"))
        ax.text(ext[0] - 0.03, -ext[1] + 0.03, "table", color=MUTED, fontsize=7, ha="right", va="bottom")
        cov, worst = coverage(s_xy, d_xy, radius)
        stats[k] = {"coverage_frac": cov, "worst_gap_m": worst, "reset_x": [float(s_xy[:, 0].min()), float(s_xy[:, 0].max())],
                    "reset_y": [float(s_xy[:, 1].min()), float(s_xy[:, 1].max())], "demo_x": [float(d_xy[:, 0].min()), float(d_xy[:, 0].max())],
                    "demo_y": [float(d_xy[:, 1].min()), float(d_xy[:, 1].max())]}
        ax.set_title(f"{k.split('/')[1]}\ndemos cover {cov*100:.0f}% of reset area (\u2264{radius*100:.0f} cm), worst gap {worst*100:.0f} cm",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("x [m]", color=INK2, fontsize=8); ax.set_ylabel("y [m]", color=INK2, fontsize=8)
        ax.set_aspect("equal"); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        if any(yaw_rows):
            ay = axes[1, i]; style(ay)
            if yaw_rows[i]:
                ys, yd = yaw_deg(samples[k][:, 3:7]), yaw_deg(demos[k][:, 3:7])
                b = np.arange(-180, 181, 15)
                ay.hist(ys, bins=b, density=True, color=BLUE, alpha=0.55, label="reset samples")
                ay.hist(yd, bins=b, density=True, histtype="step", color=ORANGE, lw=2, label="demo starts")
                ay.set_xlabel("yaw [deg]", color=INK2, fontsize=8); ay.set_xlim(-180, 180); ay.set_yticks([])
                stats[k]["yaw_reset_range"] = [float(ys.min()), float(ys.max())]; stats[k]["yaw_demo_range"] = [float(yd.min()), float(yd.max())]
            else:
                ay.axis("off")
    if goals:
        ax = axes[0, len(ents)]; style(ax)
        b, counts = next(iter(goals.items()))
        vals = sorted(set(counts)); cnt = [counts.count(v) for v in vals]
        ax.bar(vals, cnt, color=ORANGE, width=0.7)
        ax.axhline(n_demos / max(1, len(vals)), color=BLUE, lw=1.5, ls="--")
        ax.text(vals[-1] + 0.6, n_demos / max(1, len(vals)), "uniform", color=BLUE, fontsize=8, va="center")
        for v, c in zip(vals, cnt):
            ax.text(v, c + 0.3, str(c), ha="center", color=INK2, fontsize=8)
        ax.set_title(f"{b}\n(discrete goal, demos per class; dashed = uniform)", fontsize=8.5, color=INK, loc="left")
        ax.set_xticks(vals); ax.set_xticklabels([GOAL_LABELS.get(b, {}).__getitem__(v) if b in GOAL_LABELS and v < len(GOAL_LABELS[b]) else str(v) for v in vals])
        stats[b] = dict(zip(map(str, vals), cnt))
        if any(yaw_rows):
            axes[1, len(ents)].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=SEQ[4]), plt.Line2D([], [], marker="o", ls="", mfc=ORANGE, mec=SURFACE, ms=6)]
    fig.legend(handles, [f"reset distribution ({len(next(iter(samples.values())))} sampled resets, 2 cm cells)", f"demo starts ({n_demos})"],
               loc="lower left", frameon=False, fontsize=8, labelcolor=INK2, ncol=2)
    fig.suptitle(f"{task} \u2014 where objects/goals can start vs where the demos started (top-down, env frame)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 0.93, 0.94), h_pad=2.5)
    cax = fig.add_axes([0.945, 0.35, 0.012, 0.45])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax_all)), cax=cax)
    cb.set_label("resets per cell", color=INK2, fontsize=8); cb.ax.tick_params(colors=MUTED, labelsize=7); cb.outline.set_edgecolor(AXIS)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{task}.png"); fig.savefig(out, dpi=150, facecolor=SURFACE); plt.close(fig)
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", required=True, help="'<TaskId>:<category>/<Task>/<session>' (repeatable)")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "outputs", "demo_audit", "coverage"))
    ap.add_argument("--radius", type=float, default=0.05, help="Coverage radius in metres (default 0.05)")
    a = ap.parse_args()
    report = {}
    for spec in a.task:
        task, rel = spec.split(":", 1)
        out, stats = plot_task(task, rel, a.out_dir, a.radius)
        report[task] = stats; print("wrote", out)
        for k, v in stats.items():
            if "coverage_frac" in v:
                print(f"   {k:40s} covered {v['coverage_frac']*100:5.1f}%  worst gap {v['worst_gap_m']*100:4.1f} cm  reset x{v['reset_x']} y{v['reset_y']}")
    with open(os.path.join(a.out_dir, "coverage_stats.json"), "w") as fp:
        json.dump(report, fp, indent=2)


if __name__ == "__main__":
    main()
