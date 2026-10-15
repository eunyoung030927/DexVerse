# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live demonstration-diversity readout for the teleoperator.

``record_demos.py`` writes trajectories blind: the operator has no feedback on
whether the demos being collected are actually diverse until the offline
analysis runs, long after the session. This module closes that loop -- after
every recorded trajectory it re-scores the dataset and renders the result as an
image the operator can read *inside the headset*.

Pipeline per trajectory:

1. :func:`dexverse.teleop_utils.diversity_core.demos_from_episodes` wraps the
   recorder's in-memory episodes (optionally merged with previously recorded
   pickles for the same task) -- no pickle round-trip.
2. ``analyze_demos(..., include_knn=False)`` scores them. The k-NN
   multimodality stage is skipped because it is the only one whose cost grows
   with dataset size (~0.6 s at 50 episodes); everything kept here runs in
   ~35 ms, which is invisible in the reset pause after a successful demo.
3. :func:`render_report_figure` draws a dark-surface panel sized for VR.
4. :meth:`TeleopStatsPanel.publish` pushes it into the XR scene as a
   camera-facing widget, and always writes a PNG + JSON next to the demos.

The XR widget is built on ``omni.kit.xr.scene_view.utils``, the same machinery
behind Isaac Lab's ``show_instruction``. That renders into the XR scene view, so
it *is* visible over CloudXR in the headset -- unlike an ``omni.ui`` window,
which is desktop-only Kit UI and never reaches the goggles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# --- palette -----------------------------------------------------------------
# Dark-surface steps from the reference data-viz palette. Validated as a pair on
# this surface: CVD dE 26.8 (protan), normal-vision dE 31.8, both >= 3:1 contrast.
SURFACE = "#1a1a19"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"
SERIES_COVERAGE = "#3987e5"  # slot 1, blue  -- initial-state coverage
SERIES_STRATEGY = "#d95926"  # slot 2, orange -- approach strategies
GOOD = "#0ca30c"


def _fmt(value, digits: int = 2, dash: str = "--") -> str:
    if value is None:
        return dash
    try:
        if float(value) != float(value):  # NaN
            return dash
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".") or "0"
    except (TypeError, ValueError):
        return str(value)


def summarize_report(report: dict, session_count: int | None = None) -> dict:
    """Flatten the pieces of an ``analyze_demos`` report the panel shows."""
    coverage = report.get("initial_state_coverage") or {}
    saturation = coverage.get("saturation") or {}
    strategy = report.get("strategy") or {}
    octant = strategy.get("approach_octant") or {}
    onset = strategy.get("onset_step") or {}
    regrasp = strategy.get("regrasp_proxy") or {}
    motion = strategy.get("motion") or {}

    return {
        "task": report.get("task", "?"),
        "robot_type": report.get("robot_type"),
        "n_episodes": int(report.get("n_episodes", 0) or 0),
        "session_count": session_count,
        "coverage_curve": list(saturation.get("curve") or []),
        "coverage_n95": saturation.get("n95"),
        "coverage_final": saturation.get("final"),
        "mode_counts": dict(octant.get("counts") or {}),
        "mode_perplexity": octant.get("perplexity"),
        "mode_distinct": octant.get("distinct"),
        "mode_top_share": octant.get("top_share"),
        "onset_median": onset.get("median"),
        "regrasp_mean": regrasp.get("mean"),
        "regrasp_nonzero": regrasp.get("nonzero_episodes"),
        "motion": motion,
        "warnings": list(report.get("warnings") or []),
    }


def digest_text(summary: dict) -> str:
    """Compact console/fallback-panel digest of a summary."""
    n = summary["n_episodes"]
    lines = [f"{summary['task']}  |  {n} demo(s)"]
    if summary.get("session_count") is not None:
        lines[0] += f"  ({summary['session_count']} this session)"
    n95 = summary.get("coverage_n95")
    if n95 is not None:
        lines.append(f"start coverage: n95 = {n95}/{n}  (higher = starts still diversifying)")
    if summary.get("mode_perplexity") is not None:
        lines.append(
            f"approach strategies: {_fmt(summary['mode_perplexity'])} effective"
            f"  ({summary.get('mode_distinct', '?')} distinct,"
            f" top {int(round(100 * (summary.get('mode_top_share') or 0)))}%)"
        )
    if summary.get("regrasp_mean") is not None:
        lines.append(f"regrasp proxy: {_fmt(summary['regrasp_mean'])} mean")
    for w in summary.get("warnings", []):
        lines.append(f"! {w}")
    return "\n".join(lines)


# --- figure ------------------------------------------------------------------


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=11, length=3, width=1.0)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)


def _draw_kpis(ax, summary: dict) -> None:
    """Top row of hero numbers -- these are magnitudes with no comparison, so
    they are stat tiles rather than a chart (see the form heuristic)."""
    ax.set_axis_off()
    n = summary["n_episodes"]
    session = summary.get("session_count")
    tiles = [
        ("DEMOS", str(n), f"{session} this session" if session is not None else ""),
        (
            "START COVERAGE",
            f"{summary['coverage_n95']}/{n}" if summary.get("coverage_n95") is not None else "--",
            "n95 -- new starts still adding",
        ),
        (
            "EFFECTIVE STRATEGIES",
            _fmt(summary.get("mode_perplexity")),
            f"{summary.get('mode_distinct', '?')} distinct approaches",
        ),
        (
            "REGRASP PROXY",
            _fmt(summary.get("regrasp_mean")),
            f"{summary.get('regrasp_nonzero', 0)} episode(s) > 0",
        ),
    ]
    for i, (label, value, sub) in enumerate(tiles):
        x = i / len(tiles) + 0.012
        ax.text(x, 0.90, label, transform=ax.transAxes, color=INK_MUTED,
                fontsize=11, fontweight="bold", va="top", ha="left")
        ax.text(x, 0.60, value, transform=ax.transAxes, color=INK_PRIMARY,
                fontsize=30, fontweight="bold", va="top", ha="left")
        if sub:
            ax.text(x, 0.10, sub, transform=ax.transAxes, color=INK_SECONDARY,
                    fontsize=10, va="bottom", ha="left")


def _draw_coverage(ax, summary: dict) -> None:
    curve = summary.get("coverage_curve") or []
    _style_axes(ax)
    ax.set_title("Initial-state coverage", color=INK_PRIMARY, fontsize=13,
                 fontweight="bold", loc="left", pad=10)
    if len(curve) < 2:
        ax.text(0.5, 0.5, "needs 2+ demos", transform=ax.transAxes, color=INK_MUTED,
                fontsize=12, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    xs = np.arange(1, len(curve) + 1)
    ys = np.asarray(curve, dtype=float)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    # Single series -- the title names it, so no legend box.
    ax.plot(xs, ys, color=SERIES_COVERAGE, linewidth=2.0, zorder=3,
            solid_capstyle="round")
    ax.plot(xs[-1:], ys[-1:], marker="o", markersize=8, color=SERIES_COVERAGE, zorder=4)
    # One direct label on the live end, never a number on every point.
    ax.annotate(f"{ys[-1]:.0f} bins", xy=(xs[-1], ys[-1]), xytext=(-4, 10),
                textcoords="offset points", color=INK_PRIMARY, fontsize=11,
                fontweight="bold", ha="right")
    n95 = summary.get("coverage_n95")
    if n95 is not None and 0 < n95 <= len(curve):
        ax.axvline(n95, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate("n95", xy=(n95, ax.get_ylim()[0]), xytext=(4, 6),
                    textcoords="offset points", color=INK_MUTED, fontsize=10)
    ax.set_xlabel("demos")
    ax.set_ylabel("occupied bins")


def _draw_strategies(ax, summary: dict) -> None:
    counts = summary.get("mode_counts") or {}
    _style_axes(ax)
    ax.set_title("Approach strategies at onset", color=INK_PRIMARY,
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    if not counts:
        ax.text(0.5, 0.5, "no palm model for this embodiment",
                transform=ax.transAxes, color=INK_MUTED, fontsize=12,
                ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:6]
    labels = [(k if len(k) <= 14 else k[:13] + "\u2026") for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    ypos = np.arange(len(labels))
    ax.grid(True, axis="x", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    # Single measure across categories -> one hue, not a categorical rainbow.
    ax.barh(ypos, values, height=0.62, color=SERIES_STRATEGY, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color=INK_SECONDARY, fontsize=11)
    # Reserve a minimum number of category slots so a task with a single
    # observed mode shows a thin bar rather than one filling the panel.
    ax.set_ylim(-0.6, max(len(labels), 4) - 0.4)
    ax.set_xlabel("episodes")
    for y, v in zip(ypos, values):
        ax.annotate(str(v), xy=(v, y), xytext=(6, 0), textcoords="offset points",
                    color=INK_PRIMARY, fontsize=11, fontweight="bold", va="center")
    ax.set_xlim(0, max(values) * 1.18)


def render_report_figure(summary: dict, width_px: int = 1024, height_px: int = 620,
                         dpi: int = 100) -> np.ndarray:
    """Render the panel to an ``(H, W, 4)`` uint8 RGBA array.

    Uses the Agg canvas directly rather than pyplot so this stays free of global
    state and safe to call from the sim loop.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=SURFACE)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 2.0], hspace=0.42, wspace=0.30,
                          left=0.065, right=0.975, top=0.90, bottom=0.11)

    header = fig.add_subplot(gs[0, :])
    _draw_kpis(header, summary)
    _draw_coverage(fig.add_subplot(gs[1, 0]), summary)
    _draw_strategies(fig.add_subplot(gs[1, 1]), summary)

    title = summary.get("task", "?")
    if summary.get("robot_type"):
        title += f"   ·   {summary['robot_type']}"
    fig.text(0.065, 0.965, title, color=INK_SECONDARY, fontsize=12, va="top", ha="left")

    canvas.draw()
    return np.asarray(canvas.buffer_rgba()).copy()


# --- XR display --------------------------------------------------------------

_ACTIVE_CONTAINERS: dict[str, object] = {}


def _build_image_widget_class():
    """Define the image widget lazily -- ``omni.ui`` only exists inside Kit."""
    import omni.ui as ui

    class StatsImageWidget(ui.Widget):
        """Full-bleed RGBA image, for showing a rendered figure in the XR scene."""

        def __init__(self, rgba: np.ndarray, **kwargs):
            super().__init__(**kwargs)
            height, width = int(rgba.shape[0]), int(rgba.shape[1])
            self._provider = ui.ByteImageProvider()
            self._provider.set_data_array(np.ascontiguousarray(rgba, dtype=np.uint8),
                                          [width, height])
            with ui.ZStack():
                ui.Rectangle(style={"Rectangle": {"background_color": 0xFF1A1A19,
                                                  "border_radius": 0.05}})
                ui.ImageWithProvider(self._provider,
                                     fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)

    return StatsImageWidget


def show_image_panel(rgba: np.ndarray, *, target_prim_path: str = "/DexVerseStatsPanel",
                     prim_path_source: str | None = "/_xr/stage/xrCamera",
                     translation=(0.0, 0.45, -1.35), width: float = 1.1,
                     resolution_scale: float = 900.0) -> bool:
    """Show ``rgba`` as a camera-facing panel in the XR scene.

    Mirrors Isaac Lab's ``show_instruction``: the container is torn down and
    rebuilt on each update, which is the path already proven in this stack (the
    default ``WidgetComponent`` update policy only redraws on mouse hover, which
    never fires in VR).
    """
    global _ACTIVE_CONTAINERS
    try:
        import isaaclab.sim as sim_utils
        import omni.kit.commands
        from omni.kit.xr.scene_view.utils import UiContainer, WidgetComponent
        from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource
        from pxr import Gf
    except ImportError as exc:
        logger.debug("stats_panel: XR widget stack unavailable (%s)", exc)
        return False

    try:
        hide_image_panel(target_prim_path)

        height_px, width_px = int(rgba.shape[0]), int(rgba.shape[1])
        height = width * (height_px / max(width_px, 1))

        widget_component = WidgetComponent(
            _build_image_widget_class(),
            width=width,
            height=height,
            resolution_scale=resolution_scale,
            widget_args=[rgba],
        )

        stage = sim_utils.get_current_stage()
        if stage is not None and stage.GetPrimAtPath(target_prim_path).IsValid():
            sim_utils.delete_prim(target_prim_path)

        space_stack = []
        if prim_path_source:
            copied = omni.kit.commands.execute(
                "CopyPrim", path_from=prim_path_source, path_to=target_prim_path,
                exclusive_select=False, copy_to_introducing_layer=False,
            )
            if copied is not None:
                space_stack.append(SpatialSource.new_prim_path_source(target_prim_path))
        space_stack.extend([
            SpatialSource.new_translation_source(Gf.Vec3d(*translation)),
            SpatialSource.new_look_at_camera_source(),
        ])

        _ACTIVE_CONTAINERS[target_prim_path] = UiContainer(widget_component,
                                                           space_stack=space_stack)
        return True
    except Exception as exc:
        logger.warning("stats_panel: could not show XR image panel: %s", exc)
        return False


def hide_image_panel(target_prim_path: str = "/DexVerseStatsPanel") -> None:
    container = _ACTIVE_CONTAINERS.pop(target_prim_path, None)
    if container is not None:
        try:
            container.root.clear()
        except Exception:  # teardown races with Kit shutdown; not worth raising
            pass


# --- orchestration -----------------------------------------------------------


class TeleopStatsPanel:
    """Recompute and publish demo-diversity stats after each recorded trajectory.

    Every stage is defensive: a statistics failure must never take down a teleop
    session that is mid-collection, so :meth:`publish` swallows and logs rather
    than propagating.
    """

    def __init__(self, *, task: str, category: str = "", robot_type: str | None = None,
                 output_dir: Path | str | None = None, session_file: Path | str | None = None,
                 include_history: bool = True, xr: bool = False,
                 width_px: int = 1024, height_px: int = 620, verbose: bool = True):
        self.task = task
        self.category = category
        self.robot_type = robot_type
        self.session_file = Path(session_file) if session_file else None
        self.output_dir = Path(output_dir) if output_dir else (
            self.session_file.parent if self.session_file else Path.cwd())
        # Sibling pickles for this task are the "already collected" history; the
        # session's own file is excluded because finalize_episode keeps it
        # flushed and it would double-count every in-memory episode.
        self.history_dir = self.output_dir if include_history else None
        self.xr = bool(xr)
        self.width_px = width_px
        self.height_px = height_px
        self.verbose = verbose
        self.last_summary: dict | None = None

    def publish(self, episodes: list[dict]) -> dict | None:
        """Score ``episodes``, render the panel and push it to the headset."""
        try:
            from dexverse.teleop_utils import diversity_core as dc

            demos = dc.demos_from_episodes(
                episodes,
                task=self.task,
                category=self.category,
                robot_type=self.robot_type,
                history_dir=self.history_dir,
                exclude=self.session_file,
            )
            if demos.num_episodes == 0:
                return None
            report = dc.analyze_demos(demos, include_knn=False)
            if "error" in report:
                return None
            summary = summarize_report(report, session_count=len(episodes))
            self.last_summary = summary

            if self.verbose:
                print("\n" + digest_text(summary) + "\n")
            self._write_artifacts(report, summary)

            if self.xr:
                try:
                    rgba = render_report_figure(summary, self.width_px, self.height_px)
                    if not show_image_panel(rgba):
                        self._show_text_fallback(summary)
                except Exception as exc:
                    logger.warning("stats_panel: figure render failed (%s); using text panel", exc)
                    self._show_text_fallback(summary)
            return summary
        except Exception as exc:
            logger.warning("stats_panel: skipped this episode: %s", exc)
            return None

    def _write_artifacts(self, report: dict, summary: dict) -> None:
        stem = f"{self.task}_stats"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str))
        except Exception as exc:
            logger.debug("stats_panel: could not write JSON report: %s", exc)
        try:
            from PIL import Image

            rgba = render_report_figure(summary, self.width_px, self.height_px)
            Image.fromarray(rgba).save(self.output_dir / f"{stem}.png")
        except Exception as exc:
            logger.debug("stats_panel: could not write PNG: %s", exc)

    def _show_text_fallback(self, summary: dict) -> None:
        try:
            from isaaclab.ui.xr_widgets import show_instruction

            show_instruction(
                digest_text(summary), prim_path_source="/_xr/stage/xrCamera",
                translation=(0.0, 0.45, -1.35), display_duration=None,
                font_size=0.05, max_width=2.4, min_width=1.4,
                target_prim_path="/DexVerseStatsText",
            )
        except Exception as exc:
            logger.debug("stats_panel: text fallback unavailable: %s", exc)

    def close(self) -> None:
        hide_image_panel()
