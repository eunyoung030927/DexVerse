# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create the experiments/ tree: one folder per task, one sub-folder per target hand (the Bench2Dex layout).

  experiments/<task>/README.md              task: description, source hand and demos, target hands
  experiments/<task>/<target>/README.md     this (task, target) pair: what it is, where its data is, notes
  experiments/<task>/<target>/config.yaml   settings stage 3 runs with (parsed from the code defaults)

Existing README.md files are kept (they hold notes); config.yaml is rewritten so it always matches the code.
Status tables (STATUS.md, results.csv) are added per pair by status_report.py. Pure Python (no simulator).

  python scripts/retarget/init_experiments.py [--demos_root demos/demonstrations/v1]
"""

import argparse
import ast
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_queue import ALL_HANDS, find_demos, supported_robot_types  # noqa: E402
from run_target import REPO, experiments_root  # noqa: E402

PKG = REPO / "source" / "dexverse" / "dexverse"
# the cross-dex dataset: 9 v1 tasks x 5 hands (ondemand/README.md)
TASKS = ["Dexverse-GraspBleach-v1", "Dexverse-GraspPan-v1", "Dexverse-GraspKettle-v1", "Dexverse-FunctionalPourMug-v1",
         "Dexverse-FunctionalHammerStrike-v1", "Dexverse-BimanualLiftTray-v1", "Dexverse-BimanualLiftCarton-v1",
         "Dexverse-SlideUtilityKnife-v1", "Dexverse-OpenFlatFolder-v1"]
HAND_NAME = {"shadow": "Shadow Hand", "allegro": "Allegro Hand", "inspire": "Inspire Hand", "leap": "LEAP Hand",
             "sharpa": "Sharpa Wave", "wuji": "Wuji Hand"}


def _value(node):
    """A literal or a simple arithmetic default (e.g. 0.8 ** (1.0 / 3.0))."""
    try:
        return ast.literal_eval(node)
    except ValueError:
        return eval(compile(ast.Expression(node), "<default>", "eval"), {"__builtins__": {}})  # noqa: S307


def argparse_defaults(py: Path) -> dict:
    """Defaults of every ap.add_argument("--x", ..., default=...) in a script, without running it."""
    out = {}
    for n in ast.walk(ast.parse(py.read_text())):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args:
            name = str(getattr(n.args[0], "value", ""))
            if not name.startswith("--"):
                continue
            kw = {k.arg: k.value for k in n.keywords}
            if "default" in kw:
                try:
                    out[name[2:]] = _value(kw["default"])
                except Exception:  # noqa: BLE001
                    pass
            elif getattr(kw.get("action"), "value", "") == "store_true":
                out[name[2:]] = False
    return out


def cfg_defaults(py: Path, cls: str) -> dict:
    """Annotated field defaults of a (config)class."""
    for n in ast.walk(ast.parse(py.read_text())):
        if isinstance(n, ast.ClassDef) and n.name == cls:
            out = {}
            for b in n.body:
                if isinstance(b, ast.AnnAssign) and b.value is not None:
                    try:
                        v = _value(b.value)
                        out[b.target.id] = round(v, 6) if isinstance(v, float) else v
                    except Exception:  # noqa: BLE001
                        pass
            return out
    return {}


def task_description(task: str) -> tuple[str, str]:
    """(v1 implementation module, its docstring) from dexverse/benchmark.py's V1 table, read without importing."""
    src = (PKG / "benchmark.py").read_text()
    tree = ast.parse(src)
    tasks, impls = None, None
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_V1_IMPLEMENTATIONS":
            impls = ast.literal_eval(n.value)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "BASELINE_TASKS":
            tasks = ast.literal_eval(n.value)
    v1 = [t.removesuffix("-v0") + "-v1" for t in tasks]
    module, cls = dict(zip(v1, impls))[task]
    py = PKG / "baseline_v1" / "config" / (module.replace(".", "/") + ".py")
    doc = ast.get_docstring(ast.parse(py.read_text())) or ""
    return f"dexverse.baseline_v1.config.{module}.{cls}", doc.strip().split("\n\n")[0].replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos_root", default=str(REPO / "demos" / "demonstrations" / "v1"))
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--hands", nargs="*", default=ALL_HANDS)
    a = ap.parse_args()

    exp = experiments_root()
    stage3 = argparse_defaults(HERE / "stage3_spider.py")
    for k in ("task", "demos", "episode", "target", "tag", "max_frames", "seed", "no_hold_pass", "hold_only"):
        stage3.pop(k, None)
    env = cfg_defaults(PKG / "retarget" / "spider.py", "SpiderCfg")
    attempts = argparse_defaults(HERE / "run_target.py")
    supported = supported_robot_types()
    for task in a.tasks:
        impl, desc = task_description(task)
        try:
            demos = find_demos(Path(a.demos_root), task)
            with open(demos, "rb") as f:
                d = pickle.load(f)
            src, steps = d["robot_type"], [int(e["num_steps"]) for e in d["episodes"]]
            del d
            demo_line = (f"`{demos.relative_to(Path(a.demos_root).parents[1])}` (dexverse/DexVerse_release): {len(steps)} episodes, "
                         f"{min(steps)}-{max(steps)} steps (mean {np.mean(steps):.0f}) at 60 Hz")
        except FileNotFoundError:
            src = "floating_shadow_bimanual" if "Bimanual" in task or task.split("-")[1] in (
                "SlideUtilityKnife", "OpenFlatFolder") else "floating_shadow_right"
            demo_line = "not downloaded here"
        side = src.split("_", 2)[2]                                     # right | bimanual
        targets = [f"floating_{h}_{side}" for h in a.hands if h != "shadow"]
        d = exp / task
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "README.md").exists():
            (d / "README.md").write_text(
                f"# {task}\n\n{desc}\n\n"
                f"- Implementation: `{impl}`\n"
                f"- Source demos: `{src}`, {demo_line}\n"
                f"- Targets: {', '.join(f'[{t}]({t}/)' for t in targets)}\n"
                f"- Data: `<runs>/{task}/demos/epNNN/<target>/` (runs = `$DEXVERSE_RETARGET_RUNS`, default `results/retarget/`)\n")
        for t in targets:
            p = d / t
            p.mkdir(exist_ok=True)
            hand = t.split("_")[1]
            if not (p / "README.md").exists():
                (p / "README.md").write_text(
                    f"# {task}: {HAND_NAME['shadow']} -> {HAND_NAME[hand]} (`{t}`)\n\n"
                    f"Retargeting of the released {task} demos from `{src}` onto `{t}`.\n\n"
                    f"| file | content |\n|---|---|\n"
                    f"| `config.yaml` | stage-3 settings in force (from the code defaults; each attempt also stores its own in `spider_aK.json`) |\n"
                    f"| `STATUS.md`, `results.csv` | per-episode progress and verdicts (status_report.py) |\n\n"
                    f"Data: `<runs>/{task}/demos/epNNN/{t}/` (kinematic.npz, spider_aK.json / _traj.pkl.gz, run.log, status.json); "
                    f"uploaded pickle: HF `<ns>/dexverse-{task.removeprefix('Dexverse-').lower()}-{hand}-retargeting`.\n\n## Notes\n\n"
                    + ("" if t in supported else "- Not supported by the retargeting yet (`dexverse/retarget/hands.py`); "
                       "the queue skips it.\n"))
            cfg = {"task": task, "source": src, "target": t, "supported": t in supported,
                   "stage3_spider": stage3, "spider_env": env,
                   "attempts": {"max_attempts": attempts.get("max_attempts"),
                                "schedule": "attempt K: seed K, num_samples 1024*(1+K//2), iters 5+K"},
                   "success": "task success term held 10 consecutive steps (record_demos rule)",
                   "physics": "DexVerse task / robot settings, unchanged"}
            with open(p / "config.yaml", "w") as f:
                f.write("# Generated by scripts/retarget/init_experiments.py from the code defaults.\n")
                yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"experiments tree ready: {exp}")


if __name__ == "__main__":
    main()
