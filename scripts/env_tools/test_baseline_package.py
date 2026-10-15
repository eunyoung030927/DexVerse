"""Simulator-free regressions for the sibling baseline-v1 package layout."""

import ast
from pathlib import Path
import runpy
import sys
from types import ModuleType

from setuptools import find_packages

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/dexverse"
sys.path.insert(0, str(SOURCE))
from dexverse.benchmark import V1_CONFIGS


def test_baseline_package_discovery_and_original_entrypoint():
    packages = find_packages(where=str(SOURCE))
    assert "dexverse.baseline_v1" in packages
    assert "dexverse.baseline_v1.robots" not in packages
    assert (SOURCE / "dexverse/baseline_v1/control_profile.py").is_file()
    assert "dexverse.tasks.v1" not in packages
    tree = ast.parse((SOURCE / "dexverse/tasks/__init__.py").read_text())
    assert any(isinstance(node, ast.ImportFrom) and node.module == "dexverse"
               and any(alias.name == "baseline_v1" for alias in node.names) for node in tree.body)
    for module, _ in V1_CONFIGS.values():
        assert (SOURCE / "dexverse/baseline_v1/config" / (module.replace(".", "/") + ".py")).is_file()


def test_all_baselines_register_from_sibling_namespace(monkeypatch):
    calls = []
    registration = ModuleType("dexverse.tasks.utils.registration")
    registration.register_env = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, registration.__name__, registration)
    runpy.run_path(str(SOURCE / "dexverse/baseline_v1/__init__.py"), run_name="dexverse.baseline_v1")
    assert calls == [("dexverse.baseline_v1.config", task, module, cls)
                     for task, (module, cls) in V1_CONFIGS.items()]


def test_no_old_namespace_in_runtime_sources():
    old_module = "dexverse.tasks" + ".v1"
    old_path = "tasks/" + "v1/"
    paths = list((SOURCE / "dexverse").rglob("*.py"))
    paths += list((ROOT / "scripts/demo_tools").glob("*.py"))
    for path in paths:
        text = path.read_text()
        assert old_module not in text and old_path not in text, path
