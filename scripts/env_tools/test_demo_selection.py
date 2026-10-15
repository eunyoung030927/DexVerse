"""Offline task/version selection tests, including download-to-convert routing."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/demo_tools"))
from demo_release import baseline_entries, baseline_key, sha256, SCHEMA
from demo_selection import discover_groups, validate_selected_identity, worker_arguments, PickleGroup
from download_demos import _build_patterns, fetch_selected, select_release


def write_demo(root, key):
    path = root / key / "demos.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"opaque fixture: selection and transfer must not unpickle")
    return path


@pytest.mark.parametrize("key", baseline_entries())
def test_every_baseline_id_and_path_resolves_exactly(key):
    for selector in (key, key.split("/", 1)[1], key.split("/")[-1]):
        assert baseline_key(selector) == key
        assert baseline_key(selector, key[:2]) == key
        with pytest.raises(ValueError, match="conflicts"):
            baseline_key(selector, "v1" if key.startswith("v0") else "v0")


@pytest.mark.parametrize("selector", ["Dexverse-PushT", "Dexverse-PushT-v2", "../outside", "", "/abs",
                                     "v0/non_prehensile/Dexverse-PushT-v1", "functional/Dexverse-PushT-v1"])
def test_no_version_guessing_or_unsafe_paths(selector):
    with pytest.raises(ValueError):
        baseline_key(selector)


def test_selects_only_canonical_pickle_and_keeps_versions_separate(tmp_path):
    v0, v1 = (f"v{v}/non_prehensile/Dexverse-PushT-v{v}" for v in (0, 1))
    paths = [write_demo(tmp_path, key) for key in (v0, v1)]
    write_demo(tmp_path, v1 + "/local_collection/old")
    write_demo(tmp_path, "_review/" + v1)
    groups, warnings = discover_groups(tmp_path, tasks=["Dexverse-PushT-v0", "Dexverse-PushT-v1", v1])
    assert not warnings and len(groups) == 2
    assert [g.pickles for g in groups] == [[p] for p in paths]
    assert [g.expected_task for g in groups] == ["Dexverse-PushT-v0", "Dexverse-PushT-v1"]
    assert groups[0].output_stem != groups[1].output_stem
    groups, warnings = discover_groups(tmp_path, all_tasks=True, version="v1")
    assert len(groups) == 1 and len(warnings) == 19
    assert groups[0].pickles == [paths[1]]


def test_no_fallback_to_raw_sessions_or_other_version(tmp_path):
    write_demo(tmp_path, "v1/non_prehensile/Dexverse-PushT-v1/local_collection/latest")
    write_demo(tmp_path, "v0/non_prehensile/Dexverse-PushT-v0")
    with pytest.raises(FileNotFoundError, match="Download with"):
        discover_groups(tmp_path, tasks=["Dexverse-PushT-v1"])


def test_version_root_and_conflicting_selectors(tmp_path):
    path = write_demo(tmp_path, "v1/non_prehensile/Dexverse-PushT-v1")
    groups, _ = discover_groups(tmp_path / "v1", tasks=["Dexverse-PushT-v1"])
    assert groups[0].pickles == [path]
    for kwargs in (dict(version="v0"), dict(tasks=["Dexverse-PushT-v0"])):
        with pytest.raises(ValueError, match="conflicts"):
            discover_groups(tmp_path / "v1", all_tasks=not bool(kwargs.get("tasks")), **kwargs)
    with pytest.raises(ValueError):
        discover_groups(tmp_path, tasks=["Dexverse-PushT-v1"], all_tasks=True)
    with pytest.raises(ValueError):
        discover_groups(tmp_path, files=[str(path)], version="v1")


def test_download_and_release_root_select_same_pickle(tmp_path):
    release = tmp_path / "release"
    key = "v1/non_prehensile/Dexverse-PushT-v1"
    source = write_demo(release / "demonstrations", key)
    manifest = {"schema": SCHEMA, "tasks": [dict(key=k, task=k.split("/")[-1], episodes=0,
                files=[], status="missing", reason="No data") for k in baseline_entries()]}
    row = next(r for r in manifest["tasks"] if r["key"] == key)
    row.update(episodes=50, status="complete", files=[dict(path=f"demonstrations/{key}/demos.pkl",
               sha256=sha256(source), bytes=source.stat().st_size, episodes=50)])
    (release / "demonstrations/release_manifest.json").write_text(json.dumps(manifest))
    args = SimpleNamespace(all=False, baseline=False, category=[], task=["Dexverse-PushT-v1"],
                           pattern=[], version="all", legacy=False)
    selected, absent = select_release(manifest, _build_patterns(args))
    assert len(selected) == 1 and not absent
    fetch_selected(selected, tmp_path / "download", lambda name: release / name)
    for root in (release, release / "demonstrations", tmp_path / "download"):
        groups, warnings = discover_groups(root, tasks=["Dexverse-PushT-v1"])
        assert not warnings and len(groups) == 1
        assert groups[0].pickles[0].read_bytes() == source.read_bytes()
    with pytest.raises(FileNotFoundError, match="missing"):
        discover_groups(release, tasks=["Dexverse-OpenDoor-v1"])
    source.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        discover_groups(release, tasks=["Dexverse-PushT-v1"])


def test_symlink_escape_and_review_need_explicit_file(tmp_path):
    source = write_demo(tmp_path / "external", "v1/non_prehensile/Dexverse-PushT-v1")
    root = tmp_path / "root"
    root.mkdir()
    (root / "v1").symlink_to(tmp_path / "external/v1", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        discover_groups(root, tasks=["Dexverse-PushT-v1"])
    groups, _ = discover_groups(root, files=[str(source)])
    assert groups[0].pickles == [source] and groups[0].expected_task is None


def test_legacy_is_explicit_and_filters_review_and_backups(tmp_path):
    path = write_demo(tmp_path, "non_prehensile/Task/session")
    write_demo(tmp_path, "_review/non_prehensile/Task/session")
    write_demo(tmp_path, "non_prehensile/Task copy/session")
    groups, _ = discover_groups(tmp_path, all_tasks=True, legacy_collections=True)
    assert len(groups) == 1 and groups[0].pickles == [path]
    assert groups[0].expected_task is None


def test_wrong_payload_version_or_task_cannot_follow_path_label():
    task = "Dexverse-PushT-v1"
    validate_selected_identity(dict(task=task, env_name=task, task_version=1), task)
    for payload in ({}, dict(task="Dexverse-PushT-v0"), dict(task=task, env_name="Dexverse-PushT-v0"),
                    dict(task=task, task_version=0)):
        with pytest.raises(ValueError):
            validate_selected_identity(payload, task)


def test_cli_selection_precedes_simulator_and_defaults_to_cpu():
    source = (ROOT / "scripts/demo_tools/create_demo_files_sequential.py").read_text()
    assert source.index("_selected_groups, _selection_warnings = discover_groups(") < source.index("app_launcher = AppLauncher(")
    assert 'parser.set_defaults(device="cpu")' in source
    tree = ast.parse(source)
    convert = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_convert_one_group")
    calls = [n for n in ast.walk(convert) if isinstance(n, ast.Call)]
    check = next(n for n in calls if ast.unparse(n.func) == "validate_selected_identity")
    build = next(n for n in calls if ast.unparse(n.func) == "_build_env_for_pickle")
    assert check.lineno < build.lineno


def test_worker_keeps_exact_task_identity_and_nonselection_options(tmp_path):
    group = PickleGroup("v1/non_prehensile/Dexverse-PushT-v1", "Dexverse-PushT-v1", tmp_path,
                        [tmp_path / "demos.pkl"], "Dexverse-PushT-v1")
    arguments = ["--task", "Dexverse-PushT-v0", "--task=Dexverse-PushT-v1", "--version", "all",
                 "--demos-root=somewhere", "--all", "--device", "cpu", "--select-episodes", "0", "1",
                 "--obs-groups", "state", "--output-dir", "out"]
    result = worker_arguments(group, arguments)
    assert result[:-2] == ["--device", "cpu", "--select-episodes", "0", "1", "--obs-groups", "state", "--output-dir", "out"]
    data = json.loads(result[-1])
    assert data["expected_task"] == group.expected_task and data["output_stem"] == group.output_stem
    assert data["pickles"] == [str(tmp_path / "demos.pkl")]
