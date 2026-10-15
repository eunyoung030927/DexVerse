"""Offline release/transfer regressions; no simulator or HF access."""

import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo_tools"))
import demo_release as release
import download_demos as download


def args(**kwargs):
    return SimpleNamespace(**dict(all=False, baseline=False, category=[], task=[], pattern=[], version="all", legacy=False) | kwargs)


def manifest():
    return {"schema": release.SCHEMA, "tasks": [
        {"key": key, "task": key.split("/")[-1], "episodes": 0, "files": [], "status": "missing", "reason": "not collected"}
        for key in release.baseline_entries()]}


def fixture_release(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    data = manifest()
    entry = data["tasks"][0]
    name = f"demonstrations/{entry['key']}/demos.pkl"
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"test payload, no unpickling on transfer")
    entry.update(status="complete", episodes=50, files=[{"path": name, "sha256": release.sha256(path), "bytes": path.stat().st_size, "episodes": 50}])
    (root / release.MANIFEST).write_text(json.dumps(data))
    (root / "README.md").write_text("Dataset card")
    return root, data, path


def test_baseline_defaults_to_both_versions():
    patterns = download._build_patterns(args(baseline=True))
    assert len(patterns) == 40
    assert sum(p.startswith("demonstrations/v0/") for p in patterns) == 20
    assert sum(p.startswith("demonstrations/v1/") for p in patterns) == 20
    assert len(download._build_patterns(args(baseline=True, version="v1"))) == 20


def test_all_release_downloaders_share_public_repo_default():
    assert download.DEFAULT_REPO == "dexverse/DexVerse_release"
    tools = Path(__file__).resolve().parents[1] / "asset_tools"
    for script in ("download_assets.py", "download_robot_agents.py"):
        assert runpy.run_path(str(tools / script))["DEFAULT_REPO"] == download.DEFAULT_REPO


def test_explicit_task_and_category_patterns():
    assert download._build_patterns(args(task=["functional/Dexverse-GraspCup-v1"])) == [
        "demonstrations/v1/functional/Dexverse-GraspCup-v1/**"]
    assert download._build_patterns(args(category=["functional"])) == [
        "demonstrations/v0/functional/**", "demonstrations/v1/functional/**"]
    with pytest.raises(ValueError):
        download._build_patterns(args(task=["v1/functional/Dexverse-GraspCup-v1"], version="v0"))
    assert download._build_patterns(args(task=["Dexverse-PushT-v1"])) == [
        "demonstrations/v1/non_prehensile/Dexverse-PushT-v1/**"]
    for selector in ("Dexverse-PushT-v1", "non_prehensile/Dexverse-PushT-v1"):
        with pytest.raises(ValueError, match="conflicts"):
            download._build_patterns(args(task=[selector], version="v0"))
    with pytest.raises(ValueError, match="Unknown"):
        download._build_patterns(args(task=["Dexverse-PushT"]))


def test_legacy_is_explicit_v0_only():
    patterns = download._build_patterns(args(baseline=True, legacy=True))
    assert len(patterns) == 20 and all("-v0/" in p for p in patterns)
    assert all(not p.startswith("demonstrations/v0/") for p in patterns)
    with pytest.raises(ValueError):
        download._build_patterns(args(baseline=True, version="v1", legacy=True))


@pytest.mark.parametrize("path", ["../escape", "/absolute", "v1/../../escape", "a\\b", "a//b", "a/./b", ""])
def test_unsafe_paths_rejected(path):
    with pytest.raises(ValueError):
        release.safe_relative(path)


def test_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        release.contained_path(root, "escape/test")


def test_manifest_coverage_and_file_validation():
    data = manifest()
    release.validate_manifest(data)
    data["tasks"].pop()
    with pytest.raises(ValueError):
        release.validate_manifest(data)
    data = manifest()
    data["tasks"][0].update(status="complete", episodes=50)
    with pytest.raises(ValueError):
        release.validate_manifest(data)


def test_manifest_selection_reports_gaps(tmp_path):
    _, data, _ = fixture_release(tmp_path)
    entries, gaps = download.select_release(data, download._build_patterns(args(baseline=True)))
    assert len(entries) == 1 and len(gaps) == 39


def test_allow_list_blocks_extra_private_files(tmp_path):
    root, _, _ = fixture_release(tmp_path)
    release.verify_release(root)
    (root / "private_report.json").write_text("private")
    with pytest.raises(ValueError, match="extra"):
        release.verify_release(root)


def test_modified_release_rejected(tmp_path):
    root, _, path = fixture_release(tmp_path)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="mismatch"):
        release.verify_release(root)


def test_offline_validation_cli(tmp_path, monkeypatch, capsys):
    root, _, _ = fixture_release(tmp_path)
    # The validator must not need the remote client or load the opaque pickle.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(sys, "argv", ["demo_release", "--release-dir", str(root)])
    release.main()
    output = capsys.readouterr().out
    assert "Verified 3 allow-listed files" in output
    assert "50 trajectories" in output
    assert "nothing uploaded" in output


def test_download_verify_skip_and_no_overwrite(tmp_path):
    _, data, source = fixture_release(tmp_path)
    entries = data["tasks"][0]["files"]
    calls = []
    def fetch(path):
        calls.append(path)
        return source
    dest = tmp_path / "dest"
    download.fetch_selected(entries, dest, fetch)
    download.fetch_selected(entries, dest, fetch)
    assert len(calls) == 1
    target = dest / entries[0]["path"].removeprefix("demonstrations/")
    target.write_bytes(b"my own data")
    with pytest.raises(FileExistsError):
        download.fetch_selected(entries, dest, fetch)
    assert target.read_bytes() == b"my own data"


def test_corrupt_download_not_installed(tmp_path):
    _, data, source = fixture_release(tmp_path)
    entries = data["tasks"][0]["files"]
    source.write_bytes(b"x" * entries[0]["bytes"])
    with pytest.raises(ValueError, match="checksum"):
        download.fetch_selected(entries, tmp_path / "dest", lambda _: source)
    assert not list((tmp_path / "dest").rglob("*.pkl"))


@pytest.mark.parametrize("mode", ["normal", "dry", "strict"])
@pytest.mark.parametrize("repo", [None, "owner/test"])
def test_download_cli_pins_revision_and_honors_no_fetch_modes(tmp_path, monkeypatch, mode, repo):
    root, data, _ = fixture_release(tmp_path)
    calls = []
    expected_repo = repo or "dexverse/DexVerse_release"
    class API:
        def repo_info(self, **kwargs):
            assert kwargs["repo_id"] == expected_repo
            assert kwargs["revision"] == "a-tag"
            return SimpleNamespace(sha="immutable-commit")
    def fetch(**kwargs):
        assert kwargs["repo_id"] == expected_repo
        assert kwargs["revision"] == "immutable-commit"
        calls.append(kwargs["filename"])
        return str(root / kwargs["filename"])
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=API, hf_hub_download=fetch))
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", SimpleNamespace(EntryNotFoundError=FileNotFoundError))
    argv = ["download", "--revision", "a-tag", "--baseline", "--dest", str(tmp_path / "dest")]
    if repo is not None:
        argv.extend(["--repo", repo])
    if mode == "dry":
        argv.append("--dry-run")
    if mode == "strict":
        argv.append("--require-complete")
    monkeypatch.setattr(sys, "argv", argv)
    if mode == "strict":
        with pytest.raises(SystemExit, match="incomplete"):
            download.main()
    else:
        assert download.main() == 0
    assert calls[0] == release.MANIFEST
    assert len(calls) == (2 if mode == "normal" else 1)
    assert len(list((tmp_path / "dest").rglob("*.pkl"))) == (1 if mode == "normal" else 0)
