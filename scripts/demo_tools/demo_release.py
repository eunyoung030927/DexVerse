"""Simulator-free manifest and safe file operations for versioned demo releases."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEMOS = ROOT / "source/dexverse/demonstrations"
MANIFEST = "demonstrations/release_manifest.json"
SCHEMA = "dexverse-demo-release-v1"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in {"..", "."} for p in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value!r}")
    if str(path) != value:
        raise ValueError(f"Noncanonical relative path: {value!r}")
    return path


def contained_path(root, relative):
    root = Path(root).resolve()
    path = root / str(safe_relative(relative))
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes root through symlink: {relative}")
    return path


def baseline_entries(version="all"):
    result = []
    for ver in ("v0", "v1") if version == "all" else (version,):
        if ver not in {"v0", "v1"}:
            raise ValueError(f"Unknown version {ver}")
        name = "baseline_manifest.txt" if ver == "v0" else "baseline_manifest_v1.txt"
        for line in (DEMOS / name).read_text().splitlines():
            entry = line.strip()
            if entry and not entry.startswith("#"):
                safe_relative(entry)
                result.append(f"{ver}/{entry}")
    return result


def baseline_key(selector, version="all"):
    """Resolve a versioned task ID or category path; never guess a version."""
    safe_relative(selector)
    keys = baseline_entries()
    matches = [key for key in keys if selector in (key, key.split("/", 1)[1], key.split("/")[-1])]
    if len(matches) != 1:
        raise ValueError(f"Unknown baseline task {selector!r}. Use an explicit ID such as Dexverse-PushT-v0 or Dexverse-PushT-v1.")
    key = matches[0]
    if version not in {"all", "v0", "v1"} or version not in ("all", key.split("/")[0]):
        raise ValueError(f"Task {selector!r} conflicts with --version {version}")
    return key


def validate_manifest(data):
    if data.get("schema") != SCHEMA:
        raise ValueError("Unsupported demo release manifest schema")
    expected = set(baseline_entries())
    tasks, paths = set(), set()
    for task in data["tasks"]:
        key = task["key"]
        if key not in expected or key in tasks:
            raise ValueError(f"Unknown/duplicate baseline: {key}")
        tasks.add(key)
        if task["task"] != key.split("/")[-1]:
            raise ValueError(f"Task identity mismatch: {key}")
        count = task["episodes"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid episode count")
        files = task["files"]
        if task["status"] not in {"complete", "partial", "missing", "skipped"}:
            raise ValueError("Invalid coverage status")
        if (count > 0) != bool(files) or (task["status"] == "complete" and count != 50):
            raise ValueError(f"Inconsistent coverage: {key}")
        if task["status"] in {"missing", "skipped"} and (count or files):
            raise ValueError(f"Unavailable task has data: {key}")
        total = 0
        for entry in files:
            path = str(safe_relative(entry["path"]))
            if path != f"demonstrations/{key}/demos.pkl" or path in paths:
                raise ValueError(f"Unexpected/duplicate release path: {path}")
            paths.add(path)
            digest = entry["sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid SHA256")
            if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
                raise ValueError("Invalid file size")
            if type(entry["episodes"]) is not int or entry["episodes"] <= 0:
                raise ValueError("Invalid file episode count")
            total += entry["episodes"]
        if total != count:
            raise ValueError(f"File counts disagree: {key}")
    if tasks != expected:
        raise ValueError(f"Manifest must account for all baseline versions; missing {sorted(expected - tasks)}")
    return data


def read_manifest(path):
    return validate_manifest(json.loads(Path(path).read_text()))


def verify_release(root):
    root = Path(root).resolve()
    manifest = read_manifest(contained_path(root, MANIFEST))
    allowed = {MANIFEST, "README.md"}
    for task in manifest["tasks"]:
        for entry in task["files"]:
            path = contained_path(root, entry["path"])
            if path.is_symlink() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Release checksum/size mismatch: {path}")
            allowed.add(entry["path"])
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinks not allowed in a release: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != allowed:
        raise ValueError(f"Release files differ from allow-list: extra={actual-allowed}, missing={allowed-actual}")
    return manifest, sorted(allowed)


def install_file(source, destination, digest=None):
    """Atomic copy, no replacement of differing data; identical files are no-ops."""
    source, destination = Path(source), Path(destination)
    digest = digest or sha256(source)
    if sha256(source) != digest:
        raise ValueError(f"Source checksum mismatch: {source}")
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or sha256(destination) != digest:
            raise FileExistsError(f"Refusing to overwrite different local data: {destination}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".demo-download-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as stream:
            shutil.copyfileobj(stream, out)
            out.flush()
            os.fsync(out.fileno())
        if sha256(temp) != digest:
            raise ValueError("Copy checksum mismatch")
        os.link(temp, destination)
    finally:
        os.unlink(temp)
    return True


def main():
    """Offline release validation; no HF dependency, unpickling or network calls."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify a prepared demo release locally (no uploads).")
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, files = verify_release(args.release_dir)
    total = sum((args.release_dir / name).stat().st_size for name in files)
    episodes = sum(task["episodes"] for task in manifest["tasks"])
    print(f"Verified {len(files)} allow-listed files ({total:,} bytes), {episodes:,} trajectories.")
    for task in manifest["tasks"]:
        print(f"  {task['key']}: {task['status']} ({task['episodes']})")
    print("Local validation only: nothing uploaded.")


if __name__ == "__main__":
    main()
