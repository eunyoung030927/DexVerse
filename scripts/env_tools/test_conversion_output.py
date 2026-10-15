"""Offline regressions for conversion reuse and worker exit status."""

import copy
from pathlib import Path
import subprocess
import sys

import h5py
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/demo_tools"))
from conversion_output import conversion_request, request_json, validate_existing_output


def make_request(tmp_path, indices=(0, 1)):
    source = tmp_path / "source.pkl"
    if not source.exists():
        source.write_bytes(b"opaque source bytes")
    return conversion_request(
        task="Dexverse-PushT-v1", identity={"benchmark_revision": "test-revision"},
        pickles=[source], episodes=[{"episode_index": i, "actions": [0, 0, 0]} for i in indices],
        options={"set_state": True, "observation_preset": "state", "seed": 42},
    )


def write_output(path, request):
    with h5py.File(path, "w") as h:
        h.attrs.update(conversion_complete=True, conversion_request=request_json(request),
                       task=request["task"], replay_benchmark_revision=request["identity"]["benchmark_revision"],
                       requested_episodes=len(request["episodes"]), num_episodes=len(request["episodes"]))
        for i, ep in enumerate(request["episodes"]):
            g = h.create_group(f"data/demo_{i}")
            g.attrs.update(episode_index=ep["index"], num_samples=ep["steps"])
            for name in ("actions", "source_actions", "obs/state/pose", "next_obs/state/pose", "terminations/success"):
                g.create_dataset(name, shape=(ep["steps"], 1), dtype="f4")


def test_matching_output_is_read_only(tmp_path):
    request = make_request(tmp_path)
    path = tmp_path / "output.h5"
    write_output(path, request)
    before = path.read_bytes()
    validate_existing_output(path, request)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["episodes", "source", "revision", "implementation", "preset", "seed", "mode"])
def test_previous_conversion_cannot_replace_a_different_request(tmp_path, change):
    request = make_request(tmp_path, (0,))
    path = tmp_path / "output.h5"
    write_output(path, request)
    updated = copy.deepcopy(request)
    if change == "episodes":
        updated = make_request(tmp_path, (0, 1))
    elif change == "source":
        (tmp_path / "source.pkl").write_bytes(b"changed source bytes")
        updated = make_request(tmp_path, (0,))
    elif change == "revision":
        updated["identity"]["benchmark_revision"] = "new-revision"
    elif change == "implementation":
        updated["implementation_sha256"] = "changed"
    elif change == "preset":
        updated["options"]["observation_preset"] = "rgb"
    elif change == "seed":
        updated["options"]["seed"] = 43
    else:
        updated["options"]["set_state"] = False
    with pytest.raises(ValueError, match="--overwrite"):
        validate_existing_output(path, updated)


@pytest.mark.parametrize("damage", ["legacy", "incomplete", "request", "count", "indices", "actions", "observations", "corrupt"])
def test_incomplete_or_unverifiable_output_is_rejected(tmp_path, damage):
    request = make_request(tmp_path)
    path = tmp_path / "output.h5"
    write_output(path, request)
    if damage == "corrupt":
        path.write_bytes(b"not an HDF5")
    else:
        with h5py.File(path, "a") as h:
            if damage == "legacy":
                del h.attrs["conversion_complete"]
            elif damage == "incomplete":
                h.attrs["conversion_complete"] = False
            elif damage == "request":
                del h.attrs["conversion_request"]
            elif damage == "count":
                del h["data/demo_1"]
            elif damage == "indices":
                h["data/demo_1"].attrs["episode_index"] = 0
            else:
                name = "actions" if damage == "actions" else "next_obs/state/pose"
                del h[f"data/demo_0/{name}"]
                h.create_dataset(f"data/demo_0/{name}", shape=(1, 1), dtype="f4")
    with pytest.raises(ValueError, match="--overwrite"):
        validate_existing_output(path, request)


@pytest.mark.parametrize("body,expected", [("return 0", 0), ("return 7", 7),
                                           ("raise ValueError('conversion failed')", 1),
                                           ("raise KeyboardInterrupt", 130)])
def test_worker_preserves_exit_status_even_with_destructive_atexit(body, expected):
    # Models Kit shutdown replacing the status at interpreter exit.
    program = f"""
import atexit, os, sys
sys.path.insert(0, {str(ROOT / 'scripts/demo_tools')!r})
from conversion_output import exit_conversion
atexit.register(lambda: os._exit(0))
def run():
    print('flushed worker output')
    {body}
exit_conversion(run)
"""
    result = subprocess.run([sys.executable, "-B", "-c", program], capture_output=True, text=True, timeout=20)
    assert result.returncode == expected
    assert "flushed worker output" in result.stdout
    if expected == 1:
        assert "ValueError: conversion failed" in result.stderr
