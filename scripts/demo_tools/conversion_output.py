# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evidence required to reuse a completed sequential conversion, without Isaac."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import h5py

from demo_release import ROOT, sha256


@lru_cache(maxsize=1)
def implementation_digest():
    """Include local source edits as well as the declared release version."""
    paths = [ROOT / "source/dexverse/config/extension.toml"]
    for directory in (ROOT / "source/dexverse/dexverse", ROOT / "scripts/demo_tools"):
        paths.extend(p for p in directory.rglob("*") if p.suffix in {".py", ".yml", ".yaml"})
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(sha256(path).encode() + b"\0")
    return digest.hexdigest()


def conversion_request(*, task, identity, pickles, episodes, options):
    """Describe the exact requested inputs and replay/output settings."""
    return {
        "schema": 1,
        "implementation_sha256": implementation_digest(),
        "task": task,
        "identity": identity,
        "sources": [{"sha256": sha256(p), "bytes": Path(p).stat().st_size} for p in pickles],
        "episodes": [{"index": int(ep["episode_index"]), "steps": len(ep["actions"])} for ep in episodes],
        "options": options,
    }


def request_json(request):
    return json.dumps(request, sort_keys=True, separators=(",", ":"))


def validate_existing_output(path, request):
    """Refuse stale, partial, legacy or mismatched H5 instead of silently skipping."""
    try:
        with h5py.File(path, "r") as existing:
            if not existing.attrs.get("conversion_complete", False):
                raise ValueError("completion marker is missing or false")
            if existing.attrs.get("conversion_request") != request_json(request):
                raise ValueError("sources, episodes, implementation, revision or conversion settings differ")
            if existing.attrs.get("task") != request["task"]:
                raise ValueError("task identity differs")
            if existing.attrs.get("replay_benchmark_revision") != request["identity"]["benchmark_revision"]:
                raise ValueError("replay revision differs")
            expected = {ep["index"]: ep["steps"] for ep in request["episodes"]}
            data = existing["data"]
            if (len(expected) != len(request["episodes"]) or len(data) != len(expected)
                    or existing.attrs.get("requested_episodes") != len(expected)
                    or existing.attrs.get("num_episodes") != len(expected)):
                raise ValueError("episode count differs")
            seen = set()
            for episode in data.values():
                index = int(episode.attrs["episode_index"])
                if index not in expected or index in seen:
                    raise ValueError("episode indices differ")
                seen.add(index)
                steps = expected[index]
                if episode.attrs.get("num_samples") != steps:
                    raise ValueError("episode length differs")
                for name in ("actions", "source_actions"):
                    if episode[name].shape[0] != steps:
                        raise ValueError(f"truncated {name}")
                for name in ("obs", "next_obs", "terminations"):
                    if name == "terminations" and name not in episode:
                        continue

                    def check_length(_, dataset):
                        if isinstance(dataset, h5py.Dataset) and (not dataset.shape or dataset.shape[0] != steps):
                            raise ValueError(f"truncated {name}")

                    episode[name].visititems(check_length)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot reuse {path}: {exc}. Choose a new output directory or pass --overwrite."
        ) from exc


def exit_conversion(run):
    """Flush finalized outputs/logs and exit without Kit replacing our exit code.

    Each converter process owns one scene. Its run function closes H5/video
    writers before returning or raising; process exit releases the simulator.
    """
    import os
    import sys
    import traceback

    code = 1
    try:
        code = run()
    except KeyboardInterrupt:
        code = 130
        traceback.print_exc()
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
