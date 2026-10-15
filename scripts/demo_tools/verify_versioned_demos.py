"""Verify a local demo import and write an ignored, private coverage inventory.

No simulator, network access, uploads, or source modifications are performed.
The inputs must be trusted outputs from import_versioned_demos.py.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from import_versioned_demos import BASELINE_PAIRS, IDENTITY_KEYS, ROOT, array_digest, sha256, write_json


def nonfinite_arrays(value, prefix=""):
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            yield prefix
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from nonfinite_arrays(child, f"{prefix}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from nonfinite_arrays(child, f"{prefix}/{index}")


def verify(workspace):
    report = json.loads((workspace / "import_report.json").read_text())
    plan = json.loads((workspace / "plan.json").read_text())
    dest = Path(report["destination"])
    imported = [result for result in report["results"] if "output" in result]
    failures = [result for result in report["results"] if result["status"] == "failed"]
    rows, episodes_seen, duplicate_episodes, problems = [], {}, [], []
    if len(report["results"]) != len(plan["items"]):
        problems.append({"incomplete_import": {"planned": len(plan["items"]), "processed": len(report["results"])}})
    sources = {item.get("source") or item.get("remote_file"): item for item in plan["items"]}
    for result in imported:
        output = dest / result["output"]
        # Ordinary loading, deliberately without the compatibility unpickler.
        with output.open("rb") as stream:
            payload = pickle.load(stream)
        provenance = payload["demo_import"]
        item = sources[provenance["source"]]
        source = Path(item["source"]) if "source" in item else workspace / "remote_raw" / item["remote_file"]
        expected_hash = item.get("sha256") or item.get("remote_sha256") or result["source_sha256"]
        checks = {
            "source_hash_unchanged": sha256(source) == expected_hash == result["source_sha256"],
            "output_hash_unchanged": sha256(output) == result["output_sha256"],
            "identity_matches": payload["task"] == payload["env_name"] == result["target_task"]
            and payload["task_version"] == result["version"],
            "data_digest_matches": array_digest(
                {key: value for key, value in payload.items() if key not in IDENTITY_KEYS and key != "demo_import"}
            )
            == provenance["array_and_episode_digest"],
        }
        invalid = list(nonfinite_arrays(payload["episodes"]))
        checks["finite_episode_arrays"] = not invalid
        row = {**result, "checks": checks, "nonfinite_array_paths": invalid}
        rows.append(row)
        if not all(checks.values()):
            problems.append({"output": result["output"], "checks": checks})
        for index, episode in enumerate(payload["episodes"]):
            digest = array_digest({key: value for key, value in episode.items() if key != "episode_index"})
            key = (result["target_task"], digest)
            location = {"output": result["output"], "episode_index_in_file": index}
            if key in episodes_seen:
                duplicate_episodes.append({**location, "same_as": episodes_seen[key]})
            else:
                episodes_seen[key] = location
        print(f"Verified {result['output']}", flush=True)

    paths = [str(dest / row["output"]) for row in rows]
    paths.extend(str(path) for path in workspace.rglob("*") if path.is_file())
    ignored = subprocess.run(
        ["git", "check-ignore", "--stdin", "-z"],
        input="\0".join(paths) + "\0",
        text=True,
        capture_output=True,
        check=False,
        cwd=ROOT,
    )
    ignored_paths = set(ignored.stdout.split("\0"))
    unignored = sorted(set(paths) - ignored_paths)
    counts = collections.defaultdict(lambda: {"files": 0, "episodes": 0, "tasks": set(), "numpy_remapped_files": 0})
    for row in rows:
        group = f"review_v{row['version']}" if row["review_reasons"] else f"v{row['version']}"
        count = counts[group]
        count["files"] += 1
        count["episodes"] += row["summary"]["episodes"]
        count["tasks"].add(row["target_task"])
        count["numpy_remapped_files"] += bool(row["numpy_remaps"])
    for count in counts.values():
        count["tasks"] = sorted(count["tasks"])
    verified = {
        "numpy_version": np.__version__,
        "remote_revision": plan["remote_revision"],
        "counts": dict(counts),
        "failures": failures,
        "verification_problems": problems,
        "unignored_paths": unignored,
        "duplicate_episodes": duplicate_episodes,
        "excluded_older_local_sessions": plan["excluded_older_local_sessions"],
        "files": rows,
        "simulation_replay_verified": False,
    }
    write_json(workspace / "verification.json", verified)
    lines = [
        "# Local private demonstration inventory",
        "",
        "Private data: keep this inventory and all imported recordings local and Git-ignored.",
        "",
        f"Destination: `{dest}`",
        "",
        f"Remote snapshot: `{plan['remote_revision']}` (Hugging Face dataset; see plan.json).",
        "",
        f"Normalized and ordinarily reloaded with NumPy {np.__version__}. Source/output hashes and",
        "numerical/episode digests checked. Simulation replay is **not verified**.",
        "",
        "Counts are recorded episode entries; sessions were not merged or episode-filtered.",
        f"Exact repeated episodes (ignoring only episode_index): {len(duplicate_episodes)}.",
        "",
        "| Baseline family | v0 episodes (files) | v1 episodes (files) | Review episodes (files) |",
        "|---|---:|---:|---:|",
    ]
    for v0, v1 in BASELINE_PAIRS.items():
        family = [row for row in rows if row["target_task"] in {v0, v1}]
        cells = []
        for group in (0, 1, "review"):
            matching = [
                row
                for row in family
                if (
                    bool(row["review_reasons"])
                    if group == "review"
                    else not row["review_reasons"] and row["version"] == group
                )
            ]
            cells.append(
                f"{sum(row['summary']['episodes'] for row in matching)} ({len(matching)})" if matching else "—"
            )
        lines.append(f"| {v0.removeprefix('Dexverse-').removesuffix('-v0')} | {' | '.join(cells)} |")
    lines.extend(["", "## Review reasons", ""])
    reasons = collections.Counter(reason for row in rows for reason in row["review_reasons"])
    lines.extend(f"- `{reason}`: {count} files" for reason, count in sorted(reasons.items()))
    lines.extend(
        [
            "",
            "## Selection and provenance",
            "",
            "- Local v1: latest known production collection per task; older sessions, tests, backups and renders excluded.",
            "- Remote v0: explicit baseline reruns preferred; otherwise state-recorded Shadow sessions.",
            "- Known laptop placement-goal prototypes and carton sessions without target-face state are in `_review/v1`.",
            "- Empty sessions are retained in `_review`, not counted as usable demonstrations.",
            "- Missing coverage is not filled with recordings from a different task version or robot family.",
            "- Each pickle's `demo_import` retains source identity, source SHA-256 and normalization provenance.",
            "- `plan.json`, `import_report.json` and `verification.json` contain exact paths and per-file results.",
            "- Originals were not modified. No data was staged, committed, uploaded or pushed.",
            "",
            f"Import failures: {len(failures)}; verification problems: {len(problems)}; unignored data paths: {len(unignored)}.",
            "",
        ]
    )
    (workspace / "INVENTORY.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                key: value
                for key, value in verified.items()
                if key not in {"files", "excluded_older_local_sessions", "duplicate_episodes"}
            },
            indent=2,
        )
    )
    return int(bool(failures or problems or unignored))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "outputs/demo_import")
    raise SystemExit(verify(parser.parse_args().workspace.resolve()))
