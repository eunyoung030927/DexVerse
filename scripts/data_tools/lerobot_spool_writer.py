# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Consume a LeRobot SPOOL written by ``record_demos.py --lerobot_root`` into a LeRobot v3 dataset.

CPU only -- no Isaac Sim, no GPU. Runs in its own Python env that has ``lerobot`` (the ``lerobot`` conda env
in the dexverse container), so Isaac Sim's interpreter never gets LeRobot's dependencies.

``record_demos.py`` launches this automatically (``--follow --parent_pid``) and keeps recording while it
encodes. Run it BY HAND to resume a spool that was left behind (writer crashed, ``--no_spool_writer``,
machine rebooted)::

    /opt/conda/envs/lerobot/bin/python scripts/data_tools/lerobot_spool_writer.py --spool <lerobot_root>.spool

Everything it needs (dataset root, repo id, fps, features, task string) is in ``<spool>/spool.json``.

ORDER OF OPERATIONS PER EPISODE (crash-safe by construction):
  1. ``add_frame`` x T (threaded PNG writes) -> ``save_episode`` (AV1 encode of both cameras)
  2. ``finalize()`` -- parquet footers written now, so every episode so far stays readable if this process
     is killed later; the dataset is re-opened (LeRobot's resume path) before the next episode
  3. mark the spool entry WRITTEN, append its line to ``meta/isaac_tasks_episodes.jsonl``, delete it.
A kill during (1) leaves the spool entry in place; a kill between (2) and (3) is detected by the WRITTEN mark.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPOOL_PY = os.path.normpath(os.path.join(_HERE, "..", "..", "source", "dexverse", "dexverse", "data_collection",
                                          "lerobot_spool.py"))


def _load_spool_module():
    """The stdlib+numpy spool format module, loaded BY PATH (importing the dexverse package pulls Isaac Lab in)."""
    spec = importlib.util.spec_from_file_location("_dexverse_lerobot_spool", _SPOOL_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_dexverse_lerobot_spool"] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load_spool_module()


def log(msg):
    print(f"[spool-writer {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pid_alive(pid):
    if not pid:
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _features_for_lerobot(features):
    """spool.json stores lists; LeRobot wants tuples for shapes."""
    return {k: {**v, "shape": tuple(v["shape"])} for k, v in features.items()}


class Writer:
    def __init__(self, spool, root, repo_id, info, image_threads, parallel_encoding):
        self.spool, self.root, self.repo_id, self.info = spool, Path(root), repo_id, info
        self.image_threads = int(image_threads)
        self.parallel_encoding = bool(parallel_encoding)
        self.ds = None
        self.n_written = 0
        self.sidecar = self.root / "meta" / "isaac_tasks_episodes.jsonl"
        self.value_keys = [k for k, v in info["features"].items() if v["dtype"] != "video"]

    def _open(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        info_json = self.root / "meta" / "info.json"
        want = _features_for_lerobot(self.info["features"])
        if info_json.exists():
            ds = LeRobotDataset(self.repo_id, root=self.root)
            have = ds.meta.features
            for k, v in want.items():
                h = have.get(k)
                if h is None or tuple(h["shape"]) != tuple(v["shape"]) or h["dtype"] != v["dtype"] or (
                        v["dtype"] != "video" and list(h.get("names") or []) != list(v.get("names") or [])):
                    raise SystemExit(f"[spool-writer] existing dataset {self.root} has feature {k}={h}, the spool "
                                     f"wants {v}: refusing to append incompatible episodes.")
            if int(ds.meta.fps) != int(self.info["fps"]):
                raise SystemExit(f"[spool-writer] fps mismatch: dataset {ds.meta.fps} vs spool {self.info['fps']}")
            ds.start_image_writer(num_processes=0, num_threads=self.image_threads)
            log(f"opened existing dataset {self.root}: {ds.meta.total_episodes} episodes, "
                f"{ds.meta.total_frames} frames (appending)")
        else:
            if self.root.exists():
                if any(self.root.iterdir()):
                    raise SystemExit(f"[spool-writer] {self.root} exists, is not empty and has no meta/info.json "
                                     f"-- refusing to touch it.")
                self.root.rmdir()
            self.root.parent.mkdir(parents=True, exist_ok=True)
            ds = LeRobotDataset.create(repo_id=self.repo_id, fps=int(self.info["fps"]), root=self.root,
                                       robot_type=self.info.get("robot_type"), features=want,
                                       use_videos=True, image_writer_threads=self.image_threads)
            static = {k: v for k, v in self.info.items() if k != "features"}
            (self.root / "meta" / "isaac_tasks.json").write_text(
                json.dumps(static, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            log(f"created dataset {self.root} (repo_id {self.repo_id}, fps {self.info['fps']})")
        self.ds = ds

    def _sidecar_has(self, idx):
        if not self.sidecar.exists():
            return False
        with open(self.sidecar, encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line).get("episode_index") == idx:
                        return True
                except json.JSONDecodeError:
                    continue
        return False

    def _append_sidecar(self, idx, meta, ep_dir):
        if self._sidecar_has(idx):
            return
        rec = {"episode_index": int(idx), "spool_entry": os.path.basename(ep_dir), **meta}
        self.sidecar.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sidecar, "a", encoding="utf-8") as f:
            f.write(json.dumps(S._jsonable(rec), ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write(self, ep_dir):
        mark = os.path.join(ep_dir, S.WRITTEN_MARK)
        if os.path.exists(mark):  # crashed after finalize, before cleanup
            with open(mark, encoding="utf-8") as f:
                idx = int(json.load(f)["episode_index"])
            with open(os.path.join(ep_dir, S.META_FILE), encoding="utf-8") as f:
                meta = json.load(f)
            self._append_sidecar(idx, meta, ep_dir)
            shutil.rmtree(ep_dir)
            log(f"{os.path.basename(ep_dir)}: already written as episode {idx}; cleaned up")
            return
        if self.ds is None:
            self._open()
        ep = S.load_episode(ep_dir)
        meta, T = ep["meta"], ep["num_frames"]
        missing = [k for k in self.value_keys if k not in ep["arrays"]]
        if missing:
            raise SystemExit(f"[spool-writer] {ep_dir} lacks {missing}")
        task = meta.get("task") or self.info["task"]
        t0 = time.time()
        for t in range(T):
            frame = {"task": task}
            for k in self.value_keys:
                frame[k] = ep["arrays"][k][t]
            for key, frames in ep["frames"].items():
                frame[key] = frames[t]
            self.ds.add_frame(frame)
        t1 = time.time()
        self.ds.save_episode(parallel_encoding=self.parallel_encoding)
        t2 = time.time()
        self.ds.finalize()
        idx = int(self.ds.meta.total_episodes) - 1
        self.ds.stop_image_writer()
        self.ds = None
        with open(mark, "w", encoding="utf-8") as f:
            json.dump({"episode_index": idx, "time": time.time()}, f)
        meta = {**meta, "writer_timing": {"add_frames_s": t1 - t0, "save_episode_s": t2 - t1,
                                          "finalize_s": time.time() - t2}}
        self._append_sidecar(idx, meta, ep_dir)
        shutil.rmtree(ep_dir)
        self.n_written += 1
        log(f"{os.path.basename(ep_dir)} -> episode {idx}: {T} frames, add_frame {t1 - t0:.1f} s, "
            f"save_episode {t2 - t1:.1f} s, finalize {time.time() - t2:.1f} s")

    def close(self):
        if self.ds is not None:
            self.ds.finalize()
            self.ds.stop_image_writer()
            self.ds = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spool", required=True, help="<lerobot_root>.spool directory")
    ap.add_argument("--root", default=None, help="override the dataset root stored in spool.json")
    ap.add_argument("--repo_id", default=None, help="override the repo id stored in spool.json")
    ap.add_argument("--follow", action="store_true",
                    help="keep polling for new episodes until the DONE sentinel (or --parent_pid exits)")
    ap.add_argument("--parent_pid", type=int, default=None)
    ap.add_argument("--image_threads", type=int, default=8)
    ap.add_argument("--no_parallel_encoding", action="store_true")
    ap.add_argument("--poll", type=float, default=1.0)
    args = ap.parse_args(argv)

    spool = os.path.abspath(args.spool)
    info_path = os.path.join(spool, S.SPOOL_INFO)
    if not os.path.exists(info_path):
        raise SystemExit(f"[spool-writer] {info_path} not found -- not a spool directory")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    root = os.path.abspath(args.root or info["root"])
    repo_id = args.repo_id or info["repo_id"]

    lock_f = open(os.path.join(spool, S.WRITER_LOCK), "a")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if not args.follow:
            raise SystemExit(f"[spool-writer] another writer holds {os.path.join(spool, S.WRITER_LOCK)}; exiting")
        # A new recording session into the same root while the previous session's writer is still encoding:
        # wait for it to finish, then take over this session's episodes.
        log("another writer (previous session) still holds the lock; waiting for it to finish")
        fcntl.flock(lock_f, fcntl.LOCK_EX)
    lock_f.seek(0)
    lock_f.truncate()
    lock_f.write(str(os.getpid()))
    lock_f.flush()

    log(f"spool {spool} -> {root} (repo_id {repo_id}) follow={args.follow} parent={args.parent_pid} "
        f"cpu-only (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    for tmp in S.list_tmp(spool):
        log(f"NOTE: {os.path.basename(tmp)} is an unfinished attempt (in progress, or left by a crashed recorder) "
            f"-- never written")
    w = Writer(spool, root, repo_id, info, args.image_threads, not args.no_parallel_encoding)
    rc = 0
    try:
        while True:
            ready = S.list_ready(spool)
            for ep_dir in ready:
                w.write(ep_dir)
            if ready:
                continue
            if not args.follow:
                break
            if os.path.exists(os.path.join(spool, S.DONE_SENTINEL)):
                log("DONE sentinel seen and the queue is empty")
                break
            if not pid_alive(args.parent_pid):
                log(f"parent {args.parent_pid} is gone and the queue is empty -- finalizing")
                break
            time.sleep(args.poll)
    except BaseException as exc:  # noqa: BLE001
        import traceback

        log(f"!!! FAILED: {exc!r}\n{traceback.format_exc()}\nThe spool is kept; re-run this script to resume.")
        rc = 1
    finally:
        try:
            w.close()
        except Exception as exc:  # noqa: BLE001
            log(f"close failed: {exc!r}")
            rc = 1
    done = os.path.join(spool, S.DONE_SENTINEL)
    if rc == 0 and os.path.exists(done) and not S.list_ready(spool):
        os.remove(done)
    log(f"exit {rc}: wrote {w.n_written} episode(s) this session; {len(S.list_ready(spool))} left in the spool")
    return rc


if __name__ == "__main__":
    sys.exit(main())
