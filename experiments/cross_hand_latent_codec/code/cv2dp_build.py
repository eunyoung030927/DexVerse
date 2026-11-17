"""CV2DP (261115): build the 2-hand (shadow + wuji) nohot GraspKettle LeRobot v3.0 sets for the codec-v2 DP test.
New file. Source = pooled3hand_nohot/kettle3_Z (E2* 3-hand nohot Z set): the shadow and wuji episodes (same episodes,
same frames, same videos) are copied in kettle3 order (new eps 0-49 = shadow src 0-49, 50-99 = wuji src 0-49);
observation.state / action are replaced by the rows of artifacts/cv2dp/encoded_<enc>_<hand>.npz (cv2dp_encode.py).
Videos are hard-linked (identical bytes). meta/episodes: source rows with re-indexed episode/data/video indices and
recomputed per-episode stats (state, action, episode_index, index; image/timestamp/frame/task stats copied, they do
not change); meta/stats.json = lerobot aggregate_stats over the 100 per-episode stats (the procedure is first
checked to reproduce kettle3_Z's own stats.json from kettle3_Z's per-episode stats).
Verification: LeRobotDataset load, items 0/49/50/99 state/action == npz rows, image shape; stats vs rows.
Container lerobot python, cwd /workspace/xhand/latent_codec_v1:
  python -m xhand_latent.cv2dp_build --name ws_Ze2 --enc e2      (and --name ws_Zv2 --enc <v2 tag>)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.utils import write_stats

SRC = Path("/workspace/xhand/datasets/pooled3hand_nohot/kettle3_Z")
DST = Path("/workspace/xhand/datasets/cv2dp")
ENC = Path("artifacts/cv2dp")
HANDS = ("shadow", "wuji")
VKEYS = ("observation.images.third_person", "observation.images.eye_in_hand")
RECOMPUTED = ("observation.state", "action", "episode_index", "index")


def ep_stats_from_row(row, keys):
    out = {}
    for k in keys:
        out[k] = {s: np.asarray(row[f"stats/{k}/{s}"], dtype=np.float64) for s in
                  ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")}
        out[k]["count"] = out[k]["count"].astype(np.int64)
    return out


def max_abs_stats(a, b):
    m = 0.0
    for k in a:
        for s in a[k]:
            m = max(m, float(np.abs(np.asarray(a[k][s], np.float64) - np.asarray(b[k][s], np.float64)).max()))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--enc", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    info = json.loads((SRC / "meta/info.json").read_text())
    feats = info["features"]
    stat_keys = [k for k in feats]
    src_eps = pq.read_table(SRC / "meta/episodes/chunk-000/file-000.parquet")
    src_rows = src_eps.to_pylist()
    pooled = json.loads((SRC / "meta/p3h_pooled.json").read_text())

    # 0. procedure check: aggregate of kettle3_Z per-episode stats == kettle3_Z stats.json
    src_stats = json.loads((SRC / "meta/stats.json").read_text())
    agg = aggregate_stats([ep_stats_from_row(r, stat_keys) for r in src_rows])
    proc_check = max_abs_stats({k: agg[k] for k in src_stats}, src_stats)
    print(f"[build] aggregate procedure vs kettle3_Z stats.json max abs {proc_check:.3e}", flush=True)
    assert proc_check < 1e-6, proc_check

    root = DST / args.name
    if root.exists():
        if not args.force:
            raise SystemExit(f"{root} exists (use --force)")
        shutil.rmtree(root)
    (root / "data/chunk-000").mkdir(parents=True)
    for vk in VKEYS:
        (root / f"videos/{vk}/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)

    enc = {h: np.load(ENC / f"encoded_{args.enc}_{h}.npz") for h in HANDS}
    offs = {h: np.r_[0, np.cumsum(enc[h]["lengths"])] for h in HANDS}
    sel = [e for h in HANDS for e in sorted((e for e in pooled["episodes"] if e["hand"] == h),
                                            key=lambda e: e["source_episode"])]
    assert len(sel) == 100
    new_rows, new_eps_meta, ep_stats, frm = [], [], [], 0
    all_s, all_a = [], []
    for i, e in enumerate(sel):
        h, se, si = e["hand"], e["source_episode"], e["episode_index"]
        t = pq.read_table(SRC / f"data/chunk-000/file-{si:03d}.parquet")
        n = t.num_rows
        L = int(enc[h]["lengths"][se])
        assert n == L == e["length"], (h, se, n, L)
        s = enc[h]["state_Z"][offs[h][se]:offs[h][se + 1]].astype(np.float32)
        a = enc[h]["action_Z"][offs[h][se]:offs[h][se + 1]].astype(np.float32)
        assert np.array_equal(s[:, :9], np.array(t.column("observation.state").to_pylist(), np.float32)[:, :9])
        cols = {}
        for name in t.schema.names:
            if name == "observation.state":
                cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(s.reshape(-1), pa.float32()), 41)
            elif name == "action":
                cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), pa.float32()), 41)
            elif name == "episode_index":
                cols[name] = pa.array(np.full(n, i, np.int64))
            elif name == "index":
                cols[name] = pa.array(np.arange(frm, frm + n, dtype=np.int64))
            else:
                cols[name] = t.column(name)
        nt = pa.Table.from_arrays([cols[c] for c in t.schema.names], schema=t.schema)
        pq.write_table(nt, root / f"data/chunk-000/file-{i:03d}.parquet")
        for vk in VKEYS:
            os.link(SRC / f"videos/{vk}/chunk-000/file-{si:03d}.mp4", root / f"videos/{vk}/chunk-000/file-{i:03d}.mp4")
        r = dict(src_rows[si])
        assert r["episode_index"] == si and r["length"] == n
        st = ep_stats_from_row(r, stat_keys)
        new = compute_episode_stats({"observation.state": s, "action": a, "episode_index": np.full(n, i, np.int64),
                                     "index": np.arange(frm, frm + n, dtype=np.int64)}, feats)
        for k in RECOMPUTED:
            st[k] = {}
            for sname, v in new[k].items():
                v = np.asarray(v).reshape(np.asarray(r[f"stats/{k}/{sname}"]).shape)  # keep the source nesting
                st[k][sname] = v
                r[f"stats/{k}/{sname}"] = v.tolist()
        r["episode_index"] = i
        r["data/file_index"] = i
        r["dataset_from_index"], r["dataset_to_index"] = frm, frm + n
        for vk in VKEYS:
            r[f"videos/{vk}/file_index"] = i
        new_rows.append(r)
        ep_stats.append(st)
        new_eps_meta.append({"episode_index": i, "hand": h, "source_episode": se, "kettle3_episode_index": si,
                             "length": n, "dataset_from_index": frm})
        all_s.append(s); all_a.append(a)
        frm += n
    pq.write_table(pa.Table.from_pylist(new_rows, schema=src_eps.schema), root / "meta/episodes/chunk-000/file-000.parquet")
    stats = aggregate_stats(ep_stats)
    write_stats(stats, root)
    shutil.copy(SRC / "meta/tasks.parquet", root / "meta/tasks.parquet")
    info2 = dict(info)
    info2.update(total_episodes=100, total_frames=frm, splits={"train": "0:100"}, robot_type="dexverse_floating_pooled2ws_right")
    (root / "meta/info.json").write_text(json.dumps(info2, indent=4))
    enc_rep = json.loads((ENC / f"encode_{args.enc}.json").read_text())
    (root / "meta/p3h_pooled.json").write_text(json.dumps(
        {"arm": args.name, "variant": "cv2dp_2hand_nohot", "one_hot": False, "hands": list(HANDS), "state_dim": 41,
         "action_dim": 41, "codec_ckpt": enc_rep["ckpt"], "codec_md5": enc_rep["ckpt_md5"],
         "rows": f"latent_codec_v1/artifacts/cv2dp/encoded_{args.enc}_<hand>.npz (cv2dp_encode.py)",
         "source_dataset": str(SRC), "episodes": new_eps_meta}, indent=1))

    # verification
    S, A = np.concatenate(all_s), np.concatenate(all_a)
    rows_stats = {"observation.state": S, "action": A}
    sv = {k: max(float(np.abs(stats[k]["mean"] - v.mean(0)).max()), float(np.abs(stats[k]["std"] - v.std(0)).max()),
                 float(np.abs(stats[k]["min"] - v.min(0)).max()), float(np.abs(stats[k]["max"] - v.max(0)).max()))
          for k, v in rows_stats.items()}
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f"local/{args.name}", root=root)
    items = {}
    for k in (0, 49, 50, 99):
        m = new_eps_meta[k]
        it = ds[m["dataset_from_index"] + 5]
        h = m["hand"]
        o = offs[h][m["source_episode"]] + 5
        items[k] = {"hand": h, "src_ep": m["source_episode"],
                    "state_eq": bool(np.array_equal(it["observation.state"].numpy(), enc[h]["state_Z"][o])),
                    "action_eq": bool(np.array_equal(it["action"].numpy(), enc[h]["action_Z"][o])),
                    "img": [len(VKEYS)] + list(it[VKEYS[0]].shape)}
    rep = {"root": str(root), "enc": args.enc, "codec_ckpt": enc_rep["ckpt"], "frames": frm, "episodes": 100,
           "len": len(ds), "num_episodes": ds.num_episodes, "items": items, "stats_vs_rows_max_abs": sv,
           "aggregate_procedure_check_kettle3_Z": proc_check,
           "frames_per_hand": {h: int(enc[h]["lengths"].sum()) for h in HANDS}}
    assert len(ds) == frm and ds.num_episodes == 100 and all(v["state_eq"] and v["action_eq"] for v in items.values())
    assert max(sv.values()) < 1e-4, sv
    (root / "meta/cv2dp_build.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
    print(f"[build] {args.name} DONE", flush=True)


if __name__ == "__main__":
    main()
