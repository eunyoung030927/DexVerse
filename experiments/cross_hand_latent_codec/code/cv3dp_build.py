"""CV3DP (261116): build the 3-hand (shadow + wuji + allegro) nohot GraspKettle LeRobot v3.0 sets for the codec-v3 DP test
(held-out test hands sharpa, leap). Copy of cv2dp_build.py (unchanged original) with only: a third hand (allegro) whose
episodes come from ho2hand/sa_Z (E2* H2 shadow+allegro nohot Z set; same features / tasks as kettle3_Z, checked), rows
from artifacts/codec_v3/cv3dp/encoded_<enc>_<hand>.npz, output /workspace/xhand/datasets/cv3dp/<name>, 150 episodes.
New eps 0-49 = shadow src 0-49, 50-99 = wuji src 0-49 (both from kettle3_Z, as cv2dp), 100-149 = allegro src 0-49 (sa_Z).
observation.state / action are replaced by the encoded rows; videos are hard-linked (identical bytes). meta/episodes:
source rows with re-indexed episode/data/video indices and recomputed per-episode stats (state, action, episode_index,
index; image/timestamp/frame/task stats copied); meta/stats.json = lerobot aggregate_stats over the 150 per-episode stats
(the procedure is first checked to reproduce kettle3_Z's and sa_Z's own stats.json from their per-episode stats).
Verification: LeRobotDataset load, items 0/49/50/99/100/149 state/action == npz rows, image shape; stats vs rows.
Container lerobot python, cwd /workspace/xhand/latent_codec_v1:
  python -m xhand_latent.cv3dp_build --name wsa_Ze3 --enc e2      (and --name wsa_Zw3 --enc v3wbcll)
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

SRCS = {"shadow": Path("/workspace/xhand/datasets/pooled3hand_nohot/kettle3_Z"),
        "wuji": Path("/workspace/xhand/datasets/pooled3hand_nohot/kettle3_Z"),
        "allegro": Path("/workspace/xhand/datasets/ho2hand/sa_Z")}
DST = Path("/workspace/xhand/datasets/cv3dp")
ENC = Path("artifacts/codec_v3/cv3dp")
HANDS = ("shadow", "wuji", "allegro")
VKEYS = ("observation.images.third_person", "observation.images.eye_in_hand")
RECOMPUTED = ("observation.state", "action", "episode_index", "index")
N_EP = 50 * len(HANDS)


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
    roots = sorted(set(SRCS.values()))
    infos = {r: json.loads((r / "meta/info.json").read_text()) for r in roots}
    info = infos[SRCS["shadow"]]
    feats = info["features"]
    assert all(i["features"] == feats and i["fps"] == info["fps"] for i in infos.values()), "source features differ"
    stat_keys = [k for k in feats]
    src_eps = {r: pq.read_table(r / "meta/episodes/chunk-000/file-000.parquet") for r in roots}
    schema = src_eps[SRCS["shadow"]].schema
    assert all(t.schema.equals(schema) for t in src_eps.values()), "source episode-meta schemas differ"
    src_rows = {r: t.to_pylist() for r, t in src_eps.items()}
    pooled = {r: json.loads((r / "meta/p3h_pooled.json").read_text()) for r in roots}
    tasks = {r: pq.read_table(r / "meta/tasks.parquet") for r in roots}
    assert all(t.equals(tasks[SRCS["shadow"]]) for t in tasks.values()), "source tasks differ"

    # 0. procedure check: aggregate of each source's per-episode stats == its stats.json
    proc_check = {}
    for r in roots:
        src_stats = json.loads((r / "meta/stats.json").read_text())
        agg = aggregate_stats([ep_stats_from_row(x, stat_keys) for x in src_rows[r]])
        proc_check[r.name] = max_abs_stats({k: agg[k] for k in src_stats}, src_stats)
        print(f"[build] aggregate procedure vs {r.name} stats.json max abs {proc_check[r.name]:.3e}", flush=True)
        assert proc_check[r.name] < 1e-6, proc_check

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
    sel = [(h, e) for h in HANDS for e in sorted((e for e in pooled[SRCS[h]]["episodes"] if e["hand"] == h),
                                                 key=lambda e: e["source_episode"])]
    assert len(sel) == N_EP and [e["source_episode"] for h, e in sel] == list(range(50)) * len(HANDS)
    new_rows, new_eps_meta, ep_stats, frm = [], [], [], 0
    all_s, all_a = [], []
    for i, (h, e) in enumerate(sel):
        SRC = SRCS[h]
        se, si = e["source_episode"], e["episode_index"]
        t = pq.read_table(SRC / f"data/chunk-000/file-{si:03d}.parquet")
        assert t.schema.names == pq.read_schema(SRCS["shadow"] / "data/chunk-000/file-000.parquet").names
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
        r = dict(src_rows[SRC][si])
        assert r["episode_index"] == si and r["length"] == n
        for vk in VKEYS:
            assert r[f"videos/{vk}/file_index"] == si and r[f"videos/{vk}/chunk_index"] == 0, (h, se, vk)
            os.link(SRC / f"videos/{vk}/chunk-000/file-{si:03d}.mp4", root / f"videos/{vk}/chunk-000/file-{i:03d}.mp4")
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
        new_eps_meta.append({"episode_index": i, "hand": h, "source_episode": se, "source_dataset": str(SRC),
                             "source_episode_index": si, "length": n, "dataset_from_index": frm})
        all_s.append(s); all_a.append(a)
        frm += n
    pq.write_table(pa.Table.from_pylist(new_rows, schema=schema), root / "meta/episodes/chunk-000/file-000.parquet")
    stats = aggregate_stats(ep_stats)
    write_stats(stats, root)
    shutil.copy(SRCS["shadow"] / "meta/tasks.parquet", root / "meta/tasks.parquet")
    info2 = dict(info)
    info2.update(total_episodes=N_EP, total_frames=frm, splits={"train": f"0:{N_EP}"},
                 robot_type="dexverse_floating_pooled3wsa_right")
    (root / "meta/info.json").write_text(json.dumps(info2, indent=4))
    enc_rep = json.loads((ENC / f"encode_{args.enc}.json").read_text())
    (root / "meta/p3h_pooled.json").write_text(json.dumps(
        {"arm": args.name, "variant": "cv3dp_3hand_nohot", "one_hot": False, "hands": list(HANDS), "state_dim": 41,
         "action_dim": 41, "codec_ckpt": enc_rep["ckpt"], "codec_md5": enc_rep["ckpt_md5"],
         "rows": f"latent_codec_v1/artifacts/codec_v3/cv3dp/encoded_{args.enc}_<hand>.npz (cv3dp_encode.py)",
         "source_dataset": {h: str(p) for h, p in SRCS.items()}, "episodes": new_eps_meta}, indent=1))

    # verification
    S, A = np.concatenate(all_s), np.concatenate(all_a)
    rows_stats = {"observation.state": S, "action": A}
    sv = {k: max(float(np.abs(stats[k]["mean"] - v.mean(0)).max()), float(np.abs(stats[k]["std"] - v.std(0)).max()),
                 float(np.abs(stats[k]["min"] - v.min(0)).max()), float(np.abs(stats[k]["max"] - v.max(0)).max()))
          for k, v in rows_stats.items()}
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f"local/{args.name}", root=root)
    items = {}
    for k in (0, 49, 50, 99, 100, 149):
        m = new_eps_meta[k]
        it = ds[m["dataset_from_index"] + 5]
        h = m["hand"]
        o = offs[h][m["source_episode"]] + 5
        items[k] = {"hand": h, "src_ep": m["source_episode"],
                    "state_eq": bool(np.array_equal(it["observation.state"].numpy(), enc[h]["state_Z"][o])),
                    "action_eq": bool(np.array_equal(it["action"].numpy(), enc[h]["action_Z"][o])),
                    "img": [len(VKEYS)] + list(it[VKEYS[0]].shape)}
    rep = {"root": str(root), "enc": args.enc, "codec_ckpt": enc_rep["ckpt"], "frames": frm, "episodes": N_EP,
           "len": len(ds), "num_episodes": ds.num_episodes, "items": items, "stats_vs_rows_max_abs": sv,
           "aggregate_procedure_check": proc_check,
           "frames_per_hand": {h: int(enc[h]["lengths"].sum()) for h in HANDS}}
    assert len(ds) == frm and ds.num_episodes == N_EP and all(v["state_eq"] and v["action_eq"] for v in items.values())
    assert max(sv.values()) < 1e-4, sv
    (root / "meta/cv3dp_build.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
    print(f"[build] {args.name} DONE", flush=True)


if __name__ == "__main__":
    main()
