"""CV3HM (261117): build the 3-hand (shadow + wuji + allegro) nohot FunctionalHammerStrike LeRobot v3.0 sets for the
codec-v3 DP test (held-out test hands sharpa, leap). Adapted from cv3dp_build.py (unchanged original). Difference: there is
no earlier pooled Hammer set, so the sources are the per-hand dv-demos-v0 teleop LeRobot sets
(/workspace/xhand/data/dv-demos-v0/v0/teleop_lerobot/functional/Dexverse-FunctionalHammerStrike-v0/floating_<hand>_right),
and the output LAYOUT is copied from the Kettle pooled set pooled3hand_nohot/kettle3_Z (the layout cv3dp trained on):
  - data parquet: kettle3_Z column set/schema (observation.state, action, timestamp, frame_index, episode_index, index,
    task_index); state/action = encoded 41D rows (artifacts/codec_v3/cv3hm/encoded_<enc>_<hand>.npz); all other
    per-hand columns (joint_vel, ee*, tactile, touch, wrench, ...) are dropped, as in kettle3_Z.
  - meta/episodes: kettle3_Z schema; source non-stats values re-indexed; image/timestamp/frame_index/task_index stats
    copied from the source row; state/action/episode_index/index stats recomputed (compute_episode_stats); every stats
    value reshaped to the kettle3_Z nesting.
  - meta/info.json: kettle3_Z info with Hammer totals/robot_type (video feature info asserted equal to the Hammer sources);
    meta/tasks.parquet: Hammer source ('Functional hammer strike', identical in the 3 sources); meta/stats.json =
    lerobot aggregate_stats over the 150 per-episode stats.
  - videos: hard links of the source per-episode mp4s (one file per episode, from_timestamp 0, asserted).
New eps 0-49 = shadow src 0-49, 50-99 = wuji src 0-49, 100-149 = allegro src 0-49.
Checks: palm columns (0-8) of encoded_e2 and encoded_v3wbcll are bit-identical (codec-independent); LeRobotDataset load,
items 0/49/50/99/100/149 state/action == npz rows, image shape; stats vs rows.
Container lerobot python, cwd /workspace/xhand/latent_codec_v1:
  python -m xhand_latent.cv3hm_build --name wsa_Ze3h --enc e2      (and --name wsa_Zw3h --enc v3wbcll)
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

TASK = "Dexverse-FunctionalHammerStrike-v0"
SRC = Path("/workspace/xhand/data/dv-demos-v0/v0/teleop_lerobot/functional") / TASK
TEMPLATE = Path("/workspace/xhand/datasets/pooled3hand_nohot/kettle3_Z")
DST = Path("/workspace/xhand/datasets/cv3hm")
ENC = Path("artifacts/codec_v3/cv3hm")
HANDS = ("shadow", "wuji", "allegro")
VKEYS = ("observation.images.third_person", "observation.images.eye_in_hand")
RECOMPUTED = ("observation.state", "action", "episode_index", "index")
SNAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
N_EP = 50 * len(HANDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--enc", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    tinfo = json.loads((TEMPLATE / "meta/info.json").read_text())
    feats = tinfo["features"]
    stat_keys = list(feats)
    tschema = pq.read_schema(TEMPLATE / "data/chunk-000/file-000.parquet")
    tep = pq.read_table(TEMPLATE / "meta/episodes/chunk-000/file-000.parquet")
    eschema = tep.schema
    trow = tep.slice(0, 1).to_pylist()[0]
    tshape = {c: np.asarray(trow[c]).shape for c in eschema.names if c.startswith("stats/")}
    src = {h: SRC / f"floating_{h}_right" for h in HANDS}
    sinfo = {h: json.loads((src[h] / "meta/info.json").read_text()) for h in HANDS}
    for h in HANDS:
        assert sinfo[h]["fps"] == tinfo["fps"] and sinfo[h]["total_episodes"] == 50, h
        for vk in VKEYS:
            assert sinfo[h]["features"][vk] == feats[vk], (h, vk)
    stasks = {h: pq.read_table(src[h] / "meta/tasks.parquet") for h in HANDS}
    assert all(t.equals(stasks["shadow"]) for t in stasks.values()), "source tasks differ"
    # the per-hand sources keep one meta/episodes parquet per episode (file-000..049, one row each)
    srows = {h: [r for e in range(50) for r in pq.read_table(src[h] / f"meta/episodes/chunk-000/file-{e:03d}.parquet").to_pylist()]
             for h in HANDS}
    assert all(len(v) == 50 for v in srows.values()), {h: len(v) for h, v in srows.items()}

    enc = {h: np.load(ENC / f"encoded_{args.enc}_{h}.npz") for h in HANDS}
    other = "v3wbcll" if args.enc == "e2" else "e2"
    palm_cross = {}
    for h in HANDS:
        o = np.load(ENC / f"encoded_{other}_{h}.npz")
        palm_cross[h] = bool(np.array_equal(o["state_Z"][:, :9], enc[h]["state_Z"][:, :9])
                             and np.array_equal(o["action_Z"][:, :9], enc[h]["action_Z"][:, :9])
                             and np.array_equal(o["lengths"], enc[h]["lengths"]))
    assert all(palm_cross.values()), f"palm columns differ between codecs: {palm_cross}"
    offs = {h: np.r_[0, np.cumsum(enc[h]["lengths"])] for h in HANDS}

    root = DST / args.name
    if root.exists():
        if not args.force:
            raise SystemExit(f"{root} exists (use --force)")
        shutil.rmtree(root)
    (root / "data/chunk-000").mkdir(parents=True)
    for vk in VKEYS:
        (root / f"videos/{vk}/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)

    new_rows, new_eps_meta, ep_stats, frm, all_s, all_a = [], [], [], 0, [], []
    i = 0
    for h in HANDS:
        for se in range(50):
            sr = srows[h][se]
            assert sr["episode_index"] == se and sr["data/file_index"] == se and sr["data/chunk_index"] == 0
            t = pq.read_table(src[h] / f"data/chunk-000/file-{se:03d}.parquet")
            n = t.num_rows
            assert n == sr["length"] == int(enc[h]["lengths"][se]), (h, se, n)
            assert set(t.column("episode_index").to_pylist()) == {se}
            s = enc[h]["state_Z"][offs[h][se]:offs[h][se + 1]].astype(np.float32)
            a = enc[h]["action_Z"][offs[h][se]:offs[h][se + 1]].astype(np.float32)
            cols = {}
            for name in tschema.names:
                if name == "observation.state":
                    cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(s.reshape(-1), pa.float32()), 41)
                elif name == "action":
                    cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), pa.float32()), 41)
                elif name == "episode_index":
                    cols[name] = pa.array(np.full(n, i, np.int64))
                elif name == "index":
                    cols[name] = pa.array(np.arange(frm, frm + n, dtype=np.int64))
                else:
                    cols[name] = t.column(name).cast(tschema.field(name).type)
            pq.write_table(pa.Table.from_arrays([cols[c] for c in tschema.names], schema=tschema),
                           root / f"data/chunk-000/file-{i:03d}.parquet")
            for vk in VKEYS:
                assert sr[f"videos/{vk}/file_index"] == se and sr[f"videos/{vk}/chunk_index"] == 0, (h, se, vk)
                assert sr[f"videos/{vk}/from_timestamp"] == 0.0, (h, se, vk)
                assert abs(sr[f"videos/{vk}/to_timestamp"] - n / tinfo["fps"]) < 1e-6, (h, se, vk)
                os.link(src[h] / f"videos/{vk}/chunk-000/file-{se:03d}.mp4",
                        root / f"videos/{vk}/chunk-000/file-{i:03d}.mp4")
            new = compute_episode_stats({"observation.state": s, "action": a, "episode_index": np.full(n, i, np.int64),
                                         "index": np.arange(frm, frm + n, dtype=np.int64)},
                                        {k: feats[k] for k in RECOMPUTED})
            st, r = {}, {}
            for c in eschema.names:
                if not c.startswith("stats/"):
                    r[c] = sr[c]
            for k in stat_keys:
                st[k] = {}
                for sn in SNAMES:
                    c = f"stats/{k}/{sn}"
                    v = np.asarray(new[k][sn] if k in RECOMPUTED else sr[c])
                    v = v.reshape(tshape[c])
                    st[k][sn] = v.astype(np.int64) if sn == "count" else v
                    r[c] = v.tolist()
            r["episode_index"] = i
            r["data/file_index"] = i
            r["dataset_from_index"], r["dataset_to_index"] = frm, frm + n
            for vk in VKEYS:
                r[f"videos/{vk}/file_index"] = i
            r["meta/episodes/chunk_index"], r["meta/episodes/file_index"] = 0, 0
            new_rows.append(r)
            ep_stats.append(st)
            new_eps_meta.append({"episode_index": i, "hand": h, "source_episode": se, "source_dataset": str(src[h]),
                                 "source_episode_index": se, "length": n, "dataset_from_index": frm})
            all_s.append(s); all_a.append(a)
            frm += n
            i += 1
    assert i == N_EP
    pq.write_table(pa.Table.from_pylist(new_rows, schema=eschema), root / "meta/episodes/chunk-000/file-000.parquet")
    stats = aggregate_stats(ep_stats)
    write_stats(stats, root)
    shutil.copy(src["shadow"] / "meta/tasks.parquet", root / "meta/tasks.parquet")
    info2 = dict(tinfo)
    info2.update(total_episodes=N_EP, total_frames=frm, splits={"train": f"0:{N_EP}"},
                 robot_type="dexverse_floating_pooled3wsa_right")
    (root / "meta/info.json").write_text(json.dumps(info2, indent=4))
    enc_rep = json.loads((ENC / f"encode_{args.enc}.json").read_text())
    (root / "meta/p3h_pooled.json").write_text(json.dumps(
        {"arm": args.name, "variant": "cv3hm_3hand_nohot", "task": TASK, "one_hot": False, "hands": list(HANDS),
         "state_dim": 41, "action_dim": 41, "codec_ckpt": enc_rep["ckpt"], "codec_md5": enc_rep["ckpt_md5"],
         "rows": f"latent_codec_v1/artifacts/codec_v3/cv3hm/encoded_{args.enc}_<hand>.npz (cv3hm_encode.py)",
         "layout_template": str(TEMPLATE), "source_dataset": {h: str(p) for h, p in src.items()},
         "episodes": new_eps_meta}, indent=1))

    S, A = np.concatenate(all_s), np.concatenate(all_a)
    sv = {k: max(float(np.abs(stats[k]["mean"] - v.mean(0)).max()), float(np.abs(stats[k]["std"] - v.std(0)).max()),
                 float(np.abs(stats[k]["min"] - v.min(0)).max()), float(np.abs(stats[k]["max"] - v.max(0)).max()))
          for k, v in {"observation.state": S, "action": A}.items()}
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
    rep = {"root": str(root), "task": TASK, "enc": args.enc, "codec_ckpt": enc_rep["ckpt"], "frames": frm,
           "episodes": N_EP, "len": len(ds), "num_episodes": ds.num_episodes, "items": items,
           "stats_vs_rows_max_abs": sv, "palm_cols_equal_across_codecs": palm_cross,
           "frames_per_hand": {h: int(enc[h]["lengths"].sum()) for h in HANDS}}
    assert len(ds) == frm and ds.num_episodes == N_EP and all(v["state_eq"] and v["action_eq"] for v in items.values())
    assert max(sv.values()) < 1e-4, sv
    (root / "meta/cv3hm_build.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
    print(f"[build] {args.name} DONE", flush=True)


if __name__ == "__main__":
    main()
