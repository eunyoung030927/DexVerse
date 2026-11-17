"""CV3DP amendment A3 (261117): build the D-wrist arms Dw3 / De3 from the already built wsa_Zw3 / wsa_Ze3 sets.
New file; cv3dp_build.py is unchanged and its helpers are imported. Only observation.state / action columns 0-8
(palm pos 3 + rot6d 6) are replaced by EE_common (encoded_dpalm_<hand>.npz, cv3dp_encode_d.py); columns 9-40 (z),
every other parquet column, frames, episode order, tasks and the mp4s (hard links of the Z set's files) are unchanged.
Per-episode stats of observation.state / action are recomputed (lerobot compute_episode_stats), all other per-episode
stats copied from the Z set's meta/episodes; meta/stats.json = aggregate_stats over the 150 per-episode stats.
Verification: z columns bit-identical to the Z set; palm columns == npz rows; LeRobotDataset load, items; stats vs rows.
Container lerobot python, cwd /workspace/xhand/latent_codec_v1:
  python -m xhand_latent.cv3dp_build_d --name wsa_Dw3 --zset wsa_Zw3      (and --name wsa_De3 --zset wsa_Ze3)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.utils import write_stats

from .cv3dp_build import DST, ENC, HANDS, N_EP, VKEYS, ep_stats_from_row

RECOMP = ("observation.state", "action")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--zset", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    Z = DST / args.zset
    info = json.loads((Z / "meta/info.json").read_text())
    feats = info["features"]
    stat_keys = list(feats)
    zmeta = json.loads((Z / "meta/p3h_pooled.json").read_text())
    zeps = pq.read_table(Z / "meta/episodes/chunk-000/file-000.parquet")
    schema, zrows = zeps.schema, zeps.to_pylist()
    assert len(zrows) == N_EP == len(zmeta["episodes"])
    dp = {h: np.load(ENC / f"encoded_dpalm_{h}.npz") for h in HANDS}
    offs = {h: np.r_[0, np.cumsum(dp[h]["lengths"])] for h in HANDS}

    root = DST / args.name
    if root.exists():
        if not args.force:
            raise SystemExit(f"{root} exists (use --force)")
        shutil.rmtree(root)
    (root / "data/chunk-000").mkdir(parents=True)
    for vk in VKEYS:
        (root / f"videos/{vk}/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)

    new_rows, ep_stats, all_s, all_a, z_eq = [], [], [], [], True
    for i, m in enumerate(zmeta["episodes"]):
        h, se = m["hand"], m["source_episode"]
        assert m["episode_index"] == i
        t = pq.read_table(Z / f"data/chunk-000/file-{i:03d}.parquet")
        n = t.num_rows
        assert n == m["length"] == int(dp[h]["lengths"][se])
        zs = np.array(t.column("observation.state").to_pylist(), np.float32)
        za = np.array(t.column("action").to_pylist(), np.float32)
        s, a = zs.copy(), za.copy()
        s[:, :9] = dp[h]["state_D9"][offs[h][se]:offs[h][se + 1]]
        a[:, :9] = dp[h]["action_D9"][offs[h][se]:offs[h][se + 1]]
        cols = {}
        for name in t.schema.names:
            if name == "observation.state":
                cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(s.reshape(-1), pa.float32()), 41)
            elif name == "action":
                cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), pa.float32()), 41)
            else:
                cols[name] = t.column(name)
        nt = pa.Table.from_arrays([cols[c] for c in t.schema.names], schema=t.schema)
        pq.write_table(nt, root / f"data/chunk-000/file-{i:03d}.parquet")
        # read back: z columns bit-identical, palm == npz
        bt = pq.read_table(root / f"data/chunk-000/file-{i:03d}.parquet")
        bs = np.array(bt.column("observation.state").to_pylist(), np.float32)
        ba = np.array(bt.column("action").to_pylist(), np.float32)
        z_eq &= bool(np.array_equal(bs[:, 9:], zs[:, 9:]) and np.array_equal(ba[:, 9:], za[:, 9:])
                     and np.array_equal(bs, s) and np.array_equal(ba, a))
        for name in t.schema.names:
            if name not in RECOMP:
                assert bt.column(name).equals(t.column(name)), name
        for vk in VKEYS:
            os.link(Z / f"videos/{vk}/chunk-000/file-{i:03d}.mp4", root / f"videos/{vk}/chunk-000/file-{i:03d}.mp4")
        r = dict(zrows[i])
        assert r["episode_index"] == i and r["length"] == n
        st = ep_stats_from_row(r, stat_keys)
        new = compute_episode_stats({"observation.state": s, "action": a}, {k: feats[k] for k in RECOMP})
        for k in RECOMP:
            st[k] = {}
            for sname, v in new[k].items():
                v = np.asarray(v).reshape(np.asarray(r[f"stats/{k}/{sname}"]).shape)
                st[k][sname] = v
                r[f"stats/{k}/{sname}"] = v.tolist()
        new_rows.append(r)
        ep_stats.append(st)
        all_s.append(s); all_a.append(a)
    assert z_eq, "z columns differ from the Z set"
    pq.write_table(pa.Table.from_pylist(new_rows, schema=schema), root / "meta/episodes/chunk-000/file-000.parquet")
    stats = aggregate_stats(ep_stats)
    write_stats(stats, root)
    shutil.copy(Z / "meta/tasks.parquet", root / "meta/tasks.parquet")
    shutil.copy(Z / "meta/info.json", root / "meta/info.json")
    enc_rep = json.loads((ENC / "encode_dpalm.json").read_text())
    pm = dict(zmeta)
    pm.update(arm=args.name, variant="cv3dp_3hand_nohot_D", wrist="EE_common = canonical palm @ inv(f_task(q_finger))",
              f_task=enc_rep["f_task"], f_task_md5=enc_rep["f_task_md5"], z_from=str(Z),
              palm_rows="latent_codec_v1/artifacts/codec_v3/cv3dp/encoded_dpalm_<hand>.npz (cv3dp_encode_d.py)")
    (root / "meta/p3h_pooled.json").write_text(json.dumps(pm, indent=1))

    S, A = np.concatenate(all_s), np.concatenate(all_a)
    sv = {k: max(float(np.abs(stats[k]["mean"] - v.mean(0)).max()), float(np.abs(stats[k]["std"] - v.std(0)).max()),
                 float(np.abs(stats[k]["min"] - v.min(0)).max()), float(np.abs(stats[k]["max"] - v.max(0)).max()))
          for k, v in {"observation.state": S, "action": A}.items()}
    zstats = json.loads((Z / "meta/stats.json").read_text())
    other_eq = {k: all(np.allclose(np.asarray(stats[k][s_]), np.asarray(zstats[k][s_])) for s_ in zstats[k])
                for k in zstats if k not in RECOMP}
    z_dims_stats = max(float(np.abs(np.asarray(stats[k]["mean"])[9:] - np.asarray(zstats[k]["mean"])[9:]).max()) for k in RECOMP)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f"local/{args.name}", root=root)
    items = {}
    for k in (0, 49, 50, 99, 100, 149):
        m = zmeta["episodes"][k]
        it = ds[m["dataset_from_index"] + 5]
        h, o = m["hand"], offs[m["hand"]][m["source_episode"]] + 5
        items[k] = {"hand": h, "src_ep": m["source_episode"],
                    "palm_state_eq": bool(np.array_equal(it["observation.state"].numpy()[:9], dp[h]["state_D9"][o])),
                    "palm_action_eq": bool(np.array_equal(it["action"].numpy()[:9], dp[h]["action_D9"][o])),
                    "img": [len(VKEYS)] + list(it[VKEYS[0]].shape)}
    rep = {"root": str(root), "zset": str(Z), "frames": int(len(S)), "episodes": N_EP, "len": len(ds),
           "num_episodes": ds.num_episodes, "z_cols_bit_identical_to_zset": z_eq, "items": items,
           "stats_vs_rows_max_abs": sv, "non_state_action_stats_equal_zset": other_eq,
           "z_dims_mean_stat_vs_zset_max_abs": z_dims_stats,
           "frames_per_hand": {h: int(dp[h]["lengths"].sum()) for h in HANDS},
           "dpalm_md5": {h: hashlib.md5((ENC / f"encoded_dpalm_{h}.npz").read_bytes()).hexdigest() for h in HANDS}}
    assert len(ds) == len(S) == info["total_frames"] and ds.num_episodes == N_EP
    assert all(v["palm_state_eq"] and v["palm_action_eq"] for v in items.values())
    assert max(sv.values()) < 1e-4 and all(other_eq.values()), (sv, other_eq)
    (root / "meta/cv3dp_build.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
    print(f"[build_d] {args.name} DONE", flush=True)


if __name__ == "__main__":
    main()
