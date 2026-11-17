"""handmask (261115): masked copies of the cv2dp 2-hand (shadow + wuji) GraspKettle LeRobot sets.
New file; reads cv2dp datasets read-only, writes only under /workspace/xhand/datasets/handmask.
  --stage videos : for every (hand, src ep) of the chosen views: decode the episode mp4 that the cv2dp set links
                   (= pooled3hand_nohot/kettle3_Z file-<k3>.mp4, all frames), set hand pixels (state-replay segmentation
                   masks, artifacts/handmask/masks/<hand>/ep<e>.npz, optional square dilation) to 0, re-encode with the
                   LeRobot encoder settings (libsvtav1, yuv420p, g=2, crf=30, preset 12, 60 fps) ->
                   handmask/videos/<view>/<hand>_ep<e>.mp4 + per-episode image stats (LeRobot sample_images rule).
  --stage assemble --src <cv2dp root> --name <dst>: dst = src with the masked views' videos replaced (hard links),
                   data/ and other videos hard-linked, meta copied with the masked views' per-episode and aggregate
                   image stats replaced (lerobot aggregate_stats; procedure checked to reproduce src stats.json first).
                   Verification: LeRobotDataset load, state/action equal to src, masked pixels ~0 at mask positions.
lerobot python in the xhand container, cwd /workspace/xhand/latent_codec_v1:
  python -m xhand_latent.hm_build_masked --stage videos --views eye_in_hand,third_person --dilate 2
  python -m xhand_latent.hm_build_masked --stage assemble --src /workspace/xhand/datasets/cv2dp/ws_Zv2 --name ws_Zv2m
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.datasets.compute_stats import aggregate_stats, auto_downsample_height_width, get_feature_stats, sample_indices
from lerobot.datasets.utils import write_stats

LC = Path("/workspace/xhand/latent_codec_v1")
K3 = Path("/workspace/xhand/datasets/pooled3hand_nohot/kettle3_Z")
OUT = Path("/workspace/xhand/datasets/handmask")
MASKS = LC / "artifacts/handmask/masks"
VIEW_KEY = {"eye_in_hand": "observation.images.eye_in_hand", "third_person": "observation.images.third_person"}
CAM_TAG = {"eye_in_hand": "eih", "third_person": "tp"}
HANDS = ("shadow", "wuji")
STATS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def decode_all(path):
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])  # (T,H,W,3) uint8


def encode(frames, path, fps=60):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    with av.open(str(tmp), "w") as out:
        st = out.add_stream("libsvtav1", fps, options={"g": "2", "crf": "30", "preset": "12"})
        st.pix_fmt, st.width, st.height = "yuv420p", frames.shape[2], frames.shape[1]
        for fr in frames:
            for p in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                out.mux(p)
        for p in st.encode():
            out.mux(p)
    os.replace(tmp, path)


def image_stats(frames):
    """= lerobot compute_episode_stats for a video feature (sample_images on the frames instead of PNG paths)."""
    idx = sample_indices(len(frames))
    arr = np.stack([auto_downsample_height_width(frames[i].transpose(2, 0, 1)) for i in idx])
    st = get_feature_stats(arr, axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in st.items()}


def dilate(m, px):
    if px <= 0:
        return m
    t = torch.from_numpy(m).float()[:, None]
    return (torch.nn.functional.max_pool2d(t, 2 * px + 1, 1, px)[:, 0] > 0.5).numpy()


def stage_videos(a):
    pooled = json.loads((K3 / "meta/p3h_pooled.json").read_text())
    k3 = {(e["hand"], e["source_episode"]): e for e in pooled["episodes"]}
    hands = a.hands.split(",")
    rep_p = OUT / "videos/build.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    for view in a.views.split(","):
        key, tag = VIEW_KEY[view], CAM_TAG[view]
        for h in hands:
            for e in range(50):
                dst = OUT / f"videos/{view}/{h}_ep{e:03d}.mp4"
                name = f"{view}/{h}_ep{e:03d}"
                if dst.exists() and name in rep and not a.force:
                    continue
                ke = k3[(h, e)]
                frames = decode_all(K3 / f"videos/{key}/chunk-000/file-{ke['episode_index']:03d}.mp4")
                z = np.load(MASKS / f"{h}/ep{e:03d}.npz")
                T = int(z["T"])
                assert len(frames) == T == ke["length"], (name, len(frames), T, ke["length"])
                m = dilate(np.unpackbits(z[f"masks_{tag}"], axis=-1, count=640).astype(bool), a.dilate)
                frames[m] = 0
                encode(frames, dst)
                chk = decode_all(dst)
                assert len(chk) == T, (name, len(chk))
                st = image_stats(frames)
                rep[name] = {"T": T, "area": float(m.mean()), "dilate": a.dilate, "k3_episode": ke["episode_index"],
                             "masked_px_mean_after_encode": float(chk[m].mean()) if m.any() else 0.0,
                             "stats": {k: np.asarray(v).tolist() for k, v in st.items()}}
                rep_p.parent.mkdir(parents=True, exist_ok=True)
                rep_p.write_text(json.dumps(rep))
                print(f"[videos] {name} T={T} area={rep[name]['area']:.3f} masked_px_after={rep[name]['masked_px_mean_after_encode']:.2f}", flush=True)


def ep_stats_from_row(row, keys):
    return {k: {s: np.asarray(row[f"stats/{k}/{s}"], dtype=np.float64) for s in STATS} for k in keys}


def max_abs_stats(a, b):
    return max(float(np.abs(np.asarray(a[k][s], np.float64) - np.asarray(b[k][s], np.float64)).max()) for k in a for s in a[k])


def stage_assemble(a):
    src, dst = Path(a.src), OUT / a.name
    views = a.views.split(",")
    vrep = json.loads((OUT / "videos/build.json").read_text())
    info = json.loads((src / "meta/info.json").read_text())
    feats = info["features"]
    stat_keys = list(feats)
    eps_t = pq.read_table(src / "meta/episodes/chunk-000/file-000.parquet")
    rows = eps_t.to_pylist()
    pooled = json.loads((src / "meta/p3h_pooled.json").read_text())
    src_stats = json.loads((src / "meta/stats.json").read_text())
    agg0 = aggregate_stats([ep_stats_from_row(r, stat_keys) for r in rows])
    proc = max_abs_stats({k: agg0[k] for k in src_stats}, src_stats)
    print(f"[assemble] aggregate procedure vs src stats.json max abs {proc:.3e}", flush=True)
    assert proc < 1e-6, proc
    if dst.exists():
        if not a.force:
            raise SystemExit(f"{dst} exists (use --force)")
        shutil.rmtree(dst)
    (dst / "meta/episodes/chunk-000").mkdir(parents=True)
    (dst / "data/chunk-000").mkdir(parents=True)
    for f in sorted((src / "data/chunk-000").glob("*.parquet")):
        os.link(f, dst / "data/chunk-000" / f.name)
    keys_masked = {VIEW_KEY[v] for v in views}
    for vk in (k for k in feats if feats[k]["dtype"] == "video"):
        (dst / f"videos/{vk}/chunk-000").mkdir(parents=True)
        for f in sorted((src / f"videos/{vk}/chunk-000").glob("*.mp4")):
            os.link(f, dst / f"videos/{vk}/chunk-000" / f.name) if vk not in keys_masked else None
    ep_stats = []
    for r, pe in zip(rows, pooled["episodes"]):
        i = r["episode_index"]
        assert pe["episode_index"] == i
        st = ep_stats_from_row(r, stat_keys)
        for v in views:
            vk = VIEW_KEY[v]
            name = f"{v}/{pe['hand']}_ep{pe['source_episode']:03d}"
            assert vrep[name]["T"] == r["length"], name
            assert r[f"videos/{vk}/from_timestamp"] == 0.0
            os.link(OUT / f"videos/{name}.mp4", dst / f"videos/{vk}/chunk-000/file-{r[f'videos/{vk}/file_index']:03d}.mp4")
            for s in STATS:
                val = np.asarray(vrep[name]["stats"][s]).reshape(np.asarray(r[f"stats/{vk}/{s}"]).shape)
                st[vk][s] = val
                r[f"stats/{vk}/{s}"] = val.tolist()
        ep_stats.append(st)
    pq.write_table(pa.Table.from_pylist(rows, schema=eps_t.schema), dst / "meta/episodes/chunk-000/file-000.parquet")
    stats = aggregate_stats(ep_stats)
    write_stats(stats, dst)
    shutil.copy(src / "meta/tasks.parquet", dst / "meta/tasks.parquet")
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    pooled2 = dict(pooled)
    pooled2.update(handmask={"views": views, "dilate_px": sorted({vrep[f'{v}/{e["hand"]}_ep{e["source_episode"]:03d}']['dilate'] for v in views for e in pooled["episodes"]}),
                             "src_dataset": str(src), "masks": str(MASKS), "method": "Isaac instance_id_segmentation_fast of Robot prims, state replay"})
    (dst / "meta/p3h_pooled.json").write_text(json.dumps(pooled2, indent=1))
    for extra in ("cv2dp_build.json",):
        if (src / "meta" / extra).exists():
            shutil.copy(src / "meta" / extra, dst / "meta" / f"src_{extra}")

    # verification
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds_s = LeRobotDataset("local/src", root=src, video_backend="pyav")
    ds_d = LeRobotDataset(f"local/{a.name}", root=dst, video_backend="pyav")
    assert len(ds_s) == len(ds_d) and ds_s.num_episodes == ds_d.num_episodes
    chk = []
    for i in (0, 25, 49, 50, 75, 99):
        pe = pooled["episodes"][i]
        k = 37 % pe["length"]
        x, y = ds_s[pe["dataset_from_index"] + k], ds_d[pe["dataset_from_index"] + k]
        rec = {"ep": i, "hand": pe["hand"], "state_eq": bool(torch.equal(x["observation.state"], y["observation.state"])),
               "action_eq": bool(torch.equal(x["action"], y["action"]))}
        z = np.load(MASKS / f"{pe['hand']}/ep{pe['source_episode']:03d}.npz")
        for v in ("eye_in_hand", "third_person"):
            vk = VIEW_KEY[v]
            m = np.unpackbits(z[f"masks_{CAM_TAG[v]}"][k], axis=-1, count=640).astype(bool)
            im_s, im_d = x[vk].permute(1, 2, 0).numpy() * 255, y[vk].permute(1, 2, 0).numpy() * 255
            rec[v] = {"masked": vk in keys_masked, "src_in_mask": float(im_s[m].mean()) if m.any() else None,
                      "dst_in_mask": float(im_d[m].mean()) if m.any() else None,
                      "outside_mad": float(np.abs(im_s[~m] - im_d[~m]).mean())}
        chk.append(rec)
        print("[verify]", json.dumps(rec), flush=True)
    assert all(c["state_eq"] and c["action_eq"] for c in chk)
    rep = {"root": str(dst), "src": str(src), "views": views, "episodes": ds_d.num_episodes, "frames": len(ds_d),
           "aggregate_procedure_check": proc, "verify": chk,
           "image_stats": {VIEW_KEY[v]: {"src_mean": src_stats[VIEW_KEY[v]]["mean"], "dst_mean": np.asarray(stats[VIEW_KEY[v]]["mean"]).tolist()} for v in views}}
    (dst / "meta/handmask_build.json").write_text(json.dumps(rep, indent=1))
    print("[assemble] done", dst, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("videos", "assemble"))
    ap.add_argument("--views", default="eye_in_hand,third_person")
    ap.add_argument("--hands", default=",".join(HANDS))
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--src")
    ap.add_argument("--name")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    stage_videos(a) if a.stage == "videos" else stage_assemble(a)
