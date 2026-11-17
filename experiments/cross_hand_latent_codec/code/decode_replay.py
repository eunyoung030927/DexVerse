"""Decode held-out task actions offline; fingers use the learned decoder only.

Same-hand replay preserves recorded wrist commands. Cross-hand replay aligns
the source/target weighted pinch centre and palm orientation by moving only
the six floating wrist joints, following the official latent replay example.
No finger IK or target task supervision is used for decoding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import TASKS, feature_names, infer_recorded_offset, load_trajectory, trajectory_path
from .evaluate import load_codecs
from .kinematics import FINGERS, palm_transform
from .reference_kin import make_T, so3_log


def canonical_world(fk, q, root):
    poses = fk.reference.fk(q, root_T=root)
    return poses[fk.spec["palm_body"]] @ palm_transform(fk.spec)


def wrist_solve(fk, desired, root, q):
    from .wrist import floating_wrist
    target_body = desired @ np.linalg.inv(palm_transform(fk.spec))
    return floating_wrist(fk.reference, fk.spec["palm_body"], fk.spec["wrist_joints"], target_body, root, q)

def convert(source, target, q_source, source_root, target_root, codecs, fks):
    sf, tf = fks[source], fks[target]
    q_finger = q_source[:, [sf.reference.qidx[n] for n in sf.joint_names]]
    with torch.no_grad():
        normalized = sf.normalize(torch.tensor(q_finger, dtype=torch.float32)).clamp(-1, 1)
        z = codecs[source].encode(normalized)[0]
        decoded = codecs[target].decoder(z)
        predicted = tf.denormalize(decoded).numpy()
    q_target = np.zeros((len(q_source), len(tf.reference.joint_order)))
    for i, n in enumerate(tf.joint_names):
        q_target[:, tf.reference.qidx[n]] = predicted[:, i]
    if source == target:
        for n in sf.spec["wrist_joints"]:
            q_target[:, tf.reference.qidx[n]] = q_source[:, sf.reference.qidx[n]]
        return q_target, {"wrist_alignment": "recorded wrist unchanged"}
    # Copy source wrist angles as an initial seed; physical wrist poses are
    # then aligned using the exact per-hand USD transforms.
    for n in tf.spec["wrist_joints"]:
        if n in sf.reference.qidx:
            q_target[:, tf.reference.qidx[n]] = q_source[:, sf.reference.qidx[n]]
    desired = canonical_world(sf, q_source, source_root)
    with torch.no_grad():
        source_points = sf(normalized).numpy()
        target_points = tf(decoded).numpy()
    valid = (sf.valid & tf.valid).numpy()
    indices = np.flatnonzero(valid)[1:]
    weights = np.exp(-12 * np.linalg.norm(source_points[:, :1] - source_points[:, indices], axis=-1))
    weights /= weights.sum(axis=-1, keepdims=True)
    source_centre = (((source_points[:, :1] + source_points[:, indices]) * .5) * weights[..., None]).sum(axis=1)
    target_centre = (((target_points[:, :1] + target_points[:, indices]) * .5) * weights[..., None]).sum(axis=1)
    desired[:, :3, 3] += np.einsum("bij,bj->bi", desired[:, :3, :3], source_centre - target_centre)
    q_target, errors = wrist_solve(tf, desired, target_root, q_target)
    return q_target, {"wrist_alignment": "FK pinch-centre + palm orientation, floating wrist only", **errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", default="metadata")
    parser.add_argument("--data", default="data/replay")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(4)
    codecs, fks, payload = load_codecs(args.checkpoint, args.metadata)
    out = Path(args.output)
    manifest = {"checkpoint": args.checkpoint, "step": payload["step"], "jobs": [],
                "initial_state_protocol_version":2,
                "initial_state_protocol":"same-hand native state preserved; cross-hand state mapped through codec",
                "raw_control": "native physical joint targets; recorded floating-wrist offset calibrated before mapping to the live controller"}
    for task in TASKS:
        for source in codecs:
            sf = fks[source]
            action_names, _ = feature_names(args.data, task, source)
            meta = sf.spec.get("robotmeta")
            default = np.array(meta["action_offset"]) if meta else np.array([.3 if n == "z_translation_joint" else 0 for n in action_names])
            if meta:
                default = default[[meta["action_joint_names"].index(n) for n in action_names]]
            offset, offset_stats = infer_recorded_offset(args.data, task, source, action_names, default, args.episodes,
                                                        wrist_names=sf.spec["wrist_joints"])
            for target in codecs:
                tf = fks[target]
                result = {"joint_names": np.array(tf.reference.joint_order)}
                native = {"joint_names": np.array(sf.reference.joint_order)}
                diagnostics = []
                for episode in range(args.episodes):
                    data = load_trajectory(trajectory_path(args.data, task, source, episode))
                    ep = data["episodes"][0]
                    recorded_names = data["action_layout"]["robot_joint_names"]
                    a = np.asarray(ep["actions"], dtype=np.float64)
                    target_joints = a + offset
                    q = np.zeros((len(a), len(sf.reference.joint_order)))
                    for i, n in enumerate(action_names):
                        q[:, sf.reference.qidx[n]] = target_joints[:, i]
                    clipped = np.clip(q, sf.reference.lower, sf.reference.upper)
                    source_root_pose = np.asarray(ep["initial_state"]["articulation"]["robot"]["root_pose"])[0]
                    source_root = make_T(source_root_pose[:3], source_root_pose[3:])
                    native_target = load_trajectory(trajectory_path(args.data, task, target, episode))["episodes"][0]
                    target_root_pose = np.asarray(native_target["initial_state"]["articulation"]["robot"]["root_pose"])[0]
                    target_root = make_T(target_root_pose[:3], target_root_pose[3:])
                    init = np.asarray(ep["initial_state"]["articulation"]["robot"]["joint_position"])[0]
                    initial = init[[recorded_names.index(n) for n in sf.reference.joint_order]][None]
                    all_q = np.concatenate([initial, clipped])
                    decoded, stats = convert(source, target, all_q, source_root, target_root, codecs, fks)
                    key = f"ep{episode:03d}"
                    result[key] = decoded[1:].astype(np.float32)
                    result[f"init_{key}"] = (initial[0] if source == target else decoded[0]).astype(np.float32)
                    result[f"root_{key}"] = target_root_pose.astype(np.float32)
                    native[key] = q.astype(np.float32)
                    native[f"init_{key}"] = initial[0].astype(np.float32)
                    native[f"root_{key}"] = source_root_pose.astype(np.float32)
                    diagnostics.append({"episode": episode, "command_clamp_fraction": float(np.mean(np.abs(q-clipped)>1e-6)), **stats})
                directory = out / task / target
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / f"{source}_codec.npz", **result)
                if source == target:
                    np.savez_compressed(directory / f"{source}_native.npz", **native)
                manifest["jobs"].append({"task": task, "source": source, "target": target,
                                         "offset_diagnostics": offset_stats, "episodes": diagnostics})
                print(json.dumps({"decoded": task, "source": source, "target": target}), flush=True)
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
