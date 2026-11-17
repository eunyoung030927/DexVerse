"""Restricted trajectory loading and explicit recorded action/joint mappings."""
from __future__ import annotations

import io
import json
from pathlib import Path
import pickle

import numpy as np

TASKS = {
    "Dexverse-GraspKettle-v0": "functional",
    "Dexverse-GraspPan-v0": "functional",
    "Dexverse-GraspBleach-v0": "functional",
    "Dexverse-GraspTwoItems-v0": "grasping",
    "Dexverse-FunctionalHammerStrike-v0": "functional",
}
BIMANUAL_TASKS = {"Dexverse-BimanualLiftCarton-v0":"bimanual","Dexverse-BimanualLiftTray-v0":"bimanual"}

_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"), ("builtins", "slice")}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"Blocked global {module}.{name}")
        if module.startswith("numpy._core") and not hasattr(np, "_core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_trajectory(path):
    return RestrictedUnpickler(io.BytesIO(Path(path).read_bytes())).load()


def trajectory_path(root, task, hand, episode):
    category = {**TASKS,**BIMANUAL_TASKS}[task]
    side = "bimanual" if task in BIMANUAL_TASKS else "right"
    return Path(root) / "v0/teleop" / category / task / f"floating_{hand}_{side}" / f"ep{episode:03d}.pkl"


def feature_names(root, task, hand):
    category = {**TASKS,**BIMANUAL_TASKS}[task]
    side = "bimanual" if task in BIMANUAL_TASKS else "right"
    p = Path(root) / "v0/teleop_lerobot" / category / task / f"floating_{hand}_{side}/meta/info.json"
    features = json.loads(p.read_text())["features"]
    return list(features["action"]["names"]), list(features["observation.state"]["names"])


def infer_recorded_offset(root, task, hand, action_names, default_offset, episodes=10, wrist_names=None):
    samples = []
    for e in range(episodes):
        data = load_trajectory(trajectory_path(root, task, hand, e))
        ep = data["episodes"][0]
        names = data["action_layout"]["robot_joint_names"]
        q0 = np.asarray(ep["initial_state"]["articulation"]["robot"]["joint_position"])[0]
        order = [names.index(n) for n in action_names]
        samples.append(q0[order] - np.asarray(ep["actions"])[0])
    samples = np.stack(samples)
    result = np.asarray(default_offset, dtype=np.float64).copy()
    # Only floating-wrist offsets are inferred. Finger offsets come from the
    # known controller config, not from initial PD tracking error.
    indices = [i for i,n in enumerate(action_names) if n in wrist_names] if wrist_names is not None else [
        i for i,n in enumerate(action_names) if not
        (n.startswith("joint_") or n.startswith("leap_") or n.startswith(("FFJ","MFJ","RFJ","LFJ","THJ")))]
    median = np.median(samples, axis=0)
    for i in indices:
        if abs(median[i] - result[i]) > .02:
            result[i] = median[i]
    return result, {"wrist_offset_spread": np.ptp(samples[:, indices], axis=0).tolist(),
                    "offset": result.tolist(), "inferred_wrist_indices": indices}
