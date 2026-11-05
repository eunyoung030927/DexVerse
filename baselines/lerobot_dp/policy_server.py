#!/usr/bin/env python
"""Serve a LeRobot Diffusion Policy to eval_lerobot.py (lerobot env; Isaac Sim lives in the main env)."""
import argparse
from multiprocessing.connection import Listener
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_resize import install_resize_for  # noqa: E402

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="pretrained_model folder of a checkpoint.")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--authkey", required=True)
args = parser.parse_args()

device = torch.device("cuda")
policy = DiffusionPolicy.from_pretrained(args.ckpt).to(device).eval()
print(f"[policy] 240x320 resize: {install_resize_for(policy.config)}", flush=True)
preprocessor, postprocessor = make_pre_post_processors(
    policy.config, pretrained_path=args.ckpt, preprocessor_overrides={"device_processor": {"device": str(device)}}
)


def to_batch(message: dict) -> dict:
    batch = {}
    for key, (shape, dtype, data) in message.items():
        value = torch.from_numpy(np.frombuffer(data, dtype=dtype).reshape(shape).copy())
        batch[key] = value.permute(2, 0, 1).float() / 255.0 if value.ndim == 3 else value.float()
    return batch


with Listener(("127.0.0.1", args.port), authkey=args.authkey.encode()) as listener, listener.accept() as conn:
    while True:
        kind, message = conn.recv()
        if kind == "close":
            break
        if kind == "reset":
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            conn.send("ok")
            continue
        with torch.inference_mode():
            action = postprocessor(policy.select_action(preprocessor(to_batch(message))))
        conn.send(action[0].cpu().numpy().astype(np.float32).tobytes())
