#!/usr/bin/env python
"""Serve a LeRobot pi0.5 checkpoint to pi05_eval_pooled.py (261115; same multiprocessing.connection protocol as the DP
lerobot_dp/policy_server.py: ("reset", None) -> "ok", ("act", message) -> float32 action bytes, ("close", None)).

--ckpt is either a pi05_train.py export dir (trainable.safetensors + configs/processors + export_meta.json; the frozen
weights come from --base, default export_meta["base"] or $PI05_BASE) or a full LeRobot pretrained_model dir.
Memory-lean load: the model is built directly on the GPU in bf16 (stock dtype layout: paligemma_with_expert bf16 with
the stock fp32 exceptions, action/time projections fp32) and the weights are streamed tensor by tensor, so neither CPU
RAM nor GPU ever holds a second copy (~7.5 GB GPU at bs1).
--shared: serve any number of harness connections (one thread + own action queue each, one model, inference under a
lock); otherwise exactly one connection (DP behaviour, the server exits after "close").
Each "act" with an empty queue predicts a chunk (chunk_size 50, flow 10 steps) and queues the first n_action_steps
(config, 8 = DP replanning) unnormalized actions. The prompt task is --task (the dataset's single task string).
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--base", default=None)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--authkey", required=True)
parser.add_argument("--task", default="Grasp kettle")
parser.add_argument("--n_action_steps", type=int, default=None)
parser.add_argument("--shared", action="store_true")
parser.add_argument("--device", default="cuda")
parser.add_argument("--check_only", action="store_true", help="load, run one dummy chunk, print timing, exit")
args = parser.parse_args()

from safetensors import safe_open  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

PROJ = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")


def log(msg):
    print(f"[pi05_server] {msg}", flush=True)


def build_policy(ckpt: Path, device: str):
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device, cfg.compile_model, cfg.gradient_checkpointing = device, False, False
    if args.n_action_steps:
        cfg.n_action_steps = args.n_action_steps
    dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32)
    try:
        with torch.device(device):
            policy = PI05Policy(cfg)
    finally:
        torch.set_default_dtype(dt)
    for name in PROJ:  # stock layout: only paligemma_with_expert is cast to bf16
        getattr(policy.model, name).float()
    return policy, cfg


@torch.no_grad()
def stream(policy, path: Path, skip_modules=(), base=False):
    own = policy.state_dict()
    loaded = set()
    with safe_open(path, "pt", device=str(next(policy.parameters()).device)) as f:
        for k0 in f.keys():
            if base:
                fixed = policy._fix_pytorch_state_dict_keys({k0: None}, policy.config)  # per-key rename/skip
                if not fixed:
                    continue
                k = next(iter(fixed))
            else:
                k = k0
            k = k if k.startswith("model.") else f"model.{k}"
            if k.removeprefix("model.").split(".")[0] in skip_modules:
                continue
            if k not in own:
                continue
            t = f.get_tensor(k0)
            if tuple(t.shape) != tuple(own[k].shape):
                raise ValueError(f"{path}: {k} shape {tuple(t.shape)} != model {tuple(own[k].shape)}")
            own[k].copy_(t)
            loaded.add(k)
    return loaded


def load(ckpt: Path, device: str):
    t0 = time.time()
    policy, cfg = build_policy(ckpt, device)
    own = set(policy.state_dict())
    if (ckpt / "trainable.safetensors").exists():
        meta = json.loads((ckpt / "export_meta.json").read_text())
        base = Path(args.base or os.environ.get("PI05_BASE") or meta["base"])
        got = stream(policy, base / "model.safetensors", skip_modules=PROJ[:2], base=True)
        delta = stream(policy, ckpt / "trainable.safetensors")
        got |= delta
        from lerobot.policies.pi05 import modeling_pi05  # noqa: F401
        import hashlib
        named = dict(policy.named_parameters())
        for k, h in meta.get("frozen_sha16", {}).items():
            mine = hashlib.sha256(named[k].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
            if mine != h:
                raise RuntimeError(f"frozen weight {k} differs from training ({mine} != {h}): wrong --base")
        log(f"base {base} + delta {len(delta)} tensors (step {meta['step']})")
    else:
        got = stream(policy, ckpt / "model.safetensors", base=True)
    sd = policy.state_dict()
    ptrs = {sd[k].data_ptr() for k in got}  # tied weights (paligemma lm_head = embed_tokens) count as loaded
    missing = sorted(k for k in own - got if sd[k].data_ptr() not in ptrs and not k.endswith("gemma_expert.lm_head.weight"))
    if missing:
        raise RuntimeError(f"{len(missing)} weights not loaded, e.g. {missing[:5]}")
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(ckpt),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    log(f"loaded {ckpt} in {time.time() - t0:.0f} s; n_action_steps {cfg.n_action_steps}, chunk {cfg.chunk_size}, "
        f"GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB" if device.startswith("cuda") else f"loaded {ckpt} (cpu)")
    return policy, cfg, pre, post


policy, cfg, preprocessor, postprocessor = load(Path(args.ckpt), args.device)
lock = threading.Lock()
image_keys = list(cfg.image_features)


def to_batch(message: dict) -> dict:
    batch = {}
    for key, (shape, dtype, data) in message.items():
        value = torch.from_numpy(np.frombuffer(data, dtype=dtype).reshape(shape).copy())
        batch[key] = value.permute(2, 0, 1).float() / 255.0 if value.ndim == 3 else value.float()
    batch["task"] = args.task
    return batch


@torch.inference_mode()
def chunk(message: dict) -> list:
    with lock:
        batch = preprocessor(to_batch(message))
        actions = policy.predict_action_chunk(batch)[:, : cfg.n_action_steps]  # (1, n, D) normalized
        actions = postprocessor(actions)
    return [a.cpu().numpy().astype(np.float32) for a in actions[0]]


def serve(conn, name):
    queue = deque()
    n = 0
    while True:
        try:
            kind, message = conn.recv()
        except EOFError:
            break
        if kind == "close":
            break
        if kind == "reset":
            queue.clear()
            conn.send("ok")
            continue
        if not queue:
            t0 = time.perf_counter()
            queue.extend(chunk(message))
            n += 1
            if n <= 2 or n % 200 == 0:
                log(f"{name}: chunk {n} {1000 * (time.perf_counter() - t0):.0f} ms")
        conn.send(queue.popleft().tobytes())
    conn.close()


if args.check_only:
    h, w = 480, 640
    rng = np.random.default_rng(0)
    msg = {k: ((h, w, 3), "uint8", rng.integers(0, 255, (h, w, 3), np.uint8).tobytes()) for k in image_keys}
    sd = cfg.input_features["observation.state"].shape[0]
    msg["observation.state"] = ((sd,), "float32", np.zeros(sd, np.float32).tobytes())
    csum = sum(float(t.detach().double().abs().sum()) for t in policy.state_dict().values())
    log(f"state_dict checksum {csum:.6f}; image keys {image_keys}")
    for i in range(3):
        torch.manual_seed(i)
        t0 = time.perf_counter()
        out = chunk(msg)
        log(f"check chunk {i} first action[:6] {np.round(out[0][:6], 5).tolist()}")
        log(f"check chunk {i}: {len(out)} x {out[0].shape} in {1000 * (time.perf_counter() - t0):.0f} ms, "
            f"finite {all(np.isfinite(a).all() for a in out)}")
    if args.device.startswith("cuda"):
        log(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    sys.exit(0)

with Listener(("127.0.0.1", args.port), authkey=args.authkey.encode()) as listener:
    log(f"listening on 127.0.0.1:{args.port} ({'shared' if args.shared else 'single'})")
    if not args.shared:
        with listener.accept() as conn:
            serve(conn, "client")
    else:
        k = 0
        while True:
            conn = listener.accept()
            k += 1
            threading.Thread(target=serve, args=(conn, f"client{k}"), daemon=True).start()
