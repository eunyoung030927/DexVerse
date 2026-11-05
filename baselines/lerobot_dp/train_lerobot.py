#!/usr/bin/env python
"""Train LeRobot's official Diffusion Policy on one local LeRobot dataset (lerobot env).

  /workspace/envs/dexverse/lerobot/bin/python train_lerobot.py --dataset_root <dataset> --output_dir <dir>

LeRobot defaults except: one RGB encoder per camera, crop 432x576 of 480x640 (--resize: crop 216x288 of 240x320; the
default 84 is sized for 96 px images), and the DexVerse DP schedule (batch 64, 300 epochs, one snapshot per 50 epochs).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("EXPERIMENTS_ROOT", "/workspace/experiments"))  # experiments/seed_control.py: the global seed rule (EXPERIMENT_RULES.md 4)
from seed_control import DEFAULT_SEED, TRAIN_SEEDS, seed_everything  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--seed", type=int, default=DEFAULT_SEED, choices=TRAIN_SEEDS)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--snapshot_epochs", type=int, default=50)
parser.add_argument("--steps", type=int, default=None, help="Override the step count (tests).")
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument("--resize", action="store_true", help="Resize images to 240x320 (data loader; encoders catch the rest).")
parser.add_argument("--resize_stage", choices=["dataloader", "encoder"], default="dataloader",
                    help="Where --resize happens: in the data loader workers (default) or in front of the encoders (runs before 2026-11-01 14:00 KST).")
parser.add_argument("--resume", action="store_true", help="Continue from <output_dir>/checkpoints/last up to the step count.")
args = parser.parse_args()

from image_resize import FULL_CROP, RESIZED_CROP, install_dataset_resize, install_resize  # noqa: E402

crop = RESIZED_CROP if args.resize else FULL_CROP
info = json.loads((Path(args.dataset_root) / "meta/info.json").read_text())
steps_per_epoch = math.ceil(info["total_frames"] / args.batch_size)
steps = args.steps or args.epochs * steps_per_epoch
save_freq = steps if args.steps else args.snapshot_epochs * steps_per_epoch

import torch  # noqa: E402

if not torch.cuda.is_available():  # LeRobot would warn and silently train on CPU (2026-11-03 a4: MPS refused clients -> 16 s/step)
    sys.exit("[train] CUDA is not available (GPU lost or MPS stuck): refusing to train on CPU")
seed_everything(args.seed)
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", ACCELERATE_MIXED_PRECISION="no")  # 32-bit forward and gradients
sys.argv = [sys.argv[0],
            f"--dataset.repo_id=local/{Path(args.dataset_root).name}", f"--dataset.root={args.dataset_root}",
            "--policy.type=diffusion", "--policy.device=cuda", "--policy.push_to_hub=false",
            f"--policy.crop_shape=[{crop[0]},{crop[1]}]", "--policy.use_separate_rgb_encoder_per_camera=true",
            f"--batch_size={args.batch_size}", f"--num_workers={args.num_workers}", f"--steps={steps}",
            f"--save_freq={save_freq}", "--eval_freq=0", "--log_freq=200", "--wandb.enable=false",
            f"--seed={args.seed}", f"--output_dir={args.output_dir}", f"--job_name={Path(args.output_dir).name}"]
if args.resume:
    config = Path(args.output_dir) / "checkpoints/last/pretrained_model/train_config.json"
    sys.argv = [sys.argv[0], "--resume=true", f"--config_path={config}", f"--steps={steps}"]
print(f"[train] {info['total_frames']} frames, {steps_per_epoch} steps/epoch, {steps} steps, seed {args.seed}, "
      f"resize stage: {args.resize_stage if args.resize else 'none'}", flush=True)

from data_keys import install_needed_keys, install_worker_settings  # noqa: E402
from lerobot.scripts.lerobot_train import main  # noqa: E402

install_needed_keys()
install_worker_settings()
if args.resize:
    if args.resize_stage == "dataloader":
        install_dataset_resize()
    install_resize()
main()
