#!/usr/bin/env python3
"""Required run files of EXPERIMENT_RULES.md, from the values actually used (training config, logs, eval metrics)."""
import json
import shutil
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
now = lambda: datetime.now(KST).isoformat(timespec="seconds")


def start(r, task, hand, res, pkl, dataset, ckpt, epochs, command, server, num_workers):
    meta = r / "meta.json"
    if meta.exists():
        return
    git = lambda c: subprocess.run(f"git -C /workspace/codes/dexverse/DexVerse {c}", shell=True, capture_output=True, text=True).stdout.strip()
    meta.write_text(json.dumps({
        "run_id": r.name, "project": "20261031_teleop_single/teleop_dp_image", "server": server, "agent": "claude-code",
        "git_repo": git("remote get-url origin"), "git_commit": git("rev-parse HEAD"), "git_dirty": bool(git("status --porcelain")),
        "docker_image": None, "command": command, "seed": 42, "start_time": now(), "end_time": None, "status": "running",
    }, indent=2))
    (r / "config.yaml").write_text(
        f"task: Dexverse-{task if task[-3:-1] == '-v' else task + '-v0'}\nhand: {hand}\ninput_width: {res}\npickle: {pkl}\ndataset_root: {dataset}\ncheckpoints: {ckpt}\n"
        f"epochs: {epochs}\nbatch_size: 64\nseed: 42\nnum_workers: {num_workers}\npolicy: lerobot diffusion\nseparate_rgb_encoders: true\n")
    (r / "metrics.csv").write_text("step,split,metric,value\n")
    (r / "settings.md").write_text("학습 중. 종료 때 다시 생성한다.\n\n## 입력 (Input)\n## 출력 (Output)\n## 학습 하이퍼파라미터\n## 추론(평가) 하이퍼파라미터\n")
    (r / "README.md").write_text(f"## 목적\n텔레옵 데이터셋 {task} {hand}가 학습 가능한지 확인한다.\n\n## 변경점\n27세트 공통 설정 (240×320 입력, LeRobot 공식 DP).\n\n## 결과\n학습 중.\n\n## 결론\n학습 중.\n")


def eval_rate(r, kind):
    episodes = []
    for part in (0, 1):
        path = r / f"eval/{kind}_{part}/metrics.json"
        if path.exists():
            episodes += json.loads(path.read_text())["episodes"]
    return sum(e["success"] for e in episodes), len(episodes)


def end(r, status):
    meta = json.loads((r / "meta.json").read_text())
    meta.update(end_time=now(), status=status)
    (r / "meta.json").write_text(json.dumps(meta, indent=2))
    config = dict(line.split(": ", 1) for line in (r / "config.yaml").read_text().splitlines())
    rows = ["step,split,metric,value"]
    log = r / "logs/train.log"
    if log.exists():
        matches = enumerate(re.finditer(r"step:\S+ .*?loss:(\S+) .*?updt_s:(\S+) data_s:(\S+)", log.read_text()), 1)
        rows += [f"{i * 200},train,{name},{m[k]}" for i, m in matches for k, name in ((1, "loss"), (2, "update_s"), (3, "data_s"))]
    best_ok, best_n = eval_rate(r, "best")
    last_ok, last_n = eval_rate(r, "last")
    stage = re.search(r"resize stage: (\S+)", log.read_text()) if log.exists() else None
    config["resize_stage"] = stage[1] if stage else "encoder"  # runs started before the marker existed resized in the encoders
    if "resize_stage" not in (r / "config.yaml").read_text():
        with (r / "config.yaml").open("a") as f:
            f.write(f"resize_stage: {config['resize_stage']}\n")
    rows += [f"final,eval,best_success,{best_ok / max(1, best_n)}"] + ([f"final,eval,last_success,{last_ok / last_n}"] if last_n else [])
    last_text = f", last {last_ok}/{last_n}" if last_n else ""  # last was evaluated only for runs that finished before 2026-11-02 12:30 KST
    (r / "metrics.csv").write_text("\n".join(rows) + "\n")
    train = {}
    path = Path(config["checkpoints"]) / "checkpoints/last/pretrained_model/config.json"
    if path.exists():
        train = json.loads(path.read_text())
    best = (r / "eval/select/best.txt").read_text().strip() if (r / "eval/select/best.txt").exists() else "기록 없음"
    info = json.loads((Path(config["dataset_root"]) / "meta/info.json").read_text())
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    dims = {k: info["features"][k]["shape"][0] for k in ("observation.state", "action")}
    image = "480×640 그대로" if config["input_width"] == "640" else f"480×640 → 240×320 bilinear antialias resize ({config['resize_stage']} 단계)"
    (r / "settings.md").write_text(f"""# {r.name}

## 태스크·데이터
- 태스크 {config['task']}, 손 {config['hand']}
- 데모 pkl: {config['pickle']}
- 데이터셋: {config['dataset_root']} (LeRobot {info['codebase_version']}, {info['total_episodes']} 에피소드 {info['total_frames']} 프레임, {info['fps']} fps, 전체 학습, val 분할 없음)

## 입력 (Input)
- 이미지: 사용, {cameras}, {image}, crop {train.get('crop_shape', '기록 없음')} (학습 랜덤, 평가 중앙)
- 상태: observation.state {dims['observation.state']}차원 (joint_pos), 관측 길이 {train.get('n_obs_steps', '기록 없음')}
- 정규화: {train.get('normalization_mapping', '기록 없음')}

## 출력 (Output)
- action {dims['action']}차원 (env 행동 그대로), 예측 {train.get('horizon', '기록 없음')}, 실행 {train.get('n_action_steps', '기록 없음')}, 60 Hz

## 모델
- LeRobot 0.4.2 Diffusion Policy, 카메라별 ResNet18, UNet {train.get('down_dims', '기록 없음')}

## 학습 하이퍼파라미터
- Adam lr {train.get('optimizer_lr', '기록 없음')}, cosine warmup {train.get('scheduler_warmup_steps', '기록 없음')}, batch 64, {config['epochs']} epoch, grad clip 10, EMA 없음, seed 42, fp32, early stop 없음, 데이터 worker {config.get('num_workers', '기록 없음')}
- 학습 코드 commit {meta['git_commit']}

## 추론(평가) 하이퍼파라미터
- 체크포인트: best = {best} (평가), last = 최종 step (재개용 보관{", 평가함" if last_n else ", 평가 안 함"})
- DDPM {train.get('num_train_timesteps', '기록 없음')} step, 8 step 실행, 최대 3000 step
- pkl 에피소드 50개의 initial_state 복원, 시드 없음, 평가 코드 commit {meta['git_commit']}
- 결과: best {best_ok}/{best_n}{last_text}
""")
    readme = (r / "README.md").read_text().split("## 결과")[0]
    (r / "README.md").write_text(f"{readme}## 결과\nbest {best_ok}/{best_n}{last_text} (선택 {best}).\n\n## 결론\n27세트 결과와 함께 판단한다.\n")
    with (r.parent / "INDEX.csv").open("a") as f:
        f.write(f'{r.name},{status},best_success,{best_ok / max(1, best_n)},"input {config["input_width"]} best {best_ok}/{best_n}{last_text}"\n')
    # checkpoints are not copied into artifacts/: the HF repo Dexverse-CSI/dexverse-dp-image is the store (user, 2026-11-03), upload_dp_image.py reads best/last from the checkpoint dir


r = Path(sys.argv[2])
start(r, *sys.argv[3:]) if sys.argv[1] == "start" else end(r, sys.argv[3])
