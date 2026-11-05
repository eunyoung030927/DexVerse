"""Upload finished teleop DP (image) runs to the HF model repo Dexverse-CSI/dexverse-dp-image under
   v0/teleop_lerobot/<category>/<Task>/<robot_type>/      (the layout of the dv-demos-v0 dataset the model was trained on)
Each model folder: best/ (LeRobot pretrained_model of the selected snapshot), last/ (final epoch, when kept), README.md (server, hyperparameters, results),
config.yaml, settings.md, meta.json, metrics.csv, select.log.   Token on stdin, never stored.
   echo <token> | python upload_dp_image.py <run_dir> [...]     best/last read from the run's checkpoint dir (config.yaml), eval must have 50 rollouts
   echo <token> | python upload_dp_image.py --readme <file>     replace the repo README (results table made centrally)
"""
import json, re, shutil, sys, tempfile
from pathlib import Path
from huggingface_hub import HfApi

REPO = "Dexverse-CSI/dexverse-dp-image"
CAT = {"GraspPan": "functional", "GraspKettle": "functional", "FunctionalHammerStrike": "functional", "FunctionalPourMug": "functional",
       "GraspBleach": "functional", "GraspTwoItems": "grasping", "BimanualLiftTray": "bimanual", "BimanualLiftCarton": "bimanual",
       "OpenFlatFolder": "articulation", "SlideUtilityKnife": "articulation"}


def episodes(r, part):
    try: return json.loads((r / f"eval/best_{part}/metrics.json").read_text())
    except Exception: return None


def model_readme(r, meta, config, task, rt, cat, best_ok, best_n, best_step, selection, last_note):
    sel = "\n".join(f"| {s} | {ok} |" for s, ok in selection) or "| 기록 없음 | |"
    return f"""# {task} / {rt} — Diffusion Policy (image, 240×320)

LeRobot 0.4.2 official Diffusion Policy trained on the teleop demos `Dexverse-CSI/dv-demos-v0` → `v0/teleop_lerobot/{cat}/{task}/{rt}` (50 episodes).

| 항목 | 값 |
|---|---|
| run_id | `{meta['run_id']}` |
| 학습 서버 | {meta['server']} |
| 학습 기간 (KST) | {meta['start_time']} → {meta['end_time']} |
| 코드 | {meta['git_repo']} @ {meta['git_commit']} ({'dirty' if meta.get('git_dirty') else 'clean'}), 학습·평가 스크립트 `experiments/{meta['server']}/{meta['project']}/` |
| 명령 | `{meta['command']}` |
| seed | {meta['seed']} |
| best 체크포인트 | `best/` = 학습 step {best_step} (50 epoch마다 저장한 스냅숏 중 pkl 에피소드 0–9 성공 수가 가장 큰 것, 동률이면 뒤 스냅숏) |
| last 체크포인트 | {last_note} |
| **최종 평가** | **{best_ok}/{best_n} ({best_ok / max(1, best_n):.2f})** — pkl 에피소드 0–49의 initial_state 복원, DDPM 100 step, 8 step 실행, 최대 3000 step, 시드 없음 |

## 스냅숏 선택 (에피소드 0–9, 10회)

| step | 성공 |
|---|---|
{sel}

## config.yaml

```yaml
{(r / 'config.yaml').read_text().strip()}
```

## settings.md (실험 폴더 자동 생성)

{(r / 'settings.md').read_text().strip()}

## 사용

```python
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
policy = DiffusionPolicy.from_pretrained("<local path to best/>")   # lerobot==0.4.2; 입력 이미지는 240×320으로 resize해서 넣는다
```
"""


def upload_run(api, r):
    r = Path(r).resolve(); meta = json.loads((r / "meta.json").read_text())
    config = dict(line.split(": ", 1) for line in (r / "config.yaml").read_text().splitlines() if ": " in line)
    ck = Path(config["checkpoints"]) / "checkpoints"
    best = next((d for d in (r / "artifacts/best", ck.parent / "best") if (d / "model.safetensors").exists()), None)
    if best is None: print("SKIP", r.name, "no best checkpoint"); return
    last_step = ck.joinpath("last").resolve().name if (ck / "last").exists() else None
    last = next((d for d in (r / "artifacts/last", ck / "last/pretrained_model") if (d / "model.safetensors").exists()), None)
    last_note = f"`last/` = 최종 step {last_step or '기록 없음'} (300 epoch 끝, 평가 안 함, 재개용)" if last else "보관 안 함"

    ev = [episodes(r, p) for p in (0, 1)]
    if any(e is None for e in ev) or sum(e["num_episodes"] for e in ev) != 50: print("SKIP", r.name, "final evaluation incomplete"); return
    best_ok = sum(x["success"] for e in ev for x in e["episodes"]); best_n = 50
    task = config["task"]; rt = Path(config["dataset_root"]).name.split("-v0-", 1)[1]
    cat = CAT[task.replace("Dexverse-", "").rsplit("-v", 1)[0]]
    sel_file = r / "logs/select.log"; selection = re.findall(r"^(\d{6}) (\d+/10)$", sel_file.read_text(), re.M) if sel_file.exists() else []
    best_txt = r / "eval/select/best.txt"; best_step = best_txt.read_text().split()[0] if best_txt.exists() else "기록 없음"
    path = f"v0/teleop_lerobot/{cat}/{task}/{rt}"
    with tempfile.TemporaryDirectory(dir="/workspace/.tmp") as tmp:
        tmp = Path(tmp); shutil.copytree(best, tmp / "best")
        if last: shutil.copytree(last, tmp / "last")
        for f in ("config.yaml", "settings.md", "meta.json", "metrics.csv"): shutil.copy(r / f, tmp / f)
        if sel_file.exists(): shutil.copy(sel_file, tmp / "select.log")
        (tmp / "README.md").write_text(model_readme(r, meta, config, task, rt, cat, best_ok, best_n, best_step, selection, last_note))
        api.upload_folder(repo_id=REPO, repo_type="model", folder_path=str(tmp), path_in_repo=path,
                          commit_message=f"{task} {rt}: DP image, best step {best_step}{', last step ' + str(last_step) if last else ''}, {best_ok}/{best_n} ({meta['server']}, {meta['run_id']})")
    print("UPLOADED", path, f"{best_ok}/{best_n}", flush=True)


api = HfApi(token=sys.stdin.readline().strip())
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
if sys.argv[1] == "--readme":
    api.upload_file(path_or_fileobj=sys.argv[2], path_in_repo="README.md", repo_id=REPO, repo_type="model", commit_message="README: results table")
    print("README updated")
else:
    for run in sys.argv[1:]: upload_run(api, run)
