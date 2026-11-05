"""Build the README of HF Dexverse-CSI/dexverse-dp-image from the central INDEX.csv files (all servers)."""
import csv, re, datetime
from pathlib import Path
E = Path('/mnt/d/hscho_share/experiments')
CAT = {"GraspPan": "functional", "GraspKettle": "functional", "HammerStrike": "functional", "PourMug": "functional", "GraspBleach": "functional",
       "GraspTwoItems": "grasping", "LiftTray": "bimanual", "LiftCarton": "bimanual", "SlideUtilityKnife-v1": "articulation"}
TASK = {"GraspPan": "Dexverse-GraspPan-v0", "GraspKettle": "Dexverse-GraspKettle-v0", "HammerStrike": "Dexverse-FunctionalHammerStrike-v0",
        "PourMug": "Dexverse-FunctionalPourMug-v0", "GraspBleach": "Dexverse-GraspBleach-v0", "GraspTwoItems": "Dexverse-GraspTwoItems-v0",
        "LiftTray": "Dexverse-BimanualLiftTray-v0", "LiftCarton": "Dexverse-BimanualLiftCarton-v0", "SlideUtilityKnife-v1": "Dexverse-SlideUtilityKnife-v1"}
best = {}
for s in ['a4', 'amp7', 'vast-isaac']:
    for row in csv.reader(open(E / s / '20261031_teleop_single/teleop_dp_image/INDEX.csv')):
        if len(row) < 5 or row[0] == 'run_id' or '20261101-1240' in row[0]: continue
        m = re.search(r'best (\d+)/(\d+)', row[4])
        if row[1] == 'done' and m and m.group(2) == '50': best[row[0].split('_', 2)[2]] = (int(m.group(1)), s, row[0])
tasks = ['LiftCarton', 'LiftTray', 'HammerStrike', 'GraspKettle', 'GraspPan', 'PourMug', 'GraspTwoItems', 'GraspBleach', 'SlideUtilityKnife-v1']; hands = ['wuji', 'shadow', 'inspire', 'allegro', 'leap', 'sharpa']
rows = []
for t in tasks:
    cells = []
    for h in hands:
        v = best.get(f'{t.lower()}-{h}')
        cells.append(f'**{v[0] / 50:.2f}** ({v[0]}/50)' if v else '–')
    rows.append(f'| {t} | ' + ' | '.join(cells) + ' |')
paths = []
for t in tasks:
    for h in hands:
        v = best.get(f'{t.lower()}-{h}')
        if v:
            side = 'bimanual' if t in ('LiftTray', 'LiftCarton', 'SlideUtilityKnife-v1') else 'right'
            paths.append(f'| `v0/teleop_lerobot/{CAT[t]}/{TASK[t]}/floating_{h}_{side}/` | {v[0]}/50 | {v[1]} | `{v[2]}` |')
out = f"""---
license: other
tags: [lerobot, diffusion-policy, dexverse, robotics, dexterous-hand]
library_name: lerobot
---
# dexverse-dp-image — Diffusion Policy (image) trained per task × hand on DexVerse teleop demos

LeRobot 0.4.2 official Diffusion Policy, one model per (task, hand), trained on the 50-episode teleop sets of
`Dexverse-CSI/dv-demos-v0` (`v0/teleop_lerobot/...`). Layout mirrors the dataset: `v0/teleop_lerobot/<category>/<Task>/<robot_type>/`
= `best/` (selected snapshot) and `last/` (final epoch, not evaluated) as LeRobot `pretrained_model` dirs (`config.json`, `model.safetensors`, processors, `train_config.json`), `README.md` (server,
hyperparameters, results), `config.yaml`, `settings.md`, `meta.json`, `metrics.csv`, `select.log`.

Common setup: 2 cameras (third_person, eye_in_hand) 480×640 → 240×320 bilinear resize, random crop 216×288, one ResNet18 per camera,
state 28-d joint positions (n_obs 2), action 28-d (horizon 16, execute 8), UNet [512, 1024, 2048], DDPM 100, Adam 1e-4 cosine, batch 64,
300 epochs, seed 42, fp32, no EMA. Snapshot every 50 epochs; `best` = snapshot with the most successes on pkl episodes 0–9 (ties → later).
Final score = 50 rollouts of `best` restoring the initial state of pkl episodes 0–49 (max 3000 steps, no seed).

## Results (success rate of `best`, 50 rollouts) — updated {datetime.datetime.utcnow() + datetime.timedelta(hours=9):%Y-%m-%d %H:%M} KST

| task | wuji | shadow | inspire | allegro | leap | sharpa |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

`–` = no dataset yet, or training/evaluation still running.

## Models

| path | best | trained on | run_id |
|---|---|---|---|
{chr(10).join(paths)}
"""
Path('/tmp/claude-1000/-mnt-d-hscho-share/337769ae-6e9a-47a2-b8ec-8cfabddcdf2b/scratchpad/dp_repo_README.md').write_text(out)
print(len(paths), 'models listed')
