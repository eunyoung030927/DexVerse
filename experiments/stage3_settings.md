# Stage 3 (SPIDER) settings

Source of truth: `scripts/retarget/stage3_spider.py` (CLI), `source/dexverse/dexverse/retarget/spider.py` (`SpiderCfg`),
`scripts/retarget/run_target.py` (attempt schedule). Per (task, target): `<task>/<target>/config.yaml`; each run records
what it used in `<runs>/<task>/demos/epNNN/<target>/spider_aK.json`.

## Current defaults = Bench2Dex "v2", ported to 60 Hz

Bench2Dex retargets at 20 Hz; DexVerse records and controls at 60 Hz. Every frame count is x3 (the same time),
every other value is the Bench2Dex v2 value (bench2dex_retarget `experiments/stage3_settings.md`, chosen 2026-10-06).

| setting | value | Bench2Dex (20 Hz) | meaning |
|---|---|---|---|
| num_samples | 1024 | 1024 | action candidates simulated per replanning step |
| horizon / knot / commit | 90 / 15 / 15 frames | 30 / 5 / 5 | plan 1.5 s, knots every 0.25 s, execute 0.25 s, replan |
| iters / sigma | 5 / 0.5 | 5 / 0.5 | MPPI iterations and sampling std |
| guide_gain | 300 N/m -> 0 | same | virtual contact spring, annealed over the iterations; never on committed frames |
| squeeze | 0.02 m | same | contact targets moved into the object |
| finger_res_cap / wrist | 0.6 rad / 0.05 m, 0.5 rad | same | residual range around the kinematic reference |
| action_ema | 0.8^(1/3) per frame, residual only | 0.8 on the whole target | same time constant; on the residual only, so the moving reference is not delayed |
| gate_margin / gate_ramp | 45 / 15 frames | 15 / 5 | residual allowed around manipulation |
| gate_look | 90 frames | 30 | optimise only when manipulation is this close |
| drop_jump | 0.025 m | same | an active object's error > 4 cm that grew by this much since the last commit = drop |
| backtrack / max_retries | 2 commits / 3 | same | after a drop go back 2 commits, iterations x(1+level), up to 3 times per restart frame, then carry on |
| settle | 120 frames | 40 | the demo's last frame is appended 120 times and still OPTIMISED (2026-10-13; before: the last command was frozen and GraspBleach -> Allegro dropped the bottle there) |
| early stop | none | none | the run always reaches the end of the demo |

### Added 2026-10-13 (GraspBleach-v1 ep0 investigation, `experiments/benchmark_plan.md` §5)

| setting | value | meaning |
|---|---|---|
| position cost | mean over 4 keypoints per body (root, centroid, both ends of the principal axis), 10/m | root position + tilt left the bleach bottle's spout (40 cm from the root, where success is judged within 2 cm of the marker) free to swing 3-6 cm |
| squeeze direction | towards the mesh centroid | the root origin of the bottle sits at its cap, so "towards the origin" ran along the bottle |
| success_bonus | 1.0 per frame per sample | the task's own success term (`lift_and_tilt_with_contact_zones` etc.) is evaluated on every sample env and credited in the cost |
| hold pass | kept only if its mean tip shift <= `--hold_max_shift` 3 cm | the kinematic object throws a thick-fingered hand out of the grasp (Allegro 5 cm, LEAP 13 cm shift, fingers 1-3 rad past the reference); Wuji 0.7 cm keeps it |
| stage 1 contacts | tip link's collision surface (bare tip frames -> nearest collision body) within 2 cm | Shadow's thumb tip origin is 3.5 cm behind its pad: thumb contacts were never recorded, so no target thumb ever pressed |
| stage 2 penetration term | collision-geometry samples of every hand link kept >= 2 mm outside the task bodies, weight 3/point, 8 deepest per link | tip-only IK put Allegro phalanges 3-4.5 cm and LEAP's palm 5 cm inside the bottle |
| stage 2 step cap | 0.2 rad/frame (wrist translation 0.02 m) | the penetration term otherwise flips joints 0.9 rad between frames |

Results, GraspBleach-v1 ep0 (local 3090, GPU shared by 3-4 runs so minutes are ~3x solo):

| target | reference | hold pass | cost | result |
|---|---|---|---|---|
| Allegro | tip-only IK (cluster 6dd4f86) | yes | root + tilt | fail x3 attempts (drop at lift, frame 195) |
| Allegro | penetration-aware | yes | root | fail (fingers thrown 1-3 rad in the hold pass) |
| Allegro | penetration-aware, surface contacts | no | root, centroid squeeze | fail, carried to the end, spout 3.5 cm off |
| Allegro | same | no | keypoints | fail, spout 2.2 cm off at frame 335, dropped in the frozen settle |
| Allegro | same | no | keypoints + success bonus + optimised hold | **SUCCESS** (frame 401, 232 min shared) |
| Wuji | surface contacts | yes | keypoints, no bonus/optimised hold | fail (20 cm) |
| Wuji | old contacts / origin squeeze / root cost (3 ablations) | yes | + success bonus + optimised hold | **SUCCESS** x3 (337-347, 273-284 min shared) |
| LEAP | penetration-aware | - | - | reference still 2.7 cm inside, tip error 2 cm, palm 17 deg off: the hand does not fit the Shadow grasp; untested in SPIDER |

DexVerse-specific (docs/retarget_cross_hand.md):
- **PD lag compensation**: the floating wrist joints have stiffness 1000 / damping 200 (time constant 0.2 s); the
  reference command is advanced to `q_ref + (D/K) dq_ref/dt`. GraspCup ep0 -> Allegro hold-pass fingertip error
  4.1 cm -> 1.1 cm.
- **Hold pass** before SPIDER: objects pinned to the demo trajectory, the hand follows the kinematic reference with PD;
  its reached poses become the reference.

Attempts (`run_target.py`, same as the Bench2Dex cluster): attempt K = seed K, num_samples 1024 * (1 + K // 2),
iters 5 + K, up to 3 (attempts 0-2 = 1024/5, 1024/6, 2048/7), stop at the first task success. No checkpoint resume
yet (Bench2Dex resumes attempt 1 from a checkpoint).

## Rejected settings — do not retry these patterns

From Bench2Dex (06 RH56DFX -> Shadow, 2026-10-05/06), kept here because the search is the same: a dropped object is
usually re-acquired later, so anything that limits or ends the search after a few drops, or shortens the pre-grasp
window, costs successes.

| pattern | Bench2Dex result |
|---|---|
| narrower sampling window (gate_margin 10 / gate_look 10 at 20 Hz) | 64% vs 83% |
| abort after N retries | 5/5 failed, each stopped at the cap |
| per-segment retry budget | used up quickly, then abort |
| "lagging" drop trigger (error > 6 cm and growing) | budget consumed by normal lift transients |
| finger_res_cap 0.25 / 0.4, no squeeze | objects not lifted |

Judge a change only on finished episodes, not on mid-run drops.
