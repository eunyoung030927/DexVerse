#!/usr/bin/env python3
"""HO H2 driver (261115, work2 only): 2-hand (shadow + allegro) nohot DP arms Z and D (trained on vast by
ho_vast_train.sh, datasets ho2hand/sa_{Z,D}) -> pull snapshots vast -> work2, selection on the TRAINING hands, final on
the UNSEEN hands sharpa, wuji, leap. New file, host python3 stdlib, Isaac jobs via ho_lib.Runner (ho_eval_pooled.py,
HoCodec, D with the shadow+allegro f_task ho/h2/f_task/f_task_best.pt, --dataset ho2hand/sa_{arm}).
  pull   : from start (no gate); every 3 min one ssh listing of vast runs/ho2hand/{Z,D}/train/checkpoints/<step>/
           pretrained_model (model.safetensors + train_config.json, >= 3 min old) -> ho/h2/ckpt/<arm>/<step> (tar stream,
           sha256 of every file compared, .complete); key ~/.ssh/ho_vast_ed25519 (work2's own key, authorized on vast),
           host --vast. An arm whose vast training logs '[ho_vast] <arm>: GIVING UP' is dropped (-> '[ho_h2] FAIL').
  gate   : evaluation starts after '[ho_h1] ALL DONE' or '[ho_h1] FAIL' in ho/logs/h1_driver.log (H1 first, same 3-worker
           budget; ho_lib counts every eval alive in the container).
  select : per arm, shadow + allegro x eps 0-9 x snapshots of epochs 50..250 (5 steps); best = max 2-hand total /20,
           ties -> later step -> ho/h2/sel/<arm>_best.json, '[ho_h2] SELECT <arm> done best=...'.
  final  : best snapshot on sharpa, wuji, leap eps 0-9; a cell with >= 1 success is extended to eps 0-49 (leap only if
           a 50-ep leap demos pkl exists, else BLOCKED). Training-hand 50-ep reference: OFF by default (estimated > 30%
           extra wall time: 160 extra rollouts vs ~260-420 for the protocol); --train-ref enables it.
Outputs ho/h2/{sel,final}/..., ho/h2/RESULTS_H2.md, ho/h2/h2_summary.json.
Markers in ho/logs/h2_driver.log: [ho_h2] WAITING ... | [ho_h2] GATE open | [ho_h2] SELECT <arm> done | [ho_h2] FAIL ... |
  [ho_h2] ALL DONE
usage: nohup python3 ho_h2_driver.py --parallel 3 >> ho/logs/h2_driver_nohup.out 2>&1 &
"""
import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ho_lib as L  # noqa: E402

H2 = L.HO / "h2"
ARMS = ("Z", "D")
TRAIN_HANDS = ("shadow", "allegro")
UNSEEN = ("sharpa", "wuji", "leap")
F_TASK = "artifacts/pooled3hand/ho/h2/f_task/f_task_best.pt"
DATASET = "ho2hand/sa_{arm}"
DS_INFO = Path("/home/eun/xhand/datasets/ho2hand/sa_Z/meta/info.json")
VAST_RUNS = "/workspace/xhand/runs/ho2hand"

ap = argparse.ArgumentParser()
ap.add_argument("--parallel", type=int, default=3)
ap.add_argument("--poll", type=int, default=60)
ap.add_argument("--vast", default="root@<VAST_HOST>:<VAST_PORT>")
ap.add_argument("--train-ref", action="store_true")
ap.add_argument("--no-wait", action="store_true")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
LOG = L.HO / "logs/h2_driver.log"
_spe = math.ceil(json.loads(DS_INFO.read_text())["total_frames"] / 64)  # steps per epoch (batch 64)
ALL_STEPS = tuple(f"{_spe * e:06d}" for e in (50, 100, 150, 200, 250, 300))
SEL_STEPS = ALL_STEPS[:5]
USER, HP = args.vast.split(":")[0], args.vast.split(":")[1]
VAST = ["ssh", "-i", os.path.expanduser("~/.ssh/ho_vast_ed25519"), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", "-p", HP, USER]


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{L.now()} {msg}\n")


def read(p):
    try:
        return Path(p).read_text(errors="ignore")
    except OSError:
        return ""


def ckpt_dir(arm, step):
    return H2 / "ckpt" / arm / step


def ready(arm, step):
    return (ckpt_dir(arm, step) / ".complete").exists()


# ----------------------------------------------------------------------------- pull (vast -> work2)
pull_busy, last_pull = [False], [0.0]
gave_up = set()


def remote_ready():
    cmd = (f"for f in {VAST_RUNS}/{{Z,D}}/train/checkpoints/[0-9]*/pretrained_model/model.safetensors; do "
           "d=$(dirname $f); test -f $d/train_config.json || continue; "
           "echo $f $(( $(date +%s) - $(stat -c %Y $f) )) $(stat -c %s $f); done")
    cmd += f"; grep -o '\\[ho_vast\\] [ZD]: GIVING UP' {VAST_RUNS}/logs/vast_queue.log 2>/dev/null"
    out = {}
    r = L.sh(VAST + [cmd], timeout=120)
    if r.returncode not in (0, 1):  # 1 = the final grep found no GIVING UP line
        log(f"[pull] vast unreachable this pass: {r.stderr.strip()[-150:]}")
        return out
    for line in r.stdout.splitlines():
        if "GIVING UP" in line:
            arm = line.split("]")[1].strip()[0]
            if arm not in gave_up:
                gave_up.add(arm)
                log(f"[ho_h2] vast training {arm} GAVE UP -> arm dropped")
            continue
        p = line.split()
        if len(p) == 3 and p[1].isdigit() and int(p[1]) >= 180:
            parts = p[0].split("/")
            out[(parts[-6], parts[-3])] = int(p[2])
    return out


def pull_one(arm, step, size):
    rd = f"{VAST_RUNS}/{arm}/train/checkpoints/{step}"
    d = ckpt_dir(arm, step)
    tmp = d.parent / f".{step}.tmp"
    L.sh(["rm", "-rf", str(tmp)])
    tmp.mkdir(parents=True)
    t0 = time.time()
    tar = subprocess.Popen(VAST + [f"tar -C {rd} -cf - pretrained_model"], stdout=subprocess.PIPE)
    r = subprocess.run(["tar", "-xf", "-", "-C", str(tmp)], stdin=tar.stdout, timeout=3600)
    tar.wait(timeout=60)
    if r.returncode or tar.returncode:
        log(f"[pull] {arm} {step} tar failed rc={r.returncode}/{tar.returncode}")
        return
    rs = L.sh(VAST + [f"cd {rd}/pretrained_model && sha256sum *"], timeout=600).stdout.split()
    ls = L.sh(["bash", "-c", f"cd {tmp}/pretrained_model && sha256sum *"]).stdout.split()
    if not rs or sorted(rs) != sorted(ls):
        log(f"[pull] {arm} {step} sha256 mismatch, will retry")
        return
    L.sh(["rm", "-rf", str(d)])
    tmp.rename(d)
    (d / ".complete").write_text(f"pulled {L.now()} from vast {rd} sha256 ok\n")
    log(f"[pull] {arm} {step} OK {size / 1e9:.2f} GB in {time.time() - t0:.0f}s")


def pull_all():
    try:
        rr = remote_ready()
        for step in ALL_STEPS:
            for arm in ARMS:
                if not ready(arm, step) and (arm, step) in rr:
                    try:
                        pull_one(arm, step, rr[(arm, step)])
                    except Exception as exc:  # noqa: BLE001
                        log(f"[pull] {arm} {step} error {exc!r}"[:300])
    finally:
        pull_busy[0] = False


def pull_vast():
    if pull_busy[0] or time.time() - last_pull[0] < 180 or all(ready(a, s) for a in ARMS for s in ALL_STEPS):
        return
    last_pull[0] = time.time()
    pull_busy[0] = True
    threading.Thread(target=pull_all, daemon=True).start()


# ----------------------------------------------------------------------------- jobs
def job(kind, arm, step, hand, lo, hi):
    return L.Job(f"h2/{kind}/{arm}_{step}/{hand}_ep{lo}-{hi}", arm, ckpt_dir(arm, step) / "pretrained_model", hand, lo, hi,
                 F_TASK, DATASET)


def sel_jobs(arm):
    return [job("sel", arm, s, h, 0, 9) for s in SEL_STEPS for h in TRAIN_HANDS]


def best_of(arm):
    p = H2 / "sel" / f"{arm}_best.json"
    return json.loads(read(p)) if p.exists() else None


blocked = set()


def final_jobs(arm, step):
    js = []
    hands = UNSEEN + (TRAIN_HANDS if args.train_ref else ())
    for hand in hands:
        base = job("final", arm, step, hand, 0, 9) if hand in UNSEEN else None
        if hand in TRAIN_HANDS:  # reference: selection eps 0-9 already exist for the best step
            js += [job("final", arm, step, hand, lo, hi) for lo, hi in L.CHUNKS[1:]]
            continue
        js.append(base)
        if base.done() and base.successes() >= 1:
            if L.demos_rel(hand)[1] >= 50:
                js += [job("final", arm, step, hand, lo, hi) for lo, hi in L.CHUNKS[1:]]
            elif (arm, hand) not in blocked:
                blocked.add((arm, hand))
                log(f"[ho_h2] EXT BLOCKED {arm} {hand}: {base.successes()}/10 on eps 0-9, only eps 0-9 exist locally")
    return js


def update_selection():
    for arm in ARMS:
        if best_of(arm):
            continue
        js = sel_jobs(arm)
        if not all(j.done() for j in js):
            continue
        per = {s: {h: next(j for j in js if j.step_ == s and j.hand == h).successes() for h in TRAIN_HANDS} for s in SEL_STEPS}
        tot = {s: sum(per[s].values()) for s in SEL_STEPS}
        best = max(SEL_STEPS, key=lambda s: (tot[s], int(s)))
        rec = {"arm": arm, "best": best, "rule": "max 2-hand (shadow+allegro) total /20 over epochs 50..250, ties -> later",
               "totals": tot, "per_hand": per, "time": L.now()}
        (H2 / "sel").mkdir(parents=True, exist_ok=True)
        (H2 / "sel" / f"{arm}_best.json").write_text(json.dumps(rec, indent=1))
        log(f"[ho_h2] SELECT {arm} done best={best} totals " + " ".join(f"{s}:{tot[s]}/20" for s in SEL_STEPS)
            + " per_hand " + json.dumps(per[best]))


def plan():
    js = []
    arms = [a for a in ARMS if a not in gave_up or best_of(a)]
    for arm in arms:
        for j in sel_jobs(arm):
            js.append(j)
        b = best_of(arm)
        if b:
            js += final_jobs(arm, b["best"])
    complete = all(best_of(a) for a in arms) and all(j.done() for j in js)
    return js, complete


def results(js, failed):
    md = ["# RESULTS_H2 — 2-hand (shadow + allegro) nohot Z/D, zero-shot on unseen sharpa / wuji / leap (GraspKettle)", "",
          f"Generated {L.now()} by ho_h2_driver.py (work2). Training: vast, datasets ho2hand/sa_{{Z,D}} (100 eps, 27,633 frames), "
          f"300 epochs, snapshots {', '.join(ALL_STEPS)}. D uses the shadow+allegro f_task. Eval: ho_eval_pooled.py, 3000-step cap.",
          "", "## Selection (training hands, eps 0-9)", "", "| arm | " + " | ".join(SEL_STEPS) + " | best |",
          "|---|" + "---|" * (len(SEL_STEPS) + 1)]
    summ = {"steps": ALL_STEPS, "selection": {}, "final": {}, "failed": sorted(failed), "blocked": sorted(map(list, blocked)),
            "train_ref": args.train_ref}
    for arm in ARMS:
        b = best_of(arm)
        sj = sel_jobs(arm)
        cells = []
        for s in SEL_STEPS:
            per = {h: next(j for j in sj if j.step_ == s and j.hand == h) for h in TRAIN_HANDS}
            cells.append(" + ".join(f"{per[h].successes()}" for h in TRAIN_HANDS) + f" = {sum(per[h].successes() for h in TRAIN_HANDS)}/20")
        md.append(f"| {arm} | " + " | ".join(cells) + f" | {b['best'] if b else 'n/a'} |")
        summ["selection"][arm] = b
    md += ["", "(cells: shadow + allegro = total)", "", "## Final (best snapshot)", "",
           "| arm (best) | hand | eps 0-9 | all evaluated | extended |", "|---|---|---|---|---|"]
    for arm in ARMS:
        b = best_of(arm)
        if not b:
            continue
        for hand in UNSEEN + (TRAIN_HANDS if args.train_ref else ()):
            cj = [j for j in js if j.name.startswith(f"h2/final/{arm}_") and j.hand == hand]
            if hand in TRAIN_HANDS:
                cj += [j for j in sel_jobs(arm) if j.step_ == b["best"] and j.hand == hand]
            n = sum(j.n_done() for j in cj)
            s = sum(j.successes() for j in cj if j.n_done())
            b0 = next((j for j in cj if j.lo == 0), None)
            e0 = f"{b0.successes()}/{b0.n_done()}" if b0 else "-"
            ext = "yes" if any(j.lo > 0 for j in cj) else ("BLOCKED (no eps 10-49)" if (arm, hand) in blocked else "no")
            md.append(f"| {arm} ({b['best']}) | {hand}{' (train, ref)' if hand in TRAIN_HANDS else ''} | {e0} | {s}/{n} | {ext} |")
            summ["final"][f"{arm}/{hand}"] = {"ep0-9": e0, "total": [s, n], "extended": ext}
    md += ["", f"Failed jobs: {sorted(failed) or 'none'}. Training-hand 50-ep reference: {'on' if args.train_ref else 'off (>30% extra wall time)'}.",
           "leap: only eps 0-9 available locally (HF token invalid)."]
    H2.mkdir(parents=True, exist_ok=True)
    (H2 / "RESULTS_H2.md").write_text("\n".join(md) + "\n")
    (H2 / "h2_summary.json").write_text(json.dumps(summ, indent=1))


def h1_finished():
    t = read(L.HO / "logs/h1_driver.log")
    return "[ho_h1] ALL DONE" in t or "[ho_h1] FAIL" in t


_orig_init = L.Job.__init__


def _init(self, name, *a, **kw):  # remember the snapshot step of the job name (h2/<kind>/<arm>_<step>/...)
    _orig_init(self, name, *a, **kw)
    self.step_ = name.split("/")[2].split("_")[1]


L.Job.__init__ = _init


def main():
    (L.HO / "logs").mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print("steps", ALL_STEPS, "h1_finished", h1_finished(), "ready", {a: [s for s in ALL_STEPS if ready(a, s)] for a in ARMS})
        print("remote", remote_ready())
        for j in plan()[0]:
            print(j.name, j.done())
        return
    lock = open(L.HO / "logs/h2_driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("h2 driver already running")
        return
    (L.HO / "logs/h2_driver.pid").write_text(str(os.getpid()))
    log(f"h2 driver start pid {os.getpid()} parallel {args.parallel} steps {ALL_STEPS} train_ref {args.train_ref}")
    R = L.Runner("ho_h2", args.parallel, log)
    started, waiting_logged, last_beat = False, False, 0.0
    while True:
        pull_vast()
        if not started:
            if not (args.no_wait or h1_finished()):
                if not waiting_logged:
                    log("[ho_h2] WAITING for '[ho_h1] ALL DONE'/'FAIL' (H1 first); pulling snapshots meanwhile")
                    waiting_logged = True
                time.sleep(args.poll)
                continue
            started = True
            log(f"[ho_h2] GATE open; ckpts {json.dumps({a: [s for s in ALL_STEPS if ready(a, s)] for a in ARMS})}")
        R.reap()
        update_selection()
        js, complete = plan()
        if complete and not R.running:
            results(js, R.failed)
            if gave_up - {a for a in ARMS if best_of(a)}:
                log(f"[ho_h2] FAIL vast training gave up for {sorted(gave_up)}; other arms done -> {H2}/RESULTS_H2.md")
            else:
                log(f"[ho_h2] ALL DONE {len(js)} jobs -> {H2}/RESULTS_H2.md")
            return
        todo = [j for j in js if not j.done() and ready(j.arm, j.step_)]
        stuck = [j for j in js if not j.done()]
        if stuck and all(j.name in R.failed for j in stuck) and not R.running:
            results(js, R.failed)
            log(f"[ho_h2] FAIL {sorted(R.failed)}; partial results -> {H2}/RESULTS_H2.md")
            return
        todo.sort(key=lambda j: ("/final/" in j.name, ARMS.index(j.arm), j.step_, j.lo, j.hand))
        R.step(todo)
        if time.time() - last_beat > 1800:
            last_beat = time.time()
            log(f"heartbeat running={sorted(R.running)} done={sum(j.done() for j in js)}/{len(js)} "
                f"ckpts={json.dumps({a: [s for s in ALL_STEPS if ready(a, s)] for a in ARMS})} failed={sorted(R.failed)}")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
