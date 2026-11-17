#!/usr/bin/env python3
"""CV3HM eval driver (261117; work2 only): 3-hand (shadow + wuji + allegro) nohot FunctionalHammerStrike image DP, arms from
ctl/arms.json (Zw3h = codec-v3 wb_cl_l z, Ze3h = E2* z), trained on vast (runs/cv3hm/<arm>). Adapted from cv3dp_driver.py
(unchanged original, 261117 A2 version) to the CV3HM PREREG (artifacts/codec_v3/cv3hm/PREREG.md). Host python3 stdlib; Isaac
jobs via ho_lib.Runner (unchanged module; path globals pointed at artifacts/codec_v3/cv3hm) running cv3hm_eval_pooled.py.
Differences from cv3dp_driver.py:
  - work2 only (work1 has ~6 GB free disk): ckpt read in place from ckpt_archive/<arm>/<step> (PC relay
    cv3hm_relay_pc.sh pulls vast -> work2, sha256-checked, .complete). No work1 / WANT.json / host-picking logic.
  - priority BELOW cv3dp and pi0.5: no cv3hm job is launched while a cv3dp_driver.py or pi05_driver.py is alive on
    work2 and has not logged ALL DONE/FAIL (cv3dp) or ALL DONE/FAIL (pi0.5); also not while ctl/DRAIN or ctl/HOLD exist.
    On top of that the unchanged ho_lib budget (every eval alive in the container counts, GPU >= 7.8 GB, RAM >= 7.5 GB,
    150 s between launches, flock -s /tmp/isaac_gpu.lock).
  - R0 (descriptive decode-replay ceiling, PREREG): training hands x eps 0-9 x both codecs, dataset action rows through the
    eval decode path (--replay_npz r0/replay_<tag>.npz), no policy. Runs first (needs no snapshot).
  - --max_steps default 1500 (PREREG cap rule), identical for all jobs.
  select : per arm, shadow + wuji + allegro x eps 0-9 x snapshots of epochs 50..250; best = max 3-hand total /30, ties ->
           later step -> sel/<arm>_best.json.  Held-out hands never used.
  final  : best snapshot on sharpa and leap eps 0-49 (5 chunks each), then training-hand reference (eps 10-49; eps 0-9 =
           the selection rows of the best step).
Markers in artifacts/codec_v3/cv3hm/logs/driver_work2.log: [cv3hm] GATE open | SELECT <arm> | FINAL <arm> | R0 | FAIL | ALL DONE.
usage (work2 host, cwd latent_codec_v1): nohup python3 xhand_latent/cv3hm_driver.py --parallel 3 \
    >> artifacts/codec_v3/cv3hm/logs/driver_nohup_work2.out 2>&1 &
"""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ho_lib as L  # noqa: E402

CV = L.HOST_LC / "artifacts/codec_v3/cv3hm"
CCV = f"{L.CONT_LC}/artifacts/codec_v3/cv3hm"
L.HO, L.CHO = CV, CCV
L.EVAL = f"{L.CONT_LC}/xhand_latent/cv3hm_eval_pooled.py"
L.EVAL_PATTERNS = ("[c]v3hm_eval_pooled.py", "[c]v3dp_eval_pooled.py", "[c]v2dp_eval_pooled.py", "[h]m_eval_masked.py",
                   "[h]o_eval_pooled.py", "[p]3h_nh_eval_pooled.py", "[p]3h_eval_pooled.py", "[p]i05_eval_pooled.py")
TASK = "Dexverse-FunctionalHammerStrike-v0"
TRAIN_HANDS = ("shadow", "wuji", "allegro")
TEST_HANDS = ("sharpa", "leap")
EPOCHS = (50, 100, 150, 200, 250, 300)
N_SEL = 5
DEMOS = {h: f"{CCV}/demos/{TASK}_floating_{h}_right.pkl" for h in ("shadow", "wuji", "allegro", "sharpa", "leap")}
R0_CODECS = {"e2": "artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt", "v3wbcll": "artifacts/codec_v3/runs/wb_cl_l/best.pt"}

ap = argparse.ArgumentParser()
ap.add_argument("--parallel", type=int, default=3)
ap.add_argument("--poll", type=int, default=60)
ap.add_argument("--no-train-ref", dest="train_ref", action="store_false")
ap.add_argument("--no-r0", dest="r0", action="store_false")
ap.add_argument("--max_steps", type=int, default=1500)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
LOG = CV / "logs/driver_work2.log"


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{L.now()} {msg}\n")


def arms():
    try:
        return json.loads((CV / "ctl/arms.json").read_text())
    except (OSError, ValueError):
        return {}


def steps(arm):
    spe = arms()[arm]["spe"]
    return tuple(f"{spe * e:06d}" for e in EPOCHS)


def ckpt_dir(arm, step):
    return CV / "ckpt_archive" / arm / step


def ready(arm, step):
    return arm == "R0" or (ckpt_dir(arm, step) / ".complete").exists()


class Job(L.Job):
    def __init__(self, kind, arm, step, hand, lo, hi, codec=None):
        super().__init__(f"{kind}/{arm}_{step}/{hand}_ep{lo}-{hi}", arm,
                         ckpt_dir(arm, step) / "pretrained_model" if kind != "r0" else CV / "r0/no_ckpt", hand, lo, hi,
                         "artifacts/pooled3hand/p0a/f_task_best.pt",
                         arms()[arm]["dataset"] if kind != "r0" else "unused")
        self.step_, self.kind, self.seed, self.codec = step, kind, None, codec
        if kind == "r0":
            self.ckpt = None

    def done(self):
        if self.kind != "r0":
            return super().done()
        m = self.metrics()
        return bool(m) and sorted(e["episode"] for e in m["episodes"]) == self.eps and m.get("replay_npz") == self.npz()

    def npz(self):
        return f"{CCV}/r0/replay_{self.codec}.npz"

    def command(self):
        if self.kind == "r0":
            src = f"--replay_npz {self.npz()} --codec_ckpt {R0_CODECS[self.codec]}"
        else:
            src = f"--ckpt {self.ckpt} --codec_ckpt {arms()[self.arm]['codec_ckpt']} --dataset {self.dataset}"
        inner = ("export DEXVERSE_ROOT=/workspace/DexVerse OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "
                 "TZ=Asia/Seoul; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS DEXVERSE_OBJECT_MATERIAL "
                 "DEXVERSE_DUMMY_LINK_MASS; cd /workspace/DexVerse; /workspace/isaaclab/_isaac_sim/python.sh "
                 f"{L.EVAL} --arm Z --demos {DEMOS[self.hand]} {src} --f_task {self.f_task} --output_dir {self.cout} "
                 f"--episodes {','.join(map(str, self.eps))} --max_steps {args.max_steps} --no-video --headless")
        return ["flock", "-s", "/tmp/isaac_gpu.lock", "bash", "-c",
                'echo "[job] LOCKED $(date +%F_%T)"; exec docker exec xhand bash -c "$0"', inner]


def read(p):
    try:
        return Path(p).read_text(errors="ignore")
    except OSError:
        return ""


def proc_alive(pattern):
    return L.sh(["pgrep", "-f", pattern], timeout=60).returncode == 0


def higher_priority_busy():
    """None if cv3dp and pi0.5 leave work2 to cv3hm, else the reason."""
    if (CV / "ctl/DRAIN").exists() or (CV / "ctl/HOLD").exists():
        return "ctl/DRAIN or ctl/HOLD"
    cv3 = read(L.HOST_LC / "artifacts/codec_v3/cv3dp/logs/driver_work2.log")
    if proc_alive("[c]v3dp_driver.py") and "[cv3dp] ALL DONE" not in cv3 and "[cv3dp] FAIL" not in cv3:
        return "cv3dp driver alive on work2 (priority)"
    pi = read(L.HOST_LC / "artifacts/pooled3hand/pi05/logs/driver.log")
    if proc_alive("[p]i05_driver.py") and "[pi05] ALL DONE" not in pi and "[pi05] FAIL" not in pi:
        return "pi0.5 driver alive on work2 (priority)"
    return None


class Runner(L.Runner):
    def step(self, todo):
        why = higher_priority_busy()
        if why:
            if why != getattr(self, "_last_block", None):
                log(f"launch hold: {why}")
            self._last_block = why
            return
        if getattr(self, "_last_block", None):
            log("launch hold released")
        self._last_block = None
        super().step(todo)

    def write_machine(self, job, n):
        mj = {"machine": "work2", "gpu": self.gpu, "workers": self.parallel, "attempt": n, "launched": L.now(),
              "job": job.name, "codec_ckpt": (R0_CODECS[job.codec] if job.kind == "r0" else arms()[job.arm]["codec_ckpt"]),
              "max_steps": args.max_steps}
        L.sh(["docker", "exec", "-i", "xhand", "bash", "-c", f"mkdir -p {job.cout} && cat > {job.cout}/machine.json"],
             input=json.dumps(mj) + "\n", timeout=60)


def r0_jobs():
    if not args.r0:
        return []
    return [Job("r0", "R0", c, h, 0, 9, codec=c) for c in R0_CODECS for h in TRAIN_HANDS]


def sel_jobs(arm):
    return [Job("sel", arm, s, h, 0, 9) for s in steps(arm)[:N_SEL] for h in TRAIN_HANDS]


def best_of(arm):
    p = CV / "sel" / f"{arm}_best.json"
    return json.loads(p.read_text()) if p.exists() else None


def final_jobs(arm, step):
    js = [Job("final", arm, step, h, lo, hi) for h in TEST_HANDS for lo, hi in L.CHUNKS]
    if args.train_ref:
        js += [Job("final", arm, step, h, lo, hi) for h in TRAIN_HANDS for lo, hi in L.CHUNKS[1:]]
    return js


def update_selection():
    for arm in arms():
        if best_of(arm):
            continue
        js = sel_jobs(arm)
        if not all(j.done() for j in js):
            continue
        S = steps(arm)[:N_SEL]
        per = {s: {h: next(j for j in js if j.step_ == s and j.hand == h).successes() for h in TRAIN_HANDS} for s in S}
        tot = {s: sum(per[s].values()) for s in S}
        best = max(S, key=lambda s: (tot[s], int(s)))
        rec = {"arm": arm, "best": best,
               "rule": "max 3-hand (shadow+wuji+allegro) total /30 over epochs 50..250, ties -> later",
               "totals": tot, "per_hand": per, "time": L.now(), "host": "work2", "max_steps": args.max_steps}
        (CV / "sel").mkdir(parents=True, exist_ok=True)
        (CV / "sel" / f"{arm}_best.json").write_text(json.dumps(rec, indent=1))
        log(f"[cv3hm] SELECT {arm} done best={best} totals " + " ".join(f"{s}:{tot[s]}/30" for s in S)
            + " per_hand " + json.dumps(per[best]))


logged = set()


def plan():
    js = list(r0_jobs())
    if js and "r0" not in logged and all(j.done() for j in js):
        logged.add("r0")
        log("[cv3hm] R0 done " + " ".join(f"{j.codec}/{j.hand}:{j.successes()}/10" for j in js))
    A = list(arms())
    for arm in A:
        js += sel_jobs(arm)
        b = best_of(arm)
        if b:
            fj = final_jobs(arm, b["best"])
            js += fj
            sh = [j for j in fj if j.hand in TEST_HANDS]
            if arm not in logged and all(j.done() for j in sh):
                logged.add(arm)
                log(f"[cv3hm] FINAL {arm} held-out done sharpa "
                    f"{sum(j.successes() for j in sh if j.hand == 'sharpa')}/50 leap "
                    f"{sum(j.successes() for j in sh if j.hand == 'leap')}/50 (best {b['best']})")
    complete = bool(A) and all(best_of(a) for a in A) and all(j.done() for j in js)
    return js, complete


def prio(j):
    a = list(arms())
    k = {"r0": 0, "sel": 1}.get(j.kind, 2 if j.hand in TEST_HANDS else 3)
    return (k, int(j.step_) if k == 1 else 0, a.index(j.arm) if j.arm in a else -1, j.codec or "", j.lo, j.hand)


def main():
    (CV / "logs").mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print("arms", arms(), "higher_priority_busy", higher_priority_busy())
        for j in sorted(plan()[0], key=prio):
            print(j.name, j.done(), ready(j.arm, j.step_), j.command()[-1][-330:])
        return
    lock = open(CV / "logs/driver_work2.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("cv3hm driver already running")
        return
    (CV / "logs/driver_work2.pid").write_text(str(os.getpid()))
    log(f"driver start pid {os.getpid()} parallel {args.parallel} arms {list(arms())} train_ref {args.train_ref} "
        f"r0 {args.r0} max_steps {args.max_steps}")
    log("[cv3hm] GATE open (launches still wait for cv3dp/pi0.5 priority, see launch hold lines)")
    R = Runner("cv3hm", args.parallel, log)
    last_beat = 0.0
    while True:
        R.reap()
        update_selection()
        js, complete = plan()
        if complete and not R.running:
            log(f"[cv3hm] ALL DONE {len(js)} jobs")
            return
        stuck = [j for j in js if not j.done()]
        if stuck and all(j.name in R.failed for j in stuck) and not R.running:
            log(f"[cv3hm] FAIL {sorted(R.failed)}")
            return
        todo = sorted((j for j in js if not j.done() and ready(j.arm, j.step_)), key=prio)
        R.step(todo)
        if time.time() - last_beat > 1800:
            last_beat = time.time()
            log(f"heartbeat running={sorted(R.running)} done={sum(j.done() for j in js)}/{len(js)} "
                f"ready={sorted({(j.arm, j.step_) for j in js if ready(j.arm, j.step_) and j.kind != 'r0'})} "
                f"failed={sorted(R.failed)}")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
