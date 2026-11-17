#!/usr/bin/env python3
"""CV3DP eval driver (261116; runs on work1 OR work2, --host): 3-hand (shadow + wuji + allegro) nohot GraspKettle image DP,
arms from ctl/arms.json (Zw3 = codec-v3 wb_cl_l z, Ze3 = E2* z), trained on vast (runs/cv3dp/<arm>).
A3 (261117): arms.json may set "eval_arm" (default "Z"); Dw3/De3 use "D" (EE_common wrist, trained on vast runs/cv3dpD/<arm>).
Copy of cv2dp_driver.py (unchanged original) adapted to the CV3DP PREREG (artifacts/codec_v3/cv3dp/PREREG.md). Host python3
stdlib; Isaac jobs via ho_lib.Runner (unchanged module; its path globals are pointed at artifacts/codec_v3/cv3dp) running
cv3dp_eval_pooled.py (= cv2dp_eval_pooled.py + --seed) in the xhand container.
  ckpt   : work2: read in place from artifacts/codec_v3/cv3dp/ckpt_archive/<arm>/<step> (cv3dp_pull_w2.sh, sha256-checked).
           work1: ctl/WANT.json {arm, step} -> the PC relay (C:/Users/eunyo/Documents/p3h_relay/cv3dp_relay_pc.sh) copies it
           from the work2 archive into ckpt/<arm>/<step> (.complete); at most --max_local kept, unneeded ones deleted.
  gate   : ctl/EVAL_GATE_OPEN (written by the PC relay on the ONE host it picks, or by hand). Meanwhile the driver writes
           ctl/GATE_READY when its host is free: work2 = '[ho_h2] ALL DONE'/FAIL and the pi0.5 driver finished
           ('[pi05] ALL DONE'/FAIL or no pi05_driver.py) and no eval alive in the container; work1 = '[cv2dp] ALL DONE'/FAIL
           and no cv2dp_driver.py and no eval alive. ctl/DISABLED (the other host was picked) -> exit.
  select : per arm, shadow + wuji + allegro x eps 0-9 x snapshots of epochs 50..250; best = max 3-hand total /30, ties ->
           later step -> sel/<arm>_best.json, '[cv3dp] SELECT <arm> done best=...'.  Held-out hands never used.
  final  : best snapshot on sharpa and leap eps 0-49 (5 chunks each, cv2dp protocol; 261116 amendment: leap 50-ep pkl
           replaces the earlier eps 0-9 x --seed 1..5 plan, which never ran), then the
           training-hand reference (shadow, wuji, allegro eps 10-49; eps 0-9 = selection rows of the best step).
Markers in artifacts/codec_v3/cv3dp/logs/driver_<host>.log: GATE READY | [cv3dp] GATE open | SELECT <arm> | FINAL <arm> |
  FAIL ... | ALL DONE.
A4 (261117, operational only: no job, episode range, max_steps or selection-rule change) - two hosts at once:
  split  : ctl/SPLIT.json {"default": "work2", "work1": [fnmatch patterns on job names]} (read live, identical on both hosts)
           decides which host runs each job; without SPLIT.json every job is this host's (pre-A4 behaviour). A host launches
           only its own jobs; <arm>_best.json is written only by the host that owns all sel jobs of the arm. The default
           host's ALL DONE still needs every job done (the other host's results arrive by the PC mirror); the other host's
           ALL DONE needs its own jobs done and every arm selected. ctl/PEER_BUSY.txt (job names the other host already ran
           or runs, written by the PC relay) are never launched here.
  ckpt   : both hosts read artifacts/codec_v3/cv3dp/ckpt_archive/<arm>/<step> so metrics.json "ckpt" is the same string on
           both (work1: a local cache of the sha256-identical work2 archive, filled by the PC relay from ctl/WANT.json,
           pruned to --max_local); ckpt/ is no longer used.
  mirror : C:/Users/eunyo/Documents/p3h_relay/cv3dp_split_relay_pc.sh copies finished work1 job dirs into the same paths on
           work2 (so the summary reads one tree) and work2 sel/<arm>_best.json to work1; --mirror-status prints the JSON it uses.
usage: nohup python3 cv3dp_driver.py --host work2 --parallel 3 >> artifacts/codec_v3/cv3dp/logs/driver_nohup_<host>.out 2>&1 &
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ho_lib as L  # noqa: E402

CV = L.HOST_LC / "artifacts/codec_v3/cv3dp"
CCV = f"{L.CONT_LC}/artifacts/codec_v3/cv3dp"
# point the unchanged ho_lib Job/Runner at our folders and eval script
L.HO, L.CHO = CV, CCV
L.EVAL = f"{L.CONT_LC}/xhand_latent/cv3dp_eval_pooled.py"
L.EVAL_PATTERNS = ("[c]v3dp_eval_pooled.py", "[c]v2dp_eval_pooled.py", "[h]m_eval_masked.py", "[h]o_eval_pooled.py",
                   "[p]3h_nh_eval_pooled.py", "[p]3h_eval_pooled.py", "[p]i05_eval_pooled.py")
TRAIN_HANDS = ("shadow", "wuji", "allegro")
TEST_HANDS = ("sharpa", "leap")  # held-out, eps 0-49 x 1 run each (cv2dp protocol)
EPOCHS = (50, 100, 150, 200, 250, 300)
N_SEL = 5
# leap: 50-ep pkl (dv-demos-v0 ep000-049, user-approved download 261116; eps 0-9 == the old _ep0-9.pkl)
DEMOS = {h: f"{CCV}/demos/{L.TASK}_floating_{h}_right.pkl" for h in ("shadow", "wuji", "allegro", "sharpa", "leap")}

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True, choices=("work1", "work2"))
ap.add_argument("--parallel", type=int, default=3)
ap.add_argument("--poll", type=int, default=60)
ap.add_argument("--max_local", type=int, default=3)
ap.add_argument("--min_free_gb", type=float, default=3.0)
ap.add_argument("--no-train-ref", dest="train_ref", action="store_false")
ap.add_argument("--max_steps", type=int, default=1500)  # A2 261117: 3000 -> 1500
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--mirror-status", action="store_true", help="A4: print the split/mirror state as JSON and exit (PC relay)")
args = ap.parse_args()
LOG = CV / f"logs/driver_{args.host}.log"


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{L.now()} {msg}\n")


def arms():
    """ctl/arms.json: {arm: {codec_ckpt, dataset, spe}}"""
    try:
        return json.loads((CV / "ctl/arms.json").read_text())
    except (OSError, ValueError):
        return {}


def steps(arm):
    spe = arms()[arm]["spe"]
    return tuple(f"{spe * e:06d}" for e in EPOCHS)


def ckpt_dir(arm, step):
    return CV / "ckpt_archive" / arm / step  # A4: same path on both hosts (work1 = relay-filled local cache)


def split():
    """A4: ctl/SPLIT.json or None (no split: every job is this host's)."""
    try:
        return json.loads((CV / "ctl/SPLIT.json").read_text())
    except (OSError, ValueError):
        return None


def owner(name, sp=None):
    import fnmatch
    sp = split() if sp is None else sp
    if not sp:
        return args.host
    for host, pats in sp.items():
        if host in ("default", "note") or not isinstance(pats, list):
            continue
        if any(fnmatch.fnmatchcase(name, p) for p in pats):
            return host
    return sp.get("default", "work2")


def default_host():
    sp = split()
    return sp.get("default", "work2") if sp else args.host


def peer_busy():
    try:
        return {s.strip() for s in (CV / "ctl/PEER_BUSY.txt").read_text().splitlines() if s.strip()}
    except OSError:
        return set()


def ready(arm, step):
    return (ckpt_dir(arm, step) / ".complete").exists()


class Job(L.Job):
    def __init__(self, kind, arm, step, hand, lo, hi, seed=None):
        tag = hand if seed is None else f"{hand}_s{seed}"
        super().__init__(f"{kind}/{arm}_{step}/{tag}_ep{lo}-{hi}", arm, ckpt_dir(arm, step) / "pretrained_model", hand, lo,
                         hi, "artifacts/pooled3hand/p0a/f_task_best.pt", arms()[arm]["dataset"])
        self.step_, self.kind, self.seed = step, kind, seed

    def command(self):
        a = arms()[self.arm]
        seed = "" if self.seed is None else f"--seed {self.seed} "
        inner = ("export DEXVERSE_ROOT=/workspace/DexVerse OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "
                 "TZ=Asia/Seoul; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS DEXVERSE_OBJECT_MATERIAL "
                 "DEXVERSE_DUMMY_LINK_MASS; cd /workspace/DexVerse; /workspace/isaaclab/_isaac_sim/python.sh "
                 f"{L.EVAL} --arm {a.get('eval_arm', 'Z')} --demos {DEMOS[self.hand]} --ckpt {self.ckpt} {seed}"
                 f"--codec_ckpt {a['codec_ckpt']} --f_task {self.f_task} --dataset {self.dataset} --output_dir {self.cout} "
                 f"--episodes {','.join(map(str, self.eps))} --max_steps {args.max_steps} --no-video --headless")
        return ["flock", "-s", "/tmp/isaac_gpu.lock", "bash", "-c",
                'echo "[job] LOCKED $(date +%F_%T)"; exec docker exec xhand bash -c "$0"', inner]


def launch_block():
    """261117 (coordinator): work2 priority H2 -> pi0.5 -> cv3dp. No NEW cv3dp job while ctl/DRAIN exists, or (work2) while the
    pi0.5 driver is alive, not finished and has no policy server up (it needs >= 15 GB free to start one); otherwise only
    the slots left under the shared --parallel budget (ho_lib counts pi05_eval_pooled.py too) and GPU free >= 7.8 GB."""
    if (CV / "ctl/DRAIN").exists():
        return "ctl/DRAIN"
    if args.host == "work2" and not (CV / "ctl/IGNORE_PI05").exists() and proc_alive("[p]i05_driver.py"):
        pi = read(L.HOST_LC / "artifacts/pooled3hand/pi05/logs/driver.log")
        if "[pi05] ALL DONE" not in pi and "[pi05] FAIL" not in pi and L.sh(
                ["docker", "exec", "xhand", "pgrep", "-f", "[p]i05_policy_server.py --shared"], timeout=60).returncode != 0:
            return "pi0.5 driver has no server up (priority: let it start one)"
        # pi0.5 first claim on a free slot: hold while its last SERVER up / START / DONE / EXIT is < 200 s old
        # (its LAUNCH_GAP is 150 s), so cv3dp only takes a slot pi0.5 did not take when it could.
        import re
        from datetime import datetime
        ev = re.findall(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) KST (?:SERVER up|START|DONE|EXIT) ", pi, re.M)
        if ev:
            age = (datetime.strptime(L.now()[:19], "%Y-%m-%d %H:%M:%S") - datetime.strptime(ev[-1], "%Y-%m-%d %H:%M:%S")).total_seconds()
            if age < 200:
                return "pi0.5 launched/finished a job < 200 s ago (its claim first)"
    return None


class Runner(L.Runner):
    def step(self, todo):
        why = launch_block()
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
        mj = {"machine": args.host, "gpu": self.gpu, "workers": self.parallel, "attempt": n, "launched": L.now(),
              "job": job.name, "codec_ckpt": arms()[job.arm]["codec_ckpt"], "seed": job.seed}
        L.sh(["docker", "exec", "-i", "xhand", "bash", "-c", f"mkdir -p {job.cout} && cat > {job.cout}/machine.json"],
             input=json.dumps(mj) + "\n", timeout=60)


# ----------------------------------------------------------------------------- gate
def read(p):
    try:
        return Path(p).read_text(errors="ignore")
    except OSError:
        return ""


def proc_alive(pattern):
    return L.sh(["pgrep", "-f", pattern], timeout=60).returncode == 0


def host_free():
    if any(L.container_evals().values()):
        return False, "an eval is alive in the container"
    if args.host == "work2":
        h2 = read(L.HOST_LC / "artifacts/pooled3hand/ho/logs/h2_driver.log")
        if "[ho_h2] ALL DONE" not in h2 and "[ho_h2] FAIL" not in h2:
            return False, "H2 not done"
        pi = read(L.HOST_LC / "artifacts/pooled3hand/pi05/logs/driver.log")
        if "[pi05] ALL DONE" not in pi and "[pi05] FAIL" not in pi and proc_alive("[p]i05_driver.py"):
            return False, "pi0.5 eval not done"
        return True, "H2 and pi0.5 done"
    cv2 = read(L.HOST_LC / "artifacts/cv2dp/logs/driver.log")
    if "[cv2dp] ALL DONE" not in cv2 and "[cv2dp] FAIL" not in cv2:
        return False, "cv2dp not done"
    if proc_alive("[c]v2dp_driver.py"):
        return False, "cv2dp driver alive"
    return True, "cv2dp done"


# ----------------------------------------------------------------------------- ckpt (work1: PC relay)
def free_gb():
    return shutil.disk_usage(str(CV)).free / 1e9


def local_snaps():
    return {(p.parent.name, p.name) for p in CV.glob("ckpt_archive/*/[0-9]*") if p.is_dir()}


def manage_ckpts(js, R):
    if args.host == "work2":  # archive read in place, never deleted
        return
    pb = peer_busy()
    pending = [j for j in js if not j.done() and j.name not in R.failed and owner(j.name) == args.host and j.name not in pb]
    need = {(j.arm, j.step_) for j in pending}
    in_use = {(j.arm, j.step_) for j, *_ in R.running.values()}
    for key in local_snaps():
        if key not in need and key not in in_use:
            shutil.rmtree(ckpt_dir(*key), ignore_errors=True)
            log(f"[ckpt] removed local {key[0]} {key[1]} (no pending job)")
    order = []
    for j in pending:
        k = (j.arm, j.step_)
        if k not in order:
            order.append(k)
    have = {k for k in local_snaps() if ready(*k)}
    want = [k for k in order if k not in have]
    if not want or len(local_snaps()) >= args.max_local or free_gb() < args.min_free_gb + 1.2:
        return
    k = want[0]
    wp = CV / "ctl/WANT.json"
    try:
        old = json.loads(wp.read_text())
    except (OSError, ValueError):
        old = {}
    if (old.get("arm"), old.get("step")) != k:
        wp.write_text(json.dumps({"arm": k[0], "step": k[1], "time": L.now()}) + "\n")
        log(f"[pull] relay WANT {k[0]} {k[1]}")


# ----------------------------------------------------------------------------- plan
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
        if any(owner(j.name) != args.host for j in js):  # A4: only the host owning every sel job of the arm selects
            continue
        if not all(j.done() for j in js):
            continue
        S = steps(arm)[:N_SEL]
        per = {s: {h: next(j for j in js if j.step_ == s and j.hand == h).successes() for h in TRAIN_HANDS} for s in S}
        tot = {s: sum(per[s].values()) for s in S}
        best = max(S, key=lambda s: (tot[s], int(s)))
        rec = {"arm": arm, "best": best,
               "rule": "max 3-hand (shadow+wuji+allegro) total /30 over epochs 50..250, ties -> later",
               "totals": tot, "per_hand": per, "time": L.now(), "host": args.host}
        (CV / "sel").mkdir(parents=True, exist_ok=True)
        (CV / "sel" / f"{arm}_best.json").write_text(json.dumps(rec, indent=1))
        log(f"[cv3dp] SELECT {arm} done best={best} totals " + " ".join(f"{s}:{tot[s]}/30" for s in S)
            + " per_hand " + json.dumps(per[best]))


final_logged = set()


def plan():
    js = []
    A = list(arms())
    for arm in A:
        js += sel_jobs(arm)
        b = best_of(arm)
        if b:
            fj = final_jobs(arm, b["best"])
            js += fj
            sh = [j for j in fj if j.hand in TEST_HANDS]
            if arm not in final_logged and all(j.done() for j in sh):
                final_logged.add(arm)
                log(f"[cv3dp] FINAL {arm} held-out done sharpa "
                    f"{sum(j.successes() for j in sh if j.hand == 'sharpa')}/50 leap "
                    f"{sum(j.successes() for j in sh if j.hand == 'leap')}/50 (best {b['best']})")
    dh = default_host() == args.host  # A4: the default host waits for every job, the other host for its own
    complete = bool(A) and all(best_of(a) for a in A) and all(j.done() for j in js if dh or owner(j.name) == args.host)
    return js, complete


def prio(j):
    a = list(arms())
    k = 0 if j.kind == "sel" else (1 if j.hand in TEST_HANDS else 2)
    return (k, int(j.step_) if k == 0 else 0, a.index(j.arm), j.lo, j.seed or 0, j.hand)


def all_jobs():
    """every planned job (no logging, no selection writes)."""
    js = []
    for arm in arms():
        js += sel_jobs(arm)
        b = best_of(arm)
        if b:
            js += final_jobs(arm, b["best"])
    return js


def ran_here(j):
    try:
        return json.loads((j.out / "machine.json").read_text()).get("machine") == args.host
    except (OSError, ValueError):
        return False


def mirror_status():
    """A4: JSON for the PC relay (mirror + work2 DRAIN guard). Read-only."""
    sp = split()
    js = all_jobs()
    running = set().union(*L.container_evals().values())
    pb = peer_busy()
    mine = lambda j: owner(j.name, sp) == args.host  # noqa: E731
    st = {"host": args.host, "split": sp, "time": L.now(),
          "done_own": sorted(j.name for j in js if mine(j) and j.done() and ran_here(j)),
          "best_own": sorted(a for a in arms() if (best_of(a) or {}).get("host") == args.host),
          "foreign_local": sorted(j.name for j in js if not mine(j) and (j.cout in running or ran_here(j))),
          "running": sorted(j.name for j in js if j.cout in running),
          "own_ready_pending": sorted(j.name for j in js if mine(j) and not j.done() and ready(j.arm, j.step_)
                                      and j.cout not in running and j.name not in pb),
          "foreign_ready_pending": sorted(j.name for j in js if not mine(j) and not j.done() and ready(j.arm, j.step_)
                                          and j.cout not in running),
          "started_own": sorted(j.name for j in js if mine(j) and (j.cout in running or ran_here(j))),
          "n_jobs": len(js), "n_done": sum(j.done() for j in js)}
    # what a split-unaware (pre-A4) driver on this host would launch next: lowest prio among ready, not done, not running
    nxt = sorted((j for j in js if not j.done() and ready(j.arm, j.step_) and j.cout not in running), key=prio)
    st["next_pick"] = nxt[0].name if nxt else None
    print(json.dumps(st))


def main():
    (CV / "logs").mkdir(parents=True, exist_ok=True)
    if args.mirror_status:
        mirror_status()
        return
    if args.dry_run:
        print("arms", arms(), "gate", (CV / "ctl/EVAL_GATE_OPEN").exists(), "host_free", host_free(),
              "local", sorted(local_snaps()), "split", split(), "peer_busy", sorted(peer_busy()))
        for j in sorted(all_jobs(), key=prio):
            print(j.name, "owner", owner(j.name), j.done(), ready(j.arm, j.step_), j.command()[-1][-300:])
        return
    lock = open(CV / f"logs/driver_{args.host}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("cv3dp driver already running")
        return
    (CV / f"logs/driver_{args.host}.pid").write_text(str(os.getpid()))
    log(f"driver start pid {os.getpid()} host {args.host} parallel {args.parallel} arms {list(arms())} "
        f"train_ref {args.train_ref} max_steps {args.max_steps} A4split {json.dumps(split())}")
    R = Runner("cv3dp", args.parallel, log)
    started, last_beat, last_why = False, 0.0, ""
    while True:
        if not started:
            if (CV / "ctl/DISABLED").exists():
                log("[cv3dp] ctl/DISABLED (the other host was picked) -> exit")
                return
            free, why = host_free()
            if free and not (CV / "ctl/GATE_READY").exists():
                (CV / "ctl/GATE_READY").write_text(f"{L.now()} {why}\n")
                log(f"GATE READY ({why})")
            elif not free and why != last_why:
                (CV / "ctl/GATE_READY").unlink(missing_ok=True)
                log(f"waiting: {why}")
            last_why = why
            g = CV / "ctl/EVAL_GATE_OPEN"  # text 'force' = start even if the host check says busy (by hand only)
            if not (g.exists() and (free or "force" in read(g))):
                time.sleep(args.poll)
                continue
            started = True
            log("[cv3dp] GATE open")
        R.reap()
        update_selection()
        js, complete = plan()
        manage_ckpts(js, R)
        if complete and not R.running:
            log(f"[cv3dp] ALL DONE {len(js)} jobs")
            return
        pb = peer_busy()
        own = [j for j in js if owner(j.name) == args.host and j.name not in pb]  # A4
        stuck = [j for j in own if not j.done()]
        if stuck and all(j.name in R.failed for j in stuck) and not R.running:
            log(f"[cv3dp] FAIL {sorted(R.failed)}")
            return
        # A4: the non-default host takes its jobs from the far end, so a split-unaware default-host driver that steals
        # idle-time work in prio order meets it only at the last jobs (the PC relay drains it there)
        todo = sorted((j for j in own if not j.done() and ready(j.arm, j.step_)), key=prio,
                      reverse=default_host() != args.host)
        R.step(todo)
        if time.time() - last_beat > 1800:
            last_beat = time.time()
            log(f"heartbeat running={sorted(R.running)} done={sum(j.done() for j in js)}/{len(js)} "
                f"ready={sorted({(j.arm, j.step_) for j in js if ready(j.arm, j.step_)})} free={free_gb():.1f}GB "
                f"failed={sorted(R.failed)}")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
