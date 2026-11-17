#!/usr/bin/env python3
"""CV2DP eval driver (261115, work1): 2-hand (shadow + wuji) nohot GraspKettle image DP, arms from ctl/arms.json
(Ze2 = E2* z, Zv2 = codec-v2 z, optional Zv2m = Zv2 + hand-masked images), trained on vast (runs/cv2dp/<arm>).
Host python3 stdlib; Isaac jobs via ho_lib.Runner (unchanged module; its path globals are pointed at artifacts/cv2dp)
running cv2dp_eval_pooled.py (= ho_eval_pooled.py + --codec_ckpt) in the xhand container.
  pull   : on demand from vast (key ~/.ssh/p3h_vast_ed25519) -> artifacts/cv2dp/ckpt/<arm>/<step>/pretrained_model
           (tar stream, sha256 of every file compared, .complete). work1 disk is tight: at most MAX_LOCAL snapshots are
           kept locally; a snapshot no pending job needs is deleted.
  gate   : evaluation starts when ctl/EVAL_GATE_OPEN exists (created by hand once the codec-v2 agent left the GPU).
  select : per arm, shadow + wuji x eps 0-9 x snapshots of epochs 50..250; best = max 2-hand total /20, ties -> later
           step -> sel/<arm>_best.json, '[cv2dp] SELECT <arm> done best=...'.
  final  : best snapshot on sharpa (unseen by the policy) eps 0-49 (5 chunks); then the training-hand reference
           (shadow, wuji eps 10-49; eps 0-9 = selection rows of the best step).
Markers in artifacts/cv2dp/logs/driver.log: [cv2dp] GATE open | SELECT <arm> | FINAL <arm> | FAIL ... | ALL DONE
--pull_source relay (after vast is freed): instead of ssh to vast, the driver writes ctl/WANT.json {arm, step} for the
next needed snapshot and waits for .complete; the PC relay C:/Users/eunyo/Documents/p3h_relay/cv2dp_relay_pc.sh copies it
from work2 artifacts/cv2dp/ckpt_archive (sha256-verified against vast at archive time; compared again after the copy).
usage (work1): nohup python3 cv2dp_driver.py --parallel 3 >> artifacts/cv2dp/logs/driver_nohup.out 2>&1 &
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ho_lib as L  # noqa: E402

CV = L.HOST_LC / "artifacts/cv2dp"
CCV = f"{L.CONT_LC}/artifacts/cv2dp"
# point the unchanged ho_lib Job/Runner at our folders and eval script
L.HO, L.CHO = CV, CCV
L.EVAL = f"{L.CONT_LC}/xhand_latent/cv2dp_eval_pooled.py"
L.EVAL_PATTERNS = ("[c]v2dp_eval_pooled.py", "[h]m_eval_masked.py", "[h]o_eval_pooled.py", "[p]3h_nh_eval_pooled.py", "[p]3h_eval_pooled.py")
TRAIN_HANDS = ("shadow", "wuji")
TEST_HAND = "sharpa"
EPOCHS = (50, 100, 150, 200, 250, 300)
N_SEL = 5
VAST_RUNS = "/workspace/xhand/runs/cv2dp"
DEMOS = f"{L.CONT_LC}/artifacts/pooled3hand/p0b/demos/{L.TASK}_floating_{{hand}}_right.pkl"

ap = argparse.ArgumentParser()
ap.add_argument("--parallel", type=int, default=3)
ap.add_argument("--poll", type=int, default=60)
ap.add_argument("--max_local", type=int, default=3)
ap.add_argument("--min_free_gb", type=float, default=3.0)
ap.add_argument("--vast", default="root@<VAST_HOST>:<VAST_PORT>")
ap.add_argument("--no-train-ref", dest="train_ref", action="store_false")
ap.add_argument("--pull_source", choices=("vast", "relay"), default="vast",
                help="relay: write ctl/WANT.json and wait for the PC relay (cv2dp_relay_pc.sh) to place the snapshot")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
LOG = CV / "logs/driver.log"
USER, HP = args.vast.split(":")
VAST = ["ssh", "-i", os.path.expanduser("~/.ssh/p3h_vast_ed25519"), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", "-p", HP, USER]


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{L.now()} {msg}\n")


def arms():
    """ctl/arms.json: {arm: {codec_ckpt, dataset, spe, extra: [args right after the script], script (optional,
    e.g. hm_eval_masked.py for Zv2m; default cv2dp_eval_pooled.py)}} (re-read every loop: Zv2m is added later)."""
    try:
        return json.loads((CV / "ctl/arms.json").read_text())
    except (OSError, ValueError):
        return {}


def steps(arm):
    spe = arms()[arm]["spe"]
    return tuple(f"{spe * e:06d}" for e in EPOCHS)


def ckpt_dir(arm, step):
    return CV / "ckpt" / arm / step


def ready(arm, step):
    return (ckpt_dir(arm, step) / ".complete").exists()


class Job(L.Job):
    def __init__(self, kind, arm, step, hand, lo, hi):
        super().__init__(f"{kind}/{arm}_{step}/{hand}_ep{lo}-{hi}", arm, ckpt_dir(arm, step) / "pretrained_model", hand, lo,
                         hi, "artifacts/pooled3hand/p0a/f_task_best.pt", arms()[arm]["dataset"])
        self.step_, self.kind = step, kind

    def command(self):
        a = arms()[self.arm]
        extra = " ".join(a.get("extra", []))
        inner = ("export DEXVERSE_ROOT=/workspace/DexVerse OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "
                 "TZ=Asia/Seoul; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS DEXVERSE_OBJECT_MATERIAL "
                 "DEXVERSE_DUMMY_LINK_MASS; cd /workspace/DexVerse; /workspace/isaaclab/_isaac_sim/python.sh "
                 f"{a.get('script', L.EVAL)} {extra} --arm Z --demos {DEMOS.format(hand=self.hand)} --ckpt {self.ckpt} "
                 f"--codec_ckpt {a['codec_ckpt']} --f_task {self.f_task} --dataset {self.dataset} --output_dir {self.cout} "
                 f"--episodes {','.join(map(str, self.eps))} --max_steps 3000 --no-video --headless")
        return ["flock", "-s", "/tmp/isaac_gpu.lock", "bash", "-c",
                'echo "[job] LOCKED $(date +%F_%T)"; exec docker exec xhand bash -c "$0"', inner]


class Runner(L.Runner):
    def write_machine(self, job, n):
        mj = {"machine": "work1", "gpu": self.gpu, "workers": self.parallel, "attempt": n, "launched": L.now(),
              "job": job.name, "codec_ckpt": arms()[job.arm]["codec_ckpt"], "extra": arms()[job.arm].get("extra", [])}
        L.sh(["docker", "exec", "-i", "xhand", "bash", "-c", f"mkdir -p {job.cout} && cat > {job.cout}/machine.json"],
             input=json.dumps(mj) + "\n", timeout=60)


# ----------------------------------------------------------------------------- pull (vast -> work1, on demand)
pull_busy, remote = [False], {}
gave_up = set()


def remote_scan():
    cmd = (f"for f in {VAST_RUNS}/*/train/checkpoints/[0-9]*/pretrained_model/model.safetensors; do "
           "d=$(dirname $f); test -f $d/train_config.json || continue; "
           "echo $f $(( $(date +%s) - $(stat -c %Y $f) )) $(stat -c %s $f); done; "
           f"grep -ho '\\[cv2dp_vast\\] [A-Za-z0-9]*: GIVING UP' {VAST_RUNS}/logs/*.log 2>/dev/null; true")
    r = L.sh(VAST + [cmd], timeout=120)
    if r.returncode != 0:
        log(f"[pull] vast unreachable this pass: {r.stderr.strip()[-150:]}")
        return
    out = {}
    for line in r.stdout.splitlines():
        if "GIVING UP" in line:
            arm = line.split("]")[1].split(":")[0].strip()
            if arm not in gave_up:
                gave_up.add(arm)
                log(f"[cv2dp] vast training {arm} GAVE UP")
            continue
        p = line.split()
        if len(p) == 3 and p[1].isdigit() and int(p[1]) >= 180:
            parts = p[0].split("/")
            out[(parts[-6], parts[-3])] = int(p[2])
    remote.clear()
    remote.update(out)


def free_gb():
    return shutil.disk_usage(str(CV)).free / 1e9


def pull_one(arm, step):
    rd = f"{VAST_RUNS}/{arm}/train/checkpoints/{step}"
    d = ckpt_dir(arm, step)
    tmp = d.parent / f".{step}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t0 = time.time()
    tar = subprocess.Popen(VAST + [f"tar -C {rd} -cf - pretrained_model"], stdout=subprocess.PIPE)
    r = subprocess.run(["tar", "-xf", "-", "-C", str(tmp)], stdin=tar.stdout, timeout=3600)
    tar.wait(timeout=60)
    if r.returncode or tar.returncode:
        log(f"[pull] {arm} {step} tar failed rc={r.returncode}/{tar.returncode}")
        shutil.rmtree(tmp, ignore_errors=True)
        return
    rs = L.sh(VAST + [f"cd {rd}/pretrained_model && sha256sum *"], timeout=600).stdout.split()
    ls = L.sh(["bash", "-c", f"cd {tmp}/pretrained_model && sha256sum *"]).stdout.split()
    if not rs or sorted(rs) != sorted(ls):
        log(f"[pull] {arm} {step} sha256 mismatch, will retry")
        shutil.rmtree(tmp, ignore_errors=True)
        return
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    (d / ".complete").write_text(f"pulled {L.now()} from vast {rd} sha256 ok\n")
    log(f"[pull] {arm} {step} OK in {time.time() - t0:.0f}s, disk free {free_gb():.1f} GB")


def local_snaps():
    return {(p.parent.name, p.name) for p in CV.glob("ckpt/*/[0-9]*") if p.is_dir()}


def manage_ckpts(js, R):
    """delete local snapshots no pending job needs; pull the next needed one (priority order) within the budget."""
    pending = [j for j in js if not j.done() and j.name not in R.failed]
    need = {(j.arm, j.step_) for j in pending}
    in_use = {(j.arm, j.step_) for j, *_ in R.running.values()}
    for key in local_snaps():
        if key not in need and key not in in_use:
            shutil.rmtree(ckpt_dir(*key), ignore_errors=True)
            log(f"[ckpt] removed local {key[0]} {key[1]} (no pending job)")
    if pull_busy[0]:
        return
    order = []
    for j in pending:
        k = (j.arm, j.step_)
        if k not in order:
            order.append(k)
    have = {k for k in local_snaps() if ready(*k)}
    want = [k for k in order if k not in have]
    if not want or len(local_snaps()) >= args.max_local or free_gb() < args.min_free_gb + 1.2:
        return
    if args.pull_source == "relay":  # vast freed: the PC relay copies work2 ckpt_archive/<arm>/<step> here
        k = want[0]
        rec = {"arm": k[0], "step": k[1], "time": L.now()}
        wp = CV / "ctl/WANT.json"
        try:
            old = json.loads(wp.read_text())
        except (OSError, ValueError):
            old = {}
        if (old.get("arm"), old.get("step")) != k:
            wp.write_text(json.dumps(rec) + "\n")
            log(f"[pull] relay WANT {k[0]} {k[1]}")
        return
    pull_busy[0] = True

    def run():
        try:
            remote_scan()
            for k in want:
                if k in remote:
                    pull_one(*k)
                    return
        except Exception as exc:  # noqa: BLE001
            log(f"[pull] error {exc!r}"[:300])
        finally:
            pull_busy[0] = False

    threading.Thread(target=run, daemon=True).start()


# ----------------------------------------------------------------------------- plan
def sel_jobs(arm):
    return [Job("sel", arm, s, h, 0, 9) for s in steps(arm)[:N_SEL] for h in TRAIN_HANDS]


def best_of(arm):
    p = CV / "sel" / f"{arm}_best.json"
    return json.loads(p.read_text()) if p.exists() else None


def final_jobs(arm, step):
    js = [Job("final", arm, step, TEST_HAND, lo, hi) for lo, hi in L.CHUNKS]
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
        rec = {"arm": arm, "best": best, "rule": "max 2-hand (shadow+wuji) total /20 over epochs 50..250, ties -> later",
               "totals": tot, "per_hand": per, "time": L.now()}
        (CV / "sel").mkdir(parents=True, exist_ok=True)
        (CV / "sel" / f"{arm}_best.json").write_text(json.dumps(rec, indent=1))
        log(f"[cv2dp] SELECT {arm} done best={best} totals " + " ".join(f"{s}:{tot[s]}/20" for s in S)
            + " per_hand " + json.dumps(per[best]))


final_logged = set()


def plan():
    js = []
    A = [a for a in arms() if a not in gave_up or best_of(a)]
    for arm in A:
        js += sel_jobs(arm)
        b = best_of(arm)
        if b:
            fj = final_jobs(arm, b["best"])
            js += fj
            sh = [j for j in fj if j.hand == TEST_HAND]
            if arm not in final_logged and all(j.done() for j in sh):
                final_logged.add(arm)
                log(f"[cv2dp] FINAL {arm} sharpa done {sum(j.successes() for j in sh)}/50 (best {b['best']})")
    complete = bool(A) and all(best_of(a) for a in A) and all(j.done() for j in js)
    return js, complete


def prio(j):
    a = list(arms())
    k = 0 if j.kind == "sel" else (1 if j.hand == TEST_HAND else 2)
    return (k, int(j.step_) if k == 0 else 0, a.index(j.arm), j.lo, j.hand)


def main():
    (CV / "logs").mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print("arms", arms(), "gate", (CV / "ctl/EVAL_GATE_OPEN").exists(), "local", sorted(local_snaps()))
        remote_scan()
        print("remote", remote)
        for j in sorted(plan()[0], key=prio):
            print(j.name, j.done(), j.command()[-1][-260:])
        return
    lock = open(CV / "logs/driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("cv2dp driver already running")
        return
    (CV / "logs/driver.pid").write_text(str(os.getpid()))
    log(f"driver start pid {os.getpid()} parallel {args.parallel} arms {list(arms())} train_ref {args.train_ref}")
    R = Runner("cv2dp", args.parallel, log)
    started, last_beat = False, 0.0
    while True:
        if not started:
            if not (CV / "ctl/EVAL_GATE_OPEN").exists():
                time.sleep(args.poll)
                continue
            started = True
            log("[cv2dp] GATE open")
        R.reap()
        update_selection()
        js, complete = plan()
        manage_ckpts(js, R)
        if complete and not R.running:
            if (CV / "ctl/KEEP_RUNNING").exists():  # more arms (Zv2m) may still be added
                time.sleep(args.poll)
                continue
            log(f"[cv2dp] ALL DONE {len(js)} jobs")
            return
        stuck = [j for j in js if not j.done()]
        if stuck and all(j.name in R.failed for j in stuck) and not R.running:
            log(f"[cv2dp] FAIL {sorted(R.failed)}")
            return
        todo = sorted((j for j in js if not j.done() and ready(j.arm, j.step_)), key=prio)
        R.step(todo)
        if time.time() - last_beat > 1800:
            last_beat = time.time()
            log(f"heartbeat running={sorted(R.running)} done={sum(j.done() for j in js)}/{len(js)} "
                f"local={sorted(local_snaps())} free={free_gb():.1f}GB failed={sorted(R.failed)}")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
