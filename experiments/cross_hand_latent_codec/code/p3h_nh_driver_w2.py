#!/usr/bin/env python3
"""P3NH work2 driver (pooled3hand_nohot, 261114): runs the P3NH jobs (arms C', Z', J, D' = C/Z/J/D under
artifacts/pooled3hand/p3nh) that p3h_p3_partition.owner assigns to work2. New file (261114) = copy of p3h_p3_driver_w2.py
(P3, unchanged) with: P3 -> p3nh, eval p3h_nh_eval_pooled.py, arms Z/D/J/C, no local C copy (training ran on vast; ALL
snapshots come from work1 through the PC relay p3h_nh_relay_w2.sh), markers [p3h_nh_w2], and a different start gate:
  the P3 work2 driver (PID 2891159, p3h_p3_driver_w2.py) has exited (or '[p3h_p3_w2] ALL DONE' is in p3/logs/driver_w2.log)
  and no P3 p3h_eval_pooled.py job is alive in the container; then, once, the J dataset-action replay check (b)
  (p3h_nh_check_b.sh, J x 3 hands x pkl eps 0-2, result in p3nh/check/b/summary.txt) runs before any eval job.
The rest of this docstring is the P3 text (steps 1-2 replaced by the above).
Original: New file.
Host python3 (stdlib only); Isaac jobs run in work2's xhand container with exactly work1's command line
(same p3h_eval_pooled.py, demos pkl, codec, DexVerse 8d0a6b5; md5-checked 261113).

1. waits for "[p3h_queue] ALL DONE" in /home/eun/xhand/runs/pooled3hand/logs/queue.log (arm C training finished);
2. C snapshots: copied here from the training dir (root-owned -> docker exec cp), sha256-verified, <P3>/ckpt/C/<step>/.complete;
   Z/D snapshots: pushed from work1 by the PC relay (p3h_p3_relay_w2.sh), sha256-verified there;
3. launches work2-owned jobs (sel: partition.json; final: once ctl/<arm>_best.json and ctl/final_split.json arrive from
   work1 via the relay). One launch per pass, >= LAUNCH_GAP s apart, only while the GPU has >= NEED_GPU_MIB free and the
   host >= NEED_RAM_MIB available (never touches other processes), at most --parallel;
4. appends each finished job to <P3>/logs/w2_done.txt (the relay ships those dirs to work1; work1's driver does selection,
   summary and the single "[p3h_p3] ALL DONE").
Selection is never computed here. Markers in <P3>/logs/driver_w2.log:
  [p3h_nh_w2] WAITING training   [p3h_nh_w2] GPU free, start   [p3h_nh_w2] FAIL <job>   [p3h_nh_w2] ALL DONE
usage: nohup python3 p3h_p3_driver_w2.py --parallel 3 >> <P3>/logs/driver_w2_nohup.out 2>&1 &
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3h_p3_partition as PT  # noqa: E402

KST = timezone(timedelta(hours=9))
MACHINE = "work2"
HOST_LC = Path("/home/eun/xhand/latent_codec_v1")
CONT_LC = "/workspace/xhand/latent_codec_v1"
P3 = HOST_LC / "artifacts/pooled3hand/p3nh"
CP3 = f"{CONT_LC}/artifacts/pooled3hand/p3nh"
OLD_P3 = HOST_LC / "artifacts/pooled3hand/p3"
OLD_PID = 2891159
CHECK_B = Path(__file__).resolve().parent / "p3h_nh_check_b.sh"
CTL = P3 / "ctl"  # relay-synced copies of work1's partition.json, sel/<arm>_best.json, sel/final_split.json
DEMOS = f"{CONT_LC}/artifacts/pooled3hand/p0b/demos/Dexverse-GraspKettle-v0_floating_{{hand}}_right.pkl"
EVAL = f"{CONT_LC}/xhand_latent/p3h_nh_eval_pooled.py"
ARMS = ("Z", "D", "J", "C")  # vast training order (261114)
HANDS = ("sharpa", "shadow", "wuji")
SEL_STEPS = ("029600", "059200", "088800", "118400", "148000")
FINAL_CHUNKS = ((0, 9), (10, 19), (20, 29), (30, 39), (40, 49))
MAX_ATTEMPTS = 4
HANG_MIN = 45
NEED_GPU_MIB = 7800   # one worker (Isaac + policy server) peaks ~7.1 GB
NEED_RAM_MIB = 7500   # ~5.1 GB Isaac + ~1.7 GB policy server RSS
LAUNCH_GAP = 150

ap = argparse.ArgumentParser()
ap.add_argument("--parallel", type=int, default=3)
ap.add_argument("--poll", type=int, default=60)
ap.add_argument("--no-wait", action="store_true", help="skip the P3-finished gate (only if P3 on work2 is known done)")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
LOG = P3 / "logs/driver_w2.log"
DONE_LIST = P3 / "logs/w2_done.txt"


def now():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{now()} {msg}\n")


def sh(cmd, timeout=600, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


GPU = sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).stdout.strip()


def ckpt_dir(arm, step):
    return P3 / "ckpt" / arm / step


def ckpt_ready(arm, step):
    return (ckpt_dir(arm, step) / ".complete").exists()


def copy_c():
    """local C snapshots: training dir (root) -> <P3>/ckpt/C/<step>, sha256 of every file compared."""
    for step in SEL_STEPS:
        if ckpt_ready("C", step):
            continue
        src = f"{C_SRC}/{step}/pretrained_model"
        age = sh(["docker", "exec", "xhand", "bash", "-c", f"test -f {src}/model.safetensors && test -f {src}/train_config.json"
                  f" && echo $(( $(date +%s) - $(stat -c %Y {src}/model.safetensors) ))"], timeout=60).stdout.strip()
        if not age.isdigit() or int(age) < 180:
            continue
        d = ckpt_dir("C", step)
        tmp = d.parent / f".{step}.tmp"
        cd, ctmp = str(d).replace(str(HOST_LC), CONT_LC), str(tmp).replace(str(HOST_LC), CONT_LC)
        sh(["docker", "exec", "xhand", "rm", "-rf", ctmp], timeout=120)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        r = sh(["docker", "exec", "xhand", "bash", "-c", f"mkdir -p {ctmp} && cp -a {src} {ctmp}/ && "
                f"chown -R {os.getuid()}:{os.getgid()} {ctmp}"], timeout=1800)
        a = sh(["docker", "exec", "xhand", "bash", "-c", f"cd {src} && sha256sum *"], timeout=900).stdout.split()
        b = sh(["bash", "-c", f"cd {tmp}/pretrained_model && sha256sum *"], timeout=900).stdout.split()
        if r.returncode or not a or sorted(a) != sorted(b):
            log(f"[ckpt] C {step} copy/sha256 failed rc={r.returncode}, retry next pass")
            continue
        sh(["docker", "exec", "xhand", "rm", "-rf", cd], timeout=120)
        tmp.rename(d)
        (d / ".complete").write_text(f"copied {now()} on work2 from {src} sha256 ok\n")
        log(f"[ckpt] C {step} OK in {time.time() - t0:.0f}s")


class Job:
    def __init__(self, kind, arm, step, hand, lo, hi):
        self.kind, self.arm, self.step, self.hand, self.lo, self.hi = kind, arm, step, hand, lo, hi
        self.name = f"{kind}/{arm}_{step}/{hand}_ep{lo}-{hi}"
        self.out = P3 / self.name
        self.cout = f"{CP3}/{self.name}"
        self.eps = list(range(lo, hi + 1))

    def metrics(self):
        try:
            return json.loads((self.out / "metrics.json").read_text())
        except Exception:  # noqa: BLE001
            return None

    def done(self):
        m = self.metrics()
        return bool(m) and sorted(e["episode"] for e in m["episodes"]) == self.eps and m.get("ckpt", "").endswith(
            f"ckpt/{self.arm}/{self.step}/pretrained_model")

    def successes(self):
        m = self.metrics()
        return sum(e["success"] for e in m["episodes"]) if m else 0

    def attempts(self):
        return len(list((P3 / "logs/jobs").glob(self.name.replace("/", "__") + ".log.*")))

    def command(self):  # identical to work1's driver
        inner = ("export DEXVERSE_ROOT=/workspace/DexVerse OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "
                 "TZ=Asia/Seoul; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS DEXVERSE_OBJECT_MATERIAL "
                 "DEXVERSE_DUMMY_LINK_MASS; cd /workspace/DexVerse; /workspace/isaaclab/_isaac_sim/python.sh "
                 f"{EVAL} --arm {self.arm} --demos {DEMOS.format(hand=self.hand)} "
                 f"--ckpt {CP3}/ckpt/{self.arm}/{self.step}/pretrained_model --output_dir {self.cout} "
                 f"--episodes {','.join(map(str, self.eps))} --max_steps 3000 --no-video --headless")
        return ["flock", "-s", "/tmp/isaac_gpu.lock", "bash", "-c",
                'echo "[job] LOCKED $(date +%F_%T)"; exec docker exec xhand bash -c "$0"', inner]


def ctl():
    part = PT.load(CTL / "partition.json")
    split = PT.load(CTL / "final_split.json")
    best = {a: PT.load(CTL / f"{a}_best.json") for a in ARMS}
    return part, split, best


def my_jobs():
    """(jobs owned by work2 known so far, complete?) complete = final split and every arm's best are known."""
    part, split, best = ctl()
    if not part:
        return [], False
    js = [Job("sel", a, s, h, 0, 9) for s in SEL_STEPS for a in ARMS for h in HANDS]
    for a in ARMS:
        if best[a] and split is not None:
            js += [Job("final", a, best[a]["best"], h, lo, hi) for h in HANDS for lo, hi in FINAL_CHUNKS]
    complete = split is not None and all(best.values())
    return [j for j in js if PT.owner(j.name, part, split) == MACHINE], complete


running = {}
failed = set()


def write_machine(job, n):
    mj = {"machine": MACHINE, "gpu": GPU, "workers": args.parallel, "attempt": n, "launched": now(), "job": job.name}
    sh(["docker", "exec", "-i", "xhand", "bash", "-c", f"mkdir -p {job.cout} && cat > {job.cout}/machine.json"],
       input=json.dumps(mj) + "\n", timeout=60)


def launch(job):
    (P3 / "logs/jobs").mkdir(parents=True, exist_ok=True)
    n = job.attempts() + 1
    lp = P3 / "logs/jobs" / (job.name.replace("/", "__") + f".log.{n}")
    write_machine(job, n)
    fh = open(lp, "w")
    p = subprocess.Popen(job.command(), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    running[job.name] = (job, p, lp, time.time())
    log(f"START {job.name} attempt {n} pid {p.pid}")


def kill_job(job, p):
    sh(["docker", "exec", "xhand", "pkill", "-f", f"[o]utput_dir {job.cout} "], timeout=60)
    time.sleep(5)
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def reap():
    for name, (job, p, lp, start) in list(running.items()):
        rc = p.poll()
        if rc is None:
            text = lp.read_text(errors="ignore") if lp.exists() else ""
            if "[job] LOCKED" in text and time.time() - lp.stat().st_mtime > HANG_MIN * 60:
                log(f"HANG {name}: no log output for {HANG_MIN} min -> kill")
                kill_job(job, p)
            continue
        del running[name]
        m = job.metrics()
        n = len(m["episodes"]) if m else 0
        log(f"{'DONE' if job.done() else 'EXIT'} {name} rc={rc} episodes {n}/{len(job.eps)}"
            + (f" success {job.successes()}/{n}" if m else ""))


def mark_done(jobs):
    listed = set(DONE_LIST.read_text().split()) if DONE_LIST.exists() else set()
    with open(DONE_LIST, "a") as f:
        for j in jobs:
            if j.name not in listed and j.name not in running and j.done():
                f.write(j.name + "\n")


def container_running():
    r = sh(["docker", "exec", "xhand", "pgrep", "-af", "[p]3h_nh_eval_pooled.py"], timeout=60)
    out = set()
    for line in r.stdout.splitlines():
        parts = line.split()
        if "--output_dir" in parts:
            out.add(parts[parts.index("--output_dir") + 1])
    return out


def resources():
    q = sh(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).stdout.split()
    gpu_free = int(q[0]) if q and q[0].isdigit() else 0
    ram = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            ram = int(line.split()[1]) // 1024
    return gpu_free, ram


def training_done():
    """START GATE (name kept from P3): the P3 work2 driver is finished and no P3 eval job is alive in the container."""
    try:
        done = "[p3h_p3_w2] ALL DONE" in (OLD_P3 / "logs/driver_w2.log").read_text(errors="ignore")
    except OSError:
        done = False
    try:
        alive = "p3h_p3_driver_w2" in Path(f"/proc/{OLD_PID}/cmdline").read_bytes().decode(errors="ignore")
    except OSError:
        alive = False
    if alive and not done:
        return False
    r = sh(["docker", "exec", "xhand", "pgrep", "-af", "[p]3h_eval_pooled.py"], timeout=60)
    return r.returncode != 0 or not r.stdout.strip()


def check_b():
    """J dataset-action replay (once; synchronous, ~10 min). Result lines in p3nh/check/b/summary.txt."""
    sf = P3 / "check/b/summary.txt"
    if sf.exists() and "CHECK_B DONE" in sf.read_text(errors="ignore"):
        return
    log("[p3h_nh_w2] CHECK_B J replay start")
    r = sh(["bash", str(CHECK_B), "3"], timeout=7200)
    txt = sf.read_text(errors="ignore") if sf.exists() else ""
    log(f"[p3h_nh_w2] CHECK_B rc={r.returncode}: " + " | ".join(x for x in txt.splitlines() if x.strip()))


def main():
    (P3 / "logs").mkdir(parents=True, exist_ok=True)
    CTL.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        js, complete = my_jobs()
        print(f"ctl complete={complete} p3_gate_open={training_done()} resources={resources()}")
        for j in js:
            print(f"{j.name:40s} done={j.done()} ckpt={ckpt_ready(j.arm, j.step)}")
        return
    lock = open(P3 / "logs/driver_w2.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("driver_w2 already running")
        return
    (P3 / "logs/driver_w2.pid").write_text(str(os.getpid()))
    log(f"driver_w2 start pid {os.getpid()} parallel {args.parallel} gpu {GPU}")
    waiting_logged = started = False
    last_beat = last_launch = 0.0
    while True:
        js, complete = my_jobs()
        reap()
        mark_done(js)
        if complete and js and all(j.done() for j in js) and not running:
            mark_done(js)
            log(f"[p3h_nh_w2] ALL DONE {len(js)} jobs")
            return
        if not started:
            if not (args.no_wait or training_done()):
                if not waiting_logged:
                    log("[p3h_nh_w2] WAITING P3 work2 driver (PID 2891159) / P3 jobs to finish")
                    waiting_logged = True
                time.sleep(args.poll)
                continue
            started = True
            log(f"[p3h_nh_w2] GPU free, start (P3 on work2 done); resources gpu_free/ram_avail MiB {resources()}")
            try:
                check_b()
            except Exception as exc:  # noqa: BLE001
                log(f"[p3h_nh_w2] CHECK_B error {exc!r}"[:300])
        ext = container_running() - {j.cout for j, *_ in running.values()}
        busy = len(running) + len(ext)
        todo = [j for j in js if not j.done() and j.name not in running and j.name not in failed and j.cout not in ext
                and ckpt_ready(j.arm, j.step)]
        todo.sort(key=lambda j: (j.kind != "sel", ARMS.index(j.arm), j.step, HANDS.index(j.hand), j.lo))  # = work1 order
        if todo and busy < args.parallel and time.time() - last_launch > LAUNCH_GAP:
            j = todo[0]
            gpu_free, ram = resources()
            if j.attempts() >= MAX_ATTEMPTS:
                failed.add(j.name)
                log(f"[p3h_nh_w2] FAIL {j.name} after {MAX_ATTEMPTS} attempts (see logs/jobs)")
            elif gpu_free >= NEED_GPU_MIB and ram >= NEED_RAM_MIB:
                launch(j)
                last_launch = time.time()
            elif time.time() - last_beat > 600:
                log(f"hold launch: gpu_free {gpu_free} MiB ram_avail {ram} MiB (need {NEED_GPU_MIB}/{NEED_RAM_MIB}), busy {busy}")
                last_beat = time.time() - 1200
        if time.time() - last_beat > 1800:
            last_beat = time.time()
            ready = {a: [s for s in SEL_STEPS if ckpt_ready(a, s)] for a in ARMS}
            log(f"heartbeat running={sorted(running)} external={sorted(ext)} my_jobs={len(js)} "
                f"done={sum(j.done() for j in js)} ctl_complete={complete} ckpts={json.dumps(ready)} failed={sorted(failed)}")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
