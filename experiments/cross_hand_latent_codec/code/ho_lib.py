#!/usr/bin/env python3
"""HO (H1/H2, 261115): shared host-side job runner for the cross-hand evaluations on work2. New file, stdlib only.
Job/launch/reap/hang logic copied from p3h_nh_driver_w2.py (unchanged original): one Isaac worker = ho_eval_pooled.py +
policy server in the xhand container, shared flock on /tmp/isaac_gpu.lock, launch only while GPU >= 7.8 GB and RAM >=
7.5 GB are free, >= 150 s between launches, hang = no log output for 45 min, at most 4 attempts per job.
Worker budget: every eval process alive in the container counts (ours, nohot p3h_nh_eval_pooled.py, P3 p3h_eval_pooled.py),
so H1/H2 never exceed --parallel workers on the machine together with anything still running there.
"""
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
HOST_LC = Path("/home/eun/xhand/latent_codec_v1")
CONT_LC = "/workspace/xhand/latent_codec_v1"
HO = HOST_LC / "artifacts/pooled3hand/ho"
CHO = f"{CONT_LC}/artifacts/pooled3hand/ho"
EVAL = f"{CONT_LC}/xhand_latent/ho_eval_pooled.py"
TASK = "Dexverse-GraspKettle-v0"
CHUNKS = ((0, 9), (10, 19), (20, 29), (30, 39), (40, 49))
MAX_ATTEMPTS = 4
HANG_MIN = 45
NEED_GPU_MIB = 7800
NEED_RAM_MIB = 7500
LAUNCH_GAP = 150
EVAL_PATTERNS = ("[h]o_eval_pooled.py", "[p]3h_nh_eval_pooled.py", "[p]3h_eval_pooled.py")


def now():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def sh(cmd, timeout=600, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def demos_rel(hand):
    """demos pkl (relative to artifacts/pooled3hand) and number of episodes available for the hand."""
    full = HO / f"demos/{TASK}_floating_{hand}_right.pkl"
    if full.exists():
        return f"ho/demos/{full.name}", 50
    part = HO / f"demos/{TASK}_floating_{hand}_right_ep0-9.pkl"
    if part.exists():
        return f"ho/demos/{part.name}", 10
    return f"p0b/demos/{TASK}_floating_{hand}_right.pkl", 50


class Job:
    def __init__(self, name, arm, ckpt_host, hand, lo, hi, f_task, dataset):
        self.name, self.arm, self.hand, self.lo, self.hi = name, arm, hand, lo, hi
        self.ckpt_host = Path(ckpt_host)
        self.ckpt = str(self.ckpt_host).replace(str(HOST_LC), CONT_LC)
        self.f_task, self.dataset = f_task, dataset
        self.out = HO / name
        self.cout = f"{CHO}/{name}"
        self.eps = list(range(lo, hi + 1))

    def metrics(self):
        try:
            return json.loads((self.out / "metrics.json").read_text())
        except Exception:  # noqa: BLE001
            return None

    def done(self):
        m = self.metrics()
        return bool(m) and sorted(e["episode"] for e in m["episodes"]) == self.eps and m.get("ckpt") == self.ckpt

    def successes(self):
        m = self.metrics()
        return sum(e["success"] for e in m["episodes"]) if m else 0

    def n_done(self):
        m = self.metrics()
        return len(m["episodes"]) if m and m.get("ckpt") == self.ckpt else 0

    def wall(self):
        m = self.metrics()
        return [e["wall_s"] for e in m["episodes"]] if m else []

    def attempts(self):
        return len(list((HO / "logs/jobs").glob(self.name.replace("/", "__") + ".log.*")))

    def command(self):
        d, _ = demos_rel(self.hand)
        inner = ("export DEXVERSE_ROOT=/workspace/DexVerse OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "
                 "TZ=Asia/Seoul; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS DEXVERSE_OBJECT_MATERIAL "
                 "DEXVERSE_DUMMY_LINK_MASS; cd /workspace/DexVerse; /workspace/isaaclab/_isaac_sim/python.sh "
                 f"{EVAL} --arm {self.arm} --demos {CONT_LC}/artifacts/pooled3hand/{d} --ckpt {self.ckpt} "
                 f"--f_task {self.f_task} --dataset {self.dataset} --output_dir {self.cout} "
                 f"--episodes {','.join(map(str, self.eps))} --max_steps 3000 --no-video --headless")
        return ["flock", "-s", "/tmp/isaac_gpu.lock", "bash", "-c",
                'echo "[job] LOCKED $(date +%F_%T)"; exec docker exec xhand bash -c "$0"', inner]


def resources():
    q = sh(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).stdout.split()
    gpu_free = int(q[0]) if q and q[0].isdigit() else 0
    ram = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            ram = int(line.split()[1]) // 1024
    return gpu_free, ram


def container_evals():
    """{pattern: set(output_dir)} of eval processes alive in the xhand container."""
    out = {}
    for pat in EVAL_PATTERNS:
        r = sh(["docker", "exec", "xhand", "pgrep", "-af", pat], timeout=60)
        s = set()
        for line in r.stdout.splitlines():
            parts = line.split()
            if "--output_dir" in parts:
                s.add(parts[parts.index("--output_dir") + 1])
        out[pat] = s
    return out


class Runner:
    def __init__(self, tag, parallel, log):
        self.tag, self.parallel, self.log = tag, parallel, log
        self.running, self.failed = {}, set()
        self.last_launch = 0.0
        self.gpu = sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).stdout.strip()

    def write_machine(self, job, n):
        mj = {"machine": "work2", "gpu": self.gpu, "workers": self.parallel, "attempt": n, "launched": now(), "job": job.name}
        sh(["docker", "exec", "-i", "xhand", "bash", "-c", f"mkdir -p {job.cout} && cat > {job.cout}/machine.json"],
           input=json.dumps(mj) + "\n", timeout=60)

    def launch(self, job):
        (HO / "logs/jobs").mkdir(parents=True, exist_ok=True)
        n = job.attempts() + 1
        lp = HO / "logs/jobs" / (job.name.replace("/", "__") + f".log.{n}")
        self.write_machine(job, n)
        fh = open(lp, "w")
        p = subprocess.Popen(job.command(), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        self.running[job.name] = (job, p, lp, time.time())
        self.last_launch = time.time()
        self.log(f"START {job.name} attempt {n} pid {p.pid}")

    def kill_job(self, job, p):
        sh(["docker", "exec", "xhand", "pkill", "-f", f"[o]utput_dir {job.cout} "], timeout=60)
        time.sleep(5)
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def reap(self):
        for name, (job, p, lp, start) in list(self.running.items()):
            rc = p.poll()
            if rc is None:
                if lp.exists() and "[job] LOCKED" in lp.read_text(errors="ignore") and time.time() - lp.stat().st_mtime > HANG_MIN * 60:
                    self.log(f"HANG {name}: no log output for {HANG_MIN} min -> kill")
                    self.kill_job(job, p)
                continue
            del self.running[name]
            m = job.metrics()
            n = len(m["episodes"]) if m else 0
            self.log(f"{'DONE' if job.done() else 'EXIT'} {name} rc={rc} episodes {n}/{len(job.eps)}"
                     + (f" success {job.successes()}/{n}" if m else ""))

    def busy(self):
        ev = container_evals()
        mine = {j.cout for j, *_ in self.running.values()}
        ext = set().union(*ev.values()) - mine
        return len(self.running) + len(ext), ext

    def step(self, todo):
        """launch at most one job of todo (ordered) if the worker budget/resources allow."""
        busy, ext = self.busy()
        todo = [j for j in todo if not j.done() and j.name not in self.running and j.name not in self.failed
                and j.cout not in ext]
        if not todo or busy >= self.parallel or time.time() - self.last_launch < LAUNCH_GAP:
            return
        j = todo[0]
        if j.attempts() >= MAX_ATTEMPTS:
            self.failed.add(j.name)
            self.log(f"[{self.tag}] FAIL {j.name} after {MAX_ATTEMPTS} attempts (see ho/logs/jobs)")
            return
        gpu_free, ram = resources()
        if gpu_free >= NEED_GPU_MIB and ram >= NEED_RAM_MIB:
            self.launch(j)


def alive(pid, needle):
    try:
        return needle in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
    except OSError:
        return False
