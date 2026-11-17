#!/usr/bin/env python3
"""PI05 driver (261115, work2 only): pi0.5 nohot arms Z', D' (trained on vast by pi05_vast_train.sh) -> pull exports
vast -> work2 -> final evaluation. PROTOCOL (user decision 261115): NO snapshot selection; only the LAST export
(step 23670 = epoch 10) is evaluated: 3 hands (sharpa, shadow, wuji) x eps 0-49 (5 chunks of 10) per arm = 300 rollouts.
Only that export is pulled (the earlier exports stay on vast).
New file, host python3 stdlib. Jobs = pi05_eval_pooled.py (P3NH harness with the pi0.5 server) in the xhand container,
same command line as the DP jobs (demos p0b, --max_steps 3000 --no-video --headless, flock -s /tmp/isaac_gpu.lock).
GPU sharing: one pi05_policy_server.py --shared per (kind, arm, step) group (~7.6 GB) serves up to --parallel Isaac
workers (~5.3 GB each) via PI05_SERVER=<port>:<authkey>; the server is stopped when its group is finished.
  pull : from start (no gate); every 5 min one ssh listing of vast runs/pi05_nohot/{Z,D}/train/export/<step>/ (atomic
         dirs) -> pi05/ckpt/<arm>/<step>/export (tar stream, sha256 of every file compared, .complete); pi05_base once ->
         pi05/pi05_base. Key ~/.ssh/ho_vast_ed25519 (as ho_h2_driver.py). '[pi05_vast] <arm>: GIVING UP' drops the arm.
  gate : evaluation starts after '[ho_h2] ALL DONE' in ho/logs/h2_driver.log (DP queue nohot -> H1 -> H2 first) and when no
         other eval (p3h_nh / p3h / ho / pi05 harness) is alive in the container; afterwards every eval alive in the
         container counts against --parallel.
Markers in pi05/logs/driver.log: [pi05] WAITING ... | [pi05] GATE open | [pi05] FAIL ... |
  [pi05] ALL DONE
usage: nohup python3 pi05_driver.py --parallel 3 >> artifacts/pooled3hand/pi05/logs/driver_nohup.out 2>&1 &
"""
import argparse
import fcntl
import json
import os
import secrets
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
HOST_LC = Path("/home/eun/xhand/latent_codec_v1")
CONT_LC = "/workspace/xhand/latent_codec_v1"
PI = HOST_LC / "artifacts/pooled3hand/pi05"
CPI = f"{CONT_LC}/artifacts/pooled3hand/pi05"
H2_LOG = HOST_LC / "artifacts/pooled3hand/ho/logs/h2_driver.log"
DEMOS = f"{CONT_LC}/artifacts/pooled3hand/p0b/demos/Dexverse-GraspKettle-v0_floating_{{hand}}_right.pkl"
EVAL = f"{CONT_LC}/xhand_latent/pi05_eval_pooled.py"
SERVER = f"{CONT_LC}/xhand_latent/pi05_policy_server.py"
PY = "/workspace/xhand/envs/pi05_python"
ARMS = ("Z", "D")
HANDS = ("sharpa", "shadow", "wuji")
ALL_STEPS = ("004734", "009468", "014202", "018936", "023670")  # 2367 steps/epoch (37,858 frames / global bs 16) x 2,4,6,8,10
FINAL_STEP = "023670"
STEPS = (FINAL_STEP,)  # pulled + evaluated
FINAL_CHUNKS = ((0, 9), (10, 19), (20, 29), (30, 39), (40, 49))
MAX_ATTEMPTS = 4
HANG_MIN = 45
NEED_SERVER_MIB = 9000   # pi0.5 bf16 server ~7.6 GB (+ margin)
NEED_WORKER_MIB = 6000   # Isaac worker peak ~5.3 GB
NEED_RAM_MIB = 7500
LAUNCH_GAP = 150
EVAL_PATTERNS = ("[h]o_eval_pooled.py", "[p]3h_nh_eval_pooled.py", "[p]3h_eval_pooled.py", "[p]i05_eval_pooled.py")
VAST_RUNS = "/workspace/xhand/runs/pi05_nohot"
VAST_BASE = "/workspace/xhand/pi05_assets/pi05_base"

ap = argparse.ArgumentParser()
ap.add_argument("--parallel", type=int, default=3)
ap.add_argument("--poll", type=int, default=60)
ap.add_argument("--vast", default="root@<VAST_HOST>:<VAST_PORT>")
ap.add_argument("--no-wait", action="store_true")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
LOG = PI / "logs/driver.log"
USER, HP = args.vast.split(":")
VAST = ["ssh", "-i", os.path.expanduser("~/.ssh/ho_vast_ed25519"), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", "-p", HP, USER]


def now():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{now()} {msg}\n")


def sh(cmd, timeout=600, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def read(p):
    try:
        return Path(p).read_text(errors="ignore")
    except OSError:
        return ""


GPU = sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).stdout.strip()


def ckpt_dir(arm, step):
    return PI / "ckpt" / arm / step


def ready(arm, step):
    return (ckpt_dir(arm, step) / ".complete").exists() and base_ready()


def base_ready():
    return (PI / "pi05_base/.complete").exists()


# ----------------------------------------------------------------------------- pull (vast -> work2)
pull_busy, last_pull, gave_up = [False], [0.0], set()


def tar_pull(remote_parent, name, dest, what):
    tmp = dest.parent / f".{dest.name}.tmp"
    sh(["rm", "-rf", str(tmp)])
    tmp.mkdir(parents=True)
    t0 = time.time()
    tar = subprocess.Popen(VAST + [f"tar -C {remote_parent} -cf - {name}"], stdout=subprocess.PIPE)
    r = subprocess.run(["tar", "-xf", "-", "-C", str(tmp)], stdin=tar.stdout, timeout=7200)
    tar.wait(timeout=60)
    if r.returncode or tar.returncode:
        log(f"[pull] {what} tar failed rc={r.returncode}/{tar.returncode}")
        return False
    rs = sh(VAST + [f"cd {remote_parent}/{name} && sha256sum *"], timeout=1200).stdout.split()
    ls = sh(["bash", "-c", f"cd {tmp}/{name} && sha256sum *"], timeout=1200).stdout.split()
    if not rs or sorted(rs) != sorted(ls):
        log(f"[pull] {what} sha256 mismatch, will retry")
        return False
    sh(["rm", "-rf", str(dest)])
    tmp.rename(dest)
    (dest / ".complete").write_text(f"pulled {now()} from vast {remote_parent}/{name} sha256 ok\n")
    log(f"[pull] {what} OK in {time.time() - t0:.0f}s")
    return True


def pull_all():
    try:
        if not base_ready():
            tar_pull(str(Path(VAST_BASE).parent), Path(VAST_BASE).name, PI / "pi05_base", "pi05_base")
        cmd = (f"for f in {VAST_RUNS}/{{Z,D}}/train/export/[0-9]*/trainable.safetensors; do test -f $f || continue; "
               "echo $f $(( $(date +%s) - $(stat -c %Y $f) )); done; "
               f"grep -oE '\\[pi05_vast\\] ([ZD]: GIVING UP|SMOKE FAILED)' {VAST_RUNS}/logs/queue.log 2>/dev/null")
        r = sh(VAST + [cmd], timeout=120)
        if r.returncode not in (0, 1):
            log(f"[pull] vast unreachable this pass: {r.stderr.strip()[-150:]}")
            return
        avail = set()
        for line in r.stdout.splitlines():
            if "GIVING UP" in line or "SMOKE FAILED" in line:
                arms = ARMS if "SMOKE" in line else (line.split("]")[1].strip()[0],)
                for a in arms:
                    if a not in gave_up:
                        gave_up.add(a)
                        log(f"[pi05] vast training {a} GAVE UP ({line.strip()}) -> arm dropped")
                continue
            p = line.split()
            if len(p) == 2 and p[1].isdigit() and int(p[1]) >= 60:
                parts = p[0].split("/")
                avail.add((parts[-5], parts[-2]))
        for step in STEPS:
            for arm in ARMS:
                if (arm, step) in avail and not (ckpt_dir(arm, step) / ".complete").exists():
                    d = ckpt_dir(arm, step)
                    d.mkdir(parents=True, exist_ok=True)
                    tar_pull(f"{VAST_RUNS}/{arm}/train/export", step, d / "export", f"{arm} {step}")
                    if (d / "export/.complete").exists():
                        (d / ".complete").write_text(read(d / "export/.complete"))
    except Exception as exc:  # noqa: BLE001
        log(f"[pull] error {exc!r}"[:300])
    finally:
        pull_busy[0] = False


def pull_vast():
    if pull_busy[0] or time.time() - last_pull[0] < 300 or all(ready(a, s) for a in ARMS for s in STEPS):
        return
    last_pull[0] = time.time()
    pull_busy[0] = True
    threading.Thread(target=pull_all, daemon=True).start()


# ----------------------------------------------------------------------------- jobs
class Job:
    def __init__(self, kind, arm, step, hand, lo, hi):
        self.kind, self.arm, self.step, self.hand, self.lo, self.hi = kind, arm, step, hand, lo, hi
        self.group = (kind, arm, step)
        self.name = f"{kind}/{arm}_{step}/{hand}_ep{lo}-{hi}"
        self.out = PI / self.name
        self.cout = f"{CPI}/{self.name}"
        self.ckpt = f"{CPI}/ckpt/{arm}/{step}/export"
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

    def attempts(self):
        return len(list((PI / "logs/jobs").glob(self.name.replace("/", "__") + ".log.*")))

    def command(self, server):
        inner = ("export DEXVERSE_ROOT=/workspace/DexVerse OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "
                 f"TZ=Asia/Seoul PI05_SERVER={server}; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS DEXVERSE_OBJECT_MATERIAL "
                 "DEXVERSE_DUMMY_LINK_MASS; cd /workspace/DexVerse; /workspace/isaaclab/_isaac_sim/python.sh "
                 f"{EVAL} --arm {self.arm} --demos {DEMOS.format(hand=self.hand)} --ckpt {self.ckpt} "
                 f"--output_dir {self.cout} --episodes {','.join(map(str, self.eps))} --max_steps 3000 --no-video --headless")
        return ["flock", "-s", "/tmp/isaac_gpu.lock", "bash", "-c",
                'echo "[job] LOCKED $(date +%F_%T)"; exec docker exec xhand bash -c "$0"', inner]


def plan():
    js = []
    arms = [a for a in ARMS if a not in gave_up]
    for arm in arms:
        js += [Job("final", arm, FINAL_STEP, h, lo, hi) for h in HANDS for lo, hi in FINAL_CHUNKS]
    complete = bool(arms) and all(j.done() for j in js)
    return js, complete


def results(js, failed):
    md = ["# RESULTS_PI05 — pi0.5 (vision+expert, LLM frozen) nohot Z'/D', GraspKettle, pooled 3 hands", "",
          f"Generated {now()} by pi05_driver.py (work2). No selection: last export {FINAL_STEP} (epoch 10) only. "
          "Eval pi05_eval_pooled.py (P3NH harness), 3000-step cap, n_action_steps 8 of chunk 50.", "",
          "| arm | step | " + " | ".join(HANDS) + " | total |", "|---|---|" + "---|" * 4]
    summ = {"step": FINAL_STEP, "final": {}, "failed": sorted(failed)}
    for arm in ARMS:
        fj = [j for j in js if j.arm == arm]
        if not fj:
            continue
        per = {h: (sum(j.successes() for j in fj if j.hand == h), sum(len(j.eps) for j in fj if j.hand == h and j.done())) for h in HANDS}
        md.append(f"| {arm} | {FINAL_STEP} | " + " | ".join(f"{s}/{n}" for s, n in per.values())
                  + f" | {sum(s for s, _ in per.values())}/{sum(n for _, n in per.values())} |")
        summ["final"][arm] = per
    (PI / "RESULTS_PI05.md").write_text("\n".join(md) + "\n")
    (PI / "pi05_summary.json").write_text(json.dumps(summ, indent=1))


# ----------------------------------------------------------------------------- runner (shared server per group)
running, failed = {}, set()
server = {"group": None, "port": None, "key": None, "log": None}


def resources():
    q = sh(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).stdout.split()
    gpu_free = int(q[0]) if q and q[0].isdigit() else 0
    ram = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            ram = int(line.split()[1]) // 1024
    return gpu_free, ram


def container_evals():
    out = set()
    for pat in EVAL_PATTERNS:
        r = sh(["docker", "exec", "xhand", "pgrep", "-af", pat], timeout=60)
        for line in r.stdout.splitlines():
            parts = line.split()
            if "--output_dir" in parts:
                out.add(parts[parts.index("--output_dir") + 1])
    return out


def start_server(group):
    kind, arm, step = group
    port = 29600 + secrets.randbelow(300)
    key = secrets.token_hex(16)
    lg = f"{CPI}/logs/server_{kind}_{arm}_{step}.log"
    cmd = f"exec {PY} {SERVER} --shared --ckpt {CPI}/ckpt/{arm}/{step}/export --port {port} --authkey {key} >> {lg} 2>&1"
    sh(["docker", "exec", "-d", "xhand", "bash", "-c", cmd], timeout=60)
    host_log = Path(lg.replace(CONT_LC, str(HOST_LC)))
    for _ in range(60):
        time.sleep(5)
        t = read(host_log)
        if f"listening on 127.0.0.1:{port}" in t:
            server.update(group=group, port=port, key=key, log=host_log)
            log(f"SERVER up {kind} {arm} {step} port {port}: " + (t.strip().splitlines() or [""])[-2][-160:])
            return True
        if "Traceback" in t:
            break
    log(f"SERVER failed {kind} {arm} {step}: {read(host_log)[-400:]!r}")
    stop_server(port)
    return False


def stop_server(port=None):
    port = port or server["port"]
    if port:
        sh(["docker", "exec", "xhand", "pkill", "-f", f"[p]i05_policy_server.py --shared .* --port {port} "], timeout=60)
        log(f"SERVER stopped port {port}")
    server.update(group=None, port=None, key=None, log=None)


def launch(job):
    (PI / "logs/jobs").mkdir(parents=True, exist_ok=True)
    n = job.attempts() + 1
    lp = PI / "logs/jobs" / (job.name.replace("/", "__") + f".log.{n}")
    mj = {"machine": "work2", "gpu": GPU, "workers": args.parallel, "attempt": n, "launched": now(), "job": job.name,
          "policy": "pi05 shared server", "server_port": server["port"]}
    sh(["docker", "exec", "-i", "xhand", "bash", "-c", f"mkdir -p {job.cout} && cat > {job.cout}/machine.json"],
       input=json.dumps(mj) + "\n", timeout=60)
    p = subprocess.Popen(job.command(f"{server['port']}:{server['key']}"), stdout=open(lp, "w"), stderr=subprocess.STDOUT,
                         start_new_session=True)
    running[job.name] = (job, p, lp)
    log(f"START {job.name} attempt {n} pid {p.pid}")


def reap():
    for name, (job, p, lp) in list(running.items()):
        rc = p.poll()
        if rc is None:
            if lp.exists() and "[job] LOCKED" in read(lp) and time.time() - lp.stat().st_mtime > HANG_MIN * 60:
                log(f"HANG {name}: no log output for {HANG_MIN} min -> kill")
                sh(["docker", "exec", "xhand", "pkill", "-f", f"[o]utput_dir {job.cout} "], timeout=60)
                time.sleep(5)
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            continue
        del running[name]
        m = job.metrics()
        n = len(m["episodes"]) if m else 0
        log(f"{'DONE' if job.done() else 'EXIT'} {name} rc={rc} episodes {n}/{len(job.eps)}"
            + (f" success {job.successes()}/{n}" if m else ""))


def gate_open():
    return "[ho_h2] ALL DONE" in read(H2_LOG) and not container_evals()


def main():
    (PI / "logs").mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        js, complete = plan()
        print(f"gate_open={gate_open()} complete={complete} resources={resources()} base={base_ready()}")
        for j in js:
            print(f"{j.name:40s} done={j.done()} ckpt={ready(j.arm, j.step)}")
        return
    lock = open(PI / "logs/driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("pi05_driver already running")
        return
    (PI / "logs/driver.pid").write_text(str(os.getpid()))
    log(f"pi05_driver start pid {os.getpid()} parallel {args.parallel} gpu {GPU}")
    started = waiting_logged = False
    last_launch = last_beat = 0.0
    while True:
        pull_vast()
        reap()
        js, complete = plan()
        if complete and not running:
            stop_server()
            results(js, failed)
            log(f"[pi05] ALL DONE {len(js)} jobs -> {PI}/RESULTS_PI05.md")
            return
        if gave_up >= set(ARMS):
            log("[pi05] FAIL both arms gave up on vast")
            return
        if not started:
            if not (args.no_wait or gate_open()):
                if not waiting_logged:
                    log("[pi05] WAITING for '[ho_h2] ALL DONE' and an empty container; pulling exports meanwhile")
                    waiting_logged = True
                time.sleep(args.poll)
                continue
            started = True
            log(f"[pi05] GATE open; resources gpu_free/ram MiB {resources()}")
        todo = [j for j in js if not j.done() and j.name not in running and j.name not in failed and ready(j.arm, j.step)]
        todo.sort(key=lambda j: (ARMS.index(j.arm), HANDS.index(j.hand), j.lo))
        if server["port"] and sh(["docker", "exec", "xhand", "pgrep", "-f", f"[p]i05_policy_server.py --shared .* --port {server['port']} "],
                                 timeout=60).returncode != 0:
            log(f"SERVER died (port {server['port']}, group {server['group']}): {read(server['log'])[-300:]!r}")
            server.update(group=None, port=None, key=None, log=None)
        groups_alive = {j.group for j, *_ in running.values()}
        if server["group"] and server["group"] not in groups_alive and not any(j.group == server["group"] for j in todo):
            stop_server()  # group finished
        if todo and time.time() - last_launch > LAUNCH_GAP:
            ext = container_evals() - {j.cout for j, *_ in running.values()}
            busy = len(running) + len(ext)
            grp = server["group"] or todo[0].group
            cand = [j for j in todo if j.group == grp]
            if cand and busy < args.parallel:
                j = cand[0]
                gpu_free, ram = resources()
                if j.attempts() >= MAX_ATTEMPTS:
                    failed.add(j.name)
                    log(f"[pi05] FAIL {j.name} after {MAX_ATTEMPTS} attempts (see logs/jobs)")
                elif server["group"] is None:
                    if gpu_free >= NEED_SERVER_MIB + NEED_WORKER_MIB and ram >= NEED_RAM_MIB + 3000:
                        start_server(grp)
                        last_launch = time.time()
                elif gpu_free >= NEED_WORKER_MIB and ram >= NEED_RAM_MIB:
                    launch(j)
                    last_launch = time.time()
        if time.time() - last_beat > 1800:
            last_beat = time.time()
            rd = {a: [s for s in STEPS if ready(a, s)] for a in ARMS}
            log(f"heartbeat started={started} running={sorted(running)} server={server['group']} jobs={len(js)} "
                f"done={sum(j.done() for j in js)} ckpts={json.dumps(rd)} failed={sorted(failed)} gave_up={sorted(gave_up)}")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
