# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Web UI for retargeting runs: task x hand grid, and per (task, hand) the source demo, the kinematic reference and
every SPIDER attempt as video (success or failure), with the object error / success / drop events plotted
under each video and a cursor that follows playback. Videos come from render_watch.py.

Standard library only. Reads the experiments tree (hscho_share: the PC's synced copy of every server's results):
  <root>/<server>/<project>/<run_id>/artifacts/<task>/<demos stem>/epNNN/...

  python3 scripts/retarget/webui.py [--root /mnt/d/hscho_share/experiments] [--port 8765]
  -> http://localhost:8765
"""

import argparse
import json
import mimetypes
import os
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[4] / "experiments" if len(HERE.parents) > 4 else Path("experiments")
ROOT = DEFAULT_ROOT


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


LOG_KEEP = re.compile(r"^(  \S|--- |RESULT|skip|Traceback|\S*Error|object keypoints)")


def tail(p: Path, n=80):
    """Last n retarget progress lines of a run.log (Isaac's own output left out)."""
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 2_000_000))
            lines = [x for x in f.read().decode("utf-8", "replace").splitlines() if LOG_KEEP.match(x)]
            return "\n".join(lines[-n:])
    except Exception:
        return ""


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def runs():
    """Every run with retargeting episodes: [{id, server, project, run, episodes: [...]}]."""
    out = []
    for art in sorted(ROOT.glob("*/*/*/artifacts")):
        eps = sorted(art.glob("*/*/ep[0-9][0-9][0-9]/reference.json"))
        cmd = (read_json(art.parent / "meta.json") or {}).get("command")
        cmd = cmd if isinstance(cmd, str) and "run_queue.py" in cmd else ""
        if not eps and not cmd:
            continue
        run = art.parent
        meta = read_json(run / "meta.json") or {}
        episodes = []
        for ref in eps:
            ep = ref.parent
            targets = []
            for d in sorted(p for p in ep.iterdir() if p.is_dir() and p.name != "viz"):
                st = read_json(d / "status.json") or {}
                running = any(not (d / f"{p.name[:-len('_process.pkl.gz')]}.json").exists()
                              for p in d.glob("*_process.pkl.gz"))
                targets.append({"name": d.name, "state": st.get("state") or ("stage2" if not (d / "kinematic.npz").exists()
                                                                            else "running"),
                                "attempts": st.get("attempts", []), "running": running})
            episodes.append({"key": rel(ep), "task": ep.parents[1].name, "stem": ep.parent.name, "ep": ep.name,
                             "targets": targets})
        # planned but not started (no stage 1 yet): rows from the queue command, every hand waiting
        m_t, m_h, m_e = (re.search(rf"--{k} (.*?)(?: --|$)", cmd) for k in ("tasks", "hands", "episodes"))
        started = {(e["task"], e["ep"]) for e in episodes}
        for task in (m_t.group(1).split() if m_t else []):
            for ep in (m_e.group(1).split() if m_e else ["0"]):
                if (task, f"ep{int(ep):03d}") not in started:
                    episodes.append({"key": f"{rel(art)}/{task}/demos/ep{int(ep):03d}", "task": task, "stem": "demos",
                                     "ep": f"ep{int(ep):03d}", "planned": True,
                                     "targets": [{"name": f"floating_{h}_right", "state": "pending", "attempts": [],
                                                  "running": False} for h in (m_h.group(1).split() if m_h else [])]})
        out.append({"id": rel(run), "server": run.parents[1].name, "project": run.parent.name, "run": run.name,
                    "status": meta.get("status"), "episodes": episodes})
    return out


def detail(key: str):
    """Videos, results and log of one episode (all targets) or of one target dir."""
    d = (ROOT / key).resolve()
    if ROOT.resolve() not in d.parents or not d.is_dir():
        return None
    ep = d if (d / "reference.json").exists() else d.parent
    vids = []

    def add(mp4: Path, label: str):
        side = read_json(mp4.with_suffix(".json")) or {}
        vids.append({"label": label, "mp4": rel(mp4), "mtime": int(mp4.stat().st_mtime), "side": side})

    if (ep / "viz" / "source.mp4").exists():
        add(ep / "viz" / "source.mp4", "source demo")
    res = {}
    if d != ep:
        # kinematic = the stage-2 joints replayed under physics (kinonly), the same run that decides whether
        # SPIDER is needed; the teleported overlay (viz/kinematic.mp4) is not shown
        if (d / "viz" / "kinonly_process.mp4").exists():
            add(d / "viz" / "kinonly_process.mp4", "kinematic (물리 재생)")
        for mp4 in sorted((d / "viz").glob("*.mp4")) if (d / "viz").exists() else []:
            if mp4.name not in ("kinematic.mp4", "kinonly_process.mp4") and not mp4.name.endswith(".part.mp4"):
                add(mp4, mp4.stem.replace("_process", " (process)").replace("_traj", " (result)"))
        for j in sorted(d.glob("kinonly.json")) + sorted(d.glob("spider_a*.json")):
            r = read_json(j)
            if r:
                res[j.stem] = {k: r.get(k) for k in ("task_success", "success_frame", "frames", "minutes",
                                                     "optimised_steps", "retries", "obj_err_mean_cm",
                                                     "obj_err_final_cm", "hold_pass_used")}
    return {"key": rel(ep if d == ep else d), "status": read_json(d / "status.json"),
            "reference": read_json(ep / "reference.json") if d == ep else None,
            "videos": vids, "results": res, "log": tail(d / "run.log") if d != ep else ""}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path == "/":
                return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/runs":
                return self.send(200, json.dumps(runs()).encode())
            if u.path == "/api/detail":
                r = detail(q.get("key", [""])[0])
                return self.send(200 if r else 404, json.dumps(r).encode())
            if u.path.startswith("/files/"):
                return self.file(urllib.parse.unquote(u.path[len("/files/"):]))
            self.send(404, b"{}")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def file(self, relpath):
        p = (ROOT / relpath).resolve()
        if ROOT.resolve() not in p.parents or not p.is_file():
            return self.send(404, b"not found", "text/plain")
        size = p.stat().st_size
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        start, end = 0, size - 1
        if m and (m.group(1) or m.group(2)):          # browsers need ranges to seek in a video
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
        with open(p, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        hdr = {"Accept-Ranges": "bytes"}
        if m:
            hdr["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send(206 if m else 200, body, ctype, hdr)


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retarget Viewer</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--fg:#1d2330;--mut:#6b7385;--line:#dde1e8;--ok:#1f9d55;--bad:#d64545;--run:#d9a400;--none:#b8bec9;--acc:#2f6fde}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14171d;--panel:#1c2028;--fg:#e4e7ee;--mut:#8a93a6;--line:#2c323d;--none:#4a5160}}
:root[data-theme=dark]{--bg:#14171d;--panel:#1c2028;--fg:#e4e7ee;--mut:#8a93a6;--line:#2c323d;--none:#4a5160}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.45 system-ui,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:2;flex-wrap:wrap}
header h1{font-size:15px;margin:0}select,button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg)}
main{display:grid;grid-template-columns:minmax(300px,420px) 1fr;gap:16px;padding:16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}
table{border-collapse:collapse;width:100%}th,td{padding:4px 5px;text-align:center;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:500;font-size:11px}
td.task{text-align:left;font-size:12px;cursor:pointer;white-space:nowrap}td.task:hover{color:var(--acc)}
.cell{display:inline-block;min-width:34px;padding:3px 4px;border-radius:5px;color:#fff;cursor:pointer;font-size:11px}
.cell.sel{outline:2px solid var(--acc);outline-offset:1px}
.s-ok{background:var(--ok)}.s-bad{background:var(--bad)}.s-run{background:var(--run)}.s-none{background:var(--none)}
.legend span{margin-right:10px}.mut{color:var(--mut)}
.vids{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}
.vid{border:1px solid var(--line);border-radius:8px;padding:8px;min-width:0}.vid h3{margin:0 0 6px;font-size:13px;display:flex;justify-content:space-between;gap:6px}
video{width:100%;border-radius:6px;background:#000;display:block}canvas{width:100%;height:110px;display:block;margin-top:6px;cursor:crosshair}
.info{font:12px ui-monospace,monospace;margin-top:4px;min-height:1.4em}
.badge{font-size:11px;padding:1px 6px;border-radius:9px;color:#fff}
pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;max-height:300px;overflow:auto;font-size:11px;white-space:pre-wrap}
.res{font:12px ui-monospace,monospace}.res td,.res th{text-align:left}
#toast{position:fixed;right:16px;bottom:16px;display:flex;flex-direction:column;gap:6px;z-index:5;max-width:min(420px,calc(100vw - 32px))}#toast div{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:8px;padding:8px 10px;box-shadow:0 4px 14px rgba(0,0,0,.15);cursor:pointer}
.new{animation:pulse 1.2s ease-in-out 3}@keyframes pulse{50%{box-shadow:0 0 0 4px var(--acc)}}
.tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
</style></head><body>
<header><h1>Retarget Viewer</h1><select id="run"></select><span class="mut" id="upd"></span>
<span style="flex:1"></span><label>speed <select id="rate"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></label>
<button id="playall">▶ all</button><button id="pauseall">❚❚ all</button></header>
<div id="toast"></div><main><div class="card"><div class="legend mut"><span><span class="cell s-ok">✓</span> success</span><span><span class="cell s-bad">✗</span> failed</span><span><span class="cell s-run">…</span> running</span><span><span class="cell s-none">-</span> waiting</span></div>
<div id="grid"></div></div>
<div class="card" id="detail"><p class="mut">왼쪽 표에서 (task, hand) 칸을 누르면 영상이 나옵니다. task 이름을 누르면 원본 데모만 봅니다.</p></div></main>
<script>
let RUNS=[],SEL=null,RUN=null;
const $=s=>document.querySelector(s);
const short=t=>t.replace(/^Dexverse-/,'').replace(/-v\d+$/,'');
const hand=n=>n.replace(/^floating_/,'').replace(/_(right|left|bimanual)$/,'');
function cls(t){if(t.state==='recorded')return['s-ok','✓'];if(t.state==='failed_all_attempts'||/failed/.test(t.state))return['s-bad','✗'];
 if(t.running||t.state==='retrying'||t.state==='running')return['s-run','…'+(t.attempts.length?t.attempts.length:'')];return['s-none','-']}
let PREV=null,NEWK=new Set();
function toast(msg,k){const d=document.createElement('div');d.textContent=msg;d.onclick=()=>{show(k);d.remove()};$('#toast').prepend(d);setTimeout(()=>d.remove(),600000)}
function snapshot(runs){const m={};runs.forEach(r=>r.episodes.forEach(e=>e.targets.forEach(t=>m[e.key+'/'+t.name]={state:t.state,task:e.task,hand:t.name,server:r.server})));return m}
async function load(){RUNS=await (await fetch('/api/runs')).json();const cur_=snapshot(RUNS);
 if(PREV){for(const[k,v]of Object.entries(cur_)){const o=PREV[k];if(!o||o.state!==v.state){if(['recorded','failed_all_attempts','stage2_failed'].includes(v.state)){NEWK.add(k);
   const msg=`${v.server} ${short(v.task)} ${hand(v.hand)}: ${v.state==='recorded'?'성공 ✓':'실패 ✗'}`;toast(msg,k);document.title='● '+msg}}}}
 PREV=cur_;const s=$('#run'),cur=s.value;
 s.innerHTML='<option value="ALL">전체 서버 (running run 합침)</option>'+RUNS.map(r=>`<option value="${r.id}">${r.server} / ${r.run}</option>`).join('');
 s.value=cur||'ALL';if(!s.value)s.value='ALL';grid();$('#upd').textContent='갱신 '+new Date().toLocaleTimeString()}
function grid(){const v=$('#run').value;RUN=v==='ALL'?{episodes:RUNS.filter(r=>r.status==='running').flatMap(r=>r.episodes.map(e=>({...e,server:r.server})))}:RUNS.find(r=>r.id===v);if(!RUN){$('#grid').innerHTML='<p class=mut>retarget 결과 없음</p>';return}
 const hands=[...new Set(RUN.episodes.flatMap(e=>e.targets.map(t=>hand(t.name))))];
 let h='<table><tr><th style="text-align:left">task / ep</th>'+hands.map(x=>`<th>${x}</th>`).join('')+'</tr>';
 for(const e of RUN.episodes){h+=`<tr><td class="task" data-k="${e.key}">${short(e.task)} <span class=mut>${e.ep}${e.server?' · '+e.server:''}${e.planned?' · 대기':''}</span></td>`;
  for(const x of hands){const t=e.targets.find(t=>hand(t.name)===x);if(!t){h+='<td></td>';continue}const[c,l]=cls(t),k=e.key+'/'+t.name;
   h+=`<td><span class="cell ${c} ${SEL===k?'sel':''} ${NEWK.has(k)?'new':''}" data-k="${k}" title="${t.name}: ${t.state}">${l}</span></td>`}h+='</tr>'}
 $('#grid').innerHTML=h+'</table>';
 document.querySelectorAll('[data-k]').forEach(el=>el.onclick=()=>show(el.dataset.k))}
let DET=null;
async function show(k){if(RUN&&RUN.episodes.some(e=>e.planned&&k.startsWith(e.key)))return;SEL=k;NEWK.delete(k);document.title='Retarget Viewer';history.replaceState(null,'','#'+encodeURIComponent(k));grid();const d=await (await fetch('/api/detail?key='+encodeURIComponent(k))).json();DET=d;
 const parts=k.split('/');let h=`<h2 style="margin:0 0 4px;font-size:15px">${parts.slice(4).join(' / ')}</h2>`;
 if(d.status)h+=`<div class=mut>state: <b>${d.status.state||'-'}</b> · attempts: ${(d.status.attempts||[]).map(a=>a.tag+(a.success?' ✓':' ✗')+(a.minutes?` ${a.minutes}m`:'')).join(', ')||'-'}</div>`;
 const rk=Object.keys(d.results||{});if(rk.length){h+='<table class="res"><tr><th>attempt</th><th>success</th><th>frame</th><th>min</th><th>err mean cm</th><th>err final cm</th><th>retries</th></tr>';
  for(const a of rk){const r=d.results[a];h+=`<tr><td>${a}</td><td>${r.task_success?'✓':'✗'}</td><td>${r.success_frame??'-'}</td><td>${r.minutes}</td><td>${JSON.stringify(r.obj_err_mean_cm)}</td><td>${JSON.stringify(r.obj_err_final_cm)}</td><td>${Object.keys(r.retries||{}).length}</td></tr>`}h+='</table>'}
 if(d.reference&&d.reference.segments)h+=`<div class=mut>manipulation segments: ${d.reference.segments.map(s=>`${s.body} ${s.start}-${s.end}`).join(', ')}</div>`;
 h+='<div class="vids" style="margin-top:10px">'+(d.videos.length?'':'<p class=mut>아직 렌더된 영상 없음 (render_watch가 만드는 중)</p>');
 d.videos.forEach((v,i)=>{const s=v.side||{};let b='';if(s.kind==='process'||s.kind==='traj'){let st=s.status||(s.result?(s.result.task_success?'success':'fail'):'');const fin=d.status&&['recorded','failed_all_attempts','stage2_failed'].includes(d.status.state);if(st==='running'&&(fin||Date.now()/1000-(s.source_mtime||0)>1800))st='stopped';
   b=`<span class="badge ${st==='success'?'s-ok':st==='fail'?'s-bad':st==='stopped'?'s-none':'s-run'}">${{success:'성공',fail:'실패',stopped:'중단됨 (판정 끝남·재시작 전 기록)',running:'진행 중'}[st||'running']||st}</span>`}
  h+=`<div class="vid"><h3><span>${v.label}</span>${b}</h3><video id="v${i}" src="/files/${v.mp4}?t=${v.mtime}" controls muted preload="auto"></video><div class="info" id="i${i}"></div><canvas id="c${i}"></canvas></div>`});
 h+='</div>';if(d.log)h+=`<h3 style="font-size:13px">run.log (tail)</h3><pre>${d.log.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</pre>`;
 $('#detail').innerHTML=h;d.videos.forEach((v,i)=>wire(i,v.side||{}));setRate()}
function wire(i,s){const v=$('#v'+i),c=$('#c'+i),info=$('#i'+i),fps=(s.fps||60)/(s.stride||1);
 const errs=(s.err_cm||[]).map(e=>e?Math.max(...e):null),n=s.n||errs.length||0;
 if(s.kind==='kinematic'&&s.tip_err_cm){s.tip_err_cm.forEach((e,j)=>errs[j]=e?Math.max(...e.filter(x=>x>0),0):null)}
 if(!errs.length&&s.kind==='source'){c.style.display='none'}
 const draw=()=>{const W=c.clientWidth,Hh=110,r=devicePixelRatio||1;c.width=W*r;c.height=Hh*r;const g=c.getContext('2d');g.scale(r,r);
  const cs=getComputedStyle(document.documentElement),col=k=>cs.getPropertyValue(k).trim();g.clearRect(0,0,W,Hh);
  const N=Math.max(n,errs.length,1),mx=Math.max(1,...errs.filter(x=>x!=null)),X=j=>j/(N-1||1)*W,Y=e=>Hh-14-(e/mx)*(Hh-24);
  (s.success||[]).forEach((ok,j)=>{if(ok){g.fillStyle='rgba(31,157,85,.18)';g.fillRect(X(j),0,Math.max(1,W/N),Hh-12)}});
  // segments where the video time runs backward in sim frames (rollbacks)
  (s.events||[]).forEach(ev=>{const c2=ev.type==='resume'?col('--mut'):col('--bad');g.strokeStyle=c2;g.lineWidth=1.5;g.beginPath();g.moveTo(X(ev.i),0);g.lineTo(X(ev.i),Hh-12);g.stroke();
   g.fillStyle=c2;g.font='10px sans-serif';g.fillText(ev.type==='abort'?`abort f${ev.frame}`:ev.type==='resume'?`resume f${ev.frame}`:`drop f${ev.frame}→${ev.back_to} r${ev.retry}`,Math.min(X(ev.i)+2,W-90),10)});
  g.strokeStyle=col('--acc');g.lineWidth=1.2;g.beginPath();let st=false;errs.forEach((e,j)=>{if(e==null){st=false;return}st?g.lineTo(X(j),Y(e)):g.moveTo(X(j),Y(e));st=true});g.stroke();
  g.fillStyle=col('--mut');g.font='10px sans-serif';g.fillText(`${s.kind==='kinematic'?'tip err':'obj err'} max ${mx.toFixed(1)} cm`,4,Hh-2);
  const j=Math.min(N-1,Math.round(v.currentTime*fps));g.strokeStyle=col('--fg');g.lineWidth=1;g.beginPath();g.moveTo(X(j),0);g.lineTo(X(j),Hh-12);g.stroke()};
 const upd=()=>{const j=Math.min(Math.max(n-1,0),Math.round(v.currentTime*fps));let t=`video ${j}/${n}`;
  if(s.frame)t+=` · sim frame ${s.frame[j]}`;else t+=` · frame ${j*(s.stride||1)}`;
  if(errs[j]!=null)t+=` · ${s.kind==='kinematic'?'tip':'obj'} err ${errs[j].toFixed(1)} cm`;if(s.err_cm&&s.err_cm[j]&&s.err_cm[j].length>1)t+=` [${s.err_cm[j].join(', ')}]`;
  if(s.success)t+=s.success[j]?' · ✓success':'';const ev=(s.events||[]).filter(e=>e.i<=j&&e.type!=='resume').pop();if(ev&&j-ev.i<30)t+=` · ⚠ drop (${ev.objects}) → back to ${ev.back_to}`;
  info.textContent=t;draw()};
 v.addEventListener('timeupdate',upd);v.addEventListener('seeked',upd);v.addEventListener('loadedmetadata',upd);
 const loop=()=>{if(!v.paused)upd();v.requestVideoFrameCallback?v.requestVideoFrameCallback(loop):requestAnimationFrame(loop)};loop();
 c.onclick=e=>{const b=c.getBoundingClientRect();v.currentTime=(e.clientX-b.left)/b.width*Math.max(n-1,1)/fps};new ResizeObserver(draw).observe(c)}
function setRate(){document.querySelectorAll('video').forEach(v=>v.playbackRate=+$('#rate').value)}
$('#rate').onchange=setRate;$('#run').onchange=()=>{SEL=null;grid()};
$('#playall').onclick=()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play()});
$('#pauseall').onclick=()=>document.querySelectorAll('video').forEach(v=>v.pause());
const sig=d=>JSON.stringify([d.videos.map(v=>v.mp4+v.mtime),Object.keys(d.results||{}),d.status&&d.status.state]);
async function refreshDetail(){if(!SEL||!DET)return;const d=await (await fetch('/api/detail?key='+encodeURIComponent(SEL))).json();if(!d||sig(d)===sig(DET))return;
 const playing=[...document.querySelectorAll('video')].some(v=>!v.paused);if(!playing){const y=scrollY;await show(SEL);scrollTo(0,y)}
 else if(!$('#refbtn')){const b=document.createElement('button');b.id='refbtn';b.textContent='새 영상·결과 있음 → 갱신';b.onclick=()=>show(SEL);$('#detail').prepend(b)}}
setInterval(refreshDetail,60000);
load().then(()=>{const k=decodeURIComponent(location.hash.slice(1));if(k){const r=RUNS.find(r=>k.startsWith(r.id+'/'));show(k)}});setInterval(load,30000);
</script></body></html>"""


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="experiments root")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    ROOT = Path(a.root).resolve()
    print(f"serving {ROOT} on http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
