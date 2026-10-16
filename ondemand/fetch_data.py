"""Download the source demos the retargeting reads: dexverse/DexVerse_release demonstrations/<version>/*/<task>/demos.pkl.

Runs inside the Isaac Sim container (huggingface_hub comes from setup.sh). Files land as real files (not cache
symlinks) under $WS/demos/demonstrations/<version>/<category>/<task>/demos.pkl; present ones are skipped.
Assets (robot hands, objects) are fetched once by setup.sh.

Environment: TASKS ("Dexverse-GraspBleach-v1 ..."), DEMOS_DIR (default /workspace/dexverse/demos), HF_TOKEN
(anonymous downloads get rate-limited by the Hub).
"""

import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

REPO = "dexverse/DexVerse_release"
DEST = Path(os.environ.get("DEMOS_DIR", "/workspace/dexverse/demos"))
TASKS = os.environ["TASKS"].split()

try:
    print(f"[hf] authenticated as {HfApi().whoami()['name']}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[hf] WARNING: not authenticated ({type(e).__name__}); downloads may be rate-limited", flush=True)

pats = []
for t in TASKS:
    version = t.rsplit("-", 1)[-1]                       # Dexverse-GraspBleach-v1 -> v1
    pats.append(f"demonstrations/{version}/*/{t}/demos.pkl")
print(f"[demos] {len(TASKS)} tasks from {REPO} -> {DEST}", flush=True)
for attempt in range(30):
    try:
        snapshot_download(REPO, repo_type="dataset", local_dir=str(DEST), allow_patterns=pats, max_workers=4)
        break
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code not in (429, 500, 502, 503, 504) or attempt == 29:
            raise
        wait = 330 if code == 429 else 60
        print(f"  HTTP {code}, resuming in {wait} s (retry {attempt + 1}/29)", flush=True)
        time.sleep(wait)

missing = [t for t in TASKS if not list(DEST.glob(f"demonstrations/*/*/{t}/demos.pkl"))]
for t in TASKS:
    for p in DEST.glob(f"demonstrations/*/*/{t}/demos.pkl"):
        print(f"  {p.relative_to(DEST)}  {p.stat().st_size / 2**20:.0f} MB", flush=True)
if missing:
    sys.exit(f"FATAL: not in {REPO}: {missing}")
print("FETCH OK", flush=True)
