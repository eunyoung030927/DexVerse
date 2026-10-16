# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shear time series of the hybrid tactile videos (experiment): one figure, one panel per video.

    python tactile_hybrid_video_plot.py <out.png> <stem>_tacff_series.npz [...]
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
out, files = sys.argv[1], sys.argv[2:]
fig, axs = plt.subplots(len(files), 1, figsize=(11, 2.6 * len(files)), squeeze=False)
for ax, f in zip(axs[:, 0], files):
    z = np.load(f)
    t = np.arange(z["padS"].shape[0]) * float(z["dt"])
    names = [str(x) for x in z["fingers"]]
    for i, n in enumerate(names):
        if z["padF"][:, i].max() > 0.05:
            ax.plot(t, z["padS"][:, i], lw=1.2, label=f"{n} pad shear")
    ax.set_ylabel("pad shear |F_t| [N]", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t, (z["obj_z"] - z["obj_z"][0]) * 100, "k--", lw=0.8, label="object dz [cm]")
    ax2.set_ylabel("object dz [cm]", fontsize=8)
    ax.set_title(os.path.basename(f).replace("_tacff_series.npz", ""), fontsize=9)
    ax.legend(fontsize=6, loc="upper left", ncol=3)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
axs[-1, 0].set_xlabel("time [s]")
fig.tight_layout()
fig.savefig(out, dpi=120)
print("wrote", out)
