"""
score_all.py
Score OpenPIV and PIVlab results for every condition against the exact truth,
next to this package's own processor (results/metrics.json of the main run).
Writes xcheck_results.json and xcheck_summary.md.
"""
import json
import os
import sys

import numpy as np
import scipy.io as sio

from score import errors, load_flow, summary

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import artifact_node_masks  # noqa: E402
from synth_piv import load_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("POD_PIV_DATA", os.path.join(ROOT, "data"))
OUT = os.path.join(HERE, "out")
flow = load_flow(DATA)
cfg = load_config(DATA)
meta = np.load(f"{DATA}/meta.npz")
hot_pair = (meta["A_hot_amp"].max(1) > 0) | (meta["B_hot_amp"].max(1) > 0)


def load(proc, cond):
    if proc == "openpiv":
        f = f"{OUT}/openpiv_{cond}.npz"
        if not os.path.exists(f):
            return None
        z = np.load(f)
        return z["X"], z["Y"], z["U"], z["V"], "midpoint"
    f = f"{OUT}/pivlab_{cond}.mat"
    if not os.path.exists(f):
        return None
    m = sio.loadmat(f)
    return m["X"] - 1.0, m["Y"] - 1.0, np.moveaxis(m["U"], 2, 0), np.moveaxis(m["V"], 2, 0), "forward"


CONDS = {"openpiv": ["clean", "dirty", "minsub", "pod"],
         "pivlab": ["clean", "dirty", "dirty_highpass", "dirty_clahe", "minsub", "pod"]}
out = {}
for proc, conds in CONDS.items():
    out[proc] = {}
    ref = None
    for cond in conds:
        L = load(proc, cond)
        if L is None:
            continue
        X, Y, U, V, conv = L
        n = U.shape[0]
        ex, ey = errors(flow, X, Y, U, V, range(n), conv)
        if cond == "clean":
            ref = (U, V)
        art = (U - ref[0], V - ref[1]) if ref is not None else None
        ms, mh = artifact_node_masks(cfg, Y[:, 0], X[0, :], 32)
        streak = np.broadcast_to(ms, ex.shape)
        hot = hot_pair[:n, None, None] & mh[None]
        out[proc][cond] = {
            "all": summary(ex, ey, art),
            "windows_on_fixed_streaks": summary(ex[streak], ey[streak],
                                                None if art is None else (art[0][streak], art[1][streak])),
            "hotspot_windows_when_flashing": summary(ex[hot], ey[hot],
                                                     None if art is None else (art[0][hot], art[1][hot])),
            "n_pairs": int(n), "grid": list(U.shape[1:]), "truth_convention": conv}

mine = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
out["this_package"] = {c: {"all": mine["overall"][c],
                           "windows_on_fixed_streaks": mine["subsets"]["windows_on_fixed_streaks"][c],
                           "hotspot_windows_when_flashing": mine["subsets"]["hotspot_windows_when_flashing"][c]}
                       for c in ("clean", "dirty", "minsub", "pod")}
json.dump(out, open(os.path.join(ROOT, "results", "crosscheck.json"), "w"), indent=2)

LBL = {"clean": "Clean images", "dirty": "Corrupted, unfiltered", "dirty_highpass": "Corrupted, PIVlab high-pass filter", "dirty_clahe": "Corrupted, PIVlab CLAHE",
       "minsub": "Corrupted, ensemble min subtraction", "pod": "Corrupted, POD filtered TIFFs"}
PROCS = [("this_package", "This package"), ("openpiv", "OpenPIV 0.26.1"), ("pivlab", "PIVlab 3.09 (Octave)")]
lines = ["# Cross check with independent PIV processors\n",
         "All three processors read the same 16 bit TIFFs (1000 pairs). POD filtering was done once, "
         "with `pod_filter.py` on the TIFFs (10 modes per cavity), before any processor saw the images. "
         "Same pass schedule everywhere: 64 px (step 32), then three 32 px passes (step 16).\n"]
for block, title in (("all", "All vectors"), ("windows_on_fixed_streaks", "Windows on the fixed streaks"),
                     ("hotspot_windows_when_flashing", "Hotspot windows while flashing")):
    lines.append(f"\n## {title}: RMS error (px), and share of vectors off by more than 1 px\n")
    lines.append("| Condition | " + " | ".join(p[1] for p in PROCS) + " |")
    lines.append("|---" * (len(PROCS) + 1) + "|")
    for cond in ["clean", "dirty", "dirty_highpass", "dirty_clahe", "minsub", "pod"]:
        cells = []
        for key, _ in PROCS:
            r = out.get(key, {}).get(cond)
            cells.append("" if r is None else f"{r[block]['rms_px']:.3f} ({100 * r[block]['frac_gt_1px']:.2f}%)")
        if any(cells):
            lines.append(f"| {LBL[cond]} | " + " | ".join(cells) + " |")
lines.append("\n## Artifact induced RMS (px): |d_condition - d_clean| from the same processor\n")
lines.append("| Condition | " + " | ".join(p[1] for p in PROCS) + " |")
lines.append("|---" * (len(PROCS) + 1) + "|")
for cond in ["dirty", "dirty_highpass", "dirty_clahe", "minsub", "pod"]:
    cells = []
    for key, _ in PROCS:
        r = out.get(key, {}).get(cond)
        cells.append("" if r is None else f"{r['all'].get('artifact_induced_rms_px', float('nan')):.3f}")
    if any(cells):
        lines.append(f"| {LBL[cond]} | " + " | ".join(cells) + " |")
open(os.path.join(ROOT, "results", "crosscheck.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
