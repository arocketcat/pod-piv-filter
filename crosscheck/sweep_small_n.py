"""usage: python sweep_small_n.py N n_pairs
PIV error vs number of removed modes when POD is fitted on only N images per cavity."""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pod_filter import PODFilter
from piv import PIVSettings, piv_pair, node_grid
from synth_piv import load_config, KarmanStreet
N = int(sys.argv[1]); npairs = int(sys.argv[2])
d = os.environ.get("POD_PIV_DATA", os.path.join(ROOT, "data")); flow = KarmanStreet(load_config(d))
raw = {f: np.load(f"{d}/dirty_{f}.npy", mmap_mode="r")[:N] for f in "AB"}
pod = {f: PODFilter(max_store=30).fit(raw[f]) for f in "AB"}
ys, xs = node_grid(256, 256, 32, 16); Y, X = np.meshgrid(ys, xs, indexing="ij")
ks = np.linspace(0, N - 1, npairs).astype(int)
T = {k: flow.truth_displacement(X, Y, k) for k in ks}
res = {}
for r in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25]:
    e = []
    for k in ks:
        a = pod["A"].transform(raw["A"], idx=[k], r=r)[0]; b = pod["B"].transform(raw["B"], idx=[k], r=r)[0]
        _, _, dx, dy, _ = piv_pair(a, b, PIVSettings()); e.append(np.hypot(dx - T[k][0], dy - T[k][1]))
    e = np.array(e); res[r] = (float(np.sqrt((e ** 2).mean())), float((e > 1).mean()))
    print(f"N={N} r={r:2d}: RMS {res[r][0]:.3f} px, >1px {100*res[r][1]:.2f}%", flush=True)
json.dump(res, open(os.path.join(ROOT, "crosscheck", "out", f"sweep_N{N}.json"), "w"))
