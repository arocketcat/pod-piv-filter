"""
rank_rules.py
Compare automatic rules for the number of POD modes to remove, on the same spectra.

    mp_fit    this package: Marchenko-Pastur bulk fitted to the lower 75% of the
              spectrum (aspect ratio and scale fitted), edge x 1.05
    gd        Gavish & Donoho (2014) optimal hard threshold, iid noise, nominal aspect ratio
    screenot  Donoho, Gavish & Romanov (2023) ScreeNOT, noise distribution estimated
              from the spectrum (handles correlated noise); k = 40 upper bound
    mendez    Mendez et al. (2017) plateau criterion as described in the paper:
              smallest r with |sigma_k - sigma_k+1| <= eps1 and |<psi_k, 1>| <= eps2
              for all k >= r, eps1 = 0.01 sigma_plateau, eps2 = 0.01. Implementation
              choices that are mine: sigma_plateau = median singular value, and
              the ones vector is normalised to unit length.
"""
import os
import sys

import numpy as np
from screenot.ScreeNOT import createPseudoNoise, computeOptThreshold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pod_filter import PODFilter, _gd_omega  # noqa: E402


def mendez_rank(sigma, psi, eps1_rel=0.01, eps2=0.01, kmax=200):
    kmax = min(kmax, len(sigma) - 1)
    s_plateau = np.median(sigma)
    one = np.ones(psi.shape[0]) / np.sqrt(psi.shape[0])
    ok_gap = np.abs(np.diff(sigma[:kmax + 1])) <= eps1_rel * s_plateau      # k -> k+1
    ok_orth = np.abs(one @ psi[:, :kmax]) <= eps2
    ok = ok_gap & ok_orth
    # smallest r such that ok[k] holds for every k >= r (within kmax)
    r = kmax
    while r > 0 and ok[r - 1]:
        r -= 1
    return r


def all_rules(stack):
    f = PODFilter(n_modes="auto", max_store=1).fit(stack)
    s, N, P = f.sigma_, len(f.sigma_), f.n_pix_
    beta = N / P
    out = {"mp_fit": f.rank_,
           "gd": int(np.sum(s > _gd_omega(beta) * np.median(s)))}
    fz = createPseudoNoise(s, min(40, (N - 3) // 2), strategy="i")
    out["screenot"] = int(np.sum(s > computeOptThreshold(fz, beta)))
    out["mendez"] = mendez_rank(s, f.psi_)
    return out, s


if __name__ == "__main__":
    d = os.environ.get("POD_PIV_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    rows = []
    for name, n in (("clean_A", 1000), ("dirty_A", 1000), ("dirty_B", 1000), ("dirty_A", 400),
                    ("dirty_A", 200), ("dirty_A", 100), ("dirty_A", 50)):
        X = np.load(f"{d}/{name}.npy", mmap_mode="r")[:n]
        r, s = all_rules(X)
        rows.append((name, n, r))
        print(f"{name} N={n:5d}: " + "  ".join(f"{k}={v}" for k, v in r.items()), flush=True)
