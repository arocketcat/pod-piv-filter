"""
pod_filter.py
POD (snapshot SVD) filter that removes spatially coherent, repeatable
artifacts (laser reflections, streaks, flashes, glare, background glow) from an
ensemble of PIV images and reconstructs particle only images.

Idea (Mendez et al. 2017, Exp. Therm. Fluid Sci. 80:181-192)
    Stack the N images of one laser cavity as rows of X (N x Npix).
    Artifacts that repeat in space from image to image, even with random
    amplitude, jitter or intermittency, are low rank: a few spatial modes carry
    most of their energy. Particle images from uncorrelated double frame
    recordings are statistically independent from snapshot to snapshot, so
    their energy spreads almost evenly over all N modes (a flat "bulk").
    Removing the leading modes that stand above that bulk removes the
    artifacts and keeps almost all of the particle signal.

        X = sum_k  sigma_k psi_k phi_k^T          (SVD, psi temporal, phi spatial)
        X_particles = X - sum_{k<r} sigma_k psi_k phi_k^T = (I - Psi_r Psi_r^T) X

Caveats worth knowing
    * Run it separately on frame A and frame B ensembles (two laser cavities).
    * Artifacts that do NOT repeat in space (a streak at a random position,
      a flash with a random location) are not low rank and are not removed.
    * Time resolved sequences: consecutive particle images are correlated, so
      particles leak into leading modes. Subsample in time or use the
      convolution form (Mendez) or a sliding ensemble.
    * Saturated pixels are lost information; no filter can restore them.
    * Removing a few extra bulk modes costs little (each removes ~1/N of the
      particle energy); keeping an artifact mode costs a lot. Err high.

Rank selection
    'auto' (default): modes whose singular value exceeds the upper edge of the
    particle bulk. The bulk is modelled as a Marchenko-Pastur law whose
    effective aspect ratio and scale are fitted robustly to the lower
    three quarters of the spectrum, times a small safety factor (1.05) that
    covers finite N fluctuation of the largest bulk values.
    'gd'          : Gavish-Donoho (2014) optimal hard threshold, iid noise model.
    int           : fixed number of modes.

Usage as a library
    f = PODFilter(n_modes="auto").fit(stack_A)      # (N, H, W)
    clean_A = f.transform(stack_A)                   # float32, particles only

Command line, for real data (one glob per cavity):
    python pod_filter.py "run1/*_A.tif" "run1/*_B.tif" --out filtered --modes auto
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def _gd_omega(beta):
    """Gavish-Donoho approximation of omega(beta) for unknown noise level."""
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43


class PODFilter:
    def __init__(self, n_modes="auto", clip_negative=True, safety=1.05, chunk=8192, max_store=200):
        self.n_modes = n_modes
        self.clip_negative = clip_negative
        self.safety = safety
        self.chunk = chunk
        self.max_store = max_store

    # ---------------------------------------------------------------- fit
    def fit(self, stack):
        stack = np.asarray(stack) if not isinstance(stack, np.memmap) else stack
        N = stack.shape[0]
        self.shape_ = stack.shape[1:]
        P = int(np.prod(self.shape_))
        X = stack.reshape(N, P)
        # Gram (snapshot correlation) matrix, accumulated in float64 by pixel chunks
        K = np.zeros((N, N))
        for c0 in range(0, P, self.chunk):
            Xc = np.asarray(X[:, c0:c0 + self.chunk], np.float64)
            K += Xc @ Xc.T
        lam, psi = np.linalg.eigh(K)
        order = np.argsort(lam)[::-1]
        lam, psi = np.clip(lam[order], 0, None), psi[:, order]
        self.sigma_ = np.sqrt(lam)
        self.psi_ = psi                                    # temporal modes (N x N)
        self.n_pix_ = P
        self.rank_ = self._select_rank()
        # projection coefficients A = Psi_r^T X for the leading modes (kept for
        # fast reconstruction with any r <= max_store, e.g. for sweeps)
        r_store = int(min(max(self.rank_, 1) if self.max_store is None else self.max_store, N))
        self.r_store_ = max(r_store, self.rank_)
        A = np.zeros((self.r_store_, P), np.float32)
        Pr = self.psi_[:, :self.r_store_].astype(np.float64)
        for c0 in range(0, P, self.chunk):
            A[:, c0:c0 + self.chunk] = (Pr.T @ np.asarray(X[:, c0:c0 + self.chunk], np.float64)).astype(np.float32)
        self.coef_ = A
        return self

    def _select_rank(self):
        s = self.sigma_
        N = len(s)
        if isinstance(self.n_modes, (int, np.integer)):
            self.threshold_ = s[self.n_modes] if self.n_modes < N else 0.0
            return int(self.n_modes)
        beta_nominal = min(N, self.n_pix_) / max(N, self.n_pix_)
        if self.n_modes == "gd":
            tau = _gd_omega(beta_nominal) * np.median(s)
        elif self.n_modes == "auto":
            tau = self._mp_edge(s)
        else:
            raise ValueError("n_modes must be int, 'auto' or 'gd'")
        self.threshold_ = tau
        return int(np.sum(s > tau))

    def _mp_edge(self, s):
        """Fit a Marchenko-Pastur bulk to the eigenvalues (sigma^2) by quantile
        matching on the lower part of the spectrum, where artifacts do not
        live, and return the bulk upper edge (times a safety factor).

        Particle pixels are spatially correlated (a particle covers ~9 px), so
        the effective number of independent pixels, and hence beta, is not
        N/Npix. It is fitted instead of assumed."""
        lam = np.sort(s ** 2)                      # ascending
        N = len(lam)
        qs = np.linspace(0.05, 0.75, 29)           # quantiles used for the fit
        emp = np.quantile(lam, qs)
        best = None
        for beta in np.geomspace(1e-3, 1.0, 120):
            lo, hi = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
            x = np.linspace(lo, hi, 4002)[1:]
            pdf = np.sqrt(np.maximum((hi - x) * (x - lo), 0)) / (2 * np.pi * beta * x)
            cdf = np.cumsum(pdf)
            cdf /= cdf[-1]
            theo = np.interp(qs, cdf, x)
            scale = np.dot(theo, emp) / np.dot(theo, theo)   # least squares scale
            err = np.sum((emp - scale * theo) ** 2) / np.sum(emp ** 2)
            if best is None or err < best[0]:
                best = (err, beta, scale, hi)
        _, beta, scale, hi = best
        self.mp_beta_, self.mp_scale_ = beta, scale
        return self.safety * np.sqrt(scale * hi)

    # ---------------------------------------------------------- transform
    def background(self, idx=None, r=None):
        """Low rank (artifact) part for snapshots idx using r modes."""
        r = self.rank_ if r is None else int(r)
        if r > self.r_store_:
            raise ValueError(f"r={r} exceeds stored modes ({self.r_store_}); refit with larger max_store")
        psi = self.psi_ if idx is None else self.psi_[np.atleast_1d(idx)]
        bg = psi[:, :r].astype(np.float32) @ self.coef_[:r]
        return bg.reshape((-1,) + self.shape_)

    def transform(self, stack, idx=None, r=None):
        """Particle only reconstruction. `stack` must be the fitted ensemble
        (or the matching subset when idx is given)."""
        idx_arr = None if idx is None else np.atleast_1d(idx)
        X = np.asarray(stack if idx is None else stack[idx_arr], np.float32)
        out = X.reshape((-1,) + self.shape_) - self.background(idx_arr, r)
        if self.clip_negative:
            np.maximum(out, 0, out=out)
        return out if (idx is None or np.ndim(idx) > 0) else out[0]

    def spatial_mode(self, k):
        return (self.coef_[k] / max(self.sigma_[k], 1e-30)).reshape(self.shape_)


# --------------------------------------------------------------------------- #
# command line interface for real image sets
# --------------------------------------------------------------------------- #
def _read_stack(files):
    from PIL import Image
    return np.stack([np.asarray(Image.open(f)) for f in files])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("globs", nargs="+", help="one glob per laser cavity, e.g. '*_A.tif' '*_B.tif'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="auto", help="'auto', 'gd', or an integer")
    ap.add_argument("--no-clip", action="store_true", help="keep negative values (floating point output)")
    a = ap.parse_args()
    from PIL import Image
    os.makedirs(a.out, exist_ok=True)
    modes = int(a.modes) if a.modes.isdigit() else a.modes
    for g in a.globs:
        files = sorted(glob.glob(g))
        if not files:
            print("no files for", g)
            continue
        stack = _read_stack(files)
        dtype = stack.dtype
        f = PODFilter(n_modes=modes, clip_negative=not a.no_clip, max_store=None).fit(stack)
        clean = f.transform(stack)
        print(f"{g}: {len(files)} images, removed {f.rank_} modes (threshold {f.threshold_:.3g})")
        for fn, im in zip(files, clean):
            name = os.path.join(a.out, os.path.basename(fn))
            if np.issubdtype(dtype, np.integer) and not a.no_clip:
                Image.fromarray(np.clip(np.rint(im), 0, np.iinfo(dtype).max).astype(dtype)).save(name)
            else:
                Image.fromarray(im.astype(np.float32)).save(os.path.splitext(name)[0] + ".tif")


if __name__ == "__main__":
    main()
