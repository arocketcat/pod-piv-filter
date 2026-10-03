"""
synth_piv.py
Synthetic double frame PIV dataset with exact ground truth.

Flow
    A time dependent Karman vortex street: two staggered rows of Lamb-Oseen
    vortices of opposite sign convecting downstream on top of a uniform stream.
    Each infinite row is summed in closed form (cot kernel) with a local core
    correction, so the field is exact, smooth, periodic and has no seams as the
    street convects. Particles are advected with RK4 through the time dependent
    field, so the truth includes curvature and acceleration effects.

Images
    Gaussian particle images integrated over each pixel (erf form), random
    diameter, Gaussian light sheet with a small out of plane displacement,
    photon shot noise, read noise, dark offset, 12 bit quantization, saturation.

Artifacts (added only to the "dirty" copy; each frame gets its own draw)
    1. static background glow near the bottom wall             (constant)
    2. laser reflection streaks at fixed locations, with shot to shot
       amplitude fluctuation and sub pixel jitter              (low rank)
    3. hotspot flashes at fixed locations, intermittent         (low rank)
    4. global flashes, whole frame brightening with gradient   (low rank)
    5. random streaks at random positions and angles           (NOT low rank)
       Item 5 is intentional: POD cannot remove artifacts that do not repeat
       in space across the ensemble, and the benchmark should show that.

The clean and dirty copies share the same particle images, so any difference
in PIV accuracy between them is caused by the artifacts (and their shot noise).

Output (in --out directory)
    clean_A.npy, clean_B.npy, dirty_A.npy, dirty_B.npy   uint16 (n_pairs, H, W)
    meta.npz          per frame artifact flags/amplitudes + config
    optional TIFFs    --tiff, for use in PIVlab / DaVis / OpenPIV

Usage
    python synth_piv.py --out data --pairs 1000          # 2000 images per copy
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.special import erf


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class SynthConfig:
    n_pairs: int = 1000            # 2 frames per pair -> 2000 images per copy
    H: int = 256
    W: int = 256
    seed: int = 7

    # particles
    ppp: float = 0.05              # particles per pixel
    d_mean: float = 2.8            # particle image diameter (e^-2), px
    d_std: float = 0.4
    I_max: float = 900.0           # peak counts for a particle at sheet centre
    sheet_sigma: float = 0.6       # Gaussian sheet width, in units of z range [-1, 1]
    dz_out: float = 0.10           # out of plane displacement between frames (same units)

    # camera
    offset: float = 100.0          # dark level, counts
    e_per_count: float = 2.0       # photoelectrons per count (sets shot noise)
    read_noise: float = 4.0        # counts rms
    bit_depth: int = 12

    # flow (all in px and px per inter frame time)
    U0: float = 6.0                # free stream displacement
    conv_frac: float = 0.8         # vortex convection speed / U0
    wavelength: float = 160.0      # streamwise vortex spacing
    row_sep: float = 64.0          # cross stream distance between rows
    rc: float = 24.0               # Lamb-Oseen core radius
    v_peak: float = 2.0            # peak swirl displacement of one vortex
    street_step: float = 13.7      # street advance between successive pairs

    # artifacts
    glow_amp: float = 150.0
    streaks: list = field(default_factory=lambda: [
        # (x_frac, y_frac, angle_deg, length_frac, width_px, amplitude)
        (0.50, 0.78, 2.0, 1.20, 1.2, 700.0),    # reflection off a model surface
        (0.32, 0.40, 35.0, 0.60, 1.0, 500.0),   # window scratch / ghost
        (0.86, 0.50, 88.0, 0.70, 1.4, 600.0),   # edge reflection
    ])
    streak_cv: float = 0.35        # log normal amplitude fluctuation per frame
    streak_jitter: float = 0.3     # perpendicular jitter per frame, px rms
    hotspots: list = field(default_factory=lambda: [
        # (x_frac, y_frac, sigma_px)
        (0.25, 0.30, 20.0),
        (0.70, 0.62, 24.0),
    ])
    hotspot_prob: float = 0.25
    hotspot_amp: tuple = (600.0, 2800.0)
    global_prob: float = 0.08
    global_amp: tuple = (300.0, 900.0)
    random_streak_prob: float = 0.03
    random_streak_amp: tuple = (400.0, 900.0)


# --------------------------------------------------------------------------- #
# exact analytic flow: periodic Karman street of Lamb-Oseen vortices
# --------------------------------------------------------------------------- #
class KarmanStreet:
    """Displacement rate field (px per inter frame time).

    Pair k, intra pair time tau in [-0.5, 0.5] (tau = 0 is mid exposure).
    """

    # max of (1 - exp(-s^2))/s occurs at s = 1.1209 with value 0.63817
    _LO_PEAK = 0.638170

    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        self.lam = cfg.wavelength
        self.Uc = cfg.conv_frac * cfg.U0
        self.gamma = 2.0 * np.pi * cfg.rc * cfg.v_peak / self._LO_PEAK
        yc = 0.5 * cfg.H
        # (row y, circulation, x offset): staggered rows of opposite sign,
        # signs chosen (image coordinates, y down) to give a wake velocity deficit
        self.rows = [(yc - 0.5 * cfg.row_sep, +self.gamma, 0.0),
                     (yc + 0.5 * cfg.row_sep, -self.gamma, 0.5 * self.lam)]

    def _row(self, x, y, x0, y0, gam):
        """Infinite row of Lamb-Oseen vortices, returns (u, v).

        Point vortex row (closed form):  u - iv = -i G/(2 lam) cot(pi z/lam)
        Split into  [cot(a) - 1/a]  (smooth)  +  1/a  (nearest image), and give
        the nearest image its Lamb-Oseen core:  (1 - exp(-r^2/rc^2)) conj(z)/r^2,
        which is finite at r = 0. The two neighbouring images get the (tiny)
        core correction as well.
        """
        lam, rc = self.lam, self.cfg.rc
        zr = (x - x0) - lam * np.round((x - x0) / lam)       # nearest image, real part
        z = zr + 1j * (y - y0)
        a = np.pi * z / lam
        small = np.abs(a) < 1e-4
        a_safe = np.where(small, 1.0 + 0j, a)
        reg = np.where(small, -a / 3.0, np.cos(a_safe) / np.sin(a_safe) - 1.0 / a_safe)
        w = -1j * gam / (2.0 * lam) * reg
        r2 = z.real ** 2 + z.imag ** 2
        g = np.where(r2 < 1e-12, 1.0 / rc ** 2, -np.expm1(-r2 / rc ** 2) / np.maximum(r2, 1e-300))
        w = w - 1j * gam / (2.0 * np.pi) * np.conj(z) * g
        for m in (-1, 1):
            zm = z + m * lam
            w = w + 1j * gam / (2.0 * np.pi * zm) * np.exp(-np.abs(zm) ** 2 / rc ** 2)
        return w.real, -w.imag

    def velocity(self, x, y, k, tau):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shift = self.cfg.street_step * k + self.Uc * tau
        u = np.full(x.shape, self.cfg.U0)
        v = np.zeros(x.shape)
        for y0, gam, xoff in self.rows:
            du, dv = self._row(x, y, xoff + shift, y0, gam)
            u += du
            v += dv
        return u, v

    def advect(self, x, y, k, t0, t1, nsub=4):
        """RK4 integration of particle paths from tau=t0 to tau=t1."""
        h = (t1 - t0) / nsub
        t = t0
        for _ in range(nsub):
            k1 = self.velocity(x, y, k, t)
            k2 = self.velocity(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1], k, t + 0.5 * h)
            k3 = self.velocity(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1], k, t + 0.5 * h)
            k4 = self.velocity(x + h * k3[0], y + h * k3[1], k, t + h)
            x = x + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            y = y + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            t += h
        return x, y

    def pair_positions(self, xm, ym, k):
        """Positions at frame A (tau=-0.5) and frame B (tau=+0.5) from mid exposure positions."""
        xa, ya = self.advect(xm, ym, k, 0.0, -0.5)
        xb, yb = self.advect(xm, ym, k, 0.0, +0.5)
        return xa, ya, xb, yb

    def truth_displacement(self, xg, yg, k, iters=3):
        """Exact displacement (dx, dy) of the particle whose spatial midpoint
        (x_A + x_B)/2 lies at (xg, yg). This is what a central difference
        (symmetric window deformation) PIV estimate is centred on."""
        xg = np.asarray(xg, float)
        yg = np.asarray(yg, float)
        xm, ym = xg.copy(), yg.copy()
        for _ in range(iters):
            xa, ya, xb, yb = self.pair_positions(xm, ym, k)
            xm = xm - (0.5 * (xa + xb) - xg)
            ym = ym - (0.5 * (ya + yb) - yg)
        xa, ya, xb, yb = self.pair_positions(xm, ym, k)
        return xb - xa, yb - ya


# --------------------------------------------------------------------------- #
# particle image rendering
# --------------------------------------------------------------------------- #
_OFF = np.arange(-3, 4)  # 7 x 7 support, enough for d <= 4 px


def render_particles(x, y, d, I0, H, W):
    """Sum of pixel integrated Gaussian particle images."""
    s = np.sqrt(8.0) / d                                     # erf scale
    cx = np.rint(x).astype(np.int64)[:, None] + _OFF         # (N, 7)
    cy = np.rint(y).astype(np.int64)[:, None] + _OFF
    ex = 0.5 * (erf((cx + 0.5 - x[:, None]) * s[:, None]) - erf((cx - 0.5 - x[:, None]) * s[:, None]))
    ey = 0.5 * (erf((cy + 0.5 - y[:, None]) * s[:, None]) - erf((cy - 0.5 - y[:, None]) * s[:, None]))
    # normalise so a pixel centred particle peaks near I0
    norm = (0.5 * (erf(0.5 * s) - erf(-0.5 * s))) ** 2
    val = (I0 / norm)[:, None, None] * ey[:, :, None] * ex[:, None, :]
    rows = np.broadcast_to(cy[:, :, None], val.shape)
    cols = np.broadcast_to(cx[:, None, :], val.shape)
    m = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    img = np.bincount((rows[m] * W + cols[m]), weights=val[m], minlength=H * W)
    return img.reshape(H, W)


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
class ArtifactModel:
    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        H, W = cfg.H, cfg.W
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        self.xx, self.yy = xx, yy
        # static glow near the bottom wall, slightly non uniform along x
        self.glow = cfg.glow_amp * np.exp(-((yy - 0.95 * H) / (0.07 * H)) ** 2) \
            * (0.75 + 0.25 * np.cos(2 * np.pi * xx / W))
        # geometry of the fixed streaks
        self.streak_geom = []
        for xf, yf, ang, lf, wid, amp in cfg.streaks:
            self.streak_geom.append(self._line_geometry(xf * W, yf * H, ang, lf * max(H, W), wid) + (amp,))
        self.hot = [np.exp(-((xx - xf * W) ** 2 + (yy - yf * H) ** 2) / (2 * s ** 2))
                    for xf, yf, s in cfg.hotspots]
        self.global_pattern = 0.6 + 0.4 * xx / W + 0.15 * yy / H

    def _line_geometry(self, x0, y0, ang_deg, length, width):
        t = np.deg2rad(ang_deg)
        c, s = np.cos(t), np.sin(t)
        along = (self.xx - x0) * c + (self.yy - y0) * s
        perp = -(self.xx - x0) * s + (self.yy - y0) * c
        # smooth ends and a fixed along line modulation (reflections are never uniform)
        half = 0.5 * length
        env = 0.5 * (1 - np.tanh((np.abs(along) - half) / 6.0))
        env *= 0.8 + 0.2 * np.cos(2 * np.pi * along / 57.0 + 0.7)
        return along, perp, env, width

    def _streak(self, perp, env, width, amp, delta):
        return amp * env * np.exp(-0.5 * ((perp - delta) / width) ** 2)

    def draw(self, rng):
        """One frame of artifact intensity plus its metadata."""
        cfg = self.cfg
        img = self.glow.copy()
        meta = {}
        amps = []
        for along, perp, env, width, amp in self.streak_geom:
            a = amp * rng.lognormal(0.0, cfg.streak_cv)
            img += self._streak(perp, env, width, a, rng.normal(0.0, cfg.streak_jitter))
            amps.append(a)
        meta["streak_amp"] = amps
        hot_amp = []
        for pat in self.hot:
            a = rng.uniform(*cfg.hotspot_amp) if rng.random() < cfg.hotspot_prob else 0.0
            img += a * pat
            hot_amp.append(a)
        meta["hot_amp"] = hot_amp
        g = rng.uniform(*cfg.global_amp) if rng.random() < cfg.global_prob else 0.0
        img += g * self.global_pattern
        meta["global_amp"] = g
        r = 0.0
        if rng.random() < cfg.random_streak_prob:
            r = rng.uniform(*cfg.random_streak_amp)
            along, perp, env, width = self._line_geometry(
                rng.uniform(0.1, 0.9) * cfg.W, rng.uniform(0.1, 0.9) * cfg.H,
                rng.uniform(0, 180), rng.uniform(0.3, 0.8) * cfg.W, rng.uniform(1.0, 2.0))
            img += self._streak(perp, env, width, r, 0.0)
        meta["random_streak_amp"] = r
        return img, meta


def camera(signal, cfg: SynthConfig, rng):
    """Shot noise + read noise + offset + quantisation + saturation."""
    e = rng.poisson(np.maximum(signal, 0.0) * cfg.e_per_count) / cfg.e_per_count
    counts = e + cfg.offset + rng.normal(0.0, cfg.read_noise, signal.shape)
    return np.clip(np.rint(counts), 0, 2 ** cfg.bit_depth - 1).astype(np.uint16)


# --------------------------------------------------------------------------- #
# dataset generation
# --------------------------------------------------------------------------- #
def generate(cfg: SynthConfig, out: str, tiff: bool = False, verbose: bool = True):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    flow = KarmanStreet(cfg)
    art = ArtifactModel(cfg)
    H, W, N = cfg.H, cfg.W, cfg.n_pairs
    mm = {name: np.lib.format.open_memmap(os.path.join(out, f"{name}.npy"), mode="w+",
                                          dtype=np.uint16, shape=(N, H, W))
          for name in ("clean_A", "clean_B", "dirty_A", "dirty_B")}
    meta = {f"{f}_{key}": [] for f in "AB" for key in
            ("streak_amp", "hot_amp", "global_amp", "random_streak_amp")}

    pad = 14.0  # particles can enter from outside the field of view
    area = (H + 2 * pad) * (W + 2 * pad)
    n_part = int(round(cfg.ppp * area))

    for k in range(N):
        xm = rng.uniform(-pad, W + pad, n_part)
        ym = rng.uniform(-pad, H + pad, n_part)
        d = np.clip(rng.normal(cfg.d_mean, cfg.d_std, n_part), 1.5, 4.0)
        z = rng.uniform(-1.0, 1.0, n_part)
        za, zb = z - 0.5 * cfg.dz_out, z + 0.5 * cfg.dz_out
        Ia = cfg.I_max * np.exp(-0.5 * (za / cfg.sheet_sigma) ** 2)
        Ib = cfg.I_max * np.exp(-0.5 * (zb / cfg.sheet_sigma) ** 2)
        xa, ya, xb, yb = flow.pair_positions(xm, ym, k)
        pa = render_particles(xa, ya, d, Ia, H, W)
        pb = render_particles(xb, yb, d, Ib, H, W)
        for frame, p in (("A", pa), ("B", pb)):
            a_img, a_meta = art.draw(rng)
            mm[f"clean_{frame}"][k] = camera(p, cfg, rng)
            mm[f"dirty_{frame}"][k] = camera(p + a_img, cfg, rng)
            for key, val in a_meta.items():
                meta[f"{frame}_{key}"].append(val)
        if verbose and (k + 1) % 100 == 0:
            print(f"  generated {k + 1}/{N} pairs", flush=True)

    for m in mm.values():
        m.flush()
    np.savez(os.path.join(out, "meta.npz"),
             config=json.dumps(asdict(cfg)),
             **{key: np.asarray(v) for key, v in meta.items()})

    if tiff:
        from PIL import Image
        for copy in ("clean", "dirty"):
            d_out = os.path.join(out, "tiff", copy)
            os.makedirs(d_out, exist_ok=True)
            for k in range(N):
                for frame in "AB":
                    Image.fromarray(np.asarray(mm[f"{copy}_{frame}"][k])).save(
                        os.path.join(d_out, f"pair_{k:04d}_{frame}.tif"))
    return out


def load_config(data_dir: str) -> SynthConfig:
    meta = np.load(os.path.join(data_dir, "meta.npz"), allow_pickle=False)
    d = json.loads(str(meta["config"]))
    d["streaks"] = [tuple(s) for s in d["streaks"]]
    d["hotspots"] = [tuple(s) for s in d["hotspots"]]
    for key in ("hotspot_amp", "global_amp", "random_streak_amp"):
        d[key] = tuple(d[key])
    return SynthConfig(**d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--pairs", type=int, default=1000, help="image pairs (2 images each)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tiff", action="store_true", help="also write 16 bit TIFFs")
    a = ap.parse_args()
    cfg = SynthConfig(n_pairs=a.pairs, H=a.size, W=a.size, seed=a.seed)
    print(f"Generating {cfg.n_pairs} pairs ({2 * cfg.n_pairs} images per copy) at {cfg.H}x{cfg.W}")
    generate(cfg, a.out, tiff=a.tiff)
    print("done ->", a.out)
