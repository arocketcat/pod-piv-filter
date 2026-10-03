"""
piv.py
Planar PIV processor: FFT cross correlation, multipass with symmetric
(central difference) window deformation, 3 point Gaussian subpixel fit,
normalized median test (Westerweel & Scarano 2005) with median replacement.

Displacements are in pixels per frame pair. Coordinates: x = column, y = row,
node positions are window centres in pixel centre convention.

    from piv import PIVSettings, piv_pair
    ys, xs, dx, dy, info = piv_pair(img_a, img_b, PIVSettings())
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import map_coordinates, spline_filter


@dataclass
class PIVSettings:
    passes: tuple = ((64, 32), (32, 16), (32, 16), (32, 16))  # (window, step) per pass
    deform: bool = True            # symmetric image deformation on passes > 1
    interp_order: int = 3          # B spline order for image deformation
    median_eps: float = 0.2        # px, noise level in the normalized median test
    median_thresh: float = 2.0


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def node_grid(H, W, win, step):
    ys = np.arange(0, H - win + 1, step) + 0.5 * (win - 1)
    xs = np.arange(0, W - win + 1, step) + 0.5 * (win - 1)
    return ys, xs


def correlate(a, b, win, step):
    """Batched circular cross correlation of all interrogation windows.
    Returns R with shape (ny, nx, win, win), zero displacement at [win//2, win//2]."""
    wa = sliding_window_view(a, (win, win))[::step, ::step].astype(np.float32)
    wb = sliding_window_view(b, (win, win))[::step, ::step].astype(np.float32)
    wa = wa - wa.mean(axis=(-2, -1), keepdims=True)
    wb = wb - wb.mean(axis=(-2, -1), keepdims=True)
    R = np.fft.irfft2(np.conj(np.fft.rfft2(wa)) * np.fft.rfft2(wb), s=(win, win))
    return np.fft.fftshift(R, axes=(-2, -1))


def gaussian_peak(R):
    """Integer argmax + 3 point Gaussian fit in each direction.
    Returns (sy, sx, ok) where ok is False when the peak sits on the border."""
    ny, nx, w, _ = R.shape
    idx = R.reshape(ny, nx, -1).argmax(-1)
    py, px = np.divmod(idx, w)
    ok = (py > 0) & (py < w - 1) & (px > 0) & (px < w - 1)
    py, px = np.clip(py, 1, w - 2), np.clip(px, 1, w - 2)
    J, I = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    c = R[J, I, py, px]
    floor = 1e-6 * np.maximum(np.abs(c), 1e-12)
    L = lambda v: np.log(np.maximum(v, floor))
    lc = L(c)
    lym, lyp = L(R[J, I, py - 1, px]), L(R[J, I, py + 1, px])
    lxm, lxp = L(R[J, I, py, px - 1]), L(R[J, I, py, px + 1])
    dy_den = lym - 2 * lc + lyp
    dx_den = lxm - 2 * lc + lxp
    dy = np.where(np.abs(dy_den) > 1e-12, 0.5 * (lym - lyp) / np.where(dy_den == 0, 1, dy_den), 0.0)
    dx = np.where(np.abs(dx_den) > 1e-12, 0.5 * (lxm - lxp) / np.where(dx_den == 0, 1, dx_den), 0.0)
    dy, dx = np.clip(dy, -1, 1), np.clip(dx, -1, 1)
    return py + dy - w // 2, px + dx - w // 2, ok


def _neighbours(a):
    p = np.pad(a, 1, constant_values=np.nan)
    H, W = a.shape
    return np.stack([p[1 + j:1 + j + H, 1 + i:1 + i + W]
                     for j in (-1, 0, 1) for i in (-1, 0, 1) if (j, i) != (0, 0)])


def median_test(dx, dy, eps=0.1, thresh=2.0):
    """Normalized median test on a 3x3 neighbourhood. True = outlier."""
    bad = np.zeros(dx.shape, bool)
    for comp in (dx, dy):
        nb = _neighbours(comp)
        med = np.nanmedian(nb, axis=0)
        rm = np.nanmedian(np.abs(nb - med), axis=0)
        bad |= np.abs(comp - med) / (rm + eps) > thresh
    return bad


def fill_invalid(a, bad, radius=2):
    """Replace flagged vectors with a local least squares plane fit over the
    valid vectors in a (2r+1)^2 neighbourhood. Exact for linear fields, so it
    is unbiased at domain edges where a neighbour median is not. Falls back to
    a larger radius, then to the global median of valid vectors."""
    a = np.asarray(a, np.float64).copy()
    bad = np.asarray(bad, bool)
    if not bad.any():
        return a
    valid = ~bad & np.isfinite(a)
    ny, nx = a.shape
    out = a.copy()
    for j, i in zip(*np.nonzero(bad)):
        for r in (radius, radius + 1, radius + 3):
            j0, j1, i0, i1 = max(0, j - r), min(ny, j + r + 1), max(0, i - r), min(nx, i + r + 1)
            vj, vi = np.nonzero(valid[j0:j1, i0:i1])
            if len(vj) >= 5 and np.ptp(vj) > 0 and np.ptp(vi) > 0:
                dj, di = vj + j0 - j, vi + i0 - i
                w = 1.0 / np.hypot(dj, di)
                M = np.column_stack([np.ones_like(dj), dj, di]) * w[:, None]
                coef = np.linalg.lstsq(M, a[vj + j0, vi + i0] * w, rcond=None)[0]
                out[j, i] = coef[0]
                break
        else:
            out[j, i] = np.median(a[valid]) if valid.any() else 0.0
    return out


def _extrapolate_pad(f, n=2):
    """Pad a node field with n layers of linearly extrapolated ghost nodes."""
    g = np.asarray(f, np.float64)
    for axis in (0, 1):
        if g.shape[axis] < 2:
            g = np.concatenate([np.repeat(g.take([0], axis), n, axis), g,
                                np.repeat(g.take([-1], axis), n, axis)], axis)
            continue
        lo, lo2 = g.take([0], axis), g.take([1], axis)
        hi, hi2 = g.take([-1], axis), g.take([-2], axis)
        pre = [lo + (k + 1) * (lo - lo2) for k in range(n)][::-1]
        post = [hi + (k + 1) * (hi - hi2) for k in range(n)]
        g = np.concatenate(pre + [g] + post, axis)
    return g


def sample_field(ys, xs, f, yq, xq, order=3, n_pad=2):
    """Cubic spline interpolation of a node field f(ys, xs) at query points,
    with linear extrapolation beyond the outermost nodes (constant
    extrapolation would deform edge windows with the wrong displacement)."""
    g = _extrapolate_pad(f, n_pad)
    fy = (yq - ys[0]) / (ys[1] - ys[0]) + n_pad if len(ys) > 1 else np.full_like(yq, n_pad)
    fx = (xq - xs[0]) / (xs[1] - xs[0]) + n_pad if len(xs) > 1 else np.full_like(xq, n_pad)
    return map_coordinates(g, [fy, fx], order=order, mode="nearest")


# --------------------------------------------------------------------------- #
# main routine
# --------------------------------------------------------------------------- #
def piv_pair(a, b, s: PIVSettings = PIVSettings()):
    """Process one image pair.

    Returns ys, xs (node coordinates), dx, dy (displacement, px) and an info
    dict with the outlier mask of the final pass before replacement.
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    H, W = a.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ca = cb = None
    if s.deform and len(s.passes) > 1:
        ca = spline_filter(a, order=s.interp_order, mode="mirror")
        cb = spline_filter(b, order=s.interp_order, mode="mirror")

    ys = xs = dx = dy = None
    for ip, (win, step) in enumerate(s.passes):
        nys, nxs = node_grid(H, W, win, step)
        Y, X = np.meshgrid(nys, nxs, indexing="ij")
        if ip == 0:
            R = correlate(a, b, win, step)
            py = px = 0.0
        else:
            # predictor from the previous pass
            pdx = sample_field(ys, xs, dx, yy, xx)
            pdy = sample_field(ys, xs, dy, yy, xx)
            if s.deform:
                aw = map_coordinates(ca, [yy - 0.5 * pdy, xx - 0.5 * pdx], order=s.interp_order,
                                     mode="mirror", prefilter=False)
                bw = map_coordinates(cb, [yy + 0.5 * pdy, xx + 0.5 * pdx], order=s.interp_order,
                                     mode="mirror", prefilter=False)
                R = correlate(aw, bw, win, step)
                py = sample_field(ys, xs, dy, Y, X)
                px = sample_field(ys, xs, dx, Y, X)
            else:
                raise NotImplementedError("use deform=True")
        sy, sx, ok = gaussian_peak(R)
        ndy, ndx = py + sy, px + sx
        bad = ~ok | median_test(ndx, ndy, s.median_eps, s.median_thresh)
        ys, xs = nys, nxs
        dx, dy = fill_invalid(ndx, bad), fill_invalid(ndy, bad)
    return ys, xs, dx, dy, {"outliers": bad}
