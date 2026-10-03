"""
score.py
Score vector fields from an external PIV processor against the exact truth.

Each processor defines "the displacement at a node" its own way, so the truth
is computed to match it:
    midpoint : particle whose (x_A + x_B)/2 lies on the node
               (symmetric window deformation: this code, OpenPIV)
    forward  : particle that starts at the node in frame A
               (only frame B deformed: PIVlab piv_FFTmulti)

Node positions: each processor's reported position in its own pixel indexing
(PIVlab 1 based, OpenPIV 0 based), shifted to this dataset's 0 based indexing:
    PIVlab : x = xtable - 1
    OpenPIV: x = x_openpiv
Both codes interpolate the predictor for window deformation at these reported
positions, which makes them the effective node locations. The geometric window
centre differs by half a pixel; using it instead changes the clean image RMS by
under 0.003 px for either processor.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synth_piv import KarmanStreet, load_config  # noqa: E402


def truth(flow, X, Y, k, convention):
    if convention == "midpoint":
        return flow.truth_displacement(X, Y, k)
    if convention == "forward":
        xb, yb = flow.advect(X, Y, k, -0.5, 0.5)
        return xb - X, yb - Y
    raise ValueError(convention)


def errors(flow, X, Y, U, V, ks, convention):
    """U, V shaped (n, ny, nx); X, Y (ny, nx) in dataset coordinates."""
    ex, ey = np.empty_like(U), np.empty_like(V)
    for j, k in enumerate(ks):
        tx, ty = truth(flow, X, Y, k, convention)
        ex[j], ey[j] = U[j] - tx, V[j] - ty
    return ex, ey


def summary(ex, ey, ref=None):
    e = np.hypot(ex, ey)
    d = {"rms_px": float(np.sqrt(np.mean(e ** 2))), "median_px": float(np.median(e)),
         "p95_px": float(np.percentile(e, 95)), "frac_gt_0p5px": float(np.mean(e > 0.5)),
         "frac_gt_1px": float(np.mean(e > 1.0))}
    if ref is not None:
        d["artifact_induced_rms_px"] = float(np.sqrt(np.mean(np.hypot(*ref) ** 2)))
    return d


def load_flow(data_dir):
    return KarmanStreet(load_config(data_dir))
