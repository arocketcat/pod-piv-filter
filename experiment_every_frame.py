"""
experiment_every_frame.py
What if EVERY image in the dataset has streaks and flashes?

All variants: the three fixed reflection streaks (every frame, amplitude
flicker + jitter), the wall glow, and two hotspot flashes plus a global flash
in EVERY frame (no clean frames at all). Variants differ only in where the
hotspots sit:

    fixed   same place in every image                         (repeatable)
    drift   moves between pairs, sigma = 6 px                 (e.g. model vibration)
    random  anywhere in the image, new place every pair       (not repeatable)
    random_streaks  hotspots fixed, plus two THIN streaks at a random position
            and angle every pair                              (not repeatable, sharp)

In drift and random, the hotspot is at the same place in frames A and B of a
pair, because A and B are only microseconds apart: the model has not moved,
only the laser pulse (amplitude) differs. That is the realistic worst case.

Conditions: clean, dirty, POD (auto rank), spatial high pass (image minus a
Gaussian blur, sigma 5 px, clipped at 0), and POD followed by high pass.

    python experiment_every_frame.py fixed|drift|random|random_streaks [--pairs 400] [--eval 150]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from synth_piv import SynthConfig, KarmanStreet, ArtifactModel, render_particles, camera
from piv import PIVSettings, piv_pair, node_grid
from pod_filter import PODFilter

CONDS = ["clean", "dirty", "pod", "highpass", "pod+highpass"]


def make_dataset(variant, n_pairs, seed):
    cfg = SynthConfig(n_pairs=n_pairs, seed=seed)
    rng = np.random.default_rng(seed)
    flow, art = KarmanStreet(cfg), ArtifactModel(cfg)
    H, W = cfg.H, cfg.W
    yy, xx = art.yy, art.xx
    pad = 14.0
    n_part = int(round(cfg.ppp * (H + 2 * pad) * (W + 2 * pad)))
    out = {k: np.zeros((n_pairs, H, W), np.uint16) for k in ("clean_A", "clean_B", "dirty_A", "dirty_B")}
    nominal = [(xf * W, yf * H, s) for xf, yf, s in cfg.hotspots]
    for k in range(n_pairs):
        xm, ym = rng.uniform(-pad, W + pad, n_part), rng.uniform(-pad, H + pad, n_part)
        d = np.clip(rng.normal(cfg.d_mean, cfg.d_std, n_part), 1.5, 4.0)
        z = rng.uniform(-1, 1, n_part)
        Ia = cfg.I_max * np.exp(-0.5 * ((z - 0.5 * cfg.dz_out) / cfg.sheet_sigma) ** 2)
        Ib = cfg.I_max * np.exp(-0.5 * ((z + 0.5 * cfg.dz_out) / cfg.sheet_sigma) ** 2)
        xa, ya, xb, yb = flow.pair_positions(xm, ym, k)
        parts = {"A": render_particles(xa, ya, d, Ia, H, W), "B": render_particles(xb, yb, d, Ib, H, W)}
        # hotspot positions for this pair: shared by A and B
        if variant == "fixed":
            centres = nominal
        elif variant == "drift":
            centres = [(x + rng.normal(0, 6), y + rng.normal(0, 6), s) for x, y, s in nominal]
        elif variant == "random":
            centres = [(rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H, s) for _, _, s in nominal]
        elif variant == "random_streaks":
            centres = nominal
            rs = [art._line_geometry(rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * H,
                                     rng.uniform(0, 180), rng.uniform(0.5, 1.0) * W, rng.uniform(1.0, 1.5))
                  for _ in range(2)]
        else:
            raise ValueError(variant)
        for f in "AB":
            a_img = art.glow.copy()
            for along, perp, env, width, amp in art.streak_geom:
                a_img += art._streak(perp, env, width, amp * rng.lognormal(0, cfg.streak_cv),
                                     rng.normal(0, cfg.streak_jitter))
            for x0, y0, s in centres:              # every frame, independent amplitude per pulse
                a_img += rng.uniform(*cfg.hotspot_amp) * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s ** 2))
            a_img += rng.uniform(*cfg.global_amp) * art.global_pattern
            if variant == "random_streaks":        # same place in A and B, own amplitude per pulse
                for along, perp, env, width in rs:
                    a_img += art._streak(perp, env, width, rng.uniform(500, 800), 0.0)
            out[f"clean_{f}"][k] = camera(parts[f], cfg, rng)
            out[f"dirty_{f}"][k] = camera(parts[f] + a_img, cfg, rng)
    return cfg, flow, out


def highpass(img, sigma=5.0):
    return np.maximum(img - gaussian_filter(img, sigma), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["fixed", "drift", "random", "random_streaks"])
    ap.add_argument("--pairs", type=int, default=400, help="ensemble size used for POD")
    ap.add_argument("--eval", type=int, default=150, help="pairs evaluated with PIV")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    t0 = time.time()
    cfg, flow, D = make_dataset(a.variant, a.pairs, seed={"fixed": 11, "drift": 12, "random": 13, "random_streaks": 14}[a.variant])
    pod = {f: PODFilter(max_store=60).fit(D[f"dirty_{f}"]) for f in "AB"}
    print(f"[{a.variant}] data {time.time() - t0:.0f} s; POD auto rank A={pod['A'].rank_} B={pod['B'].rank_}", flush=True)

    def frame(c, f, k):
        if c == "clean":
            return D[f"clean_{f}"][k].astype(np.float32)
        if c == "dirty":
            return D[f"dirty_{f}"][k].astype(np.float32)
        if c == "pod":
            return pod[f].transform(D[f"dirty_{f}"], idx=[k])[0]
        if c == "highpass":
            return highpass(D[f"dirty_{f}"][k].astype(np.float32))
        if c == "pod+highpass":
            return highpass(pod[f].transform(D[f"dirty_{f}"], idx=[k])[0])

    s = PIVSettings()
    ys, xs = node_grid(cfg.H, cfg.W, *s.passes[-1])
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    E = {c: [] for c in CONDS}
    Dc = {c: [] for c in CONDS}
    for k in range(a.eval):
        tx, ty = flow.truth_displacement(X, Y, k)
        for c in CONDS:
            _, _, dx, dy, _ = piv_pair(frame(c, "A", k), frame(c, "B", k), s)
            E[c].append(np.hypot(dx - tx, dy - ty))
            Dc[c].append((dx, dy))
    lines = [f"### Variant: {a.variant} (every frame corrupted; POD ensemble {a.pairs} pairs, "
             f"auto rank A={pod['A'].rank_}, B={pod['B'].rank_}; PIV on {a.eval} pairs)\n",
             "| Condition | RMS error (px) | Median (px) | Vectors > 1 px | Artifact induced RMS (px) |",
             "|---|---|---|---|---|"]
    for c in CONDS:
        e = np.array(E[c])
        ai = np.sqrt(np.mean([np.hypot(dx - cx, dy - cy) ** 2 for (dx, dy), (cx, cy) in zip(Dc[c], Dc["clean"])]))
        lines.append(f"| {c} | {np.sqrt(np.mean(e ** 2)):.3f} | {np.median(e):.3f} | {100 * np.mean(e > 1):.2f}% | {ai:.3f} |")
    txt = "\n".join(lines) + "\n"
    print(txt, f"total {time.time() - t0:.0f} s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"every_frame_{a.variant}.md"), "w") as fh:
        fh.write(txt)
    np.save(os.path.join(a.out, f"every_frame_{a.variant}_example.npy"),
            np.stack([frame(c, "A", 0) for c in ("dirty", "pod", "pod+highpass")]).astype(np.float32))


if __name__ == "__main__":
    main()
