"""
benchmark.py
End to end test of the POD artifact filter against exact ground truth.

Conditions (same particles in every one):
    clean     images without artifacts                     (accuracy floor)
    dirty     images with streaks / flashes, no filtering
    minsub    dirty minus the ensemble minimum image        (standard PIV preprocessing)
    pod       dirty filtered with the POD filter (auto rank, per laser cavity)

For every pair, PIV is run on each condition and compared with the analytic
displacement at every vector node. Also reported: the artifact induced error
|d_condition - d_clean|, which removes the error sources shared by all
conditions (spatial resolution, particle noise) and isolates the artifacts.

    python benchmark.py --data data --out results           # full run
    python benchmark.py --data data --out results --pairs 200 --sweep-pairs 60   # quick
    python benchmark.py --data data --out results --max-seconds 600   # resumable chunks
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth_piv import SynthConfig, KarmanStreet, ArtifactModel, generate, load_config
from piv import PIVSettings, piv_pair, node_grid
from pod_filter import PODFilter

CONDS = ["clean", "dirty", "minsub", "pod"]
LABEL = {"clean": "Clean (no artifacts)", "dirty": "Corrupted, unfiltered",
         "minsub": "Corrupted, min subtraction", "pod": "Corrupted, POD filtered"}
COLOR = {"clean": "#2b8a3e", "dirty": "#c92a2a", "minsub": "#e67700", "pod": "#1864ab"}


# --------------------------------------------------------------------------- #
class Conditions:
    """Serves the frame pair k for any condition without storing filtered stacks."""

    def __init__(self, data_dir):
        L = lambda n: np.load(os.path.join(data_dir, n + ".npy"), mmap_mode="r")
        self.raw = {n: L(n) for n in ("clean_A", "clean_B", "dirty_A", "dirty_B")}
        self.min = {f: np.min(self.raw[f"dirty_{f}"], axis=0).astype(np.float32) for f in "AB"}
        t = time.time()
        self.pod = {f: PODFilter(n_modes="auto", max_store=80).fit(self.raw[f"dirty_{f}"]) for f in "AB"}
        self.pod_time = time.time() - t
        self.N = self.raw["clean_A"].shape[0]

    def frame(self, cond, f, k, r=None):
        if cond == "clean":
            return np.asarray(self.raw[f"clean_{f}"][k], np.float32)
        d = np.asarray(self.raw[f"dirty_{f}"][k], np.float32)
        if cond == "dirty":
            return d
        if cond == "minsub":
            return d - self.min[f]
        if cond == "pod":
            return self.pod[f].transform(self.raw[f"dirty_{f}"], idx=[k], r=r)[0]
        raise KeyError(cond)

    def pair(self, cond, k, r=None):
        return self.frame(cond, "A", k, r), self.frame(cond, "B", k, r)


def truth_on_grid(flow, ys, xs, k):
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    tx, ty = flow.truth_displacement(X, Y, k)
    return tx, ty


def summarize(ex, ey, outl, ref=None):
    e = np.hypot(ex, ey)
    d = {"rms_px": float(np.sqrt(np.mean(e ** 2))),
         "mean_abs_px": float(np.mean(e)),
         "median_px": float(np.median(e)),
         "p95_px": float(np.percentile(e, 95)),
         "p99_px": float(np.percentile(e, 99)),
         "frac_gt_0p5px": float(np.mean(e > 0.5)),
         "frac_gt_1px": float(np.mean(e > 1.0)),
         "flagged_frac": float(np.mean(outl)),
         "bias_x_px": float(np.mean(ex)), "bias_y_px": float(np.mean(ey))}
    if ref is not None:
        d["artifact_induced_rms_px"] = float(np.sqrt(np.mean(np.hypot(*ref) ** 2)))
    return d


def artifact_node_masks(cfg, ys, xs, win):
    """Nodes whose interrogation window overlaps a fixed streak or a hotspot."""
    art = ArtifactModel(cfg)
    streak = np.zeros((cfg.H, cfg.W))
    for along, perp, env, width, amp in art.streak_geom:
        streak = np.maximum(streak, env * np.exp(-0.5 * (perp / width) ** 2))
    hot = np.max(np.stack(art.hot), axis=0)
    h = win // 2
    ms = np.zeros((len(ys), len(xs)), bool)
    mh = np.zeros_like(ms)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            sl = (slice(int(y - h + 0.5), int(y + h + 0.5)), slice(int(x - h + 0.5), int(x + h + 0.5)))
            ms[j, i] = streak[sl].max() > 0.2
            mh[j, i] = hot[sl].max() > 0.3
    return ms, mh


# --------------------------------------------------------------------------- #
def run(data_dir, out, n_pairs=None, sweep_pairs=100, sweep_r=None, seed=0, max_seconds=None):
    os.makedirs(out, exist_ok=True)
    cfg = load_config(data_dir)
    flow = KarmanStreet(cfg)
    meta = np.load(os.path.join(data_dir, "meta.npz"))
    C = Conditions(data_dir)
    N = C.N if n_pairs is None else min(n_pairs, C.N)
    settings = PIVSettings()
    win_final = settings.passes[-1][0]
    ys, xs = node_grid(cfg.H, cfg.W, *settings.passes[-1])
    print(f"POD fit on {C.N} + {C.N} images: {C.pod_time:.1f} s, "
          f"modes removed A={C.pod['A'].rank_}, B={C.pod['B'].rank_}", flush=True)

    # ---------------- main run (cached per block, so an interrupted run resumes)
    cache = os.path.join(out, "cache")
    os.makedirs(cache, exist_ok=True)
    t_start = time.time()
    over_budget = lambda: max_seconds is not None and time.time() - t_start > max_seconds
    shape = (N, len(ys), len(xs))
    res = {c: {"dx": np.zeros(shape), "dy": np.zeros(shape), "out": np.zeros(shape, bool)} for c in CONDS}
    TX, TY = np.zeros(shape), np.zeros(shape)
    block = 50
    for k0 in range(0, N, block):
        k1 = min(N, k0 + block)
        fn = os.path.join(cache, f"main_{k0:05d}_{k1:05d}.npz")
        if not os.path.exists(fn):
            if over_budget():
                print("Time budget reached; rerun the same command to resume.", flush=True)
                return None
            t0 = time.time()
            blk = {}
            for k in range(k0, k1):
                tx, ty = truth_on_grid(flow, ys, xs, k)
                blk.setdefault("tx", []).append(tx); blk.setdefault("ty", []).append(ty)
                for c in CONDS:
                    a, b = C.pair(c, k)
                    _, _, dx, dy, info = piv_pair(a, b, settings)
                    for q, v in (("dx", dx), ("dy", dy), ("out", info["outliers"])):
                        blk.setdefault(f"{c}_{q}", []).append(v)
            np.savez(fn + ".tmp.npz", **{key: np.array(v) for key, v in blk.items()})
            os.replace(fn + ".tmp.npz", fn)                      # atomic: no half written cache
            print(f"  PIV pairs {k0}-{k1 - 1} done in {time.time() - t0:.0f} s", flush=True)
        z = np.load(fn)
        TX[k0:k1], TY[k0:k1] = z["tx"], z["ty"]
        for c in CONDS:
            for q in ("dx", "dy", "out"):
                res[c][q][k0:k1] = z[f"{c}_{q}"]
    np.savez_compressed(os.path.join(out, "piv_results.npz"), ys=ys, xs=xs, truth_dx=TX, truth_dy=TY,
                        **{f"{c}_{q}": res[c][q] for c in CONDS for q in ("dx", "dy", "out")})

    # ---------------- metrics
    E = {c: (res[c]["dx"] - TX, res[c]["dy"] - TY) for c in CONDS}
    A = {c: (res[c]["dx"] - res["clean"]["dx"], res[c]["dy"] - res["clean"]["dy"]) for c in CONDS}
    hotA, hotB = meta["A_hot_amp"][:N], meta["B_hot_amp"][:N]
    hot_pair = (hotA.max(1) > 0) | (hotB.max(1) > 0)
    glob_pair = (meta["A_global_amp"][:N] > 0) | (meta["B_global_amp"][:N] > 0)
    rand_pair = (meta["A_random_streak_amp"][:N] > 0) | (meta["B_random_streak_amp"][:N] > 0)
    ms, mh = artifact_node_masks(cfg, ys, xs, win_final)
    metrics = {"n_pairs": int(N), "n_vectors_per_condition": int(np.prod(shape)),
               "pod_modes_removed": {"A": int(C.pod["A"].rank_), "B": int(C.pod["B"].rank_)},
               "overall": {}, "subsets": {}}
    for c in CONDS:
        metrics["overall"][c] = summarize(*E[c], res[c]["out"], ref=A[c])
    subsets = {
        "windows_on_fixed_streaks": np.broadcast_to(ms, shape),
        "hotspot_windows_when_flashing": hot_pair[:, None, None] & mh[None],
        "pairs_with_global_flash": np.broadcast_to(glob_pair[:, None, None], shape),
        "pairs_with_random_streak": np.broadcast_to(rand_pair[:, None, None], shape),
        "artifact_free_windows_quiet_pairs": (~ms & ~mh)[None] & ~(hot_pair | glob_pair | rand_pair)[:, None, None],
    }
    for name, m in subsets.items():
        metrics["subsets"][name] = {"n_vectors": int(m.sum())}
        for c in CONDS:
            metrics["subsets"][name][c] = summarize(E[c][0][m], E[c][1][m], res[c]["out"][m],
                                                    ref=(A[c][0][m], A[c][1][m]))

    # ---------------- mode sweep on a subset of pairs
    rng = np.random.default_rng(seed)
    sp = np.sort(rng.choice(N, size=min(sweep_pairs, N), replace=False))
    r_auto = int(C.pod["A"].rank_)
    if sweep_r is None:
        sweep_r = sorted(set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 20, 30, 50, 80, r_auto]))
    sweep = {"r": [], "rms_px": [], "frac_gt_0p5px": [], "artifact_induced_rms_px": []}
    for r in sweep_r:
        fn = os.path.join(cache, f"sweep_n{len(sp)}_r{r:03d}.npz")
        if not os.path.exists(fn):
            if over_budget():
                print("Time budget reached during the mode sweep; rerun to resume.", flush=True)
                return None
            t0 = time.time()
            ex, ey, ax, ay = [], [], [], []
            for k in sp:
                a, b = C.pair("pod", k, r=r)
                _, _, dx, dy, _ = piv_pair(a, b, settings)
                ex.append(dx - TX[k]); ey.append(dy - TY[k])
                ax.append(dx - res["clean"]["dx"][k]); ay.append(dy - res["clean"]["dy"][k])
            np.savez(fn + ".tmp.npz", ex=np.array(ex), ey=np.array(ey), ax=np.array(ax), ay=np.array(ay))
            os.replace(fn + ".tmp.npz", fn)
            print(f"  sweep r={r:3d} done in {time.time() - t0:.0f} s", flush=True)
        z = np.load(fn)
        e = np.hypot(z["ex"], z["ey"])
        sweep["r"].append(int(r))
        sweep["rms_px"].append(float(np.sqrt(np.mean(e ** 2))))
        sweep["frac_gt_0p5px"].append(float(np.mean(e > 0.5)))
        sweep["artifact_induced_rms_px"].append(float(np.sqrt(np.mean(np.hypot(z["ax"], z["ay"]) ** 2))))
    ref = {c: float(np.sqrt(np.mean(np.hypot(E[c][0][sp], E[c][1][sp]) ** 2))) for c in CONDS}
    metrics["mode_sweep"] = {"pairs": sp.tolist(), **sweep, "reference_rms_same_pairs": ref}

    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    write_summary(metrics, out)
    make_figures(C, cfg, flow, meta, res, E, TX, TY, ys, xs, metrics, hot_pair, glob_pair, rand_pair, out)
    return metrics


# --------------------------------------------------------------------------- #
def write_summary(m, out):
    L = []
    L.append(f"# POD artifact filter benchmark\n\n{m['n_pairs']} image pairs ({2 * m['n_pairs']} images), "
             f"{m['n_vectors_per_condition']} vectors per condition. POD modes removed: "
             f"A = {m['pod_modes_removed']['A']}, B = {m['pod_modes_removed']['B']}.\n")
    hdr = "| Condition | RMS error (px) | Median (px) | 95th pct (px) | > 0.5 px | > 1 px | Flagged | Artifact induced RMS (px) |\n|---|---|---|---|---|---|---|---|"

    def rows(block):
        out_rows = []
        for c in CONDS:
            s = block[c]
            out_rows.append(f"| {LABEL[c]} | {s['rms_px']:.4f} | {s['median_px']:.4f} | {s['p95_px']:.4f} | "
                            f"{100 * s['frac_gt_0p5px']:.2f}% | {100 * s['frac_gt_1px']:.2f}% | "
                            f"{100 * s['flagged_frac']:.2f}% | {s['artifact_induced_rms_px']:.4f} |")
        return out_rows

    L.append("## All vectors\n")
    L.append(hdr)
    L += rows(m["overall"])
    for name, block in m["subsets"].items():
        L.append(f"\n## {name.replace('_', ' ')} ({block['n_vectors']} vectors)\n")
        L.append(hdr)
        L += rows(block)
    sw = m["mode_sweep"]
    L.append(f"\n## Mode sweep ({len(sw['pairs'])} pairs)\n\n| Modes removed | RMS error (px) | > 0.5 px | Artifact induced RMS (px) |\n|---|---|---|---|")
    for r, e, f, a in zip(sw["r"], sw["rms_px"], sw["frac_gt_0p5px"], sw["artifact_induced_rms_px"]):
        L.append(f"| {r} | {e:.4f} | {100 * f:.2f}% | {a:.4f} |")
    L.append("\nSame pairs, reference RMS: " + ", ".join(f"{c} {v:.4f}" for c, v in sw["reference_rms_same_pairs"].items()))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(txt)
    print(txt)


def _show(ax, img, title, lo=0.5, hi=99.7):
    v0, v1 = np.percentile(img, [lo, hi])
    ax.imshow(img, cmap="gray", vmin=v0, vmax=v1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def make_figures(C, cfg, flow, meta, res, E, TX, TY, ys, xs, m, hot_pair, glob_pair, rand_pair, out):
    N = TX.shape[0]
    # pick a strongly corrupted pair without a random streak for the examples
    score = meta["A_hot_amp"][:N].sum(1) + meta["B_hot_amp"][:N].sum(1)
    score = np.where(rand_pair, -1, score)
    k = int(np.argmax(score))

    # ---- fig 1: images
    fig, axs = plt.subplots(2, 4, figsize=(16, 8.8), layout="constrained")
    crop = (slice(40, 168), slice(20, 148))
    for j, c in enumerate(CONDS):
        img = C.frame(c, "A", k)
        _show(axs[0, j], img, LABEL[c] + f"\nframe A, pair {k}")
        _show(axs[1, j], img[crop], "zoom (rows 40 to 168, cols 20 to 148)")
    fig.suptitle("Particle images before and after artifact removal (grey levels auto scaled per panel)")
    fig.savefig(os.path.join(out, "fig1_images.png"), dpi=110)
    plt.close(fig)

    # ---- fig 2: POD spectrum and modes
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 6)
    ax = fig.add_subplot(gs[:, :2])
    for f, ls in (("A", "-"), ("B", "--")):
        P = C.pod[f]
        s = P.sigma_ / np.median(P.sigma_)
        ax.semilogy(np.arange(1, len(s) + 1), s, ls, color="#1864ab", lw=1.2, label=f"corrupted, cavity {f}")
    Pc = PODFilter(n_modes="auto", max_store=1).fit(C.raw["clean_A"])
    ax.semilogy(np.arange(1, len(Pc.sigma_) + 1), Pc.sigma_ / np.median(Pc.sigma_), color="#2b8a3e", lw=1.2,
                label="clean, cavity A")
    thr = C.pod["A"].threshold_ / np.median(C.pod["A"].sigma_)
    ax.axhline(thr, color="k", lw=0.8, ls=":", label=f"auto threshold ({C.pod['A'].rank_} modes)")
    ax.set_xscale("log")
    ax.set_xlabel("mode index")
    ax.set_ylabel("singular value / median")
    ax.set_title("POD spectrum: artifacts stand above the flat particle bulk")
    ax.legend(fontsize=9)
    for i in range(8):
        a2 = fig.add_subplot(gs[i // 4, 2 + i % 4])
        phi = C.pod["A"].spatial_mode(i)
        v = np.percentile(np.abs(phi), 99.5)
        a2.imshow(phi, cmap="RdBu_r", vmin=-v, vmax=v)
        a2.set_title(f"mode {i + 1}" + (" (removed)" if i < C.pod["A"].rank_ else ""), fontsize=9)
        a2.set_xticks([]); a2.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_pod_spectrum_modes.png"), dpi=110)
    plt.close(fig)

    # ---- fig 3: RMS error maps
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
    maps = {c: np.sqrt(np.mean(np.hypot(*E[c]) ** 2, axis=0)) for c in CONDS}
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=0.02, vmax=max(1.0, np.percentile(maps["dirty"], 99)))
    ext = [xs[0] - 8, xs[-1] + 8, ys[-1] + 8, ys[0] - 8]
    for ax, c in zip(axs, CONDS):
        im = ax.imshow(maps[c], cmap="magma", norm=norm, extent=ext)
        ax.set_title(f"{LABEL[c]}\nRMS {m['overall'][c]['rms_px']:.3f} px", fontsize=10)
        ax.set_xlabel("x (px)")
    axs[0].set_ylabel("y (px)")
    fig.colorbar(im, ax=axs, shrink=0.85, label="RMS error over all pairs (px, log scale)")
    fig.savefig(os.path.join(out, "fig3_error_maps.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 4: per pair error and CDF
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axs[0]
    for c in CONDS:
        pp = np.sqrt(np.mean(np.hypot(*E[c]) ** 2, axis=(1, 2)))
        ax.plot(np.arange(N), pp, lw=0.7, color=COLOR[c], label=LABEL[c])
    top = ax.get_ylim()[1]
    for mask, col, lab, yy in ((hot_pair, "#862e9c", "hotspot flash", 1.00), (glob_pair, "#495057", "global flash", 0.96),
                               (rand_pair, "#d9480f", "random streak", 0.92)):
        idx = np.nonzero(mask)[0]
        ax.plot(idx, np.full(len(idx), yy * top), "|", color=col, ms=6, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("pair index")
    ax.set_ylabel("RMS error of the pair (px)")
    ax.set_title("Per pair error; ticks mark which pairs contain which artifact")
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    ax = axs[1]
    for c in CONDS:
        e = np.sort(np.hypot(*E[c]).ravel())
        ax.semilogx(e, 1 - np.arange(len(e)) / len(e), color=COLOR[c], label=LABEL[c])
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 20)
    ax.set_xlabel("|error| (px)")
    ax.set_ylabel("fraction of vectors with larger error")
    ax.set_title("Error exceedance (tail) distribution")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_error_distribution.png"), dpi=110)
    plt.close(fig)

    # ---- fig 5: mode sweep
    sw = m["mode_sweep"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    r = np.array(sw["r"])
    ax.plot(np.maximum(r, 0.5), sw["rms_px"], "o-", color=COLOR["pod"], label="POD filtered, error vs truth")
    ax.plot(np.maximum(r, 0.5), sw["artifact_induced_rms_px"], "s--", color=COLOR["pod"], alpha=0.6,
            label="POD filtered, artifact induced part")
    for c in ("clean", "dirty", "minsub"):
        ax.axhline(sw["reference_rms_same_pairs"][c], color=COLOR[c], ls=":", lw=1.2, label=LABEL[c])
    ax.axvline(C.pod["A"].rank_, color="k", lw=0.8, ls="-.", label=f"auto rank = {C.pod['A'].rank_}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of POD modes removed (0 plotted at 0.5)")
    ax.set_ylabel("RMS displacement error (px)")
    ax.set_title(f"Sensitivity to the number of removed modes ({len(sw['pairs'])} pairs)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_mode_sweep.png"), dpi=110)
    plt.close(fig)

    # ---- fig 6: vector fields for the example pair
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.6))
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    fields = {"truth": (TX[k], TY[k])}
    fields.update({c: (res[c]["dx"][k], res[c]["dy"][k]) for c in ("clean", "dirty", "pod")})
    titles = {"truth": "Exact (analytic)", "clean": LABEL["clean"], "dirty": LABEL["dirty"], "pod": LABEL["pod"]}
    vmag = np.hypot(TX[k], TY[k])
    for j, (key, (dx, dy)) in enumerate(fields.items()):
        ax = axs[0, j]
        ax.imshow(np.hypot(dx, dy), cmap="viridis", vmin=vmag.min() - 0.5, vmax=vmag.max() + 0.5,
                  extent=[xs[0] - 8, xs[-1] + 8, ys[-1] + 8, ys[0] - 8])
        ax.quiver(X, Y, dx - cfg.U0, dy, color="w", scale=40, width=0.004)
        ax.set_title(titles[key] + "\n|d| colour, arrows = d minus free stream", fontsize=9)
        ax.set_xlim(0, cfg.W); ax.set_ylim(cfg.H, 0)
        ax = axs[1, j]
        if key == "truth":
            _show(ax, C.frame("dirty", "A", k), "Corrupted frame A (for reference)")
            continue
        err = np.hypot(dx - TX[k], dy - TY[k])
        im = ax.imshow(err, cmap="magma", vmin=0, vmax=1.0, extent=[xs[0] - 8, xs[-1] + 8, ys[-1] + 8, ys[0] - 8])
        ax.set_title(f"|error|, RMS {np.sqrt(np.mean(err ** 2)):.3f} px", fontsize=9)
    fig.colorbar(im, ax=axs[1, 1:], shrink=0.8, label="|error| (px), clipped at 1")
    fig.suptitle(f"Pair {k}: displacement fields and errors")
    fig.savefig(os.path.join(out, "fig6_vector_fields.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--pairs", type=int, default=None, help="evaluate only the first N pairs")
    ap.add_argument("--sweep-pairs", type=int, default=100)
    ap.add_argument("--generate", action="store_true", help="(re)generate the dataset first")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop after this long; rerunning resumes from the cache in <out>/cache")
    a = ap.parse_args()
    if a.generate or not os.path.exists(os.path.join(a.data, "meta.npz")):
        print("Generating dataset ...")
        generate(SynthConfig(), a.data)
    run(a.data, a.out, n_pairs=a.pairs, sweep_pairs=a.sweep_pairs, max_seconds=a.max_seconds)
