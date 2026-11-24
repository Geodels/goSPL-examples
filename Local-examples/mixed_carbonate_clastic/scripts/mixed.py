"""
Plots of the mixed_carbonate_clastic example: maps of the margin (elevation,
carbonate production, carbonate thickness and share of the deposits),
stratigraphic sections across a delta and across the coast between the deltas
(coloured by lithology: river sand yellow, river mud grey, carbonate sand light
blue, carbonate mud dark blue, reef pink) and the carbonate budget of each run.

Every function reads goSPL's HDF5 outputs directly (one file per MPI rank).
"""
import glob
import json
import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

CARB = ("stratH_reef", "stratH_carb_sand", "stratH_carb_mud")
DELTAS = (25.0, 75.0)          # y (km) of the two wet catchments
BETWEEN = 50.0                 # y (km) of the coast between them


def _nranks(sim):
    return len(glob.glob(os.path.join(sim, "h5", "topology.p*.h5")))


def steps(sim, base="gospl"):
    """Indices of the outputs of a run (``gospl`` surface or ``stratal``)."""
    return sorted(int(os.path.basename(f).split(".")[1])
                  for f in glob.glob(os.path.join(sim, "h5", "%s.*.p0.h5" % base)))


def load(sim, step, keys=("elev",), base="gospl"):
    """Coordinates (m) and fields of output ``step`` gathered from every rank.
    Stratigraphic fields (``base='stratal'``) are (nodes, layers)."""
    xy, out = [], {k: [] for k in keys}
    for p in range(_nranks(sim)):
        with h5py.File(os.path.join(sim, "h5", "topology.p%d.h5" % p), "r") as t:
            xy.append(t["coords"][:, :2])
        with h5py.File(os.path.join(sim, "h5", "%s.%d.p%d.h5" % (base, step, p)), "r") as f:
            for k in keys:
                a = f[k][:]
                out[k].append(a.ravel() if base == "gospl" else a)
    return np.vstack(xy), {k: np.concatenate(v) for k, v in out.items()}


def deposits(sim):
    """Total deposit and carbonate thickness (m, compacted) per node at the end
    of the run, from the last stratigraphic output (the initial basement layer
    excluded)."""
    last = steps(sim, "stratal")[-1]
    with h5py.File(os.path.join(sim, "h5", "stratal.%d.p0.h5" % last), "r") as f:
        keys = ["stratH"] + [k for k in CARB if k in f]
    xy, s = load(sim, last, keys, base="stratal")
    tot = s["stratH"][:, 1:].sum(axis=1)
    carb = sum(s[k][:, 1:].sum(axis=1) for k in CARB if k in s) \
        if any(k in s for k in CARB) else np.zeros_like(tot)
    return xy, tot, carb


def sea_level(csv="inputs/sealevel.csv", ax=None, end=5.0e5):
    """The sea-level forcing."""
    d = np.loadtxt(csv)
    d = d[d[:, 0] <= end]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 2.6), constrained_layout=True)
    ax.plot(d[:, 0] / 1e3, d[:, 1], color="#2a6fb0", lw=1.4)
    ax.set_xlabel("model time (kyr)")
    ax.set_ylabel("sea level (m)")
    ax.grid(alpha=0.25, lw=0.5)
    return ax


def _map(ax, xy, val, title, cmap, label, **kw):
    tri = mtri.Triangulation(xy[:, 0] / 1e3, xy[:, 1] / 1e3)
    im = ax.tripcolor(tri, val, cmap=cmap, shading="gouraud", **kw)
    plt.colorbar(im, ax=ax, label=label, shrink=0.85)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    return ax


def maps(sim, title=None):
    """Final elevation, carbonate thickness and carbonate share of the
    deposits, with the two delta and the inter-delta section lines."""
    xy, tot, carb = deposits(sim)
    _, g = load(sim, steps(sim)[-1], ("elev",))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    _map(ax[0], xy, g["elev"], "elevation (end)", "terrain", "m", vmin=-500, vmax=300)
    _map(ax[1], xy, carb, "carbonate thickness", "Blues", "m", vmin=0)
    share = np.where(tot > 1.0, carb / np.maximum(tot, 1.0e-9), np.nan)
    _map(ax[2], xy, share, "carbonate share of the deposits", "viridis", "-",
         vmin=0, vmax=1)
    for a in ax:
        for y in DELTAS:
            a.axhline(y, color="w", lw=0.8, ls="--")
        a.axhline(BETWEEN, color="w", lw=0.8)
    fig.suptitle(title or sim)
    return fig


def production(sim, steps_=None, title=None):
    """Carbonate production rate (mm/yr) at a few outputs, with the shoreline."""
    ss = steps(sim)
    steps_ = steps_ or [ss[len(ss) // 4], ss[len(ss) // 2], ss[3 * len(ss) // 4], ss[-1]]
    d = np.loadtxt("inputs/sealevel.csv")
    fig, ax = plt.subplots(1, len(steps_), figsize=(4.2 * len(steps_), 4.2),
                           constrained_layout=True, sharey=True)
    for a, s in zip(np.atleast_1d(ax), steps_):
        xy, g = load(sim, s, ("carbProd", "elev"))
        t = s * 2.0e4
        sl = np.interp(t, d[:, 0], d[:, 1])
        _map(a, xy, g["carbProd"] * 1e3, "%d kyr (sea level %.0f m)" % (t / 1e3, sl),
             "Blues", "mm/yr", vmin=0, vmax=0.8)
        tri = mtri.Triangulation(xy[:, 0] / 1e3, xy[:, 1] / 1e3)
        a.tricontour(tri, g["elev"] - sl, levels=[0.0], colors="k", linewidths=0.8)
    fig.suptitle(title or "%s: carbonate production (black: shoreline)" % sim)
    return fig


def section(sim, y, color_by="lithologies", ax=None, xlim=(80e3, 200e3),
            ylim=(-520, 60), title=None):
    """Stratigraphic section along x at ``y`` (km), from the last output."""
    from gospl.analyse.stratasection import cross_section, load_strata

    data = load_strata(os.path.join(sim, "h5"), "inputs/gospl_mesh.npz")
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    cross_section(data, kind="x", at=y * 1e3, color_by=color_by, ax=ax, layer_lines=4,
                  xlim=xlim, ylim=ylim, legend_loc="lower left")
    ax.set_title(title or "%s, y = %g km (%s)" % (sim, y, color_by), fontsize=10)
    return ax


def budget(names, labels):
    """Volume of carbonate produced (framework and loose) and of the deposits
    per run (km3), from the --summary records and the final stratigraphy."""
    rows = []
    for n, lab in zip(names, labels):
        r = [json.loads(l) for l in open("summary_%s.jsonl" % n)]
        fw = sum(q["carbonate"]["framework"] for q in r if "carbonate" in q)
        lo = sum(q["carbonate"]["loose"] for q in r if "carbonate" in q)
        rows.append((lab, fw / 1e9, lo / 1e9))
    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], color="#f15bb5", label="reef framework")
    ax.bar(x, [r[2] for r in rows], bottom=[r[1] for r in rows], color="#457b9d",
           label="loose carbonate (sand and mud)")
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("carbonate produced (km$^3$)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    return fig, rows


def forcing():
    """The two forcings: the sea-level curve and the rain map (two wet
    catchments)."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 3.6), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.3, 1]})
    sea_level(ax=ax[0])
    ax[0].set_title("sea level (synthetic, stratigraphic_record example)", fontsize=10)
    d = np.load("inputs/gospl_mesh.npz")
    _map(ax[1], d["v"][:, :2], d["rain"], "rain (wet catchments at y = 25 and 75 km)",
         "Blues", "m/yr", vmin=0)
    return fig
