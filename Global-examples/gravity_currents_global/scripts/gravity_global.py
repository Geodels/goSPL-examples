"""
Plots for the global gravity-current example: the gravity budget through time,
global maps, a regional zoom on one margin and stratigraphic sections across it.

Every map compares the run with gravity currents (``sim_gravity``) and the
reference without them (``sim_nogravity``). Outputs are read directly from
goSPL's HDF5 files (one per MPI rank); node positions are converted to
longitude and latitude.
"""
import glob
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

RUNS = [("sim_nogravity", "without gravity currents"),
        ("sim_gravity", "with gravity currents")]
MESH = "../erodep_1My/vars_40/mesh.npz"


def _nranks(sim):
    return len(glob.glob(os.path.join(sim, "h5", "topology.p*.h5")))


def last_step(sim):
    """Index of the last surface output written by a run."""
    files = glob.glob(os.path.join(sim, "h5", "gospl.*.p0.h5"))
    return max(int(os.path.basename(f).split(".")[1]) for f in files)


def load(sim, step, keys):
    """Longitude, latitude (degrees) and the requested surface fields."""
    xyz, out = [], {k: [] for k in keys}
    for p in range(_nranks(sim)):
        with h5py.File(os.path.join(sim, "h5", f"topology.p{p}.h5"), "r") as t:
            xyz.append(t["coords"][:])
        with h5py.File(os.path.join(sim, "h5", f"gospl.{step}.p{p}.h5"), "r") as f:
            for k in keys:
                out[k].append(f[k][:].ravel())
    xyz = np.vstack(xyz)
    r = np.linalg.norm(xyz, axis=1)
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    lat = np.degrees(np.arcsin(xyz[:, 2] / r))
    return lon, lat, {k: np.concatenate(v) for k, v in out.items()}


def budget(summary="summary_gravity.jsonl"):
    """Gravity budget per step from the ``--summary`` record."""
    rows = [json.loads(l) for l in open(summary)]
    g = [(r["t"], r["gravity"]) for r in rows if "gravity" in r]
    t = np.array([v[0] for v in g]) / 1e6
    sup = np.array([v[1]["supplied"] for v in g]) * 1e-9
    dep = np.array([v[1]["deposited"] for v in g]) * 1e-9
    exp = np.array([v[1]["exported"] for v in g]) * 1e-9
    clo = np.abs(sup - dep - exp) / np.maximum(sup, 1e-30)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
    ax[0].semilogy(t, sup, color="#2a6fb0", lw=1.8, label="supplied")
    ax[0].semilogy(t, np.maximum(dep, 1e-9), color="#3a9a5b", lw=1.4, ls="--", label="deposited")
    ax[0].set_xlabel("time (Ma)")
    ax[0].set_ylabel("volume per step (km$^3$)")
    ax[0].legend(frameon=False)
    ax[1].semilogy(t, np.maximum(clo, 1e-16), color="0.3", lw=1.5)
    ax[1].set_xlabel("time (Ma)")
    ax[1].set_ylabel("budget closure")
    ax[1].set_title("|supplied - deposited - exported| / supplied", fontsize=9)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    return fig


def _background(ax, lon, lat, elev, s=0.3):
    ax.scatter(lon, lat, c=np.where(elev > 0, 0.45, 0.88), s=s, cmap="gray",
               vmin=0, vmax=1, rasterized=True)


def maps(step):
    """Global maps: gravity deposit (log scale), surface fine fraction where the
    gravity currents deposited, and the difference of cumulative deposit
    between the two runs (where the currents moved sediment from and to)."""
    lon, lat, f = load("sim_gravity", step, ["elev", "gravDep", "surfFineFrac", "erodep"])
    _, _, f0 = load("sim_nogravity", step, ["erodep"])
    g = f["gravDep"]
    m = g > 0.5
    fig, ax = plt.subplots(3, 1, figsize=(11, 15), constrained_layout=True)
    for a in ax:
        _background(a, lon, lat, f["elev"])
        a.set_xlim(-180, 180)
        a.set_ylim(-90, 90)
        a.set_ylabel("latitude")
    sc = ax[0].scatter(lon[m], lat[m], c=np.log10(g[m]), s=1, cmap="viridis",
                       vmin=-0.3, vmax=2.5, rasterized=True)
    fig.colorbar(sc, ax=ax[0], label="log$_{10}$ gravity deposit (m)", shrink=0.8)
    ax[0].set_title("cumulative gravity deposit")
    sc = ax[1].scatter(lon[m], lat[m], c=f["surfFineFrac"][m], s=1, cmap="cividis_r",
                       vmin=0, vmax=1, rasterized=True)
    fig.colorbar(sc, ax=ax[1], label="fine fraction (0 = coarse)", shrink=0.8)
    ax[1].set_title("surface fine fraction where the gravity deposit exceeds 0.5 m")
    # only below the shelf: on land the two runs also differ through the global
    # flexure, which says nothing about where the currents moved sediment
    d = f["erodep"] - f0["erodep"]
    k = (np.abs(d) > 1.0) & (f["elev"] < -200.0)
    sc = ax[2].scatter(lon[k], lat[k], c=d[k], s=1, cmap="BrBG",
                       norm=TwoSlopeNorm(0, -100, 100), rasterized=True)
    fig.colorbar(sc, ax=ax[2], label="deposit difference (m)", shrink=0.8)
    ax[2].set_title("cumulative deposit below 200 m water depth, with minus "
                    "without gravity currents (brown: less, teal: more)")
    ax[2].set_xlabel("longitude")
    return fig


def zoom(step, lonlim, latlim, path=None):
    """Regional view of one margin: elevation, cumulative deposit in both runs
    and the gravity deposit, with an optional transect ``path``
    [(lon, lat), ...]."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    lon, lat, f = load("sim_gravity", step, ["elev", "erodep", "gravDep"])
    _, _, f0 = load("sim_nogravity", step, ["erodep"])
    w = (lon >= lonlim[0]) & (lon <= lonlim[1]) & (lat >= latlim[0]) & (lat <= latlim[1])
    kw = dict(s=14, marker="h", linewidths=0, rasterized=True)
    sc = ax[0, 0].scatter(lon[w], lat[w], c=f["elev"][w], cmap="terrain", vmin=-6000,
                          vmax=3000, **kw)
    fig.colorbar(sc, ax=ax[0, 0], label="elevation (m)")
    ax[0, 0].set_title("elevation, with gravity currents")
    sc = ax[0, 1].scatter(lon[w], lat[w], c=f["gravDep"][w], cmap="viridis", vmin=0,
                          vmax=300, **kw)
    fig.colorbar(sc, ax=ax[0, 1], label="gravity deposit (m)")
    ax[0, 1].set_title("cumulative gravity deposit")
    for a, e, title in ((ax[1, 0], f0["erodep"], "cumulative deposit, without"),
                        (ax[1, 1], f["erodep"], "cumulative deposit, with")):
        sc = a.scatter(lon[w], lat[w], c=e[w], cmap="BrBG",
                       norm=TwoSlopeNorm(0, -500, 500), **kw)
        fig.colorbar(sc, ax=a, label="erosion (-) / deposition (+) (m)")
        a.set_title(title)
    for a in ax.ravel():
        a.set_xlim(*lonlim)
        a.set_ylim(*latlim)
        a.set_aspect("equal")
        if path is not None:
            p = np.asarray(path)
            a.plot(p[:, 0], p[:, 1], "r-", lw=1.5)
    return fig


# Depositional facies for a global margin (water depth at deposition, m):
# the default classes of cross_section are for a regional shelf (edges at
# 0/20/50/75 m) and put everything deeper than the shelf in one class.
GLOBAL_FACIES = dict(
    facies_depths=(0.0, 200.0, 2000.0, 4000.0),
    facies_colors=("limegreen", "khaki", "sandybrown", "c", "teal"),
    facies_labels=("subaerial", "shelf (< 200 m)", "slope (200-2000 m)",
                   "rise (2000-4000 m)", "abyssal plain (> 4000 m)"),
)


def sections(step, path, color_by="lithology", ylim=None):
    """Stratigraphic sections along ``path`` [(lon, lat), ...] for both runs,
    drawn with ``gospl.analyse.stratasection.cross_section`` (filled layers
    coloured by ``color_by``: ``lithology`` = fine fraction on a common 0-1
    scale, ``facies`` with the global depth classes ``GLOBAL_FACIES``,
    ``age``, ...)."""
    from gospl.analyse.stratasection import cross_section, load_strata

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    kw = GLOBAL_FACIES if color_by == "facies" else {}
    for a, (sim, lab) in zip(ax, RUNS):
        data = load_strata(os.path.join(sim, "h5"), MESH, step=step)
        cross_section(data, path=path, color_by=color_by, ax=a, layer_lines=4,
                      ylim=ylim, legend_loc="lower left", **kw)
        if color_by in ("lithology", "fine", "coarse"):
            for coll in a.collections:          # same 0-1 scale in both panels
                if hasattr(coll, "get_array") and coll.get_array() is not None:
                    coll.set_clim(0.0, 1.0)
        a.set_title(lab, fontsize=10)
    return fig
