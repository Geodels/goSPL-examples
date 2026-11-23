"""
Plots for the gravity_currents example: map views, strike sections (along y),
stratigraphic dip sections (along x) and the gravity budget through time.

Every function compares a list of runs, by default the run with gravity
currents (``sim_gravity``) and the reference without them (``sim_nogravity``);
``FAILURE`` adds the run with slope failure (``sim_failure``). The outputs are
read directly
from goSPL's HDF5 files (one file per MPI rank) and interpolated on regular
lines or grids for plotting.
"""
import glob
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, TwoSlopeNorm
from scipy.interpolate import griddata

RUNS = [("sim_nogravity", "without gravity currents"),
        ("sim_gravity", "with gravity currents")]
FAILURE = [("sim_gravity", "gravity currents"),
           ("sim_failure", "gravity currents + slope failure")]
COLORS = ["#2a6fb0", "#d0731c", "#3a9a5b"]


def _nranks(sim):
    return len(glob.glob(os.path.join(sim, "h5", "topology.p*.h5")))


def load(sim, step, keys, kind="gospl"):
    """Coordinates (x, y) and the requested fields of output ``step``,
    gathered from every rank's file. ``kind`` is 'gospl' (surface fields) or
    'stratal' (stratigraphic layers)."""
    xy, out = [], {k: [] for k in keys}
    for p in range(_nranks(sim)):
        with h5py.File(os.path.join(sim, "h5", f"topology.p{p}.h5"), "r") as t:
            xy.append(t["coords"][:, :2])
        with h5py.File(os.path.join(sim, "h5", f"{kind}.{step}.p{p}.h5"), "r") as f:
            for k in keys:
                out[k].append(f[k][:] if kind == "stratal" else f[k][:].ravel())
    stack = np.vstack if kind == "stratal" else np.concatenate
    return np.vstack(xy), {k: stack(v) for k, v in out.items()}


def last_step(sim):
    """Index of the last surface output written by a run."""
    files = glob.glob(os.path.join(sim, "h5", "gospl.*.p0.h5"))
    return max(int(os.path.basename(f).split(".")[1]) for f in files)


def maps(step, xlim=(120e3, 200e3), dx=250.0, sealevel=None, runs=RUNS):
    """Shaded bathymetry, net gravity deposit and surface fine fraction of the
    margin, one row per run."""
    X, Y = np.meshgrid(np.arange(xlim[0], xlim[1] + 1, dx), np.arange(0, 100e3 + 1, dx))
    ext = [xlim[0] / 1e3, xlim[1] / 1e3, 0, 100]
    ls = LightSource(azdeg=315, altdeg=30)
    fig, ax = plt.subplots(len(runs), 3, figsize=(13, 4.5 * len(runs)), squeeze=False,
                           constrained_layout=True)
    for r, (sim, lab) in enumerate(runs):
        keys = ["elev", "surfFineFrac"] + (["gravDep"] if "nogravity" not in sim else [])
        xy, f = load(sim, step, keys)
        z = griddata(xy, f["elev"], (X, Y))
        shade = ls.hillshade(z, vert_exag=60, dx=dx, dy=dx)
        for a in ax[r]:
            a.imshow(shade, cmap="gray", origin="lower", extent=ext)
        ax[r, 0].contour(X / 1e3, Y / 1e3, z, levels=np.arange(-500, -50, 50),
                         colors="k", linewidths=0.3, alpha=0.6)
        ax[r, 0].set_ylabel(f"{lab}\ny (km)")
        if "gravDep" in f:
            gd = griddata(xy, f["gravDep"], (X, Y))
            im1 = ax[r, 1].imshow(np.where(np.abs(gd) > 0.05, gd, np.nan), cmap="BrBG",
                                  norm=TwoSlopeNorm(0, -15, 15), origin="lower",
                                  extent=ext, alpha=0.85)
        else:
            ax[r, 1].text(0.5, 0.5, "no gravity currents", ha="center",
                          transform=ax[r, 1].transAxes)
        ff = griddata(xy, f["surfFineFrac"], (X, Y))
        sl = sealevel if sealevel is not None else -20.0
        im2 = ax[r, 2].imshow(np.where(z < sl, ff, np.nan), cmap="cividis_r", vmin=0,
                              vmax=1, origin="lower", extent=ext, alpha=0.85)
    ax[0, 0].set_title("bathymetry (shaded, 50 m contours)")
    ax[0, 1].set_title("net gravity deposit (m)")
    ax[0, 2].set_title("surface fine fraction (0 = coarse)")
    fig.colorbar(im1, ax=ax[:, 1], shrink=0.5, location="bottom", label="gravDep (m)")
    fig.colorbar(im2, ax=ax[:, 2], shrink=0.5, location="bottom", label="fine fraction")
    for a in ax[-1]:
        a.set_xlabel("x (km)")
    return fig


def strike_sections(step, xs=(130e3, 140e3, 150e3, 160e3, 170e3, 180e3, 190e3),
                    runs=RUNS):
    """Profiles along y at fixed x: elevation (initial dashed), and the
    cumulative deposit (solid) with the flexural displacement (dotted)."""
    yy = np.linspace(0, 100e3, 501)
    colors = COLORS
    fig, ax = plt.subplots(len(xs), 2, figsize=(12, 2.1 * len(xs)), sharex=True,
                           constrained_layout=True)
    xy0, f0 = load(runs[0][0], 0, ["elev"])
    data = [load(sim, step, ["elev", "erodep", "flexIso"]) for sim, _ in runs]
    for r, xv in enumerate(xs):
        pts = np.column_stack([np.full_like(yy, xv), yy])
        ax[r, 0].plot(yy / 1e3, griddata(xy0, f0["elev"], pts), color="0.55", ls="--",
                      lw=1.2, label="initial")
        for (xy, f), (_, lab), c in zip(data, runs, colors):
            ax[r, 0].plot(yy / 1e3, griddata(xy, f["elev"], pts), color=c, lw=1.6, label=lab)
            ax[r, 1].plot(yy / 1e3, griddata(xy, f["erodep"], pts), color=c, lw=1.6)
            ax[r, 1].plot(yy / 1e3, griddata(xy, f["flexIso"], pts), color=c, lw=1.0, ls=":")
        ax[r, 1].axhline(0, color="0.7", lw=0.8)
        ax[r, 0].set_ylabel(f"x = {xv/1e3:.0f} km\nelevation (m)")
        ax[r, 1].set_ylabel("thickness (m)")
        for a in ax[r]:
            a.grid(alpha=0.25, lw=0.5)
    ax[0, 0].set_title(f"elevation along y, step {step}")
    ax[0, 1].set_title("cumulative deposit (solid), flexure (dotted)")
    ax[0, 0].legend(loc="lower right", fontsize=8, frameon=False)
    for a in ax[-1]:
        a.set_xlabel("y (km)")
    return fig


def dip_sections(step, ys=(30e3, 55e3, 80e3), color_by="lithology",
                 xlim=(120e3, 200e3), zlim=(-560, -40), mesh="data/gospl_mesh.npz",
                 runs=RUNS):
    """Stratigraphic sections along x at fixed y for each run, drawn with
    goSPL's ``gospl.analyse.stratasection.cross_section``: each layer is a
    filled band between its present-day interfaces, coloured by ``color_by``
    (``lithology`` = fine fraction, ``facies`` = depositional facies from the
    water depth at deposition, or ``age``, ``porosity``...), with the surface,
    the basement and the sea level."""
    from gospl.analyse.stratasection import cross_section, load_strata

    fig, ax = plt.subplots(len(ys), len(runs), figsize=(7 * len(runs), 3.4 * len(ys)),
                           squeeze=False, constrained_layout=True)
    for c, (sim, lab) in enumerate(runs):
        data = load_strata(os.path.join(sim, "h5"), mesh, step=step)
        for r, yv in enumerate(ys):
            a = ax[r, c]
            cross_section(data, kind="x", at=yv, color_by=color_by, ax=a,
                          layer_lines=5, xlim=xlim, ylim=zlim,
                          legend_loc="lower left")
            a.set_title(f"{lab}, y = {yv/1e3:.0f} km", fontsize=10)
            if c > 0:
                a.set_ylabel("")
            if r < len(ys) - 1:
                a.set_xlabel("")
    return fig


def budget(summary="summary_gravity.jsonl"):
    """Gravity budget per step from the ``--summary`` record: supplied and
    eroded volumes (of which failed, with slope failure) against deposited and
    exported ones, and the closure."""
    rows = [json.loads(l) for l in open(summary)]
    g = [(r["t"], r["gravity"]) for r in rows if "gravity" in r]
    t = np.array([v[0] for v in g]) / 1e3
    sup = np.array([v[1]["supplied"] for v in g]) * 1e-9
    ero = np.array([v[1].get("eroded", 0.0) for v in g]) * 1e-9
    dep = np.array([v[1]["deposited"] for v in g]) * 1e-9
    exp = np.array([v[1]["exported"] for v in g]) * 1e-9
    fail = np.array([v[1].get("failed", 0.0) for v in g]) * 1e-9
    clo = np.abs(sup + ero - dep - exp) / np.maximum(sup + ero, 1e-30)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
    ax[0].plot(t, sup, color="#2a6fb0", lw=1.8, label="supplied")
    ax[0].plot(t, dep, color="#3a9a5b", lw=1.8, label="deposited")
    ax[0].plot(t, exp, color="#d0731c", lw=1.8, label="exported")
    if fail.any():
        ax[0].plot(t, fail, color="0.35", lw=1.4, label="failed (slope failure)")
    if (ero - fail > 0).any():
        ax[0].plot(t, ero - fail, color="0.6", lw=1.2, label="eroded by the currents")
    ax[0].set_xlabel("time (kyr)")
    ax[0].set_ylabel("volume per step (km$^3$)")
    ax[0].legend(frameon=False)
    ax[1].semilogy(t, np.maximum(clo, 1e-16), color="0.3", lw=1.5)
    ax[1].set_xlabel("time (kyr)")
    ax[1].set_ylabel("budget closure")
    ax[1].set_title("|supplied + eroded - deposited - exported| / (supplied + eroded)",
                    fontsize=9)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    return fig
