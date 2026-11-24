"""
Plots of the carbonate_terraces example: the sea-level forcing, the profile of
the strip through time (the terraces are its steps), stratigraphic sections
coloured by lithology (reef in pink) and comparisons between runs.

Every function reads goSPL's HDF5 outputs directly (one file per MPI rank).
The strip is uniform along y, so profiles and sections are taken along x at
its centre line (y = 800 m).
"""
import glob
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

Y0 = 800.0                     # centre line of the strip (m)


def _nranks(sim):
    return len(glob.glob(os.path.join(sim, "h5", "topology.p*.h5")))


def steps(sim):
    """Indices of the surface outputs of a run."""
    return sorted(int(os.path.basename(f).split(".")[1])
                  for f in glob.glob(os.path.join(sim, "h5", "gospl.*.p0.h5")))


def load(sim, step, keys=("elev",)):
    """Coordinates and fields of output ``step`` gathered from every rank."""
    xy, out = [], {k: [] for k in keys}
    for p in range(_nranks(sim)):
        with h5py.File(os.path.join(sim, "h5", "topology.p%d.h5" % p), "r") as t:
            xy.append(t["coords"][:, :2])
        with h5py.File(os.path.join(sim, "h5", "gospl.%d.p%d.h5" % (step, p)), "r") as f:
            for k in keys:
                out[k].append(f[k][:].ravel())
    return np.vstack(xy), {k: np.concatenate(v) for k, v in out.items()}


def profile(sim, step, key="elev"):
    """Values along the centre line, sorted by x (km)."""
    xy, f = load(sim, step, (key,))
    row = np.abs(xy[:, 1] - Y0) < 1.0
    o = np.argsort(xy[row, 0])
    return xy[row, 0][o] / 1e3, f[key][row][o]


def sea_level(csv="data/sealevel.csv", ax=None):
    """The sea-level forcing (Spratt & Lisiecki 2016) in ka, with the main
    interglacial highstands marked."""
    d = np.loadtxt(csv, delimiter=",")
    age = (d[-1, 0] - d[:, 0]) / 1e3
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.8), constrained_layout=True)
    ax.plot(age, d[:, 1], color="#2a6fb0", lw=1.4)
    for mis, a in (("MIS 11", 405), ("MIS 9", 330), ("MIS 7", 240), ("MIS 5e", 125),
                   ("MIS 1", 5)):
        ax.axvline(a, color="0.8", lw=0.8, zorder=0)
        ax.text(a, 12, mis, ha="center", fontsize=8, color="0.3")
    ax.set_xlim(400, 0)
    ax.set_ylim(-145, 20)
    ax.set_xlabel("age (ka)")
    ax.set_ylabel("sea level (m)")
    ax.grid(alpha=0.25, lw=0.5)
    return ax


def profile_evolution(sim, every=5, ax=None, title=None):
    """Elevation along the strip every ``every`` outputs, with
    the initial foundation dashed (viridis: darker = older): the steps of the final profile are the
    terraces, the bumps at their outer edge the reef crests."""
    ss = steps(sim)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x0, z0 = profile(sim, ss[0])
    ax.plot(x0, z0, color="0.6", ls="--", lw=1, label="foundation")
    cm = plt.get_cmap("viridis")
    for k, s in enumerate(ss[every::every] + [ss[-1]]):
        x, z = profile(sim, s)
        ax.plot(x, z, color=cm(k / max(1, len(ss[every::every]))), lw=1.0)
    x, z = profile(sim, ss[-1])
    ax.plot(x, z, color="k", lw=2, label="present")
    ax.axhline(0.0, color="steelblue", lw=0.8)
    ax.set_xlabel("distance from the land edge (km)")
    ax.set_ylabel("elevation (m)")
    ax.set_title(title or sim, fontsize=10)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)
    return ax


def section(sim, color_by="lithologies", res=100, ax=None, xlim=None, ylim=None,
            title=None):
    """Stratigraphic section along the centre line, from the last output,
    drawn with gospl.analyse.stratasection on the run's mesh (``res`` m):
    ``lithologies`` (reef pink, carbonate sand light blue, carbonate mud dark
    blue), ``facies``, ``age``, ``carbonate``..."""
    from gospl.analyse.stratasection import cross_section, load_strata

    mesh = "data/mesh%d.npz" % res
    data = load_strata(os.path.join(sim, "h5"), mesh)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    cross_section(data, kind="x", at=Y0, color_by=color_by, ax=ax, layer_lines=2,
                  xlim=xlim, ylim=ylim, legend_loc="lower left")
    ax.set_title(title or "%s (%s)" % (sim, color_by), fontsize=10)
    return ax


def compare_profiles(sims, labels, ax=None, title=None):
    """Present-day profiles of several runs on one plot."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for sim, lab in zip(sims, labels):
        x, z = profile(sim, steps(sim)[-1])
        ax.plot(x, z, lw=1.5, label=lab)
    ax.axhline(0.0, color="steelblue", lw=0.8)
    ax.set_xlabel("distance from the land edge (km)")
    ax.set_ylabel("elevation (m)")
    ax.set_title(title or "present-day profiles", fontsize=10)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    return ax


def budget(summary):
    """Carbonate produced (framework and loose) and eroded at the coast, per
    step, from the --summary record."""
    rows = [json.loads(l) for l in open(summary)]
    t = np.array([r["t"] for r in rows if "carbonate" in r])
    age = (t.max() - t) / 1e3
    fw = np.array([r["carbonate"]["framework"] for r in rows if "carbonate" in r])
    lo = np.array([r["carbonate"]["loose"] for r in rows if "carbonate" in r])
    er = np.array([r.get("coastal_erosion", {}).get("eroded", 0.0)
                   for r in rows if "carbonate" in r])
    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    ax.plot(age, fw, color="#f15bb5", lw=1.2, label="reef framework")
    ax.plot(age, lo, color="#457b9d", lw=1.2, label="loose carbonate")
    ax.plot(age, er, color="0.4", lw=1.0, label="coastal erosion")
    ax.set_xlim(age.max(), 0)
    ax.set_xlabel("age (ka)")
    ax.set_ylabel("m$^3$ per step")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    return fig
