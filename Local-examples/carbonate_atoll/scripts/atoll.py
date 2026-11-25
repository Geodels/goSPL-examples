"""
Plots of the carbonate_atoll example: maps of the island through time (the
fringing reef, barrier reef, atoll sequence), maps of the wave energy and of
the reef thickness, and stratigraphic sections west-east (along the trade
winds) and south-north, coloured by lithology (volcanic sand yellow and mud
grey, carbonate sand light blue, carbonate mud dark blue, reef pink).

Every function reads goSPL's HDF5 outputs directly (one file per MPI rank).
"""
import glob
import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

C = 7000.0                     # centre of the island (m)


def _nranks(sim):
    return len(glob.glob(os.path.join(sim, "h5", "topology.p*.h5")))


def steps(sim, base="gospl"):
    """Indices of the outputs of a run (``gospl`` surface or ``stratal``)."""
    return sorted(int(os.path.basename(f).split(".")[1])
                  for f in glob.glob(os.path.join(sim, "h5", "%s.*.p0.h5" % base)))


def load(sim, step, keys=("elev",), base="gospl"):
    """Coordinates (m) and fields of output ``step`` gathered from every rank."""
    xy, out = [], {k: [] for k in keys}
    for p in range(_nranks(sim)):
        with h5py.File(os.path.join(sim, "h5", "topology.p%d.h5" % p), "r") as t:
            xy.append(t["coords"][:, :2])
        with h5py.File(os.path.join(sim, "h5", "%s.%d.p%d.h5" % (base, step, p)), "r") as f:
            for k in keys:
                a = f[k][:]
                out[k].append(a.ravel() if base == "gospl" else a)
    return np.vstack(xy), {k: np.concatenate(v) for k, v in out.items()}


def reef_thickness(sim):
    """Thickness of reef framework and of all carbonate (m) per node, end of
    the run."""
    last = steps(sim, "stratal")[-1]
    xy, s = load(sim, last, ("stratH_reef", "stratH_carb_sand", "stratH_carb_mud"),
                 base="stratal")
    reef = s["stratH_reef"][:, 1:].sum(axis=1)
    carb = reef + s["stratH_carb_sand"][:, 1:].sum(axis=1) \
        + s["stratH_carb_mud"][:, 1:].sum(axis=1)
    return xy, reef, carb


def _tri(xy):
    return mtri.Triangulation((xy[:, 0] - C) / 1e3, (xy[:, 1] - C) / 1e3)


def _map(ax, xy, val, title, cmap, label=None, cbar=True, **kw):
    im = ax.tripcolor(_tri(xy), val, cmap=cmap, shading="gouraud", **kw)
    if cbar:
        plt.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    return im


def _coast(ax, xy, elev, level=0.0, **kw):
    kw.setdefault("colors", "k")
    kw.setdefault("linewidths", 0.7)
    ax.tricontour(_tri(xy), elev, levels=[level], **kw)


def evolution(sim, n=4, title=None):
    """Elevation every few outputs (the fringing reef, barrier reef and atoll
    stages), with the shoreline (black) and the -25 m contour (grey, the
    limit of the reef builders)."""
    ss = steps(sim)
    pick = [ss[int(round(i * (len(ss) - 1) / (n - 1)))] for i in range(n)]
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 4), constrained_layout=True)
    for a, s in zip(ax, pick):
        xy, g = load(sim, s, ("elev",))
        im = _map(a, xy, g["elev"], "%d kyr" % (s * 20), "terrain", cbar=False,
                  vmin=-150, vmax=150)
        _coast(a, xy, g["elev"])
        _coast(a, xy, g["elev"], -25.0, colors="0.5")
    fig.colorbar(im, ax=ax, label="elevation (m)", shrink=0.8)
    fig.suptitle(title or sim)
    return fig


def compare(sims, labels, field="reef"):
    """Final reef thickness (``reef``), wave energy (``waveE``) or elevation
    (``elev``) of several runs side by side."""
    fig, ax = plt.subplots(1, len(sims), figsize=(4.2 * len(sims), 4.1),
                           constrained_layout=True)
    for a, sim, lab in zip(np.atleast_1d(ax), sims, labels):
        last = steps(sim)[-1]
        with h5py.File(os.path.join(sim, "h5", "gospl.%d.p0.h5" % last), "r") as f:
            has_wave = "waveE" in f
        xy, g = load(sim, last, ("elev",) + (("waveE",) if field == "waveE" and has_wave
                                             else ()))
        if field == "reef":
            xr, reef, _ = reef_thickness(sim)
            im = _map(a, xr, reef, lab, "RdPu", cbar=False, vmin=0, vmax=300)
        elif field == "waveE":
            if "waveE" in g:
                im = _map(a, xy, g["waveE"], lab, "magma", cbar=False, vmin=0, vmax=1)
            else:
                a.set_title(lab + " (no waves)", fontsize=10)
                a.set_aspect("equal")
                a.set_xlim(-5, 5)
                a.set_ylim(-5, 5)
                continue
        else:
            im = _map(a, xy, g["elev"], lab, "terrain", cbar=False, vmin=-150, vmax=150)
        _coast(a, xy, g["elev"], -10.0, colors="c", linewidths=1.8)
        a.set_xlabel("x (km)")
    lab = {"reef": "reef thickness (m)", "waveE": "relative wave energy",
           "elev": "elevation (m)"}[field]
    fig.colorbar(im, ax=ax, label=lab, shrink=0.8)
    from matplotlib.lines import Line2D
    np.atleast_1d(ax)[0].legend(
        handles=[Line2D([], [], color="c", lw=1.8, label="10 m depth contour")],
        loc="lower left", fontsize=8, framealpha=0.8)
    return fig


def section(sim, kind="x", ax=None, title=None, color_by="lithologies"):
    """Stratigraphic section through the centre of the island: ``kind='x'``
    west-east (along the trade winds), ``'y'`` south-north."""
    from gospl.analyse.stratasection import cross_section, load_strata

    data = load_strata(os.path.join(sim, "h5"), "data/island.npz")
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    lim = (C - 6000.0, C + 6000.0)
    cross_section(data, kind=kind, at=C, color_by=color_by, ax=ax, layer_lines=3,
                  xlim=lim, ylim=(-700, 50), legend_loc="lower left")
    ax.set_title(title or "%s, %s section" % (sim, "west-east" if kind == "x"
                                              else "south-north"), fontsize=10)
    return ax


def karst(sim="sim_karst", ref="sim_sealevel"):
    """Karst run (goSPL feat/gravity-carbonates-karst): cumulative dissolution, the final
    elevation with and without the karst, and the dissolution through time
    (from the --summary record, km3 per step)."""
    import json

    last = steps(sim)[-1]
    xy, g = load(sim, last, ("elev", "karstDiss"))
    xr, gr = load(ref, steps(ref)[-1], ("elev",))
    fig = plt.figure(figsize=(15, 4.3), constrained_layout=True)
    a1, a2, a3, a4 = (fig.add_subplot(1, 4, i + 1) for i in range(4))
    _map(a1, xy, g["karstDiss"], "cumulative dissolution", "YlOrBr", "m", vmin=0)
    _coast(a1, xy, g["elev"], 0.0, colors="k", linewidths=0.8)
    _map(a2, xr, gr["elev"], "no karst", "terrain", cbar=False, vmin=-150, vmax=150)
    im = _map(a3, xy, g["elev"], "with karst", "terrain", cbar=False, vmin=-150, vmax=150)
    for a, x, z in ((a2, xr, gr["elev"]), (a3, xy, g["elev"])):
        _coast(a, x, z, -10.0, colors="c", linewidths=1.8)
    fig.colorbar(im, ax=[a2, a3], label="elevation (m)", shrink=0.8)
    for a in (a1, a2, a3):
        a.set_xlabel("x (km)")
    t, v = [], []
    for line in open("summary_%s.jsonl" % sim.replace("sim_", "")):
        r = json.loads(line)
        if "karst" in r:
            t.append(r["t"] * 1e-3)
            v.append(r["karst"]["dissolved"] * 1e-9)
    a4.plot(t, np.cumsum(v), color="tab:brown")
    a4.set_xlabel("model time (kyr)")
    a4.set_ylabel("cumulative dissolution (km$^3$)")
    a4.set_title("dissolved carbonate", fontsize=10)
    sl = np.loadtxt("data/sealevel.csv", delimiter=",")
    b = a4.twinx()
    b.plot(sl[:, 0] * 1e-3, sl[:, 1], color="0.6", lw=0.8)
    b.set_ylabel("sea level (m)", color="0.4")
    return fig
