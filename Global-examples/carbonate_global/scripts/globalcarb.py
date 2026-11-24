"""
Plots of the carbonate_global example: world maps (longitude, latitude) of the
carbonate production, of the deposits by lithology (carbonate on shelves,
pelagic carbonate and clay in the basins), of the wave energy and of the
coastal erosion, and the budgets of every run (carbonate produced, pelagic
carbonate preserved and dissolved, supply limitation).

Every function reads goSPL's HDF5 outputs directly (one file per MPI rank).
"""
import glob
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

CARB = ("stratH_reef", "stratH_carb_sand", "stratH_carb_mud")


def _nranks(sim):
    return len(glob.glob(os.path.join(sim, "h5", "topology.p*.h5")))


def steps(sim, base="gospl"):
    """Indices of the outputs of a run (``gospl`` surface or ``stratal``)."""
    return sorted(int(os.path.basename(f).split(".")[1])
                  for f in glob.glob(os.path.join(sim, "h5", "%s.*.p0.h5" % base)))


def load(sim, step, keys=("elev",), base="gospl"):
    """Longitude, latitude (degrees) and fields of output ``step`` from every
    rank (stratigraphic fields are (nodes, layers))."""
    xyz, out = [], {k: [] for k in keys}
    for p in range(_nranks(sim)):
        with h5py.File(os.path.join(sim, "h5", "topology.p%d.h5" % p), "r") as t:
            xyz.append(t["coords"][:])
        with h5py.File(os.path.join(sim, "h5", "%s.%d.p%d.h5" % (base, step, p)), "r") as f:
            for k in keys:
                a = f[k][:]
                out[k].append(a.ravel() if base == "gospl" else a)
    xyz = np.vstack(xyz).astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2] / r, -1.0, 1.0)))
    return lon, lat, {k: np.concatenate(v) for k, v in out.items()}


_GRID = {}


def _raster(lon, lat, res=0.5):
    """Index of the nearest node (on the sphere) of every cell of a regular
    longitude-latitude grid, cached per mesh."""
    from scipy.spatial import cKDTree

    key = (lon.size, res)
    if key not in _GRID:
        def xyz(lo, la):
            lo, la = np.radians(lo), np.radians(la)
            return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                                    np.sin(la)])
        glo, gla = np.meshgrid(np.arange(-180.0 + res / 2, 180.0, res),
                               np.arange(-90.0 + res / 2, 90.0, res))
        _, idx = cKDTree(xyz(lon, lat)).query(xyz(glo.ravel(), gla.ravel()))
        _GRID[key] = idx.reshape(glo.shape)
    return _GRID[key]


def worldmap(lon, lat, val, ax=None, title=None, cmap="viridis", label=None,
             land=None, **kw):
    """A field on a longitude-latitude map (0.5 degree raster, nearest mesh
    node), land in grey."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    idx = _raster(lon, lat)
    img = np.asarray(val, dtype=float)[idx]
    if land is not None:
        img = np.ma.masked_where(np.asarray(land)[idx], img)
        ax.imshow(np.where(np.asarray(land)[idx], 1.0, np.nan), origin="lower",
                  extent=(-180, 180, -90, 90), cmap="Greys", vmin=0, vmax=4)
    im = ax.imshow(img, origin="lower", extent=(-180, 180, -90, 90), cmap=cmap, **kw)
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or "", fontsize=10)
    return ax


def deposits(sim):
    """Thickness (m, compacted) per node of each lithology over the run, from
    the last stratigraphic output (basement excluded): river sand (`coarse`),
    river mud and hemipelagic clay (`fine`), carbonate (reef, sand, mud)."""
    last = steps(sim, "stratal")[-1]
    with h5py.File(os.path.join(sim, "h5", "stratal.%d.p0.h5" % last), "r") as f:
        keys = ["stratH", "stratHf"] + [k for k in CARB if k in f]
    lon, lat, s = load(sim, last, keys, base="stratal")
    tot = s["stratH"][:, 1:].sum(axis=1)
    fine = s["stratHf"][:, 1:].sum(axis=1)
    carb = sum(s[k][:, 1:].sum(axis=1) for k in CARB if k in s)
    return lon, lat, {"total": tot, "fine": fine, "carbonate": carb,
                      "coarse": np.maximum(tot - fine - carb, 0.0)}


def overview(sim, title=None):
    """Final elevation, carbonate production rate, carbonate thickness and
    pelagic deposition rate."""
    last = steps(sim)[-1]
    lon, lat, g = load(sim, last, ("elev", "carbProd", "pelDep"))
    land = g["elev"] >= 0.0
    _, _, dep = deposits(sim)
    fig, ax = plt.subplots(2, 2, figsize=(17, 8.5), constrained_layout=True)
    worldmap(lon, lat, g["elev"], ax=ax[0, 0], title="elevation", cmap="terrain",
             label="m", vmin=-6000, vmax=3000)
    worldmap(lon, lat, g["carbProd"] * 1e3, ax=ax[0, 1], title="benthic carbonate production",
             cmap="Blues", label="mm/yr", land=land, vmin=0, vmax=0.2)
    worldmap(lon, lat, dep["carbonate"], ax=ax[1, 0], title="carbonate deposited (1 Myr)",
             cmap="viridis", label="m", land=land, vmin=0, vmax=50)
    worldmap(lon, lat, g["pelDep"] * 1e6, ax=ax[1, 1], title="pelagic deposition",
             cmap="magma", label="m/Myr", land=land, vmin=0, vmax=10)
    fig.suptitle(title or sim)
    return fig


def compare(sims, labels, key="pelDep", scale=1e6, label="m/Myr", vmax=10,
            cmap="magma"):
    """One output field of several runs side by side (default: the pelagic
    deposition rate in m/Myr)."""
    fig, ax = plt.subplots(1, len(sims), figsize=(8.5 * len(sims), 4.4),
                           constrained_layout=True)
    for a, sim, lab in zip(np.atleast_1d(ax), sims, labels):
        lon, lat, g = load(sim, steps(sim)[-1], ("elev", key))
        worldmap(lon, lat, g[key] * scale, ax=a, title=lab, cmap=cmap, label=label,
                 land=g["elev"] >= 0.0, vmin=0, vmax=vmax)
    return fig


def budget(names, labels):
    """Carbonate produced and pelagic carbonate preserved / dissolved over the
    run (km3) per run, from the --summary records; for a run with the
    carbonate supply, the supply scale of each step."""
    rows = []
    for n, lab in zip(names, labels):
        r = [json.loads(l) for l in open("summary_%s.jsonl" % n)]
        prod = sum(q["carbonate"]["produced"] for q in r if "carbonate" in q) / 1e9
        fw = sum(q["carbonate"]["framework"] for q in r if "carbonate" in q) / 1e9
        pc = sum(q["pelagic"]["carbonate"] for q in r if "pelagic" in q) / 1e9
        pd = sum(q["pelagic"]["dissolved"] for q in r if "pelagic" in q) / 1e9
        cl = sum(q["pelagic"]["clay"] for q in r if "pelagic" in q) / 1e9
        sc = [q["carbonate_supply"]["scale"] for q in r if "carbonate_supply" in q]
        rows.append({"run": lab, "benthic": prod, "framework": fw, "pelagic": pc,
                     "dissolved": pd, "clay": cl,
                     "scale": float(np.mean(sc)) if sc else None})
    fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    x = np.arange(len(rows))
    w = 0.2
    for k, (key, col, lab) in enumerate((("benthic", "#457b9d", "benthic carbonate"),
                                         ("pelagic", "#a8dadc", "pelagic carbonate preserved"),
                                         ("dissolved", "#e76f51", "pelagic carbonate dissolved"),
                                         ("clay", "#8d99ae", "hemipelagic clay"))):
        ax.bar(x + (k - 1.5) * w, [r[key] for r in rows], w, color=col, label=lab)
    ax.set_xticks(x, [r["run"] for r in rows])
    ax.set_ylabel("km$^3$ over the run")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    return fig, rows


def forcing(mesh="../continental_flux/vars_25_80/mesh.npz", maps="data/carb_maps.npz"):
    """The two forcing maps on the mesh: the sea-surface temperature (degC,
    `sst` block, temperature response of the factories and of the pelagic
    rain) and the pelagic carbonate rain before dissolution (m/Myr), land
    in grey."""
    d = np.load(mesh)
    m = np.load(maps)
    v = d["v"].astype(np.float64)
    r = np.linalg.norm(v, axis=1)
    lon = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(v[:, 2] / r, -1.0, 1.0)))
    land = d["z"] >= 0.0
    fig, ax = plt.subplots(1, 2, figsize=(17, 4.4), constrained_layout=True)
    worldmap(lon, lat, m["sst"], ax=ax[0], title="sea-surface temperature",
             cmap="RdYlBu_r", label="degC", land=land, vmin=0, vmax=30)
    worldmap(lon, lat, m["prod"], ax=ax[1], title="pelagic carbonate rain (before dissolution)",
             cmap="viridis", label="m/Myr", land=land, vmin=0, vmax=10)
    return fig


def supply(name="geochem"):
    """The carbonate supply of a run coupled to the geochemistry, per step
    (--summary record `carbonate_supply`): the factor scaling the burial
    (1 = not limited), the carbonate the reservoir can make (reservoir x
    yield) against the burial asked for, and the reservoir of dissolved
    carbonate."""
    r = [json.loads(l) for l in open("summary_%s.jsonl" % name)]
    s = [(q["t"], q["carbonate_supply"]) for q in r if "carbonate_supply" in q]
    t = np.array([a for a, _ in s]) / 1e6
    get = lambda k: np.array([b[k] for _, b in s])
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.4), constrained_layout=True)
    ax[0].plot(t, get("scale"), "o-", color="#457b9d")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("burial scale (1 = not limited)")
    ax[1].plot(t, get("available") / 1e9, "o-", label="supply x yield")
    ax[1].plot(t, get("demand") / 1e9, "s-", label="burial asked for")
    ax[1].set_ylabel("km$^3$ per step")
    ax[1].legend(frameon=False)
    ax[2].plot(t, get("reservoir"), "o-", color="#2a9d8f")
    ax[2].set_ylabel("dissolved carbonate reservoir (solute units)")
    for a in ax:
        a.set_xlabel("time (Ma)")
        a.grid(alpha=0.25, lw=0.5)
    return fig
