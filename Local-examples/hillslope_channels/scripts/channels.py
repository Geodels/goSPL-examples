"""Read the hillslope_channels outputs on the input mesh and measure the
channel steepness, as in the goSPL GMD paper (Sect. 3.4)."""
import glob

import h5py
import numpy as np
from scipy.spatial import cKDTree

L = 20000.0          # block size (m)
A_CHANNEL = 1.0e6    # cells draining more than 1 km2 are channels (m2)


def load_mesh(dx):
    """Input mesh of the run at spacing dx: coordinates (n, 2) and cells."""
    d = np.load(f"data/mesh{dx}.npz")
    return d["v"][:, :2], d["c"]


def read_step(run, step, xy):
    """Elevation and discharge (m3/yr) of output `step` of `run`, ordered
    like the input mesh coordinates `xy` (goSPL writes one file per rank)."""
    P = sorted(glob.glob(f"{run}/h5/topology.p*.h5"))
    E = [p.replace("topology", f"gospl.{step}") for p in P]
    pts = np.concatenate([h5py.File(p, "r")["coords"][:, :2] for p in P])
    f = {k: np.concatenate([h5py.File(e, "r")[k][:].ravel() for e in E]) for k in ("elev", "FA")}
    ids = cKDTree(pts).query(xy)[1]
    return f["elev"][ids], f["FA"][ids]


def n_steps(run):
    return len(glob.glob(f"{run}/h5/gospl.*.p0.h5"))


def steepest_slope(z, xy, cells):
    """Steepest downhill slope of every node along the mesh edges."""
    e = np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    s = (z[e[:, 0]] - z[e[:, 1]]) / np.hypot(*(xy[e[:, 0]] - xy[e[:, 1]]).T)
    S = np.zeros(len(z))
    np.maximum.at(S, e[:, 0], s)
    return S


def interior(xy):
    return (xy[:, 0] > 0) & (xy[:, 0] < L) & (xy[:, 1] > 0) & (xy[:, 1] < L)


def channel_steepness(run, dx, step=None):
    """Median k_s = S A^0.5 of the interior channel cells (A > 1 km2; with
    1 m/yr of rain the discharge in m3/yr is the drainage area in m2)."""
    xy, cells = load_mesh(dx)
    step = n_steps(run) - 1 if step is None else step
    z, A = read_step(run, step, xy)
    S = steepest_slope(z, xy, cells)
    ch = interior(xy) & (A > A_CHANNEL)
    return float(np.median(S[ch] * np.sqrt(A[ch])))


def mean_elevation(run, dx):
    """Mean interior elevation at every output."""
    xy, _ = load_mesh(dx)
    inner = interior(xy)
    return np.array([read_step(run, k, xy)[0][inner].mean() for k in range(n_steps(run))])
