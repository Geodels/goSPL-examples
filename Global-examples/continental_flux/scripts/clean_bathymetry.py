"""
Clean the bathymetry of a goSPL global mesh.

The 250 Ma paleo-bathymetry of this example (data/250.nc) goes down to
-25,368 m: closed holes far deeper than any ocean, isolated spikes and broad
regions below 11 km. goSPL's marine routing ends in them, and in 1 Myr some
fill with kilometres of sediment, so the marine record of the global examples
built on this mesh is dominated by artefact infill. This script, run on the
mesh before a simulation (it is part of build_inputs.ipynb):

1. removes the spikes: a node below `spike_below` (default -3000 m) that is
   more than `spike` (default 1500 m) deeper than the median of its two-ring
   neighbourhood is set to that median, repeated until none is left (real
   basins and slopes are wider than two rings and are not touched);
2. clips the seafloor at `floor` (default -6500 m, the deepest abyssal
   plains): trenches are not resolved at 20-100 km spacing anyway.

Land is never changed. Usage (from the example folder):

    python scripts/clean_bathymetry.py vars_25_80/mesh.npz

writes the cleaned mesh in place and keeps the original as
vars_25_80/mesh_raw.npz (re-running cleans the raw file again).
"""
import argparse
import os
import shutil

import numpy as np
import scipy.sparse as sp


def two_ring(cells, n):
    """Node -> two-ring neighbours (CSR adjacency, self excluded)."""
    e = np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    A2 = ((A @ A + A) > 0).tocsr()
    A2.setdiag(0)
    A2.eliminate_zeros()
    return A2


def ring_median(z, A2):
    med = np.empty_like(z)
    ip, ix = A2.indptr, A2.indices
    for i in range(z.size):
        med[i] = np.median(z[ix[ip[i]:ip[i + 1]]])
    return med


def clean(z, cells, floor=-6500.0, spike=1500.0, spike_below=-3000.0, maxit=20):
    z = z.astype(np.float64).copy()
    A2 = two_ring(cells, z.size)
    total = 0
    for it in range(maxit):
        med = ring_median(z, A2)
        bad = (z < spike_below) & (z < med - spike)
        if not bad.any():
            break
        z[bad] = med[bad]
        total += int(bad.sum())
        print("  pass %d: %d spikes set to their neighbourhood median" % (it + 1, bad.sum()))
    clipped = int((z < floor).sum())
    z = np.maximum(z, floor)
    print("  %d spike corrections, %d nodes clipped at %.0f m" % (total, clipped, floor))
    return z


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesh", help="mesh npz (keys v, c, z)")
    ap.add_argument("--floor", type=float, default=-6500.0)
    ap.add_argument("--spike", type=float, default=1500.0)
    ap.add_argument("--spike-below", type=float, default=-3000.0)
    a = ap.parse_args()
    raw = a.mesh.replace(".npz", "_raw.npz")
    if not os.path.exists(raw):
        shutil.copy(a.mesh, raw)
    d = dict(np.load(raw))
    print("%s: z %.0f to %.0f m" % (raw, d["z"].min(), d["z"].max()))
    d["z"] = clean(d["z"], d["c"], a.floor, a.spike, a.spike_below)
    np.savez_compressed(a.mesh, **d)
    print("wrote %s: z %.0f to %.0f m" % (a.mesh, d["z"].min(), d["z"].max()))


if __name__ == "__main__":
    main()
