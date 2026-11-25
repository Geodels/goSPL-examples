"""
Build the uniform global goSPL meshes offered for download (5, 10, 20 km).

For each cell width it runs JIGSAW through MPAS-Tools (the same routine as
the continental_flux example, ``shared_scripts/umeshFcts.buildGlobalMeshSimple``)
and writes, in ``<out>/global_<w>km/``:

- ``mesh_<w>km.nc``: the MPAS/UGRID mesh (for the uxarray workflow of the
  continental_flux ``build_inputs`` notebook);
- ``gospl_mesh_<w>km.npz``: the goSPL mesh, ``v`` (vertex coordinates, m),
  ``c`` (triangles, 0-based) and ``z`` (elevation, zeros: interpolate yours
  with ``meshdata.py`` and save it in ``z``);
- ``info.json``: node and cell counts, edge lengths.

Usage (examples environment, several GB of memory for 5 km):

    python make_global_meshes.py --out ~/Workspace/goSPL-global-meshes 20 10 5
"""

import argparse
import json
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from shared_scripts import umeshFcts as ufcts  # noqa: E402


def build(width, out):
    folder = os.path.join(out, "global_%dkm" % width)
    os.makedirs(folder, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(folder)                       # JIGSAW writes its work files here
    try:
        ufcts.buildGlobalMeshSimple(width, ".")
    finally:
        os.chdir(cwd)
    for f in ("mesh.jig", "mesh.log", "mesh.msh", "mesh-MESH.msh", "mesh-HFUN.msh",
              "mesh_triangles.nc", "cellWidthVsLatLon.nc", "graph.info"):
        p = os.path.join(folder, f)
        if os.path.exists(p):
            os.remove(p)
    ds = xr.open_dataset(os.path.join(folder, "mesh_%dkm.nc" % width))
    v = np.column_stack([ds["xCell"].values, ds["yCell"].values, ds["zCell"].values])
    c = (ds["cellsOnVertex"].values - 1).astype(np.int32)
    c = c[(c >= 0).all(axis=1)]
    de = ds["dcEdge"].values / 1e3
    np.savez_compressed(os.path.join(folder, "gospl_mesh_%dkm.npz" % width),
                        v=v, c=c, z=np.zeros(len(v)))
    deg = np.bincount(c.ravel(), minlength=len(v))
    info = {"cell_width_km": width, "nodes": int(len(v)), "triangles": int(len(c)),
            "radius_m": float(np.linalg.norm(v[0])),
            "edge_km": {"min": round(float(de.min()), 2), "mean": round(float(de.mean()), 2),
                        "max": round(float(de.max()), 2)},
            "max_vertex_degree": int(deg.max())}
    json.dump(info, open(os.path.join(folder, "info.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)
    ds.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("widths", nargs="+", type=int)
    ap.add_argument("--out", default=os.path.expanduser("~/Workspace/goSPL-global-meshes"))
    a = ap.parse_args()
    for w in a.widths:
        build(w, a.out)
