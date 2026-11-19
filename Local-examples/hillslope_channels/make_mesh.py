"""
Build the meshes and input files of the hillslope_channels example.

A 20 x 20 km block uplifted at 1 mm/yr with fixed (base-level) edges, on
jittered triangulations at 200 m and 100 m spacing. Each mesh gets two
inputs with the same land diffusivity (0.2 m2/yr): one where diffusion acts
on every cell, one with `channelQ` holding the river cells out of it.

Usage:  python make_mesh.py
"""
import os

import numpy as np
from scipy.spatial import Delaunay

L, U = 20000.0, 1.0e-3          # block size (m), uplift rate (m/yr)
KD = 0.2                        # land hillslope diffusivity (m2/yr)
QC = "1.e6"                     # channelQ (m3/yr): 1 km2 at 1 m/yr of rain

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("data", exist_ok=True)

TEMPLATE = """name: hillslope channels {tag}

domain:
    npdata: ['data/mesh{dx}','v','c','z']   # mesh [path, vertices, cells, elevation]
    flowdir: 1   # single-flow routing
    bc: 'ffff'   # fixed (base-level) edges N,E,S,W
    seadepo: False
    nodep: True   # detachment-limited: eroded material leaves the block

time:
    start: 0.
    end: 3.e6   # steady state after ~0.5 Myr; run on to check it holds
    tout: 2.5e5
    dt: 5.e3

spl:
    K: 4.e-5   # erodibility: theoretical channel steepness U/K = 25 m
    d: 0.
    m: 0.5

diffusion:
    hillslopeKa: {kd}   # land hillslope diffusivity (m2/yr)
    hillslopeKm: 0.
{chan}
sea:
    position: -10.

climate:
  - start: 0.
    uniform: 1.   # m/yr

tectonics:
  - start: 0.
    end: 1.e8
    upsub: ['data/mesh{dx}','t']   # uplift rate (m/yr), 0 on the edges

output:
    dir: 'sim_{tag}'
"""

for dx in (200, 100, 50):
    rng = np.random.default_rng(1)
    n = int(L / dx) + 1
    xs = np.linspace(0.0, L, n)
    X, Y = np.meshgrid(xs, xs)
    x, y = X.ravel().copy(), Y.ravel().copy()
    inner = (x > 0) & (x < L) & (y > 0) & (y < L)
    # jitter interior nodes by up to a quarter cell: an irregular mesh, and
    # no exact discharge ties (cell areas all differ)
    x[inner] += rng.uniform(-0.25, 0.25, inner.sum()) * dx
    y[inner] += rng.uniform(-0.25, 0.25, inner.sum()) * dx
    z = np.where(inner, rng.uniform(0.0, 1.0, len(x)), 0.0)
    t = np.where(inner, U, 0.0)
    p = np.column_stack([x, y])
    np.savez(f"data/mesh{dx}.npz", v=np.column_stack([p, np.zeros(len(x))]),
             c=Delaunay(p).simplices.astype(np.int64), z=z, t=t)
    print(f"data/mesh{dx}.npz: {len(x)} nodes")

    for chan in (False, True):
        tag = f"{dx}{'_chan' if chan else ''}"
        line = (f"    channelQ: {QC}   # river cells (discharge >= channelQ, m3/yr) "
                "are held out of the diffusion\n") if chan else ""
        with open(f"input-{tag}.yml", "w") as f:
            f.write(TEMPLATE.format(tag=tag, dx=dx, kd=KD, chan=line))
        print(f"input-{tag}.yml")
