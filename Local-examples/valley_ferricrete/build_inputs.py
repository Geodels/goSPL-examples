"""
Build a SIMPLE, controlled landscape for the two-stage tropical ferricrete
relief-inversion example. The textbook laterite-mesa story, in two stages:

  STAGE 1 (wet, tectonically quiet): a low-relief plain with broad, shallow river
  valleys. The water table sits near the surface across the wet valley floors, so an
  iron ferricrete forms there as a broad, resistant sheet (it does NOT form on the
  drier rises between the valleys). No crust is prescribed: it grows in place.

  STAGE 2 (drier, base-level fall): the interior is uplifted against a pinned outlet,
  so the rivers incise. Where incision breaches the ferricrete it guts the soft
  saprolite beneath and carves new valleys, while the ferricrete-capped former valley
  floors resist and are left standing as ferricrete-capped MESAS: the relief inverts.

The two uplift stages are stored as `t1` (quiet) and `t2` (base-level fall), both
tapered to zero at the southern outlet so it stays a fixed base level; the model only
drains south (walls on the other three edges, see bc in input.yml), which keeps the
landscape in relief instead of grading to a peneplain.

Writes inputs/gospl_mesh_valleys.npz (v, c, z, t1, t2).
Run once before the model:  python build_inputs.py
"""
import os
import numpy as np
from scipy.spatial import Delaunay

L, N = 50.0e3, 181
xs = np.linspace(0.0, L, N); ys = np.linspace(0.0, L, N)
X, Y = np.meshgrid(xs, ys)
x, y = X.ravel(), Y.ravel()
v = np.column_stack([x, y, np.zeros_like(x)])
cells = Delaunay(np.column_stack([x, y])).simplices.astype(np.int32)
rng = np.random.default_rng(11)

# A LOW-RELIEF plain with broad, shallow N-S valleys (troughs at x = 0, lam, 2*lam ...)
# separated by low rises; small amplitude so the wet valley floors are broad and the
# water table is near the surface there (a laterite plain, not deep canyons).
lam = 12.5e3                                                 # L / lam = 4 (integer)
# Phase the corrugation so RIDGE CRESTS sit on the E/W domain edges (x = 0 and x = L):
# the edges are then natural drainage divides, which (with E/W walls in input.yml)
# removes the sideways edge effect. Valleys sit fully inside the domain.
ridge = 30.0 * (0.5 + 0.5 * np.cos(2.0 * np.pi * x / lam))   # 30 m on rises (edges), 0 in valleys
z_base = 20.0 + (30.0 / L) * y                               # gentle south tilt (drains south)
z = z_base + ridge + rng.uniform(0.0, 2.0, x.shape)

# Uplift tapered to zero at the southern outlet (pinned base level), full inland.
ramp = np.clip(y / 8.0e3, 0.0, 1.0)                          # 0 at outlet -> 1 inland (>8 km)
t1 = 0.005e-3 * ramp                                         # Stage 1: near-stable (crust forms)
t2 = 0.060e-3 * ramp                                         # Stage 2: base-level fall (dissection)

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
os.makedirs(outdir, exist_ok=True)
np.savez_compressed(os.path.join(outdir, "gospl_mesh_valleys.npz"),
                    v=v, c=cells, z=z.astype(np.float64),
                    t1=t1.astype(np.float64), t2=t2.astype(np.float64))
print(f"wrote gospl_mesh_valleys: {len(x)} verts, z {z.min():.0f}..{z.max():.0f} m (low relief) | "
      f"{int(round(L/lam))} valleys | stage-1 uplift 0..{t1.max()*1e3:.3f}, "
      f"stage-2 0..{t2.max()*1e3:.3f} mm/yr (0 at outlet)")
