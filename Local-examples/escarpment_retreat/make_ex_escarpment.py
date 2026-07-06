"""
Escarpment-retreat example figure (fig_ex_escarpment.pdf) for the GMD paper.
Post-processes the pre-computed goSPL output in goSPL-examples/.../escarpment_retreat
(no simulation is run) exactly as the example's view_Results.ipynb does: it
rasterises output steps with gospl.analyse.gridexport.grid_export and builds a
2x2 panel — (a) final topography with shoreline, (b) erosion/deposition rate,
(c) cross-escarpment mean profiles through time with the crest-retreat track,
(d) the growing flexural-isostatic deflection.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gospl.analyse.gridexport import grid_export

EX = os.path.dirname(os.path.abspath(__file__))
H5, MESH, RESO = os.path.join(EX, "escarpment/h5"), os.path.join(EX, "data/escarpment.npz"), 500
STEPS = list(range(0, 51, 5))          # every 5 Myr (tout = 1 Myr)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

G = {s: grid_export(H5, MESH, s, spacing=RESO) for s in STEPS}
print("rasterised steps:", STEPS)

def masked(g, key):
    f = np.array(g[key], dtype=float)
    if "mask" in g:
        m = np.array(g["mask"], dtype=bool)
        if m.shape == f.shape:
            f = np.where(m, f, np.nan)
    return f

def mean_profile(g, key):
    """mean over x -> profile along y (km, values)."""
    x, y, f = np.asarray(g["x"]), np.asarray(g["y"]), masked(g, key)
    if f.shape == (len(y), len(x)):
        return y / 1e3, np.nanmean(f, axis=1)
    return y / 1e3, np.nanmean(f, axis=0)

g50 = G[50]
x, y = np.asarray(g50["x"]) / 1e3, np.asarray(g50["y"]) / 1e3
sl = float(np.asarray(g50["base_level"]))
elev50, ed50 = masked(g50, "elev"), masked(g50, "EDrate")

fig, ax = plt.subplots(2, 2, figsize=(10.6, 8.4))
(a, b), (c, e) = ax

# --- (a) final topography + shoreline --------------------------------------
im = a.pcolormesh(x, y, elev50, cmap="Spectral_r", shading="auto")
a.contour(x, y, elev50, levels=[sl], colors="k", linewidths=1.0)
a.set_title("(a) Topography at 50 Myr", fontsize=10, loc="left")
a.set_xlabel("x (km)"); a.set_ylabel("y (km)"); a.set_aspect("equal")
cb = fig.colorbar(im, ax=a, shrink=0.85, pad=0.02); cb.set_label("elevation (m)", fontsize=8)

# --- (b) erosion/deposition rate + shoreline -------------------------------
vmax = np.nanpercentile(np.abs(ed50), 98)
im = b.pcolormesh(x, y, ed50, cmap="bwr", vmin=-vmax, vmax=vmax, shading="auto")
b.contour(x, y, elev50, levels=[sl], colors="k", linewidths=1.0)
b.set_title("(b) Erosion / deposition rate at 50 Myr", fontsize=10, loc="left")
b.set_xlabel("x (km)"); b.set_ylabel("y (km)"); b.set_aspect("equal")
cb = fig.colorbar(im, ax=b, shrink=0.85, pad=0.02); cb.set_label("m yr$^{-1}$", fontsize=8)

# --- (c) cross-escarpment profiles + mid-scarp retreat track ---------------
def crossing(yy, prof, level):
    """y where the (monotone-ish) profile crosses `level` (m)."""
    d = prof - level
    idx = np.where(np.isfinite(d[:-1]) & np.isfinite(d[1:]) &
                   (np.sign(d[:-1]) != np.sign(d[1:])))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[len(idx) // 2]            # central crossing = the scarp face
    return yy[i] + (yy[i + 1] - yy[i]) * (level - prof[i]) / (prof[i + 1] - prof[i])

LEVEL = 500.0
cmap = plt.get_cmap("viridis", len(STEPS))
scarp_y, scarp_t = [], []
for i, s in enumerate(STEPS):
    yy, prof = mean_profile(G[s], "elev")
    lw, alpha, col = (1.8, 1.0, "k") if s in (0, 50) else (0.9, 0.6, cmap(i))
    c.plot(yy, prof, lw=lw, alpha=alpha, color=col)
    yc = crossing(yy, prof, LEVEL)
    scarp_y.append(yc); scarp_t.append(s)
c.plot(scarp_y, np.full_like(scarp_y, LEVEL), "-o", color="#c0392b", lw=2.0,
       ms=4, zorder=5, label=f"scarp face ({LEVEL:.0f} m), 0$\\to$50 Myr")
c.set_title("(c) Cross-escarpment profiles & scarp retreat", fontsize=10, loc="left")
c.set_xlabel("distance $y$ (km)"); c.set_ylabel("mean elevation (m)")
c.legend(fontsize=8, frameon=False, loc="lower right")
c.grid(True, ls=":", lw=0.5, alpha=0.5)
ret = abs(scarp_y[-1] - scarp_y[0])
print(f"scarp retreat 0->50 Myr: {ret:.1f} km  ({1000*ret/50:.1f} m/Myr)")

# --- (d) flexural-isostatic deflection through time ------------------------
for i, s in enumerate([10, 20, 30, 40, 50]):
    yy, prof = mean_profile(G[s], "flexIso")
    e.plot(yy, prof, lw=1.8, color=plt.get_cmap("Blues")(0.25 + 0.6 * i / 4),
           label=f"{s} Myr")
e.axhline(0, color="#888", lw=0.8, ls=":")
e.set_title("(d) Flexural isostatic deflection", fontsize=10, loc="left")
e.set_xlabel("distance $y$ (km)"); e.set_ylabel("cumulative deflection (m)")
e.legend(fontsize=8, frameon=False, loc="best", ncol=2)
e.grid(True, ls=":", lw=0.5, alpha=0.5)

plt.tight_layout(pad=0.8)
fig.savefig("fig_ex_escarpment.pdf", bbox_inches="tight")
fig.savefig("fig_ex_escarpment.png", dpi=160, bbox_inches="tight")
print("wrote fig_ex_escarpment.pdf / .png  | shoreline =", sl, "m")
