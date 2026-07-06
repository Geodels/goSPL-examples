"""
Flow-direction comparison figure (fig_ex_flowdir.pdf) for the GMD paper. Uses the
pre-computed, already-rasterised NetCDF outputs of the flow_direction example
(three identical runs differing only in the number of downstream directions:
SFD flowdir=1, 2-neighbour flowdir=2, MFD flowdir=6) and reproduces the notebook's
flow-discharge comparison: SFD concentrates flow into thin, mesh-sensitive
dendritic threads while MFD spreads it into broad, grid-independent drainage.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import xarray as xr

EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STEP = 2                                   # output step 2 = 10 kyr
SIMS = [("sfd", "SFD  (flowdir = 1)"),
        ("2ngb", "2 neighbours  (flowdir = 2)"),
        ("mfd", "MFD  (flowdir = 6)")]
ZOOM = dict(x=slice(30e3, 50e3), y=slice(50e3, 70e3))   # 20 x 20 km window
NORM = LogNorm(vmin=1e5, vmax=1e8)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

fig, ax = plt.subplots(2, 3, figsize=(11.0, 7.0))
im = None
for j, (pfx, title) in enumerate(SIMS):
    ds = xr.open_dataset(os.path.join(EX, f"{pfx}{STEP}.nc"))
    fa = ds.FA.where(ds.elev > 0.0)        # discharge on land only
    # --- full domain ---
    im = ax[0, j].pcolormesh(ds.x / 1e3, ds.y / 1e3, fa, norm=NORM,
                             cmap="Blues", shading="auto", rasterized=True)
    ax[0, j].add_patch(patches.Rectangle((30, 50), 20, 20, lw=1.0,
                       edgecolor="k", facecolor="none"))
    ax[0, j].set_title(title, fontsize=10, fontweight="bold")
    ax[0, j].set_aspect("equal")
    ax[0, j].set_xlabel("x (km)")
    if j == 0:
        ax[0, j].set_ylabel("y (km)")
    # --- zoom ---
    fz = fa.sel(**ZOOM)
    ax[1, j].pcolormesh(fz.x / 1e3, fz.y / 1e3, fz, norm=NORM,
                        cmap="Blues", shading="auto", rasterized=True)
    ax[1, j].set_aspect("equal")
    ax[1, j].set_xlabel("x (km)")
    if j == 0:
        ax[1, j].set_ylabel("y (km)")
    ax[1, j].set_title("zoom (20$\\times$20 km)", fontsize=9)

fig.subplots_adjust(bottom=0.13, wspace=0.25, hspace=0.30)
cax = fig.add_axes([0.30, 0.05, 0.42, 0.022])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
cb.set_label("flow discharge $Q$ (m$^3$ $\cdot$ yr$^{-1}$)")
fig.savefig("fig_ex_flowdir.pdf", bbox_inches="tight")
fig.savefig("fig_ex_flowdir.png", dpi=160, bbox_inches="tight")
print("wrote fig_ex_flowdir.pdf / .png")
