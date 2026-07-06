"""
Groundwater / geochemistry example figure (fig_ex_gwgeochem.pdf) for the GMD paper,
from the pre-computed groundwater_geochem run (sim_gw_geochem). Shows the new
near-surface hydrology and weathering-geochemistry outputs: (a) water-table depth,
(b) baseflow returned to rivers, (c) capillary-fringe duricrust thickness, (d) the
two-tracer crust composition, (e) the river dissolved-solute load, and (f) the
closed per-species solute budget through time (dissolved = precipitated + exported).
Gridded fields from results/surface10.nc; the budget from gw_solute_budget.csv.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr

EX = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ds = xr.open_dataset(os.path.join(EX, "results/surface10.nc"))
x, y = ds.x.values / 1e3, ds.y.values / 1e3

def f(key):
    return np.array(ds[key].values, float)
elev = f("elev")
duri = f("duricrust")
sc, ss = f("solute_carbonate"), f("solute_silica")
tot = sc + ss
diss_cf = np.where(tot > 1e-9, sc / np.maximum(tot, 1e-30), np.nan)   # dissolved carbonate fraction

fig = plt.figure(figsize=(13.0, 9.2))
gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.42)

def mp(ax):
    ax.set_aspect("equal"); ax.set_xlabel("x (km)")
def cbar(ax, im, label):
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.06)
    cb = fig.colorbar(im, cax=cax); cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7)

# --- (a) water-table depth --------------------------------------------------
axa = fig.add_subplot(gs[0, 0]); mp(axa); axa.set_ylabel("y (km)")
wtvmax = max(5.0, round(float(np.nanpercentile(f("wtdepth"), 96))))
im = axa.pcolormesh(x, y, f("wtdepth"), cmap="YlGnBu", vmin=0, vmax=wtvmax, shading="auto", rasterized=True)
axa.set_title("(a) water-table depth", fontsize=10); cbar(axa, im, "m")

# --- (b) baseflow -----------------------------------------------------------
axb = fig.add_subplot(gs[0, 1]); mp(axb)
bf = np.where(f("baseflow") > 1.0, f("baseflow"), np.nan)
axb.pcolormesh(x, y, elev, cmap="gray", vmin=0, vmax=2600, shading="auto", rasterized=True, alpha=0.5)
im = axb.pcolormesh(x, y, bf, cmap="Blues", norm=LogNorm(vmin=1e2, vmax=1e5), shading="auto", rasterized=True)
axb.set_title("(b) baseflow to rivers", fontsize=10); cbar(axb, im, "m$^3$ yr$^{-1}$")

# --- (c) duricrust thickness ------------------------------------------------
axc = fig.add_subplot(gs[0, 2]); mp(axc)
axc.pcolormesh(x, y, elev, cmap="gray", vmin=0, vmax=2600, shading="auto", rasterized=True, alpha=0.45)
im = axc.pcolormesh(x, y, np.where(duri > 0.01, duri, np.nan), cmap="YlOrBr",
                    vmin=0, vmax=5, shading="auto", rasterized=True)
axc.set_title("(c) capillary-fringe duricrust", fontsize=10); cbar(axc, im, "m")

# --- (d) dissolved-solute composition (two tracers, whole domain) -----------
axd = fig.add_subplot(gs[1, 0]); mp(axd); axd.set_ylabel("y (km)")
lo, hi = np.nanpercentile(diss_cf, [1, 99])
im = axd.pcolormesh(x, y, diss_cf, cmap="RdBu_r", vmin=lo, vmax=hi, shading="auto", rasterized=True)
axd.set_title("(d) dissolved-solute composition", fontsize=10)
cbar(axd, im, "carbonate fraction")

# --- (e) river dissolved-solute load ----------------------------------------
axe = fig.add_subplot(gs[1, 1]); mp(axe)
rs = np.where(f("riverSolute") > 1.0, f("riverSolute"), np.nan)
axe.pcolormesh(x, y, elev, cmap="gray", vmin=0, vmax=2600, shading="auto", rasterized=True, alpha=0.5)
im = axe.pcolormesh(x, y, rs, cmap="plasma", norm=LogNorm(vmin=1e2, vmax=1e5), shading="auto", rasterized=True)
axe.set_title("(e) river dissolved-solute load", fontsize=10); cbar(axe, im, "m$^3$ yr$^{-1}$")

# --- (f) fate of the dissolved load: exported vs. precipitated --------------
axf = fig.add_subplot(gs[1, 2])
bud = np.genfromtxt(os.path.join(EX, "sim_gw_geochem/gw_solute_budget.csv"),
                    delimiter=",", names=True)
t = bud["time"][1:] / 1e3                     # drop t=0 (zeros) for the log axis
for sp, col in [("carbonate", "#c0392b"), ("silica", "#2e6da4")]:
    axf.plot(t, bud["oceanflux_" + sp][1:], "-", color=col, lw=2.0,
             label=f"{sp}: exported (seepage)")
    axf.plot(t, bud["precipitated_" + sp][1:], "--", color=col, lw=1.4,
             label=f"{sp}: precipitated in crust")
axf.set_yscale("log")
axf.set_xlabel("time (ky)"); axf.set_ylabel("cumulative solute mass (rel. units)")
axf.set_title("(f) solute fate: export vs.\\ crust", fontsize=10)
axf.legend(fontsize=7, frameon=False, loc="center right")
axf.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)

fig.savefig("fig_ex_gwgeochem.pdf", bbox_inches="tight")
fig.savefig("fig_ex_gwgeochem.png", dpi=150, bbox_inches="tight")
print("wrote fig_ex_gwgeochem.pdf / .png | wtdepth<=%.0f duri<=%.1f" %
      (np.nanmax(f("wtdepth")), np.nanmax(duri)))
