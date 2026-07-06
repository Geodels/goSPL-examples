"""
Soil-generation example figure (fig_ex_soil.pdf) for the GMD paper, from the
pre-computed soil_generation run (sim_river_soil) and its view_Results notebook.
Uses gospl.analyse.gridexport: grid_export rasterises the surface (carrying the
soil thickness soilH), and basin_rivers traces the channel network of basin 1179.
Panels: (a) erosion/deposition, (b) soil thickness, (c) drainage basins with the
basin-1179 stream network, (d) the basin-1179 longitudinal profile with soil
thickness along the stream, and (e) the basin's soil-thickness-versus-elevation
distribution.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gospl.analyse.gridexport import grid_export, basin_rivers

EX = os.path.dirname(os.path.abspath(__file__))
STEP, RESO, BID = 10, 250, 1179
os.chdir(os.path.dirname(os.path.abspath(__file__)))

g = grid_export(os.path.join(EX, "sim_river_soil/h5"), os.path.join(EX, "data/gospl_mesh.npz"),
                STEP, spacing=RESO)
riv = basin_rivers(g, basin_id=BID, area_threshold=5e6)
ms = riv["main_stem"]

x, y = np.asarray(g["x"]) / 1e3, np.asarray(g["y"]) / 1e3
def masked(key):
    f = np.array(g[key], float); m = np.asarray(g["mask"], bool)
    return np.where(m, f, np.nan) if m.shape == f.shape else f
erodep, soilH, basin, elev = masked("erodep"), masked("soilH"), masked("basin"), masked("elev")

# soil thickness sampled along the main stem (nearest grid cell)
gx, gy = np.asarray(g["x"]), np.asarray(g["y"])
ix = np.clip(np.round((ms["x"] - gx[0]) / (gx[1] - gx[0])).astype(int), 0, gx.size - 1)
iy = np.clip(np.round((ms["y"] - gy[0]) / (gy[1] - gy[0])).astype(int), 0, gy.size - 1)
soil_stem = np.asarray(g["soilH"])[iy, ix]
edmax = round(float(np.nanpercentile(np.abs(erodep), 98)), -1) or 50.0
smax = round(float(np.nanpercentile(soilH[np.isfinite(soilH) & (soilH > 0)], 96)), 1)
print(f"basin {BID}: stem {ms['dist'].max()/1e3:.0f} km | edmax={edmax} smax={smax}")

fig = plt.figure(figsize=(12.4, 8.4))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.82], hspace=0.14, wspace=0.42)

def base_map(ax):
    ax.set_aspect("equal"); ax.set_xlabel("x (km)")

def cbar(ax, mappable=None, label=""):
    """Reserve an identical colour-bar strip on every map so all three are the
    same size; the strip is left blank on the categorical basin panel."""
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.06)
    if mappable is None:
        cax.axis("off")
    else:
        cb = fig.colorbar(mappable, cax=cax); cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)

# --- (a) erosion / deposition ----------------------------------------------
axa = fig.add_subplot(gs[0, 0]); base_map(axa); axa.set_ylabel("y (km)")
im = axa.pcolormesh(x, y, erodep, cmap="bwr", vmin=-edmax, vmax=edmax, shading="auto", rasterized=True)
axa.set_title("(a) erosion / deposition", fontsize=10)
cbar(axa, im, "m")

# --- (b) soil thickness -----------------------------------------------------
axb = fig.add_subplot(gs[0, 1]); base_map(axb)
im = axb.pcolormesh(x, y, soilH, cmap="YlOrBr", vmin=0, vmax=smax, shading="auto", rasterized=True)
axb.set_title("(b) soil thickness", fontsize=10)
cbar(axb, im, "m")

# --- (c) drainage basins + basin-1179 stream network ------------------------
axc = fig.add_subplot(gs[0, 2]); base_map(axc)
axc.pcolormesh(x, y, basin % 20, cmap="tab20", shading="auto", rasterized=True, alpha=0.55)
axc.contour(x, y, (basin == BID).astype(float), levels=[0.5], colors="k", linewidths=1.0)
for tb in riv["tributaries"]:
    axc.plot(tb["x"] / 1e3, tb["y"] / 1e3, color="#1f4e79", lw=0.3, alpha=0.7)
axc.plot(ms["x"] / 1e3, ms["y"] / 1e3, color="k", lw=1.8)
axc.plot(ms["x"][0] / 1e3, ms["y"][0] / 1e3, "v", color="w", mec="k", ms=8)  # outlet
axc.set_title(f"(c) drainage basins & basin {BID} network", fontsize=10)
cbar(axc)                                        # blank strip -> same map size as (a),(b)

# --- (d) longitudinal profile + soil thickness along the stem ---------------
axd = fig.add_subplot(gs[1, 0:2])
d = ms["dist"] / 1e3
axd.plot(d, ms["elev"], color="k", lw=1.8, label="river profile (elevation)")
axd.set_xlabel("distance from outlet (km)"); axd.set_ylabel("elevation (m)")
axd.set_title(f"(d) basin {BID} longitudinal profile", fontsize=10)
axd.grid(True, ls=":", lw=0.5, alpha=0.5)
axs = axd.twinx()
axs.plot(d, soil_stem, color="#c07a30", lw=1.4, alpha=0.9, label="soil thickness")
axs.set_ylabel("soil thickness (m)", color="#c07a30")
axs.tick_params(axis="y", colors="#c07a30")
l1, la1 = axd.get_legend_handles_labels(); l2, la2 = axs.get_legend_handles_labels()
axd.legend(l1 + l2, la1 + la2, fontsize=8, frameon=False, loc="upper left")

# --- (e) soil thickness vs elevation over the basin -------------------------
axe = fig.add_subplot(gs[1, 2])
sel = (basin == BID) & np.isfinite(soilH) & np.isfinite(elev)
ze, se = elev[sel], soilH[sel]
hb = axe.hexbin(ze, se, gridsize=40, cmap="Greys", bins="log", mincnt=1)
bins = np.linspace(np.nanpercentile(ze, 1), np.nanpercentile(ze, 99), 25)
idx = np.digitize(ze, bins)
med = [np.median(se[idx == k]) if np.any(idx == k) else np.nan for k in range(1, len(bins))]
axe.plot(0.5 * (bins[:-1] + bins[1:]), med, color="#c07a30", lw=2.0, label="median")
axe.set_xlabel("elevation (m)"); axe.set_ylabel("soil thickness (m)")
axe.set_title("(e) soil thickness vs elevation", fontsize=10)
axe.legend(fontsize=8, frameon=False, loc="upper right")
fig.colorbar(hb, ax=axe, shrink=0.8, pad=0.02).set_label("count", fontsize=8)

fig.savefig("fig_ex_soil.pdf", bbox_inches="tight")
fig.savefig("fig_ex_soil.png", dpi=150, bbox_inches="tight")
print("wrote fig_ex_soil.pdf / .png")
