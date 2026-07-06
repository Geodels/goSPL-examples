"""
Two-stage tropical ferricrete relief-inversion example figure (fig_ex_ferricrete.pdf)
for the GMD paper, from the pre-computed ferricrete_inversion run (sim_ferricrete).
The textbook laterite-mesa story on a simple, controlled landscape:

  STAGE 1 (wet, tectonically quiet, 0-5 Myr): a low-relief plain with broad river
  valleys. The water table sits near the surface across the wet valley floors, so an
  iron ferricrete forms there as a broad, resistant sheet (it does not form on the
  drier rises). No crust is prescribed: it grows in place.
  STAGE 2 (drier, base-level fall, 5-15 Myr): the interior is uplifted against a
  pinned outlet, so rivers incise. Where incision breaches the ferricrete it guts the
  soft saprolite beneath and carves new valleys, while the ferricrete-capped ground
  resists and is left standing as ferricrete-capped MESAS: the relief inverts.

Panels: (a-d) the iron ferricrete (duricrust) at 2.5, 5, 10 and 15 Myr over a grey
hillshade (broad valley-floor sheet -> caps on the dissected mesas); (e) a topographic
cross-section at 15 Myr with the ferricrete cap on the mesa tops; (f) the ratio of
valley-floor to ridge erosion through the two stages (below 1 = the ferricrete protects
the valley floors; ~0.3 while armouring, rising as dissection strips the relict crust);
(g) the ferricrete through time: total volume and where it sits in the relief (its mean
elevation rank).
Post-processed with gospl.analyse.gridexport.grid_export.
"""
import os
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gospl.analyse.gridexport import grid_export

EX = os.path.dirname(os.path.abspath(__file__))
H5, MESH, RESO = os.path.join(EX, "sim_ferricrete/h5"), os.path.join(EX, "inputs/gospl_mesh_valleys.npz"), 300
DT_OUT, STAGE1_MYR = 0.25, 5.0                 # Myr per output, end of stage 1
MAP_MYR = [2.5, 5.0, 10.0, 15.0]               # snapshot times for the ferricrete maps
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def gx(step):
    return grid_export(H5, MESH, step, spacing=RESO)
def fld(gg, k):
    return np.array(gg[k], float)

g0 = gx(0)
xe = np.asarray(g0["x"]); ye = np.asarray(g0["y"]); x, y = xe / 1e3, ye / 1e3
extent = [x.min(), x.max(), y.min(), y.max()]
lam = 12.5e3
X, Y = np.meshgrid(xe, ye)
rc = 0.5 + 0.5 * np.cos(2.0 * np.pi * X / lam)             # 1 on the rises, 0 in the valleys
inl = Y > 15e3                                             # inland of the outlet plain
z0 = fld(g0, "elev")
valley = (rc < 0.15) & inl & np.isfinite(z0)               # initial valley floors
rise = (rc > 0.85) & inl & np.isfinite(z0)                # initial rises (future mesas' neighbours)
ls = LightSource(azdeg=315, altdeg=45)

fig = plt.figure(figsize=(13.0, 7.6))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.86], hspace=0.20)
gtop = outer[0].subgridspec(1, 4, wspace=0.10)                     # ferricrete maps (tight)
gbot = outer[1].subgridspec(1, 3, wspace=0.42)                     # cross-section + evolution plots

# --- (a-d) ferricrete maps over a grey hillshade ----------------------------
dmax = 0.0
snaps = []
for tM in MAP_MYR:
    g = gx(int(round(tM / DT_OUT))); snaps.append((tM, fld(g, "elev"), fld(g, "duricrust")))
    dmax = max(dmax, float(np.nanpercentile(snaps[-1][2][snaps[-1][2] > 0.3], 98)) if (snaps[-1][2] > 0.3).any() else 1.0)
labels = "abcd"
ax0 = None
for j, (tM, z, d) in enumerate(snaps):
    ax = fig.add_subplot(gtop[0, j], sharey=ax0); ax.set_aspect("equal"); ax.set_xlabel("x (km)")
    if ax0 is None:
        ax0 = ax; ax.set_ylabel("y (km)")
    else:
        ax.tick_params(labelleft=False)                           # shared y axis
    hs = ls.hillshade(np.nan_to_num(z, nan=np.nanmin(z)), vert_exag=3.0,
                      dx=xe[1] - xe[0], dy=ye[1] - ye[0])
    ax.imshow(hs, cmap="gray", extent=extent, origin="lower", vmin=0, vmax=1, alpha=0.85)
    im = ax.imshow(np.where(d > 0.3, d, np.nan), cmap="autumn_r", extent=extent, origin="lower",
                   vmin=0, vmax=dmax, interpolation="nearest")
    ax.set_title(f"({labels[j]}) ferricrete, {tM:g} Myr", fontsize=10)
    # reserve an identical colour-bar strip on EVERY map so all four are the same size;
    # the strip is blank except on the last map, which carries the shared colour bar.
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.06)
    if j == 3:
        cb = fig.colorbar(im, cax=cax); cb.set_label("duricrust (m)", fontsize=8); cb.ax.tick_params(labelsize=7)
    else:
        cax.axis("off")

# --- time series over all outputs -------------------------------------------
steps = list(range(0, int(round(15.0 / DT_OUT)) + 1, 1))
t = np.array(steps) * DT_OUT
ero_v, ero_r, vol, cpct = [], [], [], []
for s in steps:
    g = gx(s); ed = fld(g, "erodep"); dd = fld(g, "duricrust"); zz = fld(g, "elev")
    ero_v.append(np.nanmean(ed[valley])); ero_r.append(np.nanmean(ed[rise]))
    m = inl & np.isfinite(zz); vol.append(np.nansum(dd[m]))
    cc = (dd > 0.5) & m
    if cc.sum() > 5:
        pr = np.full(zz.shape, np.nan); pr[m] = rankdata(zz[m]) / m.sum() * 100
        cpct.append(np.nanmean(pr[cc]))
    else:
        cpct.append(np.nan)
ero_v, ero_r, vol, cpct = map(np.array, (ero_v, ero_r, vol, cpct))
ratio = np.where(np.abs(ero_r) > 2.0, ero_v / ero_r, np.nan)       # valley-floor / ridge erosion

def stage_shade(ax):
    ax.axvspan(0, STAGE1_MYR, color="0.92", zorder=0)
    yl = ax.get_ylim()
    ax.text(STAGE1_MYR / 2, yl[0] + 0.1 * (yl[1] - yl[0]), "stage 1", ha="center", fontsize=8, color="0.4")
    ax.text((STAGE1_MYR + 15) / 2, yl[0] + 0.1 * (yl[1] - yl[0]), "stage 2", ha="center", fontsize=8, color="0.4")

# --- (e) cross-section: ferricrete capping the mesas at 15 Myr --------------
tX, zX, dX = snaps[-1]                                            # last snapshot (15 Myr)
capcount = ((dX > 0.3) & np.isfinite(zX)).sum(axis=1).astype(float)
capcount[y < 20] = -1                                            # transect inland of the outlet plain
iyt = int(np.nanargmax(capcount))
EXAG = 6                                                          # vertical exaggeration of the thin crust
zr, dr = zX[iyt, :], dX[iyt, :]
axe = fig.add_subplot(gbot[0, 0])
axe.fill_between(x, zr, zr + np.where(dr > 0.3, dr, 0.0) * EXAG, color="#d1662a",
                 alpha=0.9, label=f"ferricrete (×{EXAG})")
axe.plot(x, zr, color="k", lw=1.6, label="surface")
axe.set_xlabel(f"x (km) along y = {y[iyt]:.0f} km"); axe.set_ylabel("elevation (m)")
axe.set_title(f"(e) cross-section at {tX:g} Myr", fontsize=10)
axe.legend(fontsize=8, frameon=False, loc="upper left")
axe.grid(True, ls=":", lw=0.5, alpha=0.5)

# --- (f) valley-floor / ridge erosion ratio ---------------------------------
axf = fig.add_subplot(gbot[0, 1])
axf.axhline(1.0, color="k", lw=0.9, ls="--")                     # equal erosion (no armouring)
axf.fill_between(t, ratio, 1.0, where=ratio < 1.0, color="#f0d0b0", alpha=0.7)
axf.plot(t, ratio, "-", color="#8c2d04", lw=2.4)
axf.text(10., 1.03, "equal erosion", fontsize=7.5, color="0.3")
axf.set_ylim(0.0, 1.12)
axf.set_xlabel("time (Myr)"); axf.set_ylabel("valley-floor / ridge\nerosion ratio")
axf.set_title("(f) valley vs ridge erosion ratio", fontsize=10)
axf.grid(True, ls=":", lw=0.5, alpha=0.5); stage_shade(axf)

# --- (g) ferricrete evolution: volume and elevation rank --------------------
axg0 = fig.add_subplot(gbot[0, 2])
axg0.plot(t, vol / 1e3, "-", color="#8B4513", lw=2.2, label="total ferricrete volume")
axg0.set_xlabel("time (Myr)"); axg0.set_ylabel("ferricrete volume\n(10$^3$ m, gridded)", color="#8B4513")
axg0.tick_params(axis="y", colors="#8B4513")
axg0.set_title("(g) ferricrete: volume and where it sits", fontsize=10)
axg0.grid(True, ls=":", lw=0.5, alpha=0.5)
axg1 = axg0.twinx()
axg1.plot(t, cpct, "-", color="#178a5a", lw=2.2, label="mean elevation rank")
axg1.set_ylabel("mean elevation percentile\nof ferricrete (%)", color="#178a5a")
axg1.tick_params(axis="y", colors="#178a5a"); axg1.set_ylim(0, 100)
axg1.axhline(50, color="#178a5a", lw=0.7, ls=":")
stage_shade(axg0)

fig.savefig("fig_ex_ferricrete.pdf", bbox_inches="tight")
fig.savefig("fig_ex_ferricrete.png", dpi=150, bbox_inches="tight")
print(f"wrote fig_ex_ferricrete.pdf / .png | erosion ratio v/r {np.nanmin(ratio):.2f}->{ratio[-1]:.2f} | "
      f"ferricrete elevation percentile {np.nanmin(cpct):.0f}->{cpct[-1]:.0f}")
