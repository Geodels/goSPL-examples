"""
Stratigraphic-record + dual-lithology example figure (fig_ex_strat.pdf) for the GMD
paper. Uses the pre-computed output of the stratigraphic_record / dual_lithology
examples (same prograding margin) and the gospl.analyse.stratasection toolkit:
  (a) facies cross-section (strati run) with the three well sites marked;
  (b) Wheeler (chronostratigraphic) diagram;  (c) the imposed sea-level forcing;
  (d) coarse-fraction wells and (e) porosity wells at the topset/foreset/bottomset
      from the dual-lithology run, sharing a common elevation axis.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from gospl.analyse.stratasection import load_strata, cross_section, synthetic_well, wheeler

CMAP_T, CMAP_C = "Blues", "YlOrBr"      # layer thickness | coarse fraction

STR = os.path.dirname(os.path.abspath(__file__))
DUAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dual_lithology")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = load_strata(os.path.join(STR, "strati/h5"), os.path.join(STR, "inputs/gospl_mesh.npz"), step=25)
dd = load_strata(os.path.join(DUAL, "strati_dual/h5"), os.path.join(DUAL, "inputs/gospl_mesh.npz"), step=25)
sl = np.loadtxt(os.path.join(STR, "inputs/sealevel.csv"))

WELLS = [(130e3, 50e3, "topset"), (145e3, 50e3, "foreset"), (158e3, 50e3, "bottomset")]
YLIM = [-345, -20]                                   # shared elevation axis for the wells
H = np.asarray(dd["stratH"], float)
tmin, tmax = np.nanpercentile(H[H > 0], [2, 98])     # layer-thickness colour range
print(f"strati {data['nlayers']} layers | dual {dd['nlayers']} layers | thickness {tmin:.2f}-{tmax:.2f} m")

fig = plt.figure(figsize=(12.6, 11.2))
gs = GridSpec(3, 6, figure=fig, height_ratios=[1.05, 0.95, 1.15],
              hspace=0.42, wspace=0.55)

# --- (a) facies cross-section + well sites ---------------------------------
axa = fig.add_subplot(gs[0, 0:6])
cross_section(data, kind="x", color_by="facies", layer_lines=1, vexag=100,
              xlim=[0, 200.e3], ylim=[-400, 300], ax=axa)
axa.set_title("(a) Stratal architecture (facies), with well sites", fontsize=10)
# cross_section may keep its x-axis in metres or km — scale the markers to match
xr = axa.get_xlim()
xsc = 1.0 if xr[1] > 5.0e3 else 1.0e-3
for wx, wy, lab in WELLS:
    axa.plot(wx * xsc, 20.0, "kv", ms=9, mfc="w", zorder=6, clip_on=False)
    axa.annotate(lab, (wx * xsc, 20.0), xytext=(0, 8), textcoords="offset points",
                 ha="center", fontsize=7.5, zorder=6)

# --- (b) Wheeler diagram ----------------------------------------------------
axb = fig.add_subplot(gs[1, 0:3])
wheeler(data, kind="x", at=25000, color_by="facies", xlim=[100.e3, 150.e3],
        legend_loc=4, dt=5e3, ax=axb)
axb.set_title("(b) Wheeler diagram (facies)", fontsize=10)

# --- (c) imposed sea-level forcing -----------------------------------------
axc = fig.add_subplot(gs[1, 3:6])
mm = sl[:, 0] <= 2.5e5
axc.plot(sl[mm, 0] / 1e3, sl[mm, 1], "-", color="#2e6da4", lw=1.8)
axc.axhline(0, color="#888", lw=0.8, ls=":")
axc.set_xlabel("time (ky)"); axc.set_ylabel("relative sea level (m)")
axc.set_title("(c) Imposed sea-level forcing", fontsize=10)
axc.grid(True, ls=":", lw=0.5, alpha=0.5)

# --- (d) layer-thickness wells (left, with elevation axis) ------------------
ax_t, ax_c = [], []
for j, (wx, wy, lab) in enumerate(WELLS):
    ax = fig.add_subplot(gs[2, j])
    synthetic_well(dd, wx, wy, color_by="thickness", vmin=tmin, vmax=tmax, cmap=CMAP_T,
                   colorbar=False, ylim=YLIM, ax=ax)
    ax.set_title(lab, fontsize=8.5)
    ax.set_ylabel("elevation (m)" if j == 0 else ""); ax.set_ylim(YLIM)
    if j > 0:
        ax.set_yticklabels([])
    ax_t.append(ax)
# --- (e) coarse-fraction wells (right; y-axis omitted, same as left) --------
for j, (wx, wy, lab) in enumerate(WELLS):
    ax = fig.add_subplot(gs[2, 3 + j])
    synthetic_well(dd, wx, wy, color_by="coarse", vmin=0.0, vmax=1.0, cmap=CMAP_C,
                   colorbar=False, ylim=YLIM, ax=ax)
    ax.set_title(lab, fontsize=8.5)
    ax.set_ylabel(""); ax.set_ylim(YLIM); ax.set_yticklabels([])
    ax_c.append(ax)

sm_t = ScalarMappable(Normalize(tmin, tmax), cmap=CMAP_T); sm_t.set_array([])
sm_c = ScalarMappable(Normalize(0.0, 1.0), cmap=CMAP_C); sm_c.set_array([])
cbt = fig.colorbar(sm_t, ax=ax_t, orientation="horizontal", location="bottom",
                   shrink=0.85, pad=0.13, aspect=32)
cbt.set_label("(d) layer thickness (m)", fontsize=9)
cbc = fig.colorbar(sm_c, ax=ax_c, orientation="horizontal", location="bottom",
                   shrink=0.85, pad=0.13, aspect=32)
cbc.set_label("(e) coarse fraction", fontsize=9)

fig.savefig("fig_ex_strat.pdf", bbox_inches="tight")
fig.savefig("fig_ex_strat.png", dpi=150, bbox_inches="tight")
print("wrote fig_ex_strat.pdf / .png")
