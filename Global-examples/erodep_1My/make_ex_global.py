"""
Global-simulation example figure (fig_ex_global.pdf) for the GMD paper, from the
pre-computed erodep_1My global run and its view_Results notebook. PyGMT draws a
2 x 2 grid of global maps of output step 10 (after 1 Myr):
  (a) surface elevation (geo);
  (b) cumulative erosion / deposition (vik, over a greyscale topography);
  (c) per-basin river-mouth WATER discharge;
  (d) per-basin river-mouth SEDIMENT load.
Gridded field from results/surface10.nc; outlet fluxes from flowsed/{flow,sed}10.csv
(gospl-catchment). Panel titles are placed above each map. (The spherical-harmonic
flexural isostasy is solved by the run and available as an output field but is not
plotted here.)
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import pygmt

EX = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ds = xr.open_dataset(os.path.join(EX, "results/surface10.nc"))
elev, erodep, flexiso = ds.elev, ds.erodep, ds.flexIso
ero = ds.where(ds.erodep<5).where(ds.elev>0).erodep
depo = ds.where(ds.erodep>100).erodep

edmax = round(float(np.nanpercentile(np.abs(erodep.values), 99)), -2) or 2000.0
fxmax = round(float(np.nanpercentile(np.abs(flexiso.values), 99)), -2) or 1000.0

flow = pd.read_csv(os.path.join(EX, "flowsed/flow10.csv")).nlargest(200, "val")
sed = pd.read_csv(os.path.join(EX, "flowsed/sed10.csv")).nlargest(200, "val")
rFA = np.log10(flow["val"].values); rSed = np.log10(sed["val"].values)

PROJ, REG = "W8c", "d"
DX, DY = "9.4c", "-5.6c"
CBP = "jBC+o0c/-1.c+w5.6c/0.24c+h"
CB_FONT = dict(FONT_ANNOT_PRIMARY="9p,Helvetica,black", FONT_LABEL="10p,Helvetica,black")

# Font used for latitude (y-axis) tick labels on the map frames — smaller than
# the longitude (x-axis) labels, which keep the globally-set FONT_ANNOT_PRIMARY.
LAT_FONT = "4p,Helvetica,black"

fig = pygmt.Figure()
pygmt.config(FONT="5p,Helvetica,black", FONT_TITLE="8p,Helvetica",
             MAP_TITLE_OFFSET="-6p", MAP_FRAME_PEN="0.6p")

def coast():
    fig.grdcontour(grid=elev, levels=0.1, limit=[0.0, 0.09], pen="0.4p,black")

def map_frame(title):
    """Draw the map frame/gridlines/title with longitude (x) annotated at the
    default font size, then overlay latitude (y) annotations in a smaller font."""
    fig.basemap(region=REG, projection=PROJ, frame=["xafg", title])
    with pygmt.config(FONT_ANNOT_PRIMARY=LAT_FONT):
        fig.basemap(region=REG, projection=PROJ, frame=["yafg"])

# --- (a) elevation map (top-left) ------------------------
pygmt.makecpt(cmap="geo", series=[-10000, 10000])
map_frame("+t(a) elevation (1 Myr)")
fig.grdimage(elev, cmap=True); coast()
with pygmt.config(**CB_FONT):
  fig.colorbar(position=CBP, frame=[f"a2500", "x+lElevation (m)"])

# --- (b) cumulative erosion / deposition (top-right) -----------------------
fig.shift_origin(xshift=DX)
pygmt.makecpt(cmap="gray", series=[-20000, 6000])
map_frame("+t(b) erosion / deposition (1 Myr)")
fig.grdimage(elev, frame=False)
pygmt.makecpt(cmap="vik", series=[-3000, 3000])
fig.grdimage(ero, frame=False, nan_transparent=True)
fig.grdimage(depo, frame=False, nan_transparent=True); coast()
with pygmt.config(**CB_FONT):
  fig.colorbar(position=CBP, frame=[f"a1000", "x+lErosion / Deposition (m)"])

# --- (b) flexural isostatic deflection (top-right) -------------------------
# fig.shift_origin(xshift=DX)
# pygmt.makecpt(cmap="bam", series=[-fxmax, fxmax])
# map_frame("+t(b) flexural isostasy")
# fig.grdimage(flexiso, cmap=True); coast()
# with pygmt.config(**CB_FONT):
#   fig.colorbar(position=CBP, frame=[f"a{int(fxmax/2)}", "x+lflexural deflection", "y+lm"])

# --- (c) water-discharge outlets (bottom-left) -----------------------------
fig.shift_origin(xshift="-" + DX, yshift=DY)
pygmt.makecpt(cmap="gray", series=[-20000, 6000])
map_frame("+t(c) water flux at basin outlets")
fig.grdimage(elev, cmap=True); coast()
pygmt.makecpt(cmap="devon", series=[rFA.min(), rFA.max()], reverse=True)
fig.plot(x=flow["lon"], y=flow["lat"], style="cc", pen="0.1p,black",
         size=4.0e-5 * 2 ** rFA, fill=rFA, cmap=True)
with pygmt.config(**CB_FONT):
  fig.colorbar(position=CBP, frame="af+lWater discharge (log@-10@- m@+3@+ yr@+-1@+)")

# --- (d) sediment-load outlets (bottom-right) ------------------------------
fig.shift_origin(xshift=DX)
pygmt.makecpt(cmap="gray", series=[-20000, 6000])
map_frame("+t(d) sediment flux at basin outlets")
fig.grdimage(elev, cmap=True); coast()
pygmt.makecpt(cmap="buda", series=[rSed.min(), rSed.max()])
fig.plot(x=sed["lon"], y=sed["lat"], style="cc", pen="0.2p,black",
         size=4.0e-4 * 2 ** rSed, fill=rSed, cmap=True)
with pygmt.config(**CB_FONT):
  fig.colorbar(position=CBP, frame="af+lSediment load (log@-10@- m@+3@+ yr@+-1@+)")

fig.savefig("fig_ex_global.pdf")
fig.savefig("fig_ex_global.png", dpi=200)
print(f"wrote fig_ex_global.pdf / .png | edmax={edmax:.0f} fxmax={fxmax:.0f} | "
      f"FA {rFA.min():.1f}-{rFA.max():.1f} sed {rSed.min():.1f}-{rSed.max():.1f}")