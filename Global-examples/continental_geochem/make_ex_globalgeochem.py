"""
Global regional-geochemistry example figure (fig_ex_globalgeochem.pdf) for the GMD
paper, from the pre-computed continental_geochem global run (silico_geochem). The
continents are divided into latitudinal climate-zone provinces that each shed a
characteristic weathering species (tropical -> iron/ferricrete, subtropical-arid ->
carbonate/calcrete, temperate -> silica/silcrete); chemical weathering follows a
latitude- and altitude-dependent (Arrhenius) temperature field, a capillary-fringe
duricrust forms where the water table sits at fringe depth, and the dissolved load is
routed down the rivers to the ocean. PyGMT draws a 2 x 2 grid of global maps after
1 Myr (Winkel Tripel; black lines are the 0 m palaeo-shoreline):
  (a) the temperature field that drives the weathering (the Arrhenius forcing);
  (b) the dominant duricrust species (iron / carbonate / silica), the climate-zoned
      geochemistry expressed in the crust;
  (c) the capillary-fringe duricrust thickness;
  (d) the river dissolved-solute load delivered toward the coasts;
  (e) the domain-integrated per-species solute budget through time (mass exported to
      the ocean, solid, vs locked in the duricrust, dashed).
Gridded field from results/surface10.nc (gospl-grid); the temperature forcing is the
per-vertex geochem_inputs.npz:temp mapped onto the same grid; the budget from
silico_geochem/gw_solute_budget.csv.
"""
import os
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import pygmt

EX = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "continental_flux", "vars_25_80", "mesh.npz")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ds = xr.open_dataset(os.path.join(EX, "results/surface10.nc"))
elev = ds.elev
land = elev >= 0.0

# Per-species solute budget through time (domain-integrated, cumulative).
bud = np.genfromtxt(os.path.join(EX, "silico_geochem/gw_solute_budget.csv"),
                    delimiter=",", names=True)
tb = (bud["time"] - bud["time"][0]) / 1.0e6                    # Myr since start

# Dominant duricrust species per cell (0 iron, 1 carbonate, 2 silica), where a crust
# actually exists (total thickness > 0.1 m); elsewhere masked out.
ci, cc, cs = (np.nan_to_num(ds[k].values) for k in ("crust_iron", "crust_carbonate", "crust_silica"))
tot = ci + cc + cs
dom = np.argmax(np.stack([ci, cc, cs]), axis=0).astype(float)
dom[tot < 0.1] = np.nan
dom = xr.DataArray(dom, coords=[ds.lat, ds.lon], dims=["lat", "lon"])

# Duricrust thickness and river dissolved-solute load (log10), land only.
duri = ds.duricrust.where(ds.duricrust > 0.05).where(land)
rs = np.log10(ds.riverSolute.where(ds.riverSolute > 1.0e4))

# Temperature forcing: per-vertex geochem_inputs.npz:temp mapped onto the grid.
v = np.load(MESH)["v"]
r = np.linalg.norm(v, axis=1)
mlon = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
mlat = np.degrees(np.arcsin(np.clip(v[:, 2] / r, -1, 1)))
temp = np.load(os.path.join(EX, "geochem_inputs.npz"))["temp"]
LON, LAT = np.meshgrid(ds.lon.values, ds.lat.values)
tg = griddata((mlon, mlat), temp, (LON, LAT), method="nearest")
tempda = xr.DataArray(np.where(land.values, tg, np.nan), coords=[ds.lat, ds.lon], dims=["lat", "lon"])

# Categorical CPT for the dominant-species panel (keys 0/1/2 -> labelled colours).
SPCPT = "/tmp/species_geochem.cpt"
with open(SPCPT, "w") as f:
    f.write("0\t192/57/43\t;iron\n1\t46/109/164\t;carbonate\n2\t90/143/60\t;silica\n")

PROJ, REG = "W8c", "d"
DX, DY = "9.4c", "-5.6c"
CBP = "jBC+o0c/-1.c+w5.6c/0.24c+h"
CB_FONT = dict(FONT_ANNOT_PRIMARY="9p,Helvetica,black", FONT_LABEL="10p,Helvetica,black")
LAT_FONT = "4p,Helvetica,black"
GRAY = "gray"; GRAY_SERIES = [-9000, 6000]

fig = pygmt.Figure()
pygmt.config(FONT="5p,Helvetica,black", FONT_TITLE="8p,Helvetica",
             MAP_TITLE_OFFSET="-6p", MAP_FRAME_PEN="0.6p")

def coast():
    fig.grdcontour(grid=elev, levels=0.1, limit=[0.0, 0.09], pen="0.4p,black")

def map_frame(title):
    fig.basemap(region=REG, projection=PROJ, frame=["xafg", title])
    with pygmt.config(FONT_ANNOT_PRIMARY=LAT_FONT):
        fig.basemap(region=REG, projection=PROJ, frame=["yafg"])

def graybg():
    pygmt.makecpt(cmap=GRAY, series=GRAY_SERIES)
    fig.grdimage(elev, cmap=True, frame=False)

# --- (a) temperature forcing (top-left) ------------------------------------
map_frame("+t(a) weathering temperature (@.C)")
graybg()
pygmt.makecpt(cmap="vik", series=[-30, 30])                   # warm = red, cold = blue
fig.grdimage(tempda, cmap=True, nan_transparent=True); coast()
with pygmt.config(**CB_FONT):
    fig.colorbar(position=CBP, frame=["a10", "x+lTemperature (@.C)"])

# --- (b) dominant duricrust species (top-right) ----------------------------
fig.shift_origin(xshift=DX)
map_frame("+t(b) dominant duricrust species")
graybg()
fig.grdimage(dom, cmap=SPCPT, nan_transparent=True); coast()
with pygmt.config(**CB_FONT):
    fig.colorbar(cmap=SPCPT, position=CBP)

# --- (c) duricrust thickness (bottom-left) ---------------------------------
fig.shift_origin(xshift="-" + DX, yshift=DY)
map_frame("+t(c) duricrust thickness")
graybg()
pygmt.makecpt(cmap="lajolla", series=[0, 5])
fig.grdimage(duri, cmap=True, nan_transparent=True); coast()
with pygmt.config(**CB_FONT):
    fig.colorbar(position=CBP, frame=["a1", "x+lDuricrust thickness (m)"])

# --- (d) river dissolved-solute load (bottom-right) ------------------------
fig.shift_origin(xshift=DX)
map_frame("+t(d) river dissolved-solute load")
graybg()
pygmt.makecpt(cmap="plasma", series=[float(np.nanpercentile(rs.values, 2)),
                                     float(np.nanpercentile(rs.values, 99.5))])
fig.grdimage(rs, cmap=True, nan_transparent=True); coast()
with pygmt.config(**CB_FONT):
    fig.colorbar(position=CBP, frame="x+lDissolved solute (log@-10@- m@+3@+ yr@+-1@+)")

# --- (e) per-species solute budget through time (below the 2 x 2 maps) ------
SPP = [("iron", "#c0392b"), ("carbonate", "#2e6da4"), ("silica", "#5a8f3c")]
fig.shift_origin(xshift="-8.55c", yshift="-6c")
with pygmt.config(FONT_ANNOT_PRIMARY="6p,Helvetica,black", FONT_LABEL="7p,Helvetica,black",
                  FONT_TITLE="8p,Helvetica", MAP_TITLE_OFFSET="2p"):
    fig.basemap(region=[0, 1, 1e13, 3e17], projection="X16.5c/4.2cl",
                frame=["WSne+t(e) per-species solute budget (exported solid, in crust dashed)",
                       "xa0.2f0.1+lTime (Myr)", "ya1f3+lcumulative solute (m@+3@+)"])
    for sp, col in SPP:
        fig.plot(x=tb[1:], y=bud["oceanflux_" + sp][1:], pen=f"2.2p,{col}")        # exported to ocean
        fig.plot(x=tb[1:], y=bud["precipitated_" + sp][1:], pen=f"1.4p,{col},4_3")  # locked in crust
    # compact legend of the three species colours
    leg = "\n".join(f"S 0.3c - 0.6c - 2.2p,{c} 0.9c {sp}" for sp, c in SPP)
    fig.legend(spec=__import__("io").StringIO(leg), position="jTL+o0.2c+w3.4c", box="+gwhite+p0.5p")

fig.savefig("fig_ex_globalgeochem.pdf")
fig.savefig("fig_ex_globalgeochem.png", dpi=200)
nsp = {int(k): int(v) for k, v in zip(*np.unique(dom.values[np.isfinite(dom.values)], return_counts=True))}
print(f"wrote fig_ex_globalgeochem.pdf / .png | temp {np.nanmin(tempda.values):.0f}..{np.nanmax(tempda.values):.0f} C | "
      f"dominant-species cells {nsp} | duri<= {float(ds.duricrust.max()):.1f} m")
