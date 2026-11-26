# Global data on a goSPL mesh (global)

Put your own global data (elevation, rainfall, uplift, temperature, erodibility, ...) from a
**NetCDF** file on a goSPL global mesh, with checks that stop and explain how to fix a grid
that does not cover the sphere properly.

## 1. Get a global mesh

**Download** a uniform global mesh from the
[goSPL meshes folder](https://www.dropbox.com/scl/fo/kior7a0kf9ylqxnnn7vk5/AJfSqKiefxvb4fHik_uRfeQ?rlkey=mh9hpd1z0i7u8qwlgspdo2zdf&st=5pj3fxke&dl=0):

| Mesh | Node spacing (mean, range) | Nodes | `gospl_mesh_*.npz` | `mesh_*.nc` | Use |
|---|---|---|---|---|---|
| `global_20km` | 20.0 km (15.5 to 26.9) | 1,480,442 | 61 MB | 1.9 GB | tests, long runs, a workstation |
| `global_10km` | 10.0 km (7.6 to 13.6) | 5,916,934 | 249 MB | 7.5 GB | continental detail, a small cluster |
| `global_8km` | 8.0 km (6.1 to 11.0) | 9,246,354 | 390 MB | 11.7 GB | finer drainage, a cluster |
| `global_5km` | 5.0 km (3.7 to 6.8) | 23,655,606 | 1.0 GB | 29.9 GB | high resolution, an HPC facility |

All four are uniform (the same spacing everywhere), on a sphere of radius 6,371,220 m, with at
most 8 neighbours per vertex (goSPL accepts up to 12). As a guide to the cost, the paper's
strong-scaling runs used 96 to 528 processors for 5 km, 24 to 384 for 8 km and 16 to 288 for
10 km; the size form of
[Which modules do I need?](https://gospl.readthedocs.io/en/dev/user_guide/reference/modules.html)
estimates the time and memory of your own set-up.

Each folder holds:

- **`gospl_mesh_<w>km.npz`: the file goSPL needs.** `v` (vertex coordinates, m), `c` (triangles,
  0-based) and `z` (elevation, **zeros**: put your elevation in it, see below). This is all a
  goSPL run reads (`domain: npdata: ['gospl_mesh_<w>km', 'v', 'c', 'z']`), and all this notebook
  uses.
- **`mesh_<w>km.nc`: usually not needed.** The same mesh as an MPAS/UGRID NetCDF, used only by the
  uxarray workflow of the [continental_flux](../continental_flux) `build_inputs` notebook
  (`shared_scripts/umeshFcts`: `inter2UGRID` interpolates gridded data on the mesh cells,
  `get_Tectonic` derives the vertical tectonic forcing from successive paleo-elevations, and the
  file can be opened in ParaView to look at the Voronoi cells). It is large because it stores the
  full Voronoi geometry (cell areas, edges, vertices and the neighbours of each), which goSPL
  rebuilds itself from `v` and `c`. Download it only for that workflow.
- `info.json`: node and triangle counts, spacing, maximum vertex degree.

**Or build your own** (any resolution, or refined where you need it, e.g. fine on the
continents and coarse in the deep ocean) with the
[continental_flux `build_inputs` notebook](../continental_flux/build_inputs.ipynb).
`make_global_meshes.py` is the script that made the downloadable meshes.

## 2. Put your data on it

`build_inputs.ipynb` opens a NetCDF variable, checks it, interpolates it on the mesh
vertices and saves the `npz` goSPL reads (`map: ['rain_mesh', 'r']` in the input file).
The functions are in `meshdata.py`:

```python
import meshdata as md
lon, lat = md.mesh_lonlat("gospl_mesh_10km.npz")
grid = md.open_global_grid("my_rain.nc", var="precip")      # checked
rain = md.interpolate_to_mesh(grid, lon, lat)                # bilinear
np.savez("rain_mesh.npz", r=rain)
```

The grid must cover the whole sphere:

- a regular longitude/latitude grid (1-D `lon` and `lat` coordinates);
- longitudes from **-180 to 180** or from **0 to 360**, with **both ends present**
  (the 0 to 360 case is moved to -180 to 180 automatically);
- latitudes from **-90 to 90**, with **both poles present**;
- no missing value.

Otherwise `check_global_grid` stops with a message that says what is wrong and gives the
`xarray` line that fixes it (cell-centred grids without the ±180 column or the poles,
latitudes from north to south, other coordinate names, extra dimensions such as time,
missing values). The notebook shows each case and its fix.

### Check a file before using it: `check_global_grid`

`check_global_grid` (in `meshdata.py`) is the gate every grid goes through: `open_global_grid`
and `interpolate_to_mesh` call it, so a grid that would give wrong values on the mesh never
reaches it. It returns the grid with longitudes from -180 to 180 (a 0 to 360 grid is moved, the
180 meridian repeated at both ends), or stops with a `GridError` that says what is wrong and
gives the `xarray` line that fixes it. Run it on a file from a terminal:

```bash
python meshdata.py check my_data.nc precip          # variable name optional if the file has one
python meshdata.py interp gospl_mesh_10km.npz my_data.nc precip rain_mesh.npz r   # check, then put on the mesh
```

`check` prints `OK` with the grid's size and range, or `NOT USABLE` with the reason and the fix
(exit code 1, so it can stop a script). `interp` writes the npz goSPL reads (add `--nearest`
for classes). In Python, `md.check_global_grid(da)` does the same on a DataArray you already
have.

What it checks, in order, and the fix it gives:

| Check | Why goSPL needs it | Fix |
|---|---|---|
| one variable (a DataArray) | a map is one field | `da = ds['precip']` |
| coordinates named lon/lat (or longitude, latitude, x, y, ...) | to know which axis is which | `da = da.rename({'LONGXY': 'lon', 'LATIXY': 'lat'})` |
| only lon and lat dimensions | a map is one time, one level | `da = da.isel(time=0)` |
| 1-D (regular) coordinates | the interpolation is on a lon/lat grid | regrid first (e.g. xESMF) |
| numeric values | | `da = da.astype(float)` |
| latitudes increasing, no repeats | | `da = da.sortby('lat')`, `da.drop_duplicates('lat')` |
| latitudes from -90 to 90, **both poles present** | vertices near the poles would fall outside the grid | add the poles: `da.interp(lat=np.r_[-90., da.lat.values, 90.], kwargs={'fill_value': 'extrapolate'})` |
| longitudes increasing, no repeats | | `da = da.sortby('lon')`, `da.drop_duplicates('lon')` |
| longitudes from -180 to 180 or 0 to 360, **both ends present** | the grid must close around the globe; a cell-centred grid (-179.75 to 179.75) leaves a gap at the date line | close it with the opposite column: `xr.concat([da.isel(lon=-1).assign_coords(lon=-180.), da, da.isel(lon=0).assign_coords(lon=180.)], dim='lon')` |
| no missing value (NaN) | goSPL needs a value at every vertex | `da = da.fillna(0.)` or `da.interpolate_na('lon')` |

**Elevation.** goSPL reads the elevation from the mesh file (`domain: npdata:
['gospl_mesh_10km', 'v', 'c', 'z']`). Interpolate yours and save it in `z`:

```python
mesh = dict(np.load("gospl_mesh_10km.npz"))
mesh["z"] = md.interpolate_to_mesh(md.open_global_grid("my_topo.nc", var="z"), lon, lat)
np.savez_compressed("my_mesh_10km.npz", **mesh)
```

Use `method="nearest"` for integer classes (rock types, provenance regions).

## Requirements

The goSPL-examples environment (`xarray`, `netCDF4`, `scipy`, `matplotlib`).
