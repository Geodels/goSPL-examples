# Global data on a goSPL mesh (global)

Put your own global data (elevation, rainfall, uplift, temperature, erodibility, ...) from a
**NetCDF** file on a goSPL global mesh, with checks that stop and explain how to fix a grid
that does not cover the sphere properly.

## 1. Get a global mesh

**Download** a uniform global mesh from the
[goSPL meshes folder](https://www.dropbox.com/scl/fo/kior7a0kf9ylqxnnn7vk5/AJfSqKiefxvb4fHik_uRfeQ?rlkey=mh9hpd1z0i7u8qwlgspdo2zdf&st=8c3i8fir&dl=0):

| Mesh | Cell width | Mean edge | Nodes | Use |
|---|---|---|---|---|
| `global_20km` | 20 km | ~11.5 km | ~1.5 million | tests, long runs, a workstation |
| `global_10km` | 10 km | ~5.7 km | ~5.9 million | regional detail on a global model, a cluster |
| `global_5km` | 5 km | ~2.8 km | ~23.6 million | high resolution, an HPC facility |

Each folder holds:

- `gospl_mesh_<w>km.npz`: the goSPL mesh, `v` (vertex coordinates, m), `c` (triangles) and
  `z` (elevation, **zeros**: put your elevation in it, see below);
- `mesh_<w>km.nc`: the same mesh as an MPAS/UGRID NetCDF, for the uxarray workflow of the
  [continental_flux](../continental_flux) `build_inputs` notebook;
- `info.json`: node count and edge lengths.

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
