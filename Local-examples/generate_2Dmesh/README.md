# Building 2D meshes for goSPL (local)

goSPL runs on an unstructured triangular mesh (a UGRID/MPAS grid with its
Voronoi dual), so every starting surface, synthetic or from a digital
elevation model (DEM), must first be resampled onto such a mesh. This
example is a tutorial, with no model run: `build_inputs.ipynb` builds four
planar meshes and writes them as the `.npz` files a goSPL input reads, from
a regular grid made in Python and from an SRTM GeoTIFF tile, with a uniform
or an elevation-dependent resolution.

## What it produces

| Folder | Source | Resolution | Nodes | Fields | goSPL file |
|---|---|---|---|---|---|
| `slope_tec` | flat surface made in Python | 200 m | 290,007 | elevation `z`, uplift rate `t` | `slope_tec/gospl_mesh.npz` |
| `escarpment` | escarpment made in Python | 1 km | 104,454 | elevation `z` | `escarpment/escarpment2.npz` |
| `geotiff` | SRTM tile `data/srtm_36_04.tif` | 500 m | 228,566 | elevation `z` | `geotiff/gospl_mesh.npz` |
| `geotiff_refined` | the same tile | 250 m to 1 km, by elevation | 181,338 | elevation `z` | `geotiff_refined/gospl_mesh.npz` |

The node counts are those of the meshes in the folders. Each `.npz` file
holds the vertex coordinates `v` (x, y, z in m, with z = 0 on a planar mesh),
the triangles `c` (three vertex indices per row) and the elevation `z` (m) at
every vertex, plus the uplift rate `t` (m/yr) for `slope_tec`.

## The steps of `build_inputs.ipynb`

Every scenario follows the same five steps. The meshing helpers are in
`shared_scripts/umeshFcts.py` (imported as `scripts.umeshFcts`); they call
jigsaw (`jigsawpy`) and the MPAS mesh converter (`mpas_tools`), both in the
`environment.yml` of this repository.

1. **A regular grid** as an `xarray` Dataset on `x`, `y` coordinates (m),
   with the field `elev` (m), any other field to carry (here the uplift rate
   `tec`, m/yr) and `cellwidth`, the target triangle edge length (m) at every
   grid point.
2. **The mesh**: `ufcts.planarMesh(ds, output_path, fvtk='planar.vtk',
   fumpas=True, voro=True)` meshes the rectangle covered by the grid with
   jigsaw, following `cellwidth` (a uniform `cellwidth` is passed to jigsaw as
   a single size), converts it to an MPAS/UGRID file (`base2D.nc`) and writes
   the triangles to `planar.vtk`. If its output files already exist in the
   folder, `planarMesh` reuses them: delete the folder to rebuild a mesh.
3. **The fields on the mesh**: `ufcts.inter2UGRID(ds[['elev', ...]], mapds,
   output_path, var_name, type='face', latlon=False)` interpolates the grid
   fields onto the mesh nodes (the MPAS cell centres) and writes them to
   `<var_name>.nc`. The notebook prints the node and face counts and the
   minimum, maximum and mean edge lengths (km).
4. **A check**: the triangles of `planar.vtk` are read with `meshio`;
   `VoroBuild` from `gospl.mesher.meshfunc` computes the Voronoi area of each
   node (its control volume, printed as the minimum and maximum `Cell area`
   in km2), and `init.vtk` is written with the interpolated fields for
   ParaView.
5. **The goSPL file**: `np.savez_compressed` writes `v`, `c`, `z` (and `t`).

### The four scenarios

- **Slope tectonics (`slope_tec`)**: a 501 x 501 grid at 200 m (a 100 x
  100 km square). The surface is flat at 100 m plus Gaussian noise of 0.05 m
  standard deviation, which breaks the symmetry and seeds the drainage. The
  uplift rate `tec` increases linearly along x from 0 to 0.005 m/yr (5 mm/yr).
  The `implicit_timestepping` example uses a mesh built with this recipe.
- **Escarpment (`escarpment`)**: a 301 x 301 grid at 1 km (a 300 x 300 km
  square), at 100 m plus the same noise, with every row from y = 50 km
  raised by 900 m: a step of about 1000 m, the classic set-up for scarp
  retreat. No tectonic field.
- **GeoTIFF DEM (`geotiff`)**: the SRTM tile `data/srtm_36_04.tif` (elevation
  in lon/lat) is opened with `rioxarray`, reprojected to its estimated UTM
  zone (`xds.rio.estimate_utm_crs()`), clipped to x = 400 to 600 km and
  y = 4500 to 4750 km (UTM coordinates), and interpolated (cubic) onto a
  500 m grid. The mesh has the same 500 m resolution; its elevations range
  from 273 to 2365 m.
- **Variable resolution (`geotiff_refined`)**: the same 500 m DEM grid, with
  a `cellwidth` set from the elevation by the function `weightsElev`, so the
  high ground, where the slopes and the incision are strongest, is meshed
  more finely:

  | Elevation band | Target `cellwidth` |
  |---|---|
  | z < -1000 m or z > 0 m (default) | 5000 m |
  | -1000 m <= z <= 0 m | 1500 m |
  | 0 m < z <= 500 m | 1000 m |
  | 500 m < z <= 1000 m | 750 m |
  | 1000 m < z <= 1500 m | 500 m |
  | z > 1500 m | 250 m |

  The bands are applied in this order, each overriding the previous one, so
  every land elevation gets one of the last four widths.

## The output files

Each scenario folder holds:

| File | Content |
|---|---|
| `mesh2D.msh`, `mesh2D.jig`, `mesh2D-MESH.msh`, `mesh2D-HFUN.msh`, `mesh2D.log` | jigsaw geometry, configuration, mesh, sizing function and log |
| `mesh2D.nc`, `base2D.nc`, `graph.info` | the mesh converted to netCDF and to an MPAS/UGRID mesh (`base2D.nc` is the one used afterwards) |
| `planar.vtk` | the triangular mesh |
| `staticFieldsOnCells.vtp` | the Voronoi cells, to view as a wireframe in ParaView |
| `data.nc` or `data2.nc` | the fields interpolated onto the mesh |
| `init.vtk` | the mesh with the interpolated fields, for ParaView |
| `gospl_mesh.npz` (`escarpment2.npz` for the escarpment) | the goSPL input |

## Using a mesh in a goSPL input

Point the `domain: npdata` key of the input file at the `.npz` file (path
without the extension) and name its keys for the vertices, cells and
elevation. A field stored in the same file can drive a forcing, here the
uplift rate of `slope_tec` (m/yr):

```yaml
domain:
    npdata: ['slope_tec/gospl_mesh','v','c','z']
    flowdir: 1
    bc: 'oooo'        # north, east, south, west edges: o = open, f = fixed, w = wall

tectonics:
  - start: 0.
    end: 1.e5
    upsub: ['slope_tec/gospl_mesh','t']
```

This is the set-up of the `implicit_timestepping` example (which keeps its
copy of the mesh in `data/gospl_mesh.npz`). goSPL treats a mesh that has a
boundary (a global mesh has none) as a flat 2D model, and the `bc` key sets
the behaviour of its four edges.

## Running it

```bash
jupyter lab build_inputs.ipynb
```

Run the cells in order; each scenario is independent of the others after the
first cell (imports). There is nothing to run with `gospl` here: copy a
`gospl_mesh.npz` into another example, or point an input file at it as
above.

## Building your own mesh

- *Another DEM*: replace `data/srtm_36_04.tif` with your GeoTIFF (or a
  netCDF grid opened with `xarray`), and change the clipping box and the
  resolution `dx` of the regular grid.
- *Another resolution*: change `cellwidth` (uniform), or the bands of
  `weightsElev`. Any field of the grid can set the width (distance to a
  fault or a coast, slope, ...); the edge-length and cell-area printouts
  show the resolution you obtained.
- *More fields*: add them to the Dataset before `inter2UGRID` (for example a
  rainfall or an uplift map), then save them in the `.npz` file under a key of
  your choice and read them in the input file with `[file, key]`.
- *A rebuild*: `planarMesh` reuses the files already in the output folder;
  delete the folder after changing the grid or `cellwidth`.
