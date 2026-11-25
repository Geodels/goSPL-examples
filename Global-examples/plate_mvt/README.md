# Plate motion on a global mesh (global)

A 1 Myr global run starting from a 250 Ma paleogeography, in which the
surface is carried by the plate velocities while rivers, hillslopes and the
marine transport shape it. The horizontal motion uses goSPL's plate
interpolation (`advect: 'interp'`) with remaps of about one mesh spacing
(`interp_remap`), and the vertical motion is derived from two successive
paleo-elevation maps (250 and 251 Ma). The run combines the stream-power law
(`spl:`), linear hillslope diffusion and non-linear marine diffusion
(`diffusion:`), a paleo-rainfall map (`climate:`) and horizontal and vertical
tectonics (`tectonics:`). `interp_remap` is not in a goSPL release yet: run
this example with the goSPL `dev` branch.

## What it shows

- **Drainage carried by the plates.** With `interp_remap: 1` the 1 Myr
  tectonic event is applied in remaps of about one mesh spacing, here every
  50 kyr step (the fastest plates move about 0.8 cell per step), with
  barycentric weights, instead of one 1 Myr jump at the end of the event. The
  rivers then move with the plates.
- **The cost of many remaps.** Each remap is a linear interpolation, so the
  20 remaps smooth the sharpest relief a little: after 1 Myr the mean land
  elevation is the same as with a single jump (637 m), but the 99th percentile
  is about 10 % lower (2125 m against 2373 m). A larger `interp_remap` (fewer,
  longer remaps) smooths less but carries the drainage less continuously.
- **Global maps of the result**: elevation at 500 kyr and 1 Myr with the
  coastlines of both steps (the coast moves with the plates), cumulative
  erosion and deposition after 1 Myr, and the water and sediment discharge
  at the river mouths. `view_Results.ipynb` finds 4231, 3987 and 4522 basin
  outlets at 100 kyr, 500 kyr and 1 Myr (output steps 1, 5 and 10).

## The set-up, block by block (`input-global_plates.yml`)

- **`domain`**: the global mesh `vars_30/mesh.npz` (keys `v`, `c`, `z`),
  built for a target cell width of 30 km (658,605 nodes; the build notebook
  lists edges of 6.8 to 26.4 km, mean 17.2 km, for this width), carrying the
  250 Ma elevation. `flowdir: 3` sets the flow-direction method (an integer
  from 1, single flow direction, to 6, multiple flow directions).
  `advect: 'interp'` moves the surface by plate interpolation (the other
  choices are the Eulerian schemes `upwind`, `iioe1` and `iioe2`), and
  `interp_remap: 1` applies the motion in remaps of about one mesh spacing.
- **`time`**: from 250 Myr to 251 Myr (the times of the two paleo maps), an
  output every 100 kyr, internal step of 50 kyr.
- **`spl`**: `K = 1e-6`, precipitation exponent `d = 0.42`, `m = 0.5`.
- **`diffusion`**: hillslope diffusivity 0.5 m2/yr on land (`hillslopeKa`)
  and 2 m2/yr in the sea (`hillslopeKm`); non-linear diffusivity of 100 m2/yr
  for the river sediment reaching the sea (`nonlinKm`); maximum clinoform
  slope `clinSlp: 1e-5`, which caps the top of the marine deposits with the
  distance to the shore. The marine diffusion uses the lagged-diffusivity
  solver (`marineSolver: picard`) with 10 backward-Euler sub-steps per time
  step (`picardSub`) and 2 Picard iterations per sub-step (`picardIts`).
- **`sea`**: sea level fixed at 0 m.
- **`tectonics`**: one event over the whole run. `hdisp` reads the plate
  velocity of every node as a 3-D Cartesian vector (key `vxyz` of
  `vars_30/forcing250.npz`, m/yr); `upsub` reads the vertical rate `t`
  (m/yr). The commented `zfit` line would read a next-step elevation `nz`
  (not saved by default, see below).
- **`climate`**: the 250 Ma precipitation map, key `r` of
  `vars_30/forcing250.npz` (m/yr).
- **`output`**: the output folder (`silico_mvt`).

## Building the inputs (`build_inputs.ipynb`)

The notebook makes the mesh and the forcing from two regular paleo grids,
`data/250.nc` and `data/251.nc` (elevation `h` in m, plate velocity
components `vx`, `vy`, `vz` in m/yr, rainfall `rain` in m/yr). It uses jigsaw
and the MPAS mesh converter through `shared_scripts/umeshFcts.py` (imported
as `scripts.umeshFcts`). The steps:

1. **The global mesh**: `buildGlobalMeshSimple(widthCell, input_path)` builds
   a quasi-uniform spherical Voronoi mesh with `widthCell = 30` km into
   `input_30/` (`mesh_30km.nc` and the jigsaw files). An existing mesh is not
   rebuilt. The notebook's table of resolutions:

   | `widthCell` (km) | edge min (km) | edge max (km) | edge mean (km) | nodes |
   |---|---|---|---|---|
   | 5 | 1.1 | 4.5 | 2.8 | 23,632,811 |
   | 8 | 1.8 | 7.2 | 4.6 | 9,236,387 |
   | 10 | 2.2 | 8.9 | 5.7 | 5,912,778 |
   | 15 | 3.3 | 13.1 | 8.6 | 2,629,742 |
   | 20 | 4.5 | 18 | 11.5 | 1,480,168 |
   | 25 | 5.6 | 22.4 | 14.4 | 947,701 |
   | 30 | 6.8 | 26.4 | 17.2 | 658,525 |
   | 35 | 8 | 30.5 | 20.1 | 484,009 |

2. **The fields on the mesh**: `inter2UGRID` interpolates `h`, `vx`, `vy`,
   `vz` and `rain` of `data/250.nc` bilinearly onto the mesh cells
   (`vars_30/step_250.nc`), and writes `input_30/init.vtk` to check the mesh
   and the fields in ParaView (optionally also the Voronoi cells as a `.vtp`
   file, `saveVoro = True`).
3. **The vertical forcing**: the 251 Ma elevation is added as `next_h`
   (`vars_30/step_up_250.nc`). `get_Tectonic` moves the nodes along the plate
   velocities for 1 Myr, interpolates the 250 Ma elevation at the moved
   positions (inverse-distance weighting), and takes the difference with the
   251 Ma elevation; the difference is moved back to the mesh nodes and
   divided by 1 Myr to give the vertical rate `tec` (m/yr). A dynamic
   topography field can be added to it with `dkey`.
4. **The goSPL files**: `vars_30/mesh.npz` (`v`, `c`, `z`) and
   `vars_30/forcing250.npz` (`vxyz`, `t`, `r`). Uncomment `nz` to also save
   the 251 Ma elevation, which goSPL can use through the `zfit` key to pull
   selected regions towards the next paleo-elevation.

## Running it

```bash
# 1. open and run build_inputs.ipynb (mesh and forcing in input_30/ and vars_30/)
# 2. the run (8 MPI ranks)
mpirun -np 8 gospl -i input-global_plates.yml
```

Then open `view_Results.ipynb`. It grids output steps 1, 5 and 10 onto a
0.1 degree lon/lat grid with `gospl.analyse.gridexport`
(`results/surface<step>.nc`), draws global PyGMT maps of the elevation (with
the 0 m coastline of steps 5 and 10) and of the cumulative erosion and
deposition, and extracts the outlet of every drainage basin with
`gospl.analyse.catchment.catchment_flux` (`flowsed/flow<step>.csv` and
`flowsed/sed<step>.csv`, columns `basin, lon, lat, val`, `val` in m3/yr). It
maps the 200 largest water and sediment outlets at 1 Myr. Basin numbers are
labels of each gridded step, not the same catchment from one step to the
next: follow a river mouth through time by its location. From a terminal,
`gospl-catchment -i index.csv -o flowsed` does the outlet extraction, where
`index.csv` lists one `surface*.nc` per time.

## Building your own experiment

- *Another resolution*: change `widthCell` in `build_inputs.ipynb` (table
  above); the node count, and so the run time, grows quickly as the cells get
  smaller.
- *Another time interval*: replace `data/250.nc` and `data/251.nc` with two
  other paleo grids holding `h`, `vx`, `vy`, `vz` and `rain`, and adjust `dt`
  in `get_Tectonic` and the `time` and `tectonics` blocks.
- *Matching the next paleo-elevation*: save `nz` in `forcing250.npz` and
  uncomment `zfit` in the `tectonics` block.
- *Remap length*: a larger `interp_remap` smooths less but moves the drainage
  in larger jumps; without the key the 1 Myr motion is applied once, at the
  end of the event.
