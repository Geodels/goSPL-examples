# Global erosion, deposition and flexural isostasy over 1 Myr (global)

A 1 Myr global run from 251 Ma on a uniform 40 km spherical mesh. goSPL
erodes the continents with the stream power law including continental
deposition (`spl` with `G`, transport-limited), diffuses the hillslopes,
spreads the river sediment on the margins with the non-linear marine
diffusion, and computes the flexural isostatic response of the lithosphere
to the erosion and deposition with a spherical-harmonic solve and a map of
elastic thickness. The paleo-elevation, the rainfall and the elastic
thickness come from a 0.25 degree grid of the Earth at 251 Ma. The
`gravity_currents_global` example reuses its inputs.

## What it shows

- **A global model with flexure**: the `flexure` block with
  `method: 'global'`, and a spatially variable elastic thickness given as a
  map (`temap`).
- **Continental erosion and deposition**: rivers that deposit part of their
  load on land (`G`) as well as at the coast, and the marine deposits that
  build up off the river mouths.
- **The flux to the oceans**: the water and sediment discharge at the mouth
  of every drainage basin, mapped for the 200 largest rivers.
- **The isostatic response**: a map of the cumulative flexural deflection
  after 1 Myr, the response of a lithosphere of variable elastic thickness to
  the erosion and the sediment load.

## The set-up, block by block (`input-mardep.yml`)

- **`domain`**: the mesh `vars_40/mesh.npz` (keys `v`, `c`, `z`): 370,830
  nodes, edges from 31 to 51 km, 40 km on average. Five flow directions
  (`flowdir: 5`, on a scale from 1, single flow direction, to 6, multiple
  flow directions). Marine deposition is on by default.
- **`time`**: from 251 Ma to 252 Ma of model time (`start: 251.e6`,
  `end: 252.e6`, in years) in 50 kyr steps (`dt: 5.e4`), an output every
  100 kyr (`tout: 1.e5`), so 11 outputs (0 to 10).
- **`spl`**: erodibility `K: 3.e-6`, precipitation exponent `d: 0.42`,
  drainage-area exponent `m: 0.5` (slope exponent 1), and continental
  deposition coefficient `G: 2` (m/yr, compared with the rainfall: the
  fraction of the incoming load a cell deposits is `G` divided by its
  rainfall).
- **`diffusion`** (a commented copy above it explains each key):
  - `hillslopeKa: 0.5` and `hillslopeKm: 2.0` m²/yr: linear diffusivities on
    land and in the sea;
  - `nonlinKm: 1.e4` m²/yr: non-linear diffusivity of the fresh river
    sediment entering the sea, the diffusion the marine solver integrates;
  - `Gmar: 0.5`: dimensionless marine deposition coefficient;
  - `clinSlp: 1.e-5`: maximum clinoform slope, which sets the top of the
    marine deposits with the distance to the shore;
  - `marineSolver: picard` (the default solver) with `picardSub: 10`
    backward-Euler sub-steps per goSPL step (the default is 20) and
    `picardIts: 2` Picard iterations per sub-step.
- **`flexure`**: the spherical-harmonic flexure (`method: 'global'`, solved
  on the root rank on a Driscoll-Healy grid of `res_deg: 0.25` degree), with
  load density `rhoc: 2300.0` kg/m³, asthenosphere density
  `rhoa: 3300.0` kg/m³, Young's modulus `young: 65e9` Pa and Poisson ratio
  `nu: 0.25`, applied every step (`interval: 1`). The uniform elastic
  thickness `thick: 30.e3` m is replaced by the map below.
- **`temap`**: from 251 Ma, the elastic thickness map
  `vars_40/forcing251.npz` (key `te`, m; from 5 to 157 km on the mesh,
  32 km on average). With a map the global solve iterates on the varying
  thickness at each step.
- **`sea`**: sea level at 0 m.
- **`climate`**: the 251 Ma rainfall map `vars_40/forcing251.npz` (key `r`,
  from 0.001 to 5.2 m/yr, 1.07 m/yr on average), constant through the run.
- **`output`**: written to `silico_mardep/`. Besides the usual fields, the
  outputs hold `flexIso`, the cumulative flexural deflection (m).

## Building the inputs

`build_inputs.ipynb` builds the mesh and the forcing file from the grid
`data/251Ma.nc` (0.25 degree, 721 x 1441; elevation `h` in m, rainfall `rain`
in m/yr and elastic thickness `te` in m). It uses the mesh helpers of
`shared_scripts/umeshFcts.py` (imported through `scripts/`), which need
UXarray, JIGSAW, MPAS-Tools, xESMF, pyinterp and VTK.

1. It builds a uniform spherical mesh with JIGSAW (`widthCell = 40` km) in
   `input_40/` (`mesh_40km.nc`); an existing mesh is not rebuilt. The
   notebook's table gives the node counts and edge lengths of other widths,
   from 23.6 million nodes at 5 km to 484,009 at 35 km.
2. It interpolates the three fields on the mesh (`vars_40/step_251.nc`).
3. It writes `vars_40/mesh.npz` (node coordinates `v` in m, triangles `c`,
   elevation `z` in m) and `vars_40/forcing251.npz` (`te` and `r`).

Optional cells export the Voronoi mesh (`saveVoro`) and a `init.vtk` file of
the mesh and its fields (`checkMesh`) for ParaView; both are off by default.

The bathymetry is not cleaned here: the 251 Ma grid goes down to about
-24 km, and the mesh down to -23,499 m. The `continental_flux` example
cleans its 250 Ma bathymetry before saving the mesh with
`scripts/clean_bathymetry.py`; the same script can be run on this mesh (see
below).

## Running it

```bash
# 1. run all the cells of build_inputs.ipynb (writes vars_40/mesh.npz and forcing251.npz)
# 2. run the model
mpirun -np 6 gospl -i input-mardep.yml
```

Change `-np` to the number of cores you want to use (add `-v` for the
progress of each step). Then open `view_Results.ipynb`, which needs PyGMT
and joblib:

1. it grids the outputs 1, 5 and 10 on a regular 0.1 degree
   longitude-latitude grid (`gospl.analyse.gridexport.grid_export` and
   `to_netcdf`), written to `results/surface{1,5,10}.nc` with the fields
   `elev`, `erodep`, `FA` (water discharge), `sedLoad` (sediment load),
   `flexIso`, `basin`, `chi` and `drainage_area`;
2. it maps the elevation after 1 Myr (Hammer projection);
3. it maps the erosion and deposition over a grey topography: the land cells
   whose cumulative erosion/deposition is below 5 m (mostly eroded) and the
   cells with more than 100 m of deposition, on a scale of -3000 to 3000 m;
4. it maps the flexural isostasy where it exceeds 200 m up or down, on a
   scale of -2000 to 2000 m;
5. it extracts, for every drainage basin of more than 10 grid cells, the
   cell of maximum water discharge and of maximum sediment load, its river
   mouth (`gospl.analyse.catchment.catchment_flux`), written to
   `flowsed/flow{1,5,10}.csv` and `flowsed/sed{1,5,10}.csv` (columns `basin`,
   `lon`, `lat`, `val` in m³/yr);
6. it maps the 200 largest water and sediment fluxes at step 10, sized and
   coloured by their logarithm.

Basin numbers are per-step labels: follow a river through time by the
location of its mouth, not by its number.

After the notebook, `make_ex_global.py` draws a 2 x 2 figure of step 10
(elevation, erosion/deposition, water and sediment discharge at the river
mouths) from `results/surface10.nc` and `flowsed/{flow,sed}10.csv`, and
saves it as `fig_ex_global.pdf` and `fig_ex_global.png` (the figure of the
goSPL model-description paper).

## Building your own experiment

- *Another resolution*: change `widthCell` in the notebook. Finer meshes
  need more marine sub-steps: scale `picardSub` by about
  `(dx_old / dx_new)²`.
- *A uniform elastic thickness*: remove the `temap` block; `thick` is then
  used everywhere and the global solve is a single spectral pass.
- *A cheaper flexure*: a coarser Driscoll-Healy grid (`res_deg: 0.5` or
  `1.0`) changes the long-wavelength deflection little and costs much less;
  `interval` larger than 1 applies the flexure every few steps.
- *Detachment-limited rivers*: remove `G` (all the eroded material then
  leaves the rivers for the sea or the inland depressions).
- *A cleaner seafloor*: copy `../continental_flux/scripts/clean_bathymetry.py`
  and run `python clean_bathymetry.py vars_40/mesh.npz` (it keeps the
  original as `vars_40/mesh_raw.npz`).
