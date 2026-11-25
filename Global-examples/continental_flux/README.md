# Continental erosion and sediment flux to the oceans (global)

A 1 Myr global run from 250 Ma (Pangea) on a mesh refined along the
coastlines: about 25 km spacing on the continents and shelves, up to 80 km in
the deep ocean. goSPL erodes the continents with the stream power law
(`spl`), diffuses the hillslopes, carries the river sediment to the sea and
spreads it on the margins with the non-linear marine diffusion. The
paleo-elevation and the rainfall come from a 0.25 degree reconstruction grid
of the Earth at 250 Ma. The run is then post-processed into the water and
sediment discharge at every river mouth, the global maps of elevation and
erosion/deposition, and the hypsometry. Its mesh and rainfall are reused by
the `continental_geochem` and `carbonate_global` examples.

## What it shows

- **How to build a global goSPL mesh** from regular longitude-latitude grids:
  a uniform spherical mesh, and a mesh whose resolution follows the distance
  to the coast (JIGSAW through UXarray and MPAS-Tools).
- **How to clean a paleo-bathymetry** before a run: the 250 Ma grid goes
  down to -25,368 m (holes and spikes far deeper than any ocean, artefacts of
  the reconstruction). goSPL's marine routing ends in such depressions and
  fills some of them with kilometres of sediment, so the mesh is cleaned
  before it is saved (see below).
- **Continental erosion and the flux to the oceans**: the cumulative
  erosion/deposition after 1 Myr, and the water and sediment discharge at the
  mouth of every drainage basin, mapped for the 200 largest rivers.
- **The hypsometry** of the simulated surface (area per elevation bin and the
  cumulative hypsographic curve) at outputs 1 (100 kyr) and 10 (1 Myr).

## The set-up, block by block (`input-cont.yml`)

- **`domain`**: the refined mesh `vars_25_80/mesh.npz` (keys `v`, `c`, `z`:
  473,582 nodes, edges from 20 to 102 km, 33 km on average, cleaned
  bathymetry). Two flow directions (`flowdir: 2`, on a scale from 1, single
  flow direction, to 6, multiple flow directions), marine
  deposition on (`seadepo: True`).
- **`time`**: from 250 Ma to 251 Ma of model time (`start: 250.e6`,
  `end: 251.e6`, in years) in 50 kyr steps (`dt: 5.e4`), an output every
  100 kyr (`tout: 1.e5`), so 11 outputs (0 to 10).
- **`spl`**: erodibility `K: 5.e-6`, precipitation exponent `d: 0.42`,
  drainage-area exponent `m: 0.5` (slope exponent 1). No `G`: the rivers are
  detachment-limited and their load goes to the sea or to the inland
  depressions.
- **`diffusion`**:
  - `hillslopeKa: 0.5` and `hillslopeKm: 2.0` m²/yr: linear diffusivities on
    land and in the sea;
  - `nonlinKm: 5.e4` m²/yr: non-linear diffusivity of the fresh river
    sediment entering the sea, and `Gmar: 1.0` the dimensionless marine
    deposition coefficient;
  - `clinSlp: 1.e-5`: maximum clinoform slope, which sets the top of the
    marine deposits with the distance to the shore;
  - `marineSolver: picard` (the default solver) with `picardSub: 10`
    backward-Euler sub-steps per goSPL step (the default is 20) and
    `picardIts: 2` Picard iterations per sub-step;
  - `nlPitK: 5.e3` m²/yr: diffusivity used inside the large inland
    depressions, and `pitInletBias: 0.4`: the fraction of a depression's
    deposit concentrated at its inlets (default 0.1), the rest filling it
    from the bottom.
- **`sea`**: sea level at 0 m.
- **`climate`**: the 250 Ma rainfall map `vars_25_80/rain250.npz` (key `r`,
  m/yr, from 0.0007 to 5.8 m/yr), constant through the run.
- **`output`**: written to `silico_cont/`.

## Building the inputs

`build_inputs.ipynb` builds the meshes and the forcing files from the two
reconstruction grids `data/250.nc` and `data/251.nc` (0.25 degree,
721 x 1441; elevation `h` in m, rainfall `rain` in m/yr and the plate
velocities `vx`, `vy`, `vz` in m/yr). It uses the mesh helpers of
`shared_scripts/umeshFcts.py` (imported through `scripts/`), which need
UXarray, JIGSAW, MPAS-Tools, xESMF, pyinterp and VTK; the checks also use
meshio and cartopy. It has two parts.

1. **A uniform mesh** (`widthCell = 30` km, about 660,000 nodes and 17 km
   mean edge), in `input_30/`. The elevation, rainfall and plate velocities
   are interpolated on it, and the vertical tectonic rate between 250 and
   251 Ma is derived from the two elevation grids after back-tracking each
   node with the plate velocities (`get_Tectonic`, over 1 Myr). It writes
   `vars_30/mesh.npz` and `vars_30/forcing250.npz` (keys `vxyz` plate
   velocities, `t` vertical tectonic rate, `r` rainfall, `nz` the 251 Ma
   elevation, usable with `zfit`), and an `init.vtk` file and a Voronoi
   `.vtp` file to check the mesh in ParaView. This part shows how to prepare a
   run with plate motion; `input-cont.yml` does not use it.
2. **A depth-dependent mesh**, the one of the run, in `input_25_80/`. A
   50 km background mesh gives the distance of every point to the -50 m
   contour; the requested cell width grows linearly with that distance, from
   25 km on the continents and near the coast to 80 km beyond 2500 km into
   the ocean (a map of it is plotted). JIGSAW then builds the refined mesh
   `mesh_refine.nc`, and the notebook writes `vars_25_80/mesh.npz` and
   `vars_25_80/rain250.npz` (key `r`).

**The bathymetry is cleaned before each mesh is saved**, with
`scripts/clean_bathymetry.py`:

- every node below -3000 m that is more than 1500 m deeper than the median of
  its two-ring neighbourhood is set to that median, repeated until none is
  left (real basins and slopes are wider than two rings and are kept);
- the seafloor is then clipped at -6500 m, the depth of the deepest abyssal
  plains (trenches are not resolved at 20 to 100 km spacing);
- land is never changed.

On the refined mesh, 17,739 nodes (4 % of the mesh) are clipped at -6500 m
and the closed depressions deeper than 6 km drop from 1,181 (down to -25 km)
to 294 shallow dips on the -6500 m floor. The script can also clean an
existing mesh in place, keeping the original as `mesh_raw.npz`:

```bash
python scripts/clean_bathymetry.py vars_25_80/mesh.npz
```

Its options `--floor`, `--spike` and `--spike-below` change the three
thresholds (defaults -6500, 1500 and -3000 m).

## Running it

```bash
# 1. run all the cells of build_inputs.ipynb (writes vars_25_80/mesh.npz and rain250.npz)
# 2. run the model
mpirun -np 8 gospl -i input-cont.yml
```

Change `-np` to the number of cores you want to use (add `-v` for the
progress of each step). Then open `view_Results.ipynb`, which needs PyGMT
and joblib:

1. it grids the outputs 1, 5 and 10 on a regular 0.1 degree
   longitude-latitude grid (`gospl.analyse.gridexport.grid_export` and
   `to_netcdf`), written to `results/surface{1,5,10}.nc` with the fields
   `elev`, `erodep`, `FA` (water discharge), `sedLoad` (sediment load),
   `basin`, `chi` and `drainage_area`;
2. it maps the elevation after 1 Myr (Robinson and Hammer projections) and
   the cumulative erosion/deposition;
3. it extracts, for every drainage basin of more than 10 grid cells, the
   cell of maximum water discharge and of maximum sediment load, its river
   mouth (`gospl.analyse.catchment.catchment_flux`), written to
   `flowsed/flow{1,5,10}.csv` and `flowsed/sed{1,5,10}.csv` (columns `basin`,
   `lon`, `lat`, `val` in m³/yr);
4. it maps the 200 largest water and sediment fluxes at step 10, sized and
   coloured by their logarithm;
5. it draws the hypsometry of steps 1 and 10.

The same per-basin extraction runs from a terminal with
`gospl-catchment -i inputSedFlow.csv -o flowsed` (`inputSedFlow.csv` lists
the `results/surface*.nc` grid of each step). Basin numbers are per-step
labels: follow a river through time by the location of its mouth, not by its
number.

## Building your own experiment

- *Another time slice*: replace `data/250.nc` by your own grid with the same
  variables, and change `start` and `end` in `input-cont.yml`.
- *Another resolution*: change `highres`, `lowres` and `reso_contour` in the
  second part of the notebook (or `widthCell` for a uniform mesh; the
  notebook gives a table of the node counts from 5 to 35 km). Finer meshes
  need more marine sub-steps: scale `picardSub` by about
  `(dx_old / dx_new)²`.
- *Plate motion or uplift*: the uniform mesh's `forcing250.npz` holds the
  plate velocities and the vertical tectonic rate; they go in a `tectonics`
  block of the input file (see the `plate_mvt` example).
- *Your own paleo-bathymetry*: run `scripts/clean_bathymetry.py` on it, or
  check first how deep its seafloor goes.
