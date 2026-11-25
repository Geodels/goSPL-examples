# Escarpment retreat (local)

A plateau 1000 m high ends in a steep scarp above a low coastal plain, as on a
rifted (passive) margin. Rivers cut into the scarp, hillslopes creep, and the
lithosphere flexes as the margin is unloaded, so the scarp wears back inland
over 50 Myr. The example uses goSPL's stream power law (`spl:`), linear
hillslope diffusion (`diffusion:`), flexural isostasy on a flat mesh
(`flexure: method: 'fem'`) and, in a second run, orographic rain
(`orography:`). It runs with a released goSPL.

## What it shows

- **The retreat of the scarp**: the notebook collapses each output (every
  1 Myr) to cross-escarpment profiles (mean, maximum and minimum over x) and
  tracks the position and elevation of the crest through time; the slope of
  that track is the retreat rate.
- **The drainage of the margin**: maps of elevation and erosion/deposition rate
  at 10, 25, 35 and 50 Myr, the drainage basins and the chi values of the last
  output, and the long profiles of the main river and tributaries of one basin
  (picked at x = 190 km, y = 150 km) at 40 and 50 Myr.
- **The flexural response to the unloading**: the change of the flexural
  deflection (`flexIso`) between consecutive outputs along the profile, in
  m/Myr, i.e. the isostatic uplift rate as the scarp erodes.
- **The post-processing toolkit**: every product is made with goSPL's
  `gospl.analyse.gridexport` (rasterising the outputs to a regular grid,
  CF-NetCDF export, D8 basins, chi, river long profiles); the notebook also
  gives the equivalent `gospl-grid` terminal command.

## The experiments

| Input | Rain | `K` | Elastic thickness | Boundaries (N, E, S, W) | Output folder |
|---|---|---|---|---|---|
| `input-escarpment.yml` | uniform, 1 m/yr | 4e-6 | 30 km | open on all edges (`oooo`) | `escarpment` |
| `input-escarpment-oro.yml` | orographic, wind from the south | 2e-6 | 20 km | south edge fixed (`oofo`) | `escarpment-oro` |

`view_Results.ipynb` analyses `input-escarpment.yml`; the cells that grid the
orographic run (output name `oro`) are in the notebook but commented out.

## The set-up, block by block (`input-escarpment.yml`)

- **`domain`**: a 300 x 300 km square on a triangular mesh of about 1 km
  spacing (104,449 nodes, `data/escarpment.npz`, keys `v`, `c`, `z`). The
  initial surface is a coastal plain at 100 m for y below ~49 km and a plateau
  at 1000 m north of it, the scarp between the two lying within one row of
  cells, with a few centimetres of random noise. Single flow direction
  (`flowdir: 1`), no marine deposition (`seadepo: False`), all four edges open
  (`bc: 'oooo'`).
- **`time`**: 0 to 50 Myr in steps of 100 kyr (`dt: 1.e5`), an output every
  1 Myr (`tout: 1.e6`), so 51 outputs.
- **`spl`**: erosion rate `E = K (P^d A)^m S^n` with `K: 4.e-6`, `m: 0.4` and
  no rain dependence (`d: 0.`).
- **`diffusion`**: land hillslope diffusivity `hillslopeKa: 0.01` m²/yr,
  marine diffusivity `hillslopeKm: 1.0` m²/yr.
- **`flexure`**: the flat finite-volume flexure solver (`method: 'fem'`) with
  an elastic thickness of 30 km (`thick: 30.e3`), load (crust) density
  2300 kg/m³, asthenosphere density 3300 kg/m³, Young's modulus 65 GPa and
  Poisson ratio 0.25. Edge conditions: `0Slope0Shear` in the north (plateau
  side), `0Displacement0Slope` in the south (plain side), `Mirror` on the
  east and west edges.
- **`sea`**: sea level at -10 m (`position: -10.`), below the whole initial
  surface.
- **`climate`**: uniform rain of 1 m/yr from the start.
- **`output`**: the folder `escarpment`.

`input-escarpment-oro.yml` replaces the `climate` block by an `orography`
block that computes the rain from the topography and a wind blowing from the
south, across the plain towards the scarp:

```yaml
orography:
    wind_speed: 5.0            # m/s
    wind_dir: 180              # direction the wind comes from (0 = N, 90 = E, 180 = S)
    env_lapse_rate: -4         # environmental lapse rate (degC/km)
    moist_lapse_rate: -7       # moist adiabatic lapse rate (degC/km)
    ref_density: 7.4e-3        # reference saturation water-vapour density (kg/m3)
    conv_time: 1000.           # cloud water to hydrometeor conversion time (s)
    fall_time: 1000.           # hydrometeor fallout time (s)
    precip_base: 2.0           # uniform background precipitation (mm/h)
    precip_min: 0.01           # precipitation floor (mm/h)
    rainfall_frequency: 1      # 1-hour storms per day
```

It also halves the erodibility (`K: 2.e-6`), uses a thinner plate
(`thick: 20.e3`) and fixes the southern edge (`bc: 'oofo'`).

## Running it

```bash
conda activate gospl
mpirun -np 5 gospl -i input-escarpment.yml        # 5 MPI ranks, adjust to your machine
mpirun -np 5 gospl -i input-escarpment-oro.yml    # optional: the orographic-rain run
```

Then open `view_Results.ipynb`. Its first code cells rasterise the 51 outputs
of `escarpment/h5` to a 500 m grid and write them to
`results/surface<step>.nc` (in parallel with `joblib`, 8 workers); the rest of
the notebook reads these NetCDF files. The same grid can be made from a
terminal:

```bash
for s in $(seq 0 50); do
  gospl-grid --h5dir escarpment/h5 --mesh data/escarpment.npz:v:c \
      --step $s --spacing 500 --out results/surface$s.nc
done
```

To look at the orographic run, uncomment the cells that use
`h5dir = "escarpment-oro/h5"` and `out_name = "oro"`.

`make_ex_escarpment.py` rebuilds the 2 x 2 figure of the example used in the
goSPL model-description paper (final topography, erosion/deposition rate,
cross-escarpment profiles with the track of the scarp face at 500 m, and the
flexural deflection) from the existing outputs, and prints the retreat of the
scarp face from 0 to 50 Myr. It does not run the model.

## Building your own experiment

- *A faster or slower retreat*: the erodibility `K` and the exponent `m` of
  `spl`, and the rain (`climate: uniform`, or the `orography` block).
- *Rounder or sharper scarp edges*: the land diffusivity `hillslopeKa`.
- *A stiffer or softer margin*: the elastic thickness `flexure: thick` (the
  two runs use 30 and 20 km); the edge conditions `bcN`, `bcE`, `bcS`, `bcW`.
- *Another wind*: `wind_dir` (the direction the wind comes from, 0 = north)
  and `wind_speed` in `input-escarpment-oro.yml`; `precip_base` sets the
  background rain.
- *Another margin*: replace `data/escarpment.npz` by your own mesh (keys `v`
  for the vertices, `c` for the cells, `z` for the elevation), for example a
  higher plateau, a wider plain or an irregular scarp.
