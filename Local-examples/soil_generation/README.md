# Soil production on a regional landscape (local)

A regional landscape of about 200 x 250 km, between 273 and 2,365 m above sea
level, evolves for 100 kyr with a soil cover. goSPL's soil model (`soil:`)
produces soil from the bedrock at a rate that decays with the soil
thickness, moves it downslope with a non-linear creep law, and lets the rivers
erode soil and bedrock with different erodibilities. The run also uses river
incision (stream power law, `spl:`), hillslope diffusion (`diffusion:`) and
flexural isostasy (`flexure:`) under uniform rainfall.

## What it shows

The numbers below are read from the 250 m grids of `results/` (written by
`view_Results.ipynb`).

- **A soil cover that adjusts to the relief**: the soil starts at 0.5 m
  everywhere. After 10 kyr its median thickness is 0.27 m, with 5 % of the
  grid below 1 mm and 5 % above 1.37 m; at 100 kyr the median is 0.26 m and
  the 95th percentile 1.43 m. About 5 % of the grid is bare (less than 1 cm of
  soil) at 100 kyr, where erosion removes it faster than it is
  produced.
- **A maximum soil thickness**: soil production stops at the depth where it
  has decayed to `bedrockConv` of its maximum, here `-ln(1e-8) x 1 m = 18.4 m`.
  A few cells (0.04 % of the grid at 100 kyr) reach that cap.
- **Erosion and deposition**: at 100 kyr the cumulative erosion reaches 285 m
  and the deposits 105 m; the mean change is 21.8 m of
  erosion.
- **Flexural rebound**: the flexural deflection (`flexIso`, positive up) is
  4 to 43 m at 100 kyr; the lowest point of the landscape rises from 273 to
  287 m.
- **Soil along a river**: `view_Results.ipynb` extracts the river network and
  long profile of one basin (basin 1179, picked at x = 420 km,
  y = 4,600 km), and `make_ex_soil.py` plots the soil thickness along its main
  stem and against elevation over the basin.

## The set-up, block by block (`input-river-soil.yml`)

- **`domain`**: the mesh `data/gospl_mesh.npz` (keys `v`, `c`, `z`), 228,573
  nodes in projected coordinates (x from 401 to 599.5 km, y from 4,501 to
  4,749.5 km), median edge 500 m (372 to 629 m). The initial elevation is 273
  to 2,365 m, all above sea level, so the marine processes play no role.
  `flowdir: 5` (multiple flow direction, up to 5 receivers), `bc: 'oooo'`
  (open edges).
- **`time`**: 0 to 100 kyr in 5 kyr steps, an output every 10 kyr (steps 0 to
  10).
- **`spl`**: bedrock erodibility `K: 4.e-6` (SI units, with drainage area in
  m²), `m: 0.5`, `d: 0`, and `G: 0` (no river deposition on land).
- **`soil`**:
  - `soilK: 8.e-6`: erodibility of the soil, twice that of the bedrock.
  - `maxProd: 50.e-6` m/yr (50 m/Myr): maximum soil production rate, on bare
    bedrock.
  - `depthProd: 1.0` m: the production decays exponentially with the soil
    thickness over this depth.
  - `roughnessL: 1.0`: roughness length scale.
  - `decayDepth: 0.7` m: decay depth of the soil transport in the non-linear
    creep law.
  - `bedrockConv: 1.e-8`: soil-to-bedrock conversion fraction; bedrock begins
    where production falls to this fraction of `maxProd`, which also sets the
    maximum soil thickness, `-ln(bedrockConv) x depthProd` = 18.4 m.
  - `uniform: 0.5` m: initial soil thickness everywhere.
  - `solver: 'ngmres'`, `pcType: 'hypre'`, `rtol: 1.e-5`, `atol: 1.e-5`: the
    non-linear solver of the soil-aware erosion (`ngmres` with the BoomerAMG
    preconditioner is goSPL's default; the tolerances are looser than the
    default of 1e-6).
  - `mode` is not set, so it is `lumped`: the soil layer holds both the soil
    produced by weathering and the deposited sediment.
- **`diffusion`**: `hillslopeKa: 1.0` m²/yr on land, with `hillslopenl: 2.5`,
  the slope exponent of the non-critical non-linear creep law of Wang et al.
  (2024). `hillslopeKm: 2.0` m²/yr, `nonlinKm: 120` m²/yr and
  `clinSlp: 5.e-4` act only under the sea, so not here.
- **`flexure`**: `method: 'fem'`, elastic thickness 5 km, densities 2300
  (load) and 3300 kg/m³ (mantle), Young's modulus 65 GPa, Poisson ratio 0.25,
  `Mirror` on the four edges.
- **`sea`**: sea level at 0 m.
- **`climate`**: uniform rainfall of 1 m/yr.
- **`output`**: written to `sim_river_soil/` (the soil thickness is the output
  field `soilH`).

## Running it

```bash
conda activate gospl
mpirun -np 4 gospl -i input-river-soil.yml
```

The mesh is shipped in `data/`; there is no set-up notebook.

Then open `view_Results.ipynb`. It rasterises the 11 output steps onto a
250 m grid (`results/surface<step>.nc`, with elevation, erosion and
deposition, soil thickness `soilH`, flexure, discharge, basins and chi), then
plots the elevation, the cumulative erosion and deposition and the soil
thickness at steps 2, 5, 7 and 10 (20, 50, 70 and 100 kyr), the drainage
basins at 100 kyr, and the network and long profile of basin 1179.

A single step can also be rasterised from a terminal:

```bash
gospl-grid --h5dir sim_river_soil/h5 --mesh data/gospl_mesh.npz:v:c --step 10 --spacing 250 --out results/surface10.nc
```

`make_ex_soil.py` draws the figure of the goSPL model-description paper
(`fig_ex_soil.pdf` and `.png`) from `sim_river_soil` at step 10: erosion and
deposition, soil thickness, the basins with the network of basin 1179, its
long profile with the soil thickness along the main stem, and the soil
thickness against elevation over the basin.

## Building your own experiment

- *Faster or slower weathering*: `maxProd` (m/yr) and `depthProd` (m). The
  maximum soil thickness is `-ln(bedrockConv) x depthProd`; set
  `bedrockConv: 0` for no cap.
- *Weathering that depends on temperature*: add `tempMap` (a map of the mean
  annual temperature, in degrees Celsius, `[file, key]`), with `activation`
  (J/mol) and `tempRef` (degrees Celsius).
- *A soil map instead of a uniform cover*: `soilMap: [file, key]` in place of
  `uniform`.
- *Keep the deposits out of the soil*: `mode: 'regolith'` keeps only the
  weathered regolith in the soil layer and puts the deposits in the
  stratigraphy; it needs a stratigraphic interval (`strat`) in the `time`
  block.
- *Solver*: the soil solve is usually the most expensive part of the run. If
  it struggles, try another `pcType` (`gamg`, `bjacobi`, `asm`) or `solver:
  'qn'` rather than loosening `rtol`: tolerances of 1e-4 can leave single
  under-resolved nodes that show as elevation spikes.

## References

- Wang et al. (2024), Geomorphology,
  https://www.sciencedirect.com/science/article/pii/S0169555X24001053 (the
  non-critical non-linear hillslope model used with `hillslopenl`).
