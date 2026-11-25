# Glacial erosion on a mountain range (local)

A 100 kyr run of goSPL's diagnostic glacial model on a mountain range. Each
step, the surface mass balance above the equilibrium-line altitude (ELA) is
routed downhill as an ice discharge, which gives an ice thickness (Bahr
scaling) and a basal sliding velocity (Glen sliding law); the sliding ice
abrades its bed and the valley walls, and the abraded rock is carried as till
and deposited as moraine where the ice melts out. Below the ice, rivers erode
by the stream power law (`spl`), the glacial meltwater joins them, and
hillslopes creep by non-linear (critical-slope) diffusion. There is no
ice-dynamics time integration: the ice is diagnosed from the topography and
the climate at every step, which keeps the model cheap and stable over
goSPL's long time steps.

## What it shows

- **Glaciers above the ELA**: ice accumulates on the cells above 1500 m and
  is routed down the valleys, no lower than the 1200 m terminus. After
  100 kyr the thickest ice is 113 m.
- **Sliding drives the erosion**: the basal sliding velocity reaches 60 m/yr
  under the thickest, steepest ice, and the vertical abrasion rate follows it
  (`E_g = Kg * u_b`).
- **Valleys widen towards a U-shape**: with the lateral erosion coefficient
  `Kl` set, the walls next to fast ice are abraded too. The cross-valley
  transect of the notebook compares the initial and final surfaces under the
  diagnosed ice.
- **Erosion and deposition**: the cumulative erosion/deposition map shows
  where the ice and the rivers removed rock and where the till and the river
  sediment were deposited; after 100 kyr the change stays within about 50 m
  on 98 % of the cells.

## The set-up, block by block (`input-glacier.yml`)

- **`domain`**: the mesh `data/gospl_mesh.npz` (keys `v`, `c`, `z`), a
  198.5 x 248.5 km flat domain of 228,573 nodes at about 500 m spacing, with
  elevations from 273 to 2365 m. Multiple flow directions for the water
  (`flowdir: 6`), open edges on all four sides (`bc: 'oooo'`) and marine
  deposition on (`seadepo: True`, unused here: the sea is below the domain).
- **`time`**: 0 to 100 kyr (`end: 1.e5`) in 5 kyr steps (`dt: 5.e3`), an
  output every 10 kyr (`tout: 1.e4`), so 11 outputs (0 to 10).
- **`spl`**: river erodibility `K: 2.0e-6`, precipitation exponent
  `d: 0.42`, drainage-area exponent `m: 0.5` (slope exponent 1). No `G`: the
  rivers are detachment-limited.
- **`ice`**: the glacial model.
  - `hterm: 1200.0` m, `hela: 1500.0` m, `hice: 1800.0` m: the glacier
    terminus (no ice below it), the ELA (accumulation above, ablation below)
    and the ice-cap altitude (the full precipitation becomes ice above it;
    between the ELA and `hice` the fraction ramps from 0 to 1). The commented
    `evol: 'data/ice_evol.csv'` replaces these three constants by a time
    series (columns time, `hterm`, `hela`, `hice`; the file given goes from
    0 to 200 kyr), interpolated linearly at every step.
  - `icedir: 5`: number of flow directions used to route the ice discharge.
  - `eheight: 0.5` and `fwidth: 1.5`: the Bahr factors of the ice thickness
    `H = eheight * fwidth * Q^0.3`. Only their product matters.
  - `melt: 5.`: weight of the ablation below the ELA in the routed net
    balance (0 accumulation only, 1 the true net balance, more than 1 shorter
    and thinner tongues). It changes the ice extent only.
  - `slide: 0.3` and `glen: 3.0`: the sliding coefficient and the Glen
    exponent of the basal sliding law (tens of m/yr of sliding here).
  - `accum_factor: 0.9` and `accum_max: 2.0` m ice/yr: 90 % of the
    precipitation becomes ice above the ELA, and the accumulation rate is
    capped at 2 m/yr.
  - `abrasion`: vertical abrasion `Kg: 3.0e-5` with velocity exponent
    `l: 1.0` (it deepens the troughs), and lateral wall erosion
    `Kl: 3.0e-5` with exponent `lat_l: 1.0` (it widens them). Keep `Kl` of
    the same order as `Kg`: a much larger `Kl` makes the walls retreat far
    faster than the floor incises.
  - No `till` block: the defaults apply (till on, routed per catchment to the
    ice margin), and so does the default discharge-conserving meltwater
    (`melt_conserve: True`).
- **`diffusion`**: land and marine linear diffusivities
  `hillslopeKa: 0.3` and `hillslopeKm: 0.3` m²/yr, critical slope
  `hillslopeSc: 0.8` with a four-term Taylor series of the critical-slope law
  (`hillslopeNb: 4`), non-linear diffusivity of the river sediment entering
  the sea `nonlinKm: 100.` m²/yr and maximum clinoform slope
  `clinSlp: 5.e-4`. The marine keys have no effect here.
- **`flexure`** (commented out): a flat finite-element flexure (`method:
  'fem'`, elastic thickness 10 km, clamped edges). Uncomment it to add the
  isostatic response to the erosion and to the ice load.
- **`sea`**: sea level at -100 m, below the whole domain.
- **`climate`**: uniform rainfall of 1 m/yr from the start.
- **`output`**: written to `sim_glacier/`.

The glacial outputs, besides the usual elevation, erosion/deposition and
discharge, are `iceH` (ice thickness, m), `iceUb` (basal sliding velocity,
m/yr), `iceAbr` (vertical abrasion rate, m/yr; the lateral erosion is not
included), `iceMelt` (meltwater delivered to the rivers) and `iceFA` (ice
discharge).

## Running it

The mesh is provided, so the model runs directly:

```bash
mpirun -np 8 gospl -i input-glacier.yml
```

Change `-np` to the number of cores you want to use (add `-v` for the
progress of each step). Then open `view_Results.ipynb`. It grids the outputs
at steps 0 and 10 on a regular 250 m grid with
`gospl.analyse.gridexport.grid_export` and draws four panels:

- (a) the ice thickness, with the ELA contour;
- (b) the basal sliding velocity under the ice (cells with more than 1 m of
  ice);
- (c) the cumulative erosion and deposition;
- (d) a cross-valley transect through the most glaciated row of the grid,
  with the initial surface, the final surface and the diagnosed ice.

`make_ex_glacier.py` draws the same figure from the command line and saves
it as `fig_ex_glacier.pdf` and `fig_ex_glacier.png` (the figure of the goSPL
model-description paper).

## Building your own experiment

- *A changing climate*: comment out `hterm`, `hela` and `hice` and use
  `evol: 'data/ice_evol.csv'`, or edit that file (four columns: time in
  years, terminus, ELA and ice-cap altitudes in m).
- *Thicker or thinner ice*: change the product `eheight * fwidth` until the
  `iceH` output has the thickness you expect. Keep `accum_factor` at or
  below 1 (it is a fraction) and use `accum_max` as the ceiling of the
  accumulation rate.
- *Longer or shorter glacier tongues*: change `melt` (larger values give
  shorter tongues).
- *Deeper or wider troughs*: `Kg` deepens, `Kl` widens.
- *Isostasy*: uncomment the `flexure` block.

The `ice:` block and its keys are described in the goSPL user guide (surface
processes, "Ice sheets and glacial erosion"), and the equations in the
technical guide (ice sheets, glacial erosion and meltwater).
