# Stratigraphic record of a passive margin (local)

A 200 x 100 km passive margin, from 300 m high ground in the west to a 500 m
deep basin in the east, evolves for 250 kyr under an oscillating sea level.
Rivers erode the land (stream power law, `spl:`), hillslopes creep
(`diffusion:`), the sediment reaching the sea builds clinoforms (marine
deposition and diffusion), the lithosphere bends under the load (`flexure:`),
and goSPL records every deposit in a stratigraphic layer every 5 kyr
(`time: strat`), with compaction (`compaction:`). The example then extracts
the stratal architecture with goSPL's post-processing toolkit
(`gospl.analyse`): cross-sections coloured by facies, synthetic wells and a
Wheeler (chronostratigraphic) diagram. If you are new to goSPL, start here.

## What it shows

- **A prograding wedge**: over 250 kyr the shoreline of the margin (taken on
  the elevation averaged along y) moves seaward from about 104 km to about
  117 km from the western edge, while sea level falls from -29 m to -46 m.
  The deposits reach 238 m in thickness at the end of the run.
- **Cycles in the record**: sea level oscillates with a growing amplitude and
  period (lowest at -49.6 m at 160 kyr). Each fall and rise floods and exposes
  the shelf; the 50 recorded layers keep the water depth at the time of
  deposition, so a section coloured by facies shows the shoreface, offshore
  and slope deposits stacking and shifting with each cycle.
- **Reading the record**: the Wheeler diagram (distance against deposition
  time) leaves hiatuses and eroded intervals blank and draws the shoreline
  trajectory, so regressions and transgressions read directly from it.
- **Continental deposition** (`input-stratiG.yml`, `G: 3`): rivers deposit
  part of their load on land. At 250 kyr, 33 % of the land cells carry more
  than 1 m of sediment (16 % in `input-strati.yml`), and the mean land deposit
  is 10.2 m (3.8 m); the mean marine deposit is 8.9 m (66.2 m). This run also
  differs by its edges, compaction depth and the absence of flexure (see the
  table below), so it is not a one-parameter comparison.

## The experiments

| Input | Output folder | What differs | Question |
|---|---|---|---|
| `input-strati.yml` | `strati` | reference: no river deposition on land (`G` = 0), open edges, compaction depth 2000 m, flexure | what stratigraphy does a sea-level cycle record on a margin? |
| `input-stratiG.yml` | `stratiG` | `G: 3` (transport-limited rivers), `bc: 'ofof'`, compaction depth 3700 m, no flexure | how much sediment stays on land when rivers can deposit? |
| `input-advect.yml` | `advect` | horizontal motion of the whole surface (`tectonics: hdisp`, scheme `iioe2`), no stratigraphy recorded | how does the margin evolve while it is carried laterally? |

`view_Results.ipynb` analyses the `strati` run. The two other runs can be
analysed with the same cells by changing `h5dir` (no stratigraphy in
`advect`: the `strat` key is absent from its `time` block, so only the
surface cells apply).

## The set-up, block by block (`input-strati.yml`)

- **`domain`**: the mesh `inputs/gospl_mesh.npz` (keys `v`, `c`, `z`), 47,672
  nodes over 200 x 100 km. Its cell size varies with a noise map built in
  `model_setup.ipynb` (edges from 334 to 1,246 m, median 681 m). The initial
  surface is a smoothed across-margin profile (300 m at 0 km, 40 m at 50 km,
  10 m at 60 km, 0 m at 100 km, -500 m at 200 km) with Perlin-noise roughness
  added (elevations from -463 to +262 m). `flowdir: 5` routes the flow to up
  to 5 downhill neighbours (multiple flow direction); `bc: 'oooo'` makes the
  four edges open outlets.
- **`time`**: 0 to 250 kyr in 5 kyr steps, an output every 10 kyr (steps 0 to
  25), and a stratigraphic layer every 5 kyr (`strat: 5.e3`, 50 layers).
- **`spl`**: `K: 4.e-5` (SI units, with drainage area in m² and rainfall in
  m/yr), `m: 0.45`, `d: 0.42` (the rainfall enters the erosion law as
  `(P^d A)^m`). `G` is not set (0): rivers do not deposit on land, their load
  goes to depressions and to the sea.
- **`diffusion`**: land diffusivity `hillslopeKa: 0.1` m²/yr, marine
  diffusivity `hillslopeKm: 0.8` m²/yr, `nonlinKm: 1000` m²/yr for the fresh
  river sediment spread on the seafloor, and a maximum clinoform slope
  `clinSlp: 5.e-4` (the top of the marine deposit is capped by this slope
  times the distance to the shore). The marine diffusion uses goSPL's default
  solver, `picard` with 20 sub-steps per time step (`picardSub`); the
  commented lines show how to switch to `ts` or change the sub-steps.
- **`flexure`**: flexural isostasy on the flat mesh (`method: 'fem'`), elastic
  thickness 30 km, load density 2300 kg/m³, mantle density 3300 kg/m³,
  Young's modulus 65 GPa, Poisson ratio 0.25; `Mirror` north and south edges,
  `0Slope0Shear` east and west edges.
- **`sea`**: the curve `inputs/sealevel.csv` (time in years, sea level in m)
  shifted by `position: -30` m (goSPL adds `position` to the curve). The curve
  is a falling trend from 0 to -40 m over 500 kyr with an oscillation whose
  amplitude grows from 1 to 20 m and whose period grows from 20 to 100 kyr.
  The run uses its first 250 kyr: sea level goes from -29 m to -46.2 m, with a
  low of -49.6 m at 160 kyr.
- **`climate`**: uniform rainfall of 1 m/yr.
- **`compaction`**: surface porosity `phis: 0.49` and e-folding depth
  `z0s: 2000` m of the porosity-depth curve.
- **`output`**: written to `strati/` (`h5/` holds the surface outputs
  `gospl.<step>.p<rank>.h5` and the stratigraphy `stratal.<step>.p<rank>.h5`).

`input-advect.yml` adds `advect: 'iioe2'` to `domain` and a `tectonics` event
from 0 to 250 kyr with `hdisp: ['inputs/gospl_mesh','uv']`: a uniform
velocity of -0.05 m/yr along x and along y (the surface moves 12.5 km south-west
on each axis over the run).

## Running it

```bash
conda activate gospl

# optional: rebuild the mesh and the sea-level curve (inputs/)
jupyter notebook model_setup.ipynb

mpirun -np 4 gospl -i input-strati.yml
mpirun -np 4 gospl -i input-stratiG.yml
mpirun -np 4 gospl -i input-advect.yml
```

The mesh and sea-level curve are shipped in `inputs/`, so `model_setup.ipynb`
is needed only to change them. It builds the profile and the noise, meshes the
domain with `mpas_tools` (files in `meshinputs0/`), and writes
`inputs/gospl_mesh.npz` (with the velocity field `uv` used by
`input-advect.yml`) and `inputs/sealevel.csv`. The Perlin noise is random:
running it again gives a different surface and mesh, and overwrites the
shipped ones.

Then open `view_Results.ipynb`. It rasterises every output step of `strati`
onto a 500 m grid (`results/surface<step>.nc`, with elevation, erosion and
deposition, discharge, basins and chi), and plots:

- the elevation and erosion/deposition rate at steps 5, 10, 15 and 25, with
  the shoreline;
- the margin profile averaged along y through time;
- the drainage basins, and the river network and long profile of one basin
  (basin 43, picked at x = 25 km, y = 33 km);
- cross-sections of the stratigraphy at 250 kyr coloured by facies and by
  layer thickness, a porosity well at (135, 50) km, a panel of three wells at
  115, 135 and 145 km, and a Wheeler diagram along x at y = 25 km.

The facies are set by the water depth at deposition: above sea level fluvial
or deltaic plain, 0 to 20 m shoreface, 20 to 50 m distal offshore, 50 to 75 m
upper slope, deeper lower slope (the bins can be changed with
`facies_depths`).

Every plot has a terminal equivalent, for example:

```bash
gospl-grid --h5dir strati/h5 --mesh inputs/gospl_mesh.npz:v:c --step 25 --spacing 500 --out results/surface25.nc
gospl-section --h5dir strati/h5 --mesh inputs/gospl_mesh.npz:v:c --kind cross --step 25 --color-by facies --vexag 20 --out section_facies.pdf
gospl-section --h5dir strati/h5 --mesh inputs/gospl_mesh.npz:v:c --kind wheeler --step 25 --along x --at 25000 --color-by facies --strat-dt 5000 --out wheeler.pdf
mpirun -np 4 gospl-strata-volume --h5dir strati/h5 --outdir results --steps 25 --field basic
```

The last one writes the whole pile as a 3-D volume for ParaView
(`results/strata.xdmf`).

`make_ex_strat.py` draws the figure of the goSPL model-description paper
(`fig_ex_strat.pdf`): the facies section with three well sites, the Wheeler
diagram, the sea-level curve, and wells from the `dual_lithology` example
(it reads `../dual_lithology/strati_dual`, so run that example first).

## Building your own experiment

- *Another margin*: change the control points of the profile (`xdata`,
  `elev`) or the domain size in `model_setup.ipynb`; `cellwidth` sets the mesh
  resolution map (m).
- *Another sea-level history*: change `seaExt`, `ampExt` and `periodExt` of
  `buildCurve` in `model_setup.ipynb`, or give your own two-column file
  (time in years, sea level in m) to `sea: curve`. Remember that `position` is
  added to it.
- *Thinner or thicker layers*: `time: strat` sets the interval of the record;
  a shorter one gives more layers and larger output files.
- *Lateral motion that carries the valleys*: the Eulerian schemes (`upwind`,
  `iioe1`, `iioe2`) of `input-advect.yml` do not move the drainage pattern
  with the surface. Use `advect: 'interp'` with `interp_remap: 1` in `domain`
  when the valleys should move with the plate (see the goSPL user guide,
  input file page).
- *Sand and mud, or the source of the sediment*: see the `dual_lithology`
  example, which uses the same margin with two lithologies or provenance
  tracers.
