# Marine gravity currents on a passive margin (regional)

Rivers deliver most of their load to the coast, where goSPL's clinoform scheme
builds a prograding wedge of topsets and foresets. During floods, part of the
load plunges as hyperpycnal flows, and failures of the delta front make
turbidity currents: these gravity currents cross the shelf and the slope and
deposit at the foot of the slope and on the basin floor. goSPL's `gravity:`
block adds this second pathway. The currents are not resolved as flows: a
fraction of the river load reaching the sea is routed down the seafloor and
deposited gradually over a run-out length. The example runs the margin of the
`dual_lithology` example (coarse and fine sediment, falling and oscillating
sea level, flexure) with and without gravity currents. It needs the
`feat/gravity-carbonates-karst` branch of goSPL.

## What it shows

- **Deposits beyond the clinoforms**: without gravity currents the clinoform
  wedge ends at the foot of the slope (about 165 to 175 km) and the basin
  floor receives nothing. With them, sediment bypasses the shelf and the
  foresets and spreads over the basin floor, and a band of bottomset layers
  continues from the toe of the slope to the edge of the domain at 200 km.
- **Sorting by grain size**: the coarse load follows a concentrated network
  and stays nearer the base of the slope; the fine load, on a spreading
  network with a longer run-out, goes further. With `coarse_factor: 0.3` the
  gravity deposits are mixed (fine fraction about 0.4 to 0.6 near the base of
  the slope), get finer basinward, and keep finer streaks between the main
  pathways.
- **Deep-water facies**: coloured by the water depth at deposition, the gravity
  bottomsets are all lower-slope facies (deposited in more than 75 m of
  water).
- **A closed budget**: over the 250 kyr of `input-gravity.yml` the currents
  are supplied 150 km³, deposit 128 km³ and export 22 km³ through the open
  edges (sums of the per-step records in `summary_gravity.jsonl`); the
  notebook plots the closure of the budget at every step, to the solver
  tolerance.
- **Slope failure**: in `input-failure.yml`, failures start after about
  130 kyr, when the falling sea level has steepened the foresets, and grow to
  about 1 km³ per step, a quarter of the river supply to the currents
  (10.8 km³ over the run in `summary_failure.jsonl`). They concentrate in a
  few scars on the foresets; the material reappears in the bottomsets. At this
  scale the differences are small: a few metres removed from a pile about
  300 m thick.
- **Flexure, not erosion**: the basin floor drops by 30 to 50 m in both runs,
  the flexural subsidence under the sediment load.

## The experiments

| Input | `gravity:` block | Slope failure | Output folder |
|---|---|---|---|
| `input-nogravity.yml` | no | | `sim_nogravity` (reference) |
| `input-gravity.yml` | yes | no | `sim_gravity` |
| `input-failure.yml` | yes | yes (`failure_slope: 0.02`, `max_failure: 10.`) | `sim_failure` |

The three files differ only by these lines.

## The set-up, block by block (`input-gravity.yml`)

- **`domain`**: the 200 x 100 km margin of the `dual_lithology` example, an
  unstructured mesh of 615 m cells (52,805 nodes, `data/gospl_mesh.npz`). The
  initial surface slopes from +261 m on the western edge to about sea level at
  x = 80 to 90 km, then down to -463 m on the eastern edge. Multiple flow
  directions (`flowdir: 5`), all edges open (`bc: 'oooo'`).
- **`time`**: 250 kyr in 5 kyr steps (50 steps), an output every 10 kyr, a
  stratigraphic layer every step (`strat: 5.e3`).
- **`spl`**: `K: 4.e-5`, `d: 0.42` (erosion grows with the rain), `m: 0.45`.
- **`diffusion`**: land and marine hillslope diffusivities 0.1 and 0.8 m²/yr;
  the river sediment entering the sea spreads with a non-linear diffusivity of
  1000 m²/yr (`nonlinKm`), its top limited by a maximum clinoform slope of
  5e-4 (`clinSlp`).
- **`flexure`**: flat finite-volume flexure (`method: 'fem'`), elastic
  thickness 30 km, densities 2300 and 3300 kg/m³, Young's modulus 65 GPa,
  Poisson ratio 0.25; `Mirror` edges in the north and south, `0Slope0Shear`
  in the east and west.
- **`sea`**: the curve of `data/sealevel.csv` (time in years, sea level in m)
  shifted by `position: -30.`: -29 m at the start, a fall with oscillations
  to -49.6 m at 160 kyr, a rise to -38 m at 215 kyr and -46 m at the end.
- **`climate`**: uniform rain of 1 m/yr.
- **`strata`**: dual lithology (`dual: True`), coarse (sand) and fine (mud)
  sediment with their own compaction curves (surface porosity `phi0` 0.49 and
  0.63, e-folding depth `z0` 3700 and 1960 m). The fine sediment is 1.5 times
  more erodible (`k_factor`) and diffuses twice as far (`fine_diff_factor`);
  the bedrock is 60 % coarse (`bedrock_coarse_frac`). In depressions, half of
  the coarse deposit is placed at the inlets and the fine deposit fills the
  bowl (`pitInletBias`).
- **`gravity`**:
  - *Trigger*: `fraction: 0.3`, 30 % of the river load reaching the sea
    becomes gravity currents; `coarse_factor: 0.3`, the coarse load is
    diverted with 0.3 times that fraction (the currents carry mostly
    suspended load).
  - *Run-out*: the deposit decays by a factor e over `runout: 1.e4` m
    (coarse) and `runout_fine: 2.e4` m (fine).
  - *Bypass*: no deposition shallower than 50 m below sea level (the default
    `bypass_depth`) and less on gradients steeper than `bypass_slope: 0.01`
    (half the rate at 0.01), so the currents cross the shelf and the foresets.
  - *Networks*: on the actual seafloor, the coarse flux is shared among 2
    downslope neighbours in proportion to slope^4 (`receivers: 2`,
    `flow_exp: 4.`, concentrated belts), the fine flux among 6 in proportion
    to slope (`receivers_fine: 6`, `flow_exp_fine: 1.`, spreading lobes).
  - *Smoothing*: 3 passes on the routing surface (`network_smooth`) and on
    the slope used by the bypass (`slope_smooth`), and each step's deposit is
    spread laterally over 1500 m (`deposit_spread`), which removes cell-scale
    bumps.
- **Slope failure** (`input-failure.yml` only): sediment deposited during the
  run fails where the seafloor below the bypass depth is steeper than 0.02,
  down to that gradient, at most 10 m per cell and step, and joins the
  gravity currents with its composition. Bedrock never fails. The gradient
  0.02 was chosen from the foresets of this margin, which reach about 0.03.

The equations of the routing and every parameter are in the goSPL user and
technical guides (*Marine gravity currents*).

## Running it

The mesh and the sea-level curve are in `data/` (the same files as in
`dual_lithology`), so there is nothing to build.

```bash
./run_all.sh     # the three runs side by side, 4 MPI ranks each (about 2 min per run on a laptop)
# or one run:
mpirun -np 4 gospl -i input-gravity.yml --summary summary_gravity.jsonl
```

`--summary` writes a per-step record (`summary_<run>.jsonl`) that includes the
gravity budget (supplied, deposited, exported, eroded and failed volumes);
the logs go to `log_<run>.txt`.

Then open `view_Results.ipynb`: the gravity budget through time, map views of
the two runs (shaded bathymetry, net gravity deposit `gravDep`, surface fine
fraction), strike sections every 10 km from the shelf (130 km) to the basin
floor (190 km) with the cumulative deposit and the flexure, dip sections at
y = 30, 55 and 80 km coloured by fine fraction and by depositional facies
(drawn with `gospl.analyse.stratasection.cross_section`, also the
`gospl-section` command), and the same plots for the slope-failure run (dip
sections through the scars at y = 5, 58 and 88 km). The plotting functions
are in `scripts/gravity.py`.

## Building your own experiment

- *Floods instead of a fixed fraction*: replace `fraction: 0.3` by a `flood:`
  block (`sigma: 2.` for flashy rivers), which computes the fraction from the
  floods above the hyperpycnal threshold.
- *Coarse lobes*: `coarse_factor: 1.` diverts as much coarse as fine load;
  thick coarse lobes build at the foot of the slope.
- *Without spreading*: `deposit_spread: 0` (on by default, over two cell
  sizes); the concentrated coarse network then builds ridges one cell wide.
- *Deposits further into the basin*: longer `runout` and `runout_fine`.
- *More or less failure*: `failure_slope` (0.015 to 0.03) and `max_failure`
  in `input-failure.yml`.
- *Slope erosion* (`erosion_length`) is left off: it is an expert option for
  meshes of 200 m cells or finer, and at this resolution (615 m cells, 5 kyr
  steps) it only roughens the toe of the slope. See the goSPL user guide for
  the cautions.
