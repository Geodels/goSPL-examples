# Implicit time stepping: how the time step changes a landscape (local)

The same uplifted landscape is run to steady state four times, with internal
time steps of 500, 1250, 2500 and 5000 yr. goSPL solves the flow
accumulation and the stream-power law (`spl:`) implicitly, so a long time
step does not make the solution blow up as it would with an explicit,
Courant-limited scheme; the cost of a long step shows up instead as a loss
of accuracy in the transient landscape. The model is purely fluvial and
erosional: single-flow-direction routing, detachment-limited river incision,
no hillslope diffusion and no deposition.

## What it shows

- **Every run is stable and reaches steady state.** The mean elevation rises
  from 100 m to about 168 m and levels off after about 60 kyr, whatever the
  time step (mean of the gridded outputs: 164.6 m at 40 kyr and 168.8 m at
  60 kyr with dt = 500 yr; 162.7 m and 168.2 m with dt = 5 kyr).
- **The time step costs accuracy in the transient.** The longer the step,
  the more the mean-elevation curve lags during the rise (at 30 kyr: 157.2 m
  with 500 yr, 155.2 m with 5 kyr), because a long step smears out the
  migrating knickpoints.
- **The steady-state relief is similar, the detail is not.** After 100 kyr
  the mean elevations are 167.3, 168.4, 167.7 and 168.1 m (dt = 0.5, 1.25,
  2.5 and 5 kyr), less than 1 % apart. The maximum elevations are 436, 450,
  468 and 458 m: the notebook prints an average of 452.91 m with a standard
  deviation of 11.51 m. The large-scale drainage organisation is kept, but
  the valley and divide positions differ between the runs.
- **The time step also sets the run time.** The iterative solvers start from
  the previous step's solution, so when the landscape changes little between
  two steps they converge in few iterations; a shorter step does not always
  cost proportionally more time.

## The experiments

| Input | `dt` | Output interval | Output folder | Question |
|---|---|---|---|---|
| `input-500.yml` | 500 yr | 2.5 kyr | `sim_slope_500` | the reference (shortest step) |
| `input-1.25k.yml` | 1250 yr | 2.5 kyr | `sim_slope_1.25k` | |
| `input-2.5k.yml` | 2500 yr | 2.5 kyr | `sim_slope_2.5k` | |
| `input-5k.yml` | 5000 yr | 5 kyr | `sim_slope_5k` | how far can the step grow? |

The four input files are identical apart from `time: dt`, `time: tout` (5k
run only) and `output: dir`.

## The set-up, block by block (`input-500.yml`)

- **`domain`**: the mesh `data/gospl_mesh.npz` (keys `v`, `c`, `z`): a
  100 x 100 km square at 200 m resolution, 290,007 nodes. The initial surface
  is flat at 100 m with Gaussian noise of 0.05 m standard deviation (elevations
  99.8 to 100.2 m), which seeds the drainage. `flowdir: 1` routes the flow to
  a single receiver (SFD). `bc: 'oooo'` makes the four edges (north, east,
  south, west) open. `seadepo: False` and `nodep: True` switch off marine and
  continental deposition: the model is detachment-limited.
- **`time`**: 0 to 100 kyr, an output every 2.5 kyr, internal step `dt` of
  500 yr (the variable studied here).
- **`spl`**: stream-power law `E = K (P^d A)^m S^n` with `K = 2e-4`, `d = 0`
  (no precipitation dependence) and `m = 0.5` (`n = 1`). `K` is chosen so the
  landscape reaches steady state within 100 kyr.
- **`diffusion`**: `hillslopeKa: 0`, no hillslope diffusion on land.
  `hillslopeKm: 0.2` m2/yr is the marine diffusivity; the domain starts 100 m
  above sea level and has no marine deposition.
- **`sea`**: sea level fixed at 0 m.
- **`climate`**: uniform rainfall of 1 m/yr from the start.
- **`tectonics`**: one event from 0 to 100 Myr (longer than the run) with the
  uplift rate `t` of the mesh file (m/yr): it increases linearly from 0 on the
  western edge (x = 0) to 5 mm/yr on the eastern edge (x = 100 km).
- **`output`**: the output folder (`sim_slope_500`).

## Running it

The mesh is provided in `data/gospl_mesh.npz`. `model_setup.ipynb` describes
the set-up and holds the code that built it (cell with `make_mesh = False`;
set it to `True` to rebuild it, see the `generate_2Dmesh` example). Then run
the four experiments (here on 4 MPI ranks):

```bash
mpirun -np 4 gospl -i input-500.yml
mpirun -np 4 gospl -i input-1.25k.yml
mpirun -np 4 gospl -i input-2.5k.yml
mpirun -np 4 gospl -i input-5k.yml
```

Then open `view_Results.ipynb`. It rasterises every output step of the four
runs to a 200 m regular grid with `gospl.analyse.gridexport` and writes them
to `results/<run>_surface_<step>.nc`, with the prefixes `500`, `125`
(1.25 kyr), `250` (2.5 kyr) and `5k` (steps 0 to 40, or 0 to 20 for `5k`).
It then plots the final elevation maps of the four runs side by side, prints
the mean and spread of their maximum elevations, and plots the mean
elevation through time for each time step. The same grids can be made from
a terminal, one step at a time:

```bash
gospl-grid --h5dir sim_slope_500/h5 --mesh data/gospl_mesh.npz:v:c \
    --step 40 --spacing 200 --out results/500_surface_40.nc
```

`make_ex_timestep.py` reads the `results/` grids and draws the summary figure
`fig_ex_timestep.pdf` (and `.png`): the mean-elevation histories and the
steady-state topographies at dt = 0.5 and 5 kyr.

## Building your own experiment

- *Another time step*: copy an input file and change `time: dt` (and
  `output: dir`); keep `tout` a multiple of `dt`.
- *Another landscape*: change the uplift (`tectonics: upsub`), the rainfall
  (`climate`) or the erodibility `K`; a lower `K` takes longer to reach
  steady state, so lengthen `time: end`.
- *Hillslopes and deposition*: set `hillslopeKa` above 0, or remove
  `nodep: True`, to see whether the sensitivity to the time step changes
  when other processes act.
