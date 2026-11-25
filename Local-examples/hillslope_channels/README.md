# Hillslopes and channels: removing the resolution dependence with `channelQ` (local)

goSPL couples river incision (stream power law, `spl:`) and hillslope
diffusion (`diffusion:`) by operator splitting. By default the diffusion acts
on every land cell, channel cells included. A river is a line of single cells,
so the flux the hillslopes deliver into it grows as the cells shrink: the finer
the mesh, the more the hillslopes fill the valleys, and the steeper the
channels must become to carry that material away (Hergarten and Pietrek,
2023). The balance between the two processes then depends on the mesh
resolution. The optional key `diffusion: channelQ` (m³/yr) removes this
dependence. The example is the test of the goSPL model-description paper
(Sect. 3.4): an uplifted block at 200, 100 and 50 m spacing, with and without
`channelQ`. `channelQ` is not in a goSPL release yet (the latest is
v2026.7.14): it needs goSPL `dev`.

## What it shows

- **Without `channelQ` the channels steepen as the mesh is refined**: the
  steady channel steepness `k_s = S A^(1/2)` is 30.5, 40.3 and 65.1 m at 200,
  100 and 50 m spacing, against the theoretical `U/K = 25` m.
- **With `channelQ` it stays at U/K at every resolution**: 25.1, 25.2 and
  25.4 m. The diffusion now shapes only the hillslopes, and the channel slopes
  follow the fluvial law.
- **The relief follows**: the final mean elevation is 77, 120 and 203 m
  without `channelQ`, 70, 87 and 116 m with it. With `channelQ` it still grows
  a little with resolution, as finer meshes resolve smaller and steeper
  headwaters (the paper reports 70 to 96 m without any diffusion).
- **The landscapes at 50 m**: without `channelQ` broad valleys and high
  relief; with it the hillslopes grade to narrow channels.

## How `channelQ` works

Cells whose discharge (the `FA` output, m³/yr) reaches `channelQ` are channels:

- they are held at their elevation during the diffusion solve, so the
  hillslopes grade to them;
- the volume the hillslopes deliver to them joins the river load of the same
  step (with deposition on, the hillslope step runs between the incision and
  the sediment routing);
- a channel cell that would lose material to a lower neighbour keeps the loss,
  so the volume budget is exact.

It works with the linear and non-linear hillslope laws, the soil model,
stratigraphy, dual lithology and provenance tracers. It is off by default.

## The experiments

| Input | Mesh spacing | `channelQ` | Output folder |
|---|---|---|---|
| `input-200.yml` | 200 m | no (diffusion on every cell) | `sim_200` |
| `input-200_chan.yml` | 200 m | 1e6 m³/yr | `sim_200_chan` |
| `input-100.yml` | 100 m | no | `sim_100` |
| `input-100_chan.yml` | 100 m | 1e6 m³/yr | `sim_100_chan` |
| `input-50.yml` | 50 m | no | `sim_50` |
| `input-50_chan.yml` | 50 m | 1e6 m³/yr | `sim_50_chan` |

The question: does the same land diffusivity (0.2 m²/yr) give the same
channels at every resolution? All six files are written by `make_mesh.py`
from one template; a `_chan` file differs from its partner only by the
`channelQ` line.

## The set-up, block by block (`input-100_chan.yml`)

- **`domain`**: a 20 x 20 km block (`data/mesh100.npz`), a square grid of
  nodes at 100 m spacing whose interior nodes are moved randomly by up to a
  quarter of a cell, then triangulated: an irregular mesh with no exact ties
  in discharge (all cell areas differ). The initial interior surface is random
  between 0 and 1 m, the edges at 0 m. Single flow direction (`flowdir: 1`),
  fixed base-level edges (`bc: 'ffff'`), no marine deposition
  (`seadepo: False`), and `nodep: True`: detachment-limited, the eroded
  material leaves the block.
- **`time`**: 3 Myr in 5 kyr steps, an output every 250 kyr. The block
  reaches steady state after about 0.5 Myr; the runs continue to check that it
  holds.
- **`spl`**: `K: 4.e-5` (m^(1-2m)/yr), `m: 0.5`, `d: 0.`, slope exponent 1, so
  the theoretical channel steepness is `U/K = 25` m.
- **`diffusion`**: land diffusivity `hillslopeKa: 0.2` m²/yr, no marine
  diffusion (`hillslopeKm: 0.`), and `channelQ: 1.e6` m³/yr: with 1 m/yr of
  rain, cells draining more than 1 km² are held out of the diffusion.
- **`sea`**: sea level at -10 m, below the whole block.
- **`climate`**: uniform rain of 1 m/yr (so the discharge in m³/yr equals the
  drainage area in m²).
- **`tectonics`**: uplift of 1 mm/yr on the interior nodes and 0 on the edges
  (key `t` of the mesh file), from 0 to 100 Myr (longer than the run).

## Running it

```bash
conda activate gospl
python make_mesh.py      # meshes data/mesh200.npz, mesh100.npz, mesh50.npz and the six input files
./run_all.sh             # the six runs; or one run: mpirun -np 4 gospl -i input-100_chan.yml
```

`run_all.sh` runs the 200 m pair on 2 MPI ranks each and the 100 m pair on 5
side by side, then the 50 m pair on 8 each (adjust `-np` to your machine). The
six runs take about 15 minutes on 16 cores, most of it the two 50 m runs; the
logs (`log_<run>.txt`) give the run times, from about 17 s (200 m) to about
9 min (50 m).

Then open `view_Results.ipynb`: the channel steepness at steady state for the
six runs (the median of `S A^(1/2)` over the cells draining more than 1 km²,
with `S` the steepest downhill slope along the mesh edges), the mean
elevation of the block through time, and the final elevation of the two 50 m
runs with the channel cells (`FA >= channelQ`) outlined. The measuring
functions are in `scripts/channels.py`.

## Building your own experiment

- *Choosing `channelQ`*: it is a discharge, a channel-head drainage area times
  the rainfall, `channelQ = A_c x P`. Here 1 km² with 1 m/yr of rain gives
  `1.e6`. Keep `A_c` at least about ten cell areas on the coarsest mesh you
  use, and check the mask on a first output: the cells with
  `FA >= channelQ` should outline the valley network you expect.
- *Another diffusivity, uplift or block size*: `KD`, `U` and `L` at the top of
  `make_mesh.py` (and `QC` for `channelQ`), then run it again.
- *Another resolution*: add a spacing to the loop `for dx in (200, 100, 50)`
  of `make_mesh.py`, and to `SPACINGS` in the notebook.
- *Avoid thresholds on a lattice*: on a perfectly regular mesh with uniform
  rain the discharge only takes multiples of the cell area times the rain, so
  a `channelQ` set exactly on one of these values flips cells by round-off.
  The jittered meshes of this example avoid it.

## References

- Hergarten, S., Pietrek, A. (2023). Self-organization of channels and
  hillslopes in models of fluvial landform evolution and its potential for
  solving scaling issues. Earth Surface Dynamics 11, 741-755.
