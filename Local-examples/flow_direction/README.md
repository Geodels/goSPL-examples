# Flow routing: single, two and multiple flow directions (local)

goSPL routes water and sediment downhill to one or several lower neighbours
of each node, weighted by slope. The number of receivers is set by
`domain: flowdir`, from 1 (single flow direction, SFD) to 6 (multiple flow
direction, MFD). The example runs the same landscape three times with 1, 2
and 6 receivers and compares the discharge and the topography. The runs use
river incision (stream power law, `spl:`) and hillslope diffusion
(`diffusion:`) under uniform rainfall, without marine deposition.

## What it shows

- **SFD concentrates the flow in thin threads**: after 10 kyr, 28 % of the
  land surface (cells above 0 m) carries a discharge above 1e6 m³/yr with
  SFD, against 54 % with two receivers and 56 % with MFD (69 %, 93 % and 94 %
  above 1e5 m³/yr). MFD spreads the discharge over neighbouring cells and
  gives broader, smoother drainage.
- **The SFD network is a mesh artefact**: the dendritic pattern of the SFD run
  looks like a natural drainage network, but its branching depends on the
  random positions of the mesh nodes. With more receivers the network depends
  much less on the mesh, and flow can split into several branches over flat
  areas.
- **The topography follows**: after 100 kyr the elevation differs from the
  SFD run by 1.9 m on average with MFD and 2.2 m with two receivers (99 % of
  the cells within about 17 m). SFD favours more branching of the valleys;
  MFD gives a smoother surface with less local carving.
- **Recommendation**: use more than one receiver (`flowdir` above 1) for
  continental or global models, or when one mesh mixes several resolutions.

## The experiments

| Input | `flowdir` | Land diffusivity | Output folder |
|---|---|---|---|
| `input-sfd.yml` | 1 (SFD) | 0.01 m²/yr | `sim_sfd` |
| `input-2ngb.yml` | 2 | 0.1 m²/yr | `sim_2ngb` |
| `input-mfd.yml` | 6 (MFD) | 0.01 m²/yr | `sim_mfd` |

The question: how does the choice of flow routing change the drainage network
and the landscape? The three files differ by `flowdir` and, in
`input-2ngb.yml`, by a land diffusivity ten times larger (`hillslopeKa: 0.1`
instead of 0.01 m²/yr). Keep that in mind when comparing the two-receiver run
with the others: its deepest incision after 100 kyr is 47 m, against 65 m
(SFD) and 64 m (MFD). Set `hillslopeKa: 0.01` in `input-2ngb.yml` for a
comparison on `flowdir` alone.

## The set-up, block by block (`input-mfd.yml`)

- **`domain`**: the mesh `data/sombrero.npz` (keys `v`, `c`, `z`), 288,467
  nodes over 100 x 100 km at 200 m spacing (edges from 149 to 246 m). The
  surface is rotationally symmetric, with radial valleys and ridges around a
  central high (elevations from -200 to +575 m), and random noise added (the
  formula is in `model_setup.ipynb`). `flowdir: 6` (MFD), `bc: 'oooo'` (open
  edges), and `seadepo: False`: no marine deposition, the sediment reaching
  the sea leaves the model.
- **`time`**: 0 to 100 kyr in 1 kyr steps, an output every 5 kyr (steps 0 to
  20).
- **`spl`**: `K: 4.0e-6` (SI units, with drainage area in m²), `m: 0.5`,
  `d: 0` (the rainfall does not enter the erosion law).
- **`diffusion`**: land diffusivity `hillslopeKa: 0.01` m²/yr, marine
  diffusivity `hillslopeKm: 0.2` m²/yr.
- **`sea`**: sea level at `position: -100` m.
- **`climate`**: uniform rainfall of 1 m/yr.
- **`output`**: written to `sim_mfd/`.

`data/tec.npz` holds a uniform horizontal velocity (key `uv`, -0.05 m/yr
along x and y) that none of the three inputs uses; `input-sfd.yml` has a
commented `advect: 'iioe2'` line.

## Running it

```bash
conda activate gospl
mpirun -np 4 gospl -i input-sfd.yml
mpirun -np 4 gospl -i input-2ngb.yml
mpirun -np 4 gospl -i input-mfd.yml
```

The mesh is shipped in `data/`. `model_setup.ipynb` shows how it was built
(set `make_mesh = True`): a 500 x 500 grid at 200 m, the sombrero surface plus
noise, meshed with `mpas_tools`. It writes the new mesh to
`sombrero/sombrero.npz`, not to `data/`, and the noise is random, so a new mesh
differs from the shipped one.

Then open `view_Results.ipynb`. It rasterises the 21 output steps of each run
onto a 200 m grid (`results/sfd<step>.nc`, `results/2ngb<step>.nc`,
`results/mfd<step>.nc`), then plots:

- the elevation of each run at 100 kyr, with a zoom on a 20 x 20 km window
  (x from 30 to 50 km, y from 50 to 70 km);
- the elevation difference to the SFD run (two receivers minus SFD, MFD minus
  SFD);
- the discharge on land at 10 kyr (log scale, 1e5 to 1e8 m³/yr) for each run,
  with zooms on two windows (x from 30 to 50 km and from 5 to 25 km, y from 50
  to 70 km).

A single step can also be rasterised from a terminal:

```bash
gospl-grid --h5dir sim_mfd/h5 --mesh data/sombrero.npz:v:c --step 20 --spacing 200 --out results/mfd20.nc
```

`make_ex_flowdir.py` draws the figure of the goSPL model-description paper
(`fig_ex_flowdir.pdf` and `.png`): the discharge of the three runs at 10 kyr
over the whole domain and in the 20 x 20 km window. It reads
`results/sfd2.nc`, `results/2ngb2.nc` and `results/mfd2.nc`, so run the
notebook first.

## Building your own experiment

- *Another number of receivers*: any `flowdir` from 1 to 6.
- *Mesh sensitivity*: rebuild the mesh with `model_setup.ipynb` (new random
  node positions and noise) and run the SFD and MFD inputs again; compare the
  discharge maps of the two meshes.
- *Another resolution*: in `model_setup.ipynb` the surface is computed on a
  500 x 500 grid (`x1`, `y1`, step 0.025) and placed at `dx` = 200 m; change
  `dx`, `nx`, `ny` and the step of `x1`, `y1` together to test how each
  routing scheme behaves on a coarser or finer mesh.
