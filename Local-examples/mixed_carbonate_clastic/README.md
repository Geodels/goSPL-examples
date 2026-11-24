# Mixed carbonate-siliciclastic margin (regional)

A passive margin where rivers and carbonate factories share the shelf. It
starts from the `stratigraphic_record` example (same mesh, topography and
sea-level curve) and adds goSPL's carbonate module (`carbonate:`), and in some
runs the wave energy (`waves:`), coastal erosion (`coastal_erosion:`) and
gravity currents (`gravity:`). It needs the `feat/carbonates` branch of goSPL.

## What it shows

- **Carbonates where the water is clear**: two wet catchments feed two large
  deltas. The factories stop where river sediment accumulates (`sedrate`)
  and in turbid water (`turbidity`, the river mud in transit): the delta
  corridors stay free of carbonate (14 % of the delta deposits), the
  carbonate grows on the shallow flanks of the delta lobes and on the coast
  between the deltas (35 % of its deposits).
- **Why turbidity, not only deposition**: on this margin the river load
  crosses the shallow carbonate belt without settling, so the `sedrate`
  control alone cannot tell the deltas from the coast between them
  (production over the deltas / between: 0.98); with the `turbidity`
  control, 0.59.
- **Reciprocal sedimentation**: the production follows the sea level. During
  the fall the rivers build clastic clinoforms and the production almost
  stops; during the rise carbonate covers the flooded shelf again.
- **Waves**: reef builders that need wave energy move to the exposed fronts
  of the delta lobes; coastal erosion moves eroded reef as carbonate sand.
- **Gravity currents**: part of the river load runs down the slope and builds
  a siliciclastic sheet on the basin floor; with less mud on the shelf the
  factories produce more.
- **Platform-scale carbonates**: at ~700 m cells a reef is narrower than a
  cell, so `reef` is the framework share of a cell's carbonate (a margin
  facies). For resolved reefs and terraces see the `carbonate_terraces` and
  `carbonate_atoll` examples (50-200 m).

## The experiments

| Input | Carbonate | River controls | Waves (reef control) | Coastal erosion | Gravity currents | Purpose |
|---|---|---|---|---|---|---|
| `input-clastic.yml` | no | | | | | siliciclastic reference |
| `input-nocontrol.yml` | yes | none | | | | carbonates that ignore the rivers |
| `input-sedrate.yml` | yes | `sedrate` | | | | burial control alone |
| `input-mixed.yml` | yes | `sedrate` + `turbidity` | no | no | no | **mixed reference** |
| `input-wavecontrol.yml` | yes | `sedrate` + `turbidity` | yes | no | no | wave control alone |
| `input-waves.yml` | yes | `sedrate` + `turbidity` | yes | yes (rubble: carb_sand) | no | waves and erosion |
| `input-gravity.yml` | yes | `sedrate` + `turbidity` | no | no | yes | gravity currents |

Every input file is generated from one commented template by
`scripts/make_inputs.py`; its header says what the run tests and how it
differs from the reference.

| Run | Carbonate produced (km3): framework + loose | Carbonate share of the deposits: deltas / between | Production over the deltas / between |
|---|---|---|---|
| nocontrol | 160 + 452 | 0.37 / 0.52 | 1.01 |
| sedrate | 131 + 343 | 0.32 / 0.46 | 0.98 |
| mixed | 58 + 156 | 0.14 / 0.35 | 0.59 |
| wavecontrol | 39 + 148 | 0.13 / 0.28 | 0.70 |
| waves | 41 + 161 | 0.15 / 0.36 | 0.54 |
| gravity | 77 + 210 | 0.21 / 0.33 | 0.71 |

## The set-up, block by block (`input-mixed.yml`)

- **`domain`**: a 200 x 100 km margin on an unstructured mesh of ~700 m
  cells: highlands at +260 m in the west, a coastal plain, a flat shelf near
  sea level out to ~100 km and a slope down to -460 m in the east. Built in
  `stratigraphic_record/model_setup.ipynb` (`inputs/gospl_mesh.npz`).
- **`time`**: 500 kyr in 5 kyr steps, a stratigraphic layer every step.
- **`spl`, `diffusion`, `flexure`**: as in `stratigraphic_record`: rivers
  erode the highlands, the marine diffusion and the clinoform slope limit
  spread their sediment, and the margin flexes under the load.
- **`sea`**: the synthetic curve of `stratigraphic_record` (`inputs/sealevel.csv`):
  a fall from +1 to -38 m by 330 kyr with 5-15 m oscillations, then a rise to
  -20 m.
- **`climate`**: a rain map (key `rain` of the mesh file, added by
  `scripts/make_inputs.py`): 2 m/yr in two belts centred at y = 25 and 75 km,
  0.3 m/yr elsewhere. With uniform rain the rough topography drains through
  dozens of small parallel rivers that bring sediment to the whole coast; the
  wet belts make two large deltas and a starved coast between them.
- **`strata: dual`**: river sand and mud, as in the `gravity_currents`
  example; with the carbonate block the stratigraphy also records `reef`,
  `carb_sand` and `carb_mud`.
- **`carbonate`**: two factories, each with the light law of Bosscher &
  Schlager (1992), `G(w) = growth * tanh(insolation * exp(-extinction * w) /
  saturation)` at water depth `w` (`insolation`: light at the surface;
  `extinction`: how fast it fades with depth; `saturation`: the light above
  which more light no longer speeds the growth).
  - `euphotic` (reef builders, green algae): 0.5 mm/yr, in the top ~10 m
    (`extinction: 0.2`, none below 30 m), half built in place as framework.
  - `oligophotic` (foraminifera, molluscs, red algae): 0.3 mm/yr down to
    ~60 m (`extinction: 0.05`, none below 80 m), loose sand and mud only.
  - Both stop where river sediment accumulates (`sedrate`: burial, more than
    0.3 mm/yr of river sediment) and in turbid water (`turbidity`: the
    river mud in transit, spread along and off the coast over 10 km;
    production falls from full at 6e-3 m/yr to none at 1.2e-2).
    By default (`source: transport`) the plume's source is the mud the marine
    routing carried through each cell in the previous step, so the corridors
    the deltas send across the shelf are turbid; `source: mouths` would start
    it at the river mouths only, which never reaches the delta platforms here.
    The plume matters here because most river sediment crosses the shallow
    carbonate belt without settling and builds the clinoforms below it, so
    `sedrate` alone sees little of it (`input-sedrate.yml`).
- **`waves`** (wave runs): waves from the open ocean (east) losing their
  energy where the water is shallower than 20 m, over ~5 km; the euphotic
  factory works at 20 % where the energy is low.
- **`coastal_erosion`** (`input-waves.yml`): 1 mm/yr at full wave energy and
  sea level, decaying over 5 m of depth; eroded reef moved as carbonate sand
  (`rubble: carb_sand`).
- **`gravity`** (`input-gravity.yml`): the set-up of the `gravity_currents`
  example: 30 % of the river load reaching the sea, run-outs of 10 km
  (sand) and 20 km (mud), slope failure of fresh deposits.

## Lithologies: size groups and properties

The stratigraphy records five lithologies: river sand and river mud (the dual
lithology), and `carb_sand`, `carb_mud` and `reef` (added by the carbonate
block). Two separate choices define how each one behaves:

- **Its size group** (`size: coarse` or `fine`). Every carbonate lithology
  joins one of the two groups of the dual lithology: by default `carb_sand`
  and `reef` are coarse, `carb_mud` is fine. The group decides how the
  sediment moves and where it is laid: at each deposit a coarse-like
  lithology takes its share of the coarse part and a fine-like one of the
  fine part; in depressions the fine part passes the inlets
  (`pitInletBias`); in the marine spreading the coarse part settles first,
  near its source, and the fine part travels further; with gravity currents
  the fine part follows `runout_fine` and its spreading network and the
  coarse part is diverted only at `coarse_factor`. So carbonate mud travels
  with the river mud, and carbonate sand with the river sand. A factory's
  `sand` fraction splits its loose production between the two groups
  (`sand: 0.7`: 70 % `carb_sand`, 30 % `carb_mud`).
- **Its material properties**: porosity at the surface and its decay with
  burial (`phi0`, `z0`, compaction), erodibility factor (`k_factor`) and
  diffusivity factor (`diff_factor`). These are NOT inherited from the group:
  river sand and mud take them from the `strata:` block (`coarse:`, `fine:`,
  `fine_diff_factor`), each carbonate lithology from its own entry.

| Lithology | Group | phi0 | z0 (m) | k_factor | diff_factor | Set in |
|---|---|---|---|---|---|---|
| river sand | coarse | 0.49 | 3700 | 1 | 1 | `strata: coarse:` |
| river mud | fine | 0.63 | 1960 | 1.5 | 2 | `strata: fine:`, `fine_diff_factor` |
| `carb_sand` | coarse | 0.45 | 2500 | 1 | 1 | default (`carbonate: lithologies:`) |
| `carb_mud` | fine | 0.65 | 1500 | 1 | 1 | default |
| `reef` | coarse | 0.2 | 5000 | 0.1 | 0.1 | default |

The two muds therefore move as one group but compact, erode and creep a
little differently (lime mud is not siliciclastic mud). To change the
carbonate ones, for example to make the carbonate mud behave exactly like the
river mud, add to the `carbonate:` block:

```yaml
carbonate:
    lithologies:
        carb_mud: {size: fine, phi0: 0.63, z0: 1960., k_factor: 1.5, diff_factor: 2.}
        carb_sand: {size: coarse, phi0: 0.49, z0: 3700.}
```

## Running it

```bash
python scripts/make_inputs.py   # rain map and the seven input files
./run_all.sh                    # the seven runs (6 MPI ranks each, 3-5 min per run)
```

Then open `view_Results.ipynb`: the forcing, maps (elevation, carbonate
thickness, carbonate share of the deposits), the production through time,
sections across a delta and between the deltas coloured by lithology (river
sand yellow, river mud grey, carbonate sand light blue, carbonate mud dark
blue, reef pink) and the carbonate budgets.

## A 3-D view of the stratigraphy in ParaView

goSPL writes the stratigraphic layers as HDF5 (`sim_*/h5/stratal.*.h5`), which
ParaView cannot open directly. `gospl-strata-volume` (installed with goSPL,
or `python -m gospl.analyse.stratamesh`) turns them into a volume of prisms,
one per mesh triangle and layer, with an XDMF file that ParaView reads. With
`--field lithologies` every cell carries the composition of its layer.

The full time series (one volume per output, 25 steps here, ~4.5 GB for a
run of this example):

```bash
mpirun -np 5 gospl-strata-volume --h5dir sim_gravity/h5 --outdir sim_gravity/strata \
    --field lithologies --tout 2.e4
```

Only the last output (~350 MB, the largest step: it has all the layers), or a few chosen ones:

```bash
gospl-strata-volume --h5dir sim_gravity/h5 --outdir sim_gravity/strata \
    --field lithologies --tout 2.e4 --steps 25        # or --steps 5,15,25
```

- `--h5dir`: the run's `h5` folder; `--outdir`: where the volume is written
  (`strata.xdmf` plus one HDF5 file per partition and step).
- `--field lithologies`: one fraction per lithology (`frac_coarse` river
  sand, `frac_fine` river mud, `frac_carb_sand`, `frac_carb_mud`,
  `frac_reef`), the dominant lithology `dominantLith` (0 river sand, 1 river
  mud, 2 carbonate mud, 3 carbonate sand, 4 reef; the codes are printed when
  the volume is written), the `carbonateFrac` and the `porosity`. Other
  choices: `basic` (thickness, elevation, layer, porosity) and `lithology`
  (the dual coarse/fine pair only).
- `--tout 2.e4`: the output interval of the run (`time: tout`), so that
  ParaView shows the time in years; without it the time is the output index.
- `mpirun -np N` only speeds it up (the result is the same); the step
  indices are those of `sim_*/h5/gospl.<step>.p0.h5`.

In ParaView: open `strata.xdmf` (XDMF reader), colour by `dominantLith` with a
categorical colour map, or by `carbonateFrac` or one `frac_*`. A **Slice** or
**Clip** (for example a plane at y = 25 km, across a delta, or y = 50 km,
between the deltas) shows the inside, and a **Threshold** on `frac_reef`
(say above 0.5) isolates the reef bodies. With the time series, the play
button shows the margin building up.

## Building your own experiment

- *Where the rivers are*: change the rain belts in `add_rain` (positions,
  widths, rates), or give a rain map of your own.
- *Another carbonate system*: change the factories' `growth`, their depth
  range (`extinction` and the `depth` table) and their `framework` and `sand`
  fractions; add a factory, or an `exposure` response (open-water index) to
  favour the shelf edge without waves.
- *Without the turbidity control*: remove the `{control: turbidity, ...}`
  lines (as in `input-sedrate.yml`); it is off unless a factory lists it.
- *More or less sensitive to the rivers*: the `sedrate` table (`x: [0., 3.e-4]`
  stops production at 0.3 mm/yr of river sediment accumulating) and the
  `turbidity` response (its `length`, the plume decay length, and its table:
  `x: [2.e-5, 6.e-5]` on the plume's mud supply in m/yr).
- *Carbonate lithologies*: their group and properties in `carbonate:
  lithologies:` (section above).
- *Climate*: an `sst:` block and a `temperature` response make the production
  follow the sea-surface temperature through time.
- *Another sea level*: replace `inputs/sealevel.csv` (time in years, sea
  level in m, space-separated), for example with a Quaternary curve as in the
  `carbonate_terraces` example.

## References

- Bosscher, H., Schlager, W. (1992). Computer simulation of reef growth.
  Sedimentology 39, 503-512.
- Salles, T., Ding, X., Brocard, G. (2018). pyBadlands: A framework to
  simulate sediment transport, landscape dynamics and basin stratigraphic
  evolution through space and time. PLoS ONE 13(4), e0195557 (the mixed
  carbonate-siliciclastic Great Barrier Reef case).
- Hidding, J., Jarochowska, E., Hohmann, N., Liu, X., Burgess, P., Spreeuw, H.
  (2026). CarboKitten.jl, an open source toolkit for carbonate stratigraphic
  modeling. Geoscientific Model Development 19, 6207 (the factory
  parameters).
