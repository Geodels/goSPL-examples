# Mixed carbonate-siliciclastic margin (regional)

A passive margin where rivers and carbonate factories share the shelf. It
starts from the `stratigraphic_record` example (same mesh, topography and
sea-level curve) and adds goSPL's carbonate module (`carbonate:`), and in some
runs the wave energy (`waves:`), coastal erosion (`coastal_erosion:`) and
gravity currents (`gravity:`). It needs the `feat/carbonates` branch of goSPL.

## What it shows

- **Carbonates between the deltas**: two wet catchments feed two large deltas;
  the coast between them receives little river sediment and keeps a larger
  share of carbonate, and with carbonates its margin builds 13 km further out.
- **Reciprocal sedimentation**: the carbonate production follows the sea
  level. During the fall the rivers build clastic clinoforms and the
  production almost stops; during the rise carbonate covers the flooded shelf
  again and builds clinoforms of its own.
- **The `sedrate` control**: production stops where river sediment
  accumulates (turbid water, burial).
- **Waves**: reef builders that need wave energy grow on the exposed delta
  lobes, not in the sheltered, shallow coast between them; coastal erosion
  deepens that coast into an embayment.
- **Gravity currents**: part of the river load runs down the slope and builds
  a siliciclastic sheet on the basin floor.
- **Platform-scale carbonates**: at ~700 m cells a reef is narrower than a
  cell, so `reef` is the framework share of a cell's carbonate (a margin
  facies). For resolved reefs and terraces see the `carbonate_terraces`
  example (50-200 m).

## The experiments

| Input | Carbonate | `sedrate` control | Waves (reef control) | Coastal erosion | Gravity currents | Purpose |
|---|---|---|---|---|---|---|
| `input-clastic.yml` | no | | | | | siliciclastic reference |
| `input-nocontrol.yml` | yes | no | | | | carbonates that ignore the rivers |
| `input-mixed.yml` | yes | yes | no | no | no | **mixed reference** |
| `input-wavecontrol.yml` | yes | yes | yes | no | no | wave control alone |
| `input-waves.yml` | yes | yes | yes | yes (rubble: carb_sand) | no | waves and erosion |
| `input-gravity.yml` | yes | yes | no | no | yes | gravity currents |

Every input file is generated from one commented template by
`scripts/make_inputs.py`; its header says what the run tests and how it
differs from the reference.

| Run | Carbonate produced (km3): framework + loose | Carbonate share of the deposits: deltas / between |
|---|---|---|
| nocontrol | 160 + 452 | 0.37 / 0.52 |
| mixed | 132 + 345 | 0.33 / 0.45 |
| wavecontrol | 82 + 300 | 0.32 / 0.35 |
| waves | 115 + 452 | 0.42 / 0.45 |
| gravity | 120 + 313 | 0.29 / 0.42 |

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
  - Both have a `sedrate` response that stops them where more than 0.3 mm/yr
    of river sediment accumulates. The threshold is low because on this
    margin most river sediment bypasses the shallow carbonate belt and builds
    the clinoforms below it.
- **`waves`** (wave runs): waves from the open ocean (east) losing their
  energy where the water is shallower than 20 m, over ~5 km; the euphotic
  factory works at 20 % where the energy is low.
- **`coastal_erosion`** (`input-waves.yml`): 1 mm/yr at full wave energy and
  sea level, decaying over 5 m of depth; eroded reef moved as carbonate sand
  (`rubble: carb_sand`).
- **`gravity`** (`input-gravity.yml`): the set-up of the `gravity_currents`
  example: 30 % of the river load reaching the sea, run-outs of 10 km
  (sand) and 20 km (mud), slope failure of fresh deposits.

## Running it

```bash
python scripts/make_inputs.py   # rain map and the six input files
./run_all.sh                    # the six runs (6 MPI ranks each, 3-5 min per run)
```

Then open `view_Results.ipynb`: the forcing, maps (elevation, carbonate
thickness, carbonate share of the deposits), the production through time,
sections across a delta and between the deltas coloured by lithology (river
sand yellow, river mud grey, carbonate sand light blue, carbonate mud dark
blue, reef pink) and the carbonate budgets.

## Building your own experiment

- *Where the rivers are*: change the rain belts in `add_rain` (positions,
  widths, rates), or give a rain map of your own.
- *Another carbonate system*: change the factories' `growth`, their depth
  range (`extinction` and the `depth` table) and their `framework` and `sand`
  fractions; add a factory, or an `exposure` response (open-water index) to
  favour the shelf edge without waves.
- *More or less sensitive to the rivers*: the `sedrate` table (`x: [0., 3.e-4]`
  stops production at 0.3 mm/yr of river sediment).
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
