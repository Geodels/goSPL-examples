# Carbonate reef terraces (regional)

Coral reef terraces built by Quaternary sea-level changes on a margin that
rises, stays put or sinks, after the numerical experiments of Pastier et al.
(2019). The example uses goSPL's carbonate module (`carbonate:`), wave energy
(`waves:`) and coastal wave erosion (`coastal_erosion:`), on the
`feat/carbonates` branch of goSPL.

## What it shows

- **Terrace staircases on an uplifting coast**: each interglacial highstand
  builds a reef at the shoreline; uplift lifts it out of the sea, and the next
  highstand builds a new one further down. The present-day profile is a
  staircase of flat reef tops with a crest at their outer edge.
- **The control of vertical motion** (Pastier's main result): stationary and
  subsiding coasts keep few emerged terraces; on a subsiding coast the reefs
  stack and the older ones drown.
- **The role of the waves**: the wave energy concentrates the reef on the
  exposed edge and shelters the lagoon behind it; the coastal erosion cuts
  cliffs and wave-cut platforms into older reefs.
- **The effect of the mesh resolution** (50, 100 and 200 m).

## The experiments

| Input | Mesh | Vertical motion | Waves (reef control) | Coastal erosion | Purpose |
|---|---|---|---|---|---|
| `input-res100.yml` | 100 m | uplift 0.3 mm/yr | yes | 2 mm/yr | reference |
| `input-res050.yml` | 50 m | uplift | yes | 2 mm/yr | resolution |
| `input-res200.yml` | 200 m | uplift | yes | 2 mm/yr | resolution |
| `input-stationary.yml` | 100 m | none | yes | 2 mm/yr | tectonics |
| `input-subsidence.yml` | 100 m | subsidence 0.3 mm/yr | yes | 2 mm/yr | tectonics, and the full-options run of the options test |
| `input-sub_reefonly.yml` | 100 m | subsidence | no | no | options: the reef factory alone |
| `input-sub_waves.yml` | 100 m | subsidence | yes | no | options: the wave control without erosion |
| `input-erosion10.yml` | 100 m | uplift | yes | 10 mm/yr | coastal erosion strength |

The options are compared on the subsiding coast, where they matter: on the
uplifting coast each reef is lifted out of the sea within one highstand, so the
wave control and the erosion change little there. Without waves the subsiding
reef fills its whole accommodation and ends as one flat platform at sea level;
with the wave control the sheltered interior grows slowly and an ~80 m deep
lagoon opens behind a barrier reef. The coastal erosion at 2 mm/yr hardly
changes the subsiding profile, while at 10 mm/yr it cuts the emerged terraces
of the uplifting coast by tens of metres.

Every input file is commented: its header says what the run tests and how it
differs from the reference.

## The set-up, block by block (`input-res100.yml`)

- **`domain`**: a cross-shore strip, 12 km long and 1.6 km wide, on a planar
  foundation dipping 5 % offshore (+200 m on the land side, -400 m offshore).
  The open ocean is to the east (the only open edge); the other edges are
  walls. The strip is uniform along its width, so it behaves like the 2-D
  profiles of Pastier et al.
- **`time`**: 400 kyr (model time 0 = 400 ka, the end = present), time step
  500 yr, a stratigraphic layer every 5 kyr.
- **`spl`, `diffusion`, `climate`**: rivers are switched off in practice (tiny
  erodibility, a little rain so the flow routing exists), as in Pastier's
  model, which has no rivers. The loose carbonate is spread by the marine
  diffusion.
- **`tectonics`**: a uniform uplift (`up`), no motion (`zero`) or subsidence
  (`down`) read from the mesh file.
- **`sea`**: the Spratt & Lisiecki (2016) sea-level stack (0-430 ka), shifted
  so that the present is 0 m (`data/sealevel.csv`, built by
  `scripts/make_inputs.py` from `data/spratt_lisiecki_2016_v2.tab`).
- **`strata: dual`**: needed for the carbonate lithologies (`reef`,
  `carb_sand`, `carb_mud`), recorded in the stratigraphy.
- **`carbonate`**: one reef-building factory. Growth follows the light law of
  Bosscher & Schlager (1992), up to 6 mm/yr; a `depth` table stops it below
  20 m; a `wave` table cuts it to 10 % where the waves do not reach (lagoons);
  80 % of the production is built in place as reef framework, the rest is
  loose carbonate sand and mud spread by the marine transport.
- **`waves`**: waves from the east, losing their energy as the water shallows
  below 10 m: the reef crest takes the energy, the lagoon behind it is
  sheltered.
- **`coastal_erosion`**: abrasion of the shallow seafloor (decaying over 3 m
  of depth) and cliff retreat down to sea level, both scaled by the wave
  energy; the reef framework erodes ten times slower than loose sediment.

## Running it

```bash
python scripts/make_inputs.py   # meshes, uplift maps, sea-level curve, input files
./run_all.sh                    # the eight runs (4 MPI ranks each)
```

Then open `view_Results.ipynb`: sea-level forcing, profiles through time,
stratigraphic sections coloured by lithology (reef in pink), and the
comparisons between runs. The plotting functions are in `scripts/terraces.py`.

## Building your own experiment

- *Another coast*: change `SLOPE`, `ZTOP`, `LX` or the uplift `RATE` in
  `scripts/make_inputs.py` (Pastier et al. explore uplift rates from -2 to
  +2 mm/yr and foundation slopes of a few percent).
- *Another reef*: change `growth`, the `depth` table (the depth range of your
  reef builders), `framework` (more framework, more resistant reefs) or add a
  second factory (for example a deeper `oligophotic` one with `framework: 0`).
- *More or less wave action*: `shoal_depth` and `shoal_length` set where the
  waves lose their energy; `coastal_erosion: rate` how fast they cut cliffs.
- *Climate*: add an `sst:` block and a `temperature` response to the factory
  to make the reef growth follow the sea-surface temperature through glacial
  cycles.

## References

- Pastier, A.-M., Husson, L., Pedoja, K., Bézos, A., Authemayou, C.,
  Arias-Ruiz, C., Cahyarini, S.Y. (2019). Genesis and architecture of sequences
  of Quaternary coral reef terraces: insights from numerical models.
  Geochemistry, Geophysics, Geosystems 20, 4248-4272.
- Spratt, R.M., Lisiecki, L.E. (2016). A Late Pleistocene sea level stack.
  Climate of the Past 12, 1079-1092. Data: PANGAEA,
  https://doi.org/10.1594/PANGAEA.979830 (CC-BY-3.0).
- Bosscher, H., Schlager, W. (1992). Computer simulation of reef growth.
  Sedimentology 39, 503-512.
