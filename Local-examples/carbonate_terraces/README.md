# Carbonate reef terraces (regional)

Coral reef terraces built by Quaternary sea-level changes on a margin that
rises, stays put or sinks, after the numerical experiments of Pastier et al.
(2019). The example uses goSPL's carbonate module (`carbonate:`), wave energy
(`waves:`) and coastal wave erosion (`coastal_erosion:`), on the
`feat/gravity-carbonates-karst` branch of goSPL.

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
| `input-sub_reefonly.yml` | 100 m | subsidence | no | no | options: the factories alone |
| `input-sub_waves.yml` | 100 m | subsidence | yes | no | options: the wave control without erosion |
| `input-erosion10.yml` | 100 m | uplift | yes | 10 mm/yr | coastal erosion strength |
| `input-sub_erosion10.yml` | 100 m | subsidence | yes | 10 mm/yr | coastal erosion strength, subsiding coast |

The options are compared on the subsiding coast, where they matter: on the
uplifting coast each reef is lifted out of the sea within one highstand, so the
wave control and the erosion change little there. Without waves the subsiding
reef fills its whole accommodation and ends as one flat platform at sea level;
with the wave control the sheltered interior grows slowly and an ~80 m deep
lagoon, floored by platform sand and mud, opens behind a barrier reef. The coastal erosion at 2 mm/yr hardly
changes the subsiding profile (the reef grows back faster than it is cut, and
no old reef stays above sea level to be cut as a cliff); at 10 mm/yr its
rubble builds a carbonate sand apron that moves the subsiding platform
margin 500 m seaward. On the uplifting coast 10 mm/yr cuts the emerged terraces
of the uplifting coast, and its rubble (carbonate sand) builds the fore-reef
out.

The mesh is a lattice of equilateral triangles, symmetric about the centre
line (`scripts/make_inputs.py`, `strip`). Do not split a square lattice along
one diagonal: each node then has neighbours along that diagonal only, and the
marine routing piles the loose sediment against one side wall.

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
- **`carbonate`**: two factories, both following the light law of Bosscher &
  Schlager (1992). The `reef` factory grows up to 6 mm/yr; a `depth` table
  stops it below 20 m; a `wave` table cuts it to 10 % where the waves do not
  reach (lagoons); 80 % of its production is built in place as reef
  framework, the rest is loose carbonate sand and mud. The `platform` factory
  (algae, foraminifera, molluscs) grows 1 mm/yr down to 60 m, less dependent
  on light and without wave control, and makes only loose sediment (half
  sand, half mud): it fills the fore-reef slope and the lagoon floors. The
  loose carbonate is spread by the marine transport.

  The light law of each factory is `G(w) = growth * tanh(insolation *
  exp(-extinction * w) / saturation)` at water depth `w`: `insolation` is the
  light at the sea surface (W/m2), `extinction` how fast the light fades with
  depth (1/m, Beer-Lambert; clear tropical water 0.04 to 0.16), and
  `saturation` the light above which more light no longer speeds the growth
  (W/m2). Growth falls to half its potential where `insolation *
  exp(-extinction * w) / saturation` is about 0.55: here near 25 m for the reef
  (`extinction: 0.1`) and 50 m for the platform factory (0.05). The `depth`
  tables then stop the production (below 20 m and 60 m).
- **`waves`**: waves from the east, losing their energy as the water shallows
  below 10 m: the reef crest takes the energy, the lagoon behind it is
  sheltered.
- **`coastal_erosion`**: abrasion of the shallow seafloor (decaying over 3 m
  of depth) and cliff retreat down to sea level, both scaled by the wave
  energy; the reef framework erodes ten times slower than loose sediment.
  With `rubble: carb_sand` the eroded reef is moved and deposited as carbonate
  sand (reef rubble), not as pieces of reef.

## Running it

```bash
python scripts/make_inputs.py   # meshes, uplift maps, sea-level curve, input files
./run_all.sh                    # the nine runs (4 MPI ranks each)
```

Then open `view_Results.ipynb`: sea-level forcing, profiles through time,
stratigraphic sections coloured by lithology (reef in pink), and the
comparisons between runs. The plotting functions are in `scripts/terraces.py`.

## Building your own experiment

- *Another coast*: change `SLOPE`, `ZTOP`, `LX` or the uplift `RATE` in
  `scripts/make_inputs.py` (Pastier et al. explore uplift rates from -2 to
  +2 mm/yr and foundation slopes of a few percent).
- *Another reef*: change `growth`, the `depth` table (the depth range of your
  reef builders) or `framework` (more framework, more resistant reefs).
- *More or less loose carbonate*: lower the reef `framework`, or change the
  `platform` factory (`growth`, its depth range, `sand` for the sand/mud
  split). Remove `rubble` to keep eroded reef as reef.
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
