# Carbonate atoll: wave direction on a sinking volcanic island (local)

A volcanic island sinks while reef builders grow around it: Darwin's sequence
from a fringing reef to a barrier reef and an atoll. The reef builders need
wave energy, so the wave climate decides where the reef rim grows. The example
uses goSPL's carbonate module (`carbonate:`), wave energy (`waves:`), the
open-water index (`exposure`) and coastal erosion (`coastal_erosion:`), on
the `feat/gravity-carbonates-karst` branch of goSPL.

## What it shows

- **Darwin's sequence**: with waves from every direction, a reef rim grows
  around the sinking island and keeps up with the subsidence; once the summit
  drowns (~300 kyr) a lagoon opens behind the rim.
- **The control of the wave direction**: with one prevailing direction (trade
  winds from the east) only the windward side and the flanks get wave energy;
  the island shadows its lee, the leeward margin has no reef and drowns, and
  the result is a windward crescent reef in front of a sand platform. Two
  directions give reef rims facing both swells.
- **What refraction and coastal erosion change**: refraction turns the waves
  towards the shore on the flanks but cannot reach the shadow (the wave model
  has no diffraction); coastal erosion turns reef into sand that fills part of
  the lagoon.
- **Drowning under glacial sea level**: the Quaternary sea level (Spratt &
  Lisiecki 2016) moves the reef down the flanks at the lowstands, and the last
  deglacial rise drowns it: a drowned atoll.

## The experiments

| Input | Waves | Refraction | Coastal erosion | Sea level | Result (600 kyr) |
|---|---|---|---|---|---|
| `input-nowaves.yml` | none | | | constant | flat platform at sea level, no lagoon |
| `input-ring.yml` | 8 directions, equal | no | no | constant | reef rim all around, lagoon 18 m deep at the centre |
| `input-east.yml` | east (trade winds) | no | no | constant | windward crescent reef, sand platform, leeward margin drowned (to -106 m) |
| `input-season.yml` | east 70 %, SSW 30 % | no | no | constant | rims facing both swells, interior kept up, north-west margin lagging |
| `input-refraction.yml` | east | yes | no | constant | as `east` |
| `input-erosion.yml` | east | no | yes (rubble) | constant | as `east`, the drowned lee half filled with sand (to -58 m) |
| `input-sealevel.yml` | east | no | no | Spratt & Lisiecki | drowned atoll (highest point -67 m) |

Every input file is generated from one commented template by
`scripts/make_inputs.py`; its header says what the run tests and how it
differs from the others.

## The set-up, block by block (`input-east.yml`)

- **`domain`**: a 14 x 14 km square at 100 m spacing, a lattice of equilateral
  triangles (no preferred direction, so any asymmetry comes from the waves),
  open ocean on every side. The island is a cone: +150 m at the centre, flanks
  of 7.5 %, shoreline 2 km from the centre (`data/island.npz`).
- **`time`**: 600 kyr in 1 kyr steps, a stratigraphic layer every 10 kyr.
- **`spl`, `diffusion`, `climate`**: 1 m/yr of rain; rivers and creep erode the
  island slowly, and their volcanic sand and mud reach the sea.
- **`tectonics`**: uniform subsidence of 0.5 mm/yr (key `sub` of the mesh
  file): the island sinks 300 m and its summit drowns after ~300 kyr.
- **`sea`**: constant at 0 m (`input-sealevel.yml`: the curve in
  `data/sealevel.csv`, built from the long stack of Spratt & Lisiecki 2016,
  0-600 ka, shifted to 0 m at present).
- **`strata: dual`**: volcanic sand and mud, plus the carbonate lithologies
  (`reef`, `carb_sand`, `carb_mud`).
- **`carbonate`**: two factories with the light law of Bosscher & Schlager
  (1992), `G(w) = growth * tanh(insolation * exp(-extinction * w) /
  saturation)` at water depth `w`.
  - `reef`: 4 mm/yr, none below 25 m, 60 % built in place as framework,
    the rest loose sand and mud. A `wave` response (2 % of the growth where
    the wave energy is 0, full growth above 0.5) and an `exposure` response
    (the open-water index over 300 m: full growth on rims and highs, 5 % where
    the seafloor is not above its surroundings, as on a lagoon floor).
  - `lagoon`: loose sand and mud, 0.4 mm/yr down to 50 m, everywhere.
  - A lagoon opens only where the sheltered interior produces less than the
    subsidence (here 0.4 mm/yr plus a few percent of the reef growth against
    0.5 mm/yr). A control on the wave energy alone does not do it: a deeper
    lagoon dissipates less, more energy reaches its floor, the reef grows
    faster there and keeps up (the lagoon stayed ~8 m deep in every such
    test). The open-water index caps the reef on a lagoon floor whatever the
    waves bring, so the lagoon can deepen. Its length must be shorter than
    the lagoon: over 1 km it reaches the steep flanks of this small atoll and
    the whole atoll looks like a margin.
- **`waves`**: the wave climate (one or several directions with weights);
  the waves lose their energy where the water is shallower than 10 m, over
  ~300 m. Land and shallow reef flats cast a shadow downwind.
- **`coastal_erosion`** (`input-erosion.yml`): 2 mm/yr at full wave energy and
  sea level, decaying over 3 m of depth; eroded reef moved as carbonate sand
  (`rubble: carb_sand`).

## Running it

```bash
python scripts/make_inputs.py   # island mesh, sea-level curve and the seven input files
./run_all.sh                    # the seven runs (4 MPI ranks each, 3-5 min per run)
```

Then open `view_Results.ipynb`: the island through time, maps of the wave
energy, reef thickness and elevation of every run, and sections west-east and
south-north coloured by lithology (reef pink). The plotting functions are in
`scripts/atoll.py`. A 3-D view of the stratigraphy in ParaView:
`gospl-strata-volume --h5dir sim_east/h5 --outdir sim_east/strata --field
lithologies --tout 2.e4` (see the `mixed_carbonate_clastic` README).

## Building your own experiment

- *Another wave climate*: change the `climate` list (`from` in degrees, 0 =
  north, 90 = east; `weight`), or give a bearing per node with `from_map`.
- *Another island*: `ZTOP`, `SLOPE` and `SUB` in `scripts/make_inputs.py` (a
  higher island or slower subsidence delays the atoll stage).
- *A deeper or shallower lagoon*: the lagoon deepens at the subsidence rate
  minus what its floor produces (`lagoon: growth` plus the reef share allowed
  by the first `f` of the `exposure` table) and minus the sand shed from the
  rim; `exposure_length` should be a few cells, shorter than the lagoon.
- *Reefs that keep up with the glacial sea level*: start on a shallower
  platform, or widen the reef's depth window (the `depth` table); a faster
  reef alone does not help (10 mm/yr still drowns at -43 m), because the reef
  builders stop below 25 m and the deglacial rise outpaces them there.

## References

- Darwin, C. (1842). The Structure and Distribution of Coral Reefs. Smith,
  Elder and Co., London.
- Bosscher, H., Schlager, W. (1992). Computer simulation of reef growth.
  Sedimentology 39, 503-512.
- Spratt, R.M., Lisiecki, L.E. (2016). A Late Pleistocene sea level stack.
  Climate of the Past 12, 1079-1092. Data: PANGAEA,
  https://doi.org/10.1594/PANGAEA.979830 (CC-BY-3.0).
