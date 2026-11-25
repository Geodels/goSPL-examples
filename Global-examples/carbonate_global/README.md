# Global carbonate system: shelves, pelagic carbonate, the CCD and the geochemistry

A global run of goSPL's carbonate module on the 250 Ma (Pangea) mesh of the
`continental_flux` example: carbonate factories on the shelves controlled by
the sea-surface temperature, pelagic carbonate preserved above the carbonate
compensation depth (CCD), hemipelagic clay, wave energy and coastal erosion
from the global wind belts, and the coupling with the continental
geochemistry of `continental_geochem` (the dissolved carbonate delivered by
the rivers limits the burial). It needs the `feat/gravity-carbonates-karst` branch of goSPL.

## Before running

The example reuses, in place, the global mesh and rainfall of
`../continental_flux` (`vars_25_80/mesh.npz`, `rain250.npz`, ~474,000 nodes,
20-100 km spacing) and, for the geochemistry run, the temperature and
climate-province maps of `../continental_geochem` (`geochem_inputs.npz`).
Build those first (their notebooks / `build_geochem_inputs.py`), then:

```bash
python scripts/make_inputs.py   # data/carb_maps.npz (SST, pelagic productivity) and the input files
./run_all.sh                    # the four runs (8 MPI ranks; 20-40 min each on a laptop)
```

## The experiments

| Input | What it adds | Question |
|---|---|---|
| `input-carb.yml` | carbonate factories (SST window), pelagic carbonate (lysocline 3500 m, CCD 4500 m), hemipelagic clay | the reference global carbonate system |
| `input-ccd_shallow.yml` | lysocline 2500 m, CCD 3500 m | how the CCD sets the pelagic carbonate kept in the basins |
| `input-waves.yml` | waves from the zonal wind belts with a 500 km fetch, a `wave` response on the shallow factory, coastal erosion | exposed and sheltered coasts |
| `input-geochem.yml` | the soil, groundwater and geochemistry of `continental_geochem`, a `solute` response and `carbonate_supply` | carbonate burial limited by the weathering supply |

Every input file is generated from one commented template by
`scripts/make_inputs.py`; its header says what the run tests and how it
differs from the reference.

## The set-up, block by block (`input-carb.yml`)

- **`domain`, `time`**: the global mesh of `continental_flux`, 1 Myr from
  250 Ma in 50 kyr steps, an output and a stratigraphic layer every 100 kyr.
- **`spl`, `diffusion`, `climate`**: as in `continental_geochem` (rivers,
  hillslopes, marine transport, per-vertex rainfall).
- **`strata: dual`**: river sand and mud, plus the carbonate lithologies
  `reef`, `carb_sand`, `carb_mud`.
- **`sst`**: a sea-surface temperature map, 29 degC at the equator falling to
  ~0 at the poles (`data/carb_maps.npz`, key `sst`). Replace it with a
  climate-model field interpolated on the mesh for a real study; the input
  files only read the map.
- **`carbonate`**: two factories with the light law of Bosscher & Schlager
  (1992), `G(w) = growth * tanh(insolation * exp(-extinction * w) /
  saturation)`. At this resolution a reef is narrower than a cell, so `reef`
  is the framework share of a cell's carbonate (a margin facies).
  - `shallow`: tropical shelves, 0.4 mm/yr, 30 % framework, only in warm
    water (`temperature` table: none below 16 or above 32 degC, full growth
    from 18 to 28 degC).
  - `deep`: cool-water and deeper loose carbonate, 0.05 mm/yr down to 400 m.
- **`pelagic`**: pelagic carbonate rain from a productivity map (`prod`,
  2 m/Myr plus 8 in the equatorial belt), fully preserved above the
  lysocline, dissolving linearly to the CCD where none is preserved, and
  less in cold water (`temperature` table); hemipelagic clay of 1 m/Myr at
  the coast decaying as `exp(-d / 300 km)` away from it.

The other runs add:

- **`waves`** (`input-waves.yml`): `from: zonal` gives the global wind belts
  (trade winds from the east within 30 degrees of the equator, westerlies to
  60, polar easterlies beyond); `fetch` (required on a sphere) the distance
  of open water over which the waves reach their full energy; `shoal_depth`
  and `shoal_length` where they lose it. The shallow factory grows less on
  sheltered coasts (`wave` table), and `coastal_erosion` abrades shallow
  water and low coasts (eroded reef moved as carbonate sand).
- **the geochemistry** (`input-geochem.yml`): the `soil`, `provenance` and
  `groundwater` blocks of `continental_geochem` (three climate provinces
  weathering iron, carbonate and silica; dissolved load routed to the ocean
  with `river_load`, accumulated in a marine reservoir with
  `marine_coupling`), and two links:
  - `carbonate_supply`: each step the factories and the pelagic rain cannot
    bury more carbonate than the reservoir of dissolved carbonate times
    `yield` (m3 of carbonate per unit of solute); what is buried is drawn
    from the reservoir. Calibrate `yield` with the `carbonate_supply` record
    of `--summary` (available, demand, scale, reservoir): here the rivers
    deliver a few 1e12 units per step and the burial would be ~1.8e14 m3,
    so `yield: 20` limits it by about half.
  - a `solute` response on the shallow factory: the dissolved carbonate
    delivered at the coast, spread offshore over 200 km, raises its growth
    near the large rivers (full growth above 1e-8, half far from them).
  - with provenance on, the carbonate gets its own provenance class after the
    three climate provinces (sediment made in the sea).

## Results

Over 1 Myr (thousand km3; the framework is part of the benthic carbonate):

| Run | Benthic carbonate (framework) | Pelagic carbonate preserved | dissolved | Hemipelagic clay |
|---|---|---|---|---|
| reference | 1,936 (282) | 1,013 | 298 | 54 |
| shallow CCD | 1,936 (282) | 697 | 614 | 54 |
| waves and coastal erosion | 2,064 (286) | 1,058 | 297 | 55 |
| coupled to the geochemistry | 1,539 (219) | 748 | 214 | 51 |

- **CCD**: raising the lysocline and the CCD by 1000 m removes 31 % of the
  pelagic carbonate kept in the basins (the dissolved part doubles); the
  shelves are not affected.
- **Waves**: open water carries the full wave energy, the coasts in the lee
  of Pangea and the Tethys gulfs are sheltered. Coastal erosion (217,000 km3
  over the run, up to ~20 m on exposed coasts, 40 m at most) reopens shallow space that the
  factories fill again, so the benthic carbonate ends 7 % higher.
- **Geochemistry**: the dissolved carbonate delivered by the rivers limits
  the burial from the first steps (scaled by 0.53-0.94, 0.72 on average)
  and the marine reservoir is drawn down (6.6e12 to ~5.4e12 units): benthic
  carbonate -21 %, pelagic -26 %.

About 3 km3/yr of carbonate burial in the reference is a few times the
present-day estimates: a warm, shelf-rich Pangea without a supply limit and
untuned factory rates. Calibrate the factory `growth` and the pelagic `prod`
map against your target before using the numbers.

**Caveats.** The 250 Ma paleo-bathymetry of `continental_flux` went down to
-25 km (closed holes and spikes, artefacts of the reconstruction): its mesh
is now cleaned before it is saved (`../continental_flux/scripts/clean_bathymetry.py`:
spikes set to their neighbourhood median, seafloor clipped at -6500 m), so the
marine routing no longer ends in kilometre-deep holes. The continental erosion
inherited from `continental_flux` is strong, though: ~42 km3/yr of rock
eroded (about five times the present-day global sediment flux), and at 20-100
km spacing the river sediment piles up at the mouths, more than 500 m thick
in ~28,000 cells and over a kilometre at the largest deltas. The carbonate
volumes above are little affected (the factories grow away from the deltas,
the pelagic rain in the open ocean), but lower the erodibility (`spl: K`) of
the continental template before using the clastic volumes.

Open `view_Results.ipynb`: the forcing maps, the reference (carbonate
production and thickness, pelagic deposition), the CCD comparison, the
waves, the supply-limited run, and the carbonate budgets of every run. The
plotting functions are in `scripts/globalcarb.py`.

## References

- Bosscher, H., Schlager, W. (1992). Computer simulation of reef growth.
  Sedimentology 39, 503-512.
- Hidding, J., Jarochowska, E., Hohmann, N., Liu, X., Burgess, P., Spreeuw, H.
  (2026). CarboKitten.jl, an open source toolkit for carbonate stratigraphic
  modeling. Geoscientific Model Development 19, 6207.
