# Dual lithology and sediment provenance on a passive margin (local)

The passive margin of the `stratigraphic_record` example (200 x 100 km,
250 kyr of oscillating sea level), with two additions to the stratigraphic
record. The first run splits every sediment into coarse (sand) and fine
(silt and clay) fractions that compact, erode and diffuse differently
(`strata: dual`). The second run labels the sediment by the rock it was eroded
from and follows the labels through erosion, transport, deposition and the
stratigraphy (`provenance:`). Both runs use river incision (`spl:`),
hillslope and marine diffusion (`diffusion:`), marine deposition, flexure
(`flexure:`) and a sea-level curve (`sea:`).

## What it shows

- **Fines travel further**: in the dual-lithology run the fine share of the
  deposits (bulk thickness at 250 kyr) grows offshore: 0.27 between 100 and
  120 km from the western edge, 0.31 between 120 and 140 km, 0.40 between 140
  and 160 km (where the wedge is thickest, 210 m on average and up to 275 m),
  0.64 between 160 and 180 km, and 0.98 in the thin distal deposits beyond
  180 km (0.1 m on average). Coarse sediment dominates the proximal
  clinoforms.
- **Synthetic wells and sections by lithology**: the recorded fine thickness
  (`stratHf`) and porosity (`phiF`) of each layer can be drawn as a coarse
  fraction, lithology or porosity on cross-sections, wells and Wheeler
  diagrams.
- **Provenance is exact but follows the supply**: three small source areas
  (49, 71 and 84 of the 52,805 mesh nodes) are labelled 1, 2 and 3, the rest
  of the margin 0. At 250 kyr, classes 1, 2 and 3 make 0.40 %, 0.22 % and
  0.25 % of the recorded sediment, class 0 99.1 %. The record reflects what
  the rivers delivered: a source that is a small minority everywhere is best
  seen by colouring a section by its own fraction, not by the dominant class.
- **Provenance is a passive label**: it does not change the erosion,
  transport or deposition, so the provenance run evolves exactly as the same
  input file would without the block.

## The experiments

| Input | Output folder | Record | Question |
|---|---|---|---|
| `input-dual.yml` | `strati_dual` | coarse and fine fractions (`strata: dual`) | where do sand and mud end up on the margin? |
| `input-provenance.yml` | `strati_provenance` | 4 provenance classes (`provenance:`), single lithology | where does the sediment of each source area go? |

The two files are otherwise identical. The `strata` block is commented out in
`input-provenance.yml`; the two blocks can also be used together in one run.

## The set-up, block by block (`input-dual.yml`)

- **`domain`**: the mesh `inputs/gospl_mesh.npz` (keys `v`, `c`, `z`), 52,805
  nodes over 200 x 100 km (edges from 160 to 1,473 m, median 646 m). The
  initial surface is a smoothed across-margin profile (300 m at 0 km, 40 m at
  50 km, 10 m at 60 km, 0 m at 100 km, -500 m at 200 km) with Perlin-noise
  roughness (elevations from -463 to +261 m). The file also holds the source
  map `rock` used by the provenance run. `flowdir: 5` (multiple flow
  direction, up to 5 receivers), `bc: 'oooo'` (open edges).
- **`time`**: 0 to 250 kyr in 5 kyr steps, an output every 10 kyr (steps 0 to
  25), a stratigraphic layer every 5 kyr (`strat: 5.e3`, 50 layers).
- **`spl`**: `K: 4.e-5` (SI units, with drainage area in m² and rainfall in
  m/yr), `m: 0.45`, `d: 0.42`. `G` is not set (0): no river deposition on
  land.
- **`diffusion`**: `hillslopeKa: 0.1` m²/yr on land, `hillslopeKm: 0.8` m²/yr
  in the sea, `nonlinKm: 1000` m²/yr for the fresh river sediment spread on
  the seafloor, maximum clinoform slope `clinSlp: 5.e-4`. The marine diffusion
  uses goSPL's default solver, `picard` with 20 sub-steps per time step.
- **`flexure`**: `method: 'fem'`, elastic thickness 30 km, densities 2300
  (load) and 3300 kg/m³ (mantle), Young's modulus 65 GPa, Poisson ratio
  0.25; `Mirror` north and south edges, `0Slope0Shear` east and west.
- **`sea`**: the curve `inputs/sealevel.csv` (the same as in
  `stratigraphic_record`) shifted by `position: -30` m: sea level falls from
  -29 m to -46.2 m over the run, with oscillations (low of -49.6 m at
  160 kyr).
- **`climate`**: uniform rainfall of 1 m/yr.
- **`strata`**: the dual lithology.
  - `dual: True` records coarse and fine thicknesses in every layer.
  - `coarse: {phi0: 0.49, z0: 3700.}` and `fine: {phi0: 0.63, z0: 1960.}`:
    surface porosity and e-folding depth (m) of each porosity-depth curve.
  - `fine: k_factor: 1.5`: fine sediment is 1.5 times more erodible than
    coarse.
  - `fine_diff_factor: 2.0`: fine sediment diffuses twice as fast (a
    multiplier on `hillslopeKa`, `hillslopeKm` and `nonlinKm`).
  - `bedrock_coarse_frac: 0.6`: material eroded from the bedrock is 60 %
    coarse, 40 % fine.
  - `pitInletBias: {coarse: 0.5, fine: 0.0}`: in lakes and depressions coarse
    sediment builds deltas at the inlets and fine sediment settles in the
    depocentre.
- **`output`**: written to `strati_dual/`.

## The provenance block (`input-provenance.yml`)

```yaml
provenance:
    classes: 4
    source: ['inputs/gospl_mesh', 'rock']    # per-vertex source class
    cu_weight: [0.0, 1.0, 0.0, 0.3]
```

- `classes: 4`: four source classes (0 to 3).
- `source`: the class of every node, from the key `rock` of the mesh file.
  Class 1, 2 and 3 are three small areas near the western edge, between 3 and
  6 km along x and at 20 to 30 km, 50 to 60 km and 80 to 90 km along y; every
  other node is class 0. Use `uniform: 0` instead to give one class to the
  whole mesh.
- `cu_weight` (copper fertility of each class) is not read by the model: the
  weighting is applied after the run by the post-processing tool
  `gospl.analyse.provenance` (its `--cu-weights` argument), to the recorded
  compositions.

Each layer of the stratigraphy then stores the thickness of each class
(`stratP`); the classes always add up to the layer thickness.

## Running it

```bash
conda activate gospl

# optional: rebuild the mesh, the source map and the sea-level curve (inputs/)
jupyter notebook model_setup.ipynb

mpirun -np 4 gospl -i input-dual.yml
mpirun -np 4 gospl -i input-provenance.yml
```

The inputs are shipped in `inputs/`. `model_setup.ipynb` builds them as in the
`stratigraphic_record` example (profile, Perlin noise, mesh with `mpas_tools`
in `meshinputs0/`, sea-level curve) and adds the `rock` source map; the noise
is random, so running it again gives a different surface and overwrites the
shipped files.

Then open `view_Results.ipynb`. It analyses the provenance run
(`h5dir = "strati_provenance/h5"`): it rasterises every step onto a 500 m grid
(`results/surface<step>.nc`), then plots the elevation and erosion/deposition
rate at steps 5, 10, 15 and 25, the margin profile through time, the basins
and the river network of basin 98 (picked at x = 25 km, y = 43 km), and the
stratigraphy at 250 kyr: cross-sections coloured by facies and thickness, a
porosity well at (135, 50) km, three wells at 115, 135 and 145 km, and a
Wheeler diagram along x at y = 25 km.

To see the lithologies, set `h5dir = "strati_dual/h5"` and use
`color_by="coarse"` or `color_by="lithology"` in `cross_section`,
`synthetic_well` or `wheeler`; for the provenance, `color_by="provenance"` on
the `strati_provenance` run. From a terminal:

```bash
gospl-section --h5dir strati_dual/h5 --mesh inputs/gospl_mesh.npz:v:c --kind well --step 25 --xy 145000,50000 --color-by coarse --figsize 1,8 --out well_coarse.pdf
mpirun -np 4 gospl-strata-volume --h5dir strati_dual/h5 --outdir outdual --steps 25 --field lithology
mpirun -np 4 gospl-strata-volume --h5dir strati_provenance/h5 --outdir outprov --steps 25 --field provenance
```

The last two write the stratigraphic pile as a 3-D volume for ParaView.

## Building your own experiment

- *A stronger sand/mud contrast*: raise `fine_diff_factor` (fines spread
  further) or `fine: k_factor` (fines erode faster), or change
  `bedrock_coarse_frac` for a sandier or muddier source rock.
- *An initial stratigraphy with its own composition*: give an `npstrata` file
  in `domain` with the keys `strataH`, `strataZ`, `phiS` and, for the dual
  lithology, `strataHf` (fine thickness) and `phiF` (see the goSPL user guide,
  input file page).
- *Your own source map*: any integer map of the mesh nodes, with values from
  0 to `classes - 1`, saved in an `.npz` file and given as `source`. Larger or
  more erodible source areas give a stronger signal in the record; small
  outcrops give a faithful but faint one.
- *Both records at once*: uncomment the `strata` block of `input-provenance.yml`.

