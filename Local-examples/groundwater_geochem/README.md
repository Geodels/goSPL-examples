# Groundwater, duricrust & conservative solute geochemistry (local)

A local (flat-mesh) example illustrating goSPL's **water table**, **capillary-fringe
duricrust**, and **Level-B conservative solute geochemistry** with two tracers.
It is derived from [`../soil_generation`](../soil_generation) — same flat mesh and
soil/regolith setup — with the opt-in `groundwater:` block added.

## What it illustrates

- **Water table** — rainfall that does not run off infiltrates (`infiltration`)
  and feeds an implicit Dupuit–Boussinesq water table. Outputs: `recharge`,
  `wtable` (elevation), `wtdepth` (depth below surface), and — with
  `conserve_baseflow` — `baseflow` (seepage re-injected into the rivers).
- **Duricrust** — where the water table sits in the capillary fringe
  (`fringe_depth` ± `fringe_width`), an indurated crust precipitates and
  **armors the erodibility** (`armor_max`). Outputs: `duricrust` (thickness),
  `induration` (0–1 degree), `Karmor` (the erodibility multiplier). With
  stratigraphy on (`time: strat`), the per-layer degree is archived in
  `stratDuri`.
- **Conservative geochemistry** — two lumped tracers (`carbonate`, `silica`) are
  **dissolved** on land, **transported** down the groundwater flux, **precipitated**
  at the fringe (feeding the crust) and the remainder **exported** — a closed
  budget (`dissolved = precipitated + exported`). With `river_load` the export is
  routed **down the rivers** to the coast (`riverSolute`); with `marine_coupling`
  the delivered coastal flux is accumulated (`marineSoluteInput`). `silica` also
  has a small `river_decay` (in-transit loss). Outputs: `solute`, `soluteflux`,
  `crust_type` (dominant former), `riverSolute`, `marineSoluteInput`, plus the
  **per-species** `solute_<name>`, `soluteflux_<name>`, `crust_<name>`,
  `riverSolute_<name>`.

## Run

```bash
gospl -i input.yml -v            # serial
mpirun -np 4 gospl -i input.yml  # parallel
```

Output goes to `sim_gw_geochem/` (`h5/` + `xmf/` + the `.xdmf` time series).

## Inspect

Open `sim_gw_geochem/*.xdmf` in ParaView and colour by `wtable`, `duricrust`,
`Karmor`, `riverSolute`, or `crust_type`. To rasterise to NetCDF for PyGMT/ArcGIS
(carrying the geochem fields automatically):

```bash
gospl-grid --h5dir sim_gw_geochem/h5 --mesh data/gospl_mesh.npz:v:c \
    --step 10 --out surface10.nc
```

The per-layer duricrust archive is viewable with
`gospl-strata-volume` (the `induration` field). See the sibling
[`../soil_generation/view_Results.ipynb`](../soil_generation/view_Results.ipynb)
for the plotting pattern.

## Key input block

The whole capability is the `groundwater:` section of [`input.yml`](input.yml) —
each key is commented with its meaning and a sensible value. See the goSPL
[groundwater technical guide](https://gospl.readthedocs.io/en/latest/tech_guide/groundwater.html)
and the [input reference](https://gospl.readthedocs.io/en/latest/user_guide/surfproc.html#groundwater-duricrust)
for the full formulation and parameter tables.
