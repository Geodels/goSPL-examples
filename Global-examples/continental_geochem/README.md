# Global regional geochemistry — climate-zoned solute species (1 Myr)

A **standard global application** of the Level-B geochemistry: the continents are
divided into **climate-zone provinces** that each shed a characteristic solute
species, chemical weathering is driven by a **latitudinal + altitude temperature**
field (Arrhenius), the dissolved load is routed **down the rivers to the ocean**,
and a **capillary-fringe duricrust** forms where the water table sits at fringe
depth. The typical use is to *define regions with specific species and track how
they distribute through time*. Derived from
[`../continental_flux`](../continental_flux) — it **reuses that example's global
mesh and rainfall in place** (`../continental_flux/vars_25_80/…`), so run
`../continental_flux` at least once first (or keep its `vars_25_80/` folder).

## Setup

Build the per-vertex forcing maps once (latitudinal+altitude temperature and the
climate-zone rock provinces), then run the model:

```bash
python build_geochem_inputs.py         # writes geochem_inputs.npz (temp, rock)
mpirun -np 8 gospl -i input.yml -v     # global runs want MPI
```

Output goes to `silico_geochem/`.

## What it illustrates

- **Regional (climate-zoned) species.** Three provinces (from `build_geochem_inputs.py`,
  by latitude) each weather a characteristic species via a per-(province,species)
  `weatherability_by_class` table gathered from the provenance `source_class`:
  tropical → **iron** (ferricrete/laterite), subtropical arid → **carbonate**
  (calcrete), temperate → **silica** (silcrete). `crust_type` shows which species
  dominates the crust; `crust_source` attributes it to the province of origin —
  so you can watch the species redistribute (transport + river export) over time.
- **Temperature-driven weathering.** A latitudinal + altitude temperature map
  (`geochem_inputs.npz:temp`, °C) feeds the Arrhenius dependence of both soil
  production (`soil: activation`) and the `prodsoil` duricrust supply — warm
  tropics weather fastest, cold high latitudes slowest.
- **Water table & duricrust geomorphology.** A **thickness-based** aquifer
  (`aquifer_base: 50` m — a uniform unconfined-zone depth, *not* `from_soil`; a
  per-vertex map also works) plus an **arid recharge** keep the water table below
  the surface (median ~8 m), so the crust is localized in the capillary fringe of
  the shallow-water-table lowlands / margins rather than blanketing the continents.
- **River + ocean delivery.** `river_load` routes the per-species dissolved
  export to the coast (`riverSolute`, trapped in closed basins); `marine_coupling`
  accumulates the delivered flux (`marineSoluteInput`, `gw_solute_budget.csv`). The
  domain-integrated budget partitions each species between what is **locked in the
  duricrust** and what is **exported to the ocean**: iron is largely retained as
  ferricrete (comparable crust-bound and exported masses), whereas carbonate is
  dominantly exported (a soluble calcrete), and every species closes its dissolved
  budget (crust storage + export) to solver tolerance.

> **Recharge controls the water table.** Recharge (`infiltration × rainfall`) is
> the master knob: a wet value floods the aquifer to the surface (`wtdepth ≈ 0`
> everywhere, blanket crust), so an **arid** `infiltration` (here 0.005) is used
> to keep a deep-enough vadose zone for a fringe-localized duricrust. Interiors
> far from the coast still saturate (long drainage path); the fringe crust
> concentrates toward the drainages and margins.

## Extract the per-basin / per-species solute flux

Rasterise each step to NetCDF (this carries the geochem fields automatically),
then extract the per-basin fluxes at every river mouth:

```bash
# 1) grid a few steps (writes surface{N}.nc with FA, sedLoad, basin, riverSolute,
#    riverSolute_carbonate, riverSolute_silica, riverSolute_iron, ...)
for N in 2 5 10; do
  gospl-grid --h5dir silico_geochem/h5 \
      --mesh ../continental_flux/vars_25_80/mesh.npz:v:c \
      --step $N --out surface${N}.nc
done

# 2) per-basin water / sediment / SOLUTE outlets -> flowsed/{flow,sed,solute}{t}.csv
gospl-catchment -i inputSedFlow.csv -o flowsed
```

`flowsed/solute{t}.csv` has, for every drainage basin, its **solute outlet**
(river mouth) with columns `basin,lon,lat,val` **plus one column per species**
(`iron,carbonate,silica`) — each species' flux at that mouth (they sum to `val`).
The domain-integrated per-species budget over time is in
`silico_geochem/gw_solute_budget.csv`.
Looped over the time series, this gives the migrating **dissolved-flux-to-ocean**
per basin and per species. See the sibling
[`../continental_flux/view_Results.ipynb`](../continental_flux/view_Results.ipynb)
for the flux-map plotting pattern (the solute CSVs have the same layout as the
`flow`/`sed` ones, with the extra species columns).

## Key input block

See the `groundwater: geochem:` section of [`input.yml`](input.yml). Full
formulation: the goSPL
[groundwater technical guide](https://gospl.readthedocs.io/en/latest/tech_guide/groundwater.html).
