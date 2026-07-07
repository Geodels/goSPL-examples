# Valley ferricrete (absolute accumulation) & dissolved-iron export (local)

A local example of goSPL's **absolute-accumulation duricrust gate** — the
`discharge_gate` — on the same low-relief plain and river valleys as the sister
[`ferricrete_inversion`](../ferricrete_inversion) example (`inputs/gospl_mesh_valleys.npz`,
built by the identical `build_inputs.py`). Here the iron ferricrete forms by **lateral
(absolute) accumulation**: dissolved iron is transported by the groundwater flux and
re-precipitates **only where that flow converges and discharges** — the valley floors,
footslopes and seepage faces — so the crust tracks the drainage network instead of
blanketing the plain, and the un-precipitated iron is exported down the rivers.

Build the mesh, then run (about four minutes on 4–8 CPUs for 15 Myr):

```bash
python build_inputs.py
mpirun -np 8 gospl -i input.yml -v
```

Output goes to `sim_valley_ferricrete/`.

## Two styles of ferricrete

Laterite / ferricrete forms by two well-documented pathways, and goSPL can represent both
through the one `duricrust:` block:

| | `discharge_gate` | where the crust forms | example |
|---|---|---|---|
| **Relative accumulation** (in-situ) | `False` (default) | wherever the water table is shallow — the plateau / "bowal" cuirasse blanketing flat uplands | [`ferricrete_inversion`](../ferricrete_inversion) |
| **Absolute accumulation** (lateral) | `True` | only in groundwater **discharge zones** (valley floors, footslopes, seepage faces) | **this example** |

With the gate on, the capillary-fringe favourability `Φ` is multiplied by a discharge
weight

```
G = (−∇·q)⁺ / ((−∇·q)⁺ + R)   ∈ [0,1]
```

the fraction of a cell's upward discharge that was **imported by lateral convergence** of
the groundwater flux `q = −T∇h` (rather than supplied by the local recharge `R`). `G → 1`
at a convergent valley floor or seepage face and `G → 0` at a divergent recharge rise, so
the crust is confined to the valleys — **valley / "bas-fond" ferricrete**. Both terms are
area-normalised rates, so `G` is dimensionless and needs no tuning constant.

## What it illustrates

- **Absolute-accumulation localisation.** Compared with the gate-off cuirasse (which
  blankets ~100 % of the plain), the crust here stays confined to the drainage network —
  peak coverage around **20–25 %**, with the valley floors carrying a thick crust and the
  ridges essentially bare (`duricrust`, `induration`).
- **The dissolved-iron cycle.** A single conservative `iron` tracer is dissolved in the
  wet uplands, advected down the groundwater flux, precipitated at the valley discharge
  zones (feeding the crust) and the remainder **exported down the rivers**
  (`soluteflux`, `riverSolute`, `river_load: True`). With a finite `source_pool` the
  weathering is a **wet-phase pulse** that tapers as the reservoir empties (a one-time
  `source pool exhausting` note is printed); raise `source_pool` to keep it rate-limited.
- **No relief inversion.** Absolute-accumulation crust follows the active drainage, so it
  stays in the topographic lows — this example is about *where* the crust forms and the
  solute budget, not about standing mesas (see `ferricrete_inversion` for the inversion).

## Inspect

```bash
for s in $(seq 0 60); do
  gospl-grid --h5dir sim_valley_ferricrete/h5 --mesh inputs/gospl_mesh_valleys.npz:v:c \
      --step $s --spacing 300 --out results/surface$s.nc
done
```

Open `sim_valley_ferricrete/*.xdmf` in ParaView and colour by `duricrust` (confined to the
valleys), `soluteflux` / `riverSolute` (the exported dissolved iron), `wtdepth` and
`elev`. The per-species solute budget is written to
`sim_valley_ferricrete/gw_solute_budget.csv` (dissolved / precipitated / ocean-flux).

## Key input blocks

The crust localisation is set by `duricrust: discharge_gate: True` and the finite
`geochem: species: source_pool`; everything else (mesh, `bc: 'owow'`, two-phase
`climate` / `tectonics`) matches the `ferricrete_inversion` example. Full formulation:
the goSPL
[groundwater technical guide](https://gospl.readthedocs.io/en/latest/tech_guide/groundwater.html).
