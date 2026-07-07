# Ferricrete cuirasse dissection & relief inversion (local)

A local example of goSPL's **water table**, **capillary-fringe iron duricrust
(ferricrete)** and **conservative solute geochemistry** that reproduces the textbook
tropical **laterite-mesa relief inversion** in two stages, on a deliberately simple,
controlled landscape (`inputs/gospl_mesh_valleys.npz`): a low-relief plain (~50×50 km)
crossed by broad river valleys that drain south to a pinned outlet. **No crust is
prescribed — it grows in place.**

Build the mesh, then run (about four minutes on 4–8 CPUs for 15 Myr):

```bash
python build_inputs.py
mpirun -np 8 gospl -i input.yml -v
```

Output goes to `sim_ferricrete/`.

## The two-stage story

- **Stage 1 — a ferricrete cuirasse blankets the wet plain (0–5 Myr, wet, tectonically
  quiet).** The `tectonics` `t1` phase applies only a whisper of uplift, so the plain
  stays low and wet. The water table sits **near the surface across the plain**, so the
  capillary-fringe iron duricrust indurates the near-surface ground as a **broad,
  resistant sheet** (`duricrust`, `induration`, `Karmor`). This is the *in-situ /
  relative-accumulation* style of laterite — the plateau ("bowal") cuirasse that
  genuinely blankets flat, low-relief laterite uplands.

- **Stage 2 — dissection inverts the relief (5–15 Myr, drier, base-level fall).**
  The `tectonics` `t2` phase uplifts the interior against the fixed southern outlet, so
  the rivers incise. Where incision **breaches the cuirasse it cuts quickly into the soft
  saprolite beneath** (the ferricrete armours the bedrock erodibility, `armor_max`, and
  the hillslope creep, `armor_diffusion`) and carves deep valleys, while the
  **ferricrete-capped divides resist and are left standing as flat-topped mesas**. The
  armoured old land surface becomes the high ground: a relief inversion.

In the run the crust blankets the plain by the end of Stage 1, then Stage 2 strips it
from the incising valleys and preserves it on the divides: the relief grows from a few
tens of metres to **~640 m**, and the surviving ferricrete ends up firmly on the
**highest ground** (the crust–elevation correlation climbs to ~+0.7, with the capped
mesas ~20 m thick standing over deeply stripped valleys).

> **Two ferricrete styles, two examples.** This example is the *in-situ* cuirasse that
> blankets the plain and dissects into mesas. Its lateral counterpart —
> **absolute-accumulation valley ferricrete**, where dissolved iron is carried
> downslope and precipitates only in the groundwater discharge zones (valley floors,
> footslopes) via the `discharge_gate` — is the sister
> [`valley_ferricrete`](../valley_ferricrete) example. A rate-only armour model cannot
> stand a *valley-fill* cap up into a ridge (the crust adds no elevation), so relief
> inversion here is the dissected-cuirasse → capped-mesa pathway, where the crust starts
> on the ground that becomes the high divides.

## Why the boundary matters

The valleys run N–S and drain **south**, so the model must not leak flow out of the east
and west sides. Two choices in the setup remove that edge effect:

1. `build_inputs.py` phases the corrugation so **ridge crests sit exactly on the E/W
   edges** (`L / λ = 4`), making the edges natural drainage divides; and
2. `input.yml` sets `bc: 'owow'` (NESW) — **walls on E and W**, with N/S open so the
   landscape drains south to the pinned outlet.

The uplift fields `t1`/`t2` are also tapered to zero at the southern outlet, so it stays
a fixed base level while the interior is uplifted — this keeps the landscape in relief
(a dissecting, incising landscape) rather than grading to a peneplain.

## What it illustrates

- **Water table** that seeps at the channels (`wtdepth ≈ 0`, feeding `baseflow`) and
  drops away under the interfluves.
- **Capillary-fringe iron duricrust** that grows where the water table is shallow, is
  archived per stratigraphic layer (`stratDuri`/`induration`) and re-arms on exhumation,
  and **armours the bedrock erodibility** (`Karmor`) — and, with `armor_diffusion`, the
  hillslope creep — to drive the inversion.
- **Iron geochemistry**: a single conservative `iron` tracer is dissolved, advected in
  the groundwater, precipitated in the fringe (feeding the crust) and its seepage export
  routed down the rivers (`river_load`).

## Inspect

Rasterise to NetCDF (carries the geochem fields automatically):

```bash
for s in $(seq 0 60); do
  gospl-grid --h5dir sim_ferricrete/h5 --mesh inputs/gospl_mesh_valleys.npz:v:c \
      --step $s --spacing 300 --out results/surface$s.nc
done
```

The inversion is clearest as `duricrust` through time (a broad sheet early, then caps on
the dissected interfluves) alongside `elev` and `erodep` (the deepening, un-armoured
valleys). Open `sim_ferricrete/*.xdmf` in ParaView and colour by `duricrust`, `Karmor`,
`elev` or `wtdepth`.

## Key input blocks

The scenario is set by the two-phase `tectonics` block (`t1` quiet, `t2` base-level
fall), the two-phase `climate` block (wet → drier), the `bc: 'owow'` boundary, and the
`groundwater:` section of [`input.yml`](input.yml). Full formulation: the goSPL
[groundwater technical guide](https://gospl.readthedocs.io/en/latest/tech_guide/groundwater.html).
