# Tropical ferricrete & relief inversion (local)

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

- **Stage 1 — ferricrete forms in the wet valleys (0–5 Myr, wet, tectonically quiet).**
  The `tectonics` `t1` phase applies only a whisper of uplift, so the plain stays low.
  The water table sits **near the surface in the wet valley floors and deep under the
  rises**, so the capillary-fringe iron duricrust indurates the **valley ground as a
  broad, resistant sheet** and *not* the drier rises between the valleys
  (`duricrust`, `induration`, `Karmor`). This is the valley-shallow / ridge-deep
  water-table structure that localises ferricrete in river valleys.

- **Stage 2 — dissection inverts the relief (5–15 Myr, drier, base-level fall).**
  The `tectonics` `t2` phase uplifts the interior against the fixed southern outlet, so
  the rivers incise. Where incision **breaches the ferricrete it cuts quickly into the
  soft saprolite beneath** (the ferricrete armours the bedrock erodibility, `armor_max`)
  and carves new valleys, while the **ferricrete-capped ground resists and is left
  standing as ferricrete-capped mesas**. The former valley floors become the highs: a
  relief inversion.

In the run the initial valley floors erode only about **a third as fast as the bare
rises** while the crust armours them (a valley-to-ridge erosion ratio near 0.3), the
relief grows from a few tens of metres to **over 500 m**, and the ferricrete that formed
in the lows ends up on the **highest ground** (its mean elevation percentile climbs from
~50 to ~77 %).

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
  drops away under the interfluves — the gradient the capillary-fringe crust needs.
- **Capillary-fringe iron duricrust** that grows where the water table is shallow, is
  archived per stratigraphic layer (`stratDuri`/`induration`) and re-arms on exhumation,
  and **armours the bedrock erodibility** (`Karmor`) to drive the inversion.
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

The inversion is clearest as `duricrust` through time (a broad valley-floor sheet early,
then caps on the dissected interfluves) alongside `elev` and `erodep` (the deepening,
un-armoured valleys). Open `sim_ferricrete/*.xdmf` in ParaView and colour by `duricrust`,
`Karmor`, `elev` or `wtdepth`.

## Key input blocks

The scenario is set by the two-phase `tectonics` block (`t1` quiet, `t2` base-level
fall), the two-phase `climate` block (wet → drier), the `bc: 'owow'` boundary, and the
`groundwater:` section of [`input.yml`](input.yml). Full formulation: the goSPL
[groundwater technical guide](https://gospl.readthedocs.io/en/latest/tech_guide/groundwater.html).
