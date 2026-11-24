"""
Build the inputs of the carbonate_terraces example.

- data/mesh{50,100,200}.npz: a cross-shore strip (12 km x 1.6 km) at 50, 100 and
  200 m spacing, an equilateral triangle lattice (see strip), with
    v      vertex coordinates (m), c  triangles,
    z      initial elevation: a planar foundation dipping 5 % offshore, from
           +200 m on the land side (x = 0) to -400 m offshore (x = 12 km),
    up     uplift rate +0.3 mm/yr (m/yr), zero / down: 0 and -0.3 mm/yr.
- data/sealevel.csv: relative sea level (time in model years, elevation in m)
  for a 400 kyr run (model time 0 = 400 ka), from the Spratt & Lisiecki (2016)
  sea-level stack (short PC1, 0-430 ka, 1 kyr spacing), shifted so that the
  present is 0 m (the stack reads +8.5 m at 0 ka, an artefact of the
  stacking; with the shift the last interglacial highstand is about +7.5 m).

Data: Spratt, R.M., Lisiecki, L.E. (2016) A Late Pleistocene sea level stack.
Climate of the Past 12, 1079-1092; dataset version 2, PANGAEA,
https://doi.org/10.1594/PANGAEA.979830 (CC-BY-3.0), in data/.

Run from the example folder: python scripts/make_inputs.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LX, LY = 12000.0, 1600.0
SLOPE, ZTOP = 0.05, 200.0
RATE = 0.3e-3                       # m/yr
DURATION = 400.0e3                  # yr


def strip(dx):
    """Equilateral triangles: rows of nodes ``dx`` apart, every other row
    shifted by dx/2, plus the nodes on the land and ocean edges, triangulated
    by Delaunay. Do not split a square lattice along one diagonal: each node
    then has neighbours along that diagonal only, and the marine routing of
    the loose sediment leans towards it (the deposits pile up against one
    side wall). The number of row gaps is even, so the lattice is symmetric
    about the centre line y = LY / 2."""
    from scipy.spatial import Delaunay

    nrow = 2 * int(round(LY / (dx * np.sqrt(3.0)))) + 1
    ys = np.linspace(0.0, LY, nrow)
    pts = []
    for j, y in enumerate(ys):
        x = np.arange(0.0, LX + 1.0, dx)
        if j % 2:
            x = np.concatenate([[0.0], x[:-1] + 0.5 * dx, [LX]])
        pts.append(np.column_stack([x, np.full(x.size, y)]))
    xy = np.vstack(pts)
    cells = Delaunay(xy).simplices
    x = xy[:, 0]
    z = ZTOP - SLOPE * x
    n = x.size
    np.savez(os.path.join(HERE, "data", "mesh%d" % int(dx)),
             v=np.column_stack([xy, np.zeros(n)]), c=cells.astype(np.int64),
             z=z, up=np.full(n, RATE), zero=np.zeros(n), down=np.full(n, -RATE))
    return n


def sealevel():
    txt = open(os.path.join(HERE, "data", "spratt_lisiecki_2016_v2.tab")).read()
    rows = [r.split("\t") for r in txt.split("*/\n")[1].strip().split("\n")[1:]]
    age = np.array([float(r[0]) for r in rows])               # ka BP
    sl = np.array([float(r[1]) if r[1] else np.nan for r in rows])
    ok = np.isfinite(sl) & (age <= DURATION / 1e3)
    age, sl = age[ok], sl[ok]
    sl = sl - sl[age == 0.0][0]                               # present = 0 m
    t = DURATION - age * 1.0e3                                # model time (yr)
    order = np.argsort(t)
    np.savetxt(os.path.join(HERE, "data", "sealevel.csv"),
               np.column_stack([t[order], sl[order]]), delimiter=",", fmt="%.1f,%.3f")
    return sl.min(), sl.max()




# ---------------------------------------------------------------------------
# Input files. One template, fully commented; each run changes a few keys.
# ---------------------------------------------------------------------------
TEMPLATE = """\
# =============================================================================
# carbonate_terraces: {title}
#
# {purpose}
#
# Reference run: input-res100.yml (100 m, uplift 0.3 mm/yr, reef factory with
# the wave control, coastal wave erosion). This run differs by: {diff}
# =============================================================================
name: carbonate terraces, {title}

domain:
    npdata: ['data/mesh{res}','v','c','z']   # strip 12 x 1.6 km at {res} m; planar foundation dipping 5 % offshore
    bc: 'woww'          # edges N,E,S,W: walls, except the open ocean to the east (o)
    flowdir: 1          # single flow direction (no rivers to speak of here)

time:
    start: 0.           # model time 0 = 400 ka
    end: 4.e5           # 400 kyr: MIS 11 to the present
    tout: 1.e4          # output every 10 kyr
    dt: 500.            # time step (the carbonate growth does not depend on it)
    strat: 5.e3         # a stratigraphic layer every 5 kyr (80 layers)

spl:
    K: 1.e-12           # river erosion switched off in practice: like Pastier et al.
    d: 0.               #   (2019), the land is not eroded by rivers here
    m: 0.5

diffusion:
    hillslopeKa: 0.     # no creep on land
    hillslopeKm: 0.01   # marine diffusion of the loose carbonate sediment (m2/yr)
    nonlinKm: 50.       # transport of fresh marine deposits (m2/yr)

tectonics:
  - start: 0.
    end: 4.e5
    upsub: ['data/mesh{res}','{tecto}']   # {tecto_text}

sea:
    position: 0.
    curve: 'data/sealevel.csv'   # Spratt & Lisiecki (2016), shifted to 0 m at present

climate:
  - start: 0.
    uniform: 0.1        # a little rain so that the flow routing is defined

strata:
    dual: True          # needed by the carbonate lithologies (reef, carb_sand, carb_mud)

# -----------------------------------------------------------------------------
# Carbonate production: two factories (Bosscher & Schlager light law each).
# - reef: the reef builders, potential growth 6 mm/yr (Pastier et al. 2019 use
#   a potential rate of 10 mm/yr whose effective value stays below ~7 mm/yr),
#   production stopping below 20 m, 80 % built in place as reef framework, the
#   rest loose sand (80 %) and mud.
# - platform: the loose-sediment producers of the lagoon and the fore-reef
#   (calcareous algae, foraminifera, molluscs), slower (1 mm/yr), less
#   dependent on light, down to 60 m, no framework: half sand, half mud.
#   Without wave control, so they also thrive in sheltered water.
# -----------------------------------------------------------------------------
carbonate:
    insolation: 400.
    factories:
      - name: reef
        growth: 6000.          # m/Myr (6 mm/yr)
        extinction: 0.1        # 1/m: light falls with depth
        saturation: 60.        # W/m2
        framework: 0.8         # built in place (lithology `reef`)
        sand: 0.8              # loose part: 80 % sand, 20 % mud
        response:
          - {{control: depth, x: [0., 15., 20.], f: [1., 1., 0.]}}   # no production below 20 m
{wave_response}
      - name: platform
        growth: 1000.          # m/Myr (1 mm/yr)
        extinction: 0.05       # 1/m: less light-dependent than the reef
        saturation: 60.        # W/m2
        framework: 0.          # all loose sediment
        sand: 0.5              # half carbonate sand, half carbonate mud
        response:
          - {{control: depth, x: [0., 40., 60.], f: [1., 1., 0.]}}   # no production below 60 m
{waves}{coast}
output:
    dir: 'sim_{name}'
"""

WAVE_RESP = ("          - {control: wave, x: [0., 0.5], f: [0.1, 1.]}   "
             "# reef growth needs wave energy: sheltered lagoons grow less")
WAVES = """
# -----------------------------------------------------------------------------
# Waves from the open ocean (east), dissipated as the water shallows below
# 10 m: the reef crest takes the energy, the lagoon behind it is sheltered.
# -----------------------------------------------------------------------------
waves:
    climate:
      - {from: 90., weight: 1.}   # waves come from the east (the open ocean)
    decay: 1.e6                   # no attenuation in deep water across the domain
    shoal_depth: 10.              # the waves feel the bottom shallower than 10 m
    shoal_length: 500.            # and lose their energy over ~500 m at the shore
"""
COAST = """
# -----------------------------------------------------------------------------
# Coastal wave erosion (Pastier et al. 2019): the seafloor abraded down to a
# few metres, cliffs cut back to sea level; the reef framework erodes ten
# times slower (its k_factor is 0.1). The eroded reef is broken into rubble:
# it is moved and deposited as carbonate sand.
# -----------------------------------------------------------------------------
coastal_erosion:
    rate: {erate}          # m/yr at full wave energy, at sea level
    depth_scale: 3.      # abrasion decays as exp(-depth / 3 m)
    cliff_height: 15.    # land up to 15 m above sea level next to the sea is a cliff
    max_erosion: 2.      # at most 2 m per step
    rubble: carb_sand    # eroded reef framework is routed as carbonate sand
"""

RUNS = {
    # name: (res, tecto, waves, coast rate (None = off), title, purpose, diff)
    "res100": (100, "up", True, "2.e-3", "reference, 100 m",
               "Reef terraces on an uplifting coast: the reference configuration.",
               "nothing (this is the reference)."),
    "res050": (50, "up", True, "2.e-3", "50 m resolution",
               "Resolution test: the reference configuration on a finer mesh.",
               "the mesh spacing (50 m)."),
    "res200": (200, "up", True, "2.e-3", "200 m resolution",
               "Resolution test: the reference configuration on a coarser mesh.",
               "the mesh spacing (200 m)."),
    "stationary": (100, "zero", True, "2.e-3", "stationary coast",
                   "Tectonic test: no vertical motion (Pastier: few terraces, "
                   "reefs rebuilt on the same foundation each highstand).",
                   "no uplift (0 mm/yr)."),
    "subsidence": (100, "down", True, "2.e-3", "subsiding coast",
                   "Tectonic test: subsidence (Pastier: stacked reefs, barrier "
                   "reefs and lagoons, older reefs drowned). Also the full-options "
                   "run of the options test on this coast.",
                   "subsidence of 0.3 mm/yr."),
    "sub_reefonly": (100, "down", False, None, "subsiding coast, no waves",
                     "Options test on the subsiding coast: the carbonate factories "
                     "alone, without waves or coastal erosion.",
                     "subsidence of 0.3 mm/yr; no waves block (no wave control on "
                     "the reef) and no coastal erosion."),
    "sub_waves": (100, "down", True, None, "subsiding coast, waves",
                  "Options test on the subsiding coast: the wave control on the "
                  "reef factory, without coastal erosion.",
                  "subsidence of 0.3 mm/yr; no coastal erosion."),
    "erosion10": (100, "up", True, "1.e-2", "stronger coastal erosion",
                  "Erosion test: the reference coast with coastal erosion five "
                  "times stronger (10 mm/yr at full wave energy).",
                  "coastal erosion rate 1.e-2 m/yr instead of 2.e-3."),
}
TECTO = {"up": "uplift 0.3 mm/yr", "zero": "no vertical motion",
         "down": "subsidence 0.3 mm/yr"}


def write_inputs():
    for name, (res, tecto, waves, erate, title, purpose, diff) in RUNS.items():
        txt = TEMPLATE.format(
            title=title, purpose=purpose, diff=diff, res=res, tecto=tecto,
            tecto_text=TECTO[tecto], name=name,
            wave_response=WAVE_RESP if waves else
            "          # (no wave control: no waves block in this run)",
            waves=WAVES if waves else "",
            coast=COAST.format(erate=erate) if erate else "")
        open(os.path.join(HERE, "input-%s.yml" % name), "w").write(txt)
        print("input-%s.yml" % name)


if __name__ == "__main__":
    for dx in (50.0, 100.0, 200.0):
        print("mesh %3d m: %d nodes" % (dx, strip(dx)))
    lo, hi = sealevel()
    print("sea level: %.1f to %.1f m" % (lo, hi))
    write_inputs()
