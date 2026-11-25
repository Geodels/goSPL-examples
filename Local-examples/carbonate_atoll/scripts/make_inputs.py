"""
Build the inputs of the carbonate_atoll example.

- data/island.npz: a 14 x 14 km domain at 100 m spacing, a lattice of
  equilateral triangles (no preferred direction), with
    v      vertex coordinates (m), c  triangles,
    z      a volcanic cone: +150 m at the centre, flanks of 7.5 %, sea level
           at 2 km from the centre, -375 m at the middle of the domain edges,
    sub    uniform subsidence of 0.5 mm/yr (m/yr): the volcano sinks 300 m in
           600 kyr, so its summit drowns after ~300 kyr and the last 300 kyr
           are an atoll (Darwin's fringing reef -> barrier reef -> atoll).

- data/sealevel.csv: relative sea level for the 600 kyr run (model time 0 =
  600 ka), from the Spratt & Lisiecki (2016) long stack (SeaLev_longPC1,
  0-798 ka), shifted so that the present is 0 m; used by input-sealevel.yml.
  Data: PANGAEA, https://doi.org/10.1594/PANGAEA.979830 (CC-BY-3.0), in data/.

Every run uses one commented template; each experiment changes the wave
climate (and one run the sea level). Run from the example folder:
python scripts/make_inputs.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = 14000.0            # domain size (m)
DX = 100.0             # spacing (m)
ZTOP, SLOPE = 150.0, 0.075
ZMIN = -750.0
SUB = -0.5e-3          # m/yr


def island():
    """Equilateral lattice centred on the island, symmetric about both axes."""
    from scipy.spatial import Delaunay

    nrow = 2 * int(round(L / (DX * np.sqrt(3.0)))) + 1
    ys = np.linspace(-L / 2, L / 2, nrow)
    pts = []
    for j, y in enumerate(ys):
        x = np.arange(-L / 2, L / 2 + 1.0, DX)
        if j % 2:
            x = np.concatenate([[-L / 2], x[:-1] + 0.5 * DX, [L / 2]])
        pts.append(np.column_stack([x, np.full(x.size, y)]))
    xy = np.vstack(pts)
    cells = Delaunay(xy).simplices
    r = np.hypot(xy[:, 0], xy[:, 1])
    z = np.maximum(ZTOP - SLOPE * r, ZMIN)
    n = len(xy)
    np.savez(os.path.join(HERE, "data", "island"),
             v=np.column_stack([xy + L / 2, np.zeros(n)]), c=cells.astype(np.int64),
             z=z, sub=np.full(n, SUB))
    return n


DURATION = 6.0e5      # yr


def sealevel():
    """Sea-level curve from the long Spratt & Lisiecki stack (column 5)."""
    txt = open(os.path.join(HERE, "data", "spratt_lisiecki_2016_v2.tab")).read()
    rows = [r.split("\t") for r in txt.split("*/\n")[1].strip().split("\n")[1:]]
    age = np.array([float(r[0]) for r in rows])               # ka BP
    sl = np.array([float(r[5]) if len(r) > 5 and r[5] else np.nan for r in rows])
    ok = np.isfinite(sl) & (age <= DURATION / 1e3)
    age, sl = age[ok], sl[ok]
    sl = sl - sl[age == 0.0][0]                               # present = 0 m
    t = DURATION - age * 1.0e3                                # model time (yr)
    o = np.argsort(t)
    np.savetxt(os.path.join(HERE, "data", "sealevel.csv"),
               np.column_stack([t[o], sl[o]]), delimiter=",", fmt="%.1f,%.3f")
    return sl.min(), sl.max()


TEMPLATE = """\
# =============================================================================
# carbonate_atoll: {title}
#
# {purpose}
#
# All runs: a volcanic island (+150 m) sinking at 0.5 mm/yr for 600 kyr, so
# that its fringing reef becomes a barrier reef and, once the summit drowns
# (~300 kyr), an atoll (Darwin);
# constant sea level unless stated; the same two carbonate factories. They differ only by
# the waves. This run: {diff}
# =============================================================================
name: carbonate atoll, {title}

domain:
    npdata: ['data/island','v','c','z']   # 14 x 14 km at 100 m, volcanic cone
    bc: 'oooo'          # open ocean on every side
    flowdir: 5          # multiple flow directions on the island

time:
    start: 0.
    end: 6.e5           # 600 kyr: the summit drowns after ~300 kyr
    tout: 2.e4          # an output every 20 kyr
    dt: 1000.           # time step (the carbonate growth does not depend on it)
    strat: 1.e4         # a stratigraphic layer every 10 kyr (60 layers)

# The island is eroded by rivers: volcanic sand and mud reach the reef.
spl:
    K: 2.e-6
    d: 0.
    m: 0.5

diffusion:
    hillslopeKa: 0.05   # creep on the island (m2/yr)
    hillslopeKm: 0.01   # marine diffusion of the loose sediment (m2/yr)
    nonlinKm: 50.       # transport of fresh marine deposits (m2/yr)

tectonics:
  - start: 0.
    end: 6.e5
    upsub: ['data/island','sub']   # subsidence 0.5 mm/yr

{sea}
climate:
  - start: 0.
    uniform: 1.         # 1 m/yr of rain on the island

strata:
    dual: True          # volcanic sand and mud, plus the carbonate lithologies

# -----------------------------------------------------------------------------
# Carbonate production: two factories with the light law
#   G(w) = growth * tanh(insolation * exp(-extinction * w) / saturation)
# - reef: reef builders, 4 mm/yr, none below 25 m, 60 % built in place as
#   framework, the rest loose sand and mud; they grow on rims and highs,
#   not on a lagoon floor (open-water index over 300 m: 5 % of their growth
#   where the seafloor is not above its surroundings);{wave_text}
# - lagoon: loose sand and mud producers (foraminifera, molluscs, algae),
#   0.4 mm/yr down to 50 m, no framework, everywhere.
# A lagoon floor produces 0.4 mm/yr plus at most 5 % of the reef growth,
# less than the 0.5 mm/yr subsidence, so once the summit has drowned the
# lagoon deepens behind the rim. (A control on the wave energy alone does not
# open a lagoon: a deeper lagoon lets more energy through, the reef in it
# grows faster and keeps up.)
# -----------------------------------------------------------------------------
carbonate:
    insolation: 400.           # W/m2: light at the sea surface
    exposure_length: 300.      # open-water index over 300 m (3 cells): rims > 0, lagoon floor ~0
    factories:
      - name: reef
        growth: 4000.          # m/Myr (4 mm/yr)
        extinction: 0.1        # 1/m: light falls with depth
        saturation: 60.        # W/m2: light above which the growth no longer increases
        framework: 0.6         # built in place (lithology `reef`)
        sand: 0.8              # loose part: 80 % sand, 20 % mud
        response:
          - {{control: depth, x: [0., 15., 25.], f: [1., 1., 0.]}}
          - {{control: exposure, x: [0., 3.], f: [0.05, 1.]}}   # reef on rims and highs, not on the lagoon floor
{wave_resp}      - name: lagoon
        growth: 400.           # m/Myr (0.4 mm/yr): less than the subsidence
        extinction: 0.05       # 1/m: less light-dependent
        saturation: 60.
        framework: 0.          # all loose sediment
        sand: 0.5              # half carbonate sand, half carbonate mud
        response:
          - {{control: depth, x: [0., 30., 50.], f: [1., 1., 0.]}}
{waves}{coast}{karst}
output:
    dir: 'sim_{name}'
"""

WAVE_RESP = ("          - {control: wave, x: [0., 0.5], f: [0.02, 1.]}   "
             "# reef builders need wave energy: sheltered water grows less\n")
WAVE_TEXT = ("\n#   and they need wave energy (2 % of their growth in sheltered water).")
NOWAVE_TEXT = ("\n#   no wave control here.")

WAVES = """
# -----------------------------------------------------------------------------
# Waves: {wave_title}
# They lose their energy in water shallower than 10 m (over ~300 m): the reef
# crest facing them takes the energy, the water behind it is sheltered, and
# the island casts a shadow downwind.
# -----------------------------------------------------------------------------
waves:
    climate:
{climate}    decay: 1.e6                   # no attenuation in deep water
    shoal_depth: 10.              # the waves feel the bottom shallower than 10 m
    shoal_length: 300.            # and lose their energy over ~300 m (3 cells)
{refraction}"""
REFR = ("    refraction: True              # turn towards the shore in shallow water "
        "(Snell's law)\n    refraction_length: 300.       # smooth the depth contours "
        "over 3 cells\n")
COAST = """
# -----------------------------------------------------------------------------
# Coastal wave erosion: the shallow seafloor and low cliffs are abraded by the
# waves; the eroded reef is moved as carbonate sand (reef rubble).
# -----------------------------------------------------------------------------
coastal_erosion:
    rate: 2.e-3          # m/yr at full wave energy, at sea level
    depth_scale: 3.      # abrasion decays as exp(-depth / 3 m)
    cliff_height: 10.    # land up to 10 m above sea level next to the sea is a cliff
    max_erosion: 2.      # at most 2 m per step
    rubble: carb_sand
"""

EIGHT = "".join("      - {from: %g., weight: 1.}\n" % a for a in range(0, 360, 45))
EAST = "      - {from: 90., weight: 1.}   # trade winds from the east\n"
SEASON = ("      - {from: 90., weight: 0.7}   # trade winds from the east, 70 % of the energy\n"
          "      - {from: 200., weight: 0.3}  # winter swell from the south-south-west, 30 %\n")

KARST = """
# -----------------------------------------------------------------------------
# Karst: the runoff dissolves the carbonate exposed during
# the lowstands, 5e-5 m of rock per m of water (50 m/Myr under 1 m/yr of
# rain), more where the water concentrates (focus); the volcanic rock does
# not dissolve, and 0.5 m of insoluble residue protects what lies below.
# -----------------------------------------------------------------------------
karst:
    rate: 5.e-5          # m of rock dissolved per m of runoff
    focus: 0.3           # dissolution grows as (contributing area / cell)^0.3
    cover: 0.5           # m of insoluble residue that stops it
"""

SEA_CONST = """sea:
    position: 0.        # constant sea level
"""
SEA_CURVE = """sea:
    position: 0.
    curve: 'data/sealevel.csv'   # Spratt & Lisiecki (2016) long stack, 0 m at present
"""

RUNS = {
    # name: (climate, title of the waves, refraction, erosion, title, purpose)
    "nowaves": (None, None, False, False, "no waves",
                "Reference without waves: the reef grows wherever the light allows."),
    "ring": (EIGHT, "from eight directions with the same energy (no prevailing "
             "direction).", False, False, "waves from all directions",
             "Waves from every direction: the reef concentrates on a rim all "
             "around the island, a ring atoll."),
    "east": (EAST, "trade winds from the east.", False, False, "trade winds",
             "One prevailing direction (east): a reef rim on the windward side, "
             "a sheltered leeward side."),
    "season": (SEASON, "trade winds from the east (70 %) and a winter swell from "
               "the south-south-west (30 %).", False, False, "two wave directions",
               "Two directions with different energies: the rim follows both."),
    "refraction": (EAST, "trade winds from the east, refracted in shallow water.",
                   True, False, "trade winds with refraction",
                   "Refraction turns the waves towards the shore: they wrap around "
                   "the island and reach its flanks and part of its lee."),
    "erosion": (EAST, "trade winds from the east.", False, True,
                "trade winds with coastal erosion",
                "Coastal erosion on the windward side; the eroded reef is moved "
                "as carbonate sand."),
    "sealevel": (EAST, "trade winds from the east.", False, False,
                 "trade winds with glacial sea level",
                 "The trade-wind run under the Quaternary sea level (Spratt & "
                 "Lisiecki 2016): lowstands of -120 m expose the platform and move "
                 "the reef down the flanks; each highstand floods it again."),
    "karst": (EAST, "trade winds from the east.", False, False,
              "glacial sea level and karst",
              "The glacial sea-level run with karst dissolution: during the "
              "lowstands the exposed platform is dissolved by the runoff, "
              "most along its drainage."),
}


def write_inputs():
    for name, (clim, wtitle, refr, ero, title, purpose) in RUNS.items():
        if clim is None:
            diff = "no waves block (no wave control on the reef)."
        else:
            diff = "waves " + wtitle[0].lower() + wtitle[1:]
            if refr:
                diff += " Refraction on."
            if ero:
                diff += " Coastal erosion on."
            if name in ("sealevel", "karst"):
                diff += " Sea level from the Spratt & Lisiecki stack."
            if name == "karst":
                diff += " Karst dissolution of the exposed carbonate."
        txt = TEMPLATE.format(
            title=title, purpose=purpose, diff=diff, name=name,
            wave_text=WAVE_TEXT if clim else NOWAVE_TEXT,
            wave_resp=WAVE_RESP if clim else "",
            waves=WAVES.format(wave_title=wtitle, climate=clim,
                               refraction=REFR if refr else "") if clim else "",
            coast=COAST if ero else "",
            karst=KARST if name == "karst" else "",
            sea=SEA_CURVE if name in ("sealevel", "karst") else SEA_CONST)
        open(os.path.join(HERE, "input-%s.yml" % name), "w").write(txt)
        print("input-%s.yml" % name)


if __name__ == "__main__":
    print("island mesh: %d nodes" % island())
    print("sea level: %.1f to %.1f m" % sealevel())
    write_inputs()
