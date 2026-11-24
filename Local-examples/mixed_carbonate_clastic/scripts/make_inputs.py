"""
Write the input files of the mixed_carbonate_clastic example.

The margin (inputs/gospl_mesh.npz) and the sea-level curve
(inputs/sealevel.csv) are those of the stratigraphic_record example: a
200 x 100 km passive margin (land to the west, basin to the east) on an
unstructured mesh of ~700 m cells, and a synthetic curve that falls from +1 m
to -38 m (330 kyr) and rises back to -20 m (500 kyr). See
stratigraphic_record/model_setup.ipynb for how they were built.

One change to that margin: a rain map (key `rain` added to
inputs/gospl_mesh.npz) with two wet catchments, 2 m/yr in belts centred at
y = 25 and 75 km and 0.3 m/yr elsewhere. With uniform rain the rough
topography drains through dozens of small parallel rivers that bring sediment
to the whole coast; the wet belts make two large deltas with a starved coast
between them, where the carbonates can grow.

Every run uses one commented template; each experiment switches blocks on or
off. Run from the example folder: python scripts/make_inputs.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE = """\
# =============================================================================
# mixed_carbonate_clastic: {title}
#
# {purpose}
#
# Reference run: input-mixed.yml (rivers + two carbonate factories whose
# production stops where river sediment accumulates). This run differs by:
# {diff}
# =============================================================================
name: mixed carbonate-clastic margin, {title}

domain:
    npdata: ['inputs/gospl_mesh','v','c','z']   # 200 x 100 km margin, ~700 m cells
    bc: 'oooo'          # all edges open
    flowdir: 5          # multiple flow directions (5 receivers)

time:
    start: 0.
    end: 5.e5           # 500 kyr: the whole fall and rise of the sea level
    tout: 2.e4          # an output every 20 kyr
    dt: 5.e3            # 5 kyr steps (the carbonate growth does not depend on dt)
    strat: 5.e3         # a stratigraphic layer every step (100 layers)

# Rivers: the highlands (+260 m, west) are eroded and their sediment is carried
# to the coast (stream power law E = K (P^d A)^m S).
spl:
    K: 4.e-5
    d: 0.42
    m: 0.45

diffusion:
    hillslopeKa: 0.1    # creep on land (m2/yr)
    hillslopeKm: 0.8    # marine diffusion of the seafloor sediment (m2/yr)
    nonlinKm: 1000.     # transport of fresh marine deposits (m2/yr)
    clinSlp: 5.e-4      # maximum slope of the marine deposit top (clinoforms)

flexure:                # the margin flexes under the sediment load
    method: 'fem'
    thick: 30.e3
    rhoc: 2300.0
    rhoa: 3300.0
    young: 65e9
    nu: 0.25
    bcN: "Mirror"
    bcE: "0Slope0Shear"
    bcS: "Mirror"
    bcW: "0Slope0Shear"

sea:
    position: 0.
    curve: 'inputs/sealevel.csv'   # synthetic: +1 m -> -38 m (330 kyr) -> -20 m

climate:
  - start: 0.
    map: ['inputs/gospl_mesh','rain']   # 2 m/yr in two wet catchments (y = 25 and 75 km), 0.3 elsewhere

# Two siliciclastic lithologies (river sand and mud) plus, with the carbonate
# block, the carbonate lithologies (reef, carb_sand, carb_mud).
strata:
    dual: True
    coarse: {{phi0: 0.49, z0: 3700.}}
    fine:   {{phi0: 0.63, z0: 1960., k_factor: 1.5}}
    bedrock_coarse_frac: 0.6
    fine_diff_factor: 2.0
    pitInletBias: {{coarse: 0.5, fine: 0.0}}
{carbonate}{waves}{coast}{gravity}
output:
    dir: 'sim_{name}'
"""

# -----------------------------------------------------------------------------
# Carbonate factories for a ~700 m mesh: a reef is narrower than a cell, so the
# `reef` lithology is the framework share of a cell's carbonate (a margin
# facies), not a resolved reef (docs: Setting it up for your resolution).
# -----------------------------------------------------------------------------
CARB = """
# -----------------------------------------------------------------------------
# Carbonate production: two factories, both with the light law
#   G(w) = growth * tanh(insolation * exp(-extinction * w) / saturation)
# at water depth w.
# - euphotic: light-dependent reef builders and green algae, 0.5 mm/yr,
#   half of it built in place as framework (lithology `reef`); in the top
#   ~10 m only (extinction 0.2/m, and none below 30 m).
# - oligophotic: shelf producers of loose sand and mud (foraminifera,
#   molluscs, red algae), 0.3 mm/yr, down to ~60 m.
# {control_text}
# -----------------------------------------------------------------------------
carbonate:
    insolation: 400.           # W/m2: light at the sea surface
    factories:
      - name: euphotic
        growth: 500.           # m/Myr (0.5 mm/yr)
        extinction: 0.2        # 1/m: light falls with depth
        saturation: 60.        # W/m2: light above which the growth no longer increases
        framework: 0.5         # half built in place as reef framework
        sand: 0.7              # loose part: 70 % sand, 30 % mud
        response:
          - {{control: depth, x: [0., 20., 30.], f: [1., 1., 0.]}}
{sed_euph}{wave_euph}      - name: oligophotic
        growth: 300.           # m/Myr (0.3 mm/yr)
        extinction: 0.05       # 1/m: less light-dependent
        saturation: 60.
        framework: 0.          # all loose sediment
        sand: 0.5              # half carbonate sand, half carbonate mud
        response:
          - {{control: depth, x: [0., 50., 80.], f: [1., 1., 0.]}}
{sed_olig}"""
SEDRATE = ("          - {control: sedrate, x: [0., 3.e-4], f: [1., 0.]}   "
           "# no production where 0.3 mm/yr of river sediment accumulates\n")
CONTROL_ON = ("Both stop where river sediment accumulates (turbid water and burial\n"
              "# near the deltas): the `sedrate` control.")
CONTROL_OFF = ("No `sedrate` control here: the factories also grow on the delta\n"
               "# fronts, as if the river sediment did not affect them.")
WAVE_EUPH = ("          - {control: wave, x: [0., 0.5], f: [0.2, 1.]}   "
             "# the reef builders need wave energy (exposed shelf edge)\n")

WAVES = """
# -----------------------------------------------------------------------------
# Waves from the open ocean (east), dissipated as the water shallows below
# 20 m: the shelf edge takes the energy, the inner shelf is sheltered.
# -----------------------------------------------------------------------------
waves:
    climate:
      - {from: 90., weight: 1.}   # waves come from the east
    decay: 1.e6                   # no attenuation in deep water
    shoal_depth: 20.              # the waves feel the bottom shallower than 20 m
    shoal_length: 5.e3            # and lose their energy over ~5 km (7 cells)
"""
COAST = """
# -----------------------------------------------------------------------------
# Coastal wave erosion: shallow seafloor abraded, low cliffs cut back; the
# eroded reef framework is moved as carbonate sand (reef rubble).
# -----------------------------------------------------------------------------
coastal_erosion:
    rate: 1.e-3          # m/yr at full wave energy, at sea level
    depth_scale: 5.      # abrasion decays as exp(-depth / 5 m)
    cliff_height: 5.     # land up to 5 m above sea level next to the sea is a cliff
    max_erosion: 2.      # at most 2 m per step
    rubble: carb_sand
"""
GRAVITY = """
# -----------------------------------------------------------------------------
# Gravity currents (as in the gravity_currents example): part of the river
# load reaching the sea runs down the slope and is deposited at its toe and on
# the basin floor, beyond the clinoforms.
# -----------------------------------------------------------------------------
gravity:
    fraction: 0.3         # 30 % of the river load reaching the sea
    coarse_factor: 0.3    # mostly the suspended load (coarse diverted 0.3 x fraction)
    runout: 1.e4          # run-out of the coarse load (m)
    runout_fine: 2.e4     # run-out of the fine load (m)
    bypass_slope: 0.01    # less deposition on gradients steeper than 0.01
    receivers: 2          # coarse: concentrated network (channel belts)
    flow_exp: 4.
    receivers_fine: 6     # fine: spreading network (lobes)
    flow_exp_fine: 1.
    network_smooth: 3
    slope_smooth: 3
    deposit_spread: 1500.
    failure_slope: 0.02   # slope failure of fresh deposits steeper than 0.02
    max_failure: 10.
"""

RUNS = {
    # name: (carbonate, sedrate, waves, gravity, title, purpose, diff)
    "clastic": (False, False, False, False, "siliciclastic only",
                "The siliciclastic reference: rivers, deltas and clinoforms, no "
                "carbonate.",
                "no carbonate block."),
    "nocontrol": (True, False, False, False, "carbonates without sedrate control",
                  "Carbonates that ignore the river sediment: they grow on the "
                  "delta fronts as well (unrealistic, to show the control).",
                  "no `sedrate` response in the factories."),
    "mixed": (True, True, False, False, "mixed reference",
              "Mixed margin: carbonate on the shelf away from the deltas, "
              "stopped where river sediment accumulates.",
              "nothing (this is the reference)."),
    "waves": (True, True, True, False, "mixed with waves",
              "Waves from the open ocean: the reef builders need wave energy "
              "(a reef belt on the exposed shelf edge), and coastal erosion "
              "moves eroded reef as carbonate sand.",
              "a `waves` block, a `wave` response on the euphotic factory and "
              "a `coastal_erosion` block."),
    "wavecontrol": (True, True, "nocoast", False, "mixed with waves, no erosion",
                    "Waves from the open ocean control the reef builders, without "
                    "coastal erosion: separates the two effects of input-waves.yml.",
                    "a `waves` block and a `wave` response on the euphotic factory "
                    "(no coastal erosion)."),
    "gravity": (True, True, False, True, "mixed with gravity currents",
                "Gravity currents carry part of the river load down the slope "
                "to the basin floor.",
                "a `gravity` block."),
}


def add_rain():
    """Rain map with two wet catchments (key `rain`, m/yr) in the mesh file."""
    f = os.path.join(HERE, "inputs", "gospl_mesh.npz")
    d = dict(np.load(f))
    y = d["v"][:, 1] / 1.0e3
    d["rain"] = 0.3 + 1.7 * (np.exp(-((y - 25.0) / 8.0) ** 2)
                             + np.exp(-((y - 75.0) / 8.0) ** 2))
    np.savez(f, **d)


def write_inputs():
    for name, (carb, sed, waves, grav, title, purpose, diff) in RUNS.items():
        carbonate = ""
        if carb:
            carbonate = CARB.format(
                control_text=CONTROL_ON if sed else CONTROL_OFF,
                sed_euph=SEDRATE if sed else "", sed_olig=SEDRATE if sed else "",
                wave_euph=WAVE_EUPH if waves else "")
        txt = TEMPLATE.format(title=title, purpose=purpose, diff=diff, name=name,
                              carbonate=carbonate, waves=WAVES if waves else "",
                              coast=COAST if waves is True else "",
                              gravity=GRAVITY if grav else "")
        open(os.path.join(HERE, "input-%s.yml" % name), "w").write(txt)
        print("input-%s.yml" % name)


if __name__ == "__main__":
    add_rain()
    write_inputs()
