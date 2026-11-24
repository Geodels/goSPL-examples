"""
Build the inputs of the carbonate_global example.

It reuses, in place, the 250 Ma global mesh and rainfall of the sibling
example `continental_flux` (`../continental_flux/vars_25_80/`, ~474,000
nodes, 20-100 km spacing) and, for the geochemistry run, the temperature and
climate-province maps of `continental_geochem` (`../continental_geochem/
geochem_inputs.npz`): build those two examples' inputs first.

- data/carb_maps.npz: per-vertex maps on that mesh,
    sst    sea-surface temperature (degC): 29 at the equator, falling to ~0
           at the poles (cosine of the latitude), used by the temperature
           response of the carbonate factories and of the pelagic rain,
    prod   pelagic carbonate rain (m/Myr): 2 everywhere plus 8 in the
           equatorial belt (high export production), before dissolution.

Every run uses one commented template; each experiment switches blocks on or
off. Run from the example folder: python scripts/make_inputs.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH = "../continental_flux/vars_25_80/mesh"
RAIN = "../continental_flux/vars_25_80/rain250"
GEO = "../continental_geochem/geochem_inputs"


def maps():
    d = np.load(os.path.join(HERE, MESH + ".npz"))
    v = d["v"]
    R = np.linalg.norm(v, axis=1)
    lat = np.degrees(np.arcsin(np.clip(v[:, 2] / R, -1.0, 1.0)))
    sst = 29.0 * np.cos(np.radians(lat)) ** 1.5
    prod = 2.0 + 8.0 * np.exp(-(lat / 15.0) ** 2)
    np.savez(os.path.join(HERE, "data", "carb_maps"), sst=sst, prod=prod)
    return len(v)


TEMPLATE = """\
# =============================================================================
# carbonate_global: {title}
#
# {purpose}
#
# Reference run: input-carb.yml (carbonate factories with a sea-surface
# temperature window, pelagic carbonate and hemipelagic clay). This run
# differs by: {diff}
# =============================================================================
name: carbonate global, {title}

domain:
    npdata: ['{mesh}','v','c','z']   # 250 Ma global mesh of continental_flux
    flowdir: 2
    seadepo: True

time:
    start: 250.e6
    end: 251.e6         # 1 Myr
    tout: 1.e5          # 10 outputs
    dt: 5.e4
    strat: 1.e5         # a stratigraphic layer every output

spl:
    K: 5.e-6
    d: 0.42
    m: 0.5

diffusion:
    hillslopeKa: 0.5
    hillslopeKm: 2.0
    nonlinKm: 1.e4
    Gmar: 0.5
    clinSlp: 1.e-5
    marineSolver: picard
    picardSub: 10
    picardIts: 2
    nlPitK: 5.e3
    pitInletBias: 0.4
{soil}{prov}
sea:
    position: 0.

climate:
  - start: 250.e6
    map: ['{rain}','r']   # per-vertex rainfall (m/yr)

strata:
    dual: True          # river sand and mud, plus the carbonate lithologies
{gw}
# -----------------------------------------------------------------------------
# Sea-surface temperature (degC): a map by latitude (data/carb_maps.npz), the
# `temperature` response of the factories and of the pelagic rain.
# -----------------------------------------------------------------------------
sst:
    events:
      - {{start: 250.e6, map: ['data/carb_maps', 'sst']}}

# -----------------------------------------------------------------------------
# Carbonate production: two factories with the light law
#   G(w) = growth * tanh(insolation * exp(-extinction * w) / saturation)
# at a 20-100 km resolution a reef is narrower than a cell: `reef` is the
# framework share of a cell's carbonate (a margin facies).
# - shallow: tropical shelves (reefs, green algae, ooids), 0.4 mm/yr, 30 %
#   framework, only in warm water (no growth below 16 or above 32 degC, full
#   growth between 18 and 28 degC).
# - deep: cool-water and deeper loose carbonate (bryozoans, foraminifera,
#   molluscs), 0.05 mm/yr down to 400 m, at every temperature.{wave_text}{solute_text}
# -----------------------------------------------------------------------------
carbonate:
    insolation: 400.           # W/m2: light at the sea surface
    factories:
      - name: shallow
        growth: 400.           # m/Myr (0.4 mm/yr)
        extinction: 0.1        # 1/m: light falls with depth
        saturation: 60.        # W/m2: light above which the growth no longer increases
        framework: 0.3         # 30 % built in place (lithology `reef`)
        sand: 0.6              # loose part: 60 % sand, 40 % mud
        response:
          - {{control: temperature, x: [16., 18., 28., 32.], f: [0., 1., 1., 0.]}}
{wave_resp}{solute_resp}      - name: deep
        growth: 50.            # m/Myr (0.05 mm/yr)
        extinction: 0.005      # 1/m: hardly light-dependent
        saturation: 60.
        framework: 0.          # all loose sediment
        sand: 0.3
        response:
          - {{control: depth, x: [0., 200., 400.], f: [1., 1., 0.]}}

# -----------------------------------------------------------------------------
# Pelagic rain: carbonate from the surface ocean (data/carb_maps.npz `prod`,
# m/Myr, high under the equatorial belt), preserved above the lysocline,
# dissolving linearly down to the carbonate compensation depth (CCD) where
# none is preserved; and hemipelagic clay (fine) decaying away from the coasts.
# -----------------------------------------------------------------------------
pelagic:
  - start: 250.e6
    carbonate: ['data/carb_maps', 'prod']   # pelagic carbonate rain (m/Myr)
    lysocline: {lys}           # m: dissolution starts below this depth
    ccd: {ccd}                 # m: carbonate compensation depth
    clay: 1.                   # hemipelagic clay at the coast (m/Myr)
    clay_decay: 3.e5           # clay * exp(-d / 300 km) from the coast
    temperature: {{x: [5., 25.], f: [0.3, 1.]}}   # less carbonate rain in cold water
{waves}{coast}{supply}
output:
    dir: 'sim_{name}'
"""

WAVE_RESP = ("          - {control: wave, x: [0., 0.5], f: [0.3, 1.]}   "
             "# reef builders prefer exposed coasts\n")
WAVE_TEXT = ("\n# The shallow factory grows less on sheltered coasts (wave energy below 0.5).")
WAVES = """
# -----------------------------------------------------------------------------
# Waves: the global wind belts (trade winds from the east within 30 degrees of
# the equator, westerlies to 60, polar easterlies beyond), growing over a
# fetch of 500 km of open water and losing their energy in water shallower
# than 30 m: exposed coasts face the open ocean along the wind, sheltered
# ones lie behind continents and islands.
# -----------------------------------------------------------------------------
waves:
    climate:
      - {from: zonal, weight: 1.}
    fetch: 5.e5                   # growth to full energy over 500 km
    decay: 1.e7                   # no attenuation in deep water
    shoal_depth: 30.              # the waves feel the bottom shallower than 30 m
    shoal_length: 5.e4            # and lose their energy over ~50 km (1-2 cells)
"""
COAST = """
# -----------------------------------------------------------------------------
# Coastal wave erosion of the shallow seafloor and of low coasts; the eroded
# reef is moved as carbonate sand.
# -----------------------------------------------------------------------------
coastal_erosion:
    rate: 1.e-4          # m/yr at full wave energy, at sea level
    depth_scale: 20.     # abrasion decays as exp(-depth / 20 m)
    cliff_height: 50.    # land up to 50 m above sea level next to the sea
    max_erosion: 2.      # at most 2 m per step
    rubble: carb_sand
"""
SOIL = """
soil:
    mode: 'regolith'
    soilK: 8.e-6
    solver: 'ngmres'
    pcType: 'hypre'
    rtol: 1.e-5
    atol: 1.e-5
    maxProd: 30.e-6
    depthProd: 1.0
    uniform: 0.5
    tempMap: ['{geo}', 'temp']
    activation: 40.e3
    tempRef: 15.0
"""
PROV = """
provenance:
    classes: 3
    source: ['{geo}', 'rock']    # 0 tropical | 1 subtropical | 2 temperate (+3: made in the sea)
"""
GW = """
# -----------------------------------------------------------------------------
# Groundwater and geochemistry of continental_geochem: three climate provinces
# weather iron, carbonate and silica; the dissolved load goes down the rivers
# to the ocean, where it accumulates (marine_coupling).
# -----------------------------------------------------------------------------
groundwater:
    Ksat: 3.65e4
    specific_yield: 0.1
    aquifer_base: 50.0
    infiltration: 0.005
    conserve_baseflow: True
    duricrust:
        form_rate: 1.0e-4
        max_thickness: 5.0
        fringe_depth: 3.0
        fringe_width: 2.0
        armor_max: 0.9
        weathering:
            mode: prodsoil
    geochem:
        conserve: True
        river_load: True
        marine_coupling: True
        weatherability_from: source_class
        species:
          - {name: iron,      weatherability_by_class: [1.0, 0.1, 0.0], precip_rate: 1.0}
          - {name: carbonate, weatherability_by_class: [0.0, 1.0, 0.2], precip_rate: 1.0}
          - {name: silica,    weatherability_by_class: [0.1, 0.2, 1.0], precip_rate: 0.8}
"""
SOLUTE_RESP = ("          - {{control: solute, species: carbonate, length: 2.e5, x: [0., {sx}], "
               "f: [0.5, 1.]}}   # more growth where the rivers bring carbonate (alkalinity)\n")
SOLUTE_TEXT = ("\n# The shallow factory grows more where the rivers bring dissolved carbonate\n"
               "# (alkalinity), half its rate far from them (plume of 200 km).")
SUPPLY = """
# -----------------------------------------------------------------------------
# Supply-limited burial: the carbonate the factories and the pelagic rain bury
# each step cannot exceed what the dissolved carbonate delivered to the ocean
# (the marine reservoir of the geochemistry) can make, times a yield.
# -----------------------------------------------------------------------------
carbonate_supply:
    species: carbonate
    yield: {yld}         # m3 of carbonate per unit of solute
    pelagic: True        # the pelagic rain shares the supply
"""

RUNS = {
    # name: (lys, ccd, waves, geochem, title, purpose, diff)
    "carb": (3500., 4500., False, False, "reference",
             "Global carbonate system: tropical shelves, cool-water carbonate, "
             "pelagic carbonate above the CCD and hemipelagic clay.",
             "nothing (this is the reference)."),
    "ccd_shallow": (2500., 3500., False, False, "shallow CCD",
                    "A shallower lysocline and CCD (2500 and 3500 m instead of "
                    "3500 and 4500): less pelagic carbonate preserved in the basins.",
                    "lysocline 2500 m, CCD 3500 m."),
    "waves": (3500., 4500., True, False, "waves and coastal erosion",
              "The global wind belts make exposed and sheltered coasts: the shallow "
              "factory prefers exposed ones, and the waves erode the coasts.",
              "a `waves` block (zonal belts, fetch 500 km), a `wave` response on "
              "the shallow factory and a `coastal_erosion` block."),
    "geochem": (3500., 4500., False, True, "coupled to the geochemistry",
                "The carbonate is coupled to the continental geochemistry of "
                "continental_geochem: the dissolved carbonate delivered by the "
                "rivers limits the burial, and its coastal plume favours the "
                "shallow factory.",
                "the soil, provenance, groundwater and geochemistry blocks of "
                "continental_geochem, a `solute` response on the shallow factory "
                "and a `carbonate_supply` block."),
}


def write_inputs(yld=1.0e-3, sx=1.0):
    for name, (lys, ccd, waves, geo, title, purpose, diff) in RUNS.items():
        txt = TEMPLATE.format(
            title=title, purpose=purpose, diff=diff, name=name, mesh=MESH, rain=RAIN,
            lys=lys, ccd=ccd,
            soil=SOIL.format(geo=GEO) if geo else "",
            prov=PROV.format(geo=GEO) if geo else "",
            gw=GW if geo else "",
            wave_text=WAVE_TEXT if waves else "",
            solute_text=SOLUTE_TEXT if geo else "",
            wave_resp=WAVE_RESP if waves else "",
            solute_resp=SOLUTE_RESP.format(sx=sx) if geo else "",
            waves=WAVES if waves else "",
            coast=COAST if waves else "",
            supply=SUPPLY.format(yld=yld) if geo else "")
        open(os.path.join(HERE, "input-%s.yml" % name), "w").write(txt)
        print("input-%s.yml" % name)


if __name__ == "__main__":
    print("maps on %d nodes" % maps())
    write_inputs(yld=20.0, sx=1.0e-8)
