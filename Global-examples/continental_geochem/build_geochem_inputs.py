"""
Build the per-vertex forcing maps for the global geochemistry example, from the
shared continental_flux mesh:

  - `temp`  : a generic **latitudinal + altitude** near-surface air temperature
              (degC), warm at the equator, cold at the poles, cooling with
              elevation — drives the Arrhenius temperature dependence of soil
              production and chemical weathering (faster in the warm tropics).
  - `rock`  : integer **climate-zone provinces** (source-rock classes) used both
              for sediment provenance and to set which solute species each region
              yields (via `weatherability_by_class`):
                0 = tropical humid    (|lat| < 15)  -> iron / ferricrete (laterite)
                1 = subtropical arid  (15-35)       -> carbonate / calcrete
                2 = temperate + high  (|lat| >= 35) -> silica / silcrete

Writes `geochem_inputs.npz` (keys `temp`, `rock`) next to this script. Run once
before the model (the mesh is reused in place from ../continental_flux):

    python build_geochem_inputs.py
"""
import os
import numpy as np

MESH = "../continental_flux/vars_25_80/mesh.npz"

d = np.load(MESH)
v = np.asarray(d["v"], dtype=np.float64)
z = np.asarray(d["z"], dtype=np.float64)
R = np.linalg.norm(v, axis=1).mean()
lat = np.degrees(np.arcsin(np.clip(v[:, 2] / R, -1.0, 1.0)))
alat = np.abs(lat)

# Latitudinal + altitude temperature (degC): 28 at the equator (sea level),
# -0.5 degC per degree of latitude, and a 6 degC/km lapse over subaerial relief.
temp = 28.0 - 0.5 * alat - 6.0e-3 * np.maximum(z, 0.0)

# Climate-zone provinces (source-rock classes). The province boundaries follow
# climate belts, but pure lines of latitude give unrealistically straight species /
# duricrust boundaries. Perturb the latitude used for classification with a smooth
# 2-D undulation (a few zonal wavenumbers, modulated in latitude) so the belts still
# march tropical -> subtropical -> temperate but their edges wander by ~5-10 deg.
lon = np.arctan2(v[:, 1], v[:, 0])                 # radians, -pi..pi
latr = np.radians(lat)
rng = np.random.default_rng(7)
wave = np.zeros_like(alat)
for k in range(1, 5):                              # zonal wavenumbers 1..4
    wave += rng.uniform(1.5, 3.5) * np.sin(k * lon + rng.uniform(0.0, 2.0 * np.pi)
                                           + 1.2 * np.sin(2.0 * latr))
alat_eff = alat + wave                             # wavy effective latitude (deg)

rock = np.full(v.shape[0], 2, dtype=np.int64)      # temperate / high-lat (silica)
rock[alat_eff < 35.0] = 1                          # subtropical arid (carbonate)
rock[alat_eff < 15.0] = 0                          # tropical humid   (iron)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geochem_inputs.npz")
np.savez_compressed(out, temp=temp.astype(np.float64), rock=rock)
print("wrote %s" % out)
print("  temp (degC): min %.1f  mean %.1f  max %.1f" % (temp.min(), temp.mean(), temp.max()))
for k, name in enumerate(["tropical(iron)", "subtropical(carbonate)", "temperate(silica)"]):
    print("  rock class %d = %-24s : %d vertices" % (k, name, int((rock == k).sum())))
