"""
Implicit time-stepping example figure (fig_ex_timestep.pdf) for the GMD paper.
Uses the pre-computed, already-rasterised NetCDF outputs of the
implicit_timestepping example: the same uplift-driven landscape run to steady
state with four internal time steps (dt = 500, 1250, 2500, 5000 yr). Reproduces
the notebook's mean-elevation-history comparison (unconditional stability of the
implicit scheme; accuracy cost that grows with dt) and shows the near-identical
steady-state topographies at the smallest and largest dt.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# prefix, dt (yr), n output steps, output interval tout (yr), marker
RUNS = [("500", 500, 40, 2.5e3, "^"),
        ("125", 1250, 40, 2.5e3, "X"),
        ("250", 2500, 40, 2.5e3, "o"),
        ("5k", 5000, 20, 5.0e3, "v")]
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def mean_hist(pfx, n, tout):
    z, t = [], []
    for k in range(n + 1):
        d = xr.open_dataset(os.path.join(EX, f"{pfx}_surface_{k}.nc"))
        z.append(float(d.elev.mean())); t.append(k * tout)
    return np.array(t) / 1e3, np.array(z)          # ky, m

fig, ax = plt.subplots(1, 3, figsize=(12.2, 4.6),
                       gridspec_kw={"width_ratios": [1.25, 1, 1], "wspace": 0.3})
a, b, c = ax

# --- (a) mean elevation history --------------------------------------------
cmap = plt.get_cmap("viridis")
endvals = {}
for i, (pfx, dt, n, tout, mk) in enumerate(RUNS):
    t, z = mean_hist(pfx, n, tout)
    endvals[dt] = z[-1]
    a.plot(t, z, mk + "-", ms=4, lw=1.8, color=cmap(i / 3.0),
           label=f"$\\Delta t$ = {dt/1000:g} ky")
a.set_xlabel("simulation time (ky)"); a.set_ylabel("mean elevation (m)")
a.set_title("(a) Mean-elevation history", fontsize=10, loc="left")
a.legend(fontsize=8, frameon=False, loc="lower right")
a.grid(True, ls=":", lw=0.5, alpha=0.5)
gap = 100.0 * abs(endvals[5000] - endvals[500]) / endvals[500]
a.text(0.60, 0.52, f"Steady state \n end $\\Delta t$=5 ky vs 0.5 ky: {gap:.1f}%",
       transform=a.transAxes, fontsize=8, ha="center", va="center",
       bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#bbb"))

# --- (b,c) steady-state topography at extreme dt ---------------------------
d500 = xr.open_dataset(os.path.join(EX, "500_surface_40.nc"))
d5k = xr.open_dataset(os.path.join(EX, "5k_surface_20.nc"))
vmax = float(max(d500.elev.max(), d5k.elev.max()))
im = None
for ax_, ds, ttl in [(b, d500, "(b) $\\Delta t$ = 0.5 ky (step 40)"),
                     (c, d5k, "(c) $\\Delta t$ = 5 ky (step 20)")]:
    im = ax_.pcolormesh(ds.x / 1e3, ds.y / 1e3, ds.elev, cmap="Spectral_r",
                        vmin=0, vmax=vmax, shading="auto", rasterized=True)
    ax_.set_title(ttl, fontsize=10, loc="left")
    ax_.set_xlabel("x (km)"); ax_.set_box_aspect(1)
b.set_ylabel("y (km)")
cb = fig.colorbar(im, ax=[b, c], orientation="horizontal", location="bottom",
                  shrink=0.6, pad=0.18, aspect=40)
cb.set_label("elevation (m)", fontsize=8)

fig.savefig("fig_ex_timestep.pdf", bbox_inches="tight")
fig.savefig("fig_ex_timestep.png", dpi=160, bbox_inches="tight")
print(f"wrote fig_ex_timestep.pdf / .png  | end gap dt5k vs dt500 = {gap:.1f}%  "
      f"| means end: " + ", ".join(f"{k}:{v:.1f}" for k, v in endvals.items()))
