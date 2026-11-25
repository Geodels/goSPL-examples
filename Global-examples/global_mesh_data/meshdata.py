"""
Put global gridded data (NetCDF) on a goSPL global mesh.

A goSPL map input (rain, uplift, temperature, erodibility, ...) is an npz file
with one value per mesh vertex, given in the input file as ``[file, key]``.
This module checks a global longitude/latitude grid, makes its longitudes run
from -180 to 180, and interpolates it on the vertices of a global mesh
(``mesh.npz`` with ``v``, the Cartesian coordinates of the vertices).

The grid must cover the whole sphere: longitudes from -180 to 180 (or 0 to
360) with BOTH ends present, latitudes from -90 to 90 with both poles
present, no missing value. When it does not, :func:`check_global_grid` stops
with a message saying what is wrong and the xarray line that fixes it.

Typical use::

    import meshdata as md
    lon, lat = md.mesh_lonlat("mesh.npz")
    grid = md.open_global_grid("rain.nc", var="precip")
    rain = md.interpolate_to_mesh(grid, lon, lat)
    np.savez("rain_mesh.npz", r=rain)      # input file: map: ['rain_mesh', 'r']
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

LON_NAMES = ("lon", "longitude", "long", "x", "nav_lon", "LON", "Longitude", "X")
LAT_NAMES = ("lat", "latitude", "y", "nav_lat", "LAT", "Latitude", "Y")
TOL = 1.0e-6          # degrees


class GridError(ValueError):
    """A global grid that cannot be interpolated as it is (see the message)."""


def mesh_lonlat(mesh_file):
    """Longitude (-180 to 180) and latitude (-90 to 90) of every vertex of a
    goSPL global mesh (npz with ``v``, Cartesian coordinates)."""
    v = np.load(mesh_file)["v"]
    r = np.linalg.norm(v, axis=1)
    lat = np.degrees(np.arcsin(np.clip(v[:, 2] / r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    return lon, lat


def _find(names, coords, what, short):
    for n in names:
        if n in coords:
            return n
    raise GridError(
        "No %s coordinate found (looked for %s); the grid has %s.\n"
        "Fix: rename yours, e.g.  da = da.rename({'<your %s name>': '%s'})"
        % (what, ", ".join(names[:4]), sorted(set(coords)), what, short))


def check_global_grid(da):
    """
    Check a global grid and return it with longitudes from -180 to 180.

    Accepted: a 2-D DataArray on 1-D longitude and latitude coordinates, with
    longitudes either from -180 to 180 or from 0 to 360 (both ends present),
    latitudes from -90 to 90 (both poles present), and no missing value.
    Data in 0 to 360 are moved to -180 to 180 (the 180 column becomes -180 and
    is repeated at 180). Anything else raises :class:`GridError` with the
    xarray line that fixes it.
    """
    if isinstance(da, xr.Dataset):
        raise GridError("Give a DataArray (one variable), not a Dataset.\n"
                        "Fix: da = ds['%s']" % (list(da.data_vars) or ["name"])[0])
    lon_n = _find(LON_NAMES, list(da.coords) + list(da.dims), "longitude", "lon")
    lat_n = _find(LAT_NAMES, list(da.coords) + list(da.dims), "latitude", "lat")
    da = da.rename({lon_n: "lon", lat_n: "lat"}) if (lon_n, lat_n) != ("lon", "lat") else da

    extra = [d for d in da.dims if d not in ("lon", "lat")]
    if extra:
        hint = "da = da.isel(%s=0)" % extra[0] if da.sizes[extra[0]] > 1 else "da = da.squeeze('%s')" % extra[0]
        raise GridError("The grid has more dimensions than lon and lat: %s.\n"
                        "Fix: select one level/time first, e.g.  %s" % (extra, hint))
    if da.lon.ndim != 1 or da.lat.ndim != 1:
        raise GridError("lon and lat must be 1-D coordinates (a regular lon/lat grid); "
                        "this one is curvilinear.\nFix: regrid it first, e.g. with xESMF.")
    if not np.issubdtype(da.dtype, np.number):
        raise GridError("The values are not numbers (dtype %s).\nFix: da = da.astype(float)" % da.dtype)

    for name, lo, hi in (("lat", -90.0, 90.0),):
        c = da[name].values
        if np.any(np.diff(c) == 0):
            raise GridError("Repeated %s values.\nFix: da = da.drop_duplicates('%s')" % (name, name))
        if c[0] > c[-1]:
            raise GridError("Latitudes decrease (%g to %g).\nFix: da = da.sortby('lat')" % (c[0], c[-1]))
        if abs(c[0] - lo) > TOL or abs(c[-1] - hi) > TOL:
            raise GridError(
                "Latitudes run from %g to %g: the grid must include both poles (-90 and 90).\n"
                "A cell-centred grid stops half a cell short. Fix: add the poles, e.g.\n"
                "  da = da.interp(lat=np.r_[-90., da.lat.values, 90.], "
                "kwargs={'fill_value': 'extrapolate'})\n"
                "or, to repeat the last row:  da = da.pad(lat=1, mode='edge').assign_coords("
                "lat=np.r_[-90., da.lat.values, 90.])" % (c[0], c[-1]))

    lon = da.lon.values
    if np.any(np.diff(lon) == 0):
        raise GridError("Repeated longitudes.\nFix: da = da.drop_duplicates('lon')")
    if lon[0] > lon[-1]:
        raise GridError("Longitudes decrease.\nFix: da = da.sortby('lon')")
    if lon.min() >= -TOL and lon.max() > 180.0 + TOL:
        # 0 to 360: both ends needed, then move to -180..180
        if abs(lon[0]) > TOL or abs(lon[-1] - 360.0) > TOL:
            raise GridError(
                "Longitudes run from %g to %g: a 0 to 360 grid must include both 0 and 360 "
                "(the same meridian, repeated).\nFix: repeat the first column at 360:\n"
                "  da = xr.concat([da, da.isel(lon=0).assign_coords(lon=360.)], dim='lon')"
                % (lon[0], lon[-1]))
        da = da.isel(lon=slice(0, -1))                                   # drop 360 (= 0)
        da = da.assign_coords(lon=((da.lon + 180.0) % 360.0) - 180.0).sortby("lon")
        da = xr.concat([da, da.isel(lon=0).assign_coords(lon=180.0)], dim="lon")
        lon = da.lon.values
    if abs(lon[0] + 180.0) > TOL or abs(lon[-1] - 180.0) > TOL:
        raise GridError(
            "Longitudes run from %g to %g: the grid must include both -180 and 180 (or 0 and 360).\n"
            "A cell-centred grid stops half a cell short. Fix: close it with the opposite column,\n"
            "  da = xr.concat([da.isel(lon=-1).assign_coords(lon=-180.), da,\n"
            "                  da.isel(lon=0).assign_coords(lon=180.)], dim='lon')\n"
            "and interpolate the two edge columns if you need them exact." % (lon[0], lon[-1]))

    nan = int(np.isnan(da.values).sum()) if np.issubdtype(da.dtype, np.floating) else 0
    if nan:
        raise GridError(
            "%d missing values (NaN) in the grid: goSPL needs a value everywhere.\n"
            "Fix: fill them, e.g.  da = da.fillna(0.)  or  "
            "da = da.interpolate_na('lon').interpolate_na('lat')" % nan)
    return da.transpose("lat", "lon")


def open_global_grid(nc_file, var=None):
    """Open one variable of a NetCDF file and check it (see
    :func:`check_global_grid`). ``var`` is needed when the file has several."""
    try:
        ds = xr.open_dataset(nc_file)
    except Exception as exc:
        raise GridError("Cannot read %s as NetCDF (%s).\nCheck the file with: ncdump -h %s"
                        % (nc_file, exc, nc_file)) from exc
    names = list(ds.data_vars)
    if var is None:
        if len(names) != 1:
            raise GridError("The file has %d variables %s: choose one with var='...'" % (len(names), names))
        var = names[0]
    if var not in ds:
        raise GridError("No variable %r in %s; it has %s" % (var, nc_file, names))
    return check_global_grid(ds[var].load())


def interpolate_to_mesh(da, lon, lat, method="linear"):
    """Values of a checked grid at the mesh vertices (bilinear by default)."""
    da = check_global_grid(da)
    f = RegularGridInterpolator((da.lat.values, da.lon.values), da.values.astype(float),
                                method=method, bounds_error=True)
    return f(np.column_stack([np.clip(lat, -90.0, 90.0), np.clip(lon, -180.0, 180.0)]))
