"""Utilities for catchment extraction using DEM and flow accumulation."""

import numpy as np
import xarray as xr
import pandas as pd
from scipy import spatial
import rasterio
from rasterio.transform import from_origin
from pysheds.grid import Grid
import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def getCatchment(dataset,ptx,pty,fa_thres=50):
    """Delineate a catchment from DEM data and a pour point.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset containing an elevation variable and coordinates.
    ptx, pty : float
        Pour point longitude and latitude coordinates in the same CRS as the DEM.
    fa_thres : int, optional
        Flow accumulation threshold used to extract the river network branches.

    Returns
    -------
    grid : pysheds.Grid
        Grid object containing the loaded DEM, flow directions, and catchment mask.
    branches : dict
        GeoJSON-like river network features extracted from the catchment.
    dist : numpy.ndarray
        Distance from each cell to the outlet along the drainage network.
    branch_df : list of pandas.DataFrame
        Per-branch data frames with x, y, dist, fa, and z values for each extracted branch.
    """

    geotiffname = 'tmp.tif'
    elevArray = dataset.elevation.copy()

    # Assigning a value of -9999.0 to all point below sea-level
    elevArray = elevArray.where(elevArray>0, other=-9999.0)

    x = dataset.x.values
    y = dataset.y.values
    dx = dataset.x.values[1]-dataset.x.values[0]

    transform = from_origin(x.min(),y.max(), dx, dx)

    new_dataset = rasterio.open(geotiffname, 'w', driver='GTiff',
                                height = elevArray.shape[0], width = elevArray.shape[1],
                                count=1, dtype=str(elevArray.dtype),
                                crs='+proj=utm +zone=1 +datum=NAD83 +units=m',
                                transform=transform)
    new_dataset.write(np.flipud(elevArray), 1)
    new_dataset.close()

    grid = Grid.from_raster(geotiffname,nodata=-9999.0)
    dem = grid.read_raster(geotiffname, nodata=-9999.0)

    # Condition DEM
    # ----------------------
    # Fill pits in DEM
    pit_filled_dem = grid.fill_pits(dem)

    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)
        
    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Determine D8 flow directions from DEM
    # ----------------------
    # Specify directional mapping
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        
    # Compute flow directions
    # -------------------------------------
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

    # Calculate flow accumulation
    # --------------------------
    acc = grid.accumulation(fdir, dirmap=dirmap)

    # Delineate a catchment
    # ---------------------
    # Specify pour point
    x, y = ptx, pty

    # Snap pour point to high accumulation cell
    x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))

    # Delineate the catchment
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, 
                        xytype='coordinate')

    # Crop and plot the catchment
    # ---------------------------
    # Clip the bounding box to the catchment
    grid.clip_to(catch)
    clipped_catch = grid.view(catch)
    branches = grid.extract_river_network(fdir, acc > fa_thres, dirmap=dirmap)

    # Calculate distance to outlet from each cell
    # -------------------------------------------
    dist = grid.distance_to_outlet(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap,xytype='coordinate')


    xyz = np.zeros((dist.size,2)) 
    x = np.arange(grid.extent[0],grid.extent[1],250)
    y = np.arange(grid.extent[2],grid.extent[3],250)
    xx,yy = np.meshgrid(x,y)
    xyz[:,1] = xx.ravel()
    xyz[:,0] = yy.ravel()
    tree = spatial.cKDTree(xyz, leafsize=10)
    tree2 = spatial.cKDTree(dem.coords, leafsize=10)
    
    branch_df = []
    nbbranches = len(branches['features'])

    for b in range(nbbranches):
        branch = branches['features'][b]
        branchXY = np.asarray(branch['geometry']['coordinates'])
        branchXY = np.flip(branchXY,1)
        _, id = tree.query(branchXY, k=1)
        _, id2 = tree2.query(branchXY, k=1)
        dst = np.flipud(dist.data).flatten()[id]*dx
        elev = inflated_dem.flatten()[id2]
        fa = acc.flatten()[id2]
        xb = branchXY[:,1]
        yb = branchXY[:,0]

        data = {'x': xb, 'y': yb, 'dist': dst, 'fa': fa, 'z': elev}
        df = pd.DataFrame(data)

        branch_df.append(df)

    return grid, branches, dist, branch_df