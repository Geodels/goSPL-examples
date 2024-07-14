#################################################################################
###
### Running this script to create netcdf grids with the following command:
### > python3 build_ncgrids.py -i input-escarpment.yml -o nc-escarpment -s 75
### where:
###     + input-escarpment.yml: the goSPL inputfile for a specific model
###     + nc-escarpment: a specific folder where the new grids will be saved
###     + 75: a integer representing the number of time steps to process (usually set to the number of goSPL outputs)
###
### This script will basically create similar grids as the ones done when running:
###     - exportnc.ipynb
###     - diversity.ipynb
###
### This will create 2 sets of netcdf grids for each time step:
###     1. a regular nc grid with goSPL outputs (dataXXX.nc)
###     2. a regular nc grid with morphometrics information (physioXXX.nc)
###
#################################################################################

import os
import argparse
import numpy as np
import xarray as xr
from scipy import spatial
# import metpy.calc as mpcalc
# from metpy.units import units

# import xrspatial
# from xrspatial import convolution
# from xrspatial import focal
from scripts import mapOutputs2D as mout

# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="This is a simple entry to export gospl outputs to netcdf grids.", add_help=True
)
parser.add_argument("-i", "--input", help="Input file name (YAML file)", required=True)
parser.add_argument("-o", "--output", help="Output directory", required=True)
parser.add_argument("-s", "--step", help="Number of outputs to convert", required=True)

parser.add_argument(
    "-r",
    "--reso",
    help="netcdf grid resolution (m)",
    required=False,
    action="store_true",
    default=250,
)
parser.add_argument(
    "-f",
    "--flex",
    help="export flexure",
    required=False,
    action="store_true",
    default=True,
)
parser.add_argument(
    "-t",
    "--tec",
    help="export tectonic",
    required=False,
    action="store_true",
    default=False,
)

args = parser.parse_args()
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Name of each netcdf output file
ncout = os.path.join(args.output, "data")

# Initialisation of the class
grid = mout.mapOutputs(path='./', filename=args.input, step=0, 
                       uplift=args.tec, flex=args.flex, model="utm")

indices = None
for k in range(0,int(args.step)+1):
    if k>0:
        # Get goSPL variables
        grid.getData(k)
        
    # Remap the variables on the regular mesh using distance weighting interpolation
    grid.buildUTMmesh(res=int(args.reso), nghb=4, smth=1)
    
    # Export corresponding regular mesh variables as netCDF file
    grid.exportNetCDF(ncfile = ncout+str(k)+'.nc')

    # Open netcdf data file for a specific time step
    # dataset = xr.open_dataset(ncout+str(k)+'.nc')
    # x = dataset.x.data
    # y = dataset.y.data
    # cellsize_x, cellsize_y = convolution.calc_cellsize(dataset.elevation)
    # dx, dy = cellsize_x, cellsize_y

    # if indices is None:
    #     xx,yy = np.meshgrid(x,y)
    #     xyz = np.zeros((len(xx.flatten()),3))
    #     xyz[:,0] = xx.ravel()
    #     xyz[:,1] = yy.ravel()
    #     tree = spatial.cKDTree(xyz, leafsize=10)
    #     indices = tree.query(xyz, k=9)[1]

    # elev = dataset.elevation.data
    # dzdy, dzdx = mpcalc.gradient(elev, deltas=(dy, dx))
    # slope_degrees = np.arctan(np.sqrt(dzdx.magnitude**2 + dzdy.magnitude**2)) * 180./np.pi

    # curv = xrspatial.curvature(dataset.elevation)
    # aspect = xrspatial.aspect(dataset.elevation)
    # hillshade = xrspatial.hillshade(dataset.elevation)

    # data_vars = {
    #         'slp_var':(['y','x'], np.nan_to_num(slope_degrees)),
    #     }
    # coords = {'y': (['y'], y), 'x': (['x'], x)}
    # physiodiv = xr.Dataset(data_vars=data_vars, coords=coords)
    # physiodiv['slp_var'].attrs = {'units':'degrees', 'long_name':'Slope in degrees',}

    # physiodiv['curvature'] = curv.copy()
    # physiodiv['aspect'] = aspect.copy()
    # physiodiv['hillshade'] = hillshade.copy()

    # cat_slp = physiodiv.slp_var.data.copy()
    # ids1 = np.where(cat_slp>=3.)
    # ids2 = np.where(np.logical_and(cat_slp<3.,cat_slp>=2.5))
    # ids3 = np.where(np.logical_and(cat_slp<2.5,cat_slp>=2.))
    # ids4 = np.where(np.logical_and(cat_slp<2.,cat_slp>=1.5))
    # ids5 = np.where(np.logical_and(cat_slp<1.5,cat_slp>=1.))
    # ids6 = np.where(np.logical_and(cat_slp<1.,cat_slp>=0.5))
    # ids7 = np.where(cat_slp<0.5)
    # cat_slp[ids7] = 1
    # cat_slp[ids6] = 2
    # cat_slp[ids5] = 3
    # cat_slp[ids4] = 4
    # cat_slp[ids3] = 5
    # cat_slp[ids2] = 6
    # cat_slp[ids1] = 7
    # physiodiv['slp_cat'] = (['y', 'x'],  cat_slp)
    # physiodiv['slp_cat'].attrs = {'units':'category', 'long_name':'Slope categorical distribution',
    #                                 'description':'Define 7 categories for continental slope based on NSEW slope values computed in degrees. Chosen values are 1: slope>3, 2: slope in [2.5,3], 3: slope in [2,2.5], 4: slope in [1.5,2], 5: slope in [1.,1.5], 6: slope in [0.5,1.] and 7: slope<0.5'}
    
    # inner_r1 = str(cellsize_x * 1)
    # outer_r1 = str(cellsize_x * 3)
    # inner_r2 = str(cellsize_x * 4)
    # outer_r2 = str(cellsize_x * 6)
    # kernel_large = convolution.annulus_kernel(cellsize_x, cellsize_y, outer_r2, inner_r2)
    # kernel_small = convolution.annulus_kernel(cellsize_x, cellsize_y, outer_r1, inner_r1)
    # focalmean_small = focal.apply(dataset.elevation, kernel_small)
    # focalmean_large = focal.apply(dataset.elevation, kernel_large)
    # tpi_small = ((dataset.elevation - focalmean_small)+0.5).astype(int)
    # tpi_large = ((dataset.elevation - focalmean_large)+0.5).astype(int)

    # # Standardize
    # tpi_small_std = ((((tpi_small-tpi_small.mean())/tpi_small.std())*100)+0.5).astype(int)
    # tpi_large_std = ((((tpi_large-tpi_large.mean())/tpi_large.std())*100)+0.5).astype(int)

    # # Classify
    # landformclass = np.zeros(tpi_small.shape)
    # flats = physiodiv.slp_var.where(physiodiv.slp_var<=0.1)
    # noflat = physiodiv.slp_var.where(physiodiv.slp_var>0.1)

    # cond1 = tpi_small_std.where((tpi_small_std>-100)&(tpi_small_std<100)&(tpi_large_std>-100)&(tpi_large_std<100))
    # cond1.data[~flats.notnull()] = np.nan
    # cond2 = tpi_small_std.where((tpi_small_std>-100)&(tpi_small_std<100)&(tpi_large_std>-100)&(tpi_large_std<100))
    # cond2.data[~noflat.notnull()] = np.nan
    # cond3 = tpi_small_std.where((tpi_small_std>-100)&(tpi_small_std<100)&(tpi_large_std>=100))
    # cond4 = tpi_small_std.where((tpi_small_std>-100)&(tpi_small_std<100)&(tpi_large_std<=-100))
    # cond5 = tpi_small_std.where((tpi_small_std<=-100)&(tpi_large_std>-100)&(tpi_large_std<100))
    # cond6 = tpi_small_std.where((tpi_small_std>=100)&(tpi_large_std>-100)&(tpi_large_std<100))
    # cond7 = tpi_small_std.where((tpi_small_std<=-100)&(tpi_large_std>=100))
    # cond8 = tpi_small_std.where((tpi_small_std<=-100)&(tpi_large_std<=-100))
    # cond9 = tpi_small_std.where((tpi_small_std>=100)&(tpi_large_std>=100))
    # cond10 = tpi_small_std.where((tpi_small_std>=100)&(tpi_large_std<=-100))

    # landformclass[cond1.notnull()] = 5
    # landformclass[cond2.notnull()] = 6
    # landformclass[cond3.notnull()] = 7
    # landformclass[cond4.notnull()] = 4
    # landformclass[cond5.notnull()] = 2
    # landformclass[cond6.notnull()] = 9
    # landformclass[cond7.notnull()] = 3
    # landformclass[cond8.notnull()] = 1
    # landformclass[cond9.notnull()] = 10
    # landformclass[cond10.notnull()] = 8

    # tpis_l = tpi_large_std.data.astype(float)
    # tpis_l[dataset.elevation.data<0.] = np.nan
    # tpis_s = tpi_small_std.data.astype(float)
    # tpis_s[dataset.elevation.data<0.] = np.nan
    # landformclass[dataset.elevation.data<0.] = np.nan
    # physiodiv['tpis_c'] = (['y', 'x'], tpis_l)
    # physiodiv['tpis_f'] = (['y', 'x'], tpis_s)
    # physiodiv['tpi_cat'] = (['y', 'x'], landformclass)
    # physiodiv['tpi_cat'].attrs = {'units':'category', 'long_name':'Landform categorical distribution',
    #                             'description':'Landform measures the topographic position of local relief normalized to local surface roughness. This is determined by calculating the topographic position index (TPI) which compares the elevation of each cell to the mean elevation of its neighborhood cells by convoluting 2 annulus kernel. The TPI is then standardized and used to define 10 landform classes (ridge, toe slope, slope, valley, open slopes...).'}

    # # Using log-scale for flow discharge
    # flow = np.nan_to_num(dataset.fillDischarge.values.copy())
    # flow[flow==0] = 0.00001
    # fill = np.log10(flow)
    # fill[fill<0] = 0.
    # physiodiv['flow'] = (['y', 'x'], fill)
    
    # catHydro = physiodiv.flow.data.copy()
    # id1 = np.where(catHydro<7)
    # id2 = np.where(np.logical_and(catHydro>=7,catHydro<8))
    # id3 = np.where(np.logical_and(catHydro>=8,catHydro<9))
    # id4 = np.where(np.logical_and(catHydro>=9,catHydro<10))
    # id5 = np.where(catHydro>=10)
    # catHydro[id1] = 1
    # catHydro[id2] = 2
    # catHydro[id3] = 3
    # catHydro[id4] = 4
    # catHydro[id5] = 5
    # physiodiv['hydro_cat'] = (['y', 'x'], catHydro)
    # physiodiv['hydro_cat'].attrs = {'units':'category', 'long_name':'Hydrology categorical distribution',
    #                         'description':'The hydrological category is based on the discharge distribution (km3/yr) obtained from the landscape evolution modelling. From the logarithmic distribution of the water flux we defined 5 categories: 1: logQs<7, 2: logQs in [7,8], 3: logQs in [8,9], 4: logQs in [9,10], 5: logQs>10.'}


    # tmpc = physiodiv.hydro_cat.data.flatten()
    # tmpc = np.nan_to_num(tmpc)
    # nonans = np.where(tmpc>0)[0]
    # prop_hydro = np.zeros(len(tmpc))

    # tmps = physiodiv.slp_cat.data.flatten()
    # tmps = np.nan_to_num(tmps)
    # prop_slp = np.zeros(len(tmps))

    # tmpt = physiodiv.tpi_cat.data.flatten()
    # tmpt = np.nan_to_num(tmpt)
    # prop_tpi = np.zeros(len(tmpt))

    # for d in range(len(nonans)):
    #     i = nonans[d]
    #     prop_hydro[i] = np.count_nonzero(tmpc[indices[i,:]]==tmpc[i])/9
    #     prop_slp[i] = np.count_nonzero(tmps[indices[i,:]]==tmps[i])/9
    #     prop_tpi[i] = np.count_nonzero(tmpt[indices[i,:]]==tmpt[i])/9

    # prop_hydro[nonans] = prop_hydro[nonans]*np.log(prop_hydro[nonans])
    # prop_slp[nonans] = prop_slp[nonans]*np.log(prop_slp[nonans])
    # prop_tpi[nonans] = prop_tpi[nonans]*np.log(prop_tpi[nonans])

    # diversity = -np.ones(len(tmpc))
    # diversity[nonans] = -(prop_hydro[nonans] + prop_slp[nonans]+prop_tpi[nonans])/np.log(3)
    # data = diversity[nonans].copy()
    # diversity[nonans] =  (data - np.min(data)) / (np.max(data) - np.min(data))
    # diversity[diversity<0] = np.nan
    # diversity = diversity.reshape(dataset.elevation.shape)
    # physiodiv['phydiv'] = (['y', 'x'], diversity)
    # physiodiv['phydiv'].attrs = {'units':'none', 'long_name':'Physiographic diversity index',
    #                     'description':'The diversity of physiographic is measured based on Shannon’s equitability, which is calculated by normalizing the Shannon-Weaver diversity index. Here we use 3 categories to compute the index, namely the hydrology, landform, and slope categories.'}
   
    # physiodiv['elevation'] = dataset.elevation.copy()
    # catElev = physiodiv.elevation.data.copy()
    # id1 = np.where(catElev<100)
    # id2 = np.where(np.logical_and(catElev>=100,catElev<200))
    # id3 = np.where(np.logical_and(catElev>=200,catElev<300))
    # id4 = np.where(np.logical_and(catElev>=300,catElev<400))
    # id5 = np.where(np.logical_and(catElev>=400,catElev<500))
    # id6 = np.where(np.logical_and(catElev>=500,catElev<600))
    # id7 = np.where(np.logical_and(catElev>=600,catElev<700))
    # id8 = np.where(np.logical_and(catElev>=700,catElev<800))
    # id9 = np.where(np.logical_and(catElev>=800,catElev<900))
    # id10 = np.where(np.logical_and(catElev>=900,catElev<1000))
    # id11 = np.where(np.logical_and(catElev>=1000,catElev<1100))
    # id12 = np.where(catElev>=1100)
    # catElev[id1] = 1
    # catElev[id2] = 2
    # catElev[id3] = 3
    # catElev[id4] = 4
    # catElev[id5] = 5
    # catElev[id6] = 6
    # catElev[id7] = 7
    # catElev[id8] = 8
    # catElev[id9] = 9
    # catElev[id10] = 10
    # catElev[id11] = 11
    # catElev[id12] = 12
    # physiodiv['elev_cat'] = (['y', 'x'], catElev)
    # physiodiv['elev_cat'].attrs = {'units':'category', 'long_name':'Elevation categorical distribution',}

    # comp = dict(zlib=True, complevel=5)
    # encoding = {var: comp for var in physiodiv.data_vars}
    # physiofile = os.path.join(args.output, "physio"+str(k)+".nc")
    # physiodiv.to_netcdf(physiofile,encoding=encoding)

