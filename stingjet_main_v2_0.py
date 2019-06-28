#################################################################
# EDIT THESE FOLLOWING LINES TO POINT TO THE LIBRARY DIRECTORY
import getpass
myusername = getpass.getuser()

import sys
sys.path.append('/home/users/%s/pythond/methods' % (myusername))
#################################################################

import time

import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cf_units

import iris as iris
import iris.quickplot as qplt
import iris.plot as iplt

import trackmethods as track
import SJtrackStats as sjs
import sphere_methods as sphere
import atmospheric_diagnostics as adiag
import csisounding_methods as csis
from planetary_constants import Omega, R, a

def find_blobs_1lev(mask):
    '''
    Based on MATLAB function FIND_BLOBS_1LEV

    FIND_BLOBS_1LEV Find blobs of near-by gridpoints given a mask on 
    a single level.
      B = FIND_BLOBS_LOW(MASK) returns an array of N cells. Each cell
      corresponds to a different blob and contains a 2xM matrix. The first
      row contains the linear indices of the gridpoints within the blob. The
      second row contains the corresponding pressure level index.

    Oscar Martinez-Alvarado
    Department of Meteorology
    University of Reading

    Created on 11/Sep/2009
    based on the function find_blobs.m by O.M-A.
    '''

    s = mask.shape
    sm = s[0:2]
    anyblob = False
    vmaskk = mask

    IX = np.where(vmaskk)
    IXflat = np.ravel_multi_index(IX, sm)

    blob2 = []
    if IXflat.shape[0] > 0:
        anyblob = True
        blen = IX[0].shape[0]
        blobrow = np.full((blen, 9), -10)
        blobcol = np.full((blen, 9), -10)
        blob = np.full((blen, 9), -10)

        for pt in range(blen):
            # Looking for the closest neighbours
            blix = 0
            blobrow[pt, blix] = IX[0][pt]
            blobcol[pt, blix] = IX[1][pt]
            blob[pt, blix] = IXflat[pt]

            rpt = IX[0][pt]
            cpt = IX[1][pt]

            for nb in np.arange(pt+1, blen):
                rnb = IX[0][nb]
                cnb = IX[1][nb]
            
                if rnb >= rpt-1 and rnb <= rpt+1 and \
                   cnb >= cpt-1 and cnb <= cpt+1:
                    # IX(nb) is a neighbour of IX(pt)
                    blix = blix + 1
                    blobrow[pt, blix] = IX[0][nb]
                    blobcol[pt, blix] = IX[1][nb]
                    blob[pt, blix] = IXflat[nb]

        # Join partial blobs
        blcount = 1
        blix = 0
        while blix != -1:
            npblob = -1
            npbix = -1
            blobn = np.setdiff1d(blob[blix, :], -10)
            for ii in np.arange(blix+1, blen):
                blobi = np.setdiff1d(blob[ii, :], -10)
                isect = np.intersect1d(blobn, blobi)
                if isect.shape[0] > 0:
                    blobn = np.union1d(blobn, blobi)
                    blob[ii, :] = 0
                else:
                    npbix = npbix + 1
                    if npbix == 0:
                        npblob = ii
                
            if blobn.shape[0] > 0:
                blob2.append(blobn)
            else:
                blcount = blcount - 1
            
            if npbix >= 0:
                blix = npblob
                blcount = blcount + 1
            else:
                blix = -1

    return blob2

parser = argparse.ArgumentParser(description='Compute and/or plot DCAPE and DSCAPE. The plots correspond to the top and bottom of the pressure layer given by PLEVHIGH and PLEVLOW, as defined below.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('stormno', type=int, help='Storm number')
parser.add_argument('--stormdir', default='/storage/silver/NCAS-Weather/sws07om/data/stingjet', help='Storm directory')
parser.add_argument('--dataset', default='ajthp', choices=['ajthp', 'ajthr'], \
                    help='Climate run: AJTHP=Present-day, AJTHR=Future-climate')
parser.add_argument('--year', type=int, default=1996, help='Storm year')
parser.add_argument('--season', default='son', choices=['son', 'djf', 'mam'], \
                    help='Storm season')
parser.add_argument('--testing', default=False, action='store_true')
parser.add_argument('--plevlow', type=float, default=800.0, \
                    help='Lowest pressure level')
parser.add_argument('--plevhigh', type=float, default=400.0, \
                    help='Highest pressure level')
parser.add_argument('--indist', default=False, action='store_true')
parser.add_argument('--max_radius_csi', type=float, default=1000.0e3, 
                    help='Radius around storm centre to compute CSI, in m')
parser.add_argument('--mindscape', type=float, default=200.0, \
                    help='Minimum DSCAPE in CSI diagnostic')
parser.add_argument('--moisttop', default=False, action='store_true')
parser.add_argument('--minrh', type=float, default=80.0, \
                    help='Minimum RH wrt ice in CSI diagnostic')
parser.add_argument('--mingradient', type=float, default=1.0e-2, \
                    help='Minimum wet-bulb potential temperature gradient in CSI diagnostic, in K km-1')
parser.add_argument('--minadvection', type=float, default=1.0e-1, \
                    help='Minimum wet-bulb potential temperature in CSI diagnostic, in m K s-1 km-1')
parser.add_argument('--dcape', default=False, action='store_true', \
                    help='Compute DCAPE')
parser.add_argument('--printing', default=False, action='store_true')
parser.add_argument('--figdir', default='/storage/silver/NCAS-Weather/sws07om/figures/sjoper', help='Figure directory')
parser.add_argument('--dpi', type=int, default=100, help='Figure resolution, in dpi')
parser.add_argument('--saving', default=False, action='store_true')
parser.add_argument('--outdir', default='/storage/silver/NCAS-Weather/sws07om/data/sjoper', help='Output directory')

args = parser.parse_args()

start = time.time()

CaseID = 'sjmain'

FigureFormat = 'png'

km2m = 1000.0 # meter

# Maximum vorticity INDEX in input files
# MATLAB: Vmax_ix = 8; PYTHON: Vmax_ix = 7
Vmax_ix = 7

# Track number
trackno = args.stormno - 1

datadir = '%s/%s/%d/%s' % (args.stormdir, args.dataset, args.year, args.season)

print('Reading data from: %s' % (datadir))

if not args.testing and args.saving:
    outdir = '%s/%s/%d/%s' % (args.outdir, args.dataset, args.year, args.season)

# TRACK file
trackfile = 'ML_%d_%s_%s.full_track.NH.tracks.selected' % \
            (args.year, args.season, args.dataset)

print('Reading %s' % (trackfile))

TRACKS = track.readtracks_1season('%s/%s' % (datadir, trackfile))
TRACK = TRACKS[trackno]

Nvalid_tsteps = TRACK['vort'].index.size

print('The storm has %d out of a maximum of 15 time steps' % (Nvalid_tsteps))

print('Analysing STORM %d' % (args.stormno))

Coriolis = 2.0*Omega*np.sin(np.deg2rad(TRACK['mslp']['lat'].values))

stormfile = 'S%04d.nc' % (args.stormno)

print('Reading %s' % (stormfile))

# Load primary variables
u = iris.load_cube('%s/%s' % (datadir, stormfile), 'eastward_wind')
v = iris.load_cube('%s/%s' % (datadir, stormfile), 'northward_wind')
theta = iris.load_cube('%s/%s' % (datadir, stormfile), \
                       'air_potential_temperature')
q = iris.load_cube('%s/%s' % (datadir, stormfile), 'specific_humidity')
mslp = iris.load_cube('%s/%s' % (datadir, stormfile), \
                      'air_pressure_at_sea_level')
mslp = iris.util.squeeze(mslp)

# Compute secondary variables
p3d = adiag.p3d_like(theta)
temp = adiag.temperature(p3d, theta)
rh = adiag.rhwrtwater(p3d, temp, q)
rhice = adiag.rhwrtice(p3d, temp, q)
rs = adiag.saturation_mixing_ratio_water(p3d, temp)
r = adiag.mixing_ratio(q)

Ntsteps, Klev, Jlat, Ilon = theta.data.shape

if args.testing:
    print('TEST mode')
    if args.saving:
        print('Saving not allowed')
    time_steps = [0]
    sounding_i_location = [30]
    sounding_j_location = [30]
else:
    time_steps = range(Nvalid_tsteps)
    sounding_i_location = range(Ilon)
    sounding_j_location = range(Jlat)

lon = u.coord('longitude')
lat = u.coord('latitude')
pressure = u.coord('pressure')
time_from_max = u.coord('time')

Klow = np.where(pressure.points==args.plevlow)[0][0]
Khigh = np. where(pressure.points==args.plevhigh)[0][0]

pressure_dcape = iris.coords.DimCoord(pressure.points[Klow:Khigh+1], \
                                      standard_name='air_pressure', \
                                      units=pressure.units)

EQLON, EQLAT = np.meshgrid(lon.points, lat.points)

X, Y = sjs.fplane_cartesian_grid(EQLON, EQLAT)

# Define radial distance
dist = a*sphere.distance_sphere(EQLON, EQLAT, 0.0, 0.0)

wspeed = (u**2 + v**2)**0.5
wspeed.rename('wind_speed')

Mindex = np.argmax(TRACK['vort']['val'].values)

itime_offset = Vmax_ix - Mindex

for itime in time_steps:
    print('Analysing ITIME %d' % (itime))

    ifield = itime + itime_offset

    thetaw = adiag.ezwbpottemp(p3d[ifield, :, :, :], q[ifield, :, :, :], \
                               theta[ifield, :, :, :])
    
    Fx, Fy, magF = adiag.gradient_sphere(thetaw)

    velgrad = Fx*u[ifield, :, :, :] + Fy*v[ifield, :, :, :]

    eta = csis.vorticity(X, Y, u[ifield, :, :, :], v[ifield, :, :, :], \
                    Coriolis[itime], \
                    wspeed=wspeed[ifield, :, :, :])
    dr_dX = csis.d_dX(X, Y, u[ifield, :, :, :], v[ifield, :, :, :], \
                 r[ifield, :, :, :], \
                    wspeed=wspeed[ifield, :, :, :])
    dws_dLp = adiag.d_dLp(wspeed[ifield, :, :, :])

    integrand = dws_dLp / eta

    dscape_k = np.zeros((Khigh - Klow + 1, Jlat, Ilon))
    if args.dcape: dcape_k = np.zeros((Khigh - Klow + 1, Jlat, Ilon))

    for K1 in np.arange(Klow, Khigh+1):
        p_ij = np.flipud(pressure.points[:K1+1])
        for jj in sounding_j_location:
            for ii in sounding_i_location:
                compute_column = True

                if args.indist:
                    if dist[jj, ii] > args.max_radius_csi:
                        compute_column = False

                if args.moisttop:
                    if rhice[ifield, K1, jj, ii].data < args.minrh:
                        compute_column = False
                    
                if compute_column:
                    temp.convert_units('kelvin')
                    t_ij = np.flipud(temp[ifield, :K1+1, jj, ii].data)
                    r_ij = np.flipud(r[ifield, :K1+1, jj, ii].data)
                    rs_ij = np.flipud(rs[ifield, :K1+1, jj, ii].data)
                    int_ij = np.flipud(integrand[:K1+1, jj, ii].data)
                    dws_dLp_ij = np.flipud(dws_dLp[:K1+1, jj, ii].data)
                    dr_dX_ij = np.flipud(dr_dX[:K1+1, jj, ii].data)

                    F_ij = int_ij / p_ij

                    DeltaX = np.zeros_like(F_ij)

                    DeltaX[1:] = -np.cumsum((F_ij[1:] + F_ij[:-1]) * \
                                                    (p_ij[1:] - p_ij[:-1]) / \
                                            2.0)

                    if args.dcape: dcape_k[K1 - Klow, jj, ii] = \
                       adiag.dcape_sounding(p_ij, \
                                            t_ij, \
                                            r_ij)

                    temp_correction = - Coriolis[itime] * dws_dLp_ij * \
                                      DeltaX / R

                    r_correction = dr_dX_ij * DeltaX

                    t_ij = t_ij + temp_correction
                    r_ij = r_ij + r_correction

                    dscape_k[K1 - Klow, jj, ii] = adiag.dcape_sounding(p_ij, \
                                                                       t_ij, \
                                                                       r_ij)

                else:
                    if args.dcape: dcape_k[K1 - Klow, jj, ii] = np.nan
                    dscape_k[K1 - Klow, jj, ii] = np.nan

    dscape_vmax = np.nanmax(dscape_k, axis=0)

    imax = np.zeros((Jlat, Ilon), dtype=int)
    pressuremax = np.zeros((Jlat, Ilon))
    moistmax = np.zeros((Jlat, Ilon))
    maggrad = np.zeros((Jlat, Ilon))
    thwadvec = np.zeros((Jlat, Ilon))
    
    for jj in sounding_j_location:
        for ii in sounding_i_location:
            if ~np.isnan(dscape_vmax[jj, ii]):
                imax[jj, ii] = np.nanargmax(dscape_k[:, jj, ii]) + Klow

                rhice_ij = rhice[ifield, :, jj, ii].data
                magF_ij = magF[:, jj, ii].data
                velgrad_ij = velgrad[:, jj, ii].data

                pressuremax[jj, ii] = pressure.points[imax[jj, ii]]
                moistmax[jj, ii] = rhice_ij[imax[jj, ii]-1:imax[jj, ii]+2].max()
                maggrad[jj, ii] = magF_ij[imax[jj, ii]]
                thwadvec[jj, ii] = velgrad_ij[imax[jj, ii]]
            else:
                pressuremax[jj, ii] = np.nan
                moistmax[jj, ii] = np.nan
                maggrad[jj, ii] = np.nan
                thwadvec[jj, ii] = np.nan

    mask = np.where(np.logical_and( \
                            np.logical_and(dscape_vmax >= args.mindscape, \
                                           moistmax >= args.minrh), \
                            np.logical_and(maggrad >= args.mingradient, 
                                           thwadvec >= args.minadvection)), \
                    1, 0)
    
    blobs = find_blobs_1lev(mask)
    print('STORM %d, ITIME %d, NBLOBS %d' % (args.stormno, itime, len(blobs)))
                
    plt.pcolormesh(EQLON, EQLAT, mask)

    if args.printing:
        FigID = 'mask'
        FigureName = '%s_%s' % (CaseID, FigID)
        if args.moisttop:
            FigureName = '%s_mtop' % (FigureName)
        if args.testing:
            FigureName = '%s_test' % (FigureName)
        FigureName = '%s_S%04d_%s00.%s' % \
                     (FigureName, args.stormno, TRACK['vort'].index[itime], \
                      FigureFormat)
        plt.savefig('%s/%s' % (args.figdir, FigureName), format=FigureFormat, \
                    dpi=args.dpi)
        print 'Printed to: %s' % (FigureName)

    if args.testing:
        print('PRESSURE')
        print(pressure.points[Klow:Khigh+1])
        print('DSCAPE')
        print(dscape_k[:, sounding_j_location[0], sounding_i_location[0]])
        print('DSCAPE*')
        print(dscape_vmax[sounding_j_location[0], sounding_i_location[0]])
        print('IMAX')
        print(imax[sounding_j_location[0], sounding_i_location[0]])
        print('PRESSURE')
        print(pressure.points)
        print('PRESSURE*')
        print(pressuremax[sounding_j_location[0], sounding_i_location[0]])
        print('RHICE')
        print(rhice[ifield, :, sounding_j_location[0], sounding_i_location[0]].data)
        print('RHICE*')
        print(moistmax[sounding_j_location[0], sounding_i_location[0]])
        print('THWGRAD')
        print(magF[:, sounding_j_location[0], sounding_i_location[0]].data)
        print('THWGRAD*')
        print(maggrad[sounding_j_location[0], sounding_i_location[0]])
        print('THWADV')
        print(velgrad[:, sounding_j_location[0], sounding_i_location[0]].data)
        print('THWADV*')
        print(thwadvec[sounding_j_location[0], sounding_i_location[0]])

    elif args.saving:
        if args.dcape:
            dcape = iris.cube.Cube(dcape_k, \
                                   var_name='DCAPE', \
                                   units=cf_units.Unit('J kg-1'), \
                                   dim_coords_and_dims=[(pressure_dcape, 0), \
                                                        (lat, 1), \
                                                        (lon, 2)])

        dscape = iris.cube.Cube(dscape_k, \
                                var_name='DSCAPE', \
                                units=cf_units.Unit('J kg-1'), \
                                dim_coords_and_dims=[(pressure_dcape, 0), \
                                                     (lat, 1), \
                                                     (lon, 2)])

        dscapestar = iris.cube.Cube(dscape_vmax, \
                                var_name='maximum_DSCAPE', \
                                units=dscape.units, \
                                dim_coords_and_dims=[(lat, 0), \
                                                     (lon, 1)])

        pstar = iris.cube.Cube(pressuremax, \
                                var_name='pressure_at_maximum_DSCAPE', \
                                units=pressure.units, \
                                dim_coords_and_dims=[(lat, 0), \
                                                     (lon, 1)])

        rhstar = iris.cube.Cube(moistmax, \
                                var_name= \
                                'relative_humidity_wrt_ice_at_maximum_DSCAPE', \
                                units=rhice.units, \
                                dim_coords_and_dims=[(lat, 0), \
                                                     (lon, 1)])

        thwgstar = iris.cube.Cube(maggrad, \
                                  var_name= \
                                  'wet-bulb_potential_temperature_gradient_at_maximum_DSCAPE', \
                                  units=magF.units, \
                                  dim_coords_and_dims=[(lat, 0), \
                                                       (lon, 1)])

        thwastar = iris.cube.Cube(thwadvec, \
                                var_name= \
                                'wet-bulb_potential_temperature_advection_at_maximum_DSCAPE', \
                                units=velgrad.units, \
                                dim_coords_and_dims=[(lat, 0), \
                                                     (lon, 1)])

        if args.dcape: 
            clist = iris.cube.CubeList( [dcape, dscape, \
                                         dscapestar, \
                                         pstar, rhstar, \
                                         thwgstar, thwastar])
        else:
            clist = iris.cube.CubeList([dscape, \
                                        dscapestar, \
                                        pstar, rhstar, \
                                        thwgstar, thwastar])
            clist2 = iris.cube.CubeList([thetaw, mslp[ifield, :, :]])

        OutID = 'dscape'
        OutFile = '%s_%s' % (CaseID, OutID)
        if args.moisttop:
            OutFile = '%s_mtop' % (OutFile)
        OutFile = '%s_S%04d_%s00.nc' % \
                  (OutFile, args.stormno, TRACK['vort'].index[itime])
        iris.save(clist, '%s/%s' % (outdir, OutFile), \
                  netcdf_format='NETCDF4_CLASSIC')
        print 'Saving to: %s' % (OutFile)

        OutID = 'met'
        OutFile = '%s_%s' % (CaseID, OutID)
        if args.moisttop:
            OutFile = '%s_mtop' % (OutFile)
        OutFile = '%s_S%04d_%s00.nc' % \
                  (OutFile, args.stormno, TRACK['vort'].index[itime])
        iris.save(clist2, '%s/%s' % (outdir, OutFile), \
                  netcdf_format='NETCDF4_CLASSIC')
        print 'Saving to: %s' % (OutFile)

if not args.testing and args.saving:
    print 'Data saved in: %s/' % (outdir)

if args.printing:
    print 'All figures printed in: %s/' % (args.figdir)

end = time.time()

print('Time ellapsed: %5.1f min' % ((end - start)/60.0))

