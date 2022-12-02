"""
The information included at this site is for educational purposes only, neither The University of Texas at Austin
and the authors provide a warranty either expressed or impied in its application.

Nothing described within this script should be construed to lessen the need to apply sound engineering judgment nor to
carefully apply accepted engineering practices in the design, implementation, or application of
the techniques described herein.

Trend modeling script.
Version: 4.0
Author: Jose Julian Salazar
Date: December 2022

NOTE: This script is still in development
"""
import os
import sys
import time
import warnings
from random import sample

import matplotlib.colors as colors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import scipy as sp
import seaborn as sns
import torch
import torch.nn as nn
import xarray as xr
from astropy.convolution import Gaussian2DKernel, convolve
from ax.service.ax_client import AxClient
from geostatspy import geostats
from geostatspy.GSLIB import DataFrame2ndarray, Dataframe2GSLIB, GSLIB2ndarray, ndarray2GSLIB
from geostatspy.geostats import dsortem, dlocate, gauinv, backtr_value, getindex, srchnd, dpowint, krige, setrot
from geostatspy.geostats import gamv, setup_rotmat2, ctable
from matplotlib import pyplot as plt
from plotly.colors import n_colors
from plotly.subplots import make_subplots
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.ax import AxSearch
from scipy.stats import chi2, pearsonr, skew
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm

plt.rcParams.update({'font.size': 12})
warnings.filterwarnings('ignore')  # Don't show the nan warnings
pio.renderers.default = "browser"
pio.templates.default = 'simple_white'


# def KL():
#     # KL divergence
#     # self._loss = nn.KLDivLoss(reduction='sum')
#
#     def _univariate_target_(self, nbins, bin_range):
#         hist, bin_edges = np.histogram(self.df[self.main_feat], bins=nbins, density=True, range=bin_range)
#         self._nbins = nbins
#         self._min_bin_edge = bin_edges[0]
#         self._max_bin_edge = bin_edges[-1]
#         self._target_dist = torch.tensor(hist * np.diff(bin_edges)).log()
#
#
# hist, bin_edges = np.histogram(realizations_df[self.main_feat], bins=self._nbins, density=True,
#                                range=(self._min_bin_edge, self._max_bin_edge))
# realizations = torch.tensor(hist * np.diff(bin_edges))


def sgsim(df, xcol, ycol, vcol, wcol, scol, tmin, tmax, itrans, ismooth, dftrans, tcol, twtcol, zmin, zmax, ltail,
          ltpar, utail, utpar, nsim, nx, xmn, xsiz, ny, ymn, ysiz, seed, ndmin, ndmax, nodmax, mults, nmult, noct,
          radius, radius1, sang1, mxctx, mxcty, ktype, colocorr, sec_map, vario, varred=1.0):
    """Modified version from geostatspy."""
    # Parameters from sgsim.inc
    na = None
    UNEST = -99.0
    EPSLON = 1.0e-20

    # Set other parameters
    np.random.seed(seed)
    nxy = nx * ny
    sstrat = 0  # search data and nodes by default, turned off if unconditional
    radsqd = radius * radius
    sanis1 = radius1 / radius

    # load the variogram
    nst = int(vario['nst'])
    cc = np.zeros(nst)
    aa = np.zeros(nst)
    it = np.zeros(nst, dtype=int)
    ang = np.zeros(nst)
    anis = np.zeros(nst)

    c0 = vario['nug']
    cc[0] = vario['cc1']
    it[0] = vario['it1']
    ang[0] = vario['azi1']
    aa[0] = vario['hmaj1']
    anis[0] = vario['hmin1'] / vario['hmaj1']
    if nst == 2:
        cc[1] = vario['cc2']
        it[1] = vario['it2']
        ang[1] = vario['azi2']
        aa[1] = vario['hmaj2']
        anis[1] = vario['hmin2'] / vario['hmaj2']

    # Set the constants
    MAXCTX = mxctx
    MAXCTY = mxcty
    MAXCXY = MAXCTX * MAXCTY
    MAXX = nx
    MAXY = ny
    MXY = MAXX * MAXY
    MXY = 100 if MXY < 100 else MXY
    MAXNOD = nodmax
    MAXSAM = ndmax
    MAXKR1 = MAXNOD + MAXSAM + 1

    MAXKR2 = MAXKR1 * MAXKR1

    # Declare arrays
    nums = np.zeros(ndmax, dtype=int)

    # Perform some quick checks
    if ltail != 1 and ltail != 2:
        print('ERROR invalid lower tail option ' + str(ltail))
        print('      only allow 1 or 2 - see GSLIB manual ')
        sys.exit()
    if utail != 1 and utail != 2 and utail != 4:
        print('ERROR invalid upper tail option ' + str(ltail))
        print('      only allow 1,2 or 4 - see GSLIB manual ')
        sys.exit()
    if utail == 4 and utpar < 1.0:
        print('ERROR invalid power for hyperbolic tail' + str(utpar))
        print('      must be greater than 1.0!')
        sys.exit()
    if ltail == 2 and ltpar < 0.0:
        print('ERROR invalid power for power model' + str(ltpar))
        print('      must be greater than 0.0!')
        sys.exit()
    if utail == 2 and utpar < 0.0:
        print('ERROR invalid power for power model' + str(utpar))
        print('      must be greater than 0.0!')
        sys.exit()

    # Load the data
    df_extract = df.loc[(df[vcol] >= tmin) & (df[vcol] <= tmax)]  # trim values outside tmin and tmax
    nd = len(df_extract)
    ndmax = min(ndmax, nd)
    x = df_extract[xcol].values
    y = df_extract[ycol].values
    vr = df_extract[vcol].values
    vr_orig = np.copy(vr)
    if wcol > -1:
        wt = df_extract[wcol].values
    else:
        wt = np.ones(nd)
    sec = []
    sec = np.array(sec)
    if scol > -1:
        sec = df_extract[scol].values
    if itrans == 1:
        if ismooth == 1:
            dftrans_extract = dftrans.loc[(dftrans[tcol] >= tmin) & (dftrans[tcol] <= tmax)]
            ntr = len(dftrans_extract)
            vrtr = dftrans_extract[tcol].values
            if twtcol > -1:
                vrgtr = dftrans_extract[tcol].values
            else:
                vrgtr = np.ones(ntr)
        else:
            vrtr = df_extract[vcol].values
            ntr = len(df_extract)
            vrgtr = np.copy(wt)
        twt = np.sum(vrgtr)
        # sort
        vrtr, vrgtr = dsortem(0, ntr, vrtr, 2, b=vrgtr)

        # Compute the cumulative probabilities and write transformation table
        twt = max(twt, EPSLON)
        oldcp = 0.0
        cp = 0.0
        #        print('ntr') print(ntr)
        for j in range(0, ntr):
            cp = cp + vrgtr[j] / twt
            w = (cp + oldcp) * 0.5
            vrg = gauinv(w)
            oldcp = cp
            # Now, reset the weight to the normal scores value:
            vrgtr[j] = vrg

        # Normal scores transform the data
        for id_ in range(0, nd):
            if itrans == 1:
                vrr = vr[id_]
                j = dlocate(vrtr, 1, nd, vrr)
                j = min(max(0, j), (nd - 2))
                vrg = dpowint(vrtr[j], vrtr[j + 1], vrgtr[j], vrgtr[j + 1], vrr, 1.0)
                if vrg < vrgtr[0]:
                    vrg = vrgtr[0]
                if vrg > vrgtr[nd - 1]:
                    vrg = vrgtr[nd - 1]
                vr[id_] = vrg

    weighted_stats_orig = DescrStatsW(vr_orig, weights=wt)
    orig_av = weighted_stats_orig.mean
    orig_ss = weighted_stats_orig.var

    weighted_stats = DescrStatsW(vr, weights=wt)
    av = weighted_stats.mean
    ss = weighted_stats.var

    print('\n Data for SGSIM: Number of acceptable data     = ' + str(nd))
    print('                 Number trimmed                = ' + str(len(df) - nd))
    print('                 Weighted Average              = ' + str(round(orig_av, 4)))
    print('                 Weighted Variance             = ' + str(round(orig_ss, 4)))
    print('                 Weighted Transformed Average  = ' + str(round(av, 4)))
    print('                 Weighted Transformed Variance = ' + str(round(ss, 4)))

    # Read in secondary data
    sim = np.random.rand(nx * ny)
    index = 0
    for ixy in range(0, nxy):
        sim[index] = index

    lvm = []
    lvm = np.array(lvm)
    if ktype >= 2:
        # lvm = np.copy(sec_map.flatten())
        ind = 0
        lvm = np.zeros(nxy)
        for iy in range(0, ny):
            for ix in range(0, nx):
                lvm[ind] = sec_map[ny - iy - 1, ix]
                ind = ind + 1
        if ktype == 2 and itrans == 1:
            for ixy in range(0, nxy):
                # Do we to transform the secondary variable for a local mean?
                vrr = lvm[ixy]
                j = dlocate(vrtr, 1, ntr, vrr)
                j = min(max(0, j), (ntr - 2))
                vrg = dpowint(vrtr[j], vrtr[j + 1], vrgtr[j], vrgtr[j + 1], vrr, 1.0)
                if vrg < vrgtr[0]:
                    vrg = vrgtr[0]
                if vrg > vrgtr[ntr - 1]:
                    vrg = vrgtr[nd - 1]
                lvm[ixy] = vrg
        av = np.average(lvm)
        ss = np.var(lvm)
        print(' Secondary Data: Number of data             = ' + str(nx * ny))
        print('                 Equal Weighted Average     = ' + str(round(av, 4)))
        print('                 Equal Weighted Variance    = ' + str(round(ss, 4)))

        # Do we need to work with data residuals? (Locally Varying Mean)
        if ktype == 2:
            sec = np.zeros(nd)
            for idd in range(0, nd):
                ix = getindex(nx, xmn, xsiz, x[idd])
                iy = getindex(ny, ymn, ysiz, y[idd])
                index = ix + (iy - 1) * nx
                sec[idd] = lvm[index]
        # Calculation of residual moved to krige subroutine: vr(i)=vr(i)-sec(i)

        # Transform the secondary attribute to normal scores?
        if ktype == 4:
            order_sec = np.zeros(nxy)
            ind = 0
            for ixy in range(0, nxy):
                order_sec[ixy] = ind
                ind = ind + 1
            print(' Transforming Secondary Data with')
            print(' Variance reduction of ' + str(varred))
            lvm, order_sec = dsortem(0, nxy, lvm, 2, b=order_sec)
            oldcp = 0.0
            cp = 0.0
            for i in range(0, nxy):
                cp = cp + (1.0 / nxy)
                w = (cp + oldcp) / 2.0
                lvm[i] = gauinv(w)
                lvm[i] = lvm[i] * varred
                oldcp = cp
            order_sec, lvm = dsortem(0, nxy, order_sec, 2, b=lvm)

    # Set up the rotation/anisotropy matrices that are needed for the
    # variogram and search.
    print('Setting up rotation matrices for variogram and search')
    if nst == 1:
        rotmat = setrot(ang[0], ang[0], sang1, anis[0], anis[0], sanis1, nst, MAXROT=2)
    else:
        rotmat = setrot(ang[0], ang[1], sang1, anis[0], anis[1], sanis1, nst, MAXROT=2)
    isrot = 2  # search rotation is appended as 3rd

    rotmat_2d, maxcov = setup_rotmat2(c0, nst, it, cc, ang)  # will use one in the future

    # Make a KDTree for fast search of nearest neighbours
    data_locs = np.column_stack((y, x))
    tree = sp.spatial.cKDTree(data_locs, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True)

    # Set up the covariance table and the spiral search:
    cov_table, tmp, order, ixnode, iynode, nlooku, nctx, ncty = ctable(MAXNOD, MAXCXY, MAXCTX, MAXCTY, MXY,
                                                                       xsiz, ysiz, isrot, nx, ny, nst, c0, cc, aa, it,
                                                                       ang, anis, rotmat, radsqd)

    #    print('Covariance Table') print(cov_table)
    # MAIN LOOP OVER ALL THE SIMULAUTIONS:
    for isim in range(0, nsim):

        # Work out a random path for this realization:
        sim = np.random.rand(nx * ny)
        order = np.zeros(nxy)
        ind = 0
        for ixy in range(0, nxy):
            order[ixy] = ind
            ind = ind + 1

        # The multiple grid search works with multiples of 4 (yes, that is
        # somewhat arbitrary):

        if mults == 1:
            for imult in range(0, nmult):
                nny = int(max(1, ny / ((imult + 1) * 4)))
                nnx = int(max(1, nx / ((imult + 1) * 4)))
                jy = 1
                jx = 1
                for iy in range(0, nny):
                    if nny > 0:
                        jy = iy * (imult + 1) * 4
                    for ix in range(0, nnx):
                        if nnx > 0:
                            jx = ix * (imult + 1) * 4
                        index = jx + (jy - 1) * nx
                        sim[index] = sim[index] - (imult + 1)

        # Initialize the simulation:
        sim, order = dsortem(0, nxy, sim, 2, b=order)
        sim.fill(UNEST)
        print('Working on realization number ' + str(isim))

        # Assign the data to the closest grid node:

        TINY = 0.0001
        for idd in range(0, nd):
            #            print('data') print(x[idd],y[idd])
            ix = getindex(nx, xmn, xsiz, x[idd])
            iy = getindex(ny, ymn, ysiz, y[idd])
            ind = ix + (iy - 1) * nx
            xx = xmn + ix * xsiz
            yy = ymn + iy * ysiz
            #            print('xx, yy' + str(xx) + ',' + str(yy))
            test = abs(xx - x[idd]) + abs(yy - y[idd])

            # Assign this data to the node (unless there is a closer data):
            if sstrat == 1:
                if sim[ind] > 0.0:
                    id2 = int(sim[ind] + 0.5)
                    test2 = abs(xx - x(id2)) + abs(yy - y(id2))
                    if test <= test2:
                        sim[ind] = idd
                    else:
                        sim[ind] = id2

            # Assign a flag so that this node does not get simulated:
            if sstrat == 0 and test <= TINY:
                sim[ind] = 10.0 * UNEST

        # Now, enter data values into the simulated grid:
        for ind in range(0, nxy):
            idd = int(sim[ind] + 0.5)
            if idd > 0:
                sim[ind] = vr[id]
        irepo = max(1, min((nxy / 10), 10000))

        # MAIN LOOP OVER ALL THE NODES:
        for ind in range(0, nxy):
            if (int(ind / irepo) * irepo) == ind:
                print('   currently on node ' + str(ind))

            # Figure out the location of this point and make sure it has
            # not been assigned a value already:

            index = int(order[ind] + 0.5)
            if (sim[index] > (UNEST + EPSLON)) or (sim[index] < (UNEST * 2.0)):
                continue
            iy = int(index / nx)
            ix = index - iy * nx
            xx = xmn + ix * xsiz
            yy = ymn + iy * ysiz
            current_node = (yy, xx)

            # Now, we'll simulate the point ix,iy,iz.  First, get the close data
            # and make sure that there are enough to actually simulate a value,
            # we'll only keep the closest "ndmax" data, and look for previously
            # simulated grid nodes:

            if sstrat == 0:
                #                print('searching for nearest data')
                if ndmax == 1:
                    dist = np.zeros(1)
                    nums = np.zeros(1)
                    dist[0], nums[0] = tree.query(current_node, ndmax)  # use kd tree for fast nearest data search
                else:
                    dist, nums = tree.query(current_node, ndmax)

                nums = nums[dist < radius]
                dist = dist[dist < radius]
                na = len(dist)
                if na < ndmin:
                    continue  # bail if not enough data

            ncnode, icnode, cnodev, cnodex, cnodey = srchnd(ix, iy, nx, ny, xmn, ymn, xsiz, ysiz, sim, noct, nodmax,
                                                            ixnode, iynode, nlooku, nctx, ncty, UNEST)

            nclose = na

            if ktype == 2:
                gmean = lvm[index]
            else:
                gmean = 0.0

            if nclose + ncnode < 1:
                cmean = gmean
                cstdev = 1.0

            # Perform the kriging.  Note that if there are fewer than four data
            # then simple kriging is prefered so that the variance of the
            # realization does not become artificially inflated:

            else:
                lktype = ktype
                if ktype == 1 and (nclose + ncnode) < 4:
                    lktype = 0
                cmean, cstdev = krige(ix, iy, nx, ny, xx, yy, lktype, x, y, vr, sec, colocorr, lvm, nums, cov_table,
                                      nctx, ncty, icnode, ixnode, iynode, cnodev, cnodex, cnodey,
                                      nst, c0, 9999.9, cc, aa, it, ang, anis, rotmat_2d, maxcov, MAXCTX, MAXCTY, MAXKR1,
                                      MAXKR2)

            # Draw a random number and assign a value to this node:
            p = np.random.rand()
            xp = gauinv(p)
            sim[index] = xp * cstdev + cmean
            #            print('simulated value = ' + str(sim[index]))

            # Quick check for far out results:
            if abs(cmean) > 5.0 or abs(cstdev) > 5.0 or abs(sim[index]) > 6.0:
                print('WARNING: grid node location: ' + str(ix) + ',' + str(iy))
                print('         conditional mean and stdev:  ' + str(cmean) + ',' + str(cstdev))
                print('         simulated value:    ' + str(sim[index]))

        # Do we need to reassign the data to the grid nodes?
        if sstrat == 0:
            print('Reassigning data to nodes')
            for iid in range(0, nd):
                ix = getindex(nx, xmn, xsiz, x[iid])
                iy = getindex(ny, ymn, ysiz, y[iid])
                xx = xmn + ix * xsiz
                yy = ymn + iy * ysiz
                ind = ix + (iy - 1) * nx
                test = abs(xx - x[iid]) + abs(yy - y[iid])
                if test <= TINY:
                    sim[ind] = vr[iid]

        # Back transform each value and write results:
        ne = 0
        av = 0.0
        ss = 0.0
        for ind in range(0, nxy):
            simval = sim[ind]
            if -9.0 < simval < 9.0:
                ne = ne + 1
                av = av + simval
                ss = ss + simval * simval
            if itrans == 1 and simval > (UNEST + EPSLON):
                simval = backtr_value(simval, vrtr, vrgtr, zmin, zmax, ltail, ltpar, utail, utpar)
                simval = zmin if simval < zmin else simval
                simval = zmax if simval > zmax else simval
            sim[ind] = simval

        av = av / max(ne, 1.0)
        ss = (ss / max(ne, 1.0)) - av * av
        print('\n Realization ' + str(isim) + ': number   = ' + str(ne))
        print('                                   mean     = ' + str(round(av, 4)) + ' (close to 0.0?)')
        print('                                   variance = ' + str(
            round(ss, 4)) + ' (close to gammabar(V,V)? approx. 1.0)')

        # END MAIN LOOP OVER SIMULATIONS:
        sim_out = np.zeros((ny, nx))
        for ind in range(0, nxy):
            iy = int(ind / nx)
            ix = ind - iy * nx
            sim_out[ny - iy - 1, ix] = sim[ind]

    return sim_out


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """
    Compute a weighted quantile.
    Source: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy

    Parameters
    ----------
    values: numpy array
    quantiles: numpy array
    sample_weight: numpy array with same length as values

    values_sorted: bool
        If True, then will avoid sorting of initial array.

    old_style: bool
        If True, the output will be consistent with numpy percentile.

    Returns
    -------
    numpy array with computed quantiles

    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_quantiles, values)


def _data_cleansing(dataframe, xcoor, ycoor, target):
    dataset = dataframe[['UWI', xcoor, ycoor, target]].copy()
    dataset.dropna(axis=0, inplace=True)

    return dataset


def _semivariogram_intersects(azimuths, lag, gamma):
    above_sill = gamma >= 1
    points_above_sill = np.argwhere(above_sill == 1)
    _, first_intersection = np.unique(points_above_sill[:, 1], return_index=True)
    points_above_sill = points_above_sill[first_intersection]
    points_below_sill = points_above_sill + [-1, 0]

    # find the gammas and lags
    gamma_points = np.zeros((2, len(azimuths)))
    lag_points = np.zeros((2, len(azimuths)))

    gamma_points[0, :] = gamma[points_below_sill[:, 0], points_below_sill[:, 1]]
    gamma_points[1, :] = gamma[points_above_sill[:, 0], points_above_sill[:, 1]]
    lag_points[0, :] = lag[points_below_sill[:, 0], points_below_sill[:, 1]]
    lag_points[1, :] = lag[points_above_sill[:, 0], points_above_sill[:, 1]]

    return gamma_points, lag_points


def _compute_range(gamma_points, lag_points):
    range_distance = np.zeros((gamma_points.shape[1]))
    for i in range(gamma_points.shape[1]):
        poly_c = np.polyfit(gamma_points[:, i], lag_points[:, i], deg=1)
        poly_c = np.poly1d(poly_c)
        range_distance[i] = poly_c(1)

    return range_distance[0]


def _make_variogram(nug, nst, it1, cc1, azi1, hmaj1, hmin1, it2=1, cc2=0, azi2=0, hmaj2=0, hmin2=0):
    if cc2 == 0:
        nst = 1
    var = dict(
        [
            ("nug", nug),
            ("nst", nst),
            ("it1", it1),
            ("cc1", cc1),
            ("azi1", azi1),
            ("hmaj1", hmaj1),
            ("hmin1", hmin1),
            ("it2", it2),
            ("cc2", cc2),
            ("azi2", azi2),
            ("hmaj2", hmaj2),
            ("hmin2", hmin2),
        ]
    )
    return var


def _declus(df, xcol, ycol, vcol, iminmax, noff, ncell, cmin, cmax):
    """
    GSLIB's DECLUS program (Deutsch and Journel, 1998) converted from the original Fortran to Python by Michael Pyrcz,
    the University of Texas at Austin (Jan, 2019).
    Parameters
    ----------
    df
    xcol
    ycol
    vcol
    iminmax
    noff
    ncell
    cmin
    cmax

    Returns
    -------

    """
    # Load data and set up arrays
    nd = len(df)
    x = df[xcol].values
    y = df[ycol].values
    v = df[vcol].values
    wt = np.zeros(nd)
    wtopt = np.ones(nd)
    index = np.zeros(nd, np.int32)
    xcs_mat = np.zeros(ncell + 2)  # we use 1,...,n for this array
    vrcr_mat = np.zeros(ncell + 2)  # we use 1,...,n for this array
    anisy = 1.0  # hard code the cells to 2D isotropic
    roff = float(noff)

    # Calculate extents
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    # Calculate summary statistics
    vmean = np.mean(v)
    xcs_mat[0] = 0.0
    vrcr_mat[0] = vmean
    vrop = vmean  # include the naive case

    # print(f"There are {nd} data with:")
    # print(f"   mean of      {vmean} ")
    # print(f"   min and max  {vmin} and {vmax}")
    # print(f"   standard dev {vstdev} ")

    # Define a "lower" origin to use for the cell sizes
    xo1 = xmin - 0.01
    yo1 = ymin - 0.01

    # Define the increment for the cell size
    xinc = (cmax - cmin) / ncell
    yinc = xinc

    # Loop over "ncell+1" cell sizes in the grid network
    ncellx = int((xmax - (xo1 - cmin)) / cmin) + 1
    ncelly = int((ymax - (yo1 - cmin * anisy)) / cmin) + 1
    ncellt = ncellx * ncelly
    cellwt = np.zeros(ncellt)
    xcs = cmin - xinc
    ycs = (cmin * anisy) - yinc

    # Main loop over cell sizes
    # 0 index is the 0.0 cell, note n + 1 in Fortran
    for lp in range(1, ncell + 2):
        xcs = xcs + xinc
        ycs = ycs + yinc

        # Initialize the weights to zero
        wt.fill(0.0)

        # Determine the maximum number of grid cells in the network
        ncellx = int((xmax - (xo1 - xcs)) / xcs) + 1

        # Loop over all the origin offsets selected
        xfac = min((xcs / roff), (0.5 * (xmax - xmin)))
        yfac = min((ycs / roff), (0.5 * (ymax - ymin)))
        for kp in range(1, noff + 1):
            xo = xo1 - (float(kp) - 1.0) * xfac
            yo = yo1 - (float(kp) - 1.0) * yfac

            # Initialize the cumulative weight indicators
            cellwt.fill(0.0)

            # Determine which cell each datum is in
            for i in range(0, nd):
                icellx = int((x[i] - xo) / xcs) + 1
                icelly = int((y[i] - yo) / ycs) + 1
                icell = icellx + (icelly - 1) * ncellx
                index[i] = icell
                cellwt[icell] = cellwt[icell] + 1.0

            # The weight assigned to each datum is inversely proportional to the
            # number of data in the cell. We first need to get the sum of
            # weights so that we can normalize the weights to sum to one
            sumw = 0.0
            for i in range(0, nd):
                ipoint = index[i]
                sumw = sumw + (1.0 / cellwt[ipoint])
            sumw = 1.0 / sumw

            # Accumulate the array of weights (that now sum to one)
            for i in range(0, nd):
                ipoint = index[i]
                wt[i] = wt[i] + (1.0 / cellwt[ipoint]) * sumw

        # end the loop over all offsets

        # Compute the weighted average for this cell size
        sumw = 0.0
        sumwg = 0.0
        for i in range(0, nd):
            sumw = sumw + wt[i]
            sumwg = sumwg + wt[i] * v[i]
        vrcr = sumwg / sumw
        vrcr_mat[lp] = vrcr
        xcs_mat[lp] = xcs

        # See if this weighting is optimal
        if iminmax and vrcr < vrop or not iminmax and vrcr > vrop or ncell == 1:
            vrop = vrcr
            wtopt = wt.copy()  # deep copy

    # End main loop over all cell sizes

    # Get the optimal weights
    sumw = 0.0
    for i in range(0, nd):
        sumw = sumw + wtopt[i]
    facto = float(nd) / sumw
    wtopt = wtopt * facto
    return wtopt, xcs_mat, vrcr_mat


def _cova2(x1, y1, x2, y2, nst, cc, aa, it, anis, rotmat, maxcov):
    """
    Simplified version from geostatspy.

    Args:
        x1:
        y1:
        x2:
        y2:
        nst:
        cc:
        aa:
        it:
        anis:
        rotmat:
        maxcov:

    Returns:

    """
    EPSLON = 0.000000
    # Check for very small distance
    dx = x2 - x1
    dy = y2 - y1
    # the sum of two squared numbers is never zero
    if (dx * dx + dy * dy) < EPSLON:
        cova2_ = maxcov
        return cova2_

    # Non-zero distance, loop over all the structures
    cova2_ = 0.0
    for js in range(0, nst):
        # Compute the appropriate structural distance
        dx1 = dx * rotmat[0, js] + dy * rotmat[1, js]
        dy1 = (dx * rotmat[2, js] + dy * rotmat[3, js]) / anis[js]
        h = np.sqrt(max((dx1 * dx1 + dy1 * dy1), 0.0))
        if it[js] == 1:
            # Spherical model
            hr = h / aa[js]
            if hr < 1.0:
                cova2_ += cc[js] * (1.0 - hr * (1.5 - 0.5 * hr * hr))
        elif it[js] == 2:
            # Exponential model
            cova2_ += cc[js] * np.exp(-3.0 * h / aa[js])
        elif it[js] == 3:
            # Gaussian model
            hh = -3.0 * (h * h) / (aa[js] * aa[js])
            cova2_ += cc[js] * np.exp(hh)

    return cova2_


def _vmodel(nlag, xlag, azm, vario):
    """
    GSLIB's VMODEL program (Deutsch and Journel, 1998) converted from the
    original Fortran to Python by Michael Pyrcz, the University of Texas at
    Austin (Mar, 2019).
    """

    # Parameters
    DEG2RAD = np.pi / 180.0

    # Declare arrays
    index = np.zeros(nlag + 1)
    h = np.zeros(nlag + 1)
    gam = np.zeros(nlag + 1)
    cov = np.zeros(nlag + 1)
    ro = np.zeros(nlag + 1)

    # Load the variogram
    nst = vario.get("nst")
    cc = np.zeros(nst)
    aa = np.zeros(nst)
    it = np.zeros(nst)
    ang = np.zeros(nst)
    anis = np.zeros(nst)

    c0 = vario.get("nug")
    cc[0] = vario.get("cc1")
    it[0] = vario.get("it1")
    ang[0] = vario.get("azi1")
    aa[0] = vario.get("hmaj1")
    anis[0] = vario.get("hmin1") / vario.get("hmaj1")
    if nst == 2:
        cc[1] = vario.get("cc2")
        it[1] = vario.get("it2")
        ang[1] = vario.get("azi2")
        aa[1] = vario.get("hmaj2")
        anis[1] = vario.get("hmin2") / vario.get("hmaj2")

    xoff = np.sin(DEG2RAD * azm) * xlag
    yoff = np.cos(DEG2RAD * azm) * xlag
    rotmat, maxcov = geostats.setup_rotmat(c0, nst, it, cc, ang, 99999.9)

    xx = 0.0
    yy = 0.0
    for il in range(0, nlag + 1):
        index[il] = il
        cov[il] = _cova2(0.0, 0.0, xx, yy, nst, cc, aa, it, anis, rotmat, maxcov)
        gam[il] = maxcov - cov[il]
        ro[il] = cov[il] / maxcov
        h[il] = np.sqrt(max((xx * xx + yy * yy), 0.0))
        xx = xx + xoff
        yy = yy + yoff

    return h, gam


def _check_rept_coords(dataset, xcoor, ycoor):
    """
    Check if you have repeated coordinates that could cause problems in SGS python version.

    Parameters
    ----------
    dataset: pandas.DataFrame
    xcoor, ycoor: str
        Name of the X and Y coordinates.

    """
    repeated_x = np.sum(dataset[xcoor].duplicated(keep=False))
    repeated_y = np.sum(dataset[ycoor].duplicated(keep=False))

    if repeated_x == repeated_y and repeated_x != 0:
        clean = False
        nwells = repeated_x
    else:
        clean = True
        nwells = None

    return clean, nwells


def _discrete_colorscale(bvals, colores):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """

    if len(bvals) != len(colores) + 1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)
    nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]  # normalized values

    dcolorscale = []  # discrete colorscale
    for k in range(len(colores)):
        dcolorscale.extend([[nvals[k], colores[k]], [nvals[k + 1], colores[k]]])

    bvals = np.array(bvals)
    # position with respect to bvals where ticktext is displayed
    tickvals = [np.round(np.mean(bvals[k:k + 2]), 2) for k in range(len(bvals) - 1)]
    ticktext = [f'<{bvals[1]}'] + [f'{bvals[k]}-{bvals[k + 1]}' for k in range(1, len(bvals) - 2)] + [
        f'>{bvals[-2]}']

    return dcolorscale, tickvals, ticktext


def _varmap(df, xcol, ycol, vcol, nx, ny, lagdist, minpairs, bstand):
    """
    Regular spaced data, 2D wrapper for varmap from GSLIB (.exe must be available in PATH or working directory).
    Parameters
    ----------
    df: pandas.DataFrame
        The input dataframe.

    xcol, ycol, vcol: str
        The column names of the horizontal and vertical coordinates, and the feature of interest.

    nx, ny: int
        The number of nodes in the x, y coordinate directions.

    lagdist: int
        The lag tolerances or ``cell sizes'' in the X, Y directions.

    minpairs: int
        The minimum number of pairs needed to define a variogram value (set to missing if fewer than minpairs is found).

    bstand: int
        If set to 1, the semivariogram values will be divided by the variance.

    Returns
    -------

    """
    df_ext = pd.DataFrame({"X": df[xcol], "Y": df[ycol], "Z": df[vcol]})
    Dataframe2GSLIB("varmap_out.dat", df_ext)

    with open("varmap.par", "w") as f:
        f.write("              Parameters for VARMAP                                        \n")
        f.write("              *********************                                        \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETERS:                                                       \n")
        f.write("varmap_out.dat          -file with data                                    \n")
        f.write("1   3                        -   number of variables: column numbers       \n")
        f.write("-1.0e21     1.0e21           -   trimming limits                           \n")
        f.write("0                            -1=regular grid, 0=scattered values           \n")
        f.write(" 50   50    1                -if =1: nx,     ny,   nz                      \n")
        f.write("1.0  1.0  1.0                -       xsiz, ysiz, zsiz                      \n")
        f.write("1   2   0                    -if =0: columns for x,y, z coordinates        \n")
        f.write("varmap.out                   -file for variogram output                    \n")
        f.write(str(nx) + " " + str(ny) + " 0 " + "-nxlag, nylag, nzlag                     \n")
        f.write(str(lagdist) + " " + str(lagdist) + " 1.0              -dxlag, dylag, dzlag \n")
        f.write(str(minpairs) + "             -minimum number of pairs                      \n")
        f.write(str(bstand) + "               -standardize sill? (0=no, 1=yes)              \n")
        f.write("1                            -number of variograms                         \n")
        f.write("1   1   1                    -tail, head, variogram type                   \n")

    os.system("varmap.exe varmap.par")
    nnx = nx * 2 + 1
    nny = ny * 2 + 1
    varmap, _ = GSLIB2ndarray("varmap.out", 0, nnx, nny)

    return varmap


def _sgsim(nreal, df, xcol, ycol, vcol, var_min, var_max, nx, ny, hmnx, hmny, hsiz, seed, var,
           max_range, output_file, ndmax, ktype=0, rho=0.6, varred=1.0, secfl="none.dat", col_secvar=0):
    x = df[xcol]
    y = df[ycol]
    v = df[vcol]
    if col_secvar == 0:  # no secondary variable
        df_temp = pd.DataFrame({"X": x, "Y": y, "Var": v})
    else:  # cosimulation
        sec_feat = df['Sec_feat']
        df_temp = pd.DataFrame({"X": x, "Y": y, "Var": v, "Sec_feat": sec_feat})

    Dataframe2GSLIB("data_temp.dat", df_temp)
    nug = var.get("nug")
    nst = var.get("nst")
    it1 = var.get("it1")
    cc1 = var.get("cc1")
    azi1 = var.get("azi1")
    hmaj1 = var.get("hmaj1")
    hmin1 = var.get("hmin1")
    it2 = var.get("it2")
    cc2 = var.get("cc2")
    azi2 = var.get("azi2")
    hmaj2 = var.get("hmaj2")
    hmin2 = var.get("hmin2")

    # max_range = max(hmaj1, hmaj2)
    hctab = int(max_range / hsiz) * 2 + 1

    ndmax = str(ndmax)

    with open("sgsim.par", "w") as f:
        f.write("              Parameters for SGSIM                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("data_temp.dat                 -file with data                              \n")  # datafl
        f.write("1  2  0  3  0  " + str(col_secvar) + "              -  columns for X,Y,Z,vr,wt,sec.var.  \n")
        f.write("-999. 999.                    -  trimming limits                           \n")
        f.write("1                             -transform the data (0=no, 1=yes)            \n")
        f.write("none.trn                      -  file for output trans table               \n")
        f.write("0                             -  consider ref. dist (0=no, 1=yes)          \n")
        f.write("none.dat                      -  file with ref. dist distribution          \n")
        f.write("1  0                          -  columns for vr and wt                     \n")
        f.write(str(var_min) + " " + str(var_max) + "   zmin,zmax(tail extrapolation)       \n")
        f.write("1   " + str(var_min) + "      -  lower tail option, parameter              \n")
        f.write("1   " + str(var_max) + "      -  upper tail option, parameter              \n")
        f.write("0                             -debugging level: 0,1,2,3                    \n")
        f.write("nonw.dbg                      -file for debugging output                   \n")
        f.write(str(output_file) + "           -file for simulation output                  \n")
        f.write(str(nreal) + "                 -number of realizations to generate          \n")
        f.write(str(nx) + " " + str(hmnx) + " " + str(hsiz) + "                             \n")
        f.write(str(ny) + " " + str(hmny) + " " + str(hsiz) + "                             \n")
        f.write("1 0.0 1.0                     - nz zmn zsiz                                \n")
        f.write(str(seed) + "                  -random number seed                          \n")
        f.write("0     " + ndmax + "           -min and max original data for sim           \n")  # ndmin, ndmax
        f.write("12                            -number of simulated nodes to use            \n")
        f.write("0                             -assign data to nodes (0=no, 1=yes)          \n")
        f.write("1     3                       -multiple grid search (0=no, 1=yes),num      \n")
        f.write("0                             -maximum data per octant (0=not used)        \n")
        f.write(str(max_range) + " " + str(max_range) + " 1.0 -maximum search  (hmax,hmin,vert) \n")
        f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
        f.write(str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n")
        f.write(str(ktype) + "   " + str(rho) + "   " + str(varred) + "   -ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC \n")
        f.write(secfl + "                      -  file with LVM, EXDR, or COLC variable     \n")
        f.write(str(col_secvar) + "            -  column for secondary variable             \n")
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(" " + str(hmaj1) + " " + str(hmin1) + " 1.0 - a_hmax, a_hmin, a_vert        \n")
        f.write(str(it2) + " " + str(cc2) + " " + str(azi2) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(" " + str(hmaj2) + " " + str(hmin2) + " 1.0 - a_hmax, a_hmin, a_vert        \n")

    os.system("sgsim.exe sgsim.par")
    sim_array = GSLIB2ndarray(output_file, 0, nx, ny)

    return sim_array[0]


def _assign_array_to_df(dataframe, xcoor, ycoor, root, sec_feat, starts_with):
    """
    Assign the values from a 2D seismic surface to the well locations and transform the ASCII to a numpy array.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        The dataset containing UWI, well coordinates and the primary feature.

    xcoor, ycoor: str
        Names of the x and y coordinates in the dataframe.

    root: str
        Location of the seismic surface as ASCII file.

    sec_feat: str
        Name of the feature from the seismic surface.

    starts_with: str
        The sytring used for commenting.

    Returns
    -------
    pandas.DataFrame
        The original dataset plus the additional column contianing the values from the seismic surface.

    numpy.ndarray
        The ASCII file as a 2d numpy array.
    """
    with open(root, encoding='utf-8-sig') as f:
        lines = (line for line in f if not line.startswith(starts_with))
        FH = np.loadtxt(lines, delimiter=' ', skiprows=0)

    FH[FH == 0] = -9999
    min_x = np.min(FH[:, 0])
    min_y = np.min(FH[:, 1])
    max_x = np.max(FH[:, 0])
    max_y = np.max(FH[:, 1])

    denom_x = np.min(np.diff(np.unique(FH[:, 0])))
    denom_y = np.min(np.diff(np.unique(FH[:, 1])))
    half_x = int(denom_x / 2)
    half_y = int(denom_y / 2)

    Xr = np.arange(min_x - half_x, max_x + half_x, denom_x)
    Yr = np.arange(min_y - half_y, max_y + half_y, denom_y)
    data = np.histogram2d(FH[:, 1], FH[:, 0], bins=[len(Yr), len(Xr)], weights=FH[:, 2])[0]

    indices_x = (np.ceil((dataframe[xcoor] - min_x) / denom_x)).astype(int)
    indices_y = (np.ceil((dataframe[ycoor] - min_y) / denom_y)).astype(int)
    indices_x = np.where(indices_x >= len(Xr), len(Xr) - 1, indices_x)
    indices_y = np.where(indices_y >= len(Yr), len(Yr) - 1, indices_y)
    dataframe[sec_feat] = data[indices_y, indices_x].reshape(-1, 1)

    surface = np.where(data == 0., np.nan, data)
    surface = np.where(surface == -9999., 0, surface)

    return dataframe, surface, min_x, max_x, min_y, max_y, denom_x, denom_y


def resize_seismic(dataframe, xcoor, ycoor, root, class_object, sec_feat='AI', starts_with='#'):
    """
    Read the seismic ASCII file. Then resize and center it, so it is compatible with spatial modeling.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Dataset containing the coordinates and the main feature.

    xcoor, ycoor: str
        Column names of the coordinates in the dataframe.

    root: str
        Location of the seismic surface as ASCII file.

    class_object: SpatialModeler object
        SpatialModeler class AFTER runnning the 'model_trend' nested function.

    sec_feat: str
        Name of the feature from the seismic surface.

    starts_with: str
        The sytring used for commenting.

    Returns
    -------

    """

    data2, seismic, min_x, max_x, min_y, max_y, denom_x, denom_y = _assign_array_to_df(
        dataframe, xcoor, ycoor, root, sec_feat, starts_with)

    indices = np.zeros(shape=(2, 2))
    indices[0, :] = (np.ceil((class_object.coord_origin[:2] - min_x) / denom_x))
    indices[1, :] = (np.ceil((class_object.coord_origin[2:] - min_y) / denom_y))
    indices = indices.astype(int)

    resized_seismic = np.ones_like(class_object.trend_array) * np.nan
    origin_seismic = np.array([[min_x, max_x], [min_y, max_y]])
    origin_tm = np.array([
        [class_object.coord_origin[0], class_object.coord_origin[1]],
        [class_object.coord_origin[2], class_object.coord_origin[3]]]
    )
    optimal_bool = np.array([[True, False], [True, False]])
    actual_bool = (origin_tm - origin_seismic) < 0
    diff_bool = optimal_bool == actual_bool
    diff_bool = np.argwhere(diff_bool == True)
    tm_seismic_indx = np.zeros((2, 2, 2))
    tm_seismic_indx[1] = indices
    for yi, xi in zip(diff_bool[:, 0], diff_bool[:, 1]):
        if indices[yi, xi] < 0:
            tm_seismic_indx[1, yi, xi] = 0
        elif indices[yi, xi] > seismic.shape[0]:
            if yi == 1:
                tm_seismic_indx[1, yi, xi] = seismic.shape[0]
            elif xi == 0:
                tm_seismic_indx[1, yi, xi] = seismic.shape[1]

    # indices for our model
    horiz_bool = (0 <= indices[0, :]) & (indices[0, :] <= seismic.shape[1])
    ver_bool = (0 <= indices[1, :]) & (indices[1, :] <= seismic.shape[0])
    actual_bool = np.vstack((horiz_bool, ver_bool))
    diff_bool = np.argwhere(actual_bool == False)
    tm_seismic_indx[0] = np.array([[0, resized_seismic.shape[0]], [0, resized_seismic.shape[0]]])
    for yi, xi in zip(diff_bool[:, 0], diff_bool[:, 1]):
        if indices[yi, xi] < 0:
            tm_seismic_indx[0, yi, xi] = np.abs(indices[yi, xi])
        elif indices[yi, xi] > seismic.shape[0]:
            if yi == 1:
                tm_seismic_indx[0, yi, xi] = seismic.shape[0] + np.abs(indices[yi, xi - 1])
            elif xi == 0:
                tm_seismic_indx[0, yi, xi] = seismic.shape[1] + np.abs(indices[yi, xi - 1])

    tm_seismic_indx = tm_seismic_indx.astype(int)

    resized_seismic[tm_seismic_indx[0, 1, 0]:tm_seismic_indx[0, 1, 1],
    tm_seismic_indx[0, 0, 0]:tm_seismic_indx[0, 0, 1]] = seismic[
                                                         tm_seismic_indx[1, 1, 0]:tm_seismic_indx[1, 1, 1],
                                                         tm_seismic_indx[1, 0, 0]:tm_seismic_indx[1, 0, 1]
                                                         ]

    return resized_seismic, data2


def interpolate_surface(seismic, size, kernel=4, stride=4):
    """
    Interpolate the resulting seismic surface to the final size the realizations will take.

    Parameters
    ----------
    seismic: numpy.ndarray
        Resulting surface from resize seismic function.

    size: int
        Size of the realizations (from model_trend).

    kernel: int
        Size of the convolving kernel.

    stride: int
        Stride of the convolution.

    Returns
    -------
    The resized seismic with shape (size, size).
    """
    # add the batch and channel dimensions
    resized_seismic = torch.Tensor(seismic).unsqueeze(0).unsqueeze(0)
    # remove batch and channel dimensions
    resized_seismic = nn.functional.avg_pool2d(resized_seismic, kernel, stride=stride)
    resized_seismic = nn.functional.interpolate(resized_seismic, size=size).numpy()[0, 0, ...]

    return resized_seismic


def _outlier_methods(feature_array, feat_std, contamination, verbose):
    # Method 1: Standard Deviation Method (traditional)
    mask1 = np.abs(feature_array - np.mean(feature_array)) <= (3 * np.std(feature_array))
    # delete all rows that have NaNs
    outliers1 = feature_array[mask1]

    # Method 2: Isolation forest
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=13)
    yhat = iso.fit_predict(feat_std)
    mask2 = yhat != -1
    outliers2 = feature_array[mask2]

    # Method 3: Minimum Covariance Determinant
    ee = EllipticEnvelope(assume_centered=True, contamination=contamination, random_state=13)
    yhat = ee.fit_predict(feat_std)
    mask3 = yhat != -1
    outliers3 = feature_array[mask3]

    # Method 4: Local Outlier factor
    lof = LocalOutlierFactor(contamination=contamination)
    yhat = lof.fit_predict(feat_std)
    mask4 = yhat != -1
    outliers4 = feature_array[mask4]

    outliers = np.concatenate((outliers1.reshape(-1, 1), outliers2.reshape(-1, 1), outliers3.reshape(-1, 1),
                               outliers4.reshape(-1, 1), feature_array))

    indexes = [len(outliers1), len(outliers1) + len(outliers2), len(outliers1) + len(outliers2) + len(outliers3),
               len(outliers1) + len(outliers2) + len(outliers3) + len(outliers4)]

    masks = [mask1, mask2, mask3, mask4]

    if verbose:
        print('Number of points before outliers removed :', len(feat_std))
        print('Percentage of points removed with:')
        print(f'Standard Deviation: {1 - (len(outliers1) / len(feat_std)):.4%}')
        print(f'Isolation forest  : {1 - (len(outliers2) / len(feat_std)):.4%}')
        print(f'Elliptic Envelope : {1 - (len(outliers3) / len(feat_std)):.4%}')
        print(f'Outlier factor    : {1 - (len(outliers4) / len(feat_std)):.4%}')

    return outliers, indexes, masks


class UtilityFunctions:
    def __init__(self, dataset, xcoor, ycoor, feature):
        self.dataset = dataset
        self.xcoor = xcoor
        self.ycoor = ycoor
        self.feature = feature

    def half_distance_area(self):
        # Find the minimum and maximum distances
        xmax = self.dataset[self.xcoor].max()
        ymax = self.dataset[self.ycoor].max()
        xmin = self.dataset[self.xcoor].min()
        ymin = self.dataset[self.ycoor].min()

        distx = (xmax - xmin) / 2
        disty = (ymax - ymin) / 2
        if distx > disty:
            distance = distx
        else:
            distance = disty

        return distance

    def compute_lag_dist(self):
        xset = self.dataset[[self.xcoor, self.ycoor]].to_numpy()
        neighbors = NearestNeighbors(n_neighbors=2)
        neighbors_fit = neighbors.fit(xset)
        distances, _ = neighbors_fit.kneighbors(xset)

        lag_dist = np.mean(distances[:, 1:], axis=0)

        return lag_dist[0]

    def vario_plot(self, lag_distance, half_distance_area_interest, azimuths=None, extend_half_dist=1.0,
                   lag_tol_factor=2, band_factor=2):
        """
        Plot the experimental and directional variograms.

        Parameters
        ----------
        """
        iazi = None

        if azimuths is None:
            azimuths = [0, 30, 45, 60]
        tempo = [i + 90 if i < 90 else i - 90 for i in azimuths]
        azimuths += tempo  # add 90 degrees to directions you chose

        lag_tol = lag_distance * lag_tol_factor
        bandh = lag_distance * band_factor
        tmin = -999.
        tmax = 999.

        nlag = int((half_distance_area_interest * extend_half_dist) / lag_distance)
        # Arrays to store the results
        lag = np.zeros((len(azimuths), nlag + 2))
        gamma = np.zeros((len(azimuths), nlag + 2))
        npp = np.zeros((len(azimuths), nlag + 2))

        # Loop over all directions
        for iazi in range(0, len(azimuths)):
            lag[iazi, :], gamma[iazi, :], npp[iazi, :] = geostats.gamv(
                self.dataset, self.xcoor, self.ycoor, self.feature, tmin, tmax, lag_distance,
                lag_tol, nlag, azimuths[iazi], 22.5, bandh, isill=1
            )

        simbolo = ['circle', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star', 'x', 'square']
        colores = ['blue', 'red', 'black', 'green', 'blue', 'red', 'black', 'green']

        # Create traces
        fig = make_subplots(rows=1, cols=2)
        # First subplot
        for iazi in range(0, 4):
            fig.add_trace(go.Scatter(
                x=np.round(lag[iazi, :-1], 2),
                y=np.round(gamma[iazi, :], 2),
                mode='markers',
                name='Azimuth:' + str(azimuths[iazi]),
                hovertemplate='<br><b>Gamma</b>: %{y:.2f}<br>' +
                              '<b>Lag distance</b>: %{x:.2f}<br>' +
                              '<b>%{text}</b>',
                text=[f'Number of pairs {i:.0f}' for i in npp[iazi, :]],
                marker=dict(size=7, symbol=simbolo[iazi],
                            color=colores[iazi])),
                row=1, col=1)

        # Second subplot
        for iazi in range(4, 8):
            fig.add_trace(go.Scatter(
                x=np.round(lag[iazi, :-1], 2),
                y=np.round(gamma[iazi, :], 2),
                mode='markers',
                name='Azimuth' + str(azimuths[iazi - 4]) + '\u00B1 90 :' + str(azimuths[iazi]),
                hovertemplate='<br><b>Gamma</b>: %{y:.2f}<br>' +
                              '<b>Lag distance</b>: %{x:.2f}<br>' +
                              '<b>%{text}</b>',
                text=[f'Number of pairs {i:.0f}' for i in npp[iazi, :]],
                marker=dict(size=7, symbol=simbolo[iazi],
                            color=colores[iazi])),
                row=1, col=2)

        # add the sill to both subplots
        fig.add_trace(go.Scatter(x=[0, 5e9], y=[1.0, 1.0],
                                 line=dict(color='firebrick', width=3, dash='dot'),
                                 name='Sill', text=npp[iazi, :],
                                 showlegend=False),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=[0, 5e9], y=[1.0, 1.0],
                                 # dash options include 'dash', 'dot', and 'dashdot'
                                 line=dict(color='firebrick', width=3, dash='dot'),
                                 name='Sill', text=npp[iazi, :],
                                 showlegend=False),
                      row=1, col=2)

        # Update xaxis properties
        fig.update_xaxes(title_text="Lag distance <b>h(m)<b>",
                         row=1, range=[0, half_distance_area_interest * extend_half_dist], col=1)
        fig.update_xaxes(title_text="Lag distance <b>h(m)<b>",
                         row=1, range=[0, half_distance_area_interest * extend_half_dist], col=2)

        # Update yaxis properties
        fig.update_yaxes(title_text="<b>\u03B3<b>", row=1, range=[0, np.max(gamma) * 1.02],
                         col=1)
        fig.update_yaxes(title_text="<b>\u03B3<b>", row=1, range=[0, np.max(gamma) * 1.02],
                         col=2)

        # Edit the plot
        fig.update_layout(title='Directional ' + self.feature + ' Variogram',
                          autosize=False,
                          width=950,
                          height=500,
                          template='simple_white', )

        fig.show()

    def declus(self, lag_distance, max_cell, min_cell=None, maximize=None):
        """
        Compute the declustered mean to correct for sampling bias.
        Parameters
        lag_distance:
        ----------
        min_cell, max_cell: float
        Minimum and maximum cell sizes to use as bounds to compute the declustered mean. Use the recommended lag
        distance for the minium cell size.

        iminmax: bool
        0 to maximize, 1 to minimize the mean.
        """
        if maximize is None:
            skewness = skew(self.dataset[self.feature], bias=True)
            if skewness >= 0:  # there are smaller values wrt the mean
                iminmax = 0  # maximize
            else:
                iminmax = 1  # minimize
        elif maximize is True:
            iminmax = 0
        else:
            iminmax = 1

        if min_cell is None:
            min_cell = lag_distance

        _declus_wts, _, _ = _declus(self.dataset, self.xcoor, self.ycoor, self.feature, iminmax,
                                    noff=10, ncell=100, cmin=min_cell, cmax=max_cell)

        weighted_stats = DescrStatsW(self.dataset[self.feature], weights=_declus_wts, ddof=0)
        declus_mean = weighted_stats.mean
        _declus_var = weighted_stats.var

        print(f"The declustered mean is      : {declus_mean:.3f}")
        print(f"The declustered variance is  : {_declus_var:.3f}\n")

        print(f"The naive mean is            : {self.dataset[self.feature].mean():.3f}")
        print(f"The naive variance is        : {self.dataset[self.feature].var():.3f}\n")

        perc_correction = (declus_mean - self.dataset[self.feature].mean()) / self.dataset[self.feature].mean()
        print(f"The correction in means is   : {perc_correction:.2%}")

        # todo print optimal cell size for declustering

        return _declus_wts

    @staticmethod
    def vario_model_plot(vario, xrange, experimental_vario_tensor, azm1, azm2, nlag, xlag):
        title = str("Experimental Semivariogram vs Semivariogram Model")
        if nlag is None:
            nlag = int(xrange / 100)  # original
        h_major, gam_major = _vmodel(nlag, xlag, azm1, vario)
        h_minor, gam_minor = _vmodel(nlag, xlag, azm2, vario)

        major_range = vario.get("hmaj2")
        minor_range = vario.get("hmin2")

        ###########################
        # plotting
        ###########################
        fig = make_subplots(rows=1, cols=2)
        # major exper
        fig.add_trace(go.Scatter(
            x=experimental_vario_tensor[0, :, 0],
            y=experimental_vario_tensor[0, :, 1],
            mode='markers',
            name='Major: azimuth=' + str(np.round(azm1, 2)) + "; range=" + str(np.round(major_range, 2)),
            hovertemplate='<br><b>Gamma</b>: %{y:.2f}<br>' +
                          '<b>Lag distance</b>: %{x:.2f}<br>' +
                          '<b>%{text}</b>',
            text=[f'Number of pairs {i:.0f}' for i in experimental_vario_tensor[0, :, 2]],
            marker=dict(size=8, symbol='square', color='steelblue')),
            row=1, col=1)

        # major model
        fig.add_trace(go.Scatter(
            x=h_major,
            y=gam_major,
            mode='lines',
            name='Model',
            showlegend=False,
            line=dict(width=4, color='firebrick')),
            row=1, col=1)

        # minor exper
        fig.add_trace(go.Scatter(
            x=experimental_vario_tensor[1, :, 0],
            y=experimental_vario_tensor[1, :, 1],
            mode='markers',
            name='Minor: azimuth=' + str(np.round(azm2, 2)) + "; range=" + str(np.round(minor_range, 2)),
            hovertemplate='<br><b>Gamma</b>: %{y:.2f}<br>' +
                          '<b>Lag distance</b>: %{x:.2f}<br>' +
                          '<b>%{text}</b>',
            text=[f'Number of pairs {i:.0f}' for i in experimental_vario_tensor[1, :, 2]],
            marker=dict(size=8, symbol='circle', color='steelblue')),
            row=1, col=2)

        # minor model
        fig.add_trace(go.Scatter(
            x=h_minor,
            y=gam_minor,
            mode='lines',
            name='Model',
            showlegend=False,
            line=dict(width=4, color='firebrick')),
            row=1, col=2)

        # add the sill to both subplots
        fig.add_trace(go.Scatter(
            x=[0, xrange],
            y=[1.0, 1.0],
            line=dict(color='black', width=3, dash='dot'),
            showlegend=False,
            name='Sill'),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[0, xrange],
            y=[1.0, 1.0],
            line=dict(color='black', width=3, dash='dot'),
            showlegend=False,
            name='Sill'),
            row=1, col=2)

        # Update xaxis properties
        fig.update_xaxes(title_text="Lag distance <b>h(m)<b>", range=[0, xrange], row=1, col=1)
        fig.update_xaxes(title_text="Lag distance <b>h(m)<b>", range=[0, xrange], row=1, col=2)

        # Update yaxis properties
        maximum_major = np.max(experimental_vario_tensor[0, :, 1])
        maximum_minor = np.max(experimental_vario_tensor[1, :, 1])
        max_y = np.maximum(maximum_major, maximum_minor) * 1.2

        fig.update_yaxes(title_text="<b>\u03B3<b>", row=1, range=[0, max_y], col=1)  # <> is html formatting
        fig.update_yaxes(title_text="<b>\u03B3<b>", row=1, range=[0, max_y], col=2)

        fig.update_layout(title=title,
                          autosize=False,
                          width=950,
                          height=500,
                          template='simple_white', )

        fig.show()


class OutlierDetection:
    """
    Class to compute and plot the outliers based on Mahalanobis distance.
    """

    def __init__(self, dataframe, xcoor, ycoor, feature, directory):

        self.df = dataframe
        self.xcoor = xcoor
        self.ycoor = ycoor
        self.feature = feature
        self.directory = directory

    def outlier_plot(self, df_datypes, xy_ratio, width=700, height=700):
        df_datypes['Symbol'] = 'circle'
        df_datypes['marker_size'] = 7
        indices = df_datypes[df_datypes['Data'] == 'Outlier'].index
        df_datypes.loc[indices, 'Symbol'] = 'x'
        df_datypes.loc[indices, 'marker_size'] = 15

        hover_text = []
        bubble_size = []

        for index, row in df_datypes.iterrows():
            hover_text.append(('UWI: {uwi_}<br>' +
                               'X: {xcoor_}<br>' +
                               'Y: {ycoor_}<br>' +
                               'Feature: {feat_}<br>' +
                               'Classification: {classi}').format(
                uwi_=row['UWI'],
                xcoor_=row[self.xcoor],
                ycoor_=row[self.ycoor],
                feat_=row[self.feature],
                classi=row['Data']))
            bubble_size.append(np.sqrt(row[self.feature]))

        df_datypes['text'] = hover_text
        df_datypes['size'] = bubble_size

        # Dictionary with dataframes for each dftypes
        data_types = ['Not outlier', 'Outlier']
        dftypes_data = {datype: df_datypes.query("Data == '%s'" % datype)
                        for datype in data_types}

        zmin = df_datypes[self.feature].min()
        zmax = df_datypes[self.feature].max()

        show_scale_dict = {'Outlier': True, 'Not outlier': False}

        # plot
        fig = go.Figure()

        for outlier_class, df_class in dftypes_data.items():
            fig.add_trace(go.Scatter(
                x=df_class[self.xcoor],
                y=df_class[self.ycoor],
                name=outlier_class,
                text=df_class['text'],
                marker_symbol=df_class['Symbol'],
                marker_size=df_class['marker_size'],
                showlegend=True,
                mode='markers',
                marker=dict(
                    color=df_class[self.feature],
                    # showscale=show_scale_dict.get(outlier_class),
                    showscale=True,
                    colorscale="Inferno",
                    cmin=zmin,
                    cmax=zmax,
                    colorbar=dict(thickness=20),
                )))

        fig.update_layout(
            title=self.feature + " | Not-outlier: O; Outlier: X",
            xaxis=dict(
                title=self.xcoor,
                gridcolor='white',
                gridwidth=2,
            ),
            yaxis=dict(
                title=self.ycoor,
                gridcolor='white',
                gridwidth=2,
            ),
            coloraxis=dict(colorscale='Inferno'),
            autosize=False,
            width=width,
            height=height,
        )

        fig.update_xaxes(
            scaleanchor='y',
            scaleratio=xy_ratio,
        )
        fig.update_coloraxes(colorscale='Inferno')

        fig.show()

    def spatial_outliers(self, alpha, plot=True, update_data=False, xy_ratio=1.0):
        """
        Use the Mahalanobis distance to detect spatial outliers.

        Parameters
        ----------
        alpha: float
        Confidence value between 0. and 1.0. The smaller, the fewer values are outliers.

        plot: bool
        True to plot the classified data.

        update_data: bool
        True if you want to remove the outliers points from the dataset.

        xy_ratio: float

        Returns
        -------
        The lag distance
        """

        x = self.df[[self.xcoor, self.ycoor]].to_numpy()
        x_minus_mu = x - np.mean(x, axis=0)
        cov = np.cov(x.T)
        inv_covmat = sp.linalg.inv(cov)
        left_term = np.dot(x_minus_mu, inv_covmat)
        mahala = np.dot(left_term, x_minus_mu.T)
        mahala = mahala.diagonal()

        df2 = self.df.copy()
        df2['Mahal'] = mahala
        index_outlier = df2[df2['Mahal'] > chi2.ppf((1 - alpha), df=2)].index
        df2['Data'] = "Not outlier"
        df2.loc[index_outlier, 'Data'] = 'Outlier'

        if update_data:
            self.df.drop(index=index_outlier, inplace=True)

        if plot:
            self.outlier_plot(df2, xy_ratio)

    def feature_outlier(self, contamination, selection, update_data, plot, verbose):
        scaler = PowerTransformer(method='yeo-johnson')
        feature_array = self.df[self.feature].to_numpy().reshape(-1, 1)
        feat_std = scaler.fit_transform(feature_array)

        outliers, indexes, masks = _outlier_methods(feature_array, feat_std, contamination, verbose)

        outlier_df = pd.DataFrame(outliers, columns=[self.feature])
        outlier_df['Method'] = 'Std'

        outlier_df.loc[indexes[0]:indexes[1], 'Method'] = 'Isolation forest'
        outlier_df.loc[indexes[1]:indexes[2], 'Method'] = 'Elliptic envelope'
        outlier_df.loc[indexes[2]:indexes[3], 'Method'] = 'Outlier factor'
        outlier_df.loc[indexes[3]:, 'Method'] = 'Original feature'

        if update_data:
            if selection == 0:
                self.df = self.df.iloc[masks[0], :]
            elif selection == 1:
                self.df = self.df.iloc[masks[1], :]
            elif selection == 2:
                self.df = self.df.iloc[masks[2], :]
            else:
                self.df = self.df.iloc[masks[3], :]

        if plot:
            fig = px.violin(outlier_df, 'Method', self.feature, points='all', box=True)
            fig.show()


class TrendModeler:
    def __init__(self, dataframe, xcoor, ycoor, target, cell_size, declus_mean, declus_variance, declus_weights,
                 additional_cells=15):
        """

        :param dataframe:
        :param xcoor:
        :param ycoor:
        :param target:
        :param cell_size:
        """
        dataset = _data_cleansing(dataframe, xcoor, ycoor, target)
        self._xcoor = xcoor
        self._ycoor = ycoor
        self.target = target
        self._cell_size = cell_size
        self.dataset = dataset
        self.additional_cells = additional_cells

        # C order flattening
        self.xmin, self.xmax, self.ymin, self.ymax = self._max_and_min()
        xi = (np.floor((dataset[self._xcoor] - self.xmin) / self._cell_size)).astype(int)
        yi = (np.floor((dataset[self._ycoor] - self.ymin) / self._cell_size)).astype(int)
        ncells = int((self.ymax - self.ymin) / self._cell_size)
        dataset['Cell x'] = xi
        dataset['Cell y'] = yi
        dataset['Cell id'] = ncells * yi + xi
        self.dataset = dataset

        self.df_grid = DataFrame2ndarray(self.dataset, self._xcoor, self._ycoor, self.target, self.xmin,
                                         self.xmax, self.ymin, self.ymax, self._cell_size)

        # get the variance and mean of the feature of interest
        self._declus_weights = declus_weights
        self._declus_variance = declus_variance
        self._declus_mean = declus_mean

        # results
        self.xwindow = self.additional_cells
        self.ywindow = None
        self.final_trend = None

    def _max_and_min(self):
        """
        Compute the maximum and minimum values of a rectangle for modeling
        """
        # get the range in each dimension
        range_x = np.round(np.ptp(self.dataset[self._xcoor]))
        range_y = np.round(np.ptp(self.dataset[self._ycoor]))

        # add additional length to the smaller axis so the cell size is a multiple of the total length
        if range_x > range_y:
            xmax = np.round(self.dataset[self._xcoor].max()) + (self._cell_size * self.additional_cells)
            xmin = np.round(self.dataset[self._xcoor].min()) - (self._cell_size * self.additional_cells)

            range_x = np.round(xmax - xmin)
            half = np.abs(range_x - range_y) / 2

            ymin = np.round(self.dataset[self._ycoor].min()) - half
            ymax = np.round(self.dataset[self._ycoor].max()) + half
        else:
            ymax = np.round(self.dataset[self._ycoor].max()) + (self._cell_size * self.additional_cells)
            ymin = np.round(self.dataset[self._ycoor].min()) - (self._cell_size * self.additional_cells)

            range_y = np.round(ymax - ymin)
            half = np.abs(range_x - range_y) / 2

            xmin = np.round(self.dataset[self._xcoor].min()) - half
            xmax = np.round(self.dataset[self._xcoor].max()) + half

        if (xmax - xmin) % self._cell_size > 0:  # the cell size is not exact
            cells = int(np.ceil((xmax - xmin) / self._cell_size))
            half = (cells * self._cell_size - (xmax - xmin)) / 2
            ymin -= half
            ymax += half
            xmax += half
            xmin -= half

        return xmin, xmax, ymin, ymax

    def _convolve_2d(self, x_window, y_window, theta):
        """
        Convolve the sparse data with a Gaussian kernel
        :param x_window: Window size in the X direction
        :type x_window: float
        :param y_window: Window size in the Y direction
        :type y_window: float
        :return:
        """
        theta_radians = theta * np.pi / 180
        kernel = Gaussian2DKernel(x_window, y_window, theta_radians)
        trend_array = convolve(
            self.df_grid,
            kernel,
            boundary='extend',
            nan_treatment='interpolate',
            normalize_kernel=True
        )

        return trend_array

    def _mapping_to_cells(self, array):
        """
        Get the trend values at each cell using their cell ID.

        Parameters
        ----------
        array: numpy array
        Two-dimensional numpy array containing the trend map

        Returns
        -------
        The trend values at the cell locations.
        """
        trend = np.flipud(array).ravel()
        cells = self.dataset['Cell id'].to_numpy()
        trend_vector = trend[cells]

        return trend_vector

    def _trend_operations(self, x_window, y_window):
        """
        Evaluate the goodness of the trend model using geostatistical operations.
        """
        # do convolution to get the trend model
        trend_array = self._convolve_2d(x_window, y_window)
        trend_vector = self._mapping_to_cells(trend_array)
        residuals_vector = self.dataset[self.target] - trend_vector
        variance_ratio = np.var(trend_vector) / self._declus_variance
        covariance_trend_residuals = (np.cov(trend_vector, residuals_vector)[1, 0])
        variance_observed = np.var(trend_vector) + np.var(residuals_vector) + 2 * covariance_trend_residuals

        return variance_observed, variance_ratio

    def _get_trend_model(self, x_window, y_window, theta):

        self.xwindow = x_window
        self.ywindow = y_window
        trend_array = self._convolve_2d(x_window, y_window, theta)

        trend_vector = self._mapping_to_cells(trend_array)
        residuals_vector = self.dataset[self.target] - trend_vector

        return trend_array, trend_vector, residuals_vector

    def _error_analysis(self, full_trend_array, trend_at_wells, residuals):
        """
        Evalute trend model goodness using different geostatistical metrics.

        Parameters
        ----------
        full_trend_array
        trend_at_wells
        residuals

        Returns
        -------

        """
        # trend percentage error should be close to zero
        trend_perror = np.abs(self._declus_mean - np.nanmean(full_trend_array)) / self._declus_mean

        # the mean of the residuals should be close to zero
        covariance_rm = np.cov(trend_at_wells, residuals)[1, 0]
        var_experimental = np.var(residuals) + np.var(trend_at_wells) + 2 * covariance_rm

        # the experimental variance (using res, trend, and cov) should be similar to original feature variance
        error_pvar = (self._declus_variance - var_experimental) / self._declus_variance

        # the variance ratio prevent underfitting/overfitting from occuring
        # trend_stats = DescrStatsW(trend_at_wells, weights=self._declus_weights, ddof=0)
        # var_ratio = trend_stats.var / self._declus_variance
        var_ratio = np.var(trend_at_wells) / self._declus_variance

        # the covariance ratio of residuals-trend wrt original feature must be < 0.15
        cova_ratio = covariance_rm / self._declus_variance

        print(f"\nTrend mean percentage error (close to 0%?)   : {trend_perror:.2%}")
        print(f"Residuals mean (close to 0?)                 : {np.mean(residuals):.2e}")
        print(f"Variance percentage error (close to 0%?)     : {error_pvar:.2%}")
        print(f"Variance ratio                               : {var_ratio:.2%}")
        print(f"Covariance ratio (< 15%?)                    : {cova_ratio:.2%}\n")

    def get_trend_only(self, xwindow, ywindow, theta, model_trend=True, verbose=False):
        """
        Compute the trend model using given window sizes for X and Y.
        """

        df_study = self.dataset.copy()
        df_study['Cell x'] = self.dataset['Cell x'].to_numpy()
        df_study['Cell y'] = self.dataset['Cell y'].to_numpy()

        if model_trend:
            trend_present = True
            trend_array, trend_vector, residuals_vector = self._get_trend_model(xwindow, ywindow, theta)
            self.final_trend = trend_array
            df_study['Trend'] = trend_vector
            df_study['Residuals'] = residuals_vector

            # check if there are nan
            nan_present = df_study['Trend'].isnull().values.any()
            if nan_present:
                trend_vector = trend_vector[~np.isnan(trend_vector)]
                residuals_vector = residuals_vector[~np.isnan(residuals_vector)]
                df_study.dropna(inplace=True, axis=0)

            if verbose:
                self._error_analysis(trend_array, trend_vector, residuals_vector)

        else:
            trend_array = None
            trend_present = False

        return trend_array, trend_present, df_study


class SpatialContinuityParallel:
    def __init__(self, lag_distance, dataframe, xcoor, ycoor, lag_tol=None, bandh=None, nlag=None):
        """
        Model the major range of continuity.

        Parameters
        ----------

        """
        self._xcoor = xcoor
        self._ycoor = ycoor
        self._dataset = dataframe

        # geostatistical modeling parameters
        # no trimming
        self.tmin = -999
        self.tmax = 999
        # azimuth tolerance
        self._atol = 22.5
        # standardize the sill (1:yes)
        self._isill = 1
        # lag distance
        self.max_distance = self._half_distance_reservoir()
        self.lag_dist = lag_distance
        # lag tolerance
        if lag_tol is None:
            self._lag_tol = self.lag_dist * 2
        else:
            self._lag_tol = lag_tol
        # bandwidth
        if bandh is None:
            self._bandh = self.lag_dist * 2
        else:
            self._bandh = bandh
        # number of lags is computed automatically
        if nlag is None:
            self._nlag = int(self.max_distance / self.lag_dist)
        else:
            self._nlag = nlag
        self._square_max_dist = self.max_distance ** 2
        self._bayesian_realizations = None

    def _half_distance_reservoir(self):
        """
        Determine the largest distance in either X or Y direction
        :return: The maximum distance
        """
        half_dist_x = np.round(np.ptp(self._dataset[self._xcoor])) / 2
        half_dist_y = np.round(np.ptp(self._dataset[self._ycoor])) / 2

        if half_dist_x > half_dist_y:
            max_distance = half_dist_x
        else:
            max_distance = half_dist_y

        return max_distance

    def _semivariograms(self, azimuths):
        lag = np.zeros((self._nlag + 2, len(azimuths)))
        gamma = np.zeros_like(lag)
        npp = np.zeros_like(lag)

        for i, azimuth_i in enumerate(azimuths):
            lag[:, i], gamma[:, i], npp[:, i] = gamv(
                self._dataset, self._xcoor, self._ycoor, 'NFeat', self.tmin, self.tmax,
                self.lag_dist, self._lag_tol, self._nlag, azimuth_i,
                self._atol, self._bandh, self._isill
            )

        lag = lag[:-1, :]
        gamma = gamma[:-1, :]
        npp = npp[:-1, :]

        return lag, gamma, npp

    def _directional_continuity_loss(self, azimuth):
        lag, gamma, npp = self._semivariograms(azimuth)
        if np.sum(gamma > 1) != 0:
            gamma_points, lag_points = _semivariogram_intersects(azimuth, lag, gamma)
            range_distance = _compute_range(gamma_points, lag_points)
        else:  # zonal anisotropy
            min_azimuth = azimuth[0]
            min_azimuth = min_azimuth + 90 if min_azimuth < 90 else min_azimuth - 90
            lag, gamma, npp = self._semivariograms([min_azimuth])
            gamma_points, lag_points = _semivariogram_intersects(azimuth, lag, gamma)
            range_distance = _compute_range(gamma_points, lag_points)
            range_distance = self._square_max_dist / range_distance

        return -range_distance

    def _objective_loss(self, config):
        """easy_objective"""
        for i in range(config["iterations"]):
            x = config.get("azimuth")
            tune.report(
                timesteps_total=i,
                spc_loss=self._directional_continuity_loss([x]),
            )
            time.sleep(0.02)

    def _get_vario_tensors_and_range(self, azimuth):
        lag, gamma, npp = self._semivariograms([azimuth])
        gamma_points, lag_points = _semivariogram_intersects([azimuth], lag, gamma)
        rango = _compute_range(gamma_points, lag_points)

        tensor = np.hstack((lag, gamma, npp))

        azi_range = np.array([azimuth, rango])

        return tensor, azi_range

    def _get_tensors_anisotropy(self, azimuth):
        lag, gamma, npp = self._semivariograms([azimuth])

        tensor = np.hstack((lag, gamma, npp))
        azi_range = np.array([azimuth, self.max_distance * 10])

        return tensor, azi_range

    def bayesian_optimization(self, trials, minimum_azimuth, maximum_azimuth):
        algo = AxSearch()
        algo = tune.suggest.ConcurrencyLimiter(algo, max_concurrent=4)
        scheduler = AsyncHyperBandScheduler()
        analysis = tune.run(
            self._objective_loss,
            name='ax',
            metric="spc_loss",
            mode='min',
            search_alg=algo,
            scheduler=scheduler,
            num_samples=trials,
            config={
                "iterations": trials,
                "azimuth": tune.uniform(minimum_azimuth, maximum_azimuth)
            },
            stop={"timesteps_total": trials},
            verbose=0,
        )
        major_azimuth = analysis.best_config.get("azimuth")
        minor_azimuth = major_azimuth + 90 if major_azimuth < 90 else major_azimuth - 90

        return major_azimuth, minor_azimuth

    def get_directional_continuity(self, trials, minimum_azimuth, maximum_azimuth):
        major_azimuth, minor_azimuth = self.bayesian_optimization(trials, minimum_azimuth, maximum_azimuth)
        # check if zonal anisotropy
        lag, gamma, npp = self._semivariograms([major_azimuth])

        # tensor dimensions (A, B, C)
        # A: 0 corresponds to the major direction of continuity, 1 the minor
        semivar_tensor = np.zeros((2, self._nlag + 1, 3))
        azi_rang_tensor = np.zeros((2, 2))

        if np.sum(gamma > 1) != 0:
            semivar_tensor[0, ...], azi_rang_tensor[0, ...] = self._get_vario_tensors_and_range(major_azimuth)
            semivar_tensor[1, ...], azi_rang_tensor[1, ...] = self._get_vario_tensors_and_range(minor_azimuth)
        else:
            semivar_tensor[0, ...], azi_rang_tensor[0, ...] = self._get_tensors_anisotropy(major_azimuth)
            semivar_tensor[1, ...], azi_rang_tensor[1, ...] = self._get_vario_tensors_and_range(minor_azimuth)

        return azi_rang_tensor, semivar_tensor


class SpatialContinuitySequential(SpatialContinuityParallel):
    def __init__(self, lag_distance, dataframe, xcoor, ycoor, lag_tol, bandh, nlag):
        super().__init__(lag_distance, dataframe, xcoor, ycoor, lag_tol, bandh, nlag)

    def _objective_loss(self, config):
        azimuth = [config.get("azimuth")]
        return {"objective": self._directional_continuity_loss(azimuth)}

    def bayesian_optimization(self, trials, minimum_azimuth, maximum_azimuth):
        """
        Obtain the major and minor directions of spatial continuity, and a tensor containing the lag, gamma, and npp
        arrays.

        Parameters
        ----------
        trials: int
            Number of Bayesian trials to run.
        minimum_azimuth, maximum_azimuth: float
            Minimum and maximum azimuth to consider for Bayesian optimization.

        Returns
        -------

        """
        parameters_dict = [
            {
                "name": "azimuth",
                "type": "range",
                "bounds": [minimum_azimuth, maximum_azimuth],
                "value_type": "float",
            },
        ]

        ax_client = AxClient(verbose_logging=False)
        # create the experiment
        ax_client.create_experiment(
            parameters=parameters_dict,
            objective_name='objective',
            minimize=True,
        )

        with tqdm(total=trials) as pbar:
            for i in range(trials):
                parameters, trial_index = ax_client.get_next_trial()

                # Local evaluation here can be replaced with deployment to external system.
                ax_client.complete_trial(trial_index=trial_index,
                                         raw_data=self._objective_loss(parameters))
                pbar.update(1)  # progress bar

        dframe = ax_client.get_trials_data_frame().sort_values('trial_index')
        major_azimuth = dframe.loc[dframe['objective'].idxmin()]['azimuth']
        minor_azimuth = major_azimuth + 90 if major_azimuth < 90 else major_azimuth - 90

        return major_azimuth, minor_azimuth

    @staticmethod
    def save_tensors(root, name, azimuth_tensor, semivario_tensor):
        np.save(os.path.join(root, "azi_rang_tensor" + name + ".npy"), azimuth_tensor)
        np.save(os.path.join(root, "semivar_tensor" + name + ".npy"), semivario_tensor)


class SeqGaussSim:
    def __init__(self, df, xcoor, ycoor, feature, cell_size, trend_object, spatial_contin_object, vario_model,
                 handle_negatives=True):

        self.df = df
        self.xcoor = xcoor
        self.ycoor = ycoor
        self.feature = feature
        self.cell_size = cell_size

        # _TrendModeler class
        self.nx = trend_object.df_grid.shape[1]
        self.ny = trend_object.df_grid.shape[0]
        self.xmn = trend_object.xmin + self.cell_size / 2
        self.ymn = trend_object.ymin + self.cell_size / 2
        self._final_trend = trend_object.final_trend

        # _SpatialContinuity class
        self._dir_continuity = spatial_contin_object

        # stochastic Gaussian simulation parameters
        self._ndmin = 0
        self._ndmax = 10
        self.vario_model = vario_model

        self.radius = self._max_radius()
        self.handle_negatives = handle_negatives

    def _max_radius(self):
        radius = max(self.vario_model.get("hmaj1"), self.vario_model.get("hmaj2"))
        if radius > self._dir_continuity.max_distance:
            radius = np.hypot(self._dir_continuity.max_distance, self._dir_continuity.max_distance) * 1.25

        return radius

    def _add_trend(self, tensor, handle_negatives):
        if self.feature == "Residuals":
            tensor += self._final_trend
        else:
            pass

        if handle_negatives:  # handle negative petrophysical values
            tensor[tensor < 0] = np.nan  # or like this?
        return tensor

    def _geostatspy_version(self, realizations, zmin, zmax, ktype=0, pearson_r=0.0, sec_map=0, varred=1.0):
        simulation = np.zeros((realizations, self.ny, self.nx))
        if isinstance(sec_map, int):  # create an array filled with zeros when there is no cosimulation
            sec_map = np.zeros(realizations)

        with tqdm(total=realizations) as pbar:
            for i in np.arange(realizations):
                simulation[i, ...] = sgsim(
                    df=self.df, xcol=self.xcoor, ycol=self.ycoor, vcol=self.feature, wcol=-1, scol=-1,
                    tmin=self._dir_continuity.tmin, tmax=self._dir_continuity.tmax, itrans=1, ismooth=0, dftrans=0,
                    tcol=0, twtcol=0, zmin=zmin, zmax=zmax, ltail=1, ltpar=0.0, utail=1, utpar=zmax, nsim=1,
                    nx=self.nx, xmn=self.xmn, xsiz=self.cell_size, ny=self.ny, ymn=self.ymn, ysiz=self.cell_size,
                    seed=int(i), ndmin=self._ndmin, ndmax=self._ndmax, nodmax=20, mults=0, nmult=2, noct=-1,
                    radius=self.radius, radius1=1, sang1=0, mxctx=10, mxcty=10, ktype=ktype, colocorr=pearson_r,
                    sec_map=sec_map[i], vario=self.vario_model, varred=varred)

                pbar.update(1)

        simulation = self._add_trend(simulation, self.handle_negatives)

        return simulation

    def _gslib_version(self, realizations, zmin, zmax, output_file, ktype=0, pearson_r=0.6, sec_map=0, col_secvar=0,
                       varred=1.0):
        simulation = np.zeros((realizations, self.ny, self.nx))
        with tqdm(total=realizations) as pbar:
            for i in np.arange(realizations):
                if isinstance(sec_map, int):
                    sec_map_file = "none.dat"
                else:  # cosimulation
                    ndarray2GSLIB(sec_map[i], data_file="sec_feat.dat", col_name="Sec_feat")
                    sec_map_file = "sec_feat.dat"

                simulation[i, ...] = _sgsim(
                    1, self.df, self.xcoor, self.ycoor, self.feature, zmin, zmax,
                    self.nx, self.ny, self.xmn, self.ymn, self.cell_size, i + 1,
                    self.vario_model, self.radius, output_file, self._ndmax, ktype=ktype, rho=pearson_r, varred=varred,
                    secfl=sec_map_file, col_secvar=col_secvar)
                pbar.update(1)

            simulation = self._add_trend(simulation, self.handle_negatives)

        return simulation

    def _data_imputation(self, tensor, feature_to_input):
        cells_x = self.df['Cell x'].to_numpy()
        cells_y = self.ny - 1 - self.df['Cell y'].to_numpy()

        sims = tensor.copy()
        sims[:, cells_y, cells_x] = self.df[feature_to_input].to_numpy()

        return sims

    def _summarize_results(self, simulations, feature_to_input):

        simulations = self._data_imputation(simulations, feature_to_input)
        summary = np.zeros((6, simulations.shape[1], simulations.shape[2]))

        # p10
        summary[0, ...] = np.nanquantile(simulations, 0.1, axis=0)
        # p50
        summary[1, ...] = np.nanquantile(simulations, 0.5, axis=0)
        # p90
        summary[2, ...] = np.nanquantile(simulations, 0.9, axis=0)
        # uncertainty: p90 - p10
        summary[3, ...] = summary[2, ...] - summary[0, ...]
        # mean of all realizations
        summary[4, ...] = np.nanmean(simulations, axis=0)
        # trend
        summary[5, ...] = self._final_trend

        return simulations, summary

    def sgsimulation(self, realizations, feature_to_input, search_radius=None, nodmax=None, python=False, zmin=None,
                     zmax=None, output='simulatorexe.txt', ):
        """
        Perform sequential Gaussian simulation.

        Parameters
        ----------
        realizations: int
            Number of simulations to perform.
        output: str
            Dummy variable
        feature_to_input: str
            Feature in original space.
        search_radius: float
        nodmax: float

        python: bool
            False to run simulations with GSLIB (recommended), True to run with Python.
        zmin, zmax: float
            Minimum and maximum trim values to consider for simulations.

        Returns
        -------
        Tensors containing all simulations and a summary of them.
        """
        if zmin is None or zmax is None:
            zmin = self.df[self.feature].min()
            zmax = self.df[self.feature].max()

        if search_radius is not None:
            self.radius = search_radius
        if nodmax is not None:
            self._ndmax = nodmax

        if python:
            simulations = self._geostatspy_version(realizations, zmin, zmax)
        else:
            simulations = self._gslib_version(realizations, zmin, zmax, output)

        simulations, summary = self._summarize_results(simulations, feature_to_input)

        return simulations, summary


class MarkovBayesModel(SeqGaussSim):
    def __init__(self, df_study, xcoor, ycoor, feature, cell_size, trend_object, spatial_contin_object,
                 vario_model, simulation_sec, varred, handle_negatives):
        """
        Perform cosimulation of feature 1 constrained on feature 2 using the Markov-Bayes approach.

        Parameters
        ----------
        df_study: pandas.DataFrame
            Main dataset.

        xcoor, ycoor: str
            Names of the horizontal and vertical coordinates.

        feature: str
            Name of the main feature.

        cell_size: float
            Cell size for simulation.

        trend_object: trend_object object
            Attribute that contains trend information.

        spatial_contin_object: spatial_contin object
            Attribute that contains spatial continuity information.

        vario_model: dict
            Variogram model for feature in geostatspy format.

        simulation_sec: numpy array
            Contains the final realizations of feature_sec.

        """
        super().__init__(df_study, xcoor, ycoor, feature, cell_size, trend_object, spatial_contin_object,
                         vario_model, handle_negatives)
        self.simulation2 = simulation_sec
        # correlation coefficient for Markov-Bayes
        self.cor_coef, _ = pearsonr(self.df['NFeat'], self.df['NFeatSec'])
        self.varred = varred

    def sgsimulation(self, realizations, output, feature_to_input, python=False, zmin=None, zmax=None):
        """
        Perform sequential Gaussian cosimulation.

        Parameters
        ----------
        realizations: int
            Number of simulations to perform.
        output: str
            Dummy variable
        feature_to_input: str
            Feature in original space.
        python: bool
            False to run simulations with GSLIB (recommended), True to run with Python.
        zmin, zmax: float
            Minimum and maximum trim values to consider for simulations.

        Returns
        -------
        Tensors containing all simulations and a summary of them.
        """
        if zmin is None or zmax is None:
            zmin = self.df[self.feature].min()
            zmax = self.df[self.feature].max()

        if python:
            simulations = self._geostatspy_version(realizations, zmin, zmax, ktype=4, pearson_r=self.cor_coef,
                                                   sec_map=self.simulation2, varred=self.varred)
        else:
            simulations = self._gslib_version(realizations, zmin, zmax, output, ktype=4, pearson_r=self.cor_coef,
                                              sec_map=self.simulation2, col_secvar=4, varred=self.varred)

        simulations, summary = self._summarize_results(simulations, feature_to_input)

        return simulations, summary


class RealizationsQC:
    def __init__(self, df, xcoor, ycoor, feature, cell_size, realizations_array, realizations_res_space, summary,
                 extra_cells, lag_distance, lag_tol, bandh, trend_object, spatial_contin_object):
        """

        Parameters
        ----------
        df
        xcoor
        ycoor
        feature
        cell_size
        realizations_array: numpy ndarray
            Realizations in original input space (i.e., in the same units as the input)
        realizations_res_space: numpy ndarray
            Realizations in residual normal scores space (i.e., without the trend if present)
        summary
        extra_cells
        lag_distance
        trend_object
        spatial_contin_object
        """
        self._xcoor = xcoor
        self._ycoor = ycoor
        self.dataset = df
        self._cell_size = cell_size
        self.feature = feature
        self.realizations = realizations_array
        self.realizations_residuals = realizations_res_space
        self.summary = summary
        self.additional_cells = extra_cells

        self._num_sim = self.realizations.shape[0]
        self._lag_dist = lag_distance
        self._lag_tol = self._lag_dist * 2 if lag_tol is None else lag_tol
        self._bandh = self._lag_dist * 2 if bandh is None else bandh

        # variogram reproduction
        self.max_distance = spatial_contin_object.max_distance
        self._atol = 22.5
        self._isill = 1  # standardize the sill (1:yes)

        self._xmin, xmax, self._ymin, ymax = trend_object.xmin, trend_object.xmax, trend_object.ymin, trend_object.ymax
        # self.coorsx = np.arange(self._xmin, xmax + cell_size, cell_size)
        # self.coorsy = np.arange(self._ymin, ymax + cell_size, cell_size)

        # direction of spatial continuity
        self.tempo_df = self.dataset[[self._xcoor, self._ycoor]]
        # NFeat is the original variable. Do it to avoid writing more code because semivariograms uses NFeat by default
        self.tempo_df['NFeat'] = self.dataset[self.feature]

    def _compute_nbins(self):
        # compute the number of bins from the original feature
        q75 = self.dataset[self.feature].quantile(0.75)
        q25 = self.dataset[self.feature].quantile(0.25)
        maxim = self.dataset[self.feature].max()
        minim = self.dataset[self.feature].min()
        h = 2 * (q75 - q25) * len(self.dataset) ** (-1 / 3)
        nbins_input = (maxim - minim) / h

        # compute the number of bins from the original feature
        q75 = np.nanquantile(self.realizations, .75)
        q25 = np.nanquantile(self.realizations, .25)
        h = 2 * (q75 - q25) * len(self.realizations) ** (-1 / 3)
        nbins_real = (np.nanmax(self.realizations) - np.nanmin(self.realizations)) / h

        nbins = int(np.round(np.min((nbins_input, nbins_real))))

        return nbins

    def _grid_variog(self, lags, azimuth, num_sim, tmin=-999, tmax=999):
        """
        Variogram calculation of gridded data.
        Ixd, iyd specify the unit offsets that define each of the ndir directions.
        Examples:

        ixd =  0, iyd = -1: 000 azimuth, aligned along the y-axis
        ixd =  1, iyd =  0: 090 azimuth, aligned along the x-axis
        ixd =  1, iyd = -1: 045 azimuth, horizontal at 45 degrees from y
        ixd = -1, iyd = -1: 135 azimuth, horizontal at 135 degrees from y
        """
        if azimuth == 0:
            ixd = 0
            iyd = -1
        elif azimuth == 45:
            ixd = 1
            iyd = -1
        elif azimuth == 90:
            ixd = 1
            iyd = 0
        else:
            ixd = -1
            iyd = -1

        lag_model = np.zeros((num_sim, lags))
        vario_model = np.zeros((num_sim, lags))
        npar_model = np.zeros((num_sim, lags))
        with tqdm(total=num_sim) as pbar:
            for realization in np.arange(num_sim):
                A, B, C = geostats.gam(
                    self.realizations_residuals[realization, :, :],
                    tmin=tmin,
                    tmax=tmax,
                    xsiz=self._cell_size,
                    ysiz=self._cell_size,
                    ixd=ixd,
                    iyd=iyd,
                    nlag=lags,
                    isill=1.0
                )

                lag_model[realization, :] = A.T
                vario_model[realization, :] = B.T
                npar_model[realization, :] = C.T

                pbar.update(1)

        return lag_model, vario_model, npar_model

    def _array2df(self, realization_n, residual_space):
        if residual_space:
            array_realiz = self.realizations_residuals
        else:
            array_realiz = self.realizations
        xmin = self._xmin
        cellsx = array_realiz.shape[2]
        ymin = self._ymin
        cellsy = array_realiz.shape[1]
        x = np.arange(
            xmin + self._cell_size / 2, xmin + self._cell_size / 2 + self._cell_size * cellsx, self._cell_size)
        y = np.arange(
            ymin + self._cell_size / 2, ymin + self._cell_size / 2 + self._cell_size * cellsy, self._cell_size)

        xx, yy = np.meshgrid(x, y)  # create a meshgrid to simplify the addition of coordinates
        results = pd.DataFrame(xx.ravel(), columns=[self._xcoor])
        results[self._ycoor] = yy.ravel()
        results[self.feature] = array_realiz[realization_n].ravel()
        results.dropna(inplace=True)

        return results

    def semiv_reprod(self, azimuth_1, azimuth_2, model_range, model_nlag, model_xlag, vario_model, nlags_grid,
                     save_img=False, num_sim=None):
        """
        Check if realizations reproduce the spatial continuity from the samples.
        azimuth_1, azimuth_2: int
            Choose from 4 azimuths: 0, 45, 90, 135 degrees.

        """
        if num_sim is None:
            num_sim = 1
        assert num_sim <= self.realizations.shape[0],\
            f"Number of gridded variograms cannot be greater than {self.realizations.shape[0]}. "

        # realizations
        lag_grid_maj, vario_grid_maj, npar_grid_maj = self._grid_variog(nlags_grid, azimuth_1, num_sim)
        lag_grid_min, vario_grid_min, npar_grid_min = self._grid_variog(nlags_grid, azimuth_2, num_sim)

        # model
        h_model_major, gam_model_major = _vmodel(model_nlag, model_xlag, azimuth_1, vario_model)
        h_model_minor, gam_model_minor = _vmodel(model_nlag, model_xlag, azimuth_2, vario_model)

        # Plots: create two subplots and unpack the output array immediately
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        for i in range(lag_grid_maj.shape[0]):
            sns.scatterplot(
                x=lag_grid_maj[i, :],
                y=vario_grid_maj[i, :],
                alpha=0.45,
                color='black',
                marker="_",
                ax=ax1
            )

        ax1.plot(h_model_major, gam_model_major, '-', color='red', label="Model", linewidth=5)

        for i in range(lag_grid_maj.shape[0]):
            sns.scatterplot(
                x=lag_grid_min[i, :],
                y=vario_grid_min[i, :],
                alpha=0.45,
                color='black',
                marker="_",
                ax=ax2
            )

        ax2.plot(h_model_minor, gam_model_minor, '-', color='red', label="Model", linewidth=5)

        ax1.set_xlim(0, model_range)
        ax1.set_ylim(0, 2)
        ax2.set_xlim(0, model_range)
        ax2.set_ylim(0, 2)
        plt.legend()
        # plt.suptitle("Variogram comparison: " + self.feature, fontsize=20, fontweight='bold')
        ax1.set_ylabel(r'Gamma $\gamma$', fontsize=18)
        ax1.set_xlabel('Lag distance (m)', fontsize=18)
        ax2.set_xlabel('Lag distance (m)', fontsize=18)
        ax1.set_title('Azimuth: ' + str(azimuth_1), fontsize=18)
        ax2.set_title('Azimuth: ' + str(azimuth_2), fontsize=18)
        ax1.grid(False)
        ax2.grid(False)
        ax1.tick_params(axis='x', direction='in', labelsize=14)
        ax2.tick_params(axis='x', direction='in', labelsize=14)
        ax1.tick_params(axis='y', direction='in', labelsize=14)
        ax2.tick_params(axis='y', direction='in', labelsize=14)
        f.tight_layout()

        if save_img:
            plt.savefig(self.feature + '_vario_reprd.png', bbox_inches='tight', dpi=500)

        plt.show()

    def histograms(self, declus_weights, nbins=None, save_img=False):
        """
        Compare the histograms and cumulative distribution functions of the original feature and realizations.
        Parameters
        ----------
        save_img
        declus_weights
        nbins

        Returns
        -------

        """
        if nbins is None:
            nbins = self._compute_nbins()

        # Check the histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.hist(self.dataset[self.feature], facecolor='red',
                 bins=np.linspace(self.dataset[self.feature].min(), self.dataset[self.feature].max(), nbins),
                 alpha=1.,
                 density=True,
                 weights=declus_weights,
                 edgecolor='black',
                 label='Data Distribution'
                 )

        for i in np.arange(self._num_sim):
            ax1.hist(self.realizations[i, :, :].flatten(),
                     bins=np.linspace(self.dataset[self.feature].min(), self.dataset[self.feature].max(), nbins),
                     alpha=0.1,
                     density=True,
                     edgecolor='black',
                     label='Realization {0:2s}'.format(str(i + 1))
                     )

        ax1.set_xlabel(self.feature, fontsize=18)
        ax1.set_ylabel('Density', fontsize=18)
        ax1.tick_params(axis='x', direction='in', labelsize=13)
        ax1.tick_params(axis='y', direction='in', labelsize=13)
        ax1.grid(False)

        ax2.hist(self.dataset[self.feature], color='red', lw=2,
                 bins=np.linspace(self.dataset[self.feature].min(), self.dataset[self.feature].max(), 50),
                 histtype="step",
                 alpha=1.0,
                 density=True,
                 cumulative=True,
                 weights=declus_weights,
                 edgecolor='red',
                 label='Data Distribution')

        for i in np.arange(self._num_sim):
            ax2.hist(self.realizations[i, :, :].flatten(), lw=2,
                     bins=np.linspace(np.nanmin(self.realizations), np.nanmax(self.realizations), 50),
                     histtype="step",
                     alpha=0.1,
                     density=True,
                     cumulative=True,
                     label='Realization {0:2s}'.format(str(i + 1)))

        ax2.set_xlim([0.0, np.max(self.dataset[self.feature])])
        ax2.set_ylim([0, 1.05])

        ax2.set_xlabel(self.feature, fontsize=18)
        ax2.set_ylabel('Cumulative Probability', fontsize=18)
        ax2.tick_params(axis='x', direction='in', labelsize=13)
        ax2.tick_params(axis='y', direction='in', labelsize=13)
        ax2.grid(False)

        plt.suptitle("Input data vs realizations", fontsize=20, fontweight='bold')
        # plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.2, wspace=0.2, hspace=0.3)
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])

        if save_img:
            plt.savefig(self.feature + '_hist.png', bbox_inches='tight', dpi=500)

        plt.show()

    def direction_spat_continuity(self, bayesian_iterations, min_azimuth, max_azimuth, parallel):
        """
        Find the largest direction of spatial continuity of the samples in original space.

        Parameters
        ----------
        bayesian_iterations: int
            Number of bayesian trials for optimization.

        min_azimuth, max_azimuth: float
            Azimuth range used for exploration, i.e., [min_azimuth, max_azimuth]

        parallel: bool
            True to run in parallel, False to run sequentially.

        Returns
        -------
        major_azimuth, minor_azimuth: float
            The major_azimuth and minor_azimuth obtained using Bayesian optimization.
        """
        if parallel:
            spatial_continuity = SpatialContinuityParallel(self.tempo_df, self._xcoor, self._ycoor)
        else:
            spatial_continuity = SpatialContinuitySequential(self.tempo_df, self._xcoor, self._ycoor)

        major_azimuth, minor_azimuth = spatial_continuity.bayesian_optimization(
            trials=bayesian_iterations,
            minimum_azimuth=min_azimuth,
            maximum_azimuth=max_azimuth
        )

        return major_azimuth, minor_azimuth

    def variogram_map(self, nx, ny, realization_n, standardize, residual_space):
        """
        Plot the variogram map for a realization.

        Parameters
        ----------
        nx, ny: int
            The number of nodes in the x, y coordinate directions.

        realization_n: int
            What realization to consider

        standardize: int
            If set to 1, the semivariogram values will be divided by the variance.

        residual_space: bool
            True to plot the variogram map of the residual normal scores space; False to plot the varmap of the original
            input space.
        """
        dataframe = self._array2df(realization_n, residual_space)
        vario_map = _varmap(
            df=dataframe,
            xcol=self._xcoor,
            ycol=self._ycoor,
            vcol=self.feature,
            nx=nx,
            ny=ny,
            lagdist=self._cell_size,
            minpairs=20,
            bstand=standardize
        )

        xmax = (float(nx) + 0.5) * self._cell_size
        xmin = -1 * xmax
        ymax = (float(ny) + 0.5) * self._cell_size
        ymin = -1 * ymax

        xvalues = np.arange(xmin, xmax, self._cell_size)
        yvalues = np.arange(ymin, ymax, self._cell_size)

        var_max = np.nanmax(vario_map)
        fig = px.imshow(np.flipud(vario_map), x=xvalues, y=yvalues, zmin=0, zmax=var_max, origin='lower',
                        color_continuous_scale='plasma', labels=dict(x=self._xcoor, y=self._ycoor, color=self.feature))
        fig.update_layout(
            title={
                'text': "Variogram Map Realization: 0",
                'y': 0.96,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        fig.show()

    def qq_plot_f(self, realization, step, weights):
        quantiles = np.arange(0.00, 1.01, step)

        array_q = np.nanquantile(self.realizations, quantiles, axis=(1, 2))

        input_q = weighted_quantile(
            self.dataset[self.feature],
            quantiles,
            sample_weight=weights
        )

        min_identity = np.min([self.dataset[self.feature].min(), np.min(array_q[:, realization])])
        max_identity = np.max([self.dataset[self.feature].max(), np.max(array_q[:, realization])])

        texto = list(map('Target: {:.2f}; Realization: {:.2f}'.format, input_q, array_q[:, realization]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=input_q, y=array_q[:, realization],
            mode='markers',
            text=texto,
            name='q-q',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[min_identity, max_identity], y=[min_identity, max_identity],
            mode='lines',
            name='identity line',
            showlegend=False
        ))

        fig.update_layout(
            title="Q-Q plot for realization: " + str(realization),
            xaxis_title="Target " + self.feature + " distribution",
            yaxis_title=self.feature + " realization " + str(realization),
            legend_title="Legend",
            font=dict(
                size=12,
            )
        )

        fig.update_xaxes(ticks="inside")
        fig.update_yaxes(ticks="inside")

        fig.show()


class Visualization:
    def __init__(self, df, xcoor, ycoor, primary_feat, cell_size, realizations_array, summary, trend_object,
                 secondary_feat=None):
        self._xcoor = xcoor
        self._ycoor = ycoor
        self.dataset = df
        self._cell_size = cell_size
        self.primary_feat = primary_feat
        self.sec_feat = secondary_feat
        self.realizations = realizations_array
        self.summary = summary

        self._num_sim = self.realizations.shape[0]
        self.df_hover = self._hover_text()
        xmin, xmax, ymin, ymax = trend_object.xmin, trend_object.xmax, trend_object.ymin, trend_object.ymax
        self._coorsx = np.arange(xmin, xmax + cell_size, cell_size)
        self._coorsy = np.arange(ymin, ymax + cell_size, cell_size)
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

        self.cellsx = realizations_array.shape[2]
        self.cellsy = realizations_array.shape[1]
        self._mask = ~np.isnan(self.summary[0])

    def _hover_text(self):
        df2 = self.dataset.copy()
        hover_text = []

        for index, row in df2.iterrows():
            hover_text.append(('UWI: {uwi_}<br>' +
                               'X: {xcoor_}<br>' +
                               'Y: {ycoor_}<br>' +
                               'Feature: {feat_}<br>'
                               ).format(
                uwi_=row['UWI'],
                xcoor_=row[self._xcoor],
                ycoor_=row[self._ycoor],
                feat_=row[self.primary_feat],
            ))

        df2['text'] = hover_text

        return df2

    def _prob_excedance(self, threshold):
        prob_exceed = np.nansum(self.realizations >= threshold, axis=0) / self._num_sim

        return prob_exceed

    def ridge_plot(self):
        """
        Randomly plot five (or less) probability distribution functions from realizations and original feature.
        """
        if self._num_sim <= 5:
            nplots = self._num_sim - 1
        else:
            nplots = 5
        # choose 5 numbers at random
        reals = sample(range(self._num_sim), nplots)
        # convert them to a list
        names = ['Realization ' + str(real_i) for real_i in reals]
        # select the realizations from the 11 numbers
        data = [self.realizations[i, :, :].ravel() for i in reals]
        # insert the original feature name and values
        names.insert(0, self.primary_feat)
        data.insert(0, self.dataset[self.primary_feat])

        colores = n_colors('rgb(200, 10, 10)', 'rgb(5, 200, 200)', 6, colortype='rgb')
        fig = go.Figure()
        index = 0
        for data_line, color in zip(data, colores):
            fig.add_trace(go.Violin(
                x=data_line,
                line_color=color,
                name=names[index]
            ))

            index += 1

        fig.update_traces(orientation='h', side='positive', width=3, points=False)
        fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
        fig.show()

    def plot_simulation(self, xy_ratio, width=700, height=700):
        """
        Plot the summary and uncertainty maps.

        Parameters
        ----------
        xy_ratio: float
            The scale ratio for the axes.

        width, height: int
            The width and height of the image.
        -------

        """
        buttons = list([
            dict(args=[{'z': [np.flipud(self.summary[0])]}],
                 label="P10",
                 method="restyle"
                 ),
            dict(args=[{'z': [np.flipud(self.summary[1])]}],
                 label="P50",
                 method="restyle"
                 ),
            dict(args=[{'z': [np.flipud(self.summary[2])]}],
                 label="P90",
                 method="restyle"
                 ),
            dict(args=[{'z': [np.flipud(self.summary[3])]}],
                 label="Uncertainty",
                 method="restyle"
                 ),
            dict(args=[{'z': [np.flipud(self.summary[4])]}],
                 label="Expectation",
                 method="restyle"
                 ),
            dict(args=[{'z': [np.flipud(self.summary[5])]}],
                 label="Trend array",
                 method="restyle"
                 ),
        ])
        fig = go.Figure(data=go.Heatmap(
            z=np.flipud(self.summary[0]),
            x=self._coorsx,
            y=self._coorsy,
            colorscale='Inferno',
            zmin=np.nanmin(self.summary),
            zmax=np.nanmax(self.summary)
        ))

        fig.add_trace(go.Scatter(
            x=self.df_hover[self._xcoor],
            y=self.df_hover[self._ycoor],
            text=self.df_hover['text'],
            mode='markers',
            marker=dict(
                color='black',
                showscale=False,
                size=5,
                symbol='circle',
                opacity=0.8
            ),
            showlegend=False
        ))

        # Add dropdown
        fig.update_layout(
            updatemenus=[
                go.layout.Updatemenu(
                    buttons=buttons,
                    direction="down",
                ),
            ],
            title=self.primary_feat,
            xaxis_title=self._xcoor,
            yaxis_title=self._ycoor,
            autosize=False,
            width=width,
            height=height,

        )

        fig.update_xaxes(scaleanchor='y', ticks='inside', scaleratio=xy_ratio, )
        fig.update_yaxes(ticks='inside')
        fig.show()

    def get_xarray(self, x_name=None, y_name=None, metadata_dict=None):
        """
        Convenience function to transform the summary numpy array to a xarray Dataset.

        Parameters
        ----------
        x_name, y_name: str
        Name of the x-axis and y-axis.

        metadata_dict: dictionary
        Metadata regarding the spatial interpolation.

        Returns
        -------
        """

        if x_name is None and y_name is None:
            x_name = self._xcoor
            y_name = self._ycoor

        ds = xr.Dataset(dict(
            P10=((x_name, y_name), np.flipud(self.summary[0])),
            P50=((x_name, y_name), np.flipud(self.summary[1])),
            P90=((x_name, y_name), np.flipud(self.summary[2])),
            Uncertainty=((x_name, y_name), np.flipud(self.summary[3])),
            Expectation=((x_name, y_name), np.flipud(self.summary[4])),
            Trend=((x_name, y_name), np.flipud(self.summary[5]))
        ))

        nx = self.summary.shape[1]
        ny = self.summary.shape[2]
        dictio = {x_name: self._coorsx[:nx], y_name: self._coorsy[:ny]}
        ds = ds.assign_coords(dictio)

        if metadata_dict is None:
            ds.attrs["long_name"] = self.primary_feat
            ds.attrs["cell_size"] = self._cell_size
            ds.attrs["realizations"] = len(self.realizations)

        else:
            for key, value in metadata_dict.items():
                ds.attrs[key] = value

        return ds

    def local_prob_exc(self, threshold, save_img):
        prob = self._prob_excedance(threshold)
        prob[~self._mask] = np.nan

        colores = ['#225ea8', '#41b6c4', '#a1dab4', '#ffffcc']
        cmap = colors.ListedColormap(colores)
        boundaries = [0.0, 0.25, 0.5, 0.75, 1.0]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

        fig, ax = plt.subplots(1, 1)
        cax = ax.imshow(prob, interpolation=None, extent=[self._xmin, self._xmax, self._ymin, self._ymax], norm=norm,
                        cmap=cmap)
        ax.scatter(
            self.dataset[self._xcoor],
            self.dataset[self._ycoor],
            s=None,
            c=self.dataset[self.primary_feat],
            marker='s',
            cmap=cmap,
            norm=norm,
            alpha=0.8,
            linewidths=0.8,
            edgecolors="black",
        )
        ax.set_title(f"Probability to exceed {threshold} {self.primary_feat}", fontweight='bold', fontsize=20, y=1.)
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        ax.set_xlabel(self._xcoor, fontsize=18)
        ax.set_ylabel(self._ycoor, fontsize=18)
        ax.set_xlim(self._xmin, self._xmax)
        ax.set_ylim(self._ymin, self._ymax)
        ax.tick_params(axis='x', direction='in', labelsize=14)
        ax.tick_params(axis='y', direction='in', labelsize=14)
        cbar = fig.colorbar(cax, orientation="vertical", ticks=[0.125, 0.375, 0.625, 0.875])
        cbar.ax.set_yticklabels(['[0-25]%', '(25-50]%', '(50-75]%', '(75-100]%'])
        cbar.set_label("Probability", rotation=270, size=16)
        plt.gcf().set_size_inches(12, 10)

        if save_img:
            plt.savefig(self.primary_feat + '_prob_excd.png', bbox_inches='tight', dpi=500)

        plt.show()

    def unc_vs_return(self):
        """Plot risk vs P50."""
        fig = px.scatter(
            x=self.summary[1].ravel(),
            y=self.summary[3].ravel(),
            color=self.summary[4].ravel(),
            template='plotly_white'
        )

        # Update axis properties
        fig.update_xaxes(title_text="Uncertainty: P90 - P10")
        fig.update_yaxes(title_text="Return: P50")
        fig.layout.coloraxis.colorbar.title = self.primary_feat + " mean"

        fig.show(config={"displayModeBar": False, "showTips": False})

    def cosim_plot(self, secondary_realizations, sec_feat, hexsize, save_img):
        fig, ax = plt.subplots(figsize=(10, 7))
        hb = ax.hexbin(self.realizations.ravel(), secondary_realizations.ravel(), gridsize=hexsize, bins='log',
                       cmap='viridis')

        ax.scatter(self.dataset[self.primary_feat], self.dataset[sec_feat], alpha=0.6, edgecolors='black', s=40,
                   label='Input data', marker='s', color='red')

        fig.colorbar(hb, ax=ax, label='Count log10(N)')
        plt.legend()
        ax.tick_params(axis='x', direction='in', labelsize=14)
        ax.tick_params(axis='y', direction='in', labelsize=14)
        plt.xlabel(self.primary_feat, fontsize=18)
        plt.ylabel(sec_feat, fontsize=18)
        if save_img:
            plt.savefig(self.primary_feat + '_input_vs_realiz.png', bbox_inches='tight', dpi=500)
        plt.show()


class OptimizeCosim:
    def __init__(self, class_object, secondary_feat, secondary_realizations, variogram_model, lbound=0.6, rbound=1.0,
                 target_var=None):
        self.model_class = class_object
        self.sec_feat = secondary_feat
        self.sec_realizations = secondary_realizations
        self.main_feat = class_object.feature
        self.vario_model = variogram_model
        self.df = class_object.dataset

        self._parameters_dict = [{
            "name": "varred",
            "type": "range",
            "bounds": [lbound, rbound],
            "value_type": 'float'
        }]

        # varred loss
        if target_var is None:
            self._var = self.df[self.main_feat].var()
        else:
            self._var = target_var

    def _get_realizations_df(self, simulations_main, multivariate=False):
        realiz = pd.DataFrame(data=simulations_main[0].ravel(), columns=[self.main_feat])
        if multivariate:
            realiz[self.sec_feat] = self.sec_realizations[0].ravel()
        realiz.dropna(inplace=True)

        return realiz

    def _cosim_update(self, varred):
        realizations, _ = self.model_class.simulation(
            realizations=1,
            vario_model=self.vario_model,
            python=False,
            cosimulation=True,
            feat_secondary=self.sec_feat,
            simulation_sec=self.sec_realizations,
            varred=varred
        )

        return realizations

    def _univariate_loss(self, parameters):
        realizations = self._cosim_update(parameters.get("varred"))
        realizations_df = self._get_realizations_df(realizations)

        loss = np.sqrt(np.square(self._var - realizations_df[self.main_feat].var()))
        return {"varred_loss": loss}

    def varred_optim(self, trials):
        ax_client = AxClient(verbose_logging=False)
        # create the experiment
        ax_client.create_experiment(
            parameters=self._parameters_dict,
            objective_name='varred_loss',
            minimize=True,
        )

        with tqdm(total=trials) as pbar:
            for i in range(trials):
                parameters, trial_index = ax_client.get_next_trial()
                # Local evaluation here can be replaced with deployment to external system.
                ax_client.complete_trial(trial_index=trial_index, raw_data=self._univariate_loss(parameters))
                pbar.update(1)  # progress bar

        dframe = ax_client.get_trials_data_frame().sort_values('varred_loss')

        return dframe


class SpatialModeler:
    def __init__(self, dataframe, xcoor, ycoor, feature, cell_size, directory):

        assert ~dataframe[feature].eq(0).any().any(), "There is at least one zero in your " + feature + " column."

        self.dataset = dataframe
        self.xcoor = xcoor
        self.ycoor = ycoor
        self.feature = feature
        self.cell_size = cell_size
        self.directory = directory

        # from _OutlierDetection class
        self._outliers = OutlierDetection(self.dataset, self.xcoor, self.ycoor, self.feature, self.directory)
        self.lag_dist = _compute_lag_dist(self.dataset, self.xcoor, self.ycoor)[0]
        self._distance = self._outliers.distance

        # from declus function
        self.declus_mean = None
        self._declus_wts = None
        self._declus_var = None

        # from _TrendModeler class
        self.trend_array = None
        self._trend_object = None
        self._trend_present = None
        self._df_study = None
        self._extra_cells = None

        self.coord_origin = None

        # from _SpatialContinuity class
        self._spatial_contin = None
        self.azi_rang_tensor = None
        self.semivar_tensor = None
        self._feat_to_model = None  # we can either model the residuals of original feature

        # from plot_semiv_model function
        self.vario_model = None

        # from _SeGauss
        self.realizations_array = None
        self.summary_realiz = None

        # from _Quality Check class
        self._QCClass = None

        # from _Visualization class
        self._Visualization = None

    def spatial_outliers(self, alpha, plot=True, update=False, xy_ratio=1):
        """
        Use the Mahalanobis distance to detect spatial outliers.

        Parameters
        ----------
        alpha: float
        Confidence value between 0. and 1.0. The smaller, the fewer values are outliers.

        plot: bool
        True to plot the classified data.

        update: bool
        True if you want to remove the outliers points from the dataset.

        xy_ratio: float
        X to Y ratio for visualization purposes.
        """
        lag_distance = self._outliers.spatial_outliers(alpha, plot, update, xy_ratio)

        if update:
            self.lag_dist = lag_distance[0]
            self._distance = self._outliers.distance

            # save dataframe
            self.dataset = self._outliers.df
            self.dataset.to_pickle(os.path.join(self.directory, 'updated_dataset.pkl'))

    def feature_outliers(self, selection, update=False, plot=False, verbose=False, contamination=0.02):
        """
        Use the standard deviation method, isolation forest, minimum covariance determinant, and local outlier factor,
        to estimate outliers in the feature of interest.

        Parameters
        ----------
        selection: int
            Select the standard deviation method[0], isolation forest [1], minimum covariance determinant [2],
            and local outlier factor [3] results.

        update: bool
            True to update the dataset after removing the ouliers.

        plot: bool
            True to plot the violin plots of the resulting datasets and the original input.

        verbose: bool
            True to print the number of inliers kept.

        contamination: float
            The fraction considered as outliers.

        Returns
        -------
        """

        self._outliers.feature_outlier(contamination, selection, update, plot, verbose)

        if update:
            self.dataset = self._outliers.df
            self.dataset.to_pickle(os.path.join(self.directory, 'updated_dataset.pkl'))

            self._outliers = OutlierDetection(self.dataset, self.xcoor, self.ycoor, self.feature, self.directory)
            self.lag_dist = _compute_lag_dist(self.dataset, self.xcoor, self.ycoor)[0]
            self._distance = self._outliers.distance

    def vario_plot(self, azimuths=None, extend_half_dist=1.0):
        """
        Plot the experimental and directional variograms.

        Parameters
        ----------
        extend_half_dist
        azimuths: list
            Optional list containing 4 azimuths to plot semivariograms.
        """
        iazi = None

        if azimuths is None:
            azimuths = [0, 30, 45, 60]
        tempo = [i + 90 if i < 90 else i - 90 for i in azimuths]
        azimuths += tempo  # add 90 degrees to directions you chose

        lag_tol = self.lag_dist * 2
        bandh = self.lag_dist * 2
        tmin = -999.
        tmax = 999.

        nlag = int((self._distance * extend_half_dist) / self.lag_dist)
        # Arrays to store the results
        lag = np.zeros((len(azimuths), nlag + 2))
        gamma = np.zeros((len(azimuths), nlag + 2))
        npp = np.zeros((len(azimuths), nlag + 2))

        # Loop over all directions
        for iazi in range(0, len(azimuths)):
            lag[iazi, :], gamma[iazi, :], npp[iazi, :] = geostats.gamv(
                self.dataset, self.xcoor, self.ycoor, self.feature, tmin, tmax, self.lag_dist,
                lag_tol, nlag, azimuths[iazi], 22.5, bandh, isill=1
            )

        simbolo = ['circle', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star', 'x', 'square']
        colores = ['blue', 'red', 'black', 'green', 'blue', 'red', 'black', 'green']

        # Create traces
        fig = make_subplots(rows=1, cols=2)
        # First subplot
        for iazi in range(0, 4):
            fig.add_trace(go.Scatter(
                x=np.round(lag[iazi, :-1], 2),
                y=np.round(gamma[iazi, :], 2),
                mode='markers',
                name='Azimuth:' + str(azimuths[iazi]),
                hovertemplate='<br><b>Gamma</b>: %{y:.2f}<br>' +
                              '<b>Lag distance</b>: %{x:.2f}<br>' +
                              '<b>%{text}</b>',
                text=[f'Number of pairs {i:.0f}' for i in npp[iazi, :]],
                marker=dict(size=7, symbol=simbolo[iazi],
                            color=colores[iazi])),
                row=1, col=1)

        # Second subplot
        for iazi in range(4, 8):
            fig.add_trace(go.Scatter(
                x=np.round(lag[iazi, :-1], 2),
                y=np.round(gamma[iazi, :], 2),
                mode='markers',
                name='Azimuth' + str(azimuths[iazi - 4]) + '\u00B1 90 :' + str(azimuths[iazi]),
                hovertemplate='<br><b>Gamma</b>: %{y:.2f}<br>' +
                              '<b>Lag distance</b>: %{x:.2f}<br>' +
                              '<b>%{text}</b>',
                text=[f'Number of pairs {i:.0f}' for i in npp[iazi, :]],
                marker=dict(size=7, symbol=simbolo[iazi],
                            color=colores[iazi])),
                row=1, col=2)

        # add the sill to both subplots
        fig.add_trace(go.Scatter(x=[0, 5e9], y=[1.0, 1.0],
                                 line=dict(color='firebrick', width=3, dash='dot'),
                                 name='Sill', text=npp[iazi, :],
                                 showlegend=False),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=[0, 5e9], y=[1.0, 1.0],
                                 # dash options include 'dash', 'dot', and 'dashdot'
                                 line=dict(color='firebrick', width=3, dash='dot'),
                                 name='Sill', text=npp[iazi, :],
                                 showlegend=False),
                      row=1, col=2)

        # Update xaxis properties
        fig.update_xaxes(title_text="Lag distance <b>h(m)<b>",
                         row=1, range=[0, self._distance * extend_half_dist], col=1)
        fig.update_xaxes(title_text="Lag distance <b>h(m)<b>",
                         row=1, range=[0, self._distance * extend_half_dist], col=2)

        # Update yaxis properties
        fig.update_yaxes(title_text="<b>\u03B3<b>", row=1, range=[0, np.max(gamma) * 1.02],
                         col=1)
        fig.update_yaxes(title_text="<b>\u03B3<b>", row=1, range=[0, np.max(gamma) * 1.02],
                         col=2)

        # Edit the plot
        fig.update_layout(title='Directional ' + self.feature + ' Variogram',
                          autosize=False,
                          width=950,
                          height=500,
                          template='simple_white', )

        fig.show()

    def declus(self, min_cell, max_cell, iminmax=None):
        """
        Compute the declustered mean to correct for sampling bias.
        Parameters
        ----------
        min_cell, max_cell: float
        Minimum and maximum cell sizes to use as bounds to compute the declustered mean.

        iminmax:int
        0 to maximize, 1 to minimize the mean.
        """
        if iminmax is None:
            skewness = skew(self.dataset[self.feature], bias=True)
            if skewness >= 0:  # there are smaller values wrt the mean
                iminmax = 0  # maximize
            else:
                iminmax = 1  # minimize

        self._declus_wts, _, _ = _declus(self.dataset, self.xcoor, self.ycoor, self.feature, iminmax,
                                         noff=10, ncell=100, cmin=min_cell, cmax=max_cell)

        weighted_stats = DescrStatsW(self.dataset[self.feature], weights=self._declus_wts, ddof=0)
        self.declus_mean = weighted_stats.mean
        self._declus_var = weighted_stats.var

        print(f"The declustered mean is      : {self.declus_mean:.3f}")
        print(f"The declustered variance is  : {self._declus_var:.3f}\n")

        print(f"The naive mean is            : {self.dataset[self.feature].mean():.3f}")
        print(f"The naive variance is        : {self.dataset[self.feature].var():.3f}\n")

        perc_correction = (self.declus_mean - self.dataset[self.feature].mean()
                           ) / self.dataset[self.feature].mean()
        print(f"The correction in means is   : {perc_correction:.2%}")

        # todo print optimal cell size for declustering

    def model_trend(self, xwindow, ywindow, theta, generations=1, verbose=True, extra_cells=15):
        """
        Model the geological trend of the feature of interest using a Gaussian convolutional window (GCW). Note: in
        case a trend does not exist, set generations=0.

        Parameters
        ----------
        generations: int
            Number of trials to find a solution using evolutionary algorithms.

        xwindow, ywindow: float
            Window size in X and Y (in cell size units) for the GCW.

        theta: float
            Rotation angle in degrees.

        verbose: bool
            True to print diagnostic statistics.

        extra_cells: int
            Additional cells to include beyond the extremes of the area of interest.
        """

        self._extra_cells = extra_cells

        self._trend_object = TrendModeler(
            self.dataset, self.xcoor, self.ycoor, self.feature, self.cell_size,
            self.declus_mean, self._declus_var, self._declus_wts, extra_cells
        )

        self.trend_array, self._trend_present, self._df_study = self._trend_object.get_trend_only(
            xwindow, ywindow, theta, generations, verbose
        )

        self.coord_origin = [self._trend_object.xmin, self._trend_object.xmax, self._trend_object.ymin,
                             self._trend_object.ymax]

    def _update_params_trend_exists(self):
        """
        Update parameters in function a trend exists or not.
        """
        if self._trend_present:  # the trend exists, model the residuals
            self._feat_to_model = 'Residuals'
        else:  # no trend model, model the original feature
            self._feat_to_model = self.feature

        self._df_study['NFeat'], _, _ = geostats.nscore(self._df_study, self._feat_to_model)

    def find_range_lag(self, bayesian_iterations, min_azimuth=0, max_azimuth=180, parallel=False, load_tensors=False):
        """
        Find the maximum direction of continuity and model the experimental semivariogram.

        Parameters
        ----------
        bayesian_iterations: int
            Number of iterations to identify the major direction of continuity between min_azimuth and max_azimuth.

        min_azimuth, max_azimuth: float
            Minimum and maximum azimuth values to consider when estimating the major direction of continuity.

        parallel: bool
            True to run Bayesian optimization using parallelism. Default is False.

        load_tensors: bool
            True to load the tensor results from a previous 'find_range_lag' run.
        """

        self._update_params_trend_exists()  # update parameters

        if parallel:
            self._spatial_contin = SpatialContinuityParallel(self._df_study, self.xcoor, self.ycoor)
        else:
            self._spatial_contin = SpatialContinuitySequential(self._df_study, self.xcoor, self.ycoor)

        if load_tensors:
            self.azi_rang_tensor = np.load(os.path.join(self.directory, "azi_rang_tensor" + self.feature + ".npy"))
            self.semivar_tensor = np.load(os.path.join(self.directory, "semivar_tensor" + self.feature + ".npy"))

        else:
            self.azi_rang_tensor, self.semivar_tensor = self._spatial_contin.get_directional_continuity(
                trials=bayesian_iterations,
                minimum_azimuth=min_azimuth,
                maximum_azimuth=max_azimuth
            )
            # save tensors
            np.save(os.path.join(self.directory, "semivar_tensor" + self.feature + ".npy"), self.semivar_tensor)
            np.save(os.path.join(self.directory, "azi_rang_tensor" + self.feature + ".npy"), self.azi_rang_tensor)

            print(f"Major range = {self.azi_rang_tensor[0, 1]:.2f}. Azimuth = {self.azi_rang_tensor[0, 0]:.2f}")
            print(f"Recommended lag distance = {self.lag_dist:.2f}.")

    def plot_semiv_model(self, model, max_distance, nlag, xlag):
        """
        Plot the major and minor variogram models (red curves) and the experimental semivariogram values.

        model: dict
            The variogram model in the geostatspy convention.

        max_distance: float
            The maximum lag distance to include in the horizontal axis.
        """
        _vario_model_plot(
            model, max_distance, self.semivar_tensor, self.azi_rang_tensor[0, 0], self.azi_rang_tensor[1, 0],
            model.get("hmaj2"), model.get("hmin2"), nlag, xlag
        )

    def simulation(self, realizations, vario_model, python=False, cosimulation=False, feat_secondary=None,
                   simulation_sec=None, zmin=None, zmax=None, varred=1.0, handle_negatives=True):
        """
        Perform sequential Gaussian simulation for one variable.

        Parameters
        -----------
        realizations: int
            Number of realizations to simulate.

        vario_model: dict
            Variogram model in geostatspy format.

        python: bool
            True to execute with Python. False to execute with GSLIB (recommended).

        cosimulation: bool
            True to perform cosimulation (it requires the realizations of the secondary feature).

        feat_secondary: str
            Name of the secondary function provided cosimulation is True.

        simulation_sec: np.ndarray
            Realizations from SGS provided cosimulation is True.

        zmin, zmax: float
            Minimum and maximum values to simulate. If you have a trend, they should correspond to that feature space.

        varred: float
            Variance reduction factor to correct for inflated variance. Range = [0.5-1.0]

        handle_negatives: bool
            The recommended value is True. Consider False if your inputs are negatives (e.g., log10 permeability).

        Returns
        -----------
        realizations_array, summary: np.ndarray
            realizations_array contains 'realizations' simulations and are required for cosimulation. summary contains
            the P10, P50, P90, and other metrics of realizations_array.
        """
        assert self.vario_model is None, "Model the semivariogram first."
        if python:
            clean, nwells = _check_rept_coords(self._df_study, self.xcoor, self.ycoor)
            message = f"Dataset has {nwells} repeated coordinate samples.Delete repeated samples to run Python version."
            assert clean, message

        if cosimulation:
            assert isinstance(feat_secondary, str), "Include the name of the secondary feature."
            assert isinstance(simulation_sec, np.ndarray), "Include realizations of the secondary feature as a numpy " \
                                                           "array. "
            if len(simulation_sec.shape) == 2:
                print(f"You only have one surface from the secondary feature available.")
                simulation_sec = np.repeat(simulation_sec[np.newaxis, ...], repeats=realizations, axis=0)

            # get the normal scores of the secondary feature
            self._df_study['NFeatSec'], _, _ = geostats.nscore(self.dataset, feat_secondary)
            self._df_study['Sec_feat'] = self.dataset[feat_secondary]

            sec_sim_valid = np.nan_to_num(simulation_sec, nan=self.declus_mean)
            sgs = MarkovBayesModel(self._df_study, self.xcoor, self.ycoor, self._feat_to_model, self.cell_size,
                                   self._trend_object, self._spatial_contin, vario_model, sec_sim_valid, varred,
                                   handle_negatives)

        else:  # no cosimulation
            sgs = SeqGaussSim(self._df_study, self.xcoor, self.ycoor, self._feat_to_model, self.cell_size,
                              self._trend_object, self._spatial_contin, vario_model, handle_negatives)

        realizations_array, summary = sgs.sgsimulation(
            realizations, output='simulatorexe.txt', feature_to_input=self.feature, python=python, zmin=zmin, zmax=zmax
        )

        self._instantiate_classes(cosimulation, realizations_array, summary, feat_secondary)

        return realizations_array, summary

    def _instantiate_classes(self, cosimulation, realizations_array, summary, feat_secondary):
        self.realizations_array = realizations_array
        self.summary_realiz = summary
        # instantiate the post simulation class
        if self._trend_present:
            arr = self.realizations_array - self.trend_array
        else:
            arr = self.realizations_array

        self._QCClass = RealizationsQC(self._df_study, self.xcoor, self.ycoor, self.feature, self.cell_size,
                                       self.realizations_array, arr,
                                       self.summary_realiz, self._extra_cells, self.lag_dist,
                                       self._trend_object, self._spatial_contin)

        self._Visualization = Visualization(self._df_study, self.xcoor, self.ycoor, self.feature, self.cell_size,
                                            self.realizations_array, self.summary_realiz, self._trend_object)
        if cosimulation:
            self._Visualization.sec_feat = feat_secondary

    def histogram_reprod(self, nbins=None, save_img=False):
        """
        Plot the histograms of the realizations and the data.

        Parameters
        ----------
        save_img

        nbins: int
            Number of bins to consider.

        save_img: bool
            True to save image as png.
        """
        assert self._QCClass is not None, "Run the simulations first."
        assert self._declus_wts is not None, "Run declustering first."

        self._QCClass.histograms(self._declus_wts, nbins, save_img)

    def qq_plot(self, realization=0, step=0.05):
        """
        Plot the weighted Q-Q plot of one realization and the input data.
        Parameters
        ----------
        realization: int
            Realization to plot.

        step: float
            Distance among quantiles (e.g., 0.05, 0.1, etc.).
        """
        assert self._QCClass is not None, "Run the simulations first."
        assert self._declus_wts is not None, "Run declustering first."

        self._QCClass.qq_plot_f(realization, step, self._declus_wts)

    def vario_reprod(self, azimuth_1, azimuth_2, xrange, vario_model, n_variograms=1, save_img=False):
        """
        Visually check whether the realizations of the residuals in normal score (RNS) space replicate the variogram
        model of the RNS in four directions of continuity.

        Parameters
        ----------
        azimuth_1, azimuth_2: int
            Choose from 4 azimuths: 0, 45, 90, 135 degrees.

        xrange: float

        vario_model: dict
            The variogram model of the original feature in its original space.

        n_variograms: int
            Number of gridded variograms to plot. It should be smaller or equal to the number of realizations.
             The larger, the more computationally expensive.

        save_img: bool
            True to save the image as png.
        """
        assert self._QCClass is not None, "Run the simulations first."
        self._QCClass.semiv_reprod(azimuth_1, azimuth_2, xrange, vario_model, save_img, n_variograms)

    def vario_map(self, nx, ny, realization_n=0, standardize=0, residual_space=False):
        """
        Plot the variogram map for a realization.

        Parameters
        ----------
        nx, ny: int
            The number of nodes in the x, y coordinate directions.

        realization_n: int
            What realization to consider

        standardize: int
            If set to 1, the semivariogram values will be divided by the variance.

        residual_space: bool
            True to plot the variogram map of the residual normal scores space; False to plot the varmap of the original
            input space.
        """
        self._QCClass.variogram_map(nx, ny, realization_n, standardize, residual_space)

    def get_xarray(self, x_name=None, y_name=None, metadata_dict=None):
        """
        Convenience function to transform the summary numpy array to a xarray Dataset.

        Parameters
        ----------
        x_name, y_name: str
            Name of the x and y axes.
        metadata_dict: dict
            Metadata regarding the spatial interpolation.

        Returns
        -------
        xarray.Dataset
            The summary maps.
        """
        assert self._Visualization is not None, "Run the simulations first."

        xarray_set = self._Visualization.get_xarray(x_name, y_name, metadata_dict)

        return xarray_set

    def summary_plots(self, xy_ratio=None):
        """
        Plot the summary of the realizations and the trend.

        Parameters
        ----------
        xy_ratio: float
            The X to Y ratio for visualization purposes.
        """
        assert self._Visualization is not None, "Run the simulations first."
        if xy_ratio is None:
            xy_ratio = 1
        self._Visualization.plot_simulation(xy_ratio)

    def local_prob_exceedance(self, threshold, save_img=False):
        """
        Plot the probability of exceeding the threshold at all locations.

        Parameters
        ----------
        threshold: float
            The feature value in original units to consider.

        save_img: bool
            True to save image as png.
        """
        assert self._Visualization is not None, "Run the simulations first."
        self._Visualization.local_prob_exc(threshold, save_img)

    def cosim_comparison(self, secondary_realizations, sec_feat, hexsize=100, save_img=False):
        """
        Visually compare whether the cosimulations replicate the correlation between primary and secondary feature.

        Parameters
        ----------
        save_img
        secondary_realizations: np.ndarray
            Array that contains the SGS realizations of the secondary feature.

        sec_feat: str
            Name of the secondary feature in the dataframe.

        hexsize: int
            The hex bin size.
        """
        assert self._Visualization.sec_feat is not None, "Perform cosimulation first."
        self._df_study[sec_feat] = self.dataset[sec_feat]  # add the column to
        self._Visualization.cosim_plot(secondary_realizations, sec_feat, hexsize, save_img)

    def get_summary(self):
        """
        Save the summary results in a CSV file.

        Returns
        -------
        pd.DataFrame
            The summary dataframe with summary values as columns, and their coordinates.
        """
        xmin = self._trend_object.xmin
        cellsx = self.summary_realiz.shape[2]
        ymin = self._trend_object.ymin
        cellsy = self.summary_realiz.shape[1]
        x = np.arange(
            xmin + self.cell_size / 2, xmin + self.cell_size / 2 + self.cell_size * cellsx, self.cell_size)
        y = np.arange(
            ymin + self.cell_size / 2, ymin + self.cell_size / 2 + self.cell_size * cellsy, self.cell_size)

        xx, yy = np.meshgrid(x, y)  # create a meshgrid to simplify the addition of coordinates
        results = pd.DataFrame(xx.ravel(), columns=[self.xcoor])
        results[self.ycoor] = yy.ravel()
        results['P10'] = self.summary_realiz[0].ravel()
        results['P50'] = self.summary_realiz[1].ravel()
        results['P90'] = self.summary_realiz[2].ravel()
        results['Uncertainty'] = self.summary_realiz[3].ravel()
        results['Expectation'] = self.summary_realiz[4].ravel()
        results['Trend'] = self.summary_realiz[5].ravel()
        results.dropna(inplace=True)

        return results
