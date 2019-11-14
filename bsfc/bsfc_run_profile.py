# -*- coding: utf-8 -*-
"""
Run nested sampling for model selection.

This script can either be run on 1 CPU using
python bsfc_run_ns.py <SHOT>

or with MPI, using
mpirun python bsfc_run_ns.py <SHOT>
whereby the maximum number of workers will be automatically identified.

After completion of a MultiNest execution, running again this script (without mpirun!) will
pull up some useful plots.

@author: sciortino
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

#import cPickle as pkl
#import pdb
#import corner
import sys
import time as time_
import os
#import shutil
#import scipy

import argparse

from bsfc_moment_fitter import MomentFitter

from helpers import bsfc_cmod_shots

# To be removed before public release:
sys.path.insert(0,'/home/sciortino/usr/pythonmodules/PyMultiNest')

parser = argparse.ArgumentParser()
parser.add_argument("shot", type=int, help="shot number to run analysis on")
parser.add_argument("n_hermite", type=int, help="number of hermite functions")
#parser.add_argument("tbin", type=int, help="tbin of fit")
parser.add_argument('-n', "--noline", action="store_true", help="Whether or not to remove the wn5 line")
#parser.add_argument('-f', "--force", action="store_true", help="whether or not to force an overwrite of saved data")

args = parser.parse_args()

# first command line argument gives shot number
#shot = args.shot

shot = args.shot

primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)

#tbin = args.tbin


# Start counting time:
start_time=time_.time()


if 'BSFC_ROOT' not in os.environ:
    # make sure that correct directory is pointed at
    os.environ["BSFC_ROOT"]='%s/bsfc'%str(os.path.expanduser('~'))

# location of MultiNest chains
n_hermite = args.n_hermite

if args.noline:
    print "Fitting without wn5"
    basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/nl_nh%d_t%d-.'%(n_hermite, tbin))
else:
    basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/nh%d_t%d-.'%(n_hermite, tbin))

def removeFiles(basename):
    import glob
    fileList = glob.glob(basename+'*')
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            pass


# if this wasn't run before, initialize the moment fitting class

if args.noline:
    mf = MomentFitter(primary_impurity, primary_line, shot, tht=tht, nofit=['wn5', 'lyas1', 'lyas2', 'lyas3', 'lyas4'])
else:
    mf = MomentFitter(primary_impurity, primary_line, shot, tht=tht)

try:
    if args.noline:
        data = np.load('../bsfc_fits/fit_data/mf_%d_nl_nh%d_t%d.npz'%(shot,3,tbin))
    else:
        data = np.load('../bsfc_fits/fit_data/mf_%d_nh%d_t%d.npz'%(shot,3,tbin))

    meas_avg = data['meas_avg']
    meas_std = data['meas_std']
    meas_true = np.zeros((3, mf.maxChan))
    lnev = data['lnev']
    lnev_std = data['lnev_std']
except:
    meas_avg = np.zeros((3, mf.maxChan))
    meas_std = np.zeros((3, mf.maxChan))
    meas_true = np.zeros((3, mf.maxChan))
    lnev = np.zeros(mf.maxChan)
    lnev_std = np.zeros(mf.maxChan)



for chbin in range(mf.maxChan):
    print "Fitting chbin", chbin

    if lnev[chbin] != 0.0:
        print "Already fit, skipping"
        continue

    removeFiles(basename)

    # First try nonlinear fitting to see if the bin is worth fitting
    mf.fitSingleBin(tbin=tbin, chbin=chbin, nsteps=1, n_hermite=3)

    if not mf.fits[tbin][chbin].good:
        meas_avg[:,chbin] = np.nan
        lnev[chbin] = np.nan
        continue


    mf.fitSingleBin(tbin=tbin, chbin=chbin,NS=True, n_hermite=n_hermite, n_live_points=400,
                    sampling_efficiency=0.3, verbose=True, basename=basename)
    mf.fits[tbin][chbin].NS_analysis(basename)

    samples = mf.fits[tbin][chbin].samples
    sample_weights = mf.fits[tbin][chbin].sample_weights


    measurements = np.apply_along_axis(mf.fits[tbin][chbin].lineModel.modelMeasurements, 1, samples)
    moms = np.average(measurements, 0, weights=sample_weights)
    moms_std = np.sqrt(np.average((measurements-moms)**2, 0, weights=sample_weights))

    if args.noline:
        np.savez_compressed('../bsfc_fits/fit_data/mf_%d_nl_nh%d_t%d_ch%d.npz'%(shot,n_hermite,tbin,chbin),
                samples=samples, sample_weights=sample_weights, measurements=measurements)
    else:
        np.savez_compressed('../bsfc_fits/fit_data/mf_%d_nh%d_t%d_ch%d.npz'%(shot,n_hermite,tbin,chbin),
                samples=samples, sample_weights=sample_weights, measurements=measurements)
    meas_avg[:,chbin] = moms
    meas_std[:,chbin] = moms_std
    lnev[chbin], lnev_std[chbin] = mf.fits[tbin][chbin].lnev

    if args.noline:
        np.savez('../bsfc_fits/fit_data/mf_%d_nl_nh%d_t%d.npz'%(shot,n_hermite,tbin),
            meas_avg=meas_avg, meas_std=meas_std, lnev=lnev, lnev_std=lnev_std)
    else:
        np.savez('../bsfc_fits/fit_data/mf_%d_nh%d_t%d.npz'%(shot,n_hermite,tbin),
            meas_avg=meas_avg, meas_std=meas_std, lnev=lnev, lnev_std=lnev_std)

    # end time count
    elapsed_time=time_.time()-start_time
    print 'Time to run: ' + str(elapsed_time) + " s"



#with open('../bsfc_fits/mf_synth_%d_nh%d.pkl'%(shot,n_hermite),'wb') as f:
#    pkl.dump(mf, f)



