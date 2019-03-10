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

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import cPickle as pkl
import pdb
import corner
import bsfc_main
import sys
import time as time_
import os
import shutil
import scipy
from analysis.bsfc_synthetic_generator import SyntheticGenerator

import argparse

from bsfc_moment_fitter import *

# To be removed before public release:
sys.path.insert(0,'/home/sciortino/usr/pythonmodules/PyMultiNest')

parser = argparse.ArgumentParser()
#parser.add_argument("shot", type=int, help="shot number to run analysis on")
parser.add_argument('-f', "--force", action="store_true", help="whether or not to force an overwrite of saved data")

args = parser.parse_args()

# first command line argument gives shot number
#shot = args.shot

shot = 1160920007
tht = 0

# Start counting time:
start_time=time_.time()


if 'BSFC_ROOT' not in os.environ:
    # make sure that correct directory is pointed at
    os.environ["BSFC_ROOT"]='%s/bsfc'%str(os.path.expanduser('~'))

# location of MultiNest chains
basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/c-.' )
tbin = 16
chbin = 8

# try loading result
if args.force:
    # Force us to not load the pickle
    loaded = False
    # Remove all the chains
    import glob
    fileList = glob.glob(basename+'*')
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            pass

else:
    try:
        with open('../bsfc_fits/mf_synth_%d.pkl'%(shot),'rb') as f:
            mf=pkl.load(f)
        loaded = True; print "Loaded previous result"
    except:
        loaded = False

if not loaded:
    # if this wasn't run before, initialize the moment fitting class
    mf = MomentFitter('Ar', 'lya1', shot, tht=tht)
    sg = SyntheticGenerator(shot, tht, False, 'lya1', tbin)
    sg.generateSyntheticSpectrum(mf, chbin)

    # check that empty directory exists for MultiNest output:
    chains_dir = os.path.dirname(basename)

    # delete and re-create directory for MultiNest output
    if os.path.exists(chains_dir):
        if len(os.listdir(chains_dir))==0:
            # if directory exists and is empty, everything's ready
            pass
        else:
            # directory from previous run exists. Delete and re-create it
            shutil.rmtree(chains_dir)
            os.mkdir(chains_dir)
    else:
        # if directory does not exist, create it
        os.mkdir(chains_dir)


# ==================================

if loaded==False:

    # Do a single spectral fit with nested sampling
    mf.fitSingleBin(tbin=tbin, chbin=chbin,NS=True, n_hermite=5, n_live_points=400,
                    sampling_efficiency=0.3, verbose=True)

    # save fits for future use
    with open('../bsfc_fits/mf_synth_%d.pkl'%(shot),'wb') as f:
        pkl.dump(mf, f)

if loaded==True:

    # load MultiNest output
    mf.fits[tbin][chbin].NS_analysis(basename)

    samples = mf.fits[tbin][chbin].samples
    sample_weights = mf.fits[tbin][chbin].sample_weights

    # corner plot
    f = gptools.plot_sampler(
        samples,
        weights=sample_weights,
        labels=mf.fits[tbin][chbin].lineModel.thetaLabels(),
        chain_alpha=1.0,
        cutoff_weight=0.01,
        cmap='plasma',
        plot_samples=False,
        plot_chains=False,
    )

    sg = SyntheticGenerator(shot, tht, False, 'lya1', tbin)
    true_meas = sg.calculateTrueMeasurements(mf, chbin)

    mf.plotSingleBinFit(tbin=tbin, chbin=chbin)

    measurements = np.apply_along_axis(mf.fits[tbin][chbin].lineModel.modelMeasurements, 1, samples)
    moms = np.average(measurements, 0, weights=sample_weights)
    moms_std = np.sqrt(np.average((measurements-moms)**2, 0, weights=sample_weights))


    print "Counts = ", moms[0], "+/-", moms_std[0]
    print "v = ", moms[1], "+/-", moms_std[1]
    print "Ti = ", moms[2], "+/-", moms_std[2]

    print "(true) v = ", true_meas[1]
    print "(true) Ti = ", true_meas[2]


# Import mpi4py here to output timing only once
#from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    # end time count
    elapsed_time=time_.time()-start_time
    print 'Time to run: ' + str(elapsed_time) + " s"

