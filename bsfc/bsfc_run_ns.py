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
import multiprocessing
import os
import shutil
import scipy
from helpers import bsfc_cmod_shots
from helpers import bsfc_autocorr

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import argparse
from bsfc_moment_fitter import *

n_hermite=3

# To be removed before public release:
if '/home/sciortino/usr/pythonmodules/PyMultiNest' not in sys.path:
    sys.path.insert(0,'/home/sciortino/usr/pythonmodules/PyMultiNest')

parser = argparse.ArgumentParser()
parser.add_argument("shot", type=int, help="shot number to run analysis on")
parser.add_argument('-f', "--force", action="store_true", help="whether or not to force an overwrite of saved data")

args = parser.parse_args()

# first command line argument gives shot number
shot = args.shot

# Start counting time:
start_time=time_.time()

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)

if 'BSFC_ROOT' not in os.environ:
    # make sure that correct directory is pointed at
    os.environ["BSFC_ROOT"]='%s/bsfc'%str(os.path.expanduser('~'))

# location of MultiNest chains
basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/c-.' )
chains_dir = os.path.dirname(basename)


if rank==0:
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
            with open('../bsfc_fits/mf_NS_%d_tbin%d_chbin_%d.pkl'%(shot,tbin,chbin),'rb') as f:
                mf=pkl.load(f)
            loaded = True; print "Loaded previous result"
        except:
            loaded = False

    if not loaded:
        # if this wasn't run before, initialize the moment fitting class
        mf = MomentFitter(primary_impurity, primary_line, shot, tht=0)

    '''
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
    '''
else:
    mf=None
    loaded=None
        
# broadcast r object to all cores
mf = comm.bcast(mf, root = 0)
loaded = comm.bcast(loaded, root = 0)

# ==================================

if loaded==False:

    # Do a single spectral fit with nested sampling
    mf.fitSingleBin(tbin=tbin, chbin=chbin,NS=True,n_live_points=1000,
                    sampling_efficiency=0.3,verbose=True,const_eff=True,
                    n_hermite=n_hermite)
    
    # save fits for future use
    if rank==0:
        with open('../bsfc_fits/mf_NS_%d_tbin%d_chbin_%d.pkl'%(shot,tbin,chbin),'wb') as f:
            pkl.dump(mf, f)

if loaded==True:
    # DO NOT try to load and plot with multiple workers (i.e. using mpirun)!
    
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

    mf.plotSingleBinFit(tbin=tbin, chbin=chbin)

    measurements = np.apply_along_axis(mf.fits[tbin][chbin].lineModel.modelMeasurements, 1, samples)
    moms = np.average(measurements, 0, weights=sample_weights)
    moms_std = np.sqrt(np.average((measurements-moms)**2, 0, weights=sample_weights))

    print "Counts = ", moms[0], "+/-", moms_std[0]
    print "v = ", moms[1], "+/-", moms_std[1]
    print "Ti = ", moms[2], "+/-", moms_std[2]
<<<<<<< HEAD
    print "ln(ev) = ", mf.fits[tbin][chbin].lnev[0], "+/-", mf.fits[tbin][chbin].lnev[1]
    print "# Hermite polynomials: ", n_hermite
    
=======


# Import mpi4py here to output timing only once
#from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
>>>>>>> c8096d20c1ef49c3089afc2f5743e5db42ad6234

if rank==0:
    # end time count
    elapsed_time=time_.time()-start_time
    print 'Time to run: ' + str(elapsed_time) + " s"

