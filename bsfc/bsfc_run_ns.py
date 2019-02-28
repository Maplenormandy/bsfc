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

from bsfc_moment_fitter import *

# To be removed before public release:
sys.path.insert(0,'/home/sciortino/usr/pythonmodules/PyMultiNest')

# first command line argument gives shot number
shot = int(sys.argv[1])

# Start counting time:
start_time=time_.time()

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)
    
if 'BSFC_ROOT' not in os.environ:
    # make sure that correct directory is pointed at
    os.environ["BSFC_ROOT"]='%s/bsfc'%str(os.path.expanduser('~'))

# location of MultiNest chains
basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/c-.' )


# try loading result
try:
    with open('../bsfc_fits/mf_NS_%d_tbin%d_chbin_%d.pkl'%(shot,tbin,chbin),'rb') as f:
        mf=pkl.load(f)
    loaded = True; print "Loaded previous result"
    
except:
    # if this wasn't run before, initialize the moment fitting class
    mf = MomentFitter(primary_impurity, primary_line, shot, tht=0)
    loaded = False
        
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
    mf.fitSingleBin(tbin=tbin, chbin=chbin,NS=True, n_hermite=3, n_live_points=400,
                    sampling_efficiency=0.3, verbose=True)
    
    # save fits for future use
    with open('../bsfc_fits/mf_NS_%d_tbin%d_chbin_%d.pkl'%(shot,tbin,chbin),'wb') as f:
        pkl.dump(mf, f)       
    
if loaded==True:
        
    # load MultiNest output
    mf.fits[tbin][chbin].NS_analysis(basename)

    samples = mf.fits[tbin][chbin].samples
    sample_weights = mf.fits[tbin][chbin].sample_weights
    
    # corner plot
    f = gptools.plot_sampler(
        samples,
        sample_weights,
        labels=mf.fits[tbin][chbin].lineModel.thetaLabels(),
        chain_alpha=1.0,
        cutoff_weight=0.01,
        cmap='plasma',
        plot_samples=False,
        plot_chains=False,
    )
    
    mf.plotSingleBinFit(tbin=tbin, chbin=chbin)

    '''
    # Get model predictions
    # \mu = \frac{\sum_i w_i x_i}{\sum_i w_i}
    moms=np.apply_along_axis(mf.fits[tbin][chbin].lineModel.modelMeasurements, axis=1, arr=samples)
    meanw = scipy.einsum('i...,i...->i...', sample_weights, moms).sum(axis=0)/sample_weights.sum(axis=0)

    # weighted variance: s^2 = \frac{\sum_i w_i}{(\sum_i w_i)^2 - \sum_i w_i^2}\sum_i w_i (x_i - \mu)^2
    V1 = sample_weights.sum(axis=0)
    M = scipy.einsum('i...,i...->i...', sample_weights, (moms - meanw)**2).sum(axis=0)
    stdw = np.sqrt(V1/M)
    
    print "Counts = ", meanw[0], "+/-", stdw[0]
    print "v = ", meanw[1], "+/-", stdw[1]
    print "Ti = ", meanw[2], "+/-", stdw[2]
    '''
    
# Import mpi4py here to output timing only once
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    # end time count
    elapsed_time=time_.time()-start_time
    print 'Time to run: ' + str(elapsed_time) + " s"

