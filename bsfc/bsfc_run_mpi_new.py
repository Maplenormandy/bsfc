# -*- coding: utf-8 -*-
"""
Run a series of spectral fits for a tokamak discharge in a chosen time range.
To run, use  
mpirun python <SHOT> 
where <SHOT> is the CMOD shot number of interest. 

To visualize results, after the above mpirun, run
python  <SHOT> 
i.e. the same, without the 'mpirun' command. 

@author: sciortino
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import cPickle as pkl
import pdb
import corner
import sys
import time as time_
import multiprocessing
import os
import shutil
import scipy
from helpers import bsfc_cmod_shots
from helpers import bsfc_autocorr

from bsfc_moment_fitter import *

# MPI parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# first command line argument gives shot number
shot = int(sys.argv[1])

if 'BSFC_ROOT' not in os.environ:
    # make sure that correct directory is pointed at
    os.environ["BSFC_ROOT"]='%s/bsfc'%str(os.path.expanduser('~'))
        
# set resume=False if reloading previous fits should NOT be attempted
resume=True

# indicate whether to output verbose text 
verbose = True

if rank == 0:
    print "Analyzing shot ", shot
    
    # Start counting time:
    t_start=MPI.Wtime()

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)

# ==============

# Always create new object by default when running with MPI
if rank==0:
    mf = MomentFitter(primary_impurity, primary_line, shot, tht=tht)
else:
    mf = None


if size==1:
    # if only 1 core is being used, assume that script is being used for plotting

    with open('./bsfc_fits/moments_%d_tmin%f_tmax%f.pkl'%(shot,t_min,t_max),'rb') as f:
            gathered_moments=pkl.load(f)

    # clean up Hirex-Sr signals -- BR_THRESH might need to be adjusted to eliminate outliers
    moments_vals, moments_stds, time_sel = clean_moments(mf.time, mf.maxChan, t_min,t_max,
                                                         gathered_moments, BR_THRESH=2.0, BR_STD_THRESH=0.1)
    # BSFC slider visualization
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='br')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='vel')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='Temp')

    plt.show(block=True)

else:
    # Run MPI job:

    # broadcast r object to all cores
    mf = comm.bcast(mf, root = 0)  
    
    tidx_min = np.argmin(np.abs(mf.time - t_min))
    tidx_max = np.argmin(np.abs(mf.time - t_max))
    time_sel= mf.time[tidx_min: tidx_max]
    
    # create new location for MultiNest chains of each worker
    basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/c-.' )
    
    # check that empty directory exists for MultiNest output: 
    chains_dir = os.path.dirname(basename)
    
    # delete and re-create directory for MultiNest output
    if rank==0:
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

    comm.Barrier()
    # map range of channels and compute each
    map_args_tpm = list(itertools.product(range(tidx_min, tidx_max), range(mf.maxChan)))
    map_args = [list(a) for a in map_args_tpm]
    njobs = len(map_args)
    
    # run MultiNest jobs in series, parallelizing internally
    # This differs from throughput parallelization, which would MultiNest to not automatically
    # use maximum number of workers available
    assigned_bins = map_args
    
    res = np.asarray([ None for yy in range(len(assigned_bins)) ])

    # ===============================
    # Actual evaluation for each worker
    for j, binn in enumerate(assigned_bins):

        if resume:
            # create checkpoint directory if it doesn't exist yet
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')

            if not os.path.exists('checkpoints/%d_tmin%f_tmax%f'%(shot,t_min,t_max)):
                os.makedirs('checkpoints/%d_tmin%f_tmax%f'%(shot,t_min,t_max))

            try:
                # if fit has already been created, re-load it from checkpoint directory
                resfile ='./checkpoints/%d_tmin%f_tmax%f/moments_%d_bin%d_%d.pkl'%(shot,t_min,t_max,shot,binn[0],binn[1])
                with open(resfile,'rb') as f:
                    res[j] = pkl.load(f)
                print "Loaded fit moments from ", resfile

            except:
                # create new fit      
                mf.fitSingleBin(tbin=binn[0], chbin=binn[1],NS=True, n_hermite=3, n_live_points=400,
                                sampling_efficiency=0.3, verbose=verbose)
                
                if mf.fits[binn[0]][binn[1]].good ==True:

                    # load MultiNest output
                    mf.fits[binn[0]][binn[1]].NS_analysis(basename)
                    samples = mf.fits[binn[0]][binn[1]].samples
                    sample_weights = mf.fits[binn[0]][binn[1]].sample_weights
                    
                    # Get model predictions
                    # \mu = \frac{\sum_i w_i x_i}{\sum_i w_i}
                    meanw = scipy.einsum('i...,i...->i...', sample_weights,samples).sum(axis=0)/sample_weights.sum(axis=0)
                    moms=mf.fits[binn[0]][binn[1]].lineModel.modelMeasurements(meanw)
    
                    # weighted variance: s^2 = \frac{\sum_i w_i}{(\sum_i w_i)^2 - \sum_i w_i^2}\sum_i w_i (x_i - \mu)^2
                    V1 = sample_weights.sum(axis=0)
                    M = scipy.einsum('i...,i...->i...', sample_weights, (samples - meanw)**2).sum(axis=0)
                    stdw = V1 / (V1**2 - (sample_weights**2).sum(axis=0))*M
    
                    # get standard deviation of moments:
                    moms_up = mf.fits[binn[0]][binn[1]].lineModel.modelMeasurements(meanw+stdw)
                    moms_down = mf.fits[binn[0]][binn[1]].lineModel.modelMeasurements(meanw-stdw)
                    moms_std = (moms_up - moms_down)/2.0

                    res[j] = [moms, moms_std]
                else:
                    res[j] =[np.asarray([np.nan, np.nan, np.nan]), np.asarray([np.nan, np.nan, np.nan])]

                with open(resfile,'wb') as f:
                    pkl.dump(res[j],f)

        else:
                
            mf.fitSingleBin(tbin=binn[0], chbin=binn[1],NS=True, n_hermite=3, n_live_points=400,
                                sampling_efficiency=0.3, verbose=verbose)
       
            if mf.fits[binn[0]][binn[1]].good ==True:
                # load MultiNest output
                mf.fits[binn[0]][binn[1]].NS_analysis(basename)
                samples = mf.fits[binn[0]][binn[1]].samples
                sample_weights = mf.fits[binn[0]][binn[1]].sample_weights
                
                # Get model predictions
                # \mu = \frac{\sum_i w_i x_i}{\sum_i w_i}
                meanw = scipy.einsum('i...,i...->i...', sample_weights,samples).sum(axis=0)/sample_weights.sum(axis=0)
                moms=mf.fits[binn[0]][binn[1]].lineModel.modelMeasurements(meanw)
                
                # weighted variance: s^2 = \frac{\sum_i w_i}{(\sum_i w_i)^2 - \sum_i w_i^2}\sum_i w_i (x_i - \mu)^2
                V1 = sample_weights.sum(axis=0)
                M = scipy.einsum('i...,i...->i...', sample_weights, (samples - meanw)**2).sum(axis=0)
                stdw = V1 / (V1**2 - (sample_weights**2).sum(axis=0))*M
                
                # get standard deviation of moments:
                moms_up = mf.fits[binn[0]][binn[1]].lineModel.modelMeasurements(meanw+stdw)
                moms_down = mf.fits[binn[0]][binn[1]].lineModel.modelMeasurements(meanw-stdw)
                moms_std = (moms_up - moms_down)/2.0
                
                res[j] = [moms, moms_std]
            else:
                res[j] = [np.asarray([np.nan, np.nan, np.nan]), np.asarray([np.nan, np.nan, np.nan])]

                
    # ===============================

    # collect results on rank=0 process:
    gathered_res = comm.gather(res, root=0)

    #join all results
    if rank==0:
        gath_res = np.concatenate(gathered_res)
        gathered_moments= np.asarray(gath_res).reshape((tidx_max-tidx_min,mf.maxChan))

        print "*********** Completed fits *************"
    
        # save fits for future use
        with open('./bsfc_fits/moments_%d_tmin%f_tmax%f.pkl'%(shot,t_min,t_max),'wb') as f:
            pkl.dump(gathered_moments, f, protocol=pkl.HIGHEST_PROTOCOL)

        if resume:
            # eliminate checkpoint directory if this was created
            shutil.rmtree('checkpoints/%d_tmin%f_tmax%f'%(shot,t_min,t_max))
        
        # end time count
        elapsed_time = MPI.Wtime() - t_start
        
        print 'Time to run: ' + str(elapsed_time) + " s"
