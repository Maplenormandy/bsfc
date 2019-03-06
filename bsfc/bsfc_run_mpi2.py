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
import sys
import itertools
import os
import shutil
import scipy
import glob
import subprocess

#import bsfc_main
from bsfc_moment_fitter import *
from helpers import bsfc_clean_moments 
from helpers import bsfc_slider
from helpers import bsfc_cmod_shots
from bsfc_bin_fit import _fitTimeWindowWrapper

# MPI parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# first command line argument gives shot number
shot = int(sys.argv[1])

# select whether to use nested sampling
NS=True

# fix number of steps for MCMC (only used if NS=False)
nsteps = 10000

# set resume=False if reloading previous fits should NOT be attempted
resume=True

# indicate whether to output verbose text 
verbose = True

if not verbose:
    import warnings
    warnings.filterwarnings("ignore")

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
    with open('../bsfc_fits/mf_%d_tmin%f_tmax%f.pkl'%(shot,t_min,t_max),'wb') as f:
        pkl.dump(mf,f)
else:
    mf = None

    
if size==2:
    # if only 1 core is being used, assume that script is being used for plotting

    with open('../bsfc_fits/moments_%d_tmin%f_tmax%f.pkl'%(shot,t_min,t_max),'rb') as f:
        gathered_moments=pkl.load(f)

    # clean up Hirex-Sr signals -- BR_THRESH might need to be adjusted to eliminate outliers
    moments_vals, moments_stds, time_sel = clean_moments(mf.time, mf.maxChan, t_min,t_max,
                                                         gathered_moments, BR_THRESH=2e3, BR_STD_THRESH=2e3)
    
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

    # map range of channels and compute each
    map_args_tpm = list(itertools.product(range(tidx_min, tidx_max), range(mf.maxChan)))
    map_args = [list(a) for a in map_args_tpm]

    # we have comm.size cores available. Total number of jobs to be run:
    njobs = len(map_args)
    
    # Block-cyclic parallelization scheme
    extra_njobs = njobs - njobs//size * size

    if rank < extra_njobs - 1 :   #first "extra_njobs" workers take extra task (NB: Python numbering starts at 0)
        personal_njobs = njobs // size + 1
        assigned_jobs_offset = rank * personal_njobs 
    else:
        personal_njobs = njobs // size
        assigned_jobs_offset = rank * personal_njobs + extra_njobs

    assigned_bins = map_args[assigned_jobs_offset:assigned_jobs_offset+personal_njobs]
    
    # now, fill up result array for each worker:
    res = np.asarray([ None for yy in range(len(assigned_bins)) ])

    # define spectral_fit function based on whether emcee or MultiNest should be run
    if NS:
        def spectral_fit(shot,t_min,t_max,tb,cb,verbose,rank):
            # individual fit with MultiNest
            command = ['python','launch_NSfit.py',str(shot),str(t_min), str(t_max),
                       str(int(tb)),str(int(cb)),str(int(verbose)),str(int(rank))]
                       #'{:d} {:.2f} {:.2f}'.format(shot,t_min,t_max),
                       #'{:d} {:d} {:d} {:d}'.format(tb,ch,verbose,rank)]
        
            # Run without MPI:
            out = subprocess.check_output(command, stderr=subprocess.STDOUT)
            
    else:
        # emcee requires a wrapper for multiprocessing to pickle the function
        spectral_fit = _fitTimeWindowWrapper(mf,nsteps=nsteps)
                    

    
    # ===============================
    # Fitting for each worker
    for j, binn in enumerate(assigned_bins):
        
        checkpoints_dir ='checkpoints/'
        case_dir = '{:d}_tmin{:.2f}_tmax{:.2f}/'.format(shot,t_min,t_max)
        file_name = 'moments_{:d}_bin{:d}_{:d}.pkl'.format(shot,binn[0],binn[1])
        resfile = checkpoints_dir + case_dir + file_name
        
        # create checkpoint directory if it doesn't exist yet
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
            
        if not os.path.exists(checkpoints_dir+case_dir):
            os.makedirs(checkpoints_dir+case_dir)
            
        try:
            # if fit has already been created, re-load it from checkpoint directory
            with open(resdir,'rb') as f:
                res[j] = pkl.load(f)
            print "Loaded fit moments from ", resfile
                
        except:
            # if fit cannot be loaded, run fitting now:
            if NS:
                print "Fitting bin [%d,%d] ...."%(binn[0],binn[1])
                spectral_fit(shot,t_min,t_max,binn[0],binn[1],verbose,rank)

                # load result in memory
                with open(resfile,'rb') as f:
                    res[j] = pkl.load(f)
            else:
                # emcee
                spectral_fit(binn) #saves result in mf.fits[binn[0]][binn[1]]
                
                # process results to get physical measurements
                if mf.fits[binn[0]][binn[1]].good ==True:
                    chain = mf.fits[binn[0]][binn[1]].samples
                
                    # emcee samples are all equally weighed
                    moments_vals = np.apply_along_axis(mf.fits[binn[0]][binn[1]].lineModel.modelMeasurements,
                                                       axis=1, arr=chain)
                    res[j] = [np.mean(moments_vals, axis=0), np.std(moments_vals, axis=0)]
                    
                else:
                    res[j] =[np.asarray([np.nan, np.nan, np.nan]), np.asarray([np.nan, np.nan, np.nan])]
            
                # save measurement results to checkpoint file
                with open(resfile,'wb') as f:
                    pkl.dump(res[j],f)

            
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
            pkl.dump(gathered_moments, f)

        if resume:
            # eliminate checkpoint directory if this was created
            shutil.rmtree('checkpoints/%d_tmin%f_tmax%f'%(shot,t_min,t_max))
        
        # end time count
        elapsed_time = MPI.Wtime() - t_start
        
        print 'Time to run: ' + str(elapsed_time) + " s"