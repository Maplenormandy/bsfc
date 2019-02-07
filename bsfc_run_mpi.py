# -*- coding: utf-8 -*-
"""
MPI high-throughput parallelization of BSFC fitting. 
Run 
mpirun python <SHOT> <NUMBER OF STEPS> 
where <SHOT> is the CMOD shot number and the second argument is the number of steps
that MCMC analysis should run (burn in and thinning are already hard-coded). 

To visualize results, after the above mpirun, run
python  <SHOT> <NUMBER OF STEPS> 
i.e. the same, without the 'mpirun' command. 

@author: sciortino
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import bsfc_helper
import cPickle as pkl
import bsfc_autocorr
import pdb
import corner
import bsfc_main
import bsfc_slider
import bsfc_autocorr
import scipy
import sys
import time as time_
import pdb
import itertools
import os
import shutil
from bsfc_clean_moments import clean_moments
import bsfc_cmod_shots

# MPI parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# first command line argument gives shot number
shot = int(sys.argv[1])

# second command line argument gives number of MCMC steps 
nsteps = int(sys.argv[2])
    
try:
    # third command-line argument specified whether to attempt checkpointing for every bin fit
    resume=bool(int(sys.argv[3]))
except:
    resume=True

try:
    # fourth argument indicates whether to use quiet mode (requires specifying third argument)
    quiet_mode = bool(int(sys.argv[4]))
except:
    quiet_mode = False

if quiet_mode:
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
    mf = bsfc_main.MomentFitter(primary_impurity, primary_line, shot, tht=tht)
else:
    mf = None

if size==1:
    # if only 1 core is being used, assume that script is being used for plotting

    with open('./bsfc_fits/moments_%d_%dsteps_tmin%f_tmax%f.pkl'%(shot,nsteps,t_min,t_max),'rb') as f:
            gathered_moments=pkl.load(f)

    # clean up Hirex-Sr signals -- BR_THRESH might need to be adjusted to eliminate outliers
    moments_vals, moments_stds, time_sel = clean_moments(mf.time, mf.maxChan, t_min,t_max, gathered_moments, BR_THRESH=2.0, BR_STD_THRESH=0.1)
    
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
    
    # requires a wrapper for multiprocessing to pickle the function
    ff = bsfc_main._fitTimeWindowWrapper(mf,nsteps=nsteps)

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

    # ===============================
    # Actual evaluation for each worker
    for j, binn in enumerate(assigned_bins):

        if resume:
            # create checkpoint directory if it doesn't exist yet
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')

            if not os.path.exists('checkpoints/%d_%dsteps_tmin%f_tmax%f'%(shot,nsteps,t_min,t_max)):
                os.makedirs('checkpoints/%d_%dsteps_tmin%f_tmax%f'%(shot,nsteps,t_min,t_max))

            try:
                # if fit has already been created, re-load it from checkpoint directory
                with open('./checkpoints/%d_%dsteps_tmin%f_tmax%f/moments_%d_%d_bin%d_%d.pkl'%(shot,
                                                                                               nsteps,t_min,t_max,shot,nsteps, binn[0], binn[1])) as f:
                          res[j] = pkl.load(f)
                print "Loaded fit moments from ./checkpoints/%d_%dsteps_tmin%f_tmax%f/moments_%d_%d_bin%d_%d.pkl"%(shot,
                                                                                                                   nsteps,t_min,t_max,shot,nsteps, binn[0], binn[1])

            except:
                # create new fit      
                mf.fits[binn[0]][binn[1]] = ff(binn)
                if mf.fits[binn[0]][binn[1]].good ==True:
                    chain = mf.fits[binn[0]][binn[1]].samples
                    moments_vals = np.apply_along_axis(mf.fits[binn[0]][binn[1]].lineModel.modelMeasurements, axis=1, arr=chain)

                    res[j] = [np.mean(moments_vals, axis=0), np.std(moments_vals, axis=0)]
                else:
                    res[j] =[np.asarray([np.nan, np.nan, np.nan]), np.asarray([np.nan, np.nan, np.nan])]

                with open('./checkpoints/%d_%dsteps_tmin%f_tmax%f/moments_%d_%d_bin%d_%d.pkl'%(shot,
                                                                                               nsteps,t_min,t_max,shot,nsteps, binn[0], binn[1]),'wb') as f:
                    pkl.dump(res[j],f)

        else:
                
            mf.fits[binn[0]][binn[1]] = ff(binn)
            if mf.fits[binn[0]][binn[1]].good ==True:
                chain = mf.fits[binn[0]][binn[1]].samples
                moments_vals = np.apply_along_axis(mf.fits[binn[0]][binn[1]].lineModel.modelMeasurements, axis=1, arr=chain)
           
                res[j] = [np.mean(moments_vals, axis=0), np.std(moments_vals, axis=0)]
            else:
                res[j] = [np.asarray([np.nan, np.nan, np.nan]), np.asarray([np.nan, np.nan, np.nan])]
            
            pdb.set_trace()
    # ===============================

    # collect results on rank=0 process:
    gathered_res = comm.gather(res, root=0)

    #join all results
    if rank==0:
        gath_res = np.concatenate(gathered_res)
        gathered_moments= np.asarray(gath_res).reshape((tidx_max-tidx_min,mf.maxChan))

        print "*********** Completed fits *************"
    
        # save fits for future use
        with open('./bsfc_fits/moments_%d_%dsteps_tmin%f_tmax%f.pkl'%(shot,nsteps,t_min,t_max),'wb') as f:
            pkl.dump(gathered_moments, f, protocol=pkl.HIGHEST_PROTOCOL)

        if resume:
            # eliminate checkpoint directory if this was created
            shutil.rmtree('checkpoints/%d_%dsteps_tmin%f_tmax%f'%(shot,nsteps,t_min,t_max))
        
        # end time count
        elapsed_time = MPI.Wtime() - t_start
        
        print 'Time to run: ' + str(elapsed_time) + " s"
