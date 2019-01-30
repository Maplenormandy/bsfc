# -*- coding: utf-8 -*-
"""
Apply tools in bsfc_main.py to a number of test cases. 

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
    resume=False

# =====================================
# shot=1101014029
# shot=1121002022 
# shot=1101014019
# shot = 1101014030
# ====================================
if rank == 0:
    print "Analyzing shot ", shot
    
    # Start counting time:
    t_start=MPI.Wtime()


if shot==1121002022:
    primary_impurity = 'Ar'
    primary_line = 'lya1'
    tbin=5; chbin=40
    t_min=0.7; t_max=0.8
    tht=0
elif shot==1120914036:
    primary_impurity = 'Ca'
    primary_line = 'lya1'
    tbin=104; chbin=11
    t_min=1.05; t_max=1.27
    tht=5
elif shot==1101014019:
    primary_impurity = 'Ca'
    primary_line = 'w'
    tbin=128; chbin=11
    t_min=1.24; t_max=1.4
    tht=0
elif shot==1101014029:
    primary_impurity = 'Ca'
    primary_line = 'w'
    tbin=128; chbin=11
    t_min=1.17; t_max=1.3
    tht=0
elif shot==1101014030:
    primary_impurity = 'Ca'
    primary_line = 'w'
    # tbin=128; chbin=11
    tbin=116; chbin=18
    t_min=1.17; t_max=1.3
    tht=0
elif shot==1100305019:
    primary_impurity = 'Ca'
    primary_line = 'w'
    # tbin=128; chbin=11
    tbin=116; chbin=18
    t_min=0.98; t_max=1.2
    tht=9

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

    tidx_min = np.argmin(np.abs(mf.time - t_min))
    tidx_max = np.argmin(np.abs(mf.time - t_max))
    time_sel= mf.time[tidx_min: tidx_max]

    # get individual spectral moments 
    moments_vals = np.empty((tidx_max-tidx_min,mf.maxChan,3))
    moments_stds = np.empty((tidx_max-tidx_min,mf.maxChan,3))
    moments_vals[:] = None
    moments_stds[:] = None

    for tbin in range(tidx_max-tidx_min):
        for chbin in range(mf.maxChan):
            moments_vals[tbin,chbin,0] = gathered_moments[tbin,chbin][0][0]
            moments_stds[tbin,chbin,0] = gathered_moments[tbin,chbin][1][0]
            moments_vals[tbin,chbin,1] = gathered_moments[tbin,chbin][0][1]
            moments_stds[tbin,chbin,1] = gathered_moments[tbin,chbin][1][1]
            moments_vals[tbin,chbin,2] = gathered_moments[tbin,chbin][0][2]
            moments_stds[tbin,chbin,2] = gathered_moments[tbin,chbin][1][2]
            
    # exclude values with brightness greater than a certain value
    BR_THRESH = 10.0
    moments_vals[:,:,0][moments_vals[:,:,0] > BR_THRESH] = np.nan
    moments_stds[:,:,0][moments_vals[:,:,0] > BR_THRESH] = np.nan
    moments_vals[:,:,1][moments_vals[:,:,0] > BR_THRESH] = np.nan
    moments_stds[:,:,1][moments_vals[:,:,0] > BR_THRESH] = np.nan
    moments_vals[:,:,2][moments_vals[:,:,0] > BR_THRESH] = np.nan
    moments_stds[:,:,2][moments_vals[:,:,0] > BR_THRESH] = np.nan

    # normalize brightness to largest value
    idx1,idx2 = np.unravel_index(np.nanargmax(moments_vals[:,:,0]), moments_vals[:,:,0].shape)
    max_br = moments_vals[idx1,idx2,0]
    max_br_std = moments_stds[idx1,idx2,0]

    moments_vals[:, :,0] = moments_vals[:,:,0]/ max_br
    moments_stds[:,:,0] = scipy.sqrt((moments_stds[:,:,0] / max_br)**2.0 + ((moments_vals[:,:,0] / max_br)*(max_br_std / max_br))**2.0)
    
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='br')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='vel')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='Temp')

    plt.show(block=False)
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
