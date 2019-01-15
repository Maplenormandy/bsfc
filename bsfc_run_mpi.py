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

# MPI parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# first command line argument gives shot number
shot = int(sys.argv[1])

# second command line argument gives number of MCMC steps 
nsteps = int(sys.argv[2])
    

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
    t_min=1.24; t_max=1.27#1.4
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

if size==1:
    # if only 1 core is being used, assume that script is being used for plotting

    with open('./bsfc_fits/mf_%d_%d_tmin%f_tmax%f.pkl'%(shot,nsteps,t_min,t_max),'rb') as f:
            mf=pkl.load(f)

    tidx_min = np.argmin(np.abs(mf.time - t_min))
    tidx_max = np.argmin(np.abs(mf.time - t_max))
    time_sel= mf.time[tidx_min: tidx_max]

    br, br_unc, time_sel = bsfc_main.get_brightness(mf, t_min=t_min, t_max=t_max)
		
    bsfc_slider.slider_plot(
        np.asarray(range(br.shape[1])),
        time_sel,
        np.expand_dims(br.T,axis=0),
        np.expand_dims(br_unc.T,axis=0),
        xlabel=r'channel #',
        ylabel=r'$t$ [s]',
        zlabel=r'$B$ [eV]',
        labels=['Brightness'],
        plot_sum=False
    )
    
    bsfc_slider.slider_plot(
        time_sel,
        np.asarray(range(br.shape[1])),
        np.expand_dims(br,axis=0),
        np.expand_dims(br_unc,axis=0),
        xlabel=r'$t$ [s]',
        ylabel=r'channel #',
        zlabel=r'$B$ [eV]',
        labels=['Brightness'],
        plot_sum=False
    )

    plt.show(block=True)
else:
    # Run MPI job:

    # Always create new object by default when running with MPI
    if rank==0:
        mf = bsfc_main.MomentFitter(primary_impurity, primary_line, shot, tht=tht)
    else:
        mf = None

    # broadcast r object to all cores
    mf = comm.bcast(mf, root = 0)  
    
    # synchronize processes
    comm.Barrier()
    
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
    
    # number of jobs per worker if number of total jobs is exactly a multiple of the number of workers
    personal_njobs = njobs // size 
    
    # starting index of assigned bins for current worker:
    assigned_jobs_offset = rank * personal_njobs
    
    # bins to be run
    assigned_bins = map_args[assigned_jobs_offset:assigned_jobs_offset+personal_njobs]
    
    # if #total jobs is NOT exactly a multiple of the total number of workers, distribute remaining bins 
    if njobs%size != 0:
        # number of bins still to be assigned:
        extra_njobs = njobs - njobs//size * size
        
        # assign remaining 'n' bins to first workers with rank<n
        if rank < extra_njobs:
            # add one job to first #extra_njobs workers available
            assigned_bins += ([ map_args[assigned_jobs_offset+personal_njobs+rank ] ])
            

    # now, for each worker:
    fits = np.asarray([ None for yy in range(len(assigned_bins)) ])
    comm.Barrier()

    # ===============================
    # Actual evaluation for each worker
    for j, binn in enumerate(assigned_bins):
        fits[j] = ff(binn)

    # ===============================

    comm.Barrier()
    
    # collect results on rank=0 process:
    gathered_fits = comm.gather(fits, root=0)

    #join all results
    if rank==0:
        gath_fits = np.concatenate(gathered_fits)

        mf.fits = np.asarray(gath_fits).reshape((tidx_max-tidx_min,mf.maxChan))
    
        print "*********** Completed fits *************"
    
        # save fits for future use
        with open('./bsfc_fits/mf_%d_%d_tmin%f_tmax%f.pkl'%(shot,nsteps,t_min,t_max),'wb') as f:
            pkl.dump(mf, f, protocol=pkl.HIGHEST_PROTOCOL)


        # end time count
        elapsed_time = MPI.Wtime() - t_start
        
        print 'Time to run: ' + str(elapsed_time) + " s"
