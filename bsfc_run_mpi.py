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

# second command line argument tells how many processes to run in parallel
NTASKS = int(sys.argv[2])

# fourth command line argument gives number of MCMC steps 
nsteps = int(sys.argv[3])

# Start counting time:
t_start=MPI.Wtime()

# =====================================
# shot=1101014029
# shot=1121002022 
# shot=1101014019
# shot = 1101014030
# ====================================
if rank == 0:
    print "Analyzing shot ", shot
# load = True

if shot==1121002022:
    primary_line = 'lya1'
    tbin=10; chbin=20
    t_min=0.7; t_max=0.8
elif shot==1120914036:
    primary_line = 'lya1'
    tbin=126; chbin=11
elif shot==1101014019:
    primary_line = 'w'
    tbin=128; chbin=11
    t_min=1.24; t_max=1.4
elif shot==1101014029:
    primary_line = 'w'
    tbin=128; chbin=11
    t_min=1.17; t_max=1.3
elif shot==1101014030:
    primary_line = 'w'
    # tbin=128; chbin=11
    tbin=116; chbin=18
    t_min=1.17; t_max=1.3
    
# define primary line to be fitted
if primary_line=='lya1':
    lam_bounds=(3.725, 3.742)
    brancha = True
elif primary_line=='w':
    lam_bounds=(3.172, 3.188)
    brancha = False

# try loading result
if comm.rank == 0:
    try:
        with open('./bsfc_fits/mf_%d_%d_option%d_tbin%d_chbin_%d.pkl'%(shot,nsteps,option,tbin,chbin),'rb') as f:
                mf=pkl.load(f)
        loaded = True
        if rank == 0:
            print "Loaded previous result"
    except:
        # if this wasn't run before, initialize the moment fitting class
        mf = bsfc_main.MomentFitter(lam_bounds, primary_line, shot, tht=0, brancha=brancha)
        loaded = False
else:
    mf = None


# synchronize processes
comm.Barrier()

# broadcast r object to all cores
mf = comm.bcast(mf, root = 0)  

tidx_min = np.argmin(np.abs(mf.time - t_min))
tidx_max = np.argmin(np.abs(mf.time - t_max))
time_sel= mf.time[tidx_min: tidx_max]


# requires a wrapper for multiprocessing to pickle the function
ff = bsfc_main._fitTimeWindowWrapper(mf,nsteps=nsteps)

# map range of channels and compute each
map_args_tpm = list(itertools.product(range(tidx_min, tidx_max), range(self.maxChan)))
map_args = [list(a) for a in map_args_tpm]

# parallel run

# we have NTASKS cores available, so do comm.size at a time
# total number of jobs to be run:
njobs = len(map_args)
personal_njobs = njobs // comm.size

# make sure njobs is an integer multiple of comm.size
size = comm.size * personal_njobs

# indices of assigned jobs
assigned_jobs_offset = comm.rank * personal_njobs

# bins to be run
assigned_bins = map_args[assigned_jobs_offset:assigned_jobs_offse+personal_njobs]

# create empty structure
fits = np.zeros((tidx_max-tidx_min)*mf.maxChan)

comm.Barrier()

for worker in range(comm.size):
    my_fits = np.zeros(len(assigned_bins))

    for j, bins in enumerate(assigned_bins):
        my_fits[j] = ff(bins)

    comm.Allgather([my_fits, MPI.DOUBLE],
                    [fits, MPI.DOUBLE])

comm.Barrier()

mf.fits = np.asarray(fits).reshape((tidx_max-tidx_min,self.maxChan))

if comm.rank ==0:
    print "*********** Completed fits *************"

    # save fits for future use
    with open('./bsfc_fits/mf_%d_%d_option%d_tbin%d_chbin_%d.pkl'%(shot,nsteps,option,tbin,chbin),'wb') as f:
        pkl.dump(mf, f, protocol=pkl.HIGHEST_PROTOCOL)

# end time count
elapsed_time = WPI.Wtime() - t_start

if comm.rank ==0:
    print 'Time to run: ' + str(elapsed_time) + " s"