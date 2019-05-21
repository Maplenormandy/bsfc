''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This script runs a scan for the number of Hermite polynomials using PTMCMC. 
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import cPickle as pkl
import pdb
import corner
#import scipy
import sys
import time as time_
import multiprocessing
import os
import shutil

from helpers import bsfc_cmod_shots
from helpers import bsfc_autocorr

from bsfc_moment_fitter import *

# Import mpi4py here to output timing only once
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ===========
# Scan parameters
shot = 1160506007 #1101014019 #1160506007
NS=True   #if NS==True, then nsteps is useless. 
nsteps = int(1e5) #make sure this is an integer
nh_min = 3
nh_max = 5
# ============

# Use as many cores as are available (this only works on a single machine/node!)
if NS==False:
    NTASKS = multiprocessing.cpu_count()
    print "Running on ", NTASKS, "cores"
else:
    NTASKS=1

if rank==0:
    print "Analyzing shot ", shot
    
# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)

if 'BSFC_ROOT' not in os.environ:
    # make sure that correct directory is pointed at
    os.environ["BSFC_ROOT"]='%s/bsfc'%str(os.path.expanduser('~'))

# location of MultiNest chains
basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/c-.' )


# ==================================
#nh=3
#nlp = [2**i for i in range(6, 13)] # up to 4096, good to plot on log-2 scale
n_live_points=400

#for n_live_points in nlp:
for nh in range(nh_min, nh_max+1):
    # Start counting time:
    start_time=time_.time()

    # Create new fitting container for each value of n_hermite
    mf = MomentFitter(primary_impurity, primary_line, shot, tht=0)

    if rank==0 and NS:
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
        
    # actual fit. Silence warnings to avoid the RunTimeWarnings
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")

    # NB: if NS==True, PT doesn't matter
    mf.fitSingleBin(tbin=tbin, chbin=chbin, nsteps=nsteps,emcee_threads=NTASKS,
                    PT=True, NS=NS,n_live_points=n_live_points,sampling_efficiency=0.3,
                    verbose=True,const_eff=True, n_hermite=nh)

    if rank==0 and NS:
        mf.fits[tbin][chbin].NS_analysis(basename)
        
    # save each fit independently
    if rank==0:
        if NS:
            nn=n_live_points
        else:
            nn=nsteps
        with open('../bsfc_fits/mf_%d_tbin%d_chbin%d_nh%d_%d.pkl'%(shot,tbin,chbin,nh,nn),'wb') as f:
            pkl.dump(mf, f, protocol=pkl.HIGHEST_PROTOCOL)
            
        del mf
    
        # end time count
        elapsed_time=time_.time()-start_time
        print 'Time to run nh=%d: '%nh + str(elapsed_time) + " s"


