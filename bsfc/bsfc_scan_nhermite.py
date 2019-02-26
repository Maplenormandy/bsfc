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
#import bsfc_main
#import scipy
import sys
import time as time_
import multiprocessing
import os
import shutil

# make it possible to use other packages within the BSFC distribution:
from os import path
#sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))
from helpers import bsfc_cmod_shots
from helpers import bsfc_autocorr

from bsfc_moment_fitter import *

# ===========
# Scan parameters
shot = 1101014019
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
print "Analyzing shot ", shot
    
# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)

if 'BSFC_ROOT' not in os.environ:
    # make sure that correct directory is pointed at
    os.environ["BSFC_ROOT"]='%s/bsfc'%str(os.path.expanduser('~'))

# location of MultiNest chains
basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/c-.' )


# ==================================

for nh in range(nh_min, nh_max+1):
    # Start counting time:
    start_time=time_.time()

    # Create new fitting container for each value of n_hermite
    mf = MomentFitter(primary_impurity, primary_line, shot, tht=0)

    # actual fit. Silence warnings to avoid the RunTimeWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # NB: if NS==True, PT doesn't matter
        mf.fitSingleBin(tbin=tbin, chbin=chbin, nsteps=nsteps,
                        emcee_threads=NTASKS, PT=True, NS=NS, n_hermite=nh)

    # save each fit independently
    with open('../bsfc_fits/mf_%d_%d_tbin%d_chbin_%d_nh_%d_nsteps_%d.pkl'%(shot,nsteps,tbin,chbin,nh,nsteps),'wb') as f:
        pkl.dump(mf, f, protocol=pkl.HIGHEST_PROTOCOL)


    # end time count
    elapsed_time=time_.time()-start_time
    print 'Time to run nh=%d: '%nh + str(elapsed_time) + " s"


print "Concluded scan of Hermite polynomial decomposition complexitydimensionality"
print "Note that only the number of Hermite polynomials of the fitted line was varied, not the one of all lines!"

