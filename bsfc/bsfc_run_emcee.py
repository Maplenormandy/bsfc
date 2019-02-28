''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This script launches BSFC to run with emcee for parameter estimation. Use bsfc_run_ns.py to use 
nested sampling (MultiNest) for model comparison. 

'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import cPickle as pkl
import pdb
import corner
import bsfc_main
#import scipy
import sys
import time as time_
import multiprocessing
import os
import shutil

# make it possible to use other packages within the BSFC distribution:
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))
from helpers import bsfc_cmod_shots
from helpers import bsfc_autocorr

from bsfc_moment_fitter import *

# first command line argument gives shot number
shot = int(sys.argv[1])

# second command line argument gives number of MCMC steps
nsteps = int(sys.argv[2])

try: #parallel tempering
    PT = bool(int(sys.argv[3]))
except:
    PT = False

try: # possibly give number of Hermite polynomials via command line
    n_hermite = int(sys.argv[4])
except:
    n_hermite = 3 #default

# Use as many cores as are available (this only works on a single machine/node!)
NTASKS = multiprocessing.cpu_count()
print "Running on ", NTASKS, "cores"
print "Analyzing shot ", shot

# Start counting time:
start_time=time_.time()

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)

# try loading result
try:
    with open('../bsfc_fits/mf_%d_%d_tbin%d_chbin_%d.pkl'%(shot,nsteps,tbin,chbin),'rb') as f:
        mf=pkl.load(f)
  
    loaded = True
    print "Loaded previous result"    
except:
    # if this wasn't run before, initialize the moment fitting class
    mf = MomentFitter(primary_impurity, primary_line, shot, tht=0)
    loaded = False

# ==================================

if loaded==False:
    mf.fitSingleBin(tbin=tbin, chbin=chbin, nsteps=nsteps,
                    emcee_threads=NTASKS, PT=PT, NS=0, n_hermite=n_hermite)

if loaded==True:
    # the following will be empty at this stage for MultiNest
    chain = mf.fits[tbin][chbin].samples
    
    if chain==None or nsteps==1: 
        plot_posterior=False
    else: 
        plot_posterior=True

    if plot_posterior:
        figure = corner.corner(chain, labels=mf.fits[tbin][chbin].lineModel.thetaLabels(),
                               quantiles=[0.16,0.5, 0.84], show_titles=True, title_kwargs={'fontsize':12},
                               levels=(0.68,))

        plt.savefig('../cornerplots/cornerplot_%d_%d.png'%(shot,nsteps), bbox_inches='tight')

        # extract axes
        ndim = chain.shape[-1]
        axes = np.array(figure.axes).reshape((ndim,ndim))
        
        # plot empirical means on corner plot
        mean_emp = np.mean(chain, axis=0)
        for i in range(ndim):
            ax= axes[i,i]
            ax.axvline(mean_emp[i], color='r')
            
        # add empirical mean to histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi,xi]
                ax.axvline(mean_emp[xi],color='r')
                ax.axhline(mean_emp[yi], color='r')
                ax.plot(mean_emp[xi],mean_emp[yi],'sr')
                
    if chain is not None:
        mf.plotSingleBinFit(tbin=tbin, chbin=chbin)
            
        # if thinning is done, divide nsteps by ``thin''
        chain=chain.reshape((-1, nsteps, chain.shape[-1]))
        if nsteps > 1:
            bsfc_autocorr.plot_convergence(chain, dim=1, nsteps=nsteps)
        
        #plt.show(block=True)
    else:
        print " ********* "
        print "No result to plot"
        print " ********* "


# save fits for future use
with open('../bsfc_fits/mf_%d_%d_tbin%d_chbin_%d.pkl'%(shot,nsteps,tbin,chbin),'wb') as f:
    pkl.dump(mf, f, protocol=pkl.HIGHEST_PROTOCOL)


# end time count
elapsed_time=time_.time()-start_time
print 'Time to run: ' + str(elapsed_time) + " s"
