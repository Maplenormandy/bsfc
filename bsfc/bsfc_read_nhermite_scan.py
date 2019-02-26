''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This script reads the results of bsfc_scan_nhermite.py and plots relevant quantities to select
optimal number of Hermite polynomials. 
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
nsteps = int(1e5) #make sure this is an integer
nh_min = 3
nh_max = 3
# ============

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)

# ==================================

lnev = {} #np.zeros(nh_max-nh_min)
lnev_unc = {} #np.zeros_like(lnev)

for nh in range(nh_min, nh_max+1):
    
    # save each fit independently
    with open('../bsfc_fits/mf_%d_%d_tbin%d_chbin_%d_nh_%d_nsteps_%d.pkl'%(shot,nsteps,tbin,chbin,nh,nsteps),'rb') as f:
        mf = pkl.load(f)
        
    #mf.plotSingleBinFit(tbin=tbin, chbin=chbin)  

    lnev[nh], lnev_unc[nh] = mf.fits[tbin][chbin].lnev
    
    #theta_avg = mf.fits[tbin][chbin].theta_avg
    #br, Ti, v = mf.fits[tbin][chbin].lineModel.modelMoments(theta_avg)
    
    chain = mf.fits[tbin][chbin].samples
    moments_vals = np.apply_along_axis(mf.fits[tbin][chbin].lineModel.modelMeasurements, axis=1, arr=chain)
    means, stds = np.mean(moments_vals, axis=0), np.std(moments_vals, axis=0)

    print " ++++++++++++++++++++++++++"
    print "nh = ", nh
    print "lnev = ", lnev[nh], "+/-", lnev_unc[nh]
    print "br = ", means[0], "+/-", stds[0]
    print "v = ", means[1], "+/-", stds[1]
    print "Ti = ", means[2], "+/-", stds[2]


# Plot lnev scaling
lnev_arr=[]
lnev_unc_arr=[]
for nh in range(nh_min, nh_max+1):
    lnev_arr.append(lnev[nh])
    lnev_unc_arr.append(lnev_unc[nh])
    
    
plt.figure()
plt.errorbar(range(nh_min,nh_max+1), lnev_arr, lnev_unc_arr, fmt='.')
plt.xlabel('# Hermite coefficients')
plt.ylabel('ln(ev)')
plt.xlim([nh_min-1, nh_max+1])
