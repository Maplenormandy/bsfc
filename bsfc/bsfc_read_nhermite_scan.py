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

font = {'family' : 'serif',
        'serif': ['Computer Modern'],
        'size'   : 9}
import matplotlib as mpl
mpl.rc('font', **font)


# ===========
# Scan parameters
shot = 1160506007 #1101014019  #1101014019
nsteps = int(1e5) #make sure this is an integer
nh_min = 3
nh_max = 5
# ============

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)


if 'BSFC_ROOT' not in os.environ:
    # make sure that correct directory is pointed at
    os.environ["BSFC_ROOT"]='%s/bsfc'%str(os.path.expanduser('~'))

# location of MultiNest chains
basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/c-.' )

# ==================================

br = []; br_unc = []
v = []; v_unc = []
Ti = []; Ti_unc = []
lnev=[]; lnev_unc =[]
lnev_vns=[]; lnev_vns_unc=[] # from vanilla NS

nn = 400
for nh in range(nh_min, nh_max+1):
    #nh=3
    #nlp = [50,100,200,600,1000,1400,1800,2300,3000,5000]
    #nlp = [2**i for i in range(6, 13)]
    
    #for n_live_points in nlp:
    #nn=n_live_points #nsteps
         
    # save each fit independently
    print "Loading from ", '../bsfc_fits/mf_%d_tbin%d_chbin%d_nh%d_%d.pkl'%(shot,tbin,chbin,nh,nn)
    with open('../bsfc_fits/mf_%d_tbin%d_chbin%d_nh%d_%d.pkl'%(shot,tbin,chbin,nh,nn),'rb') as f:
        mf = pkl.load(f)
    
    if mf.NS==False:
        chain = mf.fits[tbin][chbin].samples
        
        moments_vals = np.apply_along_axis(mf.fits[tbin][chbin].lineModel.modelMeasurements, axis=1, arr=chain)
        means, stds = np.mean(moments_vals, axis=0), np.std(moments_vals, axis=0)
    else:
        # load MultiNest output        
        samples = mf.fits[tbin][chbin].samples
        sample_weights = mf.fits[tbin][chbin].sample_weights
        
        measurements = np.apply_along_axis(mf.fits[tbin][chbin].lineModel.modelMeasurements, 1, samples)
        means = np.average(measurements, 0, weights=sample_weights)
        stds = np.sqrt(np.average((measurements-means)**2, 0, weights=sample_weights))

    lnev_, lnev_unc_ = mf.fits[tbin][chbin].lnev
    lnev.append(lnev_); lnev_unc.append(lnev_unc_)
    lnev_vns.append(mf.fits[tbin][chbin].multinest_stats['nested sampling global log-evidence'])
    lnev_vns_unc.append(mf.fits[tbin][chbin].multinest_stats['nested sampling global log-evidence error'])
    
    print " ++++++++++++++++++++++++++"
    print "nh = ", nh
    print "lnev = ", lnev[-1], "+/-", lnev_unc[-1]
    print "br = ", means[0], "+/-", stds[0]
    print "v = ", means[1], "+/-", stds[1]
    print "Ti = ", means[2], "+/-", stds[2]
    br.append(means[0]); br_unc.append(stds[0])
    v.append(means[1]); v_unc.append(stds[1])
    Ti.append(means[2]); Ti_unc.append(stds[2])
    

# Plot lnev scaling

#for nh in range(nh_min, nh_max+1):
#    lnev_arr.append(lnev[nh])
#    lnev_unc_arr.append(lnev_unc[nh])
    
'''
plt.figure()
plt.errorbar(range(nh_min,nh_max+1), lnev, lnev_unc, fmt='.')
plt.xlabel('# Hermite coefficients', fontsize=14)
plt.ylabel('ln(ev)', fontsize=14)
plt.xlim([nh_min-1, nh_max+1])
plt.grid()
'''

aspectRatio=1.1
f, axs = plt.subplots(2, 2, sharex=True, figsize=(3.375, 3.375*aspectRatio))

ax1=axs[0,0]; ax2=axs[0,1]; ax3=axs[1,0]; ax4=axs[1,1]
ax1.errorbar(nlp, lnev, lnev_unc, fmt='.')
ax1.set_ylabel(r'$\mathcal{Z}$', fontsize=14)
ax1.grid()

ax2.errorbar(nlp, br, br_unc, fmt='.')
ax2.set_ylabel('Brightness', fontsize=14)
ax2.grid()

ax3.errorbar(nlp, v, v_unc, fmt='.')
ax3.set_xlabel('# live points', fontsize=14)
ax3.set_ylabel(r'v [km/s])', fontsize=14)
ax3.grid()

ax4.errorbar(nlp, Ti, Ti_unc, fmt='.')
ax4.set_xlabel('# live points', fontsize=14)
ax4.set_ylabel(r'$T_i$ [keV]', fontsize=14)
ax4.grid()

ax4.set_xscale('log', basex=2)

f.tight_layout()
