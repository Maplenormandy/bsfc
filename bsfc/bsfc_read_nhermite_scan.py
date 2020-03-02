''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This script reads the results of bsfc_scan_nhermite.py and plots relevant quantities to select
optimal number of Hermite polynomials.
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
try:
    import pickle as pkl # python 3+
except:
    import cPickle as pkl   # python 2.7

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
from bsfc_moment_fitter import MomentFitter
from helpers import bsfc_cmod_shots
from helpers import bsfc_autocorr

import gptools


font = {'family' : 'serif',
        'serif': ['Computer Modern'],
        'size'   : 8}
mpl.rc('font', **font)


# ===========
# Scan parameters
shot = 1160506007 #1101014019 #1160506007  #1101014019
nsteps = int(1e5) #make sure this is an integer
nh_min = 3
nh_max = 9
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
nn = 500
for nh in range(nh_min, nh_max+1):
    #nh=3
    #nlp = [50,100,200,600,1000,1400,1800,2300,3000,5000]
    #nlp = [2**i for i in range(6, 13)]

    #for n_live_points in nlp:
    #nn=n_live_points #nsteps

    # save each fit independently
    print("Loading from ", '../bsfc_fits/mf_%d_tbin%d_chbin%d_jef_nh%d_%d.pkl'%(shot,tbin,chbin,nh,nn))
    with open('../bsfc_fits/mf_%d_tbin%d_chbin%d_jef_nh%d_%d.pkl'%(shot,tbin,chbin,nh,nn),'rb') as f:
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

    print(" ++++++++++++++++++++++++++")
    print("nh = ", nh)
    print("lnev = ", lnev[-1], "+/-", lnev_unc[-1])
    print("br = ", means[0], "+/-", stds[0])
    print("v = ", means[1], "+/-", stds[1])
    print("Ti = ", means[2], "+/-", stds[2])
    br.append(means[0]); br_unc.append(stds[0])
    v.append(means[1]); v_unc.append(stds[1])
    Ti.append(means[2]); Ti_unc.append(stds[2])


#times = [109.133455991745, 127.3736081123352, 157.27472591400146, 174.1919960975647, 183.1180350780487, 237.6917450428009, 291.2915561199188]
#times = [106.88230395317078, 106.53741407394409, 112.50855994224548, 132.0632438659668, 156.06387996673584, 178.10949397087097, 173.91060090065002]
#times = [134.54520916938782, 147.2477958202362, 165.08653497695923, 178.43171787261963, 208.95905590057373, 227.8000979423523, 295.3363480567932]
times = [138.8696210384369, 157.01426005363464, 178.54218196868896, 203.29988193511963, 217.57927799224854, 253.7655689716339, 282.0700430870056]

# Plot lnev scaling

#for nh in range(nh_min, nh_max+1):
#    lnev_arr.append(lnev[nh])
#    lnev_unc_arr.append(lnev_unc[nh])


nh = range(nh_min,nh_max+1)
# %%

f = plt.figure(1, figsize=(3.375, 3.375*1.2))
gs1 = mpl.gridspec.GridSpec(5, 1, hspace=0.0)
ax0 = plt.subplot(gs1[0])
ax = [plt.subplot(gs1[j], sharex=ax0) for j in range(5)]

ax[0].errorbar(nh, br, br_unc, fmt='.')
ax[1].errorbar(nh, v, v_unc, fmt='.')
ax[2].errorbar(nh, Ti, Ti_unc, fmt='.')
ax[3].errorbar(nh, times, fmt='.')
ax[4].errorbar(nh, lnev, lnev_unc, fmt='.')

ax[4].set_xlabel('# Hermite coefficients')
ax[4].set_ylabel('ln $\mathcal{Z}$')
ax[3].set_ylabel('runtime [s]')
ax[2].set_ylabel('temp. [keV]')
ax[1].set_ylabel('vel. [km/s]')
ax[0].set_ylabel('bright. [A.U.]')

ax0.set_xlim([nh_min-0.95, nh_max+0.95])
#plt.grid()

plt.setp(ax[0].get_xticklabels(), visible=False)
plt.setp(ax[1].get_xticklabels(), visible=False)
plt.setp(ax[2].get_xticklabels(), visible=False)
plt.setp(ax[3].get_xticklabels(), visible=False)
#ax[0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))

ax[0].set_ylim([6295,6420])
ax[1].set_ylim([1.1,3.9])
ax[2].set_ylim([1.61,1.89])
ax[3].set_ylim([55, 345])
ax[4].set_ylim([-195, -171])

#ax[0].grid()
#ax[1].grid()
#ax[2].grid()
#ax[3].grid()
#ax[4].grid()

plt.savefig('/home/normandy/Pictures/BSFC/newfigs/figure2.eps')

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
'''

plt.tight_layout()

# %%

nh=3

with open('../bsfc_fits/mf_%d_tbin%d_chbin%d_jef_nh%d_%d.pkl'%(shot,tbin,chbin,nh,nn),'rb') as f:
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

# %%

plt.figure(figsize=(3.375*2.5,3.375*2.5))
#plt.close('all')

font = {'family' : 'serif',
        'serif': ['Times New Roman'],
        'size'   : 15}
#mpl.rc('font', **font)

toplot = np.array([np.log10(samples[:,3+nh]/samples[:,3]), measurements[:,1], measurements[:,2]]).T

#f = gptools.plot_sampler(
#    toplot, # index 0 is weights, index 1 is -2*loglikelihood, then samples
#    weights=sample_weights,
#    labels=['log${}_{10}$ Br. Ratio', 'Velocity [km/s]', 'Temperature [keV]'],
#    chain_alpha=1.0,
#    cutoff_weight=0.01,
#    cmap='plasma',
#    #suptitle='Posterior distribution of $D$ and $V$',
#    plot_samples=False,
#    plot_chains=False,
#    xticklabel_angle=45,
#    #yticklabel_angle=30
#    ticklabel_fontsize=16,
#)
#
#plt.savefig('/home/normandy/Pictures/BSFC/newfigs/figure3.png')
