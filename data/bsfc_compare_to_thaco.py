# -*- coding: utf-8 -*-
"""
Compare brightness results for BSFC vs. THACO. 

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

# first command line argument gives shot number
shot = int(sys.argv[1])

# second command line argument gives number of MCMC steps 
nsteps = int(sys.argv[2])

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)

mf = bsfc_main.MomentFitter(primary_impurity, primary_line, shot, tht=tht)

with open('./bsfc_fits/moments_%d_%dsteps_tmin%f_tmax%f.pkl'%(shot,nsteps,t_min,t_max),'rb') as f:
    gathered_moments=pkl.load(f)


# clean up Hirex-Sr signals -- BR_THRESH might need to be adjusted to eliminate outliers
moments_vals, moments_stds, time_sel = clean_moments(mf.time, mf.maxChan, t_min,t_max, gathered_moments, BR_THRESH=2.0, BR_STD_THRESH=0.02)
    
# BSFC slider visualization
bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel-t_min, q='br')

# load THACO results
with open('./hirex_sig_%d.pkl'%shot,'rb') as f:
    hsig = pkl.load(f)

# overplotting on the slider plot breaks the slider at this time...
#a_plots,a_sliders = bsfc_slider.visualize_moments(hsig.y_norm, hsig.std_y_norm, hsig.t, q='br',a_plots=a_plots,a_sliders=a_sliders)


chord=11
plt.figure()
plt.errorbar(hsig.t, hsig.y_norm[:,chord], hsig.std_y_norm[:,chord],fmt='.', label='THACO')
plt.errorbar(time_sel-t_min, moments_vals[:,chord,0], moments_stds[:,chord,0], fmt='.', label='BSFC')
plt.xlabel('time [s]')
plt.ylabel('Normalized Signal [A.U.]')
plt.legend()
