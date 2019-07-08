# -*- coding: utf-8 -*-
"""
Tests bsfc_main versus synthetic data, to see the quality of the fits

@author: normandy
"""

import readline
import MDSplus

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import sys
sys.path.append('/home/normandy/git/bsfc/bsfc/')

import bsfc_main
reload(bsfc_main)


# Start counting time:
#start_time=time_.time()

# %%

# Do a quick hack and load Ca lya1 lines
#mf = bsfc_main.MomentFitter('Ca', 'lya1', 1120914036, tht=0)
mf = bsfc_main.MomentFitter('Ar', 'w', 1160506007, tht=0)

tbin = 46; chbin = 27



mf.fitSingleBin(tbin=tbin, chbin=chbin, nsteps=10)
mf.plotSingleBinFit(tbin, chbin)