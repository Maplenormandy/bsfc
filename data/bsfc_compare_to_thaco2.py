# -*- coding: utf-8 -*-
"""
Compare brightness results for BSFC vs. THACO. Version 2.

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
shot = 1101014019


with open('../helpers/bsfc_hirex_%d.pkl'%shot, 'rb') as f:
    bsfc = pkl.load(f)

with open('./hirex_sig_%s.pkl'%shot, 'rb') as f:
    thaco = pkl.load(f)

# find time shift
t_start =  bsfc['time'][0]-thaco.t[0]
chord=27
plt.figure()
plt.errorbar(thaco.t, thaco.y_norm[:,chord], thaco.std_y_norm[:,chord],fmt='.', label='THACO')
plt.errorbar(bsfc['time']-t_start, bsfc['hirex_signal'][:,chord], bsfc['hirex_uncertainty'][:,chord], fmt='.', label='BSFC')
plt.xlabel('time [s]')
plt.ylabel('Normalized Signal [A.U.]')
plt.legend()
plt.title('chord #%d'%chord)


# plot relative uncertainties in Hirex chord
plt.figure()
plt.plot(thaco.t, thaco.std_y_norm[:,chord]/thaco.y_norm[:,chord],'.', label='THACO')
plt.plot(bsfc['time']-t_start, bsfc['hirex_uncertainty'][:,chord]/ bsfc['hirex_signal'][:,chord],'.', label='BSFC')
plt.xlabel('time [s]')
plt.ylabel('Rel unc')
plt.legend()
plt.title('chord #%d'%chord)


'''
plt.figure()
plt.plot(sigt[0].t, sigt[0].std_y_norm[:,chord]/sigt[0].y_norm[:,chord],'.', label='THACO')
plt.plot(sigb[0].t, sigb[0].std_y_norm[:,chord]/ sigb[0].y_norm[:,chord],'.', label='BSFC')
plt.xlabel('time [s]')
plt.ylabel('Rel unc')
plt.legend()
plt.title('chord #%d'%chord)

'''
