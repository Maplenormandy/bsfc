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

# first command line argument gives shot number
shot = int(sys.argv[1])

# second command line argument gives number of MCMC steps 
nsteps = int(sys.argv[2])

if shot==1121002022:
    t_min=0.7; t_max=0.8
elif shot==1120914036:
    t_min=1.05; t_max=1.27
elif shot==1101014019:
    t_min=1.24; t_max=1.4
elif shot==1101014029:
    t_min=1.17; t_max=1.3
elif shot==1101014030:
    t_min=1.17; t_max=1.3
elif shot==1100305019:
    t_min=0.98; t_max=1.2
else:
    # define more lines!
    raise Exception('Times of interest not set for this shot!')
