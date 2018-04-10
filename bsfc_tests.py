# -*- coding: utf-8 -*-
"""
Apply tools in bsfc_main.py to a number of test cases. 

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
import scipy

# %%

# %%
#mf = MomentFitter(lam_bounds=(3.725, 3.747), primary_line=lya1', shot=1120914036, tht=1, brancha=False)
#mf = MomentFitter(lam_bounds=(3.725, 3.742), primary_line='lya1', shot=1121002022, tht=0, brancha=False)
shot=1101014019 #1101014030 #1101014019
print "Analyzing shot ", shot
# t_min=1.17, t_max=1.3
t_min=1.2
t_max=1.4
load=True
mf = bsfc_main.MomentFitter(lam_bounds=(3.172, 3.188), primary_line='w', shot=shot, tht=0, brancha=False)

# tbin=136; chbin=28
# mf.fitSingleBin(tbin=tbin, chbin=chbin, nsteps=1024)
# # # corner.corner(mf.fits[tbin][chbin].sampler.chain[10,:,:], labels=mf.fits[tbin][chbin].lineModel.thetaLabels())
# mf.plotSingleBinFit(tbin=tbin, chbin=chbin)

# plotOverChannels(mf, tbin=126, plot=True)
# signal=inj_brightness(mf, t_min=1.2, t_max=1.4, save=True, refit=True, compare=True)


if load:
	with open('hirex_sig_%d.pkl'%shot,'rb') as f:
		signal=pkl.load(f)
else:
	sig=inj_brightness(mf, t_min=t_min, t_max=t_max, save=True, refit=True, compare=True)
	signal=sig.signal


bsfc_slider.slider_plot(
    np.asarray(range(signal.y_norm.shape[1])),
    signal.t,
    np.expand_dims(signal.y_norm.T,axis=0),
    xlabel=r'channel #',
    ylabel=r'$t$ [s]',
    zlabel=r'$n$ [A.U.]',
    labels=['Signal'],
    plot_sum=False
)