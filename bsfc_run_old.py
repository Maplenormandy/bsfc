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
import bsfc_autocorr
import scipy
import sys
import time as time_
import pdb

# first command line argument gives shot number
shot = int(sys.argv[1])

# second command line argument tells how many processes to run in parallel
NTASKS = int(sys.argv[2])

# third command line argument specifies the type of run
option = int(sys.argv[3])

# fourth command line argument gives number of MCMC steps 
nsteps = int(sys.argv[4])

# # fifth argument specifies whether to plot (choose not to if running in SLURM)
# try:
#     plot = bool(sys.argv[5])
# except:
#     plot = True

# Start counting time:
start_time=time_.time()

# =====================================
# shot=1101014029
# shot=1121002022 
# shot=1101014019
# shot = 1101014030
# ====================================
print "Analyzing shot ", shot
# load = True

if shot==1121002022:
    primary_line = 'lya1'
    imp = 'Ar'
    tbin=10; chbin=20
    t_min=0.7; t_max=0.8
elif shot==1120914036:
    primary_line = 'lya1'
    imp = 'Ca'
    tbin=126; chbin=11
elif shot==1101014019:
    primary_line = 'w'
    imp = 'Ca'
    tbin=128; chbin=11
    t_min=1.24; t_max=1.4
elif shot==1101014029:
    primary_line = 'w'
    imp = 'Ca'
    tbin=128; chbin=11
    t_min=1.17; t_max=1.3
elif shot==1101014030:
    primary_line = 'w'
    imp = 'Ca'
    # tbin=128; chbin=11
    tbin=116; chbin=18
    t_min=1.17; t_max=1.3
    
# define primary line to be fitted
if primary_line=='lya1':
    if imp == 'Ar':
        lam_bounds=(3.725, 3.742)
        brancha = True
    elif imp =='Ca':
        brancha = False
elif primary_line=='w':
    if imp == 'Ar':

    elif imp == 'Ca':
        lam_bounds=(3.172, 3.188)
        brancha = False

# try loading result
try:
    with open('./bsfc_fits/mf_%d_%d_option%d_tbin%d_chbin_%d.pkl'%(shot,nsteps,option,tbin,chbin),'rb') as f:
            mf=pkl.load(f)
    loaded = True
    print "Loaded previous result"
except:
    # if this wasn't run before, initialize the moment fitting class
    mf = bsfc_main.MomentFitter(lam_bounds, primary_line, shot, tht=0, brancha=brancha)
    loaded = False

# ==================================
if option==1:
    if loaded==False:
        mf.fitSingleBin(tbin=tbin, chbin=chbin, nsteps=nsteps, emcee_threads=4)

    if loaded==True:
        chain = mf.fits[tbin][chbin].samples
        figure = corner.corner(chain, labels=mf.fits[tbin][chbin].lineModel.thetaLabels(),
            quantiles=[0.16,0.5, 0.84], show_titles=True, title_kwargs={'fontsize':12},
            levels=(0.68,))
        
        # pdb.set_trace()
        # figure.set_figheight(10)
        # figure.set_figwidth(10)
        plt.savefig('./cornerplots/cornerplot_%d_%d.png'%(shot,nsteps), bbox_inches='tight')

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

        mf.plotSingleBinFit(tbin=tbin, chbin=chbin)

        chain=chain.reshape((-1, nsteps, chain.shape[-1]))
        bsfc_autocorr.plot_convergence(chain, dim=1, nsteps=nsteps)



# ==================================
elif option==2:
    if loaded==False:
        mf.fitTimeBin(tbin, parallel=True, nproc=NTASKS, nsteps=nsteps)

    if loaded==True:
        bsfc_main.plotOverChannels(mf, tbin=126, parallel=True)


# ==================================
elif option==3:
    signal=bsfc_main.inj_brightness(mf, t_min=t_min, t_max=t_max, refit=~loaded,  
        parallel=True, nsteps=nsteps, nproc=NTASKS, plot=False)
    print "*********** Completed fits *************"
    if loaded:
        # create slider plot for sequential visualization of results
        bsfc_slider.slider_plot(
            np.asarray(range(signal.y_norm.shape[1])),
            signal.t,
            np.expand_dims(signal.y_norm.T,axis=0),
            np.expand_dims(signal.std_y_norm.T,axis=0),
            xlabel=r'channel #',
            ylabel=r'$t$ [s]',
            zlabel=r'$n$ [A.U.]',
            labels=['Signal'],
            plot_sum=False
        )

# save fits for future use
with open('./bsfc_fits/mf_%d_%d_option%d_tbin%d_chbin_%d.pkl'%(shot,nsteps,option,tbin,chbin),'wb') as f:
    pkl.dump(mf, f, protocol=pkl.HIGHEST_PROTOCOL)

# end time count
elapsed_time=time_.time()-start_time
print 'Time to run: ' + str(elapsed_time) + " s"