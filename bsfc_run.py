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
import sys
import time as time_

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
    primary_impurity = 'Ar'
    primary_line = 'lya1'
    tbin=5; chbin=40
    t_min=0.7; t_max=0.8
elif shot==1120914036:
    primary_impurity = 'Ca'
    primary_line = 'lya1'
    tbin=104; chbin=11
    t_min=1.05; t_max=1.27
elif shot==1101014019:
    primary_impurity = 'Ca'
    primary_line = 'w'
    tbin=128; chbin=11
    t_min=1.24; t_max=1.4
elif shot==1101014029:
    primary_impurity = 'Ca'
    primary_line = 'w'
    tbin=128; chbin=13
    t_min=1.17; t_max=1.3
elif shot==1101014030:
    primary_impurity = 'Ca'
    primary_line = 'w'
    # tbin=128; chbin=11
    tbin=116; chbin=18
    t_min=1.17; t_max=1.3
elif shot==1100305019:
    primary_impurity = 'Ca'
    primary_line = 'w'
    # tbin=128; chbin=11
    tbin=116; chbin=18
    t_min=1.17; t_max=1.3

# try loading result
try:
    with open('./bsfc_fits/mf_%d_%d_option%d_tbin%d_chbin_%d.pkl'%(shot,nsteps,int(str(option%10)[0]),tbin,chbin),'rb') as f:
            mf=pkl.load(f)
    loaded = True
    print "Loaded previous result"
except:
    # if this wasn't run before, initialize the moment fitting class
    mf = bsfc_main.MomentFitter(primary_impurity, primary_line, shot, tht=0)
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

        # if thinning is done, but divide nsteps by ``thin''
        chain=chain.reshape((-1, nsteps, chain.shape[-1]))
        bsfc_autocorr.plot_convergence(chain, dim=1, nsteps=nsteps)



# ==================================
elif option==2:
    if loaded==False:
        mf.fitTimeBin(tbin, parallel=True, nproc=NTASKS, nsteps=nsteps)

    if loaded==True:
        bsfc_main.plotOverChannels(mf, tbin=126, parallel=True)


# ==================================
elif option==3: # get brightness
	tidx_min = np.argmin(np.abs(mf.time - t_min))
	tidx_max = np.argmin(np.abs(mf.time - t_max))
	time_sel= mf.time[tidx_min: tidx_max]

	if not loaded:
		print "*********** Fitting now *************"
		mf.fitTimeWindow(tidx_min=tidx_min, tidx_max=tidx_max, parallel=True, nsteps=nsteps, nproc=NTASKS)
		print "*********** Completed fits *************"

	else:
		br, br_unc, time_sel = bsfc_main.get_brightness(mf, t_min=t_min, t_max=t_max)
		# signal=bsfc_main.get_brightness(mf, t_min=t_min, t_max=t_max)

		# # create slider plot for sequential visualization of results
		# bsfc_slider.slider_plot(
		#     np.asarray(range(signal.y_norm.shape[1])),
		#     signal.t,
		#     np.expand_dims(signal.y_norm.T,axis=0),
		#     np.expand_dims(signal.std_y_norm.T,axis=0),
		#     xlabel=r'channel #',
		#     ylabel=r'$t$ [s]',
		#     zlabel=r'$B$ [A.U.]',
		#     labels=['Brightness'],
		#     plot_sum=False
		# )

		bsfc_slider.slider_plot(
		    np.asarray(range(br.shape[1])),
		    time_sel,
		    np.expand_dims(br.T,axis=0),
		    np.expand_dims(br_unc.T,axis=0),
		    xlabel=r'channel #',
		    ylabel=r'$t$ [s]',
		    zlabel=r'$B$ [eV]',
		    labels=['Brightness'],
		    plot_sum=False
		)

		bsfc_slider.slider_plot(
                    time_sel,
		    np.asarray(range(br.shape[1])),
		    np.expand_dims(br,axis=0),
		    np.expand_dims(br_unc,axis=0),
		    xlabel=r'$t$ [s]',
		    ylabel=r'channel #',
		    zlabel=r'$B$ [eV]',
		    labels=['Brightness'],
		    plot_sum=False
		)

# ==================================
elif option==33: # get rotation
	tidx_min = np.argmin(np.abs(mf.time - t_min))
	tidx_max = np.argmin(np.abs(mf.time - t_max))
	time_sel= mf.time[tidx_min: tidx_max]

	if not loaded:
		print "*********** Fitting now *************"
		mf.fitTimeWindow(tidx_min=tidx_min, tidx_max=tidx_max, parallel=True, nsteps=nsteps, nproc=NTASKS)
		print "*********** Completed fits *************"

	else:
	    rot, rot_unc, time_sel=bsfc_main.get_rotation(mf, t_min=t_min, t_max=t_max)

	    # create slider plot for sequential visualization of results
	    bsfc_slider.slider_plot(
	        np.asarray(range(rot.shape[1])),
	        time_sel,
	        np.expand_dims(rot.T,axis=0),
	        np.expand_dims(rot_unc.T,axis=0),
	        xlabel=r'channel #',
	        ylabel=r'$t$ [s]',
	        zlabel=r'$v$ [km/s]',
	        labels=['rotation velocity'],
	        plot_sum=False
	    )

# ==================================
elif option==333: # get temperature
	tidx_min = np.argmin(np.abs(mf.time - t_min))
	tidx_max = np.argmin(np.abs(mf.time - t_max))
	time_sel = mf.time[tidx_min: tidx_max]

	if not loaded:
		print "*********** Fitting now *************"
		mf.fitTimeWindow(tidx_min=tidx_min, tidx_max=tidx_max, parallel=True, nsteps=nsteps, nproc=NTASKS)
		print "*********** Completed fits *************"

	else:
	    Temp, Temp_unc, time_sel=bsfc_main.get_temperature(mf, t_min=t_min, t_max=t_max)

	    # create slider plot for sequential visualization of results
	    bsfc_slider.slider_plot(
	        np.asarray(range(Temp.shape[1])),
	        time_sel,
	        np.expand_dims(Temp.T,axis=0),
	        np.expand_dims(Temp_unc.T,axis=0),
	        xlabel=r'channel #',
	        ylabel=r'$t$ [s]',
	        zlabel=r'$T_i$ [eV]',
	        labels=['Ion temperature'],
	        plot_sum=False
	    )


# ==================================
elif option==3333: # get temperature
	tidx_min = np.argmin(np.abs(mf.time - t_min))
	tidx_max = np.argmin(np.abs(mf.time - t_max))
	time_sel= mf.time[tidx_min: tidx_max]

	if not loaded:
		print "*********** Fitting now *************"
		mf.fitTimeWindow(tidx_min=tidx_min, tidx_max=tidx_max, parallel=True, nsteps=nsteps, nproc=NTASKS)
		print "*********** Completed fits *************"

	else:
		moments, moments_std, time_sel=bsfc_main.get_meas(mf, t_min=t_min, t_max=t_max)
    	
    	br = moments[:,:,0]; br_std = moments_std[:,:,0]
    	rot = moments[:,:,1]; rot_std = moments_std[:,:,1]
    	Temp = moments[:,:,2]; Temp_std = moments_std[:,:,2]

    	# create slider plot for sequential visualization of results
    	bsfc_slider.slider_plot(
		    np.asarray(range(br.shape[1])),
		    time_sel,
		    np.expand_dims(br.T,axis=0),
		    np.expand_dims(br_std.T,axis=0),
		    xlabel=r'channel #',
		    ylabel=r'$t$ [s]',
		    zlabel=r'$B$ [eV]',
		    labels=['Brightness'],
		    plot_sum=False)

    	bsfc_slider.slider_plot(
		    np.asarray(range(rot.shape[1])),
		    time_sel,
		    np.expand_dims(rot.T,axis=0),
		    np.expand_dims(rot_std.T,axis=0),
		    xlabel=r'channel #',
		    ylabel=r'$t$ [s]',
		    zlabel=r'$v$ [eV]',
		    labels=['Rotation'],
		    plot_sum=False)

    	bsfc_slider.slider_plot(
		    np.asarray(range(Temp.shape[1])),
		    time_sel,
		    np.expand_dims(Temp.T,axis=0),
		    np.expand_dims(Temp_std.T,axis=0),
		    xlabel=r'channel #',
		    ylabel=r'$t$ [s]',
		    zlabel=r'$T_i$ [eV]',
		    labels=['Ion temperature'],
		    plot_sum=False)

# === Show measurements for a single time/channel bin ===
elif option==11:
    if loaded==True:
        chain = mf.fits[tbin][chbin].samples
        moments = np.apply_along_axis(mf.fits[tbin][chbin].lineModel.modelMeasurements, axis=1, arr=chain)
        f, a = plt.subplots(3, 1, figsize=(8,8))
        a[0].hist(moments[:,0], bins=1000)
        a[1].hist(moments[:,1], bins=1000)
        a[2].hist(moments[:,2], bins=1000)
        a[0].set_xlabel('B [A.U.]')
        a[1].set_xlabel('v_ll [km/s]')
        a[2].set_xlabel('Ti [keV]')
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)


        mf.plotSingleBinFit(tbin=tbin, chbin=chbin)


# save fits for future use
with open('./bsfc_fits/mf_%d_%d_option%d_tbin%d_chbin_%d.pkl'%(shot,nsteps,int(str(option%10)[0]),tbin,chbin),'wb') as f:
    pkl.dump(mf, f, protocol=pkl.HIGHEST_PROTOCOL)


# end time count
elapsed_time=time_.time()-start_time
print 'Time to run: ' + str(elapsed_time) + " s"
