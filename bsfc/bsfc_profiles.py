# -*- coding: utf-8 -*-
"""
Run BSFC scans over wide time and channel windows of Hirex-Sr data. 
Use bsfc_run.py to look into individual bin statistics. 

@author: sciortino
"""

import numpy as np
import cPickle as pkl
import pdb
import sys
import time as time_
import matplotlib.pyplot as plt
plt.ion()

# make it possible to use other packages within the BSFC distribution:
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))

import bsfc_main
from helpers import bsfc_slider
from helpers import bsfc_cmod_shots


# first command line argument gives shot number
shot = int(sys.argv[1])

# second command line argument tells how many processes to run in parallel
NTASKS = int(sys.argv[2])

# third command line argument gives number of MCMC steps
nsteps = int(sys.argv[3])

# Start counting time:
start_time=time_.time()

print "Analyzing shot ", shot

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)


# try loading result
try:
    with open('./bsfc_fits/bsfc_profiles_%d_%d.pkl'%(shot,nsteps),'rb') as f:
            bsfc_profiles=pkl.load(f)
    print "Result already available in ./bsfc_fits/bsfc_profiles_%d_%d.pkl"%(shot,nsteps)
    print 'Run plot_bsfc_profiles(bsfc_profiles) to plot profiles.' 

except:
	print "*********** Fitting *************"
	# if this wasn't run before, initialize the moment fitting class
	mf = bsfc_main.MomentFitter(primary_impurity, primary_line, shot, tht=tht)

	tidx_min = np.argmin(np.abs(mf.time - t_min))
	tidx_max = np.argmin(np.abs(mf.time - t_max))
	time_sel= mf.time[tidx_min: tidx_max]

	
	mf.fitTimeWindow(tidx_min=tidx_min, tidx_max=tidx_max, parallel=True, nsteps=nsteps, nproc=NTASKS)
	print "*********** Completed fits *************"

	print "------->  Computing moments "
	
	# moments, moments_std, time_sel=bsfc_main.get_meas(mf, t_min=t_min, t_max=t_max)
	br, br_unc, time_sel = bsfc_main.get_brightness(mf, t_min=t_min, t_max=t_max)
	rot, rot_unc, time_sel = bsfc_main.get_rotation(mf, t_min=t_min, t_max=t_max)
	Temp, Temp_unc, time_sel = bsfc_main.get_temperature(mf, t_min=t_min, t_max=t_max)

	# pdb.set_trace()
	# br = moments[:,:,0]; br_unc = moments_std[:,:,0]
	# rot = moments[:,:,1]; rot_unc = moments_std[:,:,1]
	# Temp = moments[:,:,2]; Temp_unc = moments_std[:,:,2]

	# save fits for future use
	with open('./bsfc_fits/bsfc_profiles_%d_%d.pkl'%(shot,nsteps),'wb') as f:
	    pkl.dump((time_sel, br, br_unc, rot,rot_unc, Temp, Temp_unc), f, protocol=pkl.HIGHEST_PROTOCOL)

	print "------->  Saved ./bsfc_fits/bsfc_profiles_%d_%d.pkl"%(shot,nsteps)


# end time count
elapsed_time=time_.time()-start_time
print 'Time to run: ' + str(elapsed_time) + " s"


def plot_bsfc_profiles(bsfc_profiles):
	time_sel, br, br_std, rot,rot_std, Temp, Temp_std = bsfc_profiles
	# create slider plot for sequential visualization of results
	bsfc_slider.slider_plot(
	    np.asarray(range(br.shape[1])),
	    time_sel,
	    np.expand_dims(br.T,axis=0),
	    np.expand_dims(br_std.T,axis=0),
	    xlabel=r'channel #',
	    ylabel=r'$t$ [s]',
	    zlabel=r'$B$ [A.U.]',
	    labels=['Brightness'],
	    plot_sum=False)

	bsfc_slider.slider_plot(
	    np.asarray(range(rot.shape[1])),
	    time_sel,
	    np.expand_dims(rot.T,axis=0),
	    np.expand_dims(rot_std.T,axis=0),
	    xlabel=r'channel #',
	    ylabel=r'$t$ [s]',
	    zlabel=r'$v$ [km/s]',
	    labels=['Rotation'],
	    plot_sum=False)

	bsfc_slider.slider_plot(
	    np.asarray(range(Temp.shape[1])),
	    time_sel,
	    np.expand_dims(Temp.T,axis=0),
	    np.expand_dims(Temp_std.T,axis=0),
	    xlabel=r'channel #',
	    ylabel=r'$t$ [s]',
	    zlabel=r'$T_i$ [keV]',
	    labels=['Ion temperature'],
	    plot_sum=False)
