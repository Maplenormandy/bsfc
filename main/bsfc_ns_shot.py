# -*- coding: utf-8 -*-
"""
Run a series of spectral fits with MultiNest for a tokamak discharge in a chosen time range.
To run, use  
>> python bsfc_ns_shot.py <SHOT> 
where <SHOT> is the CMOD shot number of interest. This runs MPI internally to MultiNest if there exists 
a MPI installation on the running machine. 


To visualize results, after running, use
python bsfc_ns_shot.py  <SHOT> -p
If results are stored and found, this will try to plot them. 

This is a simpler script than bsfc_run_mpi.py, which offers more options to also run emcee and parallelize in high-throughput fashion. 

@author: sciortino, Nov 2019
"""
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import chr
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import time as _time
import pickle as pkl
import sys, itertools, os, shutil
import scipy, glob, subprocess, argparse

from .bsfc_moment_fitter import *
from helpers import bsfc_clean_moments 
from helpers import bsfc_slider
from helpers import bsfc_cmod_shots
from .bsfc_bin_fit import _fitTimeWindowWrapper
import datetime


# ---------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("shot", type=int, help="shot number to run analysis on.")

# optional argument to plot
parser.add_argument("-p", "--plot", action='store_true', help="Boolean to indicate whether to try to fetch previous results and plot. Only available if results were previously computed!")

# optional arguments for fit procedure
parser.add_argument("-n", "--chosen_n_hermite", type=int, default=3, help="Number of Hermite polynomial terms to include in each fit. Default is 3. ")
parser.add_argument("-v", "--verbose", action='store_true', help="Enable output verbosity.")
parser.add_argument("--no_mpirun", action='store_true', help='Flag to prevent MultiNest from running with MPI (internally parallelizing over line points)')

args = parser.parse_args()
# ---------------------------------

if not args.verbose:
    import warnings
    warnings.filterwarnings("ignore")


print("Analyzing shot ", args.shot)
    
# Start counting time:
t_start=_time.time()

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(args.shot)

# ==============

# Always create new object by default 

mf = MomentFitter(primary_impurity, primary_line, args.shot, tht=tht)
with open('../bsfc_fits/mf_%d_tmin%f_tmax%f_%sline.pkl'%(args.shot,t_min,t_max,primary_line),'wb') as f:
    pkl.dump(mf,f)

    
if args.plot: 

    with open('../bsfc_fits/moments_%d_tmin%f_tmax%f_%sline.pkl'%(args.shot,t_min,t_max,primary_line),'rb') as f:
        gathered_moments=pkl.load(f)

    # clean up Hirex-Sr signals -- BR_THRESH might need to be adjusted to eliminate outliers
    moments_vals, moments_stds, time_sel = bsfc_clean_moments.clean_moments(
        mf.time, mf.maxChan, t_min,t_max,gathered_moments, BR_THRESH=1e4, BR_STD_THRESH=2e8,BR_REL_THRESH=1e8, normalize=False)

    # BSFC slider visualization
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='br')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='vel')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='Temp')

    plt.show(block=True)

else:
    # Run all fit jobs:


    if 'BSFC_ROOT' not in os.environ:
        # make sure that correct directory is pointed at
        os.environ["BSFC_ROOT"]='%s/bsfc'%str(os.path.expanduser('~'))

    # create new chains directory
    chains_dirs = [f for f in os.listdir('..') if f.startswith('mn_chains')]
    nn=0
    while True:
        if 'mn_chains_'+str(chr(nn+97)).upper() in chains_dirs:
            nn+=1
        else:
            chains_dir = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains_'+str(chr(nn+97)).upper())
            break

    if not os.path.exists(chains_dir):
        os.mkdir(chains_dir)
    
    tidx_min = np.argmin(np.abs(mf.time - t_min))
    tidx_max = np.argmin(np.abs(mf.time - t_max))
    time_sel= mf.time[tidx_min: tidx_max]

    # map range of channels and compute each
    map_args_tpm = list(itertools.product(list(range(tidx_min, tidx_max)), list(range(mf.maxChan))))
    map_args = [list(a) for a in map_args_tpm]
    
    # now, fill up result array for each worker:
    res = np.asarray([None for yy in range(len(map_args)) ])
    
    # define spectral_fit function 

    def spectral_fit(shot,t_min,t_max,tb,cb,n_hermite,verbose):
        
        # set up directory for MultiNest to run
        basename_inner = os.path.abspath(chains_dir+'/mn_chains_ns_shot/c-.')
        chains_dir_inner = os.path.dirname(basename_inner)
        
        if not os.path.exists(chains_dir_inner):
            # if chains directory does not exist, create it
            os.mkdir(chains_dir_inner)
        else:
            # if chains directory exists, make sure it's empty
            fileList = glob.glob(basename_inner+'*')
            for filePath in fileList:
                try:
                    os.remove(filePath)
                except:
                    pass
        
        # individual fit with MultiNest
        command = ['python','launch_NSfit.py',str(shot),str(primary_line), str(t_min), str(t_max),
                       str(int(tb)),str(int(cb)),str(int(n_hermite)), str(int(verbose)),basename_inner]
        
        if not args.no_mpirun: command = ['mpirun'] + command
        
        # print out command to be run on screen and time
        print(' '.join(command))
        now = datetime.datetime.now()
        print(now.isoformat())
            
        # Run externally:
        out = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if verbose: print(out)

                    
    # ===============================
    # Fitting
    checkpoints_dir ='checkpoints/'
    case_dir = '{:d}_tmin{:.2f}_tmax{:.2f}/'.format(args.shot,t_min,t_max)
    
    # create checkpoint directory if it doesn't exist yet
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(checkpoints_dir+case_dir):
            os.makedirs(checkpoints_dir+case_dir)  
    
    for j, binn in enumerate(map_args):
        
        file_name = 'moments_{:d}_bin{:d}_{:d}_{:s}line.pkl'.format(args.shot,binn[0],binn[1],primary_line)
        resfile = checkpoints_dir + case_dir + file_name
        
        try:
            # if fit has already been created, re-load it from checkpoint directory
            with open(resfile,'rb') as f:
                res[j] = pkl.load(f)
            print("Loaded fit moments from ", resfile)
                
        except:
            # if fit cannot be loaded, run fitting now:
            print("Fitting bin [%d,%d] ...."%(binn[0],binn[1]))
            spectral_fit(args.shot,t_min,t_max,binn[0],binn[1],args.chosen_n_hermite,args.verbose)
            
            # load result in memory
            with open(resfile,'rb') as f:
                res[j] = pkl.load(f)
                
            # save measurement results to checkpoint file 
            with open(resfile,'wb') as f:
                pkl.dump(res[j],f)

            
    # ===============================

    gathered_moments= np.asarray(res).reshape((tidx_max-tidx_min,mf.maxChan))
    
    print("*********** Completed fits *************")
    
    # save fits for future use
    with open('../bsfc_fits/moments_%d_tmin%f_tmax%f_%sline.pkl'%(args.shot,t_min,t_max, primary_line),'wb') as f:
        pkl.dump(gathered_moments, f)

    #if args.resume:
    #    # eliminate checkpoint directory if this was created
    #    shutil.rmtree('checkpoints/%d_tmin%f_tmax%f'%(args.shot,t_min,t_max))
    
    # remove chains directory
    shutil.rmtree(chains_dir)
        
    # end time count
    elapsed_time = _time.time() - t_start
    
    print('Time to run: ' + str(elapsed_time) + " s")
    print('Completed BSFC analysis for shot ', args.shot)
