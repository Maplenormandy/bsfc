# -*- coding: utf-8 -*-
"""
Run a series of spectral fits for a tokamak discharge in a chosen time range.
To run, use  
>> python <SHOT> 
where <SHOT> is the CMOD shot number of interest. If using nested sampling (NS) and if MultiNest was 
installed with MPI, then this automatically defaults to parallelizing over live point evaluations. 
This is NOT a high-throughput parallelization, but it works well. See other options in script.

To use a high-throughput parallelization, i.e. parallelized over all time and spatial bins, run this script
with 
>> mpirun python <SHOT>
i.e. the same as above, but invoking the `mpirun` command. 

To visualize results, after running, use
python  <SHOT> -p
without the 'mpirun' command. If results are stored and found, this will try to plot them. 

@author: sciortino
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import cPickle as pkl
import sys, itertools, os, shutil
import scipy, glob, subprocess, argparse

#import bsfc_main
from bsfc_moment_fitter import *
from helpers import bsfc_clean_moments 
from helpers import bsfc_slider
from helpers import bsfc_cmod_shots
from bsfc_bin_fit import _fitTimeWindowWrapper

# MPI parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("shot", type=int, help="shot number to run analysis on.")

# optional argument to plot
parser.add_argument("-p", "--plot", action='store_true', help="Boolean to indicate whether to try to fetch previous results and plot. Only available if results were previously computed!")

# optional arguments for fit procedure
parser.add_argument("-n", "--chosen_n_hermite", type=int, default=3, help="Number of Hermite polynomial terms to include in each fit. Default is 3. ")
parser.add_argument("-v", "--verbose", action='store_true', help="Enable output verbosity.")
parser.add_argument("-ht","--high_throughput", action='store_true', help='Flag to prevent MultiNest from running with MPI (internally parallelizing over line points)')

# optional arguments for emcee runs
parser.add_argument("--emcee", action="store_true", help="Boolean to indicate whether to use emcee. Default is False, so that MultiNest is used instead.")
parser.add_argument("--nsteps", type=int, help="Number of emcee steps. Only used if emcee is being used.")
args = parser.parse_args()
# ---------------------------------

if args.emcee:
    NS=False
else:
    NS=True

if not args.verbose:
    import warnings
    warnings.filterwarnings("ignore")

if rank == 0:
    print "Analyzing shot ", args.shot
    
    # Start counting time:
    t_start=MPI.Wtime()

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(args.shot)

# ==============

# Always create new object by default when running with MPI
if rank==0:
    mf = MomentFitter(primary_impurity, primary_line, args.shot, tht=tht)
    with open('../bsfc_fits/mf_%d_tmin%f_tmax%f_%sline.pkl'%(args.shot,t_min,t_max,primary_line),'wb') as f:
        pkl.dump(mf,f)
else:
    mf = None
    chains_dir = None
    
if args.plot: 
    # if only 1 core is being used, assume that script is being used for plotting

    with open('../bsfc_fits/moments_%d_tmin%f_tmax%f_%sline.pkl'%(args.shot,t_min,t_max,primary_line),'rb') as f:
        gathered_moments=pkl.load(f)

    # clean up Hirex-Sr signals -- BR_THRESH might need to be adjusted to eliminate outliers
    moments_vals, moments_stds, time_sel = bsfc_clean_moments.clean_moments(
        mf.time, mf.maxChan, t_min,t_max,gathered_moments, BR_THRESH=2e8, BR_STD_THRESH=2e8)

    #from IPython import embed
    #embed()
    # BSFC slider visualization
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='br')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='vel')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='Temp')

    plt.show(block=True)

else:
    # Run MPI job:

    if rank==0 and NS:
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
            
    # broadcast mf object to all cores (this also acts for synchronization)
    mf = comm.bcast(mf, root = 0)
    chains_dir = comm.bcast(chains_dir, root=0)
    
    tidx_min = np.argmin(np.abs(mf.time - t_min))
    tidx_max = np.argmin(np.abs(mf.time - t_max))
    time_sel= mf.time[tidx_min: tidx_max]

    # map range of channels and compute each
    map_args_tpm = list(itertools.product(range(tidx_min, tidx_max), range(mf.maxChan)))
    map_args = [list(a) for a in map_args_tpm]

    # we have comm.size cores available. Total number of jobs to be run:
    njobs = len(map_args)
    
    # Block-cyclic parallelization scheme
    extra_njobs = njobs - njobs//size * size

    if rank < extra_njobs - 1 :   #first "extra_njobs" workers take extra task (NB: Python numbering starts at 0)
        personal_njobs = njobs // size + 1
        assigned_jobs_offset = rank * personal_njobs 
    else:
        personal_njobs = njobs // size
        assigned_jobs_offset = rank * personal_njobs + extra_njobs

    assigned_bins = map_args[assigned_jobs_offset:assigned_jobs_offset+personal_njobs]
    
    # now, fill up result array for each worker:
    res = np.asarray([ None for yy in range(len(assigned_bins)) ])

    # define spectral_fit function based on whether emcee or MultiNest should be run
    if NS:

        def spectral_fit(shot,t_min,t_max,tb,cb,n_hermite,verbose,rank):
    
            # set up directory for MultiNest to run
            basename_inner = os.path.abspath(chains_dir+'/mn_chains_%s/c-.'%rank )
            chains_dir_inner = os.path.dirname(basename_inner)
            
            if not os.path.exists(chains_dir_inner):
                # if chains directory for this worker/rank does not exist, create it
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

            if not args.high_throughput: command = ['mpirun'] + command
        
            # print out command to be run on screen
            print ' '.join(command)
            
            # Run externally:
            out = subprocess.check_output(command, stderr=subprocess.STDOUT)
            if verbose: print out
    else:
        # emcee requires a wrapper for multiprocessing to pickle the function
        spectral_fit = _fitTimeWindowWrapper(mf,nsteps=args.nsteps)
                    

    
    # ===============================
    # Fitting for each worker
    for j, binn in enumerate(assigned_bins):
        
        checkpoints_dir ='checkpoints/'
        case_dir = '{:d}_tmin{:.2f}_tmax{:.2f}/'.format(args.shot,t_min,t_max)
        file_name = 'moments_{:d}_bin{:d}_{:d}_{:s}line.pkl'.format(args.shot,binn[0],binn[1],primary_line)
        resfile = checkpoints_dir + case_dir + file_name
        
        # create checkpoint directory if it doesn't exist yet
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
            
        if not os.path.exists(checkpoints_dir+case_dir):
            os.makedirs(checkpoints_dir+case_dir)

        try:
            # if fit has already been created, re-load it from checkpoint directory
            with open(resfile,'rb') as f:
                res[j] = pkl.load(f)
            print "Loaded fit moments from ", resfile
                
        except:
            # if fit cannot be loaded, run fitting now:
            if NS:
                print "Fitting bin [%d,%d] ...."%(binn[0],binn[1])
                spectral_fit(args.shot,t_min,t_max,binn[0],binn[1],args.chosen_n_hermite,args.verbose,rank)

                # load result in memory
                with open(resfile,'rb') as f:
                    res[j] = pkl.load(f)
                
            else:
                # emcee
                spectral_fit(binn) #saves result in mf.fits[binn[0]][binn[1]]
                
                # process results to get physical measurements
                if mf.fits[binn[0]][binn[1]].good ==True:
                    chain = mf.fits[binn[0]][binn[1]].samples
                
                    # emcee samples are all equally weighed
                    moments_vals = np.apply_along_axis(mf.fits[binn[0]][binn[1]].lineModel.modelMeasurements,
                                                       axis=1, arr=chain)
                    res[j] = [np.mean(moments_vals, axis=0), np.std(moments_vals, axis=0)]
                    
                else:
                    res[j] =[np.asarray([np.nan, np.nan, np.nan]), np.asarray([np.nan, np.nan, np.nan])]
            
            # save measurement results to checkpoint file 
            with open(resfile,'wb') as f:
                pkl.dump(res[j],f)

            
    # ===============================

    # collect results on rank=0 process:
    gathered_res = comm.gather(res, root=0)

    #join all results
    if rank==0:
        gath_res = np.concatenate(gathered_res)
        gathered_moments= np.asarray(gath_res).reshape((tidx_max-tidx_min,mf.maxChan))

        print "*********** Completed fits *************"
    
        # save fits for future use
        with open('../bsfc_fits/moments_%d_tmin%f_tmax%f_%sline.pkl'%(args.shot,t_min,t_max, primary_line),'wb') as f:
            pkl.dump(gathered_moments, f)

        #if args.resume:
        #    # eliminate checkpoint directory if this was created
        #    shutil.rmtree('checkpoints/%d_tmin%f_tmax%f'%(args.shot,t_min,t_max))

        # remove chains directory
        shutil.rmtree(chains_dir)
        
        # end time count
        elapsed_time = MPI.Wtime() - t_start
        
        print 'Time to run: ' + str(elapsed_time) + " s"
        print 'Completed BSFC analysis for shot ', args.shot
