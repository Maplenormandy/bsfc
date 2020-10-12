# -*- coding: utf-8 -*-
"""
Run nested sampling for model selection.

This script can either be run on 1 CPU using
python bsfc_run_ns.py <SHOT>

or with MPI, using
mpirun python bsfc_run_ns.py <SHOT>
whereby the maximum number of workers will be automatically identified.

After completion of a MultiNest execution, running again this script (without mpirun!) will pull up some useful plots.

@author: sciortino
"""
#from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import pickle as pkl   
import pdb
#import corner
import sys
import time as time_
import multiprocessing
import os
import shutil
import scipy
from helpers import bsfc_cmod_shots
from helpers import bsfc_autocorr

import matplotlib as mpl
from IPython import embed

mpl.rcParams['axes.labelsize']=20
mpl.rcParams['legend.fontsize']=20 #16
mpl.rcParams['xtick.labelsize']=18 #14
mpl.rcParams['ytick.labelsize']=18 #14

import argparse
#from .bsfc_moment_fitter import *
from bsfc_moment_fitter import *

n_hermite=3

parser = argparse.ArgumentParser()
parser.add_argument("shot", type=int,
                    help="shot number to run analysis on")
parser.add_argument("method", type=int, default=2,
                    help="Sampling method to use, among {0: emcee ES, 1: emcee PT, 2: MultiNest, 3: dyPolyChord}")
parser.add_argument("-l", "--line_name",
                    help="name of atomic line of interest for post-fitting analysis.Leave empty for primary line.")
parser.add_argument('-f', "--force", action="store_true",
                    help="whether or not to force an overwrite of saved data")
parser.add_argument('-d', "--dynamic", action="store_true",
                    help="If using method=3, this enables dynamic allocation of PolyChord live points.")
parser.add_argument('-ins', "--INS", action="store_true", default=False,
                    help="If using MultiNest, activate usage of Importance Nested Sampling.")
parser.add_argument('-ce', "--const_eff", action="store_true", default=False,
                    help="If using MultiNest, activate 'constant efficiency' mode.")
parser.add_argument('-cp', "--corner_plot", action="store_true", default=False,
                    help="When plotting, get posterior corner plot.")

args = parser.parse_args()

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#rank = 0

if rank==0:
    methods_dict = {0: 'emcee ES', 1: 'emcee PT', 2: 'MultiNest', 3: 'PolyChord'}
    print('C-Mod shot ', str(args.shot))
    print('-------------')
    print('BSFC options: ')
    print('Method: ', methods_dict[args.method])
    if args.INS: print('Using Importance Nested Sampling')
    if args.method==2 and args.const_eff: print('Using constant efficiency')
    if args.method==3 and args.dynamic: print('Using dynamic live points allocation')
    if args.force: print('Forcing creation of new chains directories')
    if args.line_name is not None: print('Line selection: ', str(args.line_name))
    print('-------------')
    
# Start counting time:
start_time=time_.time()

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(args.shot)

if 'BSFC_ROOT' not in os.environ:
    # make sure that correct directory is pointed at
    os.environ["BSFC_ROOT"]='%s/bsfc'%str(os.path.expanduser('~'))

if args.method==2:
    # location of MultiNest chains
    basename = os.path.abspath(os.environ['BSFC_ROOT']+'/main/mn_chains/bsfc-' )
elif args.method==3:
    if args.dynamic:
        basename = os.path.abspath(os.environ['BSFC_ROOT']+'/dypc_chains/bsfc-' )
    else:
        basename = os.path.abspath(os.environ['BSFC_ROOT']+'/pc_chains/bsfc-' )
        
chains_dir = os.path.dirname(basename)
        
if rank==0:
    # try loading result
    if args.force:
        # Force us to not load the pickle
        loaded = False
        # Remove all the chains
        import glob
        fileList = glob.glob(basename+'*')
        for filePath in fileList:
            try:
                os.remove(filePath)
            except:
                pass
    else:
        try:
            with open('../bsfc_fits/mf_NS_%d_tbin%d_chbin_%d.pkl'%(args.shot,tbin,chbin),'rb') as f:
                mf=pkl.load(f)
            loaded = True; print("Loaded previous result")
        except:
            loaded = False

    if not os.path.exists(chains_dir):  # clear directory
        os.mkdir(chains_dir)

    if not loaded:
        # if this wasn't run before, initialize the moment fitting class
        mf = MomentFitter(primary_impurity, primary_line, args.shot, tht=tht,
                          nofit=['lyas1', 'lyas2', 'lyas3','m','s','t'])

else:
    mf=None
    loaded=None

# broadcast mf object to all cores
mf = comm.bcast(mf, root = 0)
loaded = comm.bcast(loaded, root = 0)

# ==================================

if loaded==False:
    
    # Do a single spectral fit with MultiNest (nested sampling)
    mf.fitSingleBin(tbin=tbin, chbin=chbin, method=args.method,n_live_points='auto',
                    INS=True if args.INS else False,
                    const_eff=True if args.const_eff else False,
                    sampling_efficiency=0.05 if args.const_eff else 0.3,
                    verbose=True,
                    dynamic=args.dynamic,  # used to choose between vanilla PolyChord and dyPolychord
                    n_hermite=n_hermite,
                    basename=basename)

    # save fits for future use
    if rank==0:
        with open('../bsfc_fits/mf_NS_%d_tbin%d_chbin_%d.pkl'%(args.shot,tbin,chbin),'wb') as f:
            pkl.dump(mf, f)

if loaded==True:
    # DO NOT try to load and plot with multiple workers (i.e. using mpirun)!
    bf = mf.fits[tbin][chbin]
    
    if args.method==2:
        # load MultiNest output
        bf.MN_analysis(basename)
    elif args.method==3:
        # load PolyChord output
        logx,nlive,w_rel = bf.PC_analysis('bsfc', dynamic=args.dynamic, plot=True)
        
    if args.corner_plot:
        # corner plot of main posterior cluster
        f = gptools.plot_sampler(
            bf.samples,
            weights=bf.sample_weights,
            labels=bf.lineModel.thetaLabels(),
            chain_alpha=1.0,
            cutoff_weight=0.01,
            cmap='plasma',
            plot_samples=False,
            plot_chains=False,
        )

        
    mf.plotSingleBinFit(tbin=tbin, chbin=chbin) #,plot_clusters=True)

    # ---------
    # allow user to get info about lines other than the primary line
    if args.line_name is None:
        # if user didn't request a specific line, assume that primary line is of interest
        args.line_name = mf.primary_line

    if args.line_name=='all':
        args.line_name='w'
        
    try:
        line_id = np.where(bf.lineModel.linesnames==args.line_name)[0][0]
    except:
        raise ValueError('Requested line cannot be found in nested sampling output!')
    
    modelMeas = lambda x: bf.lineModel.modelMeasurements(x, line=line_id)
    
    # ---------

    # samples and weights from highest-posterior cluster
    samples = bf.samples
    weights = bf.sample_weights
    if hasattr(bf, 'clusters') and 'lnev' in bf.clusters:
        lnevs = np.array(bf.clusters['lnev'])[:,0]
        
        #Bayes factors with respect to highest-posterior cluster
        BFs = np.exp(lnevs - bf.lnev[0])
        print('Bayes Factors for model averaging: ')
        print(BFs)
        
        # Model-average other clusters according to BFs
        for n in np.arange(len(bf.clusters['posterior'])):
            samples = np.concatenate([samples, bf.clusters['posterior'][n].samples])
            weights = np.concatenate([weights, bf.clusters['posterior'][n].weights*BFs[n]])

    measurements = np.apply_along_axis(modelMeas, 1, samples)
    moms = np.average(measurements, 0, weights=weights)
    moms_std = np.sqrt(np.average((measurements-moms)**2, 0, weights=weights))

    print("Counts = ", moms[0], "+/-", moms_std[0])
    print("v = ", moms[1], "+/-", moms_std[1])
    print("Ti = ", moms[2], "+/-", moms_std[2])
    print("ln(ev) = ", bf.lnev[0], "+/-", bf.lnev[1])
    print("# Hermite polynomials: ", n_hermite)

    # Find variation in measurements from various clusters
    if hasattr(bf, 'clusters') and 'lnev' in bf.clusters:
        moms_c = []
        moms_std_c = []
        
        for n in np.arange(len(bf.clusters['posterior'])):
            samp = bf.clusters['posterior'][n].samples
            weig = bf.clusters['posterior'][n].weights
            
            meas = np.apply_along_axis(modelMeas, 1, samp)
            moms_c.append( np.average(meas, 0, weights=weig) )
            moms_std_c.append( np.sqrt(np.average((meas-moms_c[-1])**2, 0, weights=weig)) )

        moms_c = np.array(moms_c)
        moms_std_c = np.array(moms_std_c)
        
        fig,ax = plt.subplots(3,1,sharex=True)
        ax[0].errorbar(np.arange(moms_c.shape[0]),moms_c[:,0], moms_std_c[:,0],fmt='.')
        ax[1].errorbar(np.arange(moms_c.shape[0]),moms_c[:,1], moms_std_c[:,1],fmt='.')
        ax[2].errorbar(np.arange(moms_c.shape[0]),moms_c[:,2], moms_std_c[:,2],fmt='.')
        ax[2].set_xticks(np.arange(moms_c.shape[0]))
        ax[2].set_xlabel(r'Posterior cluster #')
        ax[0].set_ylabel('counts')
        ax[1].set_ylabel('v [km/s]')
        ax[2].set_ylabel(r'$T_i$ [keV]')
        plt.tight_layout()
        
# Import mpi4py here to output timing only once
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

if rank==0:
    # end time count
    elapsed_time=time_.time()-start_time
    print('Time to run: ' + str(elapsed_time) + " s")


plt.show(block=True)


#plt.subplots_adjust(hspace=0.1,right=0.99,left=0.15)
