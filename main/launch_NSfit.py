import sys
import pickle as pkl
import os
import numpy as np
import glob, argparse

# inputs to identify MomentFitter file and bin to fit

parser = argparse.ArgumentParser()
parser.add_argument("shot", type=int, help="shot number to run analysis on.")
parser.add_argument("primary_line", type=str, help="name of primary atomic line of fit.")
parser.add_argument("primary_impurity", type=str, help="name of primary atomic impurity giving rise to line of fit.")
parser.add_argument("t_min", type=float, help="name of primary atomic line of fit.")
parser.add_argument("t_max", type=float,help="name of primary atomic line of fit.")
parser.add_argument("tb", type=int,help="name of primary atomic line of fit.")
parser.add_argument("cb", type=int, help="name of primary atomic line of fit.")
parser.add_argument("n_hermite",type=int, help="name of primary atomic line of fit.")
parser.add_argument("verbose",type=bool, help="name of primary atomic line of fit.")
parser.add_argument("basename", type=str,help="name of primary atomic line of fit.")

# optional argument to request saving of a non-primary line (all lines in chosen wavelength range are anyway fit)
parser.add_argument("-l", "--line_name", help="name of atomic line of interest for post-fitting analysis. For the primary line, just leave to None (no need to pass argument).")

args = parser.parse_args()


# load moment fitter setup 
with open('../bsfc_fits/mf_{args.shot:d}_tmin{args.t_min:.2f}_tmax{args.t_max:.2f}_{args.primary_line:s}line_{args.primary_impurity:s}.pkl','rb') as f:
    mf=pkl.load(f)


# do fit in the directory of 'basename'
mf.fitSingleBin(tbin=args.tb, chbin=args.cb, n_hermite=args.n_hermite,
                method=2, INS=True, const_eff=True, n_live_points='auto', sampling_efficiency=0.3,  # fixed
                verbose=args.verbose, basename=args.basename)



# now import MPI to run MultiNest data analysis only once
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    # load MultiNest output
    good = mf.fits[args.tb][args.cb].MN_analysis(args.basename)
    
    # process samples and weights
    samples = mf.fits[args.tb][args.cb].samples
    sample_weights = mf.fits[args.tb][args.cb].sample_weights

    if args.line_name is None:
        # if user didn't request a specific line, assume that primary line is of interest
        args.line_name = mf.primary_line

    try:
        line_id = np.where(mf.fits[args.tb][args.cb].lineModel.linesnames==args.line_name)[0][0]
        print(f'Analyzing {args.line_name} line')
    except:
        raise ValueError('Requested line cannot be found in MultiNest output!')

    # collect results from fits
    modelMeas = lambda x: mf.fits[args.tb][args.cb].lineModel.modelMeasurements(x, line=line_id)
    measurements = np.apply_along_axis(modelMeas, 1, samples)
    moments_vals = np.average(measurements, 0, weights=sample_weights)
    moments_stds = np.sqrt(np.average((measurements-moments_vals)**2, 0, weights=sample_weights))
    res_fit = [np.asarray(moments_vals), np.asarray(moments_stds)]
    
    # checkpoints and case_dir directories must be created externally to this script!
    checkpoints_dir ='checkpoints/'
    case_dir = '{:d}_tmin{:.2f}_tmax{:.2f}_{:s}/'.format(args.shot,args.t_min,args.t_max, args.primary_impurity)
    file_name = 'moments_{:d}_bin{:d}_{:d}_{:s}line_{:s}.pkl'.format(args.shot,args.tb,args.cb,args.primary_line, args.primary_impurity)
    resfile = checkpoints_dir + case_dir + file_name
    
    # save measurement results to checkpoint file
    with open(resfile,'wb') as f:
        pkl.dump(res_fit,f)
        
