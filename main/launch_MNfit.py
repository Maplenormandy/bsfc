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
with open(f'../bsfc_fits/mf_{args.shot:d}_tmin{args.t_min:.2f}_tmax{args.t_max:.2f}_{args.primary_line:s}_{args.primary_impurity:s}.pkl','rb') as f:
    mf=pkl.load(f)


# do fit in the directory of 'basename'
mf.fitSingleBin(tbin=args.tb, chbin=args.cb, n_hermite=args.n_hermite,
                method=2, INS=True, const_eff=True, n_live_points='auto', sampling_efficiency=0.3,  # fixed
                verbose=args.verbose, basename=args.basename)


with open(f'../bsfc_fits/mf_{args.shot:d}_tmin{args.t_min:.2f}_tmax{args.t_max:.2f}_{args.primary_line:s}_{args.primary_impurity:s}.pkl','wb') as f:
    pkl.dump(mf, f)

    
