'''
This script creates an input file for BITE from Hirex-Sr analysis using BSFC results. This require that BSFC has been previously run for a time window of interest for a chosen shot. 

'''
import os, sys, argparse
import numpy as np
import pickle as pkl  #python 3+

import bsfc_moment_fitter as bmf
import bsfc_data
from helpers import bsfc_clean_moments 
from helpers import bsfc_slider
from helpers import bsfc_cmod_shots


parser = argparse.ArgumentParser()
parser.add_argument("shot", type=int, help="shot number.")
parser.add_argument("-l", "--line_name", help="name of atomic line of interest for post-fitting analysis. For the primary line, just leave to None (no need to pass argument).")
parser.add_argument("-p", "--plot", action='store_true', help="Indicate whether results should be plotted")
args = parser.parse_args()


# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(args.shot)


mf = bmf.MomentFitter(primary_impurity, primary_line, args.shot, tht=tht)


# make sure to have the right pos vector for emission forward modeling:
pos = bsfc_data.hirexsr_pos(args.shot, mf.hirex_branch, tht, primary_line, primary_impurity)


# ====== load BSFC results ======= #
if args.line_name is None:
    line_name = primary_line
else:
    line_name = args.line_name

# new format from bsfc_ns_shot
with open(f'../bsfc_fits/moments_{args.shot:d}_tmin{t_min:.2f}_tmax{t_max:.2f}_{primary_line:s}_{primary_impurity:s}_{line_name:s}line.pkl','rb') as f:
    gathered_moments=pkl.load(f)
    
# clean up Hirex-Sr signals -- BR_THRESH might need to be adjusted to eliminate outliers
out = bsfc_clean_moments.clean_moments(mf.time,mf.maxChan,t_min,t_max,
                                       gathered_moments,BR_THRESH=1e8, #2e3,
                                       BR_STD_THRESH=2e8, normalize=False) # do not normalize at this stage

moments_vals, moments_stds, time_sel = out


if args.plot:
    # BSFC slider visualization
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='br')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='vel')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='Temp')
    

data = {} 
data['time'] = time_sel   #mf.time includes all measurement times
data['pos'] = pos
data['hirex_signal'] = moments_vals[:,:,0]
data['hirex_uncertainty'] = moments_stds[:,:,0]
data['shot'] = args.shot
data['time_1'] = t_min
data['time_2'] = t_max

# save with name only based on line name, not primary_line (which isn't useful fit info for BSFC)
with open(f'bsfc_hirex_{args.shot:d}_{line_name:s}_{primary_impurity:s}.pkl', 'wb') as f:
    pkl.dump(data, f)

print(f'Saved bsfc_hirex_{args.shot:d}_{line_name:s}_{primary_impurity:s}.pkl locally!')
