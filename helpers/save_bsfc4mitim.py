'''
This script creates an input file for MITIM from Hirex-Sr analysis using BSFC results. This require that BSFC has been previously run for a time window of interest for a chosen shot. 

'''

import MDSplus, os, sys, argparse
import numpy as np
import cPickle as pkl

from bsfc_moment_fitter import *
from helpers import bsfc_clean_moments 
from helpers import bsfc_slider
from helpers import bsfc_cmod_shots


parser = argparse.ArgumentParser()
parser.add_argument("shot", type=int, help="shot number.")
parser.add_argument("-imp", "--primary_impurity", type=str,
                    default=None, help="Overrun primary_impurity from bsfc_cmod_shots.")
parser.add_argument("-l", "--primary_line",type=str, default=None,
                    help="Overrun primary_line from bsf_cmod_shots.")
parser.add_argument("-p", "--plot", action='store_true', help="Indicate whether results should be plotted")


args = parser.parse_args()


# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(args.shot,
                                                imp_override=args.primary_impurity if args.primary_impurity is not None else '' )
if args.primary_impurity is not None: primary_impurity = args.primary_impurity # overrun
if args.primary_line is not None: primary_line = args.primary_line # overrun


mf = MomentFitter(primary_impurity, primary_line, args.shot, tht=tht)


# make sure to have the right pos vector for emission forward modeling:
pos = hirexsr_pos(args.shot, mf.hirex_branch, tht, primary_line, primary_impurity)



# ====== load BSFC results ======= #
home = os.path.expanduser('~')

with open(home+'/bsfc/bsfc_fits/moments_%d_tmin%f_tmax%f_%sline_%s.pkl'%(args.shot,t_min,t_max,primary_line,primary_impurity),'rb') as f:
    gathered_moments=pkl.load(f)

# clean up Hirex-Sr signals -- BR_THRESH might need to be adjusted to eliminate outliers
out = bsfc_clean_moments.clean_moments(mf.time,mf.maxChan,t_min,t_max,
                                       gathered_moments,BR_THRESH=1e3, #2e3,
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

with open('bsfc_hirex_%d_%s_%s.pkl'%(args.shot,primary_line, primary_impurity), 'wb') as f:
    pkl.dump(data, f)
        
print('Saved bsfc_hirex_%d_%s_%s.pkl locally!'%(args.shot,primary_line, primary_impurity))
