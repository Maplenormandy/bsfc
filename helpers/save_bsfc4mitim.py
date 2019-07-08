import subprocess
import MDSplus
import numpy as np
import cPickle as pkl

from bsfc_moment_fitter import *
from helpers import bsfc_clean_moments 
from helpers import bsfc_slider
from helpers import bsfc_cmod_shots


'''
This script creates an input file for MITIM from Hirex-Sr analysis using BSFC results. This require that BSFC has been previously run for a time window of interest for a chosen shot. 

'''
shot=1140729021 #1120914029
visual=False

# get key info for requested shot:
primary_impurity, primary_line, tbin,chbin, t_min, t_max,tht = bsfc_cmod_shots.get_shot_info(shot)

mf = MomentFitter(primary_impurity, primary_line, shot, tht=tht)

# ===== load pos vector ======= #
specTree = MDSplus.Tree('spectroscopy', shot)
ana = '.ANALYSIS'

if tht > 0:
    ana += str(tht)

rootPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana

try: # branch A
    # Hack for now; usually the POS variable is in LYA1 on branch B
    branchNode = specTree.getNode(rootPath+'.HELIKE')
    lam_all = branchNode.getNode('SPEC:LAM').data()
    pos_tmp = branchNode.getNode('MOMENTS.LYA1:POS').data()
    
except:  #branch B
    branchNode = specTree.getNode(rootPath+'.HLIKE')
    lam_all = branchNode.getNode('SPEC:LAM').data()
    # Otherwise, load the POS variable as normal
    try:
        pos_tmp = branchNode.getNode('MOMENTS.'+primary_line.upper()+':POS').data()
    except:
        pos_tmp = branchNode.getNode('MOMENTS.LYA1:POS').data()
        
pos=np.squeeze(pos_tmp[np.where(pos_tmp[:,0]!=-1),:])


# ====== load BSFC results ======= #

with open('../bsfc_fits/moments_%d_tmin%f_tmax%f.pkl'%(shot,t_min,t_max),'rb') as f:
    gathered_moments=pkl.load(f)

# clean up Hirex-Sr signals -- BR_THRESH might need to be adjusted to eliminate outliers
out = bsfc_clean_moments.clean_moments(mf.time,mf.maxChan,t_min,t_max,
                                       gathered_moments,BR_THRESH=2e8, BR_STD_THRESH=2e8)

moments_vals, moments_stds, time_sel = out


if visual:
    # BSFC slider visualization
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='br')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='vel')
    bsfc_slider.visualize_moments(moments_vals, moments_stds, time_sel, q='Temp')
    

data = {} 
data['time'] = time_sel   #mf.time includes all measurement times
data['pos'] = pos
data['hirex_signal'] = moments_vals[:,:,0]
data['hirex_uncertainty'] = moments_stds[:,:,0]
data['shot'] = shot
data['time_1'] = t_min
data['time_2'] = t_max

with open('bsfc_hirex_%d.pkl'%shot, 'wb') as f:
    pkl.dump(data, f)
        
