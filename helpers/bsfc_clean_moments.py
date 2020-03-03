# -*- coding: utf-8 -*-
"""
Convenience function to clean Hirex-Sr signals after spectral line fitting. 

@author: sciortino
"""
from __future__ import division
from builtins import range
from past.utils import old_div

import numpy as np


def clean_moments(mf_time,mf_maxChan, t_min, t_max, gathered_moments, BR_THRESH=2.0, BR_STD_THRESH=0.1, BR_REL_THRESH=0.5, normalize=True):
    ''' Re-organize gathered moments into a numpy array. 
    Exclude brightness data points whose value exceeds 'BR_THRESH'. 

    Parameters:
    mf_time: Time array in the BSFC Moment-Fitter object, given in mf.time
    mf_maxChan: number of channels in the BSFC Moment-Fitter object, given in mf.maxChan
    t_min [s]: minimum time to be kept in results
    t_max [s]: maximum time to be kept in results
    gathered_moments: result of running bsfc_run_mpi.py, saved in ./bsfc_fits

    BR_THRESH: threshold to exclude outliers based on their brightness value
    BR_STDS_THRESH: threshold to exclude outliers based on their brightness standard deviation
    BR_REL_THRESH: threshold to exclude outliers based on their relative uncertainty

    normalize: normalize brightness to largest value across chords.
    '''
    # activate this for future BSFC runs:
    #ind_sel = np.where(np.logical_and(mf_time>t_min, mf_time<t_max))
    #time_sel= np.array(mf_time)[ind_sel] #[tidx_min: tidx_max]
    #tidx_min = ind_sel[0][0]
    #tidx_max = ind_sel[0][-1] +1  # python indexing
    
    tidx_min = np.argmin(np.abs(mf_time - t_min))
    tidx_max = np.argmin(np.abs(mf_time - t_max))
    time_sel = np.array(mf_time)[tidx_min:tidx_max]
    
    
    # get individual spectral moments 
    moments_vals = np.empty((tidx_max-tidx_min,mf_maxChan,3))
    moments_stds = np.empty((tidx_max-tidx_min,mf_maxChan,3))
    moments_vals[:] = None
    moments_stds[:] = None

    for tbin in range(tidx_max-tidx_min):
        for chbin in range(mf_maxChan):
            moments_vals[tbin,chbin,0] = gathered_moments[tbin,chbin][0][0]
            moments_stds[tbin,chbin,0] = gathered_moments[tbin,chbin][1][0]
            moments_vals[tbin,chbin,1] = gathered_moments[tbin,chbin][0][1]
            moments_stds[tbin,chbin,1] = gathered_moments[tbin,chbin][1][1]
            moments_vals[tbin,chbin,2] = gathered_moments[tbin,chbin][0][2]
            moments_stds[tbin,chbin,2] = gathered_moments[tbin,chbin][1][2]
            
    # exclude values with brightness value or uncertainty greater than certain values
    moments_vals[:,:,0][np.logical_or(moments_vals[:,:,0] > BR_THRESH , moments_stds[:,:,0] > BR_STD_THRESH)] = np.nan
    moments_stds[:,:,0][np.logical_or(moments_vals[:,:,0] > BR_THRESH , moments_stds[:,:,0] > BR_STD_THRESH)] = np.nan
    moments_vals[:,:,1][np.logical_or(moments_vals[:,:,0] > BR_THRESH , moments_stds[:,:,0] > BR_STD_THRESH)] = np.nan
    moments_stds[:,:,1][np.logical_or(moments_vals[:,:,0] > BR_THRESH , moments_stds[:,:,0] > BR_STD_THRESH)] = np.nan
    moments_vals[:,:,2][np.logical_or(moments_vals[:,:,0] > BR_THRESH , moments_stds[:,:,0] > BR_STD_THRESH)] = np.nan
    moments_stds[:,:,2][np.logical_or(moments_vals[:,:,0] > BR_THRESH , moments_stds[:,:,0] > BR_STD_THRESH)] = np.nan

    # threshold on relative uncertainty
    '''
    moments_vals[:,:,0][moments_stds[:,:,0]/(moments_vals[:,:,0]+1e-10) > BR_REL_THRESH] = np.nan
    moments_stds[:,:,0][moments_stds[:,:,0]/(moments_vals[:,:,0]+1e-10) > BR_REL_THRESH] = np.nan
    '''
    if normalize:
        # normalize brightness to largest value
        idx1,idx2 = np.unravel_index(np.nanargmax(moments_vals[:,:,0]), moments_vals[:,:,0].shape)
        max_br = moments_vals[idx1,idx2,0]
        max_br_std = moments_stds[idx1,idx2,0]

        moments_vals[:, :,0] = old_div(moments_vals[:,:,0], max_br)
        moments_stds[:,:,0] = np.sqrt((old_div(moments_stds[:,:,0], max_br))**2.0 + ((old_div(moments_vals[:,:,0], max_br))*(old_div(max_br_std, max_br)))**2.0)

    return moments_vals, moments_stds, time_sel
