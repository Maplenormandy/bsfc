''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

BSFC functions to obtain moments from spectral fitting

'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# make it possible to use other packages within the BSFC distribution:
from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))

# BSFC packages
from helpers import bsfc_helper
from helpers import bsfc_autocorr

from bsfc_moment_fitter import *
from bsfc_line_model import *

def get_brightness(mf, t_min=1.2, t_max=1.4, plot=False, save=False):
    '''
    Function to obtain time series of Hirex-Sr brightnesses in all channels.
    If data has already been fitted for a shot, one may set nofit=True.
    '''
    # # select times of interest
    tidx_min = np.argmin(np.abs(mf.time - t_min))
    tidx_max = np.argmin(np.abs(mf.time - t_max))
    time_sel= mf.time[tidx_min: tidx_max]

    # collect moments and respective standard deviations
    moments, moments_std = unpack_moments(mf, tidx_min, tidx_max)

    # load Hirex-Sr position vector
    pos = mf.pos

    # Get fitted results for brightness
    br = np.zeros((tidx_max-tidx_min, mf.maxChan))
    br_unc = np.zeros((tidx_max-tidx_min, mf.maxChan))
    for chbin in range(mf.maxChan):
        t=0
        for tbin in range(tidx_max-tidx_min): #range(tidx_min, tidx_max):
            if mf.fits[tbin][chbin].good:
                br[t,chbin] = mf.fits[tbin][chbin].m_avg[0]
                br_unc[t,chbin] = mf.fits[tbin][chbin].m_std[0]
            else:
                br[t,chbin] = np.nan
                br_unc[t,chbin] = np.nan
            t+=1

    # adapt this mask based on experience
    # mask=np.logical_and(br>0.2, br>0.05)
    # br[mask]=np.nan
    # br_unc[mask]=np.nan

    if save:
        # store signals in format for MITIM analysis
        inj = bsfc_helper.Injection(t_min, t_min-0.02, t_max)
        sig=bsfc_helper.HirexData(shot=mf.shot,
            sig=br,
            unc=br_unc,
            pos=pos,
            time=time_sel,
            tht=mf.tht,
            injection=inj,
            debug_plots=plot)

    if plot:
        # plot time series of Hirex-Sr signals for all channels
        plt.figure()
        for i in range(hirex_signal.shape[1]):
            plt.errorbar(time_sel, hirex_signal[:,i], hirex_uncertainty[:,i], fmt='-.', label='ch. %d'%i)
        leg=plt.legend(fontsize=8)
        leg.draggable()

        # # compare obtained normalized fits with those from THACO fits of 1101014019
        # with open('signals_1101014019.pkl','rb') as f:
        #     signals=pkl.load(f)

        # plt.figure()
        # plt.subplot(211)
        # for i in range(sig.signal.y.shape[1]):
        #     plt.errorbar(sig.signal.t, sig.signal.y_norm[:,i], sig.signal.std_y_norm[:,i])#, '.-')
        # plt.title('new fits')

        # plt.subplot(212)
        # for i in range(sig.signal.y.shape[1]):
        #     plt.errorbar(signals[0].t, signals[0].y_norm[:,i], signals[0].std_y_norm[:,i])
        # plt.title('saved fits from 1101014019')
    if save:
        return sig.signal
    else:
        return br_vals, br_stds, time_sel



def get_rotation(mf, t_min=1.2, t_max=1.4, line=0, plot=False):
    '''
    Function to obtain time series of Hirex-Sr rotation in all channels.
    If data has already been fitted for a shot, one may set nofit=True.
    '''
    # # select times of interest
    tidx_min = np.argmin(np.abs(mf.time - t_min))
    tidx_max = np.argmin(np.abs(mf.time - t_max))
    time_sel= mf.time[tidx_min: tidx_max]

    # collect moments and respective standard deviations
    moments, moments_std = unpack_moments(mf, tidx_min, tidx_max)

    # load Hirex-Sr position vector
    pos = mf.pos

    c = 2.998e+5 # speed of light in km/s

    rot = np.zeros((tidx_max-tidx_min, mf.maxChan))
    rot_unc = np.zeros((tidx_max-tidx_min, mf.maxChan))
    for chbin in range(mf.maxChan):
        t=0
        for tbin in range(tidx_min, tidx_max):
            if mf.fits[tbin][chbin].good:
                m0 = mf.fits[tbin][chbin].m_avg[0]
                m1 = mf.fits[tbin][chbin].m_avg[1]
                m0_std = mf.fits[tbin][chbin].m_std[0]
                m1_std = mf.fits[tbin][chbin].m_std[1]
                linesLam = mf.fits[tbin][chbin].lineModel.linesLam[line]

                rot[t,chbin] = (m1 / (m0 * linesLam)) *1e-3 * c
                rot_unc[t,chbin] = (1e-3 * c/ linesLam) * np.sqrt((m1_std**2/m0**2)+(m1**2*m0_std**2/m0**4))

            else:
                rot[t,chbin] = np.nan
                rot_unc[t,chbin] = np.nan
            t+=1

    if plot:
        # plot time series of Hirex-Sr signals for all channels
        plt.figure()
        for i in range(rot.shape[1]):
            plt.errorbar(time_sel, rot[:,i], rot_unc[:,i], fmt='-.', label='ch. %d'%i)
        leg=plt.legend(fontsize=8)
        leg.draggable()

    return rot_vals, rot_stds, time_sel




def get_temperature(mf, t_min=1.2, t_max=1.4, line=0, plot=False):
    '''
    Function to obtain time series of Hirex-Sr rotation in all channels.
    If data has already been fitted for a shot, one may set nofit=True.
    '''
    # # select times of interest
    tidx_min = np.argmin(np.abs(mf.time - t_min))
    tidx_max = np.argmin(np.abs(mf.time - t_max))
    time_sel= mf.time[tidx_min: tidx_max]

    # collect moments and respective standard deviations
    moments, moments_std = unpack_moments(mf, tidx_min, tidx_max)

    # load Hirex-Sr position vector
    pos = mf.pos

    c = 2.998e+5 # speed of light in km/s

    Temp = np.zeros((tidx_max-tidx_min, mf.maxChan))
    Temp_unc = np.zeros((tidx_max-tidx_min, mf.maxChan))

    for chbin in range(mf.maxChan):
        t=0
        for tbin in range(tidx_min, tidx_max):
            linesLam = mf.fits[tbin][chbin].lineModel.linesLam[line]
            linesFit = mf.fits[tbin][chbin].lineModel.linesFit
            m_kev = mf.fits[tbin][chbin].lineModel.lineData.m_kev[linesFit][line]
            w = linesLam**2 / m_kev

            if mf.fits[tbin][chbin].good:
                m0 = mf.fits[tbin][chbin].m_avg[0]
                m1 = mf.fits[tbin][chbin].m_avg[1]
                m2 = mf.fits[tbin][chbin].m_avg[2]
                m0_std = mf.fits[tbin][chbin].m_std[0]
                m1_std = mf.fits[tbin][chbin].m_std[1]
                m2_std = mf.fits[tbin][chbin].m_std[2]

                Temp[t,chbin] = m2*1e-6/m0 / w #(m2/(linesLam**2 *m0)) * m_kev *1e-6
                Temp_unc[t,chbin] = (1e-6/ w) * np.sqrt((m2_std**2/m0**2)+((m1**2*m0_std**2)/m0**4))

            else:
                Temp[t,chbin] = np.nan
                Temp_unc[t,chbin] = np.nan
            t+=1

    if plot:
        # plot time series of Hirex-Sr signals for all channels
        plt.figure()
        for i in range(rot.shape[1]):
            plt.errorbar(time_sel, Temp[:,i], Temp_unc[:,i], fmt='-.', label='ch. %d'%i)
        leg=plt.legend(fontsize=8)
        leg.draggable()

    return Temp_vals, Temp_stds, time_sel


############################
def get_meas(mf, t_min=1.2, t_max=1.4, line=0, plot=False):
    #print("Computing brightness, rotation and ion temperature")
    # # select times of interest
    tidx_min = np.argmin(np.abs(mf.time - t_min))
    tidx_max = np.argmin(np.abs(mf.time - t_max))
    time_sel= mf.time[tidx_min: tidx_max]

    # collect moments and respective standard deviations
    moments = np.empty((tidx_max-tidx_min,mf.maxChan,3))
    moments_std = np.empty((tidx_max-tidx_min,mf.maxChan,3))
    moments[:] = None
    moments_std[:] = None


    for chbin in range(mf.maxChan):
        t=0
        for tbin in range(tidx_min, tidx_max):
            if mf.fits[tbin][chbin].good:
                chain = mf.fits[tbin][chbin].samples
                moments_vals = np.apply_along_axis(mf.fits[tbin][chbin].lineModel.modelMeasurements, axis=1, arr=chain)
                moments[t, chbin,:] = np.mean(moments_vals, axis=0)
                moments_std[t, chbin,:] = np.std(moments_vals, axis=0)
            t+=1

    return moments_vals, moments_std, time_sel


def plotOverChannels(mf, tbin=126, parallel=True, nproc=None, nsteps=1000):
    '''
    Function to fit signals for a specified time bin, across all channels.
    Optionally, plots 0th, 1st and 2nd moment across channels.
    '''

    moments = [None] * mf.maxChan
    moments_std = [None] * mf.maxChan

    for chbin in range(mf.maxChan):
        if mf.fits[tbin][chbin].good:
            moments[chbin] = mf.fits[tbin][chbin].m_avg
            moments_std[chbin] = mf.fits[tbin][chbin].m_std
        else:
            moments[chbin] = np.zeros(3)
            moments_std[chbin] = np.zeros(3)

    moments = np.array(moments)
    moments_std = np.array(moments_std)

    f, a = plt.subplots(3, 1, sharex=True)

    a[0].errorbar(range(mf.maxChan), moments[:,0], yerr=moments_std[:,0], fmt='.')
    a[0].set_ylabel('0th moment')
    a[1].errorbar(range(mf.maxChan), moments[:,1], yerr=moments_std[:,1], fmt='.')
    a[1].set_ylabel('1st moment')
    a[2].errorbar(range(mf.maxChan), moments[:,2], yerr=moments_std[:,2], fmt='.')
    a[2].set_ylabel('2nd moment')
    a[2].set_xlabel(r'channel')



# %% =====================================
def unpack_moments(mf, tidx_min, tidx_max):
    # collect moments and respective standard deviations
    moments = np.empty((tidx_max-tidx_min,mf.maxChan,3))
    moments_std = np.empty((tidx_max-tidx_min,mf.maxChan,3))#[None] * (tidx_max - tidx_min)

    for chbin in range(mf.maxChan):
        t=0
        for tbin in range(tidx_max-tidx_min): #range(tidx_min, tidx_max):
            if mf.fits[tbin][chbin].good:
                moments[t,chbin,:] = mf.fits[tbin][chbin].m_avg
                moments_std[t,chbin,:] = mf.fits[tbin][chbin].m_std
            else:
                moments[t,chbin,:] = np.zeros(3)
                moments_std[t,chbin,:] = np.zeros(3)
            t+=1

    moments = np.array(moments)
    moments_std = np.array(moments_std)

    return moments, moments_std

