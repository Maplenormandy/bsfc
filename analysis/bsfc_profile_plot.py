# -*- coding: utf-8 -*-
"""
Tests bsfc_main versus synthetic data, to see the quality of the fits

@author: normandy
"""

import readline
import MDSplus

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.ion()

import corner

# %%

shot = 1160506007
n_hermite = 3

data2 = np.load('../bsfc_fits/fit_data/mf_%d_nl_nh%d.npz'%(shot,3))
data3 = np.load('../bsfc_fits/fit_data/mf_%d_nh%d.npz'%(shot,3))
data4 = np.load('../bsfc_fits/fit_data/mf_%d_nh%d.npz'%(shot,4))
data5 = np.load('../bsfc_fits/fit_data/mf_%d_nh%d.npz'%(shot,5))

# %%

data = data4

plt.figure()
gs1 = mpl.gridspec.GridSpec(3, 1)
ax0 = plt.subplot(gs1[0])
ax1 = plt.subplot(gs1[1], sharex=ax0)
ax2 = plt.subplot(gs1[2], sharex=ax0)


brightChange = 1.0

#brightChange = np.nanmean(data['meas_true'][0,:]) / np.nanmean(data['meas_avg'][0,:])

bins = np.array(range(data['meas_avg'].shape[1]))
#ax0.plot(bins, data['meas_true'][0,:], marker='.')
ax0.errorbar(bins, data['meas_avg'][0,:]*brightChange, marker='.', yerr=data['meas_std'][0,:]*brightChange, c='g')

#ax1.plot(bins, data['meas_true'][1,:], marker='.')
ax1.errorbar(bins, data['meas_avg'][1,:], marker='.', yerr=data['meas_std'][1,:], c='g')

#ax2.plot(bins, data['meas_true'][2,:]-0.5, marker='.')
ax2.errorbar(bins, data['meas_avg'][2,:], marker='.', yerr=data['meas_std'][2,:], c='g')

ax0.set_ylabel('Brightness [a.u.]')
ax1.set_ylabel('Velocity [km/s]')
ax2.set_ylabel('Temperature [keV]')

plt.xlabel('Channel #')

# %%

plt.figure()
plt.errorbar(bins, data3['lnev'], yerr=data3['lnev_std'], c='b')
plt.errorbar(bins, data4['lnev'], yerr=data4['lnev_std'], c='g')
plt.errorbar(bins, data5['lnev'], yerr=data5['lnev_std'], c='r')
plt.errorbar(bins, data2['lnev'], yerr=data2['lnev_std'], c='m')

# %%

chbin = 3
bin_data = np.load('../bsfc_fits/fit_data/mf_%d_nh%d_ch%d.npz'%(shot,3,chbin))

meas = bin_data['measurements']
weights = bin_data['sample_weights']
plt.hist(meas[:,1], weights=weights, bins=128)
plt.axvline(np.average(meas[:,1], weights=weights), c='r', ls='--')

vsorted = np.argsort(meas[:,1])
vs = meas[vsorted,1]
ws = np.cumsum(weights[vsorted])
median = np.interp(0.5, ws, vs)

plt.axvline(median, c='g', ls='--')

# %%

corner.corner(bin_data['samples'], weights=bin_data['sample_weights'])
plt.show()

# %%

def modelMoments(lm, theta, line=0, order=-1):
    """
    Calculate the moments predicted by the model.

    Note, THACO actually calculates the zeroth moment
    """
    noise, center, scale, herm = lm.unpackTheta(theta)

    # Since the Probablist's Hermite functions are orthogonal given the unit normal,
    # to integrate the mean and variance just get the weights multiplied by x.
    hermx = hermemulx(herm[line][0:1])
    hermxx = hermemulx(hermx)

    normFactor = np.sqrt(2*np.pi)*scale[line]
    m0 = normFactor*herm[line][0]
    m1 = (center[line] - lm.linesLam[line])*m0 + normFactor*hermx[0]*scale[line]
    m2 = normFactor*hermxx[0]*scale[line]**2

    return np.array([m0, m1*1e3, m2*1e6])
    
def modelMeasurements(lm, theta, line=0, order=-1, thaco=True):
    """
    Calculate the counts, v, Ti predicted by the model
    counts in #, v in km/s, Ti in keV.

    Note that THACO doesn't calculate the total counts, and instead uses
    M0 as the A.U. brightness.
    """
    c = 2.998e+5 # speed of light in km/s

    noise, center, scale, herm = lm.unpackTheta(theta)

    moments = modelMoments(lm, theta, line, order)
    m0 = moments[0]
    # velocity is normalized M1 divided by rest wavelength times c
    # Note that this needs to be projected onto the toroidal component
    v = moments[1]*1e-3/moments[0] / lm.linesLam[line] * c
    # width of a 1 kev line = rest wavelength ** 2 / mass in kev
    w = lm.linesLam[line]**2 / lm.lineData.m_kev[lm.linesFit][line]
    ti = moments[2]*1e-6/moments[0] / w
    if thaco:
        counts = m0/scale[line]
    else:
        counts = m0

    return np.array([counts, v, ti])