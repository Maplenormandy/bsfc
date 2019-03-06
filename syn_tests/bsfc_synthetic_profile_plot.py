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

shot = 1150903021
n_hermite = 3

data2 = np.load('../bsfc_fits/synth_data/mf_synth_%d_nl_nh%d.npz'%(shot,3))
data3 = np.load('../bsfc_fits/synth_data/mf_synth_%d_nh%d.npz'%(shot,3))
data4 = np.load('../bsfc_fits/synth_data/mf_synth_%d_nh%d.npz'%(shot,4))
data5 = np.load('../bsfc_fits/synth_data/mf_synth_%d_nh%d.npz'%(shot,5))

# %%

data = data2

plt.figure()
gs1 = mpl.gridspec.GridSpec(3, 1)
ax0 = plt.subplot(gs1[0])
ax1 = plt.subplot(gs1[1], sharex=ax0)
ax2 = plt.subplot(gs1[2], sharex=ax0)

brightChange = np.nanmean(data['meas_true'][0,:]) / np.nanmean(data['meas_avg'][0,:])

bins = np.array(range(50))
ax0.plot(bins, data['meas_true'][0,:], marker='.')
ax0.errorbar(bins, data['meas_avg'][0,:]*brightChange, marker='.', yerr=data['meas_std'][0,:]*brightChange)

ax1.plot(bins, data['meas_true'][1,:], marker='.')
ax1.errorbar(bins, data['meas_avg'][1,:], marker='.', yerr=data['meas_std'][1,:])

ax2.plot(bins, data['meas_true'][2,:]-0.62, marker='.')
ax2.errorbar(bins, data['meas_avg'][2,:]-0.62, marker='.', yerr=data['meas_std'][2,:])

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

chbin = 30
bin_data = np.load('../bsfc_fits/synth_data/mf_synth_%d_nh%d_ch%d.npz'%(shot,3,chbin))

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