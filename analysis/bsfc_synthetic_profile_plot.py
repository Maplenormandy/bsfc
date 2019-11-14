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


font = {'family' : 'serif',
        'serif': ['Times New Roman'],
        'size'   : 8}

mpl.rc('font', **font)

# %%

#shot = 1160506007
#shot = 1150903021
shot = 1160920007

line = 'w'
##data1 = np.load('../bsfc_fits/synth_data/mf_synth_%d_%snl_nh%d.npz'%(shot,line,5))
#data2 = np.load('../bsfc_fits/synth_data/mf_synth_%d_%snl_nh%d.npz'%(shot,line,3))
#data3 = np.load('../bsfc_fits/synth_data/mf_synth_%d_%snh%d.npz'%(shot,line,3))
##data4 = np.load('../bsfc_fits/synth_data/mf_synth_%d_%snh%d.npz'%(shot,line,4))
#data5 = np.load('../bsfc_fits/synth_data/mf_synth_%d_%snh%d.npz'%(shot,line,5))

# New Jeffreys priors
data2 = np.load('../bsfc_fits/synth_data/mf_synth_%d_%s_jef_nl_nh%d.npz'%(shot,line,3))
data3 = np.load('../bsfc_fits/synth_data/mf_synth_%d_%s_jef_nh%d.npz'%(shot,line,3))
#data4 = np.load('../bsfc_fits/synth_data/mf_synth_%d_%snh%d.npz'%(shot,line,4))
#data5 = np.load('../bsfc_fits/synth_data/mf_synth_%d_%s_jef_nh%d.npz'%(shot,line,5))
#data5 = np.load('../bsfc_fits/synth_data/mf_synth_%d_nl_nh%d.npz'%(shot,3))
#data3 = np.load('../bsfc_fits/synth_data/mf_synth_%d_nh%d.npz'%(shot,3))

# %%

data = data2

gdata = data['meas_avg']
gstd = data['meas_std']

bins = np.array(range(data['meas_avg'].shape[1]))

"""
for chbin in bins:
    if np.isnan(gdata[0,chbin]):
        continue

    bin_data = np.load('../bsfc_fits/synth_data/mf_synth_%d_%snh%d_ch%d.npz'%(shot,line,3,chbin))

    meas = bin_data['measurements']
    weights = bin_data['sample_weights']
    scales = bin_data['samples'][:,2]
    meas[:,0] = meas[:,0] * scales

    gdata[0,chbin] = np.average(meas[:,0], weights=weights)
    gstd[0,chbin] = np.sqrt(np.average((meas[:,0] - gdata[0,chbin])**2, weights=weights))
"""

brightChange = 1.0

brightChange = np.nanmean(data['meas_true'][0,:]) / np.nanmean(gdata[0,:])

# %%

plt.figure(1, figsize=(3.375, 3.375*1.2))
gs1 = mpl.gridspec.GridSpec(3, 1, hspace=0.0)
ax0 = plt.subplot(gs1[0])
ax1 = plt.subplot(gs1[1], sharex=ax0)
ax2 = plt.subplot(gs1[2], sharex=ax0)


bins = np.array(range(data['meas_true'].shape[1]))
ax0.plot(bins, data['meas_true'][0,:], marker='.', label='synthetic')
ax0.errorbar(bins, gdata[0,:]*brightChange, marker='.', yerr=gstd[0,:]*brightChange, label='BSFC')

ax1.plot(bins, data['meas_true'][1,:], marker='.')
ax1.errorbar(bins, gdata[1,:], marker='.', yerr=gstd[1,:])

ax2.plot(bins, data['meas_true'][2,:], marker='.')
ax2.errorbar(bins, gdata[2,:], marker='.', yerr=gstd[2,:])

ax0.set_ylabel('Brightness [a.u.]')
ax1.set_ylabel('Velocity [km/s]')
ax2.set_ylabel('Temperature [keV]')

plt.xlabel('Channel #')

#ax0.set_ylim([-0.05, 0.7])
#ax1.set_ylim([-0.5, 5.5])
#ax2.set_ylim([0.0,1.35])
#
ax0.set_ylim([0.09, 0.45])
ax1.set_ylim([-13, -3])
ax2.set_ylim([0.8,2.1])

ax0.set_xlim([0, len(bins)])

plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)

ax0.legend(loc='lower right')

plt.xlabel('Spatial Channel #')
plt.tight_layout()
plt.tight_layout()

plt.savefig('/home/normandy/Pictures/BSFC/newfigs/figure5.eps')

# %%

plt.figure()
plt.errorbar(bins, data3['lnev'], yerr=data3['lnev_std'], c='b')
##plt.errorbar(bins, data4['lnev'], yerr=data4['lnev_std'], c='g')
#plt.errorbar(bins, data5['lnev'], yerr=data5['lnev_std'], c='r')
plt.errorbar(bins, data2['lnev'], yerr=data2['lnev_std'], c='m')
##plt.errorbar(bins, data1['lnev'], yerr=data1['lnev_std'], c='y')

# %%


chbin = 30
#bin_data = np.load('../bsfc_fits/synth_data/mf_synth_%d_%s_jef_nh%d_ch%d.npz'%(shot,line,3,chbin))

"""
plt.figure()
meas = bin_data['measurements']
weights = bin_data['sample_weights']
plt.hist(meas[:,2], weights=weights, bins=128)
plt.axvline(np.average(meas[:,2], weights=weights), c='r', ls='--')

vsorted = np.argsort(meas[:,2])
vs = meas[vsorted,2]
ws = np.cumsum(weights[vsorted])
median = np.interp(0.5, ws, vs)

plt.axvline(median, c='g', ls='--')
"""

#corner.corner(bin_data['samples'], weights=bin_data['sample_weights'], range=[0.99]*bin_data['samples'].shape[1])
#plt.plot()

# %%

#toplot = np.array([bin_data['samples'][:,6], meas[:,2]]).T

#corner.corner(toplot, weights=bin_data['sample_weights'], range=[0.99,0.99])
plt.show()
