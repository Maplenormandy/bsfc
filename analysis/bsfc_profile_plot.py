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
import gptools




font = {'family' : 'serif',
        'serif': ['Times New Roman'],
        'size'   : 8}

mpl.rc('font', **font)

# %%

shot = 1160506007
#shot = 1101014019
n_hermite = 2

#tbin = 125
#tbin = 12
tbin=46

data2 = np.load('../bsfc_fits/fit_data/mf_%d_nl_nh%d_t%d.npz'%(shot,3,tbin))
data3 = np.load('../bsfc_fits/fit_data/mf_%d_nh%d_t%d.npz'%(shot,3,tbin))
data4 = np.load('../bsfc_fits/fit_data/mf_%d_nh%d_t%d.npz'%(shot,4,tbin))
#data5 = np.load('../bsfc_fits/fit_data/mf_%d_nh%d.npz'%(shot,5))

# %%

specTree = MDSplus.Tree('spectroscopy', shot)
#momNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HLIKE.MOMENTS.LYA1')
momNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.MOMENTS.W')

momsRaw = momNode.getNode('mom').data()
momsErr = momNode.getNode('err').data()

# %%

maxChan = data2['meas_avg'].shape[1]

lamw = 3.94912
#lamw = 3.73114
mAr = 39.948
c = 2.998e+5

normError = (momsErr[:,tbin,:maxChan] / momsRaw[:,tbin,:maxChan])**2
moms = momsRaw[:,tbin,:maxChan]

v_thaco = moms[1] / moms[0] * c/lamw
v_err_thaco = np.sqrt(v_thaco**2 * (normError[1] + normError[0]))

t_thaco = moms[2] / moms[0] * mAr / lamw**2 * 1e6
t_err_thaco = np.sqrt(t_thaco**2 * (normError[2] + normError[0]))

# %%

data = data2


gdata = data['meas_avg']
gstd = data['meas_std']

bins = np.array(range(data['meas_avg'].shape[1]))

"""
for chbin in bins:
    if np.isnan(gdata[0,chbin]):
        continue

    bin_data = np.load('../bsfc_fits/fit_data/mf_%d_nh%d_ch%d.npz'%(shot,3,chbin))

    meas = bin_data['measurements']
    weights = bin_data['sample_weights']
    scales = bin_data['samples'][:,2]
    meas[:,0] = meas[:,0] * scales

    gdata[0,chbin] = np.average(meas[:,0], weights=weights)
    gstd[0,chbin] = np.sqrt(np.average((meas[:,0] - gdata[0,chbin])**2, weights=weights))
"""

gdata[:,8] = np.nan
moms[0,8] = np.nan
v_thaco[8] = np.nan
t_thaco[8] = np.nan

gdata[:,24] = np.nan
moms[0,24] = np.nan
v_thaco[24] = np.nan
t_thaco[24] = np.nan

brightChange = 1.0
lbin = 26

brightChange = np.nanmean(moms[0,lbin:]) / np.nanmean(gdata[0,lbin:])

# %%

plt.figure(2, figsize=(3.375, 3.375*1.2))
gs1 = mpl.gridspec.GridSpec(3, 1, hspace=0.0)
ax0 = plt.subplot(gs1[0])
ax1 = plt.subplot(gs1[1], sharex=ax0)
ax2 = plt.subplot(gs1[2], sharex=ax0)

#ax0.errorbar(bins[:lbin], moms[0,:lbin], marker='.', c='r', yerr=momsErr[0,tbin,:lbin])
#ax0.errorbar(bins[:lbin], gdata[0,:lbin]*brightChange, marker='.', yerr=gstd[0,:lbin]*brightChange, c='g')

ax0.errorbar(bins[lbin:], moms[0,lbin:], marker='.', c='r', yerr=momsErr[0,tbin,lbin:maxChan], label='THACO')
ax0.errorbar(bins[lbin:], gdata[0,lbin:]*brightChange, marker='.', yerr=gstd[0,lbin:]*brightChange, c='g', label='BSFC')

#ax1.plot(bins, data['meas_true'][1,:], marker='.')
ax1.errorbar(bins[lbin:], v_thaco[lbin:], marker='.', yerr=v_err_thaco[lbin:], c='r')
ax1.errorbar(bins[lbin:], gdata[1,lbin:], marker='.', yerr=gstd[1,lbin:], c='g')

#ax2.plot(bins, data['meas_true'][2,:]-0.5, marker='.')
ax2.errorbar(bins[lbin:], t_thaco[lbin:], marker='.', yerr=t_err_thaco[lbin:], c='r')
ax2.errorbar(bins[lbin:], gdata[2,lbin:], marker='.', yerr=gstd[2,lbin:], c='g')

ax0.set_ylabel('Brightness [a.u.]')
ax1.set_ylabel('Velocity [km/s]')
ax2.set_ylabel('Temperature [keV]')


#ax0.set_ylim([-0.1, 1.9])
#ax1.set_ylim([-32, 12])
#ax2.set_ylim([-0.1,2.2])
#
#
#ax0.set_xlim([24, 56])
ax0.set_ylim([1.25, 2.1])
ax1.set_ylim([-3, 7])
ax2.set_ylim([1.6,1.97])

ax0.legend(loc='lower right')


#ax0.set_ylim([-0.01, 0.14])
#ax1.set_ylim([-8, 38])
#ax2.set_ylim([0.0,2.8])


ax0.set_xlim([lbin-1, len(bins)])

plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)

plt.xlabel('Spatial Channel #')
plt.tight_layout()
plt.tight_layout()

plt.savefig('/home/normandy/Pictures/figure6.eps')

# %%

plt.figure()
plt.errorbar(bins, data3['lnev'], yerr=data3['lnev_std'], c='b')
plt.errorbar(bins, data4['lnev'], yerr=data4['lnev_std'], c='g')
#plt.errorbar(bins, data5['lnev'], yerr=data5['lnev_std'], c='r')
plt.errorbar(bins, data2['lnev'], yerr=data2['lnev_std'], c='m')

# %%
"""

chbin = 36
n_hermite = 3
bin_data = np.load('../bsfc_fits/fit_data/mf_%d_nh%d_ch%d.npz'%(shot,n_hermite,chbin))

meas = bin_data['measurements']
weights = bin_data['sample_weights']
scales = bin_data['samples'][:,2]

toplot = np.array([bin_data['samples'][:,3+n_hermite], meas[:,1], meas[:,2]]).T

#corner.corner(bin_data['samples'], weights=bin_data['sample_weights'], range=[0.99]*bin_data['samples'].shape[1])
plt.show()

"""
"""
plt.hist(meas[:,1], weights=weights, bins=128)
plt.axvline(np.average(meas[:,1], weights=weights), c='r', ls='--')

vsorted = np.argsort(meas[:,1])
vs = meas[vsorted,1]
ws = np.cumsum(weights[vsorted])
median = np.interp(0.5, ws, vs)

plt.axvline(median, c='g', ls='--')
"""

# %%

#plt.figure(figsize=(3.375*2.5,3.375*2.5))
#plt.close('all')

"""
f = gptools.plot_sampler(
    toplot, # index 0 is weights, index 1 is -2*loglikelihood, then samples
    weights=bin_data['sample_weights'],
    labels=['Satellite Br. [a.u.]', 'Velocity [km/s]', 'Temperature [keV]'],
    chain_alpha=1.0,
    cutoff_weight=0.001,
    cmap='plasma',
    #suptitle='Posterior distribution of $D$ and $V$',
    plot_samples=False,
    plot_chains=False,
    xticklabel_angle=45,
    #yticklabel_angle=30
    ticklabel_fontsize=16,
)
"""

# %%



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
