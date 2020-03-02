# -*- coding: utf-8 -*-
"""
Tests bsfc_main versus synthetic data, to see the quality of the fits

@author: normandy
"""

import readline
import MDSplus

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import sys
sys.path.append('/home/normandy/git/bsfc/bsfc/')

import bsfc_main
reload(bsfc_main)


# Start counting time:
#start_time=time_.time()

# %%

# Do a quick hack and load Ca lya1 lines
#mf = bsfc_main.MomentFitter('Ca', 'lya1', 1120914036, tht=0)



# %%

def shiftMoLine(mf, mo_pos):
    mf.lines.lam[2] = mf.lines.lam[0] * (1-mo_pos) + mf.lines.lam[1] * mo_pos


def momentsFromMeasurements(mf, measurement, j):
    """
    Takes in a 3xN array of measurements (m0, v [km/s], Ti [keV]) and converts
    it to a form usable in generating synthetic spectra
    """
    c = 2.998e+5 # speed of light in km/s
    moments = np.zeros(measurement.shape)
    moments[0,:] = measurement[0,:]
    moments[1,:] = measurement[1,:] * mf.lines.lam[j] / c * 1e3
    w = mf.lines.lam[j]**2 / mf.lines.m_kev[j]
    moments[2,:] = measurement[2,:] * w * 1e6

    return moments

# Ca lya1 @ 3.0185, Ca lya2 @ 3.02391, Mo 10d @ 3.0198
# mo_pos is between 0 and 1, places it between the Ca lya1 and lya2 lines
def generateSyntheticSpectrum(mf, tbin, chbin, noise, g_lya1, g_lya2, g_mo10d):
    """
    Generates synethetic spectra and calculates their true unnormalized moments
    the g_[line] arguments should be 3xN arrays of M0, m1*1e3, m2*1e6 where M0
    is the zeroth moment, m1 is the normalized 1st moment, and m2 is the normalize
    and centered second moment.

    Also places a sampled version of the spectrum into the moment fitter
    """
    moments = lambda g: np.array([np.sum(g[0,:]), np.sum(g[1,:]*g[0,:]), np.sum(g[0,:]*g[2,:] + g[0,:]*(g[1,:]**2))])

    m_lya1 = moments(g_lya1)
    m_lya2 = moments(g_lya2)
    m_mo10d = moments(g_mo10d)

    true_moments = np.array([m_lya1, m_lya2, m_mo10d])
    true_moments[:,1] = true_moments[:,1] / true_moments[:,0]
    true_moments[:,2] = true_moments[:,2] / true_moments[:,0]

    lam = mf.lam_all[:,tbin,chbin]
    lamEdge = np.zeros(len(lam)+1)
    lamEdge[1:-1] = (lam[1:] + lam[:-1]) / 2
    lamEdge[-1] = 2 * lamEdge[-2] - lamEdge[-3]
    lamEdge[0] = 2 * lamEdge[1] - lamEdge[2]

    evalLine = lambda g, l, j: g[0,:]*np.exp(-(l[:, np.newaxis] -mf.lines.lam[j] - g[1,:]*1e-3)**2/2/(g[2,:]*1e-6))/np.sqrt(2*np.pi)

    edge_lya1 = evalLine(g_lya1, lamEdge, 0)
    edge_lya2 = evalLine(g_lya2, lamEdge, 1)
    edge_mo10d = evalLine(g_mo10d, lamEdge, 2)

    ev_lya1 = (4*evalLine(g_lya1, lam, 0) + edge_lya1[1:] + edge_lya1[:-1])/6.0
    ev_lya2 = (4*evalLine(g_lya2, lam, 1) + edge_lya2[1:] + edge_lya2[:-1])/6.0
    ev_mo10d = (4*evalLine(g_mo10d, lam, 2) + edge_mo10d[1:] + edge_mo10d[:-1])/6.0

    mean = noise + np.sum(ev_lya1 + ev_lya2 + ev_mo10d, axis=1)

    #specBr = np.random.randn(len(lam))*np.sqrt(mean) + mean
    specBr = mean

    # Normalized 3rd and 4th moments, for calculating cumulants
    m3 = np.sum(g_lya1[0,:]*(g_lya1[1,:]*(g_lya1[1,:]**2+3*g_lya1[2,:]))) / m_lya1[0]
    m4 = np.sum(g_lya1[0,:]*(g_lya1[1,:]**4 + 6*g_lya1[1,:]**2*g_lya1[2,:] + 3*g_lya1[2,:]**2)) / m_lya1[0]
    # Calculate cumulants; formulas taken from wikipedia
    k3 = m3 - 3*m_lya1[2]*m_lya1[1] + 2*m_lya1[1]**3
    k4 = m4 - 4*m3*m_lya1[1] - 3*m_lya1[2]**2 + 12*m_lya1[2]*m_lya1[1]**2 - 6*m_lya1[1]**4

    #plt.plot(lam, specBr, marker='.')

    mf.specBr_all[:,tbin,chbin] = specBr
    mf.sig_all[:,tbin,chbin] = np.sqrt(specBr)

    return true_moments, k3, k4


def generateMeasurements(b0, b1, v0, v1, t0, t1):
    rho = np.linspace(0, 1)
    rhoc = 1.0-rho
    b = (rho*b0 + rhoc*b1)/len(rho)
    v = rho*v0 + rhoc*v1
    t = rho*t0 + rhoc*t1

    return np.array([b, v, t])

def modelMoments(mf, tbin, chbin):
    mf.plotSingleBinFit(0, 0)
    bf = mf.fits[0][0]
    g_lya1 = bf.lineModel.modelMoments(bf.theta_ml, line=0)
    g_lya2 = bf.lineModel.modelMoments(bf.theta_ml, line=2)
    g_mo10d = bf.lineModel.modelMoments(bf.theta_ml, line=1)
    noise, center, scale, herm = bf.lineModel.unpackTheta(bf.theta_ml)
    

    moments = np.array([g_lya1, g_mo10d, g_lya2])
    moments[:,1] = moments[:,1] / moments[:,0]
    moments[:,2] = moments[:,2] / moments[:,0]
    
    moments[0,0] = moments[0,0] / scale[0]
    moments[1,0] = moments[1,0] / scale[1]
    moments[2,0] = moments[2,0] / scale[2]

    return moments

m_lya1 = generateMeasurements(1e4, 1e3, 20, 0, 3.5, 1.5)
m_lya2 = generateMeasurements(5e2, 5e1, 20, 0, 3.5, 1.5)
m_mo10d = generateMeasurements(1e3, 1e2, 20, 0, 3.5, 1.5)

shiftMoLine(mf, 0.2)
g_lya1 = momentsFromMeasurements(mf, m_lya1, 0)
g_lya2 = momentsFromMeasurements(mf, m_lya2, 1)
g_mo10d = momentsFromMeasurements(mf, m_mo10d, 2)

true_mom, k3, k4 = generateSyntheticSpectrum(mf, 0, 0, 1e3, g_lya1, g_lya2, g_mo10d)
mf.fitSingleBin(0, 0, nsteps=1)
bf = mf.fits[0][0]

print("true")
print(true_mom)
print("fit")
print(modelMoments(mf, 0, 0))
print("fit measurements", bf.lineModel.modelMeasurements(bf.theta_ml))
print("cumulants", k3, k4)
print("percent")
print((modelMoments(mf, 0, 0) - true_mom) / true_mom * 100)
