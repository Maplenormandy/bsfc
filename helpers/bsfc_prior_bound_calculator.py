# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:19:46 2019

@author: normandy
"""

import numpy as np
from numpy.polynomial.hermite_e import hermeval, hermeroots
#import scipy.optimize as op
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import emcee
import corner
import scipy.linalg
import scipy.stats

# %%

def lnprob(theta):
    derivCoef = np.concatenate([[0.0, 1.0], theta])
    r = hermeroots(derivCoef)
    n = len(r)-1
    isolated_roots = np.logical_or(np.abs(np.imag(r)) > 1e-6, np.abs(r) > np.sqrt(1.25*n)+1)
    origin_roots = np.logical_and(np.abs(np.real(r)) < 0.2, np.logical_not(isolated_roots))
    
    if (np.sum(origin_roots) == 1) and (np.sum(isolated_roots) == n):
        return 0.0
    else:
        return -np.inf

def calculateCovariance(ndim):
    nwalkers = max(8, ndim**2)
    nwalkers = nwalkers + (nwalkers%2)
    nburn = 8192
    nsamples = 8192*4
    p0 = [np.random.rand(ndim)*1e-4 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    pos, prob, state = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nsamples)
    
    data = sampler.chain.reshape((-1, ndim))
    
    covMat = np.cov(data.T)
    mean = np.mean(data, axis=0)
    transMatrix = scipy.linalg.sqrtm(covMat)
    invMatrix = np.linalg.inv(transMatrix)
    
    zero = np.dot(invMatrix, np.zeros(ndim) - mean)
    rescaled = np.dot(data - mean[np.newaxis], invMatrix) - zero[np.newaxis,:]
    
    minVal = np.abs(np.amin(rescaled, axis=0))
    maxVal = np.amax(rescaled, axis=0)
    
    axes = np.minimum(minVal, maxVal)
        
    return mean, transMatrix, zero, axes

# %%

transforms = [calculateCovariance(x) for x in range(2,9)]

# %%

means = [a for a,b,c,d in transforms]
transMatrices = [b for a,b,c,d in transforms]
zeros = [c for a,b,c,d in transforms]
axes = [d for a,b,c,d in transforms]

np.savez(r'/home/normandy/git/bsfc/data/prior_bound_means.npz', *means)
np.savez(r'/home/normandy/git/bsfc/data/prior_bound_matrices.npz', *transMatrices)
np.savez(r'/home/normandy/git/bsfc/data/prior_bound_zeros.npz', *zeros)
np.savez(r'/home/normandy/git/bsfc/data/prior_bound_axes.npz', *axes)


