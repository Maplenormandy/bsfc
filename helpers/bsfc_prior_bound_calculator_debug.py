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
    derivCoef = np.concatenate([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], theta])
    r = hermeroots(derivCoef)
    n = len(r)-1
    isolated_roots = np.logical_or(np.abs(np.imag(r)) > 1e-6, np.abs(r) > np.sqrt(1.25*n)+1)
    origin_roots = np.logical_and(np.abs(np.real(r)) < 0.2, np.logical_not(isolated_roots))
    
    if (np.sum(origin_roots) == 1) and (np.sum(isolated_roots) == n):
        return 0.0
    else:
        return -np.inf
        
ndim = 2
nwalkers = max(8, ndim**2)
nwalkers = nwalkers + (nwalkers%2)
nburn = 100
nsamples = 1000
p0 = [np.random.rand(ndim)*1e-4 for i in np.arange(nwalkers)]
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
rescaled = np.dot(data - mean[np.newaxis], invMatrix)




# %%

figure = corner.corner(data, labels=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'])
# %%



rescaled = np.dot((data-mean[np.newaxis,:]), invTrans)
# %%

figure = corner.corner(rescaled)

# %%

rscaled = np.sum(rescaled**2, axis=1)
hist, bin_edges = np.histogram(rscaled, bins=256, density=True)
bins = (bin_edges[1:] + bin_edges[:-1])/2.0
plt.loglog(bins, hist)
plt.loglog(bins, bins*1e-1)
#x = np.linspace(0,np.max(rscaled))
#plt.plot(x, scipy.stats.chi2.pdf(x, 8))

# %%

plt.figure()
ax = plt.subplot(111, projection='3d')
good = np.abs(data[:,2]+0.1) < 1e-3
ax.scatter(rescaled[good,0], rescaled[good,1], rescaled[good,3])

# %%
dataMax = np.amax(rescaled, axis=0)
dataMin = np.amin(rescaled, axis=0)

# %%

#def advancedHypercubeToHermiteSampleFunction(a0_max, n_hermite):
#    if n_hermite%2 == 0:
#        raise NotImplementedError("Only odd numbers of hermite functions are supported right now")

def hypercubeToSimplex(z, r):
    """
    Takes a length-n vector with components between 0-1 (e.g. sampled from an n-dimensional
    hypercube), and a m x (n+1) dimensional array consisting of the (n+1) vertices
    of the simplex each with m specified coordinates

    Parameters:
    z : array (n-elements)
         vector with components between 0 and 1
    r : array m x (n+1)  (NB: usually m=n)
        vertices of the simplex to be sampled.
    """
    z_sorted = np.concatenate(([0], np.sort(z), [1]))
    bary_coords = np.diff(z_sorted)
    return np.dot(r, bary_coords)
    
def advancedCubePrior(z):
    y = 2*z-1
    signY = np.sign(np.sign(y) + 1e-2)
    r = np.diag(2*signY, k=1)
    r = r[:-1,:]
        
    x = hypercubeToSimplex(np.abs(y), r)
    return np.dot(transMatrix, x) + mean



# %%

cube = np.random.rand(2, 1000)
output = np.apply_along_axis(advancedCubePrior, 0, cube)

plt.figure()
plt.scatter(output[0,:], output[1,:])

# %%

def samplePlane(n1, n2, nsamples):
    max1 = np.max(data[:,n1-1])
    max2 = np.max(data[:,n2-2])
    
    samples = np.random.rand(2, nsamples)
    thetas = np.zeros((8, nsamples))
    if n1%2 == 1:
        thetas[n1-1,:] = (2.5 * samples[0,:] - 1.25) * max1
    else:
        thetas[n1-1,:] = (1.75 * samples[0,:] - 0.4) * max1
    if n2%2 == 1:
        thetas[n2-1,:] = (2.5 * samples[1,:] - 1.25) * max2
    else:
        thetas[n2-1,:] = (1.75 * samples[1,:] - 0.4) * max2
    
    probs = np.apply_along_axis(lnprob, 0, thetas)
    good = np.isfinite(probs)
    bad = np.logical_not(good)
    
    transThetas = np.dot((thetas.T), invTrans).T
    
    plt.figure()
    plt.scatter(transThetas[n1-1,good], transThetas[n2-1,good], c='b')
    #plt.scatter(thetas[n1-1,bad], thetas[n2-1,bad], c='r')
    
samplePlane(2,4, 10000)
