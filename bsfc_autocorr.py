# -*- coding: utf-8 -*-
"""
Functions to assess convergence of MCMC chains. 
Most of this code is an adaptation of the one given in 
http://emcee.readthedocs.io/en/latest/tutorials/autocorr/

@author: sciortino
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import emcee
import pdb

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

# following suggestions by @fardal on Github 
# https://github.com/dfm/emcee/issues/209
def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

def plot_convergence(sampler, dim=1):
	# look at flattened chains in the dim dimension
	# pdb.set_trace()
	if sampler.chain.shape[2]>dim:
		chain = sampler.chain[:, :, dim].T
	else:
		print "chosen chain dimension is too large! Using dim=1 instead"
		chain = sampler.chain[:, :, 0].T

	plt.figure()
	plt.hist(chain.flatten(), 100)
	plt.title('chains histogram long dim=%d'%dim)
	plt.gca().set_yticks([])
	plt.xlabel(r"$\theta$")
	plt.ylabel(r"$p(\theta)$");

	# Compute the estimators for a few different chain lengths
	N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
	gw2010 = np.empty(len(N))
	new = np.empty(len(N))
	for i, n in enumerate(N):
	    gw2010[i] = autocorr_gw2010(chain[:, :n])
	    new[i] = autocorr_new(chain[:, :n])

	# Plot the comparisons
	plt.figure()
	plt.loglog(N, gw2010, "o-", label="G\&W 2010")
	plt.loglog(N, new, "o-", label="new")
	ylim = plt.gca().get_ylim()
	plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
	plt.ylim(ylim)
	plt.xlabel("number of samples, $N$")
	plt.ylabel(r"$\tau$ estimates")
	plt.legend(fontsize=14);