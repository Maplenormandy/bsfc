''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This is the main script containing major classes for spectral fitting. 
Refer to the description of individual classes and functions for a description
of their functionalities. 

'''

import numpy as np
from numpy.polynomial.hermite_e import hermeval, hermemulx
import scipy.optimize as op
from collections import namedtuple
import pdb
import multiprocessing
import cPickle as pkl
import itertools
import time as time_
import os
import sys
import warnings
import matplotlib.pyplot as plt
plt.ion()

# packages that are specific to tokamak fits:
import MDSplus

# make it possible to use other packages within the BSFC distribution:
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))
from helpers import bsfc_helper
from helpers import bsfc_autocorr
from helpers.simplex_sampling import hypercubeToSimplex, hypercubeToHermiteSampleFunction

# packages that require extra installation/care:
import emcee
import gptools

try:
    # this is only necessary on engaging, where jcwright's pymultinest
    # version is normally loaded by default
    sys.path.insert(0,'/home/sciortino/usr/pythonmodules/PyMultiNest')
    import pymultinest
except:
    # assume that user has pymultinest in PYTHONPATH
    import pymultinest


# counter:
counter = 0

# %%
class LineModel:
    """
    Models a spectra. Uses 2nd order Legendre fitting on background noise,
    and n-th order Hermite on the lines
    """
    def __init__(self, lam, lamNorm, specBr, sig, lineData, linesFit, hermFuncs):
        self.lam = lam
        self.specBr = specBr
        self.sig = sig
        self.lineData = lineData
        # It's assumed that the primary line is at index 0 of linesFit
        self.linesFit = linesFit

        self.linesLam = self.lineData.lam[self.linesFit]
        self.linesnames = self.lineData.names[self.linesFit]
        self.linesSqrtMRatio = self.lineData.sqrt_m_ratio[self.linesFit]

        # Normalized lambda, for evaluating background noise
        self.lamNorm = lamNorm

        # Get the edge of the lambda bins, for integrating over finite pixels
        lamEdge = np.zeros(len(lam)+1)
        lamEdge[1:-1] = (lam[1:] + lam[:-1]) / 2
        lamEdge[-1] = 2 * lamEdge[-2] - lamEdge[-3]
        lamEdge[0] = 2 * lamEdge[1] - lamEdge[2]
        self.lamEdge = lamEdge

        self.noiseFuncs = 1

        self.nfit = len(linesFit)
        # Number of hermite polynomials to use for each line, 1 being purely Gaussian
        if hermFuncs == None:
            self.hermFuncs = [3] * self.nfit
        else:
            self.hermFuncs = hermFuncs
        #self.hermFuncs[0] = 9

    """
    Helper functions for theta (i.e. the model parameters).
    Definition of theta is here!
    """
    def thetaLength(self):
        return self.noiseFuncs+2+np.sum(self.hermFuncs)

    def unpackTheta(self, theta):
        # 2nd order Legendre noise
        noise = theta[0:self.noiseFuncs]

        # Extract center and scale, one for each line to fit
        center = theta[self.noiseFuncs]*1e-4 + self.linesLam
        scale = (theta[self.noiseFuncs+1]/self.linesSqrtMRatio)*1e-4

        # Ragged array of hermite function coefficients
        herm = [None]*self.nfit
        cind = self.noiseFuncs+2
        for i in range(self.nfit):
            herm[i] = theta[cind:cind+self.hermFuncs[i]]
            cind = cind + self.hermFuncs[i]

        return noise, center, scale, herm

    def thetaLabels(self):
        labels=[]
        # noise labels
        if self.noiseFuncs>=1:
            labels.append('$c_n$')
        elif self.noiseFuncs>=2:
            labels.append('$m_n$')
        elif self.noiseFuncs == 3:
            labels.append('$q_n$')

        # labels for center shift and scale
        labels.append('$\lambda_c$')
        labels.append('$s$')

        # labels for Hermite function coefficients
        for line in range(self.nfit):
            for h in range(self.hermFuncs[line]):
                labels.append('$%s_%d$'%(self.linesnames[line],h))

        return labels


    def hermiteConstraints(self):
        """
        Constraint function helper
        """
        constraints = []

        h0cnstr = lambda theta, n: theta[n]
        # Don't allow functions to grow more than 10% of the original Gaussian
        hncnstr = lambda theta, n, m: theta[n] - np.abs(10*theta[n+m])

        cind = self.noiseFuncs+2
        for i in range(self.nfit):
            for j in range(self.hermFuncs[i]):
                if j == 0:
                    constraints.append({
                        'type': 'ineq',
                        'fun': h0cnstr,
                        'args': [cind]
                        })
                else:
                    constraints.append({
                        'type': 'ineq',
                        'fun': hncnstr,
                        'args': [cind, j]
                        })

            cind = cind + self.hermFuncs[i]

        return constraints

    """
    Functions for actually producing the predictions from the model.
    """
    def modelPredict(self, theta):
        """
        Full prediction given theta
        """
        noise, center, scale, herm = self.unpackTheta(theta)

        # Shift and scale lambdas to evaluation points
        lamEv = (self.lam[:,np.newaxis]-center)/scale
        lamEdgeEv = (self.lamEdge[:,np.newaxis]-center)/scale

        # Evaluate gaussian functions
        gauss = np.exp(-lamEv**2 / 2)
        gaussEdge = np.exp(-lamEdgeEv**2 / 2)

        hn = np.zeros(lamEv.shape)
        hnEdge = np.zeros(lamEdgeEv.shape)

        # Compute hermite functions to model lineData
        for i in range(self.nfit):
            hn[:,i] = hermeval(lamEv[:,i], herm[i]) * gauss[:,i]
            hnEdge[:,i] = hermeval(lamEdgeEv[:,i], herm[i]) * gaussEdge[:,i]

        # Compute integral over finite pixel size
        # Note: I've found that since lambda is stored as a float32 instead of a float64,
        # the floating point rounding error introduces some unwanted numerical noise if
        # delta-lambda is used here. Thus, I have simply assumed delta-lambda to be approx
        # constant.
        hnEv = (4 * hn + hnEdge[1:] + hnEdge[:-1])/6.0

        # Evaluate noise as 2nd order Legendre fit
        if self.noiseFuncs == 1:
            noiseEv = noise[0]
        elif self.noiseFuncs == 3:
            noiseEv = noise[0] + noise[1]*self.lamNorm + noise[2]*(3*self.lamNorm**2-1)/2

        # Sum over all lineData
        pred = noiseEv + np.sum(hnEv, axis=1)

        return pred

    def modelNoise(self, theta):
        """
        Get only the background noise
        """
        noise, center, scale, herm = self.unpackTheta(theta)

        # Evaluate noise as 2nd order Legendre fit
        if self.noiseFuncs == 1:
            noiseEv = noise[0] * np.ones(self.lamNorm.shape)
        elif self.noiseFuncs == 3:
            noiseEv = noise[0] + noise[1]*self.lamNorm + noise[2]*(3*self.lamNorm**2-1)/2.0
        return noiseEv

    def modelLine(self, theta, line=0, order=-1):
        """
        Get only a single line. Update this to plot a continuous line rather than discretely at
        the same values of wavelengths.
        """
        noise, center, scale, herm = self.unpackTheta(theta)

        # Shift and scale lambdas to evaluation points
        lamEv = (self.lam[:,np.newaxis]-center)/scale
        lamEdgeEv = (self.lamEdge[:,np.newaxis]-center)/scale

        # Evaluate gaussian functions
        gauss = np.exp(-lamEv**2 / 2)
        gaussEdge = np.exp(-lamEdgeEv**2 / 2)

        hn = np.zeros(lamEv.shape)
        hnEdge = np.zeros(lamEdgeEv.shape)

        i = line
        if order > len(herm[i]):
            order = len(herm[i])

        hn[:,i] = hermeval(lamEv[:,i], herm[i]) * gauss[:,i]
        hnEdge[:,i] = hermeval(lamEdgeEv[:,i], herm[i]) * gaussEdge[:,i]

        # Compute integral over finite pixel size
        hnEv = (4 * hn + hnEdge[1:] + hnEdge[:-1])/6.0

        return np.sum(hnEv, axis=1)


    def modelMoments(self, theta, line=0, order=-1):
        """
        Calculate the moments predicted by the model.

        Note, THACO actually calculates the zeroth moment
        """
        noise, center, scale, herm = self.unpackTheta(theta)

        # Since the Probablist's Hermite functions are orthogonal given the unit normal,
        # to integrate the mean and variance just get the weights multiplied by x.
        hermx = hermemulx(herm[line])
        hermxx = hermemulx(hermx)

        normFactor = np.sqrt(2*np.pi)*scale[line]
        m0 = normFactor*herm[line][0]
        m1 = (center[line] - self.linesLam[line])*m0 + normFactor*hermx[0]*scale[line]
        m2 = normFactor*hermxx[0]*scale[line]**2

        return np.array([m0, m1*1e3, m2*1e6])

    def modelMeasurements(self, theta, line=0, order=-1, thaco=True):
        """
        Calculate the counts, v, Ti predicted by the model
        counts in #, v in km/s, Ti in keV.

        Note that THACO doesn't calculate the total counts, and instead uses
        M0 as the A.U. brightness.
        """
        c = 2.998e+5 # speed of light in km/s

        noise, center, scale, herm = self.unpackTheta(theta)

        moments = self.modelMoments(theta, line, order)
        m0 = moments[0]
        # velocity is normalized M1 divided by rest wavelength times c
        # Note that this needs to be projected onto the toroidal component
        v = moments[1]*1e-3/moments[0] / self.linesLam[line] * c
        # width of a 1 kev line = rest wavelength ** 2 / mass in kev
        w = self.linesLam[line]**2 / self.lineData.m_kev[self.linesFit][line]
        ti = moments[2]*1e-6/moments[0] / w
        if thaco:
            counts = m0/scale[line]
        else:
            counts = m0

        return np.array([counts, v, ti])


    """
    Helper functions for initializing fits
    """
    def guessFit(self):
        """
        Returns a theta0 that is the 'zeroth order' guess
        """
        noise0 = np.percentile(self.specBr, 5)
        center = 0.0
        scale = 0.0

        # Ragged array of hermite function coefficients
        herm = [None]*self.nfit
        for i in range(self.nfit):
            herm[i] = np.zeros(self.hermFuncs[i])
            l0 = np.searchsorted(self.lam, self.linesLam[i])

            if i == 0:
                lamFit = self.lam[l0-4:l0+5]
                specFit = self.specBr[l0-4:l0+5]-noise0

                center = np.average(lamFit, weights=specFit)
                scale = np.sqrt(np.average((lamFit-center)**2, weights=specFit))*1e4

            herm[i][0] = np.max(self.specBr[l0]-noise0, 0)

        hermflat = np.concatenate(herm)
        if self.noiseFuncs == 3:
            thetafirst = np.array([noise0, 0.0, 0.0, center, scale])
        elif self.noiseFuncs == 1:
            thetafirst = np.array([noise0, center, scale])

        return np.concatenate((thetafirst, hermflat))


    def copyFit(self, oldLineFit, oldTheta):
        """ Copies over an old fit; the new fit must completely subsume the old fit """
        thetafirst = oldTheta[0:self.noiseFuncs+2]

        cind = self.noiseFuncs+2
        herm = [None]*self.nfit
        for i in range(self.nfit):
            herm[i] = np.zeros(self.hermFuncs[i])

            if i < oldLineFit.nfit:
                herm[i][:oldLineFit.hermFuncs[i]] = oldTheta[cind:cind+oldLineFit.hermFuncs[i]]
                cind = cind + oldLineFit.hermFuncs[i]
            else:
                l0 = np.searchsorted(self.lam, self.linesLam)
                herm[0] = np.max(self.specBr[l0]-oldTheta[0], 0)

        hermflat = np.concatenate(herm)
        return np.concatenate((thetafirst, hermflat))


    # ==============================================
    #                             Likelihood and prior functions
    # ==============================================

    def lnlike(self, theta):
        '''
        Log-likelihood. This is the basic function called by nonlinear optimizers as well as all
        sampling methods that we apply.
        '''
        # save current theta at every evaluation of the log-likelihood
        self.current_theta = theta

        # get chi2 by comparing model prediction and experimental signals
        pred = self.modelPredict(theta)
        return -np.sum((self.specBr-pred)**2/self.sig**2)



    def lnprior(self, theta):
        '''
        Log-prior. This is the simplest version of this function, used by emcee's Ensemble Sampler.
        This function sets correlations between Hermite polynomial coefficients.

        '''
        # unpack parameters:
        noise, center, scale, herm = self.unpackTheta(theta)
        herm0 = np.array([h[0] for h in herm])
        herm1= np.array([h[1] for h in herm])
        herm2 = np.array([h[2] for h in herm])

        # correlated prior:
        if (np.all(noise[0]>0) and np.all(scale>0) and np.all(herm0>0) and
            np.all(herm0-8*np.abs(herm1)>0) and np.all(herm0-8*np.abs(herm2)>0)):
            return 0.0
        else:
            return -np.inf


    def lnprob(self, theta):
        ''' Posterior probability. This simply sums the log-likelihood and the log-prior.
        This function is used by MCMC methods that do not need to distinguish between
        the prior and the likelihood, but only deal with a single probability distribution (the posterior).
        For example, application of a M-H algorithm, the Enseble Sampling MCMC implementation in emcee
        require this function. A nonlinear optimizer can also work with this function, but it's simpler to
        just optimize the log-likelihood and set constraints (which might be better handled by the specific
        algorithm at hand.
        '''

        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp+self.lnlike(theta)



    def get_NS_prior(self):
        '''
        Construct a prior using gptools, which has methods to transform probability
        distributions from the unit hypercube to the real parameter space using inverse
        CDF's. Offers capabilities to use uniform, Gaussian and Gamma prior distributions.

        This function is used to interact with Multinest.

        Using this method, correlations between parameters can be imposed in the
        multinest prior
        '''
        prior = gptools.UniformJointPrior(
            [(0, 1e3)]* self.noiseFuncs  +  # noise must be positive (large upper bound?)
            [(-1e2, 1e2)]  + # wavelength must be positive (large upper bound?)
            [(0, 1e2)]  # scale must be positive (large upper bound?)
            )

        for i in range(self.nfit):
            prior = prior * gptools.UniformJointPrior([(0, 1e5)] )
            for j in range(1,self.hermFuncs[i]): #loop over all Hermite coeffs (only j=1,2,3 normally)
                prior = prior * gptools.UniformJointPrior([(-1e4, 1e4)] )

        return prior


    def hypercube_lnprior(self, cube, ndim, nparams):
        """Prior distribution function for :py:mod:`pymultinest`.
        ! ==========================================
        This version attempts to set correlations between parameters in the prior.
        ! ==========================================

        Maps the (free) parameters in `cube` from [0, 1] to real space.
        This is necessary because MultiNest only works in a unit hypercube.

        Do NOT attempt to change function arguments, even if they are not used!
        This function signature is internally required in Fortran by MultiNest.

        Parameters
        ----------
        cube : array of float, (`num_free_params`,)
            The variables in the unit hypercube.
        ndim : int
            The number of dimensions (meaningful length of `cube`).
        nparams : int
            The number of parameters (length of `cube`).
        """
        #u = self.get_NS_prior().elementwise_cdf(self.current_theta)
       # u[~self.fixed_params] = cube[:ndim]

        global counter
        counter+=1
        if counter%10000==0:
            print counter, " calls to hypercube_lnprior"

        #pdb.set_trace()

        p = self.get_NS_prior().sample_u(cube[:ndim]) # sampled parameters

        ''' # Simple option:
        for k in range(0, ndim):
            cube[k] = p[k]
        '''

        # prior for noise, center and scale parameters:
        for k in range(self.noiseFuncs+2):
            cube[k] = p[k]

        #herm = [None]*self.nfit
        cind = self.noiseFuncs+2

        noise, center, scale, herm = self.unpackTheta(p)

        # loop over number of spectral lines:
        for i in range(self.nfit):

            # 0th Hermite polynomial coefficient:
            herm0 =p[cind]
            cube[cind] =  p[cind]

            #loop over higher order Hermite coeffs (only j=2 and j=3 normally)
            for j in range(1, self.hermFuncs[i]):

                # force higher order Hermite coeffs to be at least 8 times smaller than the 0th coeff
                if (herm0 - 8 * p[cind+j]) >0 :
                    cube[cind+j]  = p[cind+j]
                else:
                    cube[cind+j] = - np.infty  # if condition is not met, return ln-prior=-np.infty

            # increase count by number of Hermite polynomials considered.
            cind = cind + self.hermFuncs[i]


    # Routines for multinest:
    def hypercube_lnprior_simple(self, cube, ndim, nparams):
        """Prior distribution function for :py:mod:`pymultinest`.
        This version does NOT set any correlations between parameters in the prior.

        Maps the (free) parameters in `cube` from [0, 1] to real space.
        This is necessary because MultiNest only works in a unit hypercube.

        Do NOT attempt to change function arguments, even if they are not used!
        This function signature is internally required in Fortran by MultiNest.

        Parameters
        ----------
        cube : array of float, (`num_free_params`,)
            The variables in the unit hypercube.
        ndim : int
            The number of dimensions (meaningful length of `cube`).
        nparams : int
            The number of parameters (length of `cube`).
        """

        # noise:
        for kk in range(self.noiseFuncs):
            cube[kk] = cube[kk] * 1e3 # noise must be positive

        # center wavelength (in 1e-4 A units)
        cube[self.noiseFuncs] = cube[self.noiseFuncs]*2e2 - 1e2 # wavelength must be positive

        # scale (in 1e-4 A units)
        cube[self.noiseFuncs+1] = cube[self.noiseFuncs+1]*1e2 # scale must be positive

        # Hermite coefficients:
        herm = [None]*self.nfit
        cind = self.noiseFuncs+2

        # loop over number of spectral lines:
        for i in range(self.nfit):
            cube[cind]  = cube[cind] *1e5
            #loop over other Hermite coeffs (only j=2,3 normally)
            for j in range(1, self.hermFuncs[i]):
                cube[cind+j]  = cube[cind+j] *2e4 - 1e4  # Hermite coeff must be +ve (large upper bound?)

            # increase count by number of Hermite polynomials considered.
            cind = cind + self.hermFuncs[i]


    def hypercube_lnprior_simplex(self,cube, ndim, nparams):
        """Prior distribution function for :py:mod:`pymultinest`.
        This version does NOT set any correlations between parameters in the prior.

        Maps the (free) parameters in `cube` from [0, 1] to real space.
        This is necessary because MultiNest only works in a unit hypercube.

        Do NOT attempt to change function arguments, even if they are not used!
        This function signature is internally required in Fortran by MultiNest.

        Parameters
        ----------
        cube : array of float, (`num_free_params`,)
            The variables in the unit hypercube.
        ndim : int
            The number of dimensions (meaningful length of `cube`).
        nparams : int
            The number of parameters (length of `cube`).
        """


        # set simplex limits so that a1 and a2 are 1/8 of a0 at most
        # a0 is set to be >0 and smaller than 1e5
        f_simplex = hypercubeToHermiteSampleFunction(1e5, 0.125, 0.125)

        # noise:
        for kk in range(self.noiseFuncs):
            cube[kk] = cube[kk] * 1e3 # noise must be positive

        # center wavelength (in 1e-4 A units)
        cube[self.noiseFuncs] = cube[self.noiseFuncs]*2e2 - 1e2 # wavelength must be positive

        # scale (in 1e-4 A units)
        cube[self.noiseFuncs+1] = cube[self.noiseFuncs+1]*1e2 # scale must be positive

        # Hermite coefficients:
        herm = [None]*self.nfit
        cind = self.noiseFuncs+2

        # loop over number of spectral lines:
        for i in range(self.nfit):

            # map hypercube to constrained simplex:
            [cube[cind], cube[cind+1],cube[cind+2]] = f_simplex([cube[cind],cube[cind+1], cube[cind+2] ])

            # increase count by number of Hermite polynomials considered.
            cind = cind + self.hermFuncs[i]



    def hypercube_lnlike(self, cube, ndim, nparams, lnew):
        """Log-likelihood function for py:mod:`pymultinest`.

        Do NOT attempt to change function arguments, even if they are not used!
        This function signature is internally required by MultiNest.

        Parameters
        ----------
        cube : array of float, (`num_free_params`,)
            The free parameters.
        ndim : int
            The number of dimensions (meaningful length of `cube`).
        nparams : int
            The number of parameters (length of `cube`).
        lnew : float
            New log-likelihood. Probably just there for FORTRAN compatibility?
        """

        ll = -np.inf

        try:
            # parameters are in the hypercube space defined by multinest_lnprior
            theta = [cube[i] for i in range(0, ndim)]

            pred = self.modelPredict(theta)
            ll = -np.sum((self.specBr-pred)**2/self.sig**2)

        except:
            warnings.warn("Log-Likelihood evaluation failed in MultiNest!")

        return ll

# =====================================================


class _LnPost_Wrapper(object):
    """wrapper for log-posterior evaluation in parallel emcee.
    This is needed since instance methods are not pickeable.
    """
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, theta):
        out = self.bf.lineModel.lnprob(theta)

        return out

class _LnLike_Wrapper(object):
    """wrapper for log-likelihood evaluation in parallel emcee.
    This is needed since instance methods are not pickeable.
    """
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, theta):
        out = self.bf.lineModel.lnlike(theta)

        return out

class _LnPrior_Wrapper(object):
    """wrapper for log-prior evaluation in parallel emcee.
    This is needed since instance methods are not pickeable.
    """
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, theta):
        out = self.bf.lineModel.lnprior(theta)

        return out


# =====================================================

class BinFit:
    """
    Performs a nonlinear fit and MCMC error estimate of given binned data
    """
    def __init__(self, lam, specBr, sig, lineData, linesFit):
        self.lam = lam
        self.specBr = specBr
        self.sig = sig
        self.lineData = lineData
        self.linesFit = linesFit

        # Normalized lambda, for evaluating noise
        self.lamNorm = (lam-np.average(lam))/(np.max(lam)-np.min(lam))*2

        # ML is the maximum likelihood theta
        self.result_ml = None
        self.theta_ml = None

        # self.chain = None
        # self.sampler = None
        self.samples = None

        self.good = False

        hermFuncs = [3]*len(linesFit)
        hermFuncs[0] = 3

        self.lineModel = LineModel(lam, self.lamNorm, specBr, sig, lineData, linesFit, hermFuncs)


    def optimizeFit(self, theta0):
        ''' Function to obtain non-linear optimization of log-likelihood. Constraints are added on the Hermite
        polynomial coefficients (see self.lineModel.hermiteConstraints method). The minimizer defaults to
        the COBYLA algorithm implementation in scipy.optimize (I think).

        Parameters
        ----------
        theta0 : array, optional
            guesses for all parameters, given to the optimizer.

        '''
        nll = lambda *args: -self.lineModel.lnlike(*args)

        constraints = self.lineModel.hermiteConstraints()
        result_ml = op.minimize(nll, theta0, tol=1e-6, constraints = constraints)

        return result_ml


    def mcmcSample(self, theta_ml, emcee_threads, nsteps=1000, PT=True, ntemps=5, thin=1, burn=1000):
        ''' Helper function to do MCMC sampling. This uses the emcee implementations of Ensemble
        Sampling MCMC or Parallel-Tempering MCMC (PT-MCMC). In the latter case, a number of
        ``temperatures'' must be defined. These are used to modify the likelihood in a way that
        makes exploration of multimodal distributions easier. PT-MCMC also allows an estimation of
        the model log-evidence, although to lower accuracy than nested sampling.

        theta_ml : array
            Initial parameters to start MCMC exploration from.
        nsteps : int, optional
            Number of steps required from MCMC. Default is 1000. The number of steps is rounded to a multiple
            of the ``thin'' factor.
        PT : bool, optional
            Boolean used to activate Parallel Tempering. With this option, an evaluation of log-evidence
            is possible, but is expected to be less accurate than in nested sampling.
        emcee_threads : int, optional
            Number of threads used within emcee. Parallelization is possible only within a single machine/node.
        ntemps : int, optional
            Number of temperatures used in PT-MCMC. This is a rather arbitrary number. Note that in emcee
            the temperature ladder is implemented such that each temperature is larger by a factor of \sqrt{2}.
        burn : int, optional
            Burn-in of chains. Default is 1000
        '''

        # round number of steps to multiple of thinning factor:
        nsteps = nsteps - nsteps%thin

        if PT: print "Using PT-MCMC!"
        ndim, nwalkers = len(theta_ml), len(theta_ml)*4

        if PT == False:
            # pos has shape (nwalkers, ndim)
            pos = [theta_ml + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

            # use affine-invariant Ensemble Sampling
            sampler = emcee.EnsembleSampler(nwalkers, ndim, _LnPost_Wrapper(self), threads=emcee_threads) # self.lineModel.lnprob
            # get MCMC samples, adding 'burn' steps which will be burnt later
            sampler.run_mcmc(pos, nsteps+burn)

            # flatten chain (but keep ndim)
            samples = sampler.chain[:,burn::thin, :].reshape((-1, ndim))
        else:
            # pos has shape (ntemps, nwalkers, ndim)
            pos = [[theta_ml + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)] for t in range(ntemps)]

            # use PT-MCMC
            sampler = emcee.PTSampler(ntemps, nwalkers, ndim, _LnLike_Wrapper(self), _LnPrior_Wrapper(self), threads=emcee_threads)

            # burn-in 'burn' iterations
            for p, lnprob, lnlike in sampler.sample(pos, iterations=burn):
                pass
            sampler.reset()

            # now sample (and thin by a factor of `thin`):
            for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob, lnlike0=lnlike, iterations=nsteps, thin=thin):
                pass

            # Keep only samples corresponding to T=0 chains
            samples = sampler.chain[0,:,:,:].reshape((-1, ndim))

        # sanity check
        if PT==True:
            assert sampler.chain.shape == (ntemps, nwalkers, nsteps//thin, ndim)
            assert samples.shape == ( nwalkers * nsteps//thin, ndim)
        else:
            assert sampler.chain.shape == (nwalkers, burn+nsteps, ndim)
            assert samples.shape == (nwalkers * nsteps//thin, ndim)

        return samples, sampler

    def MCMCfit(self, mcmc=True, PT=False, nsteps=10000, plot_convergence=False, emcee_threads=1):
        ''' Method to fit a line model. If mcmc==True, then Ensamble Sampler MCMC or Parallel-Tempering
        MCMC (PT-MCMC) are applied. If PT=True, PT-MCMC is used.

        Convergence plots can be produced to evaluate whether the number of MCMC steps was sufficient
        for effective chain mixing. EMCEE runs are parallelized on a single machine/node, but are not
        currently MPI compatible.

        Parameters
        ----------
        mcmc : bool, optional
            Activate MCMC sampling. If set to False, the result of nonlinear optimizer (from scipy.optimize)
            is used. In this case, the standard deviation of samples (self.m_std) should of course be 0.
        nsteps : int, optional
            Number of MCMC steps. Default if 10k.
        plot_convergence : bool, optional
            Boolean indicating whether the autocorrelation time of MCMC chains should be plotted as a
            means for diagnosing MCMC chain mixing and convergence. This is ineffective if mcmc==False.
        emcee_threads : int, optional
             Number of threads to parallelize EMCEE. This defaults to 1 because it is convenient for us to
             parallelize fits for different channels and times rather than to parallelize chains. However, the
             function is written in a general enough way that it could be adapted in the future to parallelize
             each emcee search.

        '''
        if emcee_threads==None:
            emcee_threads = multiprocessing.cpu_count()

        # get a good guess for the fit parameters for each line amd unpack them
        theta0 = self.lineModel.guessFit()
        noise, center, scale, herm = self.lineModel.unpackTheta(theta0)

        # if amplitude of primary line is less than 10% of the noise, not worth fitting better
        if herm[0][0] < noise[0]*0.05: #0.1 or 0.05?
            self.m0_ml = 0.0
            self.good = False
            return False
        else:
            # obtain maximum likelihood fits
            self.result_ml = self.optimizeFit(theta0)
            self.theta_ml = self.result_ml['x']

            # if MCMC is requested, use result from non-linear optimization as an MCMC start
            if mcmc:
                self.samples, sampler = self.mcmcSample(self.theta_ml, emcee_threads, nsteps=nsteps, PT=PT)
                #sampler = self.mcmcSample(self.theta_ml, emcee_threads, nsteps=nsteps, PT=PT)
                #self.samples = sampler.chain
            else:
                # if MCMC is not requested, do nothing...? (we might want to make this smarter)
                self.samples = np.array([self.theta_ml]*50)

            if plot_convergence and mcmc:
                # use external scripts to plot chains' autocorrelation function
                bsfc_autocorr.plot_convergence(sampler.chain, dim=1, nsteps=nsteps)

            if PT:
                # get log-evidence estimate and uncertainty
                self.lnev = sampler.thermodynamic_integration_log_evidence()
            else:
                self.lnev = None

            # find moments of line fits from the samples/chains obtained above
            self.m_samples = np.apply_along_axis(self.lineModel.modelMoments, axis=1, arr=self.samples)

            # save moments obtained from maximum likelihood optimization
            self.m_ml = self.lineModel.modelMoments(self.theta_ml)

            self.theta_avg = np.average(self.samples, axis=0)
            self.m_avg = np.average(self.m_samples, axis=0)
            self.m_std = np.std(self.m_samples, axis=0) #, ddof=len(theta0))

            self.good = True
            return True


    def NSfit(self, lnev_tol=0.5, n_live_points=100, sampling_efficiency=0.8, INS=True, const_eff=True, basename=None):
        ''' Fit with Nested Sampling (MultiNest algorithm).
        For Nested Sampling, the prior and likelihood are not simply combined into a posterior
        (which is the only function passed to EMCEE), but they are used differently to explore the
        probability space and get an estimated for the log-evidence.

         Parameters
        ----------
        lnev_tol : bool, optional
            Tolerance in the log-evidence. This sets the termination condition for the nested sampling
            algorithm. Default is 0.5.
        n_live_points : int, optional
            Number of live points. Default of 100.
        sampling_efficiency : float, optional
            Sets the enlargement factor for ellipsoidal fits in MultiNest. Default is 0.3 (appropriate for
            model selection).
        INS : bool, optional
            Setting to activate Importance Nested Sampling in MultiNest. Refer to Feroz et al. 2014.
        basename : str, optional
        '''
        # obtain maximum likelihood fits
        theta0 = self.lineModel.guessFit()
        self.result_ml = self.optimizeFit(theta0)
        self.theta_ml = self.result_ml['x']

        # save moments obtained from maximum likelihood optimization
        self.m_ml = self.lineModel.modelMoments(self.theta_ml)

        # dimensionality of the problem
        ndim = self.lineModel.thetaLength()

        print("Basename = ", basename)

        pymultinest.run(
            self.lineModel.hypercube_lnlike,   # log-likelihood
            self.lineModel.hypercube_lnprior_simplex, #self.lineModel.hypercube_lnprior_simple,   # log-prior
            ndim,
            outputfiles_basename=basename,
            n_live_points=n_live_points,
            n_params = None, # defaults to same as ndim
            n_clustering_params = None, # defaults to same as ndim
            wrapped_params = None,
            importance_nested_sampling = INS,
            multimodal = True,
            const_efficiency_mode = const_eff, # only appropriate with INS
            evidence_tolerance = lnev_tol,
            sampling_efficiency = sampling_efficiency,
            n_iter_before_update = 100, #MultiNest internally multiplies by 10
            null_log_evidence = -1e90,
            max_modes = 100,
            mode_tolerance = -1e90,  #keeps all modes
            seed = -1,
            verbose = True,
            resume = True,
            context = 0,   # additional info by user (leave empty)
            write_output = True,
            log_zero = -1e99,
            max_iter = 0,   #unlimited
            dump_callback = None
        )

        self.good=True

        # after MultiNest run, read results
        a = pymultinest.Analyzer(
            n_params=ndim,
            outputfiles_basename=basename
        )

        # get chains and weights
        data = a.get_data()

        self.samples = data[:,2:]
        self.sample_weights = data[:,0]
        self.sample_n2ll = data[:,1]

        # save statistics
        stats = a.get_stats()
        self.multinest_stats = stats

        self.modes=stats['modes'][0]
        self.maximum= self.modes['maximum']
        self.maximum_a_posterior= self.modes['maximum a posterior']
        self.mean=np.asarray(self.modes['mean'])
        self.sigma=np.asarray(self.modes['sigma'])

        # get log-evidence estimate and uncertainty
        self.lnev = (stats['global evidence'], stats['global evidence error'])

        f = gptools.plot_sampler(
            data[:, 2:], # index 0 is weights, index 1 is -2*loglikelihood, then samples
            weights=data[:, 0],
            labels=self.lineModel.thetaLabels(),
            chain_alpha=1.0,
            cutoff_weight=0.01,
            cmap='plasma',
            #suptitle='Posterior distribution of $D$ and $V$',
            plot_samples=False,
            plot_chains=False,
            #xticklabel_angle=120,
            #ticklabel_fontsize=15,
        )

        g=gptools.summarize_sampler(data[:, 2:],
                                    weights=data[:, 0],
                                    burn=0,
                                    ci=0.95, chain_mask=None)
        self.params_mean = np.asarray(g[0])
        self.params_ci_l = np.asarray(g[1])
        self.params_ci_u = np.asarray(g[2])

        # summarize results
        self.m_map = self.lineModel.modelMoments(self.maximum_a_posterior)
        self.m_map_mean = self.lineModel.modelMoments(self.mean)
        m1 = self.lineModel.modelMoments(self.mean+self.sigma)
        m2 = self.lineModel.modelMoments(self.mean - self.sigma)
        self.m_map_std = (m1 - m2)/2.0   #temporary

        # marginalized (fully-Bayesian) results:
        self.m_bayes_marg = self.lineModel.modelMoments(self.params_mean)
        self.m_bayes_marg_low = self.lineModel.modelMoments(self.params_ci_l)
        self.m_bayes_marg_up = self.lineModel.modelMoments(self.params_ci_u)

        # temporary, for compatibility with MCMC methods:
        self.theta_avg = self.params_mean
        self.m_avg = self.m_bayes_marg
        self.m_std = self.m_map_std

        return True





    def NS_analyze(self):
        ''' Function to analyze Nested Sampling chains.


        '''
        # after MultiNest run, read results
        a = pymultinest.Analyzer(
            n_params=ndim,
            outputfiles_basename=self.MN_basename
        )

        # get chains and weights
        data = a.get_data()
        # save statistics
        s = a.get_stats()
        self.multinest_stats = s
        self.m_lnev = (s['global evidence'], s['global evidence error'])

        # get inferred parameters:
        for p, m in zip(parameters, s['marginals']):
            lo, hi = m['1sigma']
            med = m['median']
            sigma = (hi - lo) / 2

        print('creating marginal plot ...')
        p = pymultinest.PlotMarginal(a)

        values = a.get_equal_weighted_posterior()
        assert n_params == len(s['marginals'])
        modes = s['modes']

        # more plots
        dim2 = os.environ.get('D', '1' if n_params > 20 else '2') == '2'
        nbins = 100 if n_params < 3 else 20
        if dim2:
            plt.figure(figsize=(5*n_params, 5*n_params))
            for i in range(n_params):
                plt.subplot(n_params, n_params, i + 1)
                plt.xlabel(parameters[i])

                m = s['marginals'][i]
                plt.xlim(m['5sigma'])

                oldax = plt.gca()
                x,w,patches = oldax.hist(values[:,i], bins=nbins, edgecolor='grey', color='grey', histtype='stepfilled', alpha=0.2)
                oldax.set_ylim(0, x.max())

                newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
                p.plot_marginal(i, ls='-', color='blue', linewidth=3)
                newax.set_ylim(0, 1)

                ylim = newax.get_ylim()
                y = ylim[0] + 0.05*(ylim[1] - ylim[0])
                center = m['median']
                low1, high1 = m['1sigma']
                #print(center, low1, high1)
                newax.errorbar(x=center, y=y,
                        xerr=numpy.transpose([[center - low1, high1 - center]]),
                        color='blue', linewidth=2, marker='s')
                oldax.set_yticks([])
                #newax.set_yticks([])
                newax.set_ylabel("Probability")
                ylim = oldax.get_ylim()
                newax.set_xlim(m['5sigma'])
                oldax.set_xlim(m['5sigma'])
                #plt.close()

                for j in range(i):
                        plt.subplot(n_params, n_params, n_params * (j + 1) + i + 1)
                        p.plot_conditional(i, j, bins=20, cmap = plt.cm.gray_r)
                        for m in modes:
                                plt.errorbar(x=m['mean'][i], y=m['mean'][j], xerr=m['sigma'][i], yerr=m['sigma'][j])
                        plt.xlabel(parameters[i])
                        plt.ylabel(parameters[j])

        else:
            from matplotlib.backends.backend_pdf import PdfPages
            sys.stderr.write('1dimensional only. Set the D environment variable \n')
            sys.stderr.write('to D=2 to force 2d marginal plots.\n')

            for i in range(n_params):
                plt.figure(figsize=(3, 3))
                plt.xlabel(parameters[i])
                plt.locator_params(nbins=5)

                m = s['marginals'][i]
                iqr = m['q99%'] - m['q01%']
                xlim = m['q01%'] - 0.3 * iqr, m['q99%'] + 0.3 * iqr
                #xlim = m['5sigma']
                plt.xlim(xlim)

                oldax = plt.gca()
                x,w,patches = oldax.hist(values[:,i], bins=numpy.linspace(xlim[0], xlim[1], 20), edgecolor='grey', color='grey', histtype='stepfilled', alpha=0.2)
                oldax.set_ylim(0, x.max())

                newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
                p.plot_marginal(i, ls='-', color='blue', linewidth=3)
                newax.set_ylim(0, 1)

                ylim = newax.get_ylim()
                y = ylim[0] + 0.05*(ylim[1] - ylim[0])
                center = m['median']
                low1, high1 = m['1sigma']
                #print center, low1, high1
                newax.errorbar(x=center, y=y,
                        xerr=numpy.transpose([[center - low1, high1 - center]]),
                        color='blue', linewidth=2, marker='s')
                oldax.set_yticks([])
                newax.set_ylabel("Probability")
                ylim = oldax.get_ylim()
                newax.set_xlim(xlim)
                oldax.set_xlim(xlim)

        return True





# =====================================================
# %%
# =====================================================




class _TimeBinFitWrapper(object):
    """ Wrapper to support parallelization of different channels in a
    specific time bin. This is needed since instance methods are not pickeable.

    """
    def __init__(self, mf, nsteps, tbin):
        self.mf = mf
        self.nsteps = nsteps
        self.tbin = tbin

    def __call__(self, chbin):

        w0, w1 = np.searchsorted(self.mf.lam_all[:,self.tbin,chbin], self.mf.lam_bounds)
        lam = self.mf.lam_all[w0:w1,self.tbin,chbin]
        specBr = self.mf.specBr_all[w0:w1,self.tbin,chbin]
        sig = self.mf.sig_all[w0:w1,self.tbin,chbin]

        #
        bf = BinFit(lam, specBr, sig, self.mf.lines, range(len(self.mf.lines.names)))

        print "Now fitting tbin =", self.tbin, ',chbin =', chbin, "with nsteps =", self.nsteps
        good = bf.MCMCfit(nsteps=self.nsteps)
        if not good:
            print "not worth fitting"

        return bf



class _ChBinFitWrapper(object):
    """ Wrapper to support parallelization of different time bins in a
    specific channel bin. This is needed since instance methods are not pickeable.

    """
    def __init__(self, mf, nsteps, chbin):
        self.mf = mf
        self.nsteps = nsteps
        self.chbin = chbin

    def __call__(self, tbin):

        w0, w1 = np.searchsorted(self.mf.lam_all[:,tbin,self.chbin], self.mf.lam_bounds)
        lam = self.mf.lam_all[w0:w1,tbin,self.chbin]
        specBr = self.mf.specBr_all[w0:w1,tbin,self.chbin]
        sig = self.mf.sig_all[w0:w1,tbin,self.chbin]

        # create bin-fit
        bf = BinFit(lam, specBr, sig, self.mf.lines, range(len(self.mf.lines.names)))

        print "Now fitting tbin=", tbin, ',chbin=', self.chbin, "with nsteps=", self.nsteps
        good = bf.MCMCfit(nsteps=self.nsteps)
        if not good:
            print "not worth fitting"

        return bf


class _fitTimeWindowWrapper(object):
    """ Wrapper to support parallelization of different time bins in a
    specific channel bin. This is needed since instance methods are not pickeable.

    """
    def __init__(self, mf, nsteps):
        self.mf = mf
        self.nsteps = nsteps

    def __call__(self, bins):
        tbin, chbin = bins

        w0, w1 = np.searchsorted(self.mf.lam_all[:,tbin,chbin], self.mf.lam_bounds)
        lam = self.mf.lam_all[w0:w1,tbin,chbin]
        specBr = self.mf.specBr_all[w0:w1,tbin,chbin]
        sig = self.mf.sig_all[w0:w1,tbin,chbin]

        # create bin-fit
        bf = BinFit(lam, specBr, sig, self.mf.lines, range(len(self.mf.lines.names)))

        print "Now fitting tbin=", tbin, ', chbin=', chbin, " with nsteps=", self.nsteps
        try:
            good = bf.MCMCfit(nsteps=self.nsteps)
        except ValueError:
            print "BinFit.fit() failed."
            print "++++++++++++++++++++++++++++++++"
            print "Failed at fitting tbin=", tbin, ', chbin=', chbin, " with nsteps=", self.nsteps
            print "++++++++++++++++++++++++++++++++"
            good = False
        if not good:
            print "Fitting not available. Result will be None."

        return bf


# =====================================================


# %%

LineInfo = namedtuple('LineInfo', 'lam m_kev names symbol z sqrt_m_ratio'.split())

class MomentFitter:
    def __init__(self, primary_impurity, primary_line, shot, tht, lam_bounds = None, experiment='CMOD', instrument='Hirex-Sr'):
        ''' Class to store experimental data and inferred spectral fits.

        Parameters:
        primary_impurity:
        primary_line:
        shot:
        tht:
        lam_bounds:
        experiment: {'CMOD','D3D',...}
                Experimental device of interest. Only pre-defined choices for which data fetching is made available
                are acceptable inputs. Default is 'CMOD'.
        instrument: {'Hirex-Sr','XEUS','LOWEUS', 'CER', ...}
                Instrument/diagnostic for which spectral data should be fitted. Note that said instrument must be
                available for the experiment given above. Default is 'Hirex-Sr'.

        '''
        self.lines = LineInfo(None, None, None, None, None, None)
        self.primary_line = primary_line
        self.tht=tht
        self.shot = shot

        self.experiment = experiment
        if experiment=='CMOD':
            if instrument in ['Hirex-Sr','XEUS','LOWEUS']:
                self.instrument = instrument
            else:
                raise ValueError('%s instrument not available for CMOD!'%str(instrument))
        elif experiment=='D3D':
            if instrument in ['CER','XEUS','LOWEUS']:
                self.instrument = instrument
            else:
                raise ValueError('%s instrument not available for D3D!'%str(instrument))

        if experiment=='CMOD':
            if instrument=='Hirex-Sr':
                self.load_hirex_data(primary_impurity, primary_line, shot, tht, lam_bounds)
            else:
                raise ValueError('Instruments other than Hirex-Sr not yet implemented for CMOD!')
        elif  experiment=='D3D':
            if instrument=='CER':
                self.load_D3D_cer(primary_impurity, primary_line, shot, tht, lam_bounds)
            else:
                raise ValueError('Instruments other than CER are not yet implemented for D3D!')
        else:
            raise ValueError('Experiments other than CMOD not yet implemented!')

    def load_hirex_data(self, primary_impurity, primary_line, shot, tht, lam_bounds, hirexsr_file='../data/hirexsr_wavelengths.csv'):
        '''
        Function to load Hirex-Sr data for CMOD. Assumes that rest wavelengths, ionization stages and
        atomic line names are given in a file provided as input.
        '''
        self.hirexsr_file = str(hirexsr_file)

        # Load all wavelength data
        with open(hirexsr_file, 'r') as f:
            lineData = [s.strip().split(',') for s in f.readlines()]
            lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
            lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
            lineName = np.array([ld[3] for ld in lineData[2:]])

        amuToKeV = 931494.095 # amu in keV
        #speedOfLight = 2.998e+5 # speed of light in km/s

        # Load atomic data, for calculating line widths, etc...
        with open('atomic_data.csv', 'r') as f:
            atomData = [s.strip().split(',') for s in f.readlines()]
            atomSymbol = np.array([ad[1].strip() for ad in atomData[1:84]])
            atomMass = np.array([float(ad[3]) for ad in atomData[1:84]]) * amuToKeV

        if lam_bounds == None:
            if primary_impurity == 'Ca':
                if primary_line == 'w':
                    lam_bounds = (3.172, 3.188)
                elif primary_line == 'lya1':
                    lam_bounds = (3.010, 3.027)
                else:
                    raise NotImplementedError("Line is not yet implemented")
            elif primary_impurity == 'Ar':
                if primary_line == 'w':
                    lam_bounds = (3.945, 3.954)
                elif primary_line == 'z':
                    raise NotImplementedError("Not implemented yet (needs line tying)")
                elif primary_line == 'lya1':
                    lam_bounds = (3.725, 3.742)
                else:
                    raise NotImplementedError("Line is not yet implemented")

        self.lam_bounds = lam_bounds

        # Populate the line data
        lineInd = np.logical_and(lineLam>lam_bounds[0], lineLam<lam_bounds[1])
        #satelliteLines = np.array(['s' not in l for l in lineName])
        #lineInd = np.logical_and(satelliteLines, lineInd)
        ln = lineName[lineInd]
        ll = lineLam[lineInd]
        lz = lineZ[lineInd]
        lm = atomMass[lz-1]
        ls = atomSymbol[lz-1]

        # Get the index of the primary line
        self.pl = np.where(ln==primary_line)[0][0]

        lr = np.sqrt(lm / lm[self.pl])

        self.lines = LineInfo(ll, lm, ln, ls, lz, lr)

        # Sort lines by distance from primary line
        pl_sorted = np.argsort(np.abs(self.lines.lam-self.lines.lam[self.pl]))
        for data in self.lines:
            data = data[pl_sorted]

        print 'Fitting:', [self.lines.symbol[i] +
                ' ' + self.lines.names[i] + ' @ ' +
                str(self.lines.lam[i]) for i in range(len(self.lines.names))]

        specTree = MDSplus.Tree('spectroscopy', shot)

        ana = '.ANALYSIS'
        if tht > 0:
            ana += str(tht)

        # Determine which, if any, detector has the desired lam_bounds
        rootPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana
        branchB = False
        lamInRange = False
        try:
            branchNode = specTree.getNode(rootPath+'.HELIKE')
            self.lam_all = branchNode.getNode('SPEC:LAM').data()
            if np.any(np.logical_and(self.lam_all>lam_bounds[0], self.lam_all<lam_bounds[1])):
                print "Fitting on Branch A"
                lamInRange = True
                branchB = False
        except:
            pass

        if not lamInRange:
            try:
                branchNode = specTree.getNode(rootPath+'.HLIKE')
                self.lam_all = branchNode.getNode('SPEC:LAM').data()
                if np.any(np.logical_and(self.lam_all>lam_bounds[0], self.lam_all<lam_bounds[1])):
                    print "Fitting on Branch B"
                    lamInRange = True
                    branchB = True
            except:
                pass

        if not lamInRange:
            raise ValueError("Fit range does not appear to be on detector")

        # Indices are [lambda, time, channel]
        self.specBr_all = branchNode.getNode('SPEC:SPECBR').data()
        self.sig_all = branchNode.getNode('SPEC:SIG').data()

        if branchB:
            # Hack for now; usually the POS variable is in LYA1 on branch B
            pos_tmp = branchNode.getNode('MOMENTS.LYA1:POS').data()
        else:
            # Otherwise, load the POS variable as normal
            pos_tmp = branchNode.getNode('MOMENTS.'+primary_line.upper()+':POS').data()

        self.pos=np.squeeze(pos_tmp[np.where(pos_tmp[:,0]!=-1),:])

        # Maximum number of channels, time bins
        self.maxChan = np.max(branchNode.getNode('BINNING:CHMAP').data())+1
        self.maxTime = np.max(branchNode.getNode('BINNING:TMAP').data())+1

        # get time basis
        tmp=np.asarray(branchNode.getNode('SPEC:SIG').dim_of(1))
        mask = [tmp>-1]
        self.time = tmp[mask]

        self.fits = [[None for y in range(self.maxChan)] for x in range(self.maxTime)] #[[None]*self.maxChan]*self.maxTime

    def load_D3D_cer(self, primary_impurity, primary_line, shot, tht, lam_bounds, cer_file='cer_wavelengths.csv'):
        '''
        Function to load CER data for D3D. Assumes that rest wavelengths, ionization stages and
        atomic line names are given in a file provided as input.
        '''
        self.cer_file = str(cer_file)

        # Load all wavelength data
        with open(cer_file, 'r') as f:
            lineData = [s.strip().split(',') for s in f.readlines()]
            lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
            lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
            lineName = np.array([ld[3] for ld in lineData[2:]])

        amuToKeV = 931494.095 # amu in keV

        # Load atomic data, for calculating line widths, etc...
        with open('atomic_data.csv', 'r') as f:
            atomData = [s.strip().split(',') for s in f.readlines()]
            atomSymbol = np.array([ad[1].strip() for ad in atomData[1:84]])
            atomMass = np.array([float(ad[3]) for ad in atomData[1:84]]) * amuToKeV

        if lam_bounds == None:
            if primary_impurity == 'Ca':
                if primary_line == 'w':
                    lam_bounds = (3.172, 3.188)
                elif primary_line == 'lya1':
                    lam_bounds = (3.010, 3.027)
                else:
                    raise NotImplementedError("Line is not yet implemented")
            elif primary_impurity == 'Ar':
                if primary_line == 'w':
                    lam_bounds = (3.945, 3.960)
                elif primary_line == 'z':
                    raise NotImplementedError("Not implemented yet (needs line tying)")
                elif primary_line == 'lya1':
                    lam_bounds = (3.725, 3.742)
                else:
                    raise NotImplementedError("Line is not yet implemented")

        self.lam_bounds = lam_bounds

        # Populate the line data
        lineInd = np.logical_and(lineLam>lam_bounds[0], lineLam<lam_bounds[1])
        #satelliteLines = np.array(['s' not in l for l in lineName])
        #lineInd = np.logical_and(satelliteLines, lineInd)
        ln = lineName[lineInd]
        ll = lineLam[lineInd]
        lz = lineZ[lineInd]
        lm = atomMass[lz-1]
        ls = atomSymbol[lz-1]

        # Get the index of the primary line
        self.pl = np.where(ln==primary_line)[0][0]

        lr = np.sqrt(lm / lm[self.pl])

        self.lines = LineInfo(ll, lm, ln, ls, lz, lr)

        # Sort lines by distance from primary line
        pl_sorted = np.argsort(np.abs(self.lines.lam-self.lines.lam[self.pl]))
        for data in self.lines:
            data = data[pl_sorted]

        print 'Fitting:', [self.lines.symbol[i] +
                ' ' + self.lines.names[i] + ' @ ' +
                str(self.lines.lam[i]) for i in range(len(self.lines.names))]

        # MODIFY for D3D!!!!
        specTree = MDSplus.Tree('spectroscopy', shot)

        ana = '.ANALYSIS'
        if tht > 0:
            ana += str(tht)

        # Determine which, if any, detector has the desired lam_bounds
        rootPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana
        lamInRange = False
        try:
            branchNode = specTree.getNode(rootPath+'.HELIKE')
            self.lam_all = branchNode.getNode('SPEC:LAM').data()
            if np.any(np.logical_and(self.lam_all>lam_bounds[0], self.lam_all<lam_bounds[1])):
                print "Fitting on Branch A"
                lamInRange = True
        except:
            pass

        if not lamInRange:
            try:
                branchNode = specTree.getNode(rootPath+'.HLIKE')
                self.lam_all = branchNode.getNode('SPEC:LAM').data()
                if np.any(np.logical_and(self.lam_all>lam_bounds[0], self.lam_all<lam_bounds[1])):
                    print "Fitting on Branch B"
                    lamInRange = True
            except:
                pass

        if not lamInRange:
            raise ValueError("Fit range does not appear to be on detector")

        # Indices are [lambda, time, channel]
        self.specBr_all = branchNode.getNode('SPEC:SPECBR').data()
        self.sig_all = branchNode.getNode('SPEC:SIG').data()

        pos_tmp = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HLIKE.MOMENTS.LYA1.POS').data()
        self.pos=np.squeeze(pos_tmp[np.where(pos_tmp[:,0]!=-1),:])

        # Maximum number of channels, time bins
        self.maxChan = np.max(branchNode.getNode('BINNING:CHMAP').data())+1
        self.maxTime = np.max(branchNode.getNode('BINNING:TMAP').data())+1

        # get time basis
        tmp=np.asarray(branchNode.getNode('SPEC:SIG').dim_of(1))
        mask = [tmp>-1]
        self.time = tmp[mask]

        self.fits = [[None for y in range(self.maxChan)] for x in range(self.maxTime)] #[[None]*self.maxChan]*self.maxTime

    def fitSingleBin(self, tbin, chbin, nsteps=1024, emcee_threads=1, PT=False, NS=False):
        ''' Basic function to launch fitting methods. If NS==True, this uses Nested Sampling
        with MultiNest. In this case, the number of steps (nsteps) doesn't matter since the
        algorithm runs until meeting a convergence threshold. Parallelization is activated by
        default in MultiNest if MPI libraries are available.

        If NS==False, emcee is used. Either an affine-invariant Ensemble Sampler or
        Parallel-Tempering MCMC are used. Both require specification of a certain number of
        steps and a number of threads for parallelization. It is recommended to keep this to 1
        if the user is already using parallelization to compute multiple spectal images at the same
        time (i.e. avoid double-layer parallelization).

        '''

        self.NS = NS
        w0, w1 = np.searchsorted(self.lam_all[:,tbin,chbin], self.lam_bounds)
        lam = self.lam_all[w0:w1,tbin,chbin]
        specBr = self.specBr_all[w0:w1,tbin,chbin]
        sig = self.sig_all[w0:w1,tbin,chbin]

        bf = BinFit(lam, specBr, sig, self.lines, range(len(self.lines.names)))

        self.fits[tbin][chbin] = bf

        print "Now fitting tbin=", tbin, ', chbin=', chbin, " with nsteps=", nsteps
        if NS==False:
            good = bf.MCMCfit(nsteps=nsteps, emcee_threads=emcee_threads, PT=PT)
        else:
            print "Using Nested Sampling!"
            # create output
            basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/c-.' )
            chains_dir = os.path.dirname(basename)
            if chains_dir and not os.path.exists(chains_dir):
                os.mkdir(chains_dir)

            good = bf.NSfit(lnev_tol= 0.5, n_live_points=400, sampling_efficiency=0.8, INS=True, basename=basename)

        if not good:
            print "not worth fitting"
        else:
            print "--> done"


    def fitTimeBin(self, tbin, parallel=True, nproc=None, nsteps=1024, emcee_threads=1):
        '''
        Fit signals from all channels in a specific time bin.
        Functional parallelization.

        '''
        if parallel:
            if nproc==None:
                nproc = multiprocessing.cpu_count()
            print "running fitTimeBin in parallel with nproc=", nproc
            pool = multiprocessing.Pool(processes=nproc)

            # requires a wrapper for multiprocessing to pickle the function
            ff = _TimeBinFitWrapper(self,nsteps=nsteps, tbin=tbin)

            # map range of channels and compute each
            self.fits[tbin][:] = pool.map(ff, range(self.maxChan))
        else:
            # fit channel bins sequentially
            for chbin in range(self.maxChan):
                # note that emcee multithreads cannot be used with inter-bin fitting multiprocessing
                self.fitSingleBin(tbin, chbin, emcee_threads=emcee_threads)

    def fitChBin(self, chbin, parallel=True, nproc=None, nsteps=1024, emcee_threads=1):
        '''
        Fit signals from all times in a specific channel.
        Functional parallelization.

        '''
        if parallel:
            if nproc==None:
                nproc = multiprocessing.cpu_count()
            print "running fitChBin in parallel with nproc=", nproc
            pool = multiprocessing.Pool(processes=nproc)

            # requires a wrapper for multiprocessing to pickle the function
            ff = _ChBinFitWrapper(self, nsteps=nsteps, chbin=chbin)

            # map range of channels and compute each
            self.fits[:][chbin] = pool.map(ff, range(self.maxTime))

        else:
            # fit time bins sequentially
            for tbin in range(self.maxTime):
                self.fitSingleBin(tbin, chbin, emcee_threads=emcee_threads)


    def fitTimeWindow(self, tidx_min=None, tidx_max=None, parallel=True,
        nproc=None, nsteps=1000, emcee_threads=1):
        '''
        Fit all signals within a time window, across all channels.
        Optional parallelization.

        If tidx_min and tidx_max are not specified (i.e. left as "None"),
        then the routine defaults to compute results across the entire time window of
        Hirex-Sr's measurements.
        '''
        if tidx_min==None:
            tidx_min=0
        if tidx_max==None:
            tidx_max=self.MaxTime

        if parallel:
            if nproc==None:
                nproc = multiprocessing.cpu_count()
            print "running fitTimeWindow in parallel with nproc=", nproc
            pool = multiprocessing.Pool(processes=nproc)

            # requires a wrapper for multiprocessing to pickle the function
            ff = _fitTimeWindowWrapper(self,nsteps=nsteps)

            # map range of channels and compute each
            map_args_tpm = list(itertools.product(range(tidx_min, tidx_max), range(self.maxChan)))
            map_args = [list(a) for a in map_args_tpm]

            # parallel run
            fits_tmp = pool.map(ff, np.asarray(map_args))
            fits = np.asarray(fits_tmp).reshape((tidx_max-tidx_min,self.maxChan))

            # recollect results into default fits structure
            t=0
            for tbin in range(tidx_min, tidx_max):
                self.fits[tbin][:] = fits[t,:]
                t+=1

        else:
            for chbin in range(self.maxChan):
                for tbin in range(tidx_min, tidx_max):
                    self.fitSingleBin(tbin, chbin, emcee_threads=emcee_threads)

    #####
    def plotSingleBinFit(self, tbin, chbin):
        ''' Function designed to plot spectrum from a single time and a single channel bin.
        This allows visualization and comparison of the results of nonlinear optimization,
        MCMC sampling or Nested Sampling.

        '''
        bf = self.fits[tbin][chbin]

        if bf == None:
            return

        f0, (a0, a1) = plt.subplots(2, 1, sharex=True, gridspec_kw = {'height_ratios': [4,1]})
        a0.errorbar(bf.lam, bf.specBr, yerr=bf.sig, c='m', fmt='.')

        if bf.good:
            # color list, one color for each spectral line
            #color=cm.rainbow(np.linspace(0,1,len(self.lines.names)))
            color = ['b','g','c','m','y','k']
            # plot in red the overall reconstructed spectrum (sum of all spectral lines)
            pred = bf.lineModel.modelPredict(bf.theta_ml)
            a0.plot(bf.lam, pred, c='r')

            # plot some samples: noise floor in black, spectral lines all in different colors
            for samp in range(25):
                theta = bf.samples[np.random.randint(len(bf.samples))]
                noise = bf.lineModel.modelNoise(theta)
                a0.plot(bf.lam, noise, c='k', alpha=0.08)

                for i in range(len(self.lines.names)):
                    line = bf.lineModel.modelLine(theta, i)
                    a0.plot(bf.lam, line+noise, c=color[i], alpha=0.08)

            # add average inferred noise
            noise = bf.lineModel.modelNoise(bf.theta_avg)
            a0.plot(bf.lam, noise, c='k', label='Inferred noise')
            a0.set_title('tbin='+str(tbin)+', chbin='+str(chbin))

            # plot all fitted spectral lines, one in each color
            for i in range(len(self.lines.names)):
                line = bf.lineModel.modelLine(bf.theta_avg, i)
                a0.plot(bf.lam, line+noise, c=color[i])

            # on second subplot, plot residuals
            a1.errorbar(bf.lam, bf.specBr - pred, yerr=bf.sig, c='r', fmt='.')
            a1.axhline(c='m', ls='--')

            for i in range(len(self.lines.names)):
                a1.axvline(self.lines.lam[i], c='b', ls='--')
                a0.axvline(self.lines.lam[i], c='b', ls='--')

        plt.show()





# %% =======================================================
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

    return moments_vals, moments_std


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
    #print "Computing brightness, rotation and ion temperature"
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
