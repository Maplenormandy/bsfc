''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This script contains the LineModel class, which define the Hermite polynomial decomposition for each line, model noise, the likelihood function and prior function.

'''
from __future__ import division
from builtins import range
from builtins import object
from past.utils import old_div

import numpy as np
from numpy.polynomial.hermite_e import hermeval, hermemulx

# make it possible to use other packages within the BSFC distribution:
#from os import path
#import sys
#sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))

# BSFC modules
#from helpers import bsfc_helper
#from helpers import bsfc_autocorr
#from helpers.simplex_sampling import hypercubeToSimplex, hypercubeToHermiteSampleFunction
from helpers.simplex_sampling import hypercubeToHermiteSampleFunction
from helpers.simplex_sampling import generalizedHypercubeToHermiteSampleFunction, generalizedHypercubeConstraintFunction


# %%
class LineModel(object):
    """
    Models a spectra. Uses 2nd order Legendre fitting on background noise,
    and n-th order Hermite on the lines
    """
    def __init__(self, lam, lamNorm, specBr, sig, whitefield, lineData, linesFit, hermFuncs, scaleFree=True, sqrtPrior=False):
        """
        Note scaleFree uses scale-free priors (~1/s) on the scale parameters
        if scaleFree=True, sqrtPrior uses the square root prior (~1/sqrt(s)) on Poisson processes instead of the scale-free prior
        Note that these two parameters have only been implemented for the nested sampling routines.
        """
        self.lam = lam
        self.specBr = specBr
        self.sig = sig
        self.lineData = lineData
        # It's assumed that the primary line is at index 0 of linesFit
        self.linesFit = linesFit

        self.linesLam = self.lineData.lam[self.linesFit]
        self.linesnames = self.lineData.names[self.linesFit]
        self.linesSqrtMRatio = self.lineData.sqrt_m_ratio[self.linesFit]
        self.whitefield = whitefield

        # Normalized lambda, for evaluating background noise
        self.lamNorm = lamNorm

        # Get the edge of the lambda bins, for integrating over finite pixels
        lamEdge = np.zeros(len(lam)+1)
        lamEdge[1:-1] = old_div((lam[1:] + lam[:-1]), 2)
        lamEdge[-1] = 2 * lamEdge[-2] - lamEdge[-3]
        lamEdge[0] = 2 * lamEdge[1] - lamEdge[2]
        self.lamEdge = lamEdge
        self.dlam = np.mean(np.diff(lam))

        self.noiseFuncs = 1

        self.nfit = len(linesFit)
        # Number of hermite polynomials to use for each line, 1 being purely Gaussian
        if hermFuncs == None:
            self.hermFuncs = [3] * self.nfit
        else:
            self.hermFuncs = hermFuncs
        #self.hermFuncs[0] = 9

        # Check if the j line is present, which necessitates a special treatment by tying it to the k line
        self.jlinePresent = 'j' in self.linesnames
        if self.jlinePresent:
            self.jindex = np.argwhere(self.linesnames=='j')[0][0]
            # Set it so there are no free hermite function coefficients for the j line
            self.hermFuncs[self.jindex] = 0
            if 'k' in self.linesnames:
                self.kindex = np.argwhere(self.linesnames=='k')[0][0]
            else:
                raise ValueError("Need k line if fitting j line")

        self.simpleConstraints = False

        self.scaleFree = scaleFree
        self.sqrtPrior = sqrtPrior

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
        scale = (old_div(theta[self.noiseFuncs+1],self.linesSqrtMRatio))*1e-4

        # Ragged array of hermite function coefficients
        herm = [None]*self.nfit
        cind = self.noiseFuncs+2
        for i in range(self.nfit):
            herm[i] = np.array(theta[cind:cind+self.hermFuncs[i]])
            cind = cind + self.hermFuncs[i]

        if self.jlinePresent:
            # In case the j line is present, tie the j line hermite coefficients to the k line
            herm[self.jindex] = 1.3576*herm[self.kindex]

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
        Constraint function helper for non-linear fitting stage
        """
        constraints = []


        h0cnstr = lambda theta, n: theta[n]
        # Don't allow functions to grow more than 10% of the original Gaussian
        hncnstr = lambda theta, n, m: theta[n] - np.abs(10*theta[n+m])

        cind = self.noiseFuncs+2

        # Add constraint for noise
        constraints.append({
            'type': 'ineq',
            'fun': h0cnstr,
            'args': [0]
            })

        if self.simpleConstraints:
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
        else:
            for i in range(self.nfit):
                if self.hermFuncs[i] != 0:
                    constraints.append({
                        'type': 'ineq',
                        'fun': h0cnstr,
                        'args': [cind]
                        })

                    constraints.append({
                        'type': 'ineq',
                        'fun': generalizedHypercubeConstraintFunction(cind, self.hermFuncs[i], 0.75),
                        'args': []
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
        lamEv = old_div((self.lam[:,np.newaxis]-center),scale)
        lamEdgeEv = old_div((self.lamEdge[:,np.newaxis]-center),scale)

        # Evaluate gaussian functions
        gauss = np.exp(old_div(-lamEv**2, 2))
        gaussEdge = np.exp(old_div(-lamEdgeEv**2, 2))

        hn = np.zeros(lamEv.shape)
        hnEdge = np.zeros(lamEdgeEv.shape)

        # Compute hermite functions to model lineData
        for i in range(self.nfit):
            hn[:,i] = hermeval(lamEv[:,i], herm[i]) * gauss[:,i]
            hnEdge[:,i] = hermeval(lamEdgeEv[:,i], herm[i]) * gaussEdge[:,i]

        # Compute integral over finite pixel size
        hnEv = (4 * hn + hnEdge[1:] + hnEdge[:-1])/6.0*self.dlam/scale[np.newaxis,:]

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
        lamEv = old_div((self.lam[:,np.newaxis]-center),scale)
        lamEdgeEv = old_div((self.lamEdge[:,np.newaxis]-center),scale)

        # Evaluate gaussian functions
        gauss = np.exp(old_div(-lamEv**2, 2))
        gaussEdge = np.exp(old_div(-lamEdgeEv**2, 2))

        hn = np.zeros(lamEv.shape)
        hnEdge = np.zeros(lamEdgeEv.shape)

        i = line
        if order > len(herm[i]):
            order = len(herm[i])

        hn[:,i] = hermeval(lamEv[:,i], herm[i]) * gauss[:,i]
        hnEdge[:,i] = hermeval(lamEdgeEv[:,i], herm[i]) * gaussEdge[:,i]

        # Compute integral over finite pixel size
        hnEv = (4 * hn + hnEdge[1:] + hnEdge[:-1])/6.0*self.dlam/scale[np.newaxis,:]

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
        w = old_div(self.linesLam[line]**2, self.lineData.m_kev[self.linesFit][line])
        ti = moments[2]*1e-6/moments[0] / w
        if thaco:
            counts = old_div(m0, scale[line])
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
                l_lower = np.max((0, l0-4))
                l_upper = np.min((len(self.lam)-1, l0+5))
                lamFit = self.lam[l_lower:l_upper]
                specFit =  np.maximum( self.specBr[l_lower:l_upper]-noise0, 0) #element-wise

                center = np.average(lamFit, weights=specFit)
                scale = np.sqrt(np.average((lamFit-center)**2, weights=specFit))*1e4

            if self.hermFuncs[i] > 0:
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
    #         Likelihood and prior functions
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

        ll_sig = -0.5 * np.sum(np.log(np.abs(pred)*self.whitefield))
        ll_cnst = -len(pred) * 0.5 * np.log(2.0 * np.pi)
        ll = -np.sum((self.specBr-pred)**2/np.abs(pred)*self.whitefield * 0.5)

        return ll + ll_sig + ll_cnst
        #return -np.sum((self.specBr-pred)**2/np.abs(pred)*self.whitefield)



    def lnprior(self, theta):
        '''
        Log-prior. This is the simplest version of this function, used by emcee's Ensemble Sampler.
        This function sets correlations between Hermite polynomial coefficients.
        '''
        # unpack parameters:
        noise, center, scale, herm = self.unpackTheta(theta)

        if self.simpleConstraints:
            herm0 = np.array([h[0] for h in herm])
            herm1= np.array([h[1] for h in herm])
            herm2 = np.array([h[2] for h in herm])

            # correlated prior:
            if (np.all(noise[0]>0) and np.all(scale>0) and np.all(herm0>0) and
                np.all(herm0-8*np.abs(herm1)>0) and np.all(herm0-8*np.abs(herm2)>0)):
                return 0.0
            else:
                return -np.inf
        else:
            hermiteConstraintsEv = self.hermiteConstraints()

            if (np.all(noise[0]>0) and np.all(scale>0)):
                for f in hermiteConstraintsEv:
                    if f['fun'](theta, *f['args'])<0:
                        return -np.inf
                    else:
                        continue
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
        algorithm at hand).
        '''

        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp+self.lnlike(theta)


    # Routines for multinest:
    def hypercube_lnprior_simple(self, cube):
        """Prior distribution function for :py:mod:`pymultinest`.
        This version does NOT set any correlations between parameters in the prior.

        Maps the (free) parameters in `cube` from [0, 1] to real space.
        This is necessary because MultiNest only works in a unit hypercube.

        Parameters
        ----------
        cube : array of float, (`num_free_params`,)
            The variables in the unit hypercube.
        """

        # noise:
        for kk in range(self.noiseFuncs):
            cube[kk] = cube[kk] * 1e2 # noise must be positive

        # center wavelength (in 1e-4 A units)
        cube[self.noiseFuncs] = cube[self.noiseFuncs]*2e1 - 1e1 # wavelength must be positive

        # scale (in 1e-4 A units)
        cube[self.noiseFuncs+1] = cube[self.noiseFuncs+1]*1e2 # scale must be positive

        # Hermite coefficients:
        #herm = [None]*self.nfit
        cind = self.noiseFuncs+2

        # loop over number of spectral lines:
        for i in range(self.nfit):
            cube[cind]  = cube[cind] *1e5
            #loop over other Hermite coeffs (only j=2,3 normally)
            for j in range(1, self.hermFuncs[i]):
                cube[cind+j]  = cube[cind+j] *2e3 - 1e3  # Hermite coeff must be +ve (large upper bound?)

            # increase count by number of Hermite polynomials considered.
            cind = cind + self.hermFuncs[i]

        return cube

    

    def hypercube_lnprior_simplex(self,cube):
        """Prior distribution function for :py:mod:`pymultinest`.
        This version sets smart bounds within hypercube method of MultiNest.

        Maps the (free) parameters in `cube` from [0, 1] to real space.
        This is necessary because MultiNest only works in a unit hypercube.

        Parameters
        ----------
        cube : array of float, (`num_free_params`,)
            The variables in the unit hypercube.

        """
        # set simplex limits so that a1 and a2 are 1/8 of a0 at most
        # a0 is set to be >0 and smaller than 1e5 (widest bound)
        f_simplex = hypercubeToHermiteSampleFunction(1e3, 0.125, 0.125)

        # noise:
        for kk in range(self.noiseFuncs):
            cube[kk] = cube[kk] * 1e2 # noise must be positive

        # center wavelength (in 1e-4 A units)
        cube[self.noiseFuncs] = cube[self.noiseFuncs]*2e1 - 1e1

        # scale (in 1e-4 A units)
        cube[self.noiseFuncs+1] = cube[self.noiseFuncs+1]*20 + 1 # scale must be positive (>1)

        # Hermite coefficients:
        cind = self.noiseFuncs+2

        # loop over number of spectral lines:
        for i in range(self.nfit):

            # map hypercube to constrained simplex:
            [cube[cind], cube[cind+1],cube[cind+2]] = f_simplex([cube[cind],cube[cind+1], cube[cind+2] ])

            for nn in range(3, self.hermFuncs[i]):
                # constrain any further coefficients to be +/-0.3 h_0
                cube[cind + 3] = 0.3 *  cube[cind] * (2 * cube[cind+3] +1)

            # increase count by number of Hermite polynomials considered.
            cind = cind + self.hermFuncs[i]

        return cube


    
    def hypercube_lnprior_generalized_simplex(self,cube):
        """Prior distribution function for :py:mod:`pymultinest`.
        This version sets smart bounds within hypercube method of MultiNest.

        Maps the (free) parameters in `cube` from [0, 1] to real space.
        This is necessary because MultiNest only works in a unit hypercube.

        Parameters
        ----------
        cube : array of float, (`num_free_params`,)
            The variables in the unit hypercube.
        """
        noise, center, scale, herm = self.unpackTheta(self.theta_ml)


        #print(cube)
        # FS: make sure that noise is a positive variable (it seems to swing a little near 0)
        noise[0] = max([noise[0],1e-10])
        
        # noise params
        if self.scaleFree:
            if self.sqrtPrior:
                # In this case, sqrt(rate) is uniformly distributed
                cube[0] = ((cube[0]-0.5)+np.sqrt(noise[0]))**2
            else:
                # Otherwise log(rate) is uniformly distributed
                cube[0] = np.exp((cube[0]-0.5)+np.log(noise[0]))

            for kk in range(self.noiseFuncs):
                if kk == 0:
                    continue
                else:
                    cube[kk] = (cube[kk]-0.5)*2.0 * cube[0]
        else:
            for kk in range(self.noiseFuncs):
                cube[kk] = cube[kk] * noise[0] * (1 + old_div(10.0, np.sqrt(noise[0])))  # noise must be positive

        # center wavelength (in 1e-4 A units)
        cube[self.noiseFuncs] = cube[self.noiseFuncs]*2e1 - 1e1

        # scale (in 1e-4 A units)
        if self.scaleFree:
            #cube[self.noiseFuncs+1] = np.exp((cube[self.noiseFuncs+1]-0.5)*0.5+np.log(scale[0]))
            cube[self.noiseFuncs+1] = np.exp((cube[self.noiseFuncs+1])*4.0)
        else:
            cube[self.noiseFuncs+1] = cube[self.noiseFuncs+1]*20 + 1 # scale must be positive (>1)

        #cube[self.noiseFuncs+1] = cube[self.noiseFuncs+1]*20 + 1 # scale must be positive (>1)
        # Hermite coefficients:
        cind = self.noiseFuncs+2

        # loop over number of spectral lines:
        for i in range(self.nfit):
            if self.hermFuncs[i] == 0:
                continue

            if self.scaleFree:
                a0 = herm[i][0]
                f_simplex = generalizedHypercubeToHermiteSampleFunction(a0, self.hermFuncs[i], scaleFree=self.scaleFree, sqrtPrior=self.sqrtPrior)
            else:
                a0 = herm[i][0]
                a_max = (a0 + noise[0]) * 1.1
                f_simplex = generalizedHypercubeToHermiteSampleFunction(a_max, self.hermFuncs[i], scaleFree=self.scaleFree, sqrtPrior=self.sqrtPrior)

            cubeCoords = np.array([cube[cind+j] for j in range(self.hermFuncs[i])])

            # map hypercube to constrained simplex:
            hermCoefs = f_simplex(cubeCoords)

            for j in range(self.hermFuncs[i]):
                cube[cind+j] = hermCoefs[j]

            # increase count by number of Hermite polynomials considered.
            cind = cind + self.hermFuncs[i]

        return cube
    

    def hypercube_lnlike(self, theta):
        """Log-likelihood function for py:mod:`pymultinest`.

        Parameters
        ----------
        params : array of float, (`num_free_params`,)
            The free parameters.

        """
        # parameters are in the hypercube space defined by multinest_lnprior
        pred = self.modelPredict(theta)
        ll_sig = -0.5 * np.sum(np.log(np.abs(pred)*self.whitefield))
        ll_cnst = -len(pred) * 0.5 * np.log(2.0 * np.pi)
        ll = -np.sum((self.specBr-pred)**2/np.abs(pred)*self.whitefield * 0.5)

        return ll + ll_sig + ll_cnst
