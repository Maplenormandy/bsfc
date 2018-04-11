# -*- coding: utf-8 -*-
"""
Defines and implements several classes to do fitting of THACO line data in a
generic way. Uses 2nd order Legendre fitting on background noise, and 2nd order Hermite
function fitting on the lines

@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from numpy.polynomial.hermite_e import hermeval, hermemulx
import MDSplus
import scipy.optimize as op
import emcee
from collections import namedtuple
import bsfc_helper
# import cPickle as pkl
import bsfc_autocorr
import pdb
import corner
import multiprocessing
import dill as pkl
import itertools
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
            self.hermFuncs = [1] * self.nfit
        else:
            self.hermFuncs = hermFuncs

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
        coeffs_labels=['a','b','c']
        for line in range(self.nfit):
            for h in range(self.hermFuncs[0]):
                labels.append('$%s_%d$'%(coeffs_labels[line],h))
      
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
            noiseEv = noise[0] + noise[1]*self.lamNorm + noise[2]*(3*self.lamNorm**2-1)/2
        return noiseEv

    def modelLine(self, theta, line=0, order=-1):
        """
        Get only a single line
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
        Calculate the moments predicted by the model
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



    """
    Likelihood functions
    """
    def lnlike(self, theta):
        pred = self.modelPredict(theta)
        return -np.sum((self.specBr-pred)**2/self.sig**2)

    def lnprior(self, theta):
        noise, center, scale, herm = self.unpackTheta(theta)
        herm0 = np.array([h[0] for h in herm])
        # pdb.set_trace()
        herm1= np.array([h[1] for h in herm])
        herm2 = np.array([h[2] for h in herm])

        if (np.all(noise[0]>0) and np.all(scale>0) and np.all(herm0>0) and 
            np.all(herm0-8*np.abs(herm1)>0) and np.all(herm0-8*np.abs(herm2)>0)):
            return 0.0
        else:
            return -np.inf

    def lnprob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp+self.lnlike(theta)




# =====================================================


class _LnProbWrapper(object):
    """wrapper for log-posterior evaluation in parallel emcee.
    This is needed since instance methods are not pickeable. 
    """
    def __init__(self, bf):
        self.bf = bf
       
    def __call__(self, theta):
        out = self.bf.lineModel.lnprob(theta)

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
        nll = lambda *args: -self.lineModel.lnlike(*args)

        constraints = self.lineModel.hermiteConstraints()
        result_ml = op.minimize(nll, theta0, tol=1e-6, constraints = constraints)

        return result_ml


    def mcmcSample(self, theta_ml, nsteps):
        ndim, nwalkers = len(theta_ml), len(theta_ml)*4
        pos = [theta_ml + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, _LnProbWrapper(self))#, threads=4) # self.lineModel.lnprob
        sampler.run_mcmc(pos, nsteps)
        
        samples = sampler.chain[:, int(nsteps/2.0):, :].reshape((-1, ndim))

        return samples, sampler


    def fit(self, mcmc=True, nsteps=10000, plot_convergence=False):
        theta0 = self.lineModel.guessFit()
        noise, center, scale, herm = self.lineModel.unpackTheta(theta0)
        if herm[0][0] < noise[0]*0.1: # maybe we should set the multiplier to 0.05? 
            # not worth fitting in this case; i.e. the primary line is under the median noise level
            self.m0_ml = 0.0
            self.good = False
            return False
        else:
            self.result_ml = self.optimizeFit(theta0)
            self.theta_ml = self.result_ml['x']
            if mcmc:
                self.samples, sampler = self.mcmcSample(self.theta_ml, nsteps)
            else:
                self.samples = np.array([self.theta_ml]*50)

            if plot_convergence:
                bsfc_autocorr.plot_convergence(sampler, dim=1, nsteps=nsteps)
            

            self.m_samples = np.apply_along_axis(self.lineModel.modelMoments, axis=1, arr=self.samples)
            self.m_ml = self.lineModel.modelMoments(self.theta_ml)

            self.theta_avg = np.average(self.samples, axis=0)
            self.m_avg = np.average(self.m_samples, axis=0)
            self.m_std = np.std(self.m_samples, axis=0, ddof=len(theta0))

            self.good = True
            return True


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
        # pdb.set_trace()

        print "Now fitting tbin=", self.tbin, ',chbin=', chbin, "with nsteps=", self.nsteps
        good = bf.fit(nsteps=self.nsteps)
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
        good = bf.fit(nsteps=self.nsteps)
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

        w0, w1 = np.searchsorted(self.mf.lam_all[:,tbin,chbin], mf.lam_bounds)
        lam = self.mf.lam_all[w0:w1,tbin,chbin]
        specBr = self.mf.specBr_all[w0:w1,tbin,chbin]
        sig = self.mf.sig_all[w0:w1,tbin,chbin]

        # create bin-fit
        bf = BinFit(lam, specBr, sig, self.mf.lines, range(len(self.mf.lines.names)))

        print "Now fitting tbin=", tbin, ',chbin=', chbin, "with nsteps=", self.nsteps
        good = bf.fit(nsteps=self.nsteps)
        if not good:
            print "not worth fitting"

        return bf


# =====================================================


# %%

LineInfo = namedtuple('LineInfo', 'lam m_kev names symbol z sqrt_m_ratio'.split())

class MomentFitter:
    def __init__(self, lam_bounds, primary_line, shot, tht, brancha=True):
        self.lines = LineInfo(None, None, None, None, None, None)
        self.lam_bounds = lam_bounds
        self.primary_line = primary_line
        self.tht=tht

        amuToKeV = 931494.095 # amu in keV
        #speedOfLight = 2.998e+5 # speed of light in km/s

        # Load all wavelength data
        with open('/home/normandy/idl/HIREXSR/hirexsr_wavelengths.csv', 'r') as f:
            lineData = [s.strip().split(',') for s in f.readlines()]
            lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
            lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
            lineName = np.array([ld[3] for ld in lineData[2:]])

        # Load atomic data, for calculating line widths, etc...
        with open('/home/normandy/git/psfc-misc/ReallyMiscScripts/atomic_data.csv', 'r') as f:
            atomData = [s.strip().split(',') for s in f.readlines()]
            atomSymbol = np.array([ad[1] for ad in atomData[1:84]])
            atomMass = np.array([float(ad[3]) for ad in atomData[1:84]]) * amuToKeV


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

        print 'Fitting:', self.lines.names

        self.shot = shot

        specTree = MDSplus.Tree('spectroscopy', shot)

        ana = '.ANALYSIS'
        if tht > 0:
            ana += str(tht)
        if brancha:
            br = '.HELIKE'
        else:
            br = '.HLIKE'

        branchPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana+br

        branchNode = specTree.getNode(branchPath)

        # pdb.set_trace()
        # Indices are [lambda, time, channel]
        self.specBr_all = branchNode.getNode('SPEC:SPECBR').data()
        self.sig_all = branchNode.getNode('SPEC:SIG').data()
        self.lam_all = branchNode.getNode('SPEC:LAM').data()
        
        # Maximum number of channels, time bins
        self.maxChan = np.max(branchNode.getNode('BINNING:CHMAP').data())+1
        self.maxTime = np.max(branchNode.getNode('BINNING:TMAP').data())+1

        # get time basis
        tmp=branchNode.getNode('SPEC:SIG').dim_of(1)
        mask = [tmp[0]>-1][0 ]
        self.time = np.asarray(tmp[0][mask])

        self.fits = [[None for y in range(self.maxChan)] for x in range(self.maxTime)] #[[None]*self.maxChan]*self.maxTime

    def fitSingleBin(self, tbin, chbin, nsteps=1024):
        w0, w1 = np.searchsorted(self.lam_all[:,tbin,chbin], self.lam_bounds)
        lam = self.lam_all[w0:w1,tbin,chbin]
        specBr = self.specBr_all[w0:w1,tbin,chbin]
        sig = self.sig_all[w0:w1,tbin,chbin]

        bf = BinFit(lam, specBr, sig, self.lines, range(len(self.lines.names)))

        self.fits[tbin][chbin] = bf

        print "Now fitting tbin=", tbin, ', chbin=', chbin, " with nsteps=", nsteps
        good = bf.fit(nsteps=nsteps)

        # print self.fits[tbin][chbin].good
        if not good:
            print "not worth fitting"
        else:
            print "--> done"

    def fitSingleBin_par(self, tbin, chbin, nsteps=1024):
        w0, w1 = np.searchsorted(self.lam_all[:,tbin,chbin], self.lam_bounds)
        lam = self.lam_all[w0:w1,tbin,chbin]
        specBr = self.specBr_all[w0:w1,tbin,chbin]
        sig = self.sig_all[w0:w1,tbin,chbin]

        bf = BinFit(lam, specBr, sig, self.lines, range(len(self.lines.names)))

        print "Now fitting tbin=", tbin, ', chbin=', chbin, " with nsteps=", nsteps
        good = bf.fit(nsteps=nsteps)
        if not good:
            print "not worth fitting"
        # else:
        #     print "--> done"

        return bf

    def fitTimeBin(self, tbin, parallel=True, nproc=None, nsteps=1024):
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
                self.fitSingleBin(tbin, chbin)

    def fitChBin(self, chbin, parallel=True, nproc=None, nsteps=1024):
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
                self.fitSingleBin(tbin, chbin)


    def fitTimeWindow(self, tidx_min=None, tidx_max=None, parallel=True, 
        nproc=None, nsteps=1000):
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

            # initialize empty array to hold fits from parallel process
            # fits = np.empty((tidx_max-tidx_min, self.maxChan))

            # parallel run
            fits_tmp = pool.map(ff, np.asarray(map_args))
            fits = np.asarray(fits_tmp).reshape((tidx_max-tidx_min,self.maxChan))
            pdb.set_trace()
            # recollect results into default fits structure
            t=0
            for tbin in range(tidx_min, tidx_max):
                self.fits[tbin][:] = fits[t,:]
                t+=1
            pdb.set_trace()

        else:
            for chbin in range(self.maxChan):
                for tbin in range(tidx_min, tidx_max):
                    self.fitSingleBin(tbin, chbin)

    #####
    def plotSingleBinFit(self, tbin, chbin):
        # bf = self.fits[tbin,chbin]
        bf = self.fits[tbin][chbin]

        if bf == None:
            return

        f0, (a0, a1) = plt.subplots(2, 1, sharex=True, gridspec_kw = {'height_ratios': [4,1]})
        a0.errorbar(bf.lam, bf.specBr, yerr=bf.sig, c='m', fmt='.')

        if bf.good:
            pred = bf.lineModel.modelPredict(bf.theta_ml)
            a0.plot(bf.lam, pred, c='r')

            for samp in range(25):
                theta = bf.samples[np.random.randint(len(bf.samples))]
                noise = bf.lineModel.modelNoise(theta)
                a0.plot(bf.lam, noise, c='g', alpha=0.08)

                for i in range(len(self.lines.names)):
                    line = bf.lineModel.modelLine(theta, i)
                    a0.plot(bf.lam, line+noise, c='c', alpha=0.08)

            noise = bf.lineModel.modelNoise(bf.theta_avg)
            a0.plot(bf.lam, noise, c='g')
            a0.set_title('tbin='+str(tbin)+', chbin='+str(chbin))

            for i in range(len(self.lines.names)):
                line = bf.lineModel.modelLine(bf.theta_avg, i)
                a0.plot(bf.lam, line+noise, c='c')

            a1.errorbar(bf.lam, bf.specBr - pred, yerr=bf.sig, c='r', fmt='.')
            a1.axhline(c='m', ls='--')

            for i in range(len(self.lines.names)):
                a1.axvline(self.lines.lam[i], c='b', ls='--')
                a0.axvline(self.lines.lam[i], c='b', ls='--')

        plt.show()





# %% =======================================================


def plotOverChannels(mf, tbin=126, plot=True, parallel=True, nproc=None, nsteps=1000): 
    '''
    Function to fit signals for a specified time bin, across all channels. 
    Optionally, plots 0th, 1st and 2nd moment across channels.
    '''

    mf.fitTimeBin(tbin, parallel=parallel, nproc=None, nsteps=nsteps)
    # mf = fitTimeBin(mf, tbin, parallel=True, nproc=None)

    # mf.fitSingleBin(tbin, 29)
    #mf.plotSingleBinFit(tbin, 29)

    # %% Plot stuff
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

    if plot:
        f, a = plt.subplots(3, 1, sharex=True)

        a[0].errorbar(range(mf.maxChan), moments[:,0], yerr=moments_std[:,0], fmt='.')
        a[0].set_ylabel('0th moment')
        a[1].errorbar(range(mf.maxChan), moments[:,1], yerr=moments_std[:,1], fmt='.')
        a[1].set_ylabel('1st moment')
        a[2].errorbar(range(mf.maxChan), moments[:,2], yerr=moments_std[:,2], fmt='.')
        a[2].set_ylabel('2nd moment')
        a[2].set_xlabel(r'channel')



# %% =====================================

def inj_brightness(mf, t_min=1.2, t_max=1.4, save=True, refit=False, compare=True, 
    parallel=True, nsteps=1000):
    '''
    Function to obtain time series of Hirex-Sr signals in all channels. 
    This saves files containing the signals needed for MITIM analysis. 
    If data has already been fitted for a shot, one may set refit=False.

    Optionally, set compare=True to plot fits against those obtained for shot
    1101014019 using THACO's fitting routines. 
    '''
    # # select times of interest
    # t_min = 1.2 #1.17 #injection at 1.2 for 1101014030
    # t_max = 1.4 #1.3
    tidx_min = np.argmin(np.abs(mf.time - t_min))
    tidx_max = np.argmin(np.abs(mf.time - t_max))
    time_sel= mf.time[tidx_min: tidx_max]

    print "fitting time bins between ", tidx_min, " and ", tidx_max
    # if fitting has been previously done, avoid doing it all again 
    if refit:
        mf.fitTimeWindow(tidx_min=tidx_min, tidx_max=tidx_max, parallel=True, nsteps=nsteps)

    pdb.set_trace()

    # collect moments and respective standard deviations
    moments = np.empty((tidx_max-tidx_min,mf.maxChan,3))
    moments_std = np.empty((tidx_max-tidx_min,mf.maxChan,3))#[None] * (tidx_max - tidx_min)

    pdb.set_trace()

    for chbin in range(mf.maxChan):
        t=0
        for tbin in range(tidx_min, tidx_max):
            if mf.fits[tbin][chbin].good:
                moments[t,chbin,:] = mf.fits[tbin][chbin].m_avg
                moments_std[t,chbin,:] = mf.fits[tbin][chbin].m_std
            else:
                moments[t,chbin,:] = np.zeros(3)
                moments_std[t,chbin,:] = np.zeros(3)
            t+=1

    moments = np.array(moments)
    moments_std = np.array(moments_std)

    # chsel=3
    # mom = moments[:,chsel,:]
    # mom_std = moments_std[:,chsel,:]
    # mask = np.logical_and(mom[:,0]<10.0,mom[:,1]<10.0,mom[:,2]<10.0)#[0]
    # f, a = plt.subplots(3, 1, sharex=True)

    # a[0].errorbar(time_sel[mask], mom[mask,0], yerr=mom_std[mask,0], fmt='.')
    # a[0].set_ylabel('0th moment')
    # a[1].errorbar(time_sel[mask], mom[mask,1], yerr=mom_std[mask,1], fmt='.')
    # a[1].set_ylabel('1st moment')
    # a[2].errorbar(time_sel[mask], mom[mask,2], yerr=mom_std[mask,2], fmt='.')
    # a[2].set_ylabel('2nd moment')
    # a[2].set_xlabel(r'$\lambda$ [nm]')

    # load previous fit of 1101014019 from THACO's routines
    with open('/home/sciortino/shot_1101014019/signals_1101014019.pkl','rb') as f:
        signals=pkl.load(f)
    # TEMPORARILY get Hirex-Sr position structure from previous runs
    pos=signals[0].pos

    # 
    # 
    pdb.set_trace()
    # Get fitted results for brightness
    hirex_signal = np.zeros((tidx_max-tidx_min, mf.maxChan))
    hirex_uncertainty = np.zeros((tidx_max-tidx_min, mf.maxChan))
    for chbin in range(mf.maxChan):
        t=0
        for tbin in range(tidx_min, tidx_max):
            if mf.fits[tbin][chbin].good:
                hirex_signal[t,chbin] = mf.fits[tbin][chbin].m_avg[0]
                hirex_uncertainty[t,chbin] = mf.fits[tbin][chbin].m_std[0]
            else:
                hirex_signal[t,chbin] = np.nan
                hirex_uncertainty[t,chbin] = np.nan
            t+=1

    pdb.set_trace()
    # adapt this mask based on experience
    mask=np.logical_and(hirex_signal>0.2, hirex_uncertainty>0.05)
    hirex_signal[mask]=np.nan
    hirex_uncertainty[mask]=np.nan

    # plot time series of Hirex-Sr signals for all channels
    plt.figure()
    for i in range(hirex_signal.shape[1]):
        plt.errorbar(time_sel, hirex_signal[:,i], hirex_uncertainty[:,i], fmt='-.', label='ch. %d'%i)
    leg=plt.legend(fontsize=8)
    leg.draggable()

    # store signals in format for MITIM analysis
    inj = bsfc_helper.Injection(t_min, t_min-0.02, t_max)
    sig=bsfc_helper.HirexData(shot=shot,
        sig=hirex_signal, 
        unc=hirex_uncertainty, 
        pos=pos,
        time=time_sel,
        tht=mf.tht,
        injection=inj, 
        debug_plots=True)

    # optionally, save signal to current directory in pkl format
    if save:
        with open('hirex_sig_%d.pkl'%shot,'wb') as f:
            pkl.dump(sig.signal, f, protocol=pkl.HIGHEST_PROTOCOL)

    # optionally, compare obtained normalized fits with those from THACO fits of 1101014019
    if compare:
        plt.figure()
        plt.subplot(211)
        for i in range(sig.signal.y.shape[1]):
            plt.errorbar(sig.signal.t, sig.signal.y_norm[:,i], sig.signal.std_y_norm[:,i])#, '.-')
        plt.title('new fits')

        plt.subplot(212)
        for i in range(sig.signal.y.shape[1]):
            plt.errorbar(signals[0].t, signals[0].y_norm[:,i], signals[0].std_y_norm[:,i])
        plt.title('saved fits from 1101014019')

    return sig.signal








# %%
#mf = MomentFitter(lam_bounds=(3.725, 3.747), primary_line=lya1', shot=1120914036, tht=1, brancha=False)
#mf = MomentFitter(lam_bounds=(3.725, 3.742), primary_line='lya1', shot=1121002022, tht=0, brancha=False)
shot=1101014030 #1101014030 #1101014019
print "Analyzing shot ", shot
mf = MomentFitter(lam_bounds=(3.172, 3.188), primary_line='w', shot=shot, tht=0, brancha=False)

tbin=126; chbin=10 #136
nsteps=100

# mf.fitSingleBin(tbin=tbin, chbin=chbin, nsteps=nsteps)

# with open('mf_%d_%d.pkl'%(shot,nsteps),'wb') as f:
#     pkl.dump(mf, f, protocol=pkl.HIGHEST_PROTOCOL)

# chain = mf.fits[tbin][chbin].sampler.chain[:,int(nsteps/2):,:]
# chain=chain.reshape((-1, chain.shape[-1]))

# chain = mf.fits[tbin][chbin].samples
# corner.corner(chain, labels=mf.fits[tbin][chbin].lineModel.thetaLabels())

# corner.corner(chain, labels=mf.fits[tbin][chbin].lineModel.thetaLabels())
# mf.plotSingleBinFit(tbin=tbin, chbin=chbin)

# plotOverChannels(mf, tbin=126, plot=True, parallel=True)
signal=inj_brightness(mf, t_min=1.2, t_max=1.210, save=False, refit=True, compare=True, nsteps=nsteps)

# signal=inj_brightness(mf, t_min=1.17, t_max=1.3, save=True, refit=True, compare=True)