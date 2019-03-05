''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This script contains the BinFit class, where methods for spectral fitting are defined.

'''

import numpy as np
import scipy.optimize as op
import pdb  #may delete before release

# make it possible to use other packages within the BSFC distribution:
from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))

# BSFC modules
from helpers import bsfc_helper
from helpers import bsfc_autocorr
from bsfc_line_model import *

# packages that require extra installation/care:
import emcee
import gptools

# =====================================================

class BinFit:
    """
    Performs a nonlinear fit and MCMC error estimate of given binned data
    """
    def __init__(self, lam, specBr, sig, lineData, linesFit, n_hermite):
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

        # number of Hermite polynomial terms:
        hermFuncs = [3]*len(linesFit)
        hermFuncs[0] = n_hermite

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


    def mcmcSample(self, theta_ml, emcee_threads, nsteps=1000, PT=True, ntemps=20, Tmax=None, betas=None, thin=1, burn=1000):
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
        Tmax : float, optional
            Maximum temperature allowed for emcee PTSampler. If ``ntemps`` is not given, this argument
            controls the number of temperatures.  Temperatures are chosen according to the spacing criteria until
            the maximum temperature exceeds ``Tmax`. Default is to set ``ntemps`` and leave Tmax=None.
        betas : array, optional
            Array giving the inverse temperatures, :math:`\\beta=1/T`, used in the ladder. The default is chosen
            so that a Gaussian posterior in the given number of dimensions will have a 0.25 tswap acceptance rate.
        thin: int, optional
            thinning factor (choose 1 to avoid thinning)
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
            # testing purposes ONLY
            #betas=np.asarray([1.0,0.9,0.8,0.7,0.6, 0.5])

            # use PT-MCMC
            sampler = emcee.PTSampler(ntemps, nwalkers, ndim, _LnLike_Wrapper(self), _LnPrior_Wrapper(self),
                                      threads=emcee_threads, Tmax=Tmax, betas=betas)

            # if Tmax is not None, ntemps was set internally in PTSampler
            ntemps = sampler.ntemps

            # pos has shape (ntemps, nwalkers, ndim)
            pos = [[theta_ml + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)] for t in range(ntemps)]

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
            import multiprocessing
            emcee_threads = multiprocessing.cpu_count()

        # get a good guess for the fit parameters for each line amd unpack them
        theta0 = self.lineModel.guessFit()
        noise, center, scale, herm = self.lineModel.unpackTheta(theta0)

        # if amplitude of primary line is less than 10% of the noise, not worth fitting better
        if herm[0][0] < noise[0]*0.5: #0.1 or 0.05?
            self.m0_ml = 0.0
            self.good = False
            return False
        else:
            # obtain maximum likelihood fits
            self.result_ml = self.optimizeFit(theta0)
            self.theta_ml = self.result_ml['x']

            # if MCMC is requested, use result from non-linear optimization as an MCMC start
            # If nsteps=1, assume this is a debug run and don't sample
            if mcmc and nsteps>1:
                self.samples, sampler = self.mcmcSample(self.theta_ml, emcee_threads, nsteps=nsteps, PT=PT)
                #sampler = self.mcmcSample(self.theta_ml, emcee_threads, nsteps=nsteps, PT=PT)
                #self.samples = sampler.chain
            else:
                # if MCMC is not requested, do nothing...? (we might want to make this smarter)
                self.samples = np.array([self.theta_ml]*50)

            if plot_convergence and mcmc and not PT:
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


    def NSfit(self, lnev_tol=0.1, n_live_points=400, sampling_efficiency=0.3,
              INS=True, const_eff=True,basename=None, verbose=True):
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
            Setting to activate Importance Nested Sampling in MultiNest. Refer to [Feroz et al., MNRAS 2014].
        basename : str, optional
            Location where MultiNest output is written.
        verbose: bool, optional
            Boolean indicating whether to output verbose MultiNest progress. 
        '''
        # obtain maximum likelihood fits
        theta0 = self.lineModel.guessFit()
        self.result_ml = self.optimizeFit(theta0)
        self.theta_ml = self.result_ml['x']

        # save theta_ml also in the lineModel object, so that constraints may be set based on ML result
        self.lineModel.theta_ml = self.theta_ml
        
        # save moments obtained from maximum likelihood optimization
        self.m_ml = self.lineModel.modelMoments(self.theta_ml)
        
        # dimensionality of the problem
        ndim = self.lineModel.thetaLength()

        try:
            import pymultinest
        except:
            print "********************"
            print "Could not import pyMultiNest! Make sure that both this is in your PYTHONPATH."
            print "MultiNest must also be on your LD_LIBRARY_PATH"
            raise ValueError("Abort BSFC fit")

        pymultinest.run(
            self.lineModel.hypercube_lnlike,   # log-likelihood
            self.lineModel.hypercube_lnprior_generalized_simplex,   # log-prior
            ndim,
            outputfiles_basename=basename,
            n_live_points=n_live_points,
            importance_nested_sampling = INS,
            const_efficiency_mode = const_eff, # only appropriate with INS
            evidence_tolerance = lnev_tol,
            sampling_efficiency = sampling_efficiency,
            n_iter_before_update = 1000, #MultiNest internally multiplies by 10
            max_modes = 100,
            mode_tolerance = -1e90,  #keeps all modes
            verbose = verbose,
            resume = False,
        )

        self.good=True

        return True


    def NS_analysis(self, basename):
        '''
        Splitting of MultiNest output analysis

        '''

        try:
            import pymultinest
        except:
            print "********************"
            print "Could not import pyMultiNest! Make sure that both this is in your PYTHONPATH."
            print "MultiNest must also be on your LD_LIBRARY_PATH"
            raise ValueError("Abort BSFC fit")

        # after MultiNest run, read results
        a = pymultinest.Analyzer(
            n_params= self.lineModel.thetaLength(),
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

        #self.modes=stats['modes'][0]
        #self.maximum= self.modes['maximum']
        #self.maximum_a_posterior= self.modes['maximum a posterior']
        #self.mean=np.asarray(self.modes['mean'])
        #self.sigma=np.asarray(self.modes['sigma'])

        # get log-evidence estimate and uncertainty (from INS, if this is used)
        self.lnev = (stats['global evidence'], stats['global evidence error'])


        g=gptools.summarize_sampler(data[:, 2:],
                                    weights=data[:, 0],
                                    burn=0,
                                    ci=0.95, chain_mask=None)
        self.params_mean = np.asarray(g[0])
        self.params_ci_l = np.asarray(g[1])
        self.params_ci_u = np.asarray(g[2])

        # summarize results
        #self.m_map = self.lineModel.modelMoments(self.maximum_a_posterior)
        #self.m_map_mean = self.lineModel.modelMoments(self.mean)
        #m1 = self.lineModel.modelMoments(self.mean+self.sigma)
        #m2 = self.lineModel.modelMoments(self.mean - self.sigma)
        #self.m_map_std = (m1 - m2)/2.0   #temporary

        # marginalized (fully-Bayesian) results:
        self.m_bayes_marg = self.lineModel.modelMoments(self.params_mean)
        self.m_bayes_marg_low = self.lineModel.modelMoments(self.params_ci_l)
        self.m_bayes_marg_up = self.lineModel.modelMoments(self.params_ci_u)

        # temporary, for compatibility with MCMC methods:
        self.theta_avg = self.params_mean
        self.m_avg = self.m_bayes_marg
        #self.m_std = self.m_map_std

        return True



### =============================
#
#           Wrappers
#
### =============================

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
