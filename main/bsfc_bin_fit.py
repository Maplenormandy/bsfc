''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This script contains the BinFit class, where methods for spectral fitting are defined.

'''
import numpy as np
import scipy.optimize as op
from IPython import embed

# make it possible to use other packages within the BSFC distribution:
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BSFC modules
from helpers import bsfc_helper
from helpers import bsfc_autocorr
from bsfc_line_model import LineModel

# packages that require extra installation/care:
import emcee
import gptools

# =====================================================

class BinFit:
    """
    Performs a nonlinear fit and MCMC error estimate of given binned data
    """
    def __init__(self, lam, amp, amp_unc, whitefield, lineData, linesFit, n_hermite):
        self.lam = lam
        self.amp = amp
        self.amp_unc = amp_unc  
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

        self.lineModel = LineModel(lam, self.lamNorm, amp, amp_unc, whitefield, lineData, linesFit, hermFuncs)


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

        if PT: print("Using PT-MCMC!")
        ndim, nwalkers = len(theta_ml), len(theta_ml)*4

        if PT == False:
            # pos has shape (nwalkers, ndim)
            pos = [theta_ml + 1e-4 * np.random.randn(ndim) for i in np.arange(nwalkers)]

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
            pos = [[theta_ml + 1e-4 * np.random.randn(ndim) for i in np.arange(nwalkers)] for t in np.arange(ntemps)]

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


    def PC_loglike(self, theta):
        ''' Convenience function for PolyChord/dyPolyChord).
        pyPolyChord requires any derived quantities to be returned after the log-likelihood value,
        but here we don't set any.
        '''
        return self.lineModel.ns_lnlike(theta), [] #[0.0]*nderived
    
        
    def PC_fit(self,nlive_const='auto', dynamic=True,dynamic_goal=1.0, ninit=100, 
                 basename='dypc_chains', verbose=True, plot=False):
        '''
        Parameters
        ----------
        dynamic_goal : float, opt
            Parameter in [0,1] determining whether algorithm prioritizes accuracy in 
            evidence accuracy (goal near 0) or parameter estimation (goal near 1).
        ninit : int, opt
            Number of live points to use in initial exploratory run. 
        nlive_const : int, opt
            Total computational budget, equivalent to non-dynamic nested sampling with nlive_const live points.
        dynamic : bool, opt
            If True, use dynamic nested sampling via dyPolyChord. Otherwise, use the
            standard PolyChord.
        basename : str, opt
            Location in which chains will be stored. 
        verbose : bool, opt
            If True, text will be output on the terminal to check how run is proceeding. 
        plot : bool, opt
            Display some sample plots to check result of dynamic slice nested sampling. 
        '''
        if dynamic:
            print('Dynamic slice nested sampling')
        else:
            print('Slice nested sampling')
            
        # obtain maximum likelihood fits
        theta0 = self.lineModel.guessFit()
        self.result_ml = self.optimizeFit(theta0)
        self.theta_ml = self.result_ml['x']

        # save theta_ml also in the lineModel object,
        # so that constraints may be set based on ML result
        self.lineModel.theta_ml = self.theta_ml
        
        # save moments obtained from maximum likelihood optimization
        self.m_ml = self.lineModel.modelMoments(self.theta_ml)
        
        # dimensionality of the problem
        self.ndim = int(self.lineModel.thetaLength())

        if dynamic:
            # dyPolyChord (dynamic slice nested sampling)
            # ------------------------------
            try:
                import dyPolyChord.pypolychord_utils
                import dyPolyChord
            except:
                print("********************")
                print("Could not import dyPolyChord! Make sure that this is in your PYTHONPATH.")
                print("PolyChord must also be on your LD_LIBRARY_PATH")
                raise ValueError("Abort BSFC fit")
        
            #Make a callable for running dyPolyChord
            my_callable = dyPolyChord.pypolychord_utils.RunPyPolyChord(
                self.PC_loglike,
                self.lineModel.hypercube_lnprior_generalized_simplex,
                self.ndim
            )
        
            # Specify sampler settings (see run_dynamic_ns.py documentation for more details)
            settings_dict = {'file_root': 'bsfc',
                             'base_dir': basename,
                             'seed': 1}

            # Run dyPolyChord
            MPI_parallel=True
            if MPI_parallel:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                dyPolyChord.run_dypolychord(my_callable, dynamic_goal, settings_dict,
                                            ninit=ninit,
                                            nlive_const=int(25*self.ndim) if nlive_const=='auto' else nlive_const,
                                            comm=comm)
            else:
                dyPolyChord.run_dypolychord(my_callable, dynamic_goal, settings_dict,
                                            ninit=ninit,
                                            nlive_const=int(25*self.ndim) if nlive_const=='auto' else nlive_const)
                
        else:
            # PolyChord (slice nested sampling)
            # ------------------------------
            try:
                import pypolychord
                from pypolychord.settings import PolyChordSettings
            except:
                print("********************")
                print("Could not import pypolychord! Make sure that this is in your PYTHONPATH.")
                raise ValueError("Abort BSFC fit")
            
            nDerived=0
            settings = PolyChordSettings(self.ndim, nDerived)
            settings.file_root = 'bsfc'
            settings.base_dir = basename
            settings.nlive = int(25*self.ndim) if nlive_const=='auto' else int(nlive_const)
            #settings.do_clustering = True
            #settings.read_resume = False
            settings.feedback = 3
            
            def dumper(live, dead, logweights, logZ, logZerr):
                #print("Last dead point:", dead[-1])
                print("logZ = "+str(logZ)+"+/-"+str(logZerr))
                
            self.polychord_output = pypolychord.run_polychord(self.PC_loglike,
                                               self.ndim,
                                               nDerived,
                                               settings,
                                               self.lineModel.hypercube_lnprior_generalized_simplex,
                                               dumper)

        self.good=True


    def PC_analysis(self, file_root, base_dir=None, dynamic=True, plot=False):
        ''' Analysis of (dy)PolyChord output.
        If dynamic==True, dyPolyChord output is processed, else a default PolyChord 
        run is assumed. 
        '''
        if base_dir is None:
            if dynamic:
                base_dir='dypc_chains'
            else:
                base_dir='pc_chains'

        # Read output:
        try:
            import nestcheck.data_processing
            import nestcheck.estimators as estim
        except:
            print("********************")
            print("Could not import nestcheck! Make sure that this is in your PYTHONPATH.")
            raise ValueError("Abort BSFC analysis")
        
        # load the run
        run = nestcheck.data_processing.process_polychord_run(
            file_root, base_dir)

        # temporary
        self.ndim = run['theta'].shape[1]
        
        #print('The log evidence estimate using the first run is {}'
        #      .format(estim.logz(run)))
        #print('The estimated parameter means are: ')
        #print(*[estim.param_mean(run, param_ind=i) for i in np.arange(self.ndim)])

        import pypolychord
        out = pypolychord.output.PolyChordOutput('pc_chains', 'bsfc')

        paramnames = [('p%i' % i, self.lineModel.thetaLabels()[i].strip('$')) for i in np.arange(self.ndim)]

        # get paramnames files for getdist
        for n in np.arange(out.ncluster):
            out.cluster_paramnames_file(n)
        out.make_paramnames_files(paramnames)   # what's the difference from `..file'?

        self.clusters = {}
        if out.ncluster>1:
            self.clusters['lnev'] = []
            self.clusters['posterior'] = []
            for n in np.arange(out.ncluster):
                self.clusters['lnev'].append([out.logZs[n], out.logZerrs[n]])
                try:
                    self.clusters['posterior'].append(out.cluster_posterior(n+1))
                except:
                    # not all clusters seem to readable for some reason...
                    pass

        post = out.posterior

        self.lnev = [out.logZ, out.logZerr]
        
        self.samples = post.samples #run['theta']
        self.sample_weights = post.weights 
        self.cum_sample_weights = np.cumsum(self.sample_weights)

        self.m_bayes_marg = self.lineModel.modelMoments(post.means)
        #self.m_bayes_marg_low = self.lineModel.modelMoments(self.params_ci_l)
        #self.m_bayes_marg_up = self.lineModel.modelMoments(self.params_ci_u)
        
        # for compatibility with MCMC methods
        self.theta_avg = post.means #self.params_mean
        self.m_avg = self.m_bayes_marg
        
        if dynamic:
            try:
                import dyPolyChord.pypolychord_utils
                import dyPolyChord
            except:
                print("********************")
                print("Could not import dyPolyChord! Make sure that this is in your PYTHONPATH.")
                print("PolyChord must also be on your LD_LIBRARY_PATH")
                raise ValueError("Abort BSFC analysis")

            import nestcheck.ns_run_utils
            
            # get the sample's estimated logX co-ordinates and relative posterior mass
            logx = nestcheck.ns_run_utils.get_logx(run['nlive_array'])
            logw = logx + run['logl']
            w_rel = np.exp(logw - logw.max())

        else:
            from pypolychord.settings import PolyChordSettings
            
        if plot:
            
            import getdist.plots
            import matplotlib.pyplot as plt
            plt.ion()
            plt.switch_backend("Qt5Agg")   # getdist changes backend to Agg without warning...
            
            g = getdist.plots.get_single_plotter() #getSubplotPlotter()
            g.triangle_plot(post, filled=True)
            #g.export('posterior.pdf')

            if dynamic:
                # plot nlive and w_rel on same axis
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax2 = ax1.twinx()
                l1 = ax1.plot(logx, run['nlive_array'], label='number of live points', color='blue')
                l2 = ax2.plot(logx, w_rel, label='relative posterior mass', color='black', linestyle='dashed')
                lines = l1 + l2
                ax1.legend(lines, [l.get_label() for l in lines], loc=0)
                ax1.set_xlabel('estimated $\log X$')
                ax1.set_xlim(right=0.0)
                ax1.set_ylim(bottom=0.0)
                ax1.set_ylabel('number of live points')
                ax2.set_ylim(bottom=0.0)
                ax2.set_yticks([])
                ax2.set_ylabel('relative posterior mass')

            #embed()
            #import pdb
            #pdb.set_trace()
            return logx, run['nlive_array'],w_rel
                 
    def MN_fit(self, lnev_tol=0.1, n_live_points='auto', sampling_efficiency=0.3,
               INS=True, const_eff=True,basename=None, verbose=False, resume=False):
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
            Number of live points. If set to 'auto', use a default of 25*ndim
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

        # save theta_ml also in the lineModel object,
        # so that constraints may be set based on ML result
        self.lineModel.theta_ml = self.theta_ml
        
        # save moments obtained from maximum likelihood optimization
        self.m_ml = self.lineModel.modelMoments(self.theta_ml)
        
        # dimensionality of the problem
        ndim = self.lineModel.thetaLength()

        try:
            import pymultinest
        except:
            print("********************")
            print("Could not import pyMultiNest! Make sure that both this is in your PYTHONPATH.")
            print("MultiNest must also be on your LD_LIBRARY_PATH")
            raise ValueError("Abort BSFC fit")

        pymultinest.solve(
            self.lineModel.ns_lnlike,   # log-likelihood
            self.lineModel.hypercube_lnprior_generalized_simplex,   # log-prior
            ndim,
            outputfiles_basename=basename,
            n_live_points=200+int(25*ndim) if n_live_points=='auto' else int(n_live_points),
            importance_nested_sampling = INS,
            const_efficiency_mode = const_eff, # only appropriate with INS
            evidence_tolerance = lnev_tol,
            sampling_efficiency = sampling_efficiency,
            n_iter_before_update = 1000, #MultiNest internally multiplies by 10
            max_modes = 100,
            mode_tolerance = -1e90,  #keeps all modes
            verbose = verbose,
            resume = resume,
        )

        self.good=True

        # Read output:
        #import nestcheck.data_processing
        #import nestcheck.estimators as estim
    
        #run = nestcheck.data_processing.process_multinest_run(
        #    basename, basename)
        
            


    def MN_analysis(self, basename):
        '''
        Analysis of MultiNest output.
        '''

        try:
            import pymultinest
        except:
            print("********************")
            print("Could not import pyMultiNest! Make sure that both this is in your PYTHONPATH.")
            print("MultiNest must also be on your LD_LIBRARY_PATH")
            raise ValueError("Abort BSFC fit")

        ####
        ##Only works for MultiNest v3.11+, but we use MultiNest 3.10
        ## Read output:
        #try:
        #    import nestcheck.data_processing
        #    import nestcheck.estimators as estim
        #except:
        #    print("********************")
        #    print("Could not import nestcheck! Make sure that this is in your PYTHONPATH.")
        #    raise ValueError("Abort BSFC analysis")
        # 
        ## for testing
        #run = nestcheck.data_processing.process_multinest_run(
        #    basename.split('/')[-1],
        #    os.path.dirname(basename)+'/')
        
        ####
        
        # after MultiNest run, read results
        a = pymultinest.Analyzer(
            n_params= self.lineModel.thetaLength(),
            outputfiles_basename=basename
        )

        # get chains and weights
        data = a.get_data()

        self.samples = data[:,2:]
        self.sample_weights = data[:,0]
        # Used for sampling
        self.cum_sample_weights = np.cumsum(self.sample_weights)
        self.sample_n2ll = data[:,1]

        # save statistics
        stats = a.get_stats()
        self.multinest_stats = stats

        self.modes=stats['modes'][0]
        self.maximum= self.modes['maximum']
        self.maximum_a_posterior= self.modes['maximum a posterior']
        self.mean=np.asarray(self.modes['mean'])
        self.sigma=np.asarray(self.modes['sigma'])

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

class _LnPost_Wrapper:
    """wrapper for log-posterior evaluation in parallel emcee.
    This is needed since instance methods are not pickeable.
    """
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, theta):
        out = self.bf.lineModel.lnprob(theta)

        return out

class _LnLike_Wrapper:
    """wrapper for log-likelihood evaluation in parallel emcee.
    This is needed since instance methods are not pickeable.
    """
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, theta):
        out = self.bf.lineModel.lnlike(theta)

        return out

class _LnPrior_Wrapper:
    """wrapper for log-prior evaluation in parallel emcee.
    This is needed since instance methods are not pickeable.
    """
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, theta):
        out = self.bf.lineModel.lnprior(theta)

        return out




class _TimeBinFitWrapper:
    """ Wrapper to support parallelization of different channels in a
    specific time bin. This is needed since instance methods are not pickeable.

    """
    def __init__(self, mf, nsteps, tbin, n_hermite):
        self.mf = mf
        self.nsteps = nsteps
        self.tbin = tbin
        self.n_hermite = n_hermite

    def __call__(self, chbin):

        w0, w1 = np.searchsorted(self.mf.lam_all[:,self.tbin,chbin], self.mf.lam_bounds)
        lam = self.mf.lam_all[w0:w1,self.tbin,chbin]
        amp = self.mf.amp_all[w0:w1,self.tbin,chbin]
        amp_unc = self.mf.amp_unc_all[w0:w1,self.tbin,chbin]

        # create bin-fit
        bf = BinFit(lam, amp, amp_unc, self.mf.whitefield, self.mf.lines,
                    list(np.arange(len(self.mf.lines.names))), n_hermite=self.n_hermite)
         
        print("Now fitting tbin =", self.tbin, ',chbin =', chbin, "with nsteps =", self.nsteps)
        good = bf.MCMCfit(nsteps=self.nsteps)
        if not good:
            print("not worth fitting")

        return bf



class _ChBinFitWrapper:
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
        amp = self.mf.amp_all[w0:w1,tbin,self.chbin]
        amp_unc = self.mf.amp_unc_all[w0:w1,tbin,self.chbin]

        # create bin-fit
        bf = BinFit(lam, amp, amp_unc, self.mf.whitefield, self.mf.lines,
                    list(np.arange(len(self.mf.lines.names))), n_hermite=self.n_hermite)

        print("Now fitting tbin=", tbin, ',chbin=', self.chbin, "with nsteps=", self.nsteps)
        good = bf.MCMCfit(nsteps=self.nsteps)
        if not good:
            print("not worth fitting")

        return bf


class _fitTimeWindowWrapper:
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
        amp = self.mf.amp_all[w0:w1,tbin,chbin]
        amp_unc = self.mf.amp_unc_all[w0:w1,tbin,chbin]

        # create bin-fit
        bf = BinFit(lam, amp, amp_unc, self.mf.whitefield, self.mf.lines,
                    list(np.arange(len(self.mf.lines.names))), n_hermite=self.n_hermite)

        print("Now fitting tbin=", tbin, ', chbin=', chbin, " with nsteps=", self.nsteps)
        try:
            good = bf.MCMCfit(nsteps=self.nsteps)
        except ValueError:
            print("BinFit.fit() failed.")
            print("++++++++++++++++++++++++++++++++")
            print("Failed at fitting tbin=", tbin, ', chbin=', chbin, " with nsteps=", self.nsteps)
            print("++++++++++++++++++++++++++++++++")
            good = False
        if not good:
            print("Fitting not available. Result will be None.")

        return bf
