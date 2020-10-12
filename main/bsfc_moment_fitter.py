''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This script contains the MomentFitter class. This loads experimental data and stores the final spectral fit.

'''

import numpy as np
from collections import namedtuple
import multiprocessing
import itertools
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.ion()
from IPython import embed
import bsfc_bin_fit
import bsfc_data  # methods related to data loading


LineInfo = namedtuple('LineInfo', 'lam m_kev names symbol z sqrt_m_ratio'.split())



class MomentFitter:
    def __init__(self, primary_impurity, primary_line, shot, tht=None,
                 lam_bounds=None, nofit=[], experiment='CMOD', instrument='Hirex-Sr'):
        ''' Class to store experimental data and inferred spectral fits.

        INPUTS:
        -------
        primary_impurity: e.g. 'Ar'
        primary_line: e.g. 'w'
        shot: experiment number
        tht: identifier for C-Mod XICS data loading
        lam_bounds: list or array-like, lower and upper wavelength bounds for spectrum of interest
        experiment: {'CMOD','D3D',...}
                Experimental device of interest. Only pre-defined choices for which data fetching is made available
                are acceptable inputs. Default is 'CMOD'.
        instrument: {'Hirex-Sr','XEUS','LOWEUS', 'CER', ...}
                Instrument/diagnostic for which spectral data should be fitted. Note that said instrument must be
                available for the experiment given above. Default is 'Hirex-Sr'.

        '''
        self.lines = LineInfo(None, None, None, None, None, None)
        self.primary_impurity = primary_impurity
        self.primary_line = primary_line
        self.tht=tht
        self.shot = shot
        self.nofit = nofit
        
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

        # Fetch data:
        if experiment=='CMOD':
            if instrument=='Hirex-Sr':
                out = bsfc_data.load_hirex_data(primary_impurity, primary_line, shot, tht, lam_bounds, nofit)
                self.maxChan,self.maxTime,self.whitefield,self.lam_bounds,self.hirex_branch,self.lines = out[0]
                self.time,self.lam_all,self.pos,self.amp_all,self.amp_unc_all = out[1]
                
                self.fits = [[None for y in range(self.maxChan)] for x in range(self.maxTime)] 
            else:
                raise ValueError('Instruments other than Hirex-Sr not yet implemented for CMOD!')
        elif  experiment=='D3D':
            if instrument=='CER':
                raise ValueError('CER has not been implemented for D3D!')
                #out = bsfc_data.load_D3D_cer(primary_impurity, primary_line, shot, tht, lam_bounds)
            else:
                raise ValueError('Instruments other than CER are not yet implemented for D3D!')
        else:
            raise ValueError('Experiments other than CMOD not yet implemented!')

    def fitSingleBin(self, tbin, chbin, nsteps=1024, emcee_threads=1, PT=False,
                     method=2,n_hermite=3, n_live_points='auto', sampling_efficiency=0.3,
                     dynamic=True, # only used for PolyChord
                     INS=True, const_eff=True,
                     verbose=True, basename=None):
        ''' Basic function to launch fitting methods. If method>1, this uses Nested Sampling
        with MultiNest (method=1) or dyPolyChord (method=2). 
        In this case, the number of steps (nsteps) doesn't matter since the
        algorithm runs until meeting a convergence threshold. Parallelization is activated by
        default in MultiNest or PolyChord if MPI libraries are available.

        The sampling algorithm is specified via `method`. Options:
        {0: vanilla emcee, 1: parallel-tempering emcee, 2: MultiNest (default), 3: dyPolyChord}

        The "n_live_points" argument is only used for nested sampling algorithms. Set to 'auto'
        to use the default value of 25*ndims.

        If method=3, use PolyChord. If dynamic==True as well, allow dynamic allocation of live 
        points using dyPolyChord.
        '''
        self.method = method
        
        if isinstance(tbin,float):
            # find (integer) time bin corresponding to given time
            tbin = np.argmin(np.abs(self.time - tbin))

        w0, w1 = np.searchsorted(self.lam_all[:,tbin,chbin], self.lam_bounds)
        lam = self.lam_all[w0:w1,tbin,chbin]
        amp = self.amp_all[w0:w1,tbin,chbin]
        amp_unc = self.amp_unc_all[w0:w1,tbin,chbin]
        whitefield = self.whitefield[w0:w1,tbin,chbin]

        # Fix NaN whitefield values by taking the average of the two neighbors
        nans = np.isnan(whitefield)
        if np.any(nans):
            notnan = np.logical_not(nans)
            whitefield[nans] = np.interp(lam[nans], lam[notnan], whitefield[notnan])
            print(whitefield)
            pass

        # create bin fit
        bf = bsfc_bin_fit.BinFit(lam, amp, amp_unc, whitefield, self.lines, list(np.arange(len(self.lines.names))), n_hermite=n_hermite)

        self.fits[tbin][chbin] = bf

        #print("Now fitting tbin=", tbin, ', chbin=', chbin, " with nsteps=", nsteps)
        if self.method==0 or self.method==1:
            # MCMC fit
            bf.MCMCfit(nsteps=nsteps, emcee_threads=emcee_threads, PT=PT)
            
        elif self.method==2:
            # Using nested rejection sampling within ellipsoidal bounds (MultiNest)
            if basename==None:
                basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/c-.' )

            bf.MN_fit(lnev_tol= 0.1, n_live_points=n_live_points,
                      sampling_efficiency=sampling_efficiency,
                      basename=basename, verbose=verbose,
                      INS=INS, const_eff=const_eff)

        elif self.method==3:
            # Dynamic nested slice sampling (dyPolyChord)
            if basename==None:
                basename = 'dypc_chains' if dynamic else 'pc_chains'  #os.path.abspath(os.environ['BSFC_ROOT']+'/dypc_chains/c-.' )
                
            bf.PC_fit(nlive_const=n_live_points,dynamic_goal=1.0, dynamic=False, ninit=100,
                 basename=basename, verbose=verbose, plot=False)


        else:
            raise ValueError('Unrecognized sampling method!')


    def fitTimeBin(self, tbin, parallel=True, nproc=None, nsteps=1024, emcee_threads=1):
        '''
        Fit signals from all channels in a specific time bin.
        Functional parallelization.

        '''
        if parallel:
            if nproc==None:
                nproc = multiprocessing.cpu_count()
            print("running fitTimeBin in parallel with nproc=", nproc)
            pool = multiprocessing.Pool(processes=nproc)

            # requires a wrapper for multiprocessing to pickle the function
            ff = bsfc_bin_fit._TimeBinFitWrapper(self,nsteps=nsteps, tbin=tbin)

            # map range of channels and compute each
            self.fits[tbin][:] = pool.map(ff, list(np.arange(self.maxChan)))
        else:
            # fit channel bins sequentially
            for chbin in np.arange(self.maxChan):
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
            print("running fitChBin in parallel with nproc=", nproc)
            pool = multiprocessing.Pool(processes=nproc)

            # requires a wrapper for multiprocessing to pickle the function
            ff = bsfc_bin_fit._ChBinFitWrapper(self, nsteps=nsteps, chbin=chbin)

            # map range of channels and compute each
            self.fits[:][chbin] = pool.map(ff, list(np.arange(self.maxTime)))

        else:
            # fit time bins sequentially
            for tbin in np.arange(self.maxTime):
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
            print("running fitTimeWindow in parallel with nproc=", nproc)
            pool = multiprocessing.Pool(processes=nproc)

            # requires a wrapper for multiprocessing to pickle the function
            ff = bsfc_bin_fit._fitTimeWindowWrapper(self,nsteps=nsteps)

            # map range of channels and compute each
            map_args_tpm = list(itertools.product(list(np.arange(tidx_min, tidx_max)), list(np.arange(self.maxChan))))
            map_args = [list(a) for a in map_args_tpm]

            # parallel run
            fits_tmp = pool.map(ff, np.asarray(map_args))
            fits = np.asarray(fits_tmp).reshape((tidx_max-tidx_min,self.maxChan))

            # recollect results into default fits structure
            t=0
            for tbin in np.arange(tidx_min, tidx_max):
                self.fits[tbin][:] = fits[t,:]
                t+=1

        else:
            for chbin in np.arange(self.maxChan):
                for tbin in np.arange(tidx_min, tidx_max):
                    self.fitSingleBin(tbin, chbin, emcee_threads=emcee_threads)

    #####
    def plotSingleBinFit(self, tbin, chbin, plot_clusters=False):
        ''' Function designed to plot spectrum from a single time and a single channel bin.
        This allows visualization and comparison of the results of nonlinear optimization,
        MCMC sampling or Nested Sampling.

        If set to True, the ``plot_clusters'' argument requests that any posterior clusters 
        that may be saved in self.clusters are added to the plot. 
        
        '''
        bf = self.fits[tbin][chbin]

        if bf == None:
            return

        f0, (a0, a1) = plt.subplots(2, 1, sharex=True, gridspec_kw = {'height_ratios': [4,1]})
        a0.errorbar(bf.lam, bf.amp, yerr=bf.amp_unc, c='m', fmt='.')

        if bf.good:
            # color list, one color for each spectral line
            #from matplotlib import cm 
            #color=cm.rainbow(np.linspace(0,1,len(self.lines.names)))
            from matplotlib import colors as mcolors
            colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
            
            color =['b','g','c'] #['g','b','c'] # [colors['darkviolet'],colors['black'],colors['orange']] #['m','k','y']  #['b','g','c']

            # plot in red the overall reconstructed spectrum (sum of all spectral lines)
            pred = bf.lineModel.modelPredict(bf.theta_ml)
            a0.plot(bf.lam, pred, c='r')

            # plot some samples: noise floor in black, spectral lines all in different colors
            for samp in np.arange(160):
                if self.method==0 or self.method==1:  # emcee ES or PT
                    theta = bf.samples[np.random.randint(len(bf.samples))]
                elif self.method==2: # MultiNest
                    # With nested sampling, sample the samples according to the weights
                    sampleIndex = np.searchsorted(bf.cum_sample_weights, np.random.rand())
                    theta = bf.samples[sampleIndex]
                elif self.method==3: # dyPolyChord
                    sampleIndex = np.searchsorted(bf.cum_sample_weights, np.random.rand())
                    theta = bf.samples[sampleIndex]
                    #raise ValueError('Not implemented yet!')
                else:
                    raise ValueError('Unrecognized method!')

                noise = bf.lineModel.modelNoise(theta)
                a0.plot(bf.lam, noise, c='k', alpha=0.04)

                for i in np.arange(len(self.lines.names)):
                    line = bf.lineModel.modelLine(theta, i)
                    a0.plot(bf.lam, line+noise, c=color[i], alpha=0.04)

            # add average inferred noise
            noise = bf.lineModel.modelNoise(bf.theta_avg)
            a0.plot(bf.lam, noise, c='k', label='Inferred noise')

            a1.set_xlabel('Wavelength [$\AA$]')
            a0.set_ylabel('Brightness [a.u.]')
            a1.set_ylabel('Residual [a.u.]')
            
            # plot all fitted spectral lines, one in each color
            for i in np.arange(len(self.lines.names)):
                line = bf.lineModel.modelLine(bf.theta_avg, i)
                a0.plot(bf.lam, line+noise, c=color[i])

            # on second subplot, plot residuals
            a1.errorbar(bf.lam, bf.amp - pred, yerr=bf.amp_unc, c='r', fmt='.')
            a1.axhline(c='m', ls='--')

            for i in np.arange(len(self.lines.names)):
                a1.axvline(self.lines.lam[i], c='b', ls='--')
                a0.axvline(self.lines.lam[i], c='b', ls='--')

            a0.set_ylabel('signal [AU]')
            a1.set_ylabel('res. [AU]')
            a1.set_xlabel(r'$\lambda$ [A]')

        if plot_clusters:
            # show predictions for primary line from each posterior cluster
            cmap = plt.get_cmap('viridis')
            cols = cmap(np.linspace(0, 1, len(bf.clusters['posterior'])))
            for ii, cluster in enumerate(bf.clusters['posterior']):
                col = cols[ii]
                samples = cluster.samples
                weights = cluster.weights
                cum_sample_weights = np.cumsum(weights)
                
                for samp in np.arange(100):
                    sampleIndex = np.searchsorted(cum_sample_weights, np.random.rand())
                    theta = samples[sampleIndex]

                    ii = np.searchsorted(self.lines.names,self.primary_line)
                    line = bf.lineModel.modelLine(theta, ii)
                    a0.plot(bf.lam, line+noise, c=col, alpha=0.04)

        plt.show()


