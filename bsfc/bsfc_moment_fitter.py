''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This script contains the MomentFitter class. This loads experimental data and stores the final spectral fit. 

'''
import numpy as np
from numpy.polynomial.hermite_e import hermeval, hermemulx
import scipy.optimize as op
from collections import namedtuple
import pdb
import multiprocessing
import itertools
import os
import sys
import warnings
import matplotlib.pyplot as plt
plt.ion()
import shutil

# packages that are specific to tokamak fits:
import MDSplus

from bsfc_bin_fit import BinFit

# packages that require extra installation/care:
import emcee
import gptools

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
        with open('../data/atomic_data.csv', 'r') as f:
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

        '''
        print 'Fitting:', [self.lines.symbol[i] +
                ' ' + self.lines.names[i] + ' @ ' +
                str(self.lines.lam[i]) for i in range(len(self.lines.names))]
        '''

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
                #print "Fitting on Branch A"
                lamInRange = True
                branchB = False
        except:
            pass

        if not lamInRange:
            try:
                branchNode = specTree.getNode(rootPath+'.HLIKE')
                self.lam_all = branchNode.getNode('SPEC:LAM').data()
                if np.any(np.logical_and(self.lam_all>lam_bounds[0], self.lam_all<lam_bounds[1])):
                    #print "Fitting on Branch B"
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

        '''
        print 'Fitting:', [self.lines.symbol[i] +
                ' ' + self.lines.names[i] + ' @ ' +
                str(self.lines.lam[i]) for i in range(len(self.lines.names))]
        '''

        # TODO: modify for D3D!
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
                #print "Fitting on Branch A"
                lamInRange = True
        except:
            pass

        if not lamInRange:
            try:
                branchNode = specTree.getNode(rootPath+'.HLIKE')
                self.lam_all = branchNode.getNode('SPEC:LAM').data()
                if np.any(np.logical_and(self.lam_all>lam_bounds[0], self.lam_all<lam_bounds[1])):
                    #print "Fitting on Branch B"
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

        
    def fitSingleBin(self, tbin, chbin, nsteps=1024, emcee_threads=1, PT=False,
                     NS=False,n_hermite=3, n_live_points=400, sampling_efficiency=0.3, verbose=True):
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

        bf = BinFit(lam, specBr, sig, self.lines, range(len(self.lines.names)), n_hermite=n_hermite)

        self.fits[tbin][chbin] = bf

        #print "Now fitting tbin=", tbin, ', chbin=', chbin, " with nsteps=", nsteps
        if NS==False:
            # MCMC fit
            good = bf.MCMCfit(nsteps=nsteps, emcee_threads=emcee_threads, PT=PT)
        else:
            #print "Using Nested Sampling!"
            basename = os.path.abspath(os.environ['BSFC_ROOT']+'/mn_chains/c-.' )
            good = bf.NSfit(lnev_tol= 0.1, n_live_points=n_live_points,sampling_efficiency=sampling_efficiency,
                            basename=basename, verbose=verbose)

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
            for samp in range(100):
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


