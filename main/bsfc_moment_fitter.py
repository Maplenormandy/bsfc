''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This script contains the MomentFitter class. This loads experimental data and stores the final spectral fit.

'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np
from collections import namedtuple
import multiprocessing
import itertools
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.ion()
import shutil

# packages that are specific to tokamak fits:
import MDSplus

#from .bsfc_bin_fit import BinFit
#from bsfc_bin_fit import BinFit
import bsfc_bin_fit

# packages that require extra installation/care:
import emcee
try:
    import gptools3 as gptools
except:
    import gptools

# %%

LineInfo = namedtuple('LineInfo', 'lam m_kev names symbol z sqrt_m_ratio'.split())




def hirexsr_pos(shot, hirex_branch, tht, primary_line, primary_impurity,
                plot_pos=False, plot_on_tokamak=False, check_with_tree=False):
    '''
    Get the POS vector as defined in the THACO manual. 
    Unlike in THACO, here we use POS vectors averaged over the wavelength range of the line of interest, rather than over the wavelength range that is fit (including various satellite lines). This reduces the averaging quite significantly. 

    Plotting functions make use of the eqtools and TRIPPy packages.
    '''
    
    specTree = MDSplus.Tree('spectroscopy', shot)
    
    if hirex_branch=='B':
        # pos vectors for detector modules 1-3
        pos1 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD1:POS').data()
        pos2 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD2:POS').data()
        pos3 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD3:POS').data()
        
        # wavelengths for each module
        lam1 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD1:LAMBDA').data()
        lam2 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD2:LAMBDA').data()
        lam3 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD3:LAMBDA').data()
        
        pos_tot = np.hstack([pos1,pos2,pos3])
        lam_tot = np.hstack([lam1,lam2,lam3])
    else:
        # 1 detector module
        pos_tot = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD4:POS').data()
    
        # wavelength
        lam_tot = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD4:LAMBDA').data()
        
        
    branchNode = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS{:s}.{:s}LIKE'.format(
        str(tht) if tht!=0 else '','HE' if hirex_branch=='B' else 'H'))
        
    # mapping from pixels to chords (wavelength x space pixels, but wavelength axis is just padding)
    chmap = branchNode.getNode('BINNING:CHMAP').data()
    pixels_to_chords = chmap[0,:]
    
    # find over which wavelengths the pos vector should be averaged at every time
    # get lambda bounds for specific BSFC line for accurate impurity forward modeling:
    lam_bounds = get_hirexsr_lam_bounds(primary_impurity, primary_line, reduced=True)

    lam_all = branchNode.getNode('SPEC:LAM').data()

    # exclude empty chords
    mask = lam_all[0,0,:]!=-1
    lam_masked = lam_all[:,:,mask]

    # lambda vector does not change over time, so just use tbin=0
    tbin=0
    w0=[]; w1=[]
    for chbin in np.arange(lam_masked.shape[2]):
        bb = np.searchsorted(lam_masked[:,tbin,chbin], lam_bounds)
        w0.append(bb[0])
        w1.append(bb[1])
        
    # form chords
    pos_ave = []
    for chord in np.arange(lam_masked.shape[2]):
        pos_ave.append( np.mean(pos_tot[w0[chord]:w1[chord], pixels_to_chords == chord,:], axis=(0,1) ))
    pos_ave = np.array(pos_ave)


    if plot_pos:
        # show each component of the pos vector separately
        fig,ax = plt.subplots(2,2)
        axx = ax.flatten()
        for i in [0,1,2,3]:
            pcm = axx[i].pcolormesh(pos_tot[:,:,i].T)
            axx[i].axis('equal')
            fig.colorbar(pcm, ax=axx[i])
            
    if plot_on_tokamak:
        import TRIPPy
        import eqtools
        
        # visualize chords
        efit_tree = eqtools.CModEFITTree(shot)
        tokamak = TRIPPy.plasma.Tokamak(efit_tree)
        
        #pos_ave[:,0]*=1.2
        # pos[:,3] indicate spacing between rays
        rays = [TRIPPy.beam.pos2Ray(p, tokamak) for p in pos_ave]   #pos_old]
        
        t0=1.25
        
        weights = TRIPPy.invert.fluxFourierSens(
            rays,
            efit_tree.rz2psinorm,
            tokamak.center,
            t0,
            np.linspace(0,1, 150),
            ds=1e-5
        )[0]
        
        from TRIPPy.plot.pyplot import plotTokamak, plotLine
        
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        # Only plot the tokamak if an axis was not provided:
        plotTokamak(tokamak)
        
        for r in rays:
            plotLine(r, pargs='r')
            
            
        i_flux = np.searchsorted(efit_tree.getTimeBase(), t0)

        a.contour(
            efit_tree.getRGrid(),
            efit_tree.getZGrid(),
            efit_tree.getFluxGrid()[i_flux, :, :],
            80
        )

    if check_with_tree:
        try:
            pos_on_tree = branchNode.getNode('MOMENTS.{:s}:POS'.format(primary_line.upper())).data()
        except:
            pos_on_tree = branchNode.getNode('MOMENTS.LYA1:POS').data()
        return pos_ave, pos_on_tree
    else:
        return pos_ave





def get_hirexsr_lam_bounds(primary_impurity='Ca', primary_line='w', reduced=False):
    '''
    Get wavelength ranges for Hirex-Sr at C-Mod.
    
    reduced : bool, optional
        Boolean specifying whether a reduced or full wavelength range near the indicated primary
        line should be returned. The reduced range only gives the wavelengths where the line is normally 
        observed. The extended range includes satellite lines that must be fitted together with the 
        primary line. Use the reduced range to calculate the POS vector and the full range for fitting. 
    '''

    if primary_impurity == 'Ca':
        if primary_line == 'w':    # w-line at 3.177 mA
            lam_bounds = (3.175, 3.181) if reduced else (3.172, 3.188)
        elif primary_line == 'z':   # z-line at 3.211 mA
            lam_bounds = (3.208, 3.215) if reduced else (3.205, 3.215)
        elif primary_line == 'lya1':
            lam_bounds = (3.010, 3.027)
        elif primary_line == 'z':
            lam_bounds = (3.205, 3.215)
        elif primary_line == 'all':
            primary_line = 'w' # substitute to allow routine to recognize line name
            lam_bounds = (3.172, 3.215)
        else:
            raise NotImplementedError("Line is not yet implemented")

    elif primary_impurity == 'Ar':
        if primary_line == 'w':
            # not much of a reduction in lam space
            lam_bounds = (3.946,3.952) if reduced else (3.945, 3.954)
        elif primary_line == 'z':
            lam_bounds = (3.991,3.998) if reduced else (3.987,3.998) # (3.897, 3.998)
        elif primary_line == 'lya1':
            lam_bounds = (3.725, 3.745)
        elif primary_line == 'zz': # stricter bounds near z (and j)
            primary_line = 'z'   # substitute to allow routine to recognize line name
            lam_bounds = (3.975,3.998) #(3.725, 3.998)
        elif primary_line == 'zzz': # very strict bounds near z (and j)
            primary_line = 'z'   # substitute to allow routine to recognize line name
            lam_bounds = (3.992,3.998) #(3.725, 3.998)                    
        else:
            raise NotImplementedError("Line is not yet implemented")


    return lam_bounds


class MomentFitter(object):
    def __init__(self, primary_impurity, primary_line, shot, tht, lam_bounds = None, nofit=[], experiment='CMOD', instrument='Hirex-Sr'):
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
                self.load_hirex_data(primary_impurity, primary_line, shot, tht, lam_bounds, nofit)
            else:
                raise ValueError('Instruments other than Hirex-Sr not yet implemented for CMOD!')
        elif  experiment=='D3D':
            if instrument=='CER':
                raise ValueError('CER has not been implemented for D3D!')
                #self.load_D3D_cer(primary_impurity, primary_line, shot, tht, lam_bounds)
            else:
                raise ValueError('Instruments other than CER are not yet implemented for D3D!')
        else:
            raise ValueError('Experiments other than CMOD not yet implemented!')

    def load_hirex_data(self, primary_impurity, primary_line, shot, tht, lam_bounds, nofit, hirexsr_file='../data/hirexsr_wavelengths.csv'):
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
            lam_bounds = get_hirexsr_lam_bounds(primary_impurity, primary_line)

        self.lam_bounds = lam_bounds

        # Populate the line data
        lineInd = np.logical_and(lineLam>lam_bounds[0], lineLam<lam_bounds[1])
        lineInd = np.logical_and(lineInd, np.in1d(lineName, nofit, invert=True))
        #satelliteLines = np.array(['s' not in l for l in lineName])
        #lineInd = np.logical_and(satelliteLines, lineInd)
        ln = lineName[lineInd]
        ll = lineLam[lineInd]
        lz = lineZ[lineInd]
        lm = atomMass[lz-1]
        ls = atomSymbol[lz-1]

        # Get the index of the primary line
        self.pl = np.where(ln==primary_line)[0][0]

        lr = np.sqrt(old_div(lm, lm[self.pl]))

        self.lines = LineInfo(ll, lm, ln, ls, lz, lr)

        # Sort lines by distance from primary line
        pl_sorted = np.argsort(np.abs(self.lines.lam-self.lines.lam[self.pl]))
        for data in self.lines:
            data = data[pl_sorted]
        
        specTree = MDSplus.Tree('spectroscopy', shot)

        ana = '.ANALYSIS'
        if tht > 0:
            ana += str(tht)

        # Determine which, if any, detector has the desired lam_bounds
        rootPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana
        self.hirex_branch='A'
        lamInRange = False
        
        try:
            branchNode = specTree.getNode(rootPath+'.HLIKE')  
            self.lam_all = branchNode.getNode('SPEC:LAM').data()
            if np.any(np.logical_and(self.lam_all>lam_bounds[0], self.lam_all<lam_bounds[1])):
                lamInRange = True
                self.hirex_branch = 'A'
        except:
            lamInRange=False


        if not lamInRange:
            try:
                branchNode = specTree.getNode(rootPath+'.HELIKE')  
                self.lam_all = branchNode.getNode('SPEC:LAM').data()
                if np.any(np.logical_and(self.lam_all>lam_bounds[0], self.lam_all<lam_bounds[1])):
                    lamInRange = True
                    self.hirex_branch = 'B'
            except:
                raise ValueError("Fit range does not appear to be on detector")

        # Indices are [lambda, time, channel]
        self.specBr_all = branchNode.getNode('SPEC:SPECBR').data()
        self.sig_all = branchNode.getNode('SPEC:SIG').data()
        with np.errstate(divide='ignore', invalid='ignore'):  #temporarily ignore divide by 0 warnings
            self.whitefield = old_div(self.specBr_all, self.sig_all**2)

        # load pos vector:
        self.pos = hirexsr_pos(shot, self.hirex_branch, tht, primary_line, primary_impurity)

        # Maximum number of channels, time bins
        self.maxChan = np.max(branchNode.getNode('BINNING:CHMAP').data())+1
        self.maxTime = np.max(branchNode.getNode('BINNING:TMAP').data())+1

        # get time basis
        tmp=np.asarray(branchNode.getNode('SPEC:SIG').dim_of(1))
        
        mask = tmp>-1
        self.time = tmp[mask]
        #print('Available times for analysis:', self.time)
        
        self.fits = [[None for y in range(self.maxChan)] for x in range(self.maxTime)] #[[None]*self.maxChan]*self.maxTime



    def fitSingleBin(self, tbin, chbin, nsteps=1024, emcee_threads=1, PT=False,
                     method=1,n_hermite=3, n_live_points=400, sampling_efficiency=0.3,
                     const_eff=True, verbose=True, basename=None):
        ''' Basic function to launch fitting methods. If method>1, this uses Nested Sampling
        with MultiNest (method=1) or dyPolyChord (method=2). 
        In this case, the number of steps (nsteps) doesn't matter since the
        algorithm runs until meeting a convergence threshold. Parallelization is activated by
        default in MultiNest or PolyChord if MPI libraries are available.

        The sampling algorithm is specified via `method`. Options:
        {0: vanilla emcee, 1: parallel-tempering emcee, 2: MultiNest, 3: dyPolyChord}
        '''
        self.method = method
        
        w0, w1 = np.searchsorted(self.lam_all[:,tbin,chbin], self.lam_bounds)
        lam = self.lam_all[w0:w1,tbin,chbin]
        specBr = self.specBr_all[w0:w1,tbin,chbin]
        sig = self.sig_all[w0:w1,tbin,chbin]
        whitefield = self.whitefield[w0:w1,tbin,chbin]

        # Fix NaN whitefield values by taking the average of the two neighbors
        nans = np.isnan(whitefield)
        if np.any(nans):
            notnan = np.logical_not(nans)
            whitefield[nans] = np.interp(lam[nans], lam[notnan], whitefield[notnan])
            print(whitefield)
            pass

        bf = bsfc_bin_fit.BinFit(lam, specBr, sig, whitefield, self.lines, list(range(len(self.lines.names))), n_hermite=n_hermite)

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
                            basename=basename, verbose=verbose, const_eff=const_eff)

        elif self.method==3:
            # Dynamic nested slice sampling (dyPolyChord)
            if basename==None:
                basename = os.path.abspath(os.environ['BSFC_ROOT']+'/dypc_chains/c-.' )
                
            bf.dyPC_fit(dynamic_goal=1.0, ninit=100, nlive_const=n_live_points,
                 dypc_basename=basename, verbose=verbose, plot=False)


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
            self.fits[tbin][:] = pool.map(ff, list(range(self.maxChan)))
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
            print("running fitChBin in parallel with nproc=", nproc)
            pool = multiprocessing.Pool(processes=nproc)

            # requires a wrapper for multiprocessing to pickle the function
            ff = bsfc_bin_fit._ChBinFitWrapper(self, nsteps=nsteps, chbin=chbin)

            # map range of channels and compute each
            self.fits[:][chbin] = pool.map(ff, list(range(self.maxTime)))

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
            print("running fitTimeWindow in parallel with nproc=", nproc)
            pool = multiprocessing.Pool(processes=nproc)

            # requires a wrapper for multiprocessing to pickle the function
            ff = bsfc_bin_fit._fitTimeWindowWrapper(self,nsteps=nsteps)

            # map range of channels and compute each
            map_args_tpm = list(itertools.product(list(range(tidx_min, tidx_max)), list(range(self.maxChan))))
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
    def plotSingleBinFit(self, tbin, chbin, forPaper=False):
        ''' Function designed to plot spectrum from a single time and a single channel bin.
        This allows visualization and comparison of the results of nonlinear optimization,
        MCMC sampling or Nested Sampling.

        '''
        bf = self.fits[tbin][chbin]

        if bf == None:
            return

        if forPaper:
            font = {'family' : 'serif',
                    'serif': ['Times New Roman'],
                    'size'   : 8}

            mpl.rc('font', **font)

            f0, (a0, a1) = plt.subplots(2, 1, sharex=True, gridspec_kw = {'height_ratios': [4,1]}, figsize=(3.375, 3.375*0.8))
        else:
            f0, (a0, a1) = plt.subplots(2, 1, sharex=True, gridspec_kw = {'height_ratios': [4,1]})
        a0.errorbar(bf.lam, bf.specBr, yerr=bf.sig, c='m', fmt='.')

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
            for samp in range(160):
                if self.method==0 or self.method==1:  # emcee ES or PT
                    theta = bf.samples[np.random.randint(len(bf.samples))]
                elif self.method==2: # MultiNest
                    # With nested sampling, sample the samples according to the weights
                    sampleIndex = np.searchsorted(bf.cum_sample_weights, np.random.rand())
                    theta = bf.samples[sampleIndex]
                elif self.method==3: # dyPolyChord
                    raise ValueError('Not implemented yet!')
                else:
                    raise ValueError('Unrecognized method!')

                noise = bf.lineModel.modelNoise(theta)
                a0.plot(bf.lam, noise, c='k', alpha=0.04)

                for i in range(len(self.lines.names)):
                    line = bf.lineModel.modelLine(theta, i)
                    a0.plot(bf.lam, line+noise, c=color[i], alpha=0.04)

            # add average inferred noise
            noise = bf.lineModel.modelNoise(bf.theta_avg)
            a0.plot(bf.lam, noise, c='k', label='Inferred noise')

            if not forPaper:
                a0.set_title('')
                #a0.set_title('tbin='+str(tbin)+', chbin='+str(chbin))
            else:
                a1.set_xlabel('Wavelength [$\AA$]')
                a0.set_ylabel('Brightness [a.u.]')
                a1.set_ylabel('Residual [a.u.]')

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

            a0.set_ylabel('signal [AU]')
            a1.set_ylabel('res. [AU]')
            a1.set_xlabel(r'$\lambda$ [A]')

                          
        if forPaper:
            a1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.05)
            #plt.savefig(os.path.expanduser+'/Pictures/BSFC/newfigs/figure1_new.png')
        plt.show()


