''' Bayesian Spectral Fitting Code (BSFC)
by N. Cao & F. Sciortino

This script contains functions used to load and process experimental data.

'''
import numpy as np, os
import MDSplus   # to fetch tokamak data
from collections import namedtuple
import matplotlib.pyplot as plt
plt.ion()
from IPython import embed
from matplotlib import cm


        
LineInfo = namedtuple('LineInfo', 'lam m_kev names symbol z sqrt_m_ratio'.split())

# location of local directory
loc_dir = os.path.dirname(os.path.realpath(__file__))

def load_hirex_data(primary_impurity, primary_line, shot, tht=0, lam_bounds=None, nofit=[],
                    plot=False, plot_time_s=1.0, plot_ch=1):
    '''
    Function to load Hirex-Sr data for CMOD. Assumes that rest wavelengths, ionization stages and
    atomic line names are given in a file provided as input.

    INPUTS:
    ------
    primary_impurity : str
        Ion symbol, e.g. 'Ar'
    primary_line : str
        Symbol of primary line of interest, e.g. 'w' or 'z'
    shot : int
        C-Mod shot number
    tht : int
        Hirex-Sr analysis identifier.
    lam_bounds : 2-element list or array-like
        lower and upper bounds to consider in spectrum
    no_fit : list or array-like
        List of symbols for lines that should be excluded from a fit
    plot : bool
        If True, plot spectrum for the chosen time and channel
    plot_time_s : float
        Time to plot spectrum at.
    plot_ch : int
        Hirex-Sr channel number to plot. NB: this is normally between 1 and 32 or 64 (i.e. no Python indexing)

    OUTPUTS: 
    -------
    info : list
        List containing [maxChan,maxTime,whitefield,lam_bounds, hirex_branch, lines] fields
    data : list
        List containg [time,lam_all,pos,specBr_all,specBr_unc_all,fits] fields
    See bsfc_moment_fitter.py for usage of these output fields.
    
    MWE: 
    info,data = load_hirex_data('Ca','w', 1101014006, tht=0,plot=True)
    '''

    # Load all wavelength data
    with open(loc_dir+'/../data/hirexsr_wavelengths.csv', 'r') as f:
        lineData = [s.strip().split(',') for s in f.readlines()]
        lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
        lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
        lineName = np.array([ld[3] for ld in lineData[2:]])

    amuToKeV = 931494.095 # amu in keV

    # Load atomic data, for calculating line widths, etc...
    with open(loc_dir+'/../data/atomic_data.csv', 'r') as f:
        atomData = [s.strip().split(',') for s in f.readlines()]
        atomSymbol = np.array([ad[1].strip() for ad in atomData[1:84]])
        atomMass = np.array([float(ad[3]) for ad in atomData[1:84]]) * amuToKeV

    if lam_bounds == None:
        lam_bounds,primary_line = get_hirexsr_lam_bounds(primary_impurity, primary_line)

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
    pl_index = np.where(lineName[lineInd]==primary_line)[0][0]
    lr = np.sqrt(lm/lm[pl_index])

    lines = LineInfo(ll, lm, ln, ls, lz, lr)

    # Sort lines by distance from primary line
    pl_sorted = np.argsort(np.abs(lines.lam-lines.lam[pl_index]))
    for data in lines:
        data = data[pl_sorted]

    specTree = MDSplus.Tree('spectroscopy', shot)

    ana = '.ANALYSIS'
    if tht > 0:
        ana += str(tht)

    # Determine which, if any, detector has the desired lam_bounds
    rootPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana
    hirex_branch='A'
    lamInRange = False

    try:
        branchNode = specTree.getNode(rootPath+'.HLIKE')  
        lam_all = branchNode.getNode('SPEC:LAM').data()
        if np.any(np.logical_and(lam_all>lam_bounds[0], lam_all<lam_bounds[1])):
            lamInRange = True
            hirex_branch = 'A'
    except:
        lamInRange=False

    if not lamInRange:
        try:
            branchNode = specTree.getNode(rootPath+'.HELIKE')  
            lam_all = branchNode.getNode('SPEC:LAM').data()
            if np.any(np.logical_and(lam_all>lam_bounds[0], lam_all<lam_bounds[1])):
                lamInRange = True
                hirex_branch = 'B'
        except:
            raise ValueError("Fit range does not appear to be on detector")

    # Indices are [lambda, time, channel]
    specBr_all = branchNode.getNode('SPEC:SPECBR').data()
    specBr_unc_all = branchNode.getNode('SPEC:SIG').data()  # uncertainties
    
    with np.errstate(divide='ignore', invalid='ignore'):  #temporarily ignore divide by 0 warnings
        whitefield = specBr_all/specBr_unc_all**2

    # load pos vector:
    pos = hirexsr_pos(shot, hirex_branch, tht, primary_line, primary_impurity)

    # Maximum number of channels, time bins
    maxChan = np.max(branchNode.getNode('BINNING:CHMAP').data())+1
    maxTime = np.max(branchNode.getNode('BINNING:TMAP').data())+1

    # get time basis
    all_times =np.asarray(branchNode.getNode('SPEC:SIG').dim_of(1))
    mask = all_times >-1
    time = all_times[mask]

    # select only available time elements
    whitefield = whitefield[:,mask,:]
    lam_all = lam_all[:,mask,:]
    specBr_all = specBr_all[:,mask,:]
    specBr_unc_all = specBr_unc_all[:,mask,:]

    #print('Available times for Hirex-Sr analysis:', time)

    # collect all useful outputs
    info = [maxChan,maxTime,whitefield,lam_bounds, hirex_branch, lines]
    data = [time,lam_all,pos,specBr_all,specBr_unc_all]

    if plot:
        # plot spectrum at chosen time and channel, displaying known lines in database
        tidx = np.argmin(np.abs(all_times - plot_time_s))
        fig = plt.figure()
        fig.set_size_inches(10,7, forward=True)
        ax1 = plt.subplot2grid((10,1),(0,0),rowspan = 1, colspan = 1, fig=fig)
        ax2 = plt.subplot2grid((10,1),(1,0),rowspan = 9, colspan = 1, fig=fig, sharex=ax1)
        
        ax2.errorbar(lam_all[:,tidx,plot_ch-1], specBr_all[:,tidx,plot_ch-1], specBr_unc_all[:,tidx,plot_ch-1], fmt='.')
        ax2.set_xlabel(r'$\lambda$ [$A$]',fontsize=14)
        ax2.set_ylabel(r'Signal [A.U.]',fontsize=14)
        #ax2.axvline(lam_bounds[0], c='r', ls='--')
        #ax2.axvline(lam_bounds[1], c='r', ls='--')
        ax1.set_xlim([ax2.get_xlim()[0], ax2.get_xlim()[1]])
        for ii,_line in enumerate(lineLam):
            if _line>ax2.get_xlim()[0] and _line<ax2.get_xlim()[1]:
                ax2.axvline(_line, c='r', ls='--')
                ax1.axvline(_line, c='r', ls='--')
                
                ax1.text(_line, 0.5, lineName[ii], rotation=90, fontdict={'fontsize':14}) #, transform=ax1.transAxes)
        ax1.axis('off')
        
    return info, data



def plot_time_dept_spectrum(primary_impurity, primary_line, shot, tht=0, chbin=0, t0=1.0):
    '''Plot spectrum after Bragg diffraction remapping to spectra over wavelength, time and channel number. 

    MWE:
    plot_time_dept_spectrum('Ca','lya1',1120914029,tht=0,chbin=10, t0=0.91)

    '''
    info,data = load_hirex_data(primary_impurity, primary_line, shot, tht=tht,
                                plot=True, plot_time_s=t0, plot_ch=1)

    time,lam_all,pos,specBr_all,sig_all = data
    maxChan,maxTime,whitefield,lam_bounds, hirex_branch, lines = info

    # take approx wavelength as mean over time bins
    lams = np.mean(lam_all,axis=1)
        
    #fig,ax = plt.subplots()
    #ax.contourf(lams[:,chbin],time,specBr_all[:,:,chbin].T, levels=100, cmap=cm.grey)

    fig,ax = plt.subplots()
    dlam = np.mean(np.diff(lams[:,chbin]))   # NB: lams are actually not equally spaced; this is only for a quick view
    dt = np.mean(np.diff(time))

    img0 = ax.pcolorfast(np.r_[lams[:,chbin]-dlam/2.,lams[-1,chbin]+dlam/2.],
                         np.r_[time-dt/2., time[-1]+dt],
                         specBr_all[:,:,chbin].T,
                         cmap=cm.inferno)

    cbar0 = plt.colorbar(img0, format='%.2g', ax=ax)
    try:
        # if draggable colorbar is available
        cbar0 = DraggableColorbar(cbar0,img0)
        cid1 = cbar0.connect()
    except:
        print('Could not use draggable-colorbar')
        pass
    
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'time [s]')

                
def hirexsr_pos(shot, hirex_branch, tht, primary_line, primary_impurity,
                plot_pos=False, plot_on_tokamak=False, check_with_tree=False, t0=1.25):
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
    lam_bounds,primary_line = get_hirexsr_lam_bounds(primary_impurity, primary_line, reduced=True)

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
            plotLine(r, pargs='r',lw=1.0)
            
            
        i_flux = np.searchsorted(efit_tree.getTimeBase(), t0)

        # mask out coils, where flux is highest
        flux = efit_tree.getFluxGrid()[i_flux, :, :]
        #flux[flux>np.percentile(flux, 75)] = np.nan
        #flux[:,efit_tree.getRGrid()>0.9] = np.nan
        
        cset = a.contour(
            efit_tree.getRGrid(),
            efit_tree.getZGrid(),
            flux,
            80
        )
        #f.colorbar(cset,ax=a)

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
        elif primary_line == 'z':
            lam_bounds = (3.205, 3.215)
        elif primary_line == 'x':  # when indicating x, y-line is also fitted
            lam_bounds = (3.186, 3.1947)
        elif primary_line == 'lya1':  # lya1 and lya2 (H-like), see Rice J.Phys. B 47 (2014) 075701
            lam_bounds = (3.010, 3.030)
        elif primary_line == 'q': # q, r, a (exclude w4n)
            lam_bounds = (3.197, 3.205)
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
        elif primary_line == 'x':
            lam_bounds = (3.964, 3.973)  # includes st, yn4, yn3
        elif primary_line == 'zz': # stricter bounds near z (and j)
            primary_line = 'z'   # substitute to allow routine to recognize line name
            lam_bounds = (3.975,3.998) #(3.725, 3.998)
        elif primary_line == 'zzz': # very strict bounds near z (and j)
            primary_line = 'z'   # substitute to allow routine to recognize line name
            lam_bounds = (3.992,3.998) #(3.725, 3.998)                    
        else:
            raise NotImplementedError("Line is not yet implemented")


    return lam_bounds,primary_line

