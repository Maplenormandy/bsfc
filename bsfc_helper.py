# -*- coding: utf-8 -*-
"""
Define useful classes and functions to store impurity injection information and 
spectroscopic data structures.

@author: sciortino
"""
import scipy
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.gridspec as mplgs
import matplotlib.widgets as mplw
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import profiletools
import Tkinter as tk
import gptools
import re

# Threshold for size of HiReX-SR errorbar to reject:
HIREX_THRESH = 0.03

# Regex used to split lists up. This will let the list be delimted by any
# non-numeric characters, where the decimal point and minus sign are NOT
# considered numeric.
LIST_REGEX = r'([0-9]+)[^0-9]*'

class HirexData(object):
    """Helper object to load and process the HiReX-SR data.
    
    The sequence of operations is as follows:
    
    * Load the relevant data from 'run_data.sav'.
    * Launch a GUI to flag/unflag possible outliers.
    * If requested, plot the raw data.
    * Parcel the data into injections, normalize and combine.
    * Generate a :py:class:`Signal` containing the data. This is stored in the
      attribute :py:attr:`self.signal` for later use.
    
    Parameters
    ----------
    injections : list of :py:class:`Injection`
        The injections the data are to be grouped into.
    debug_plots : bool, optional
        If True, plots are made. Default is False.
    """
    def __init__(self, shot, sig, unc, pos, time, tht, injection, debug_plots=False):
        
        self.shot =shot
        self.hirex_signal = scipy.asarray(sig, dtype=float)
        self.hirex_uncertainty = scipy.asarray(unc, dtype=float)
        self.hirex_pos = scipy.asarray(pos, dtype=float)
        self.hirex_time = scipy.asarray(time, dtype=float)
        self.hirex_tht = tht
    
        self.time_1 = injection.t_start
        self.time_2 = injection.t_stop
        
        self.hirex_flagged = (
            (self.hirex_uncertainty > HIREX_THRESH) |
            (self.hirex_uncertainty == 0.0) |
            (np.isnan(self.hirex_signal)) |
            (np.isnan(self.hirex_uncertainty)) |
            (self.hirex_signal > 1.5)
        )
        
        # Flag bad points:
        # this won't work unless we can access a valid pos array 
        root = HirexWindow(self)
        root.mainloop()
        
        if debug_plots:
            f = self.plot_data()
        
        # Process the injections:
        t = []; y = []; std_y = []; y_norm = []; std_y_norm = []
        
        t_hirex_start, t_hirex_stop = profiletools.get_nearest_idx(
            [injection.t_start, injection.t_stop],
            self.hirex_time
        )
        hirex_signal = self.hirex_signal[t_hirex_start:t_hirex_stop + 1, :]
        hirex_flagged = self.hirex_flagged[t_hirex_start:t_hirex_stop + 1, :]
        hirex_uncertainty = self.hirex_uncertainty[t_hirex_start:t_hirex_stop + 1, :]
        hirex_time = self.hirex_time[t_hirex_start:t_hirex_stop + 1] - injection.t_inj
        
        # Normalize to the brightest interpolated max on the brightest
        # chord:
        maxs = scipy.zeros(hirex_signal.shape[1])
        s_maxs = scipy.zeros_like(maxs)
        for j in xrange(0, hirex_signal.shape[1]):
            good = ~hirex_flagged[:, j]
            max_idx = np.argmax(hirex_signal[good,j])
            maxs[j] = hirex_signal[good,j][max_idx]
            s_maxs[j] = hirex_uncertainty[good,j][max_idx]

            # maxs[j], s_maxs[j] = interp_max(
            #     hirex_time[good],
            #     hirex_signal[good, j],
            #     err_y=hirex_uncertainty[good, j],
            #     debug_plots=debug_plots,
            #     method='spline', 
            # )
    
        i_max = maxs.argmax()
        m = maxs[i_max]
        s = s_maxs[i_max]
        
        hirex_signal[hirex_flagged] = scipy.nan
        
        t.append(hirex_time)
        y.append(hirex_signal)
        std_y.append(hirex_uncertainty)
        y_norm.append(hirex_signal / m)
        std_y_norm.append(scipy.sqrt((hirex_uncertainty / m)**2.0 + ((hirex_signal / m)*(s / m))**2.0))
        
        # import pdb
        # pdb.set_trace()

        self.signal = Signal(
            scipy.vstack(y),
            scipy.vstack(std_y),
            scipy.vstack(y_norm),
            scipy.vstack(std_y_norm),
            scipy.hstack(t),
            'HiReX-SR',
            0,
            pos=self.hirex_pos,
            m=m,
            s=s
        )

    
    def plot_data(self, z_max=None):
        """Make a 3d scatterplot of the data.
        
        Parameters
        ----------
        z_max : float, optional
            The maximum value for the z axis. Default is None (no limit).
        norm : bool, optional
            If True, plot the normalized, combined data. Default is False (plot
            the unnormalized, raw data).
        """
        f = plt.figure()
        a = f.add_subplot(1, 1, 1, projection='3d')
        t = self.hirex_time
        keep = ~(self.hirex_flagged.ravel())
        signal = self.hirex_signal
        uncertainty = self.hirex_uncertainty
        CHAN, T = scipy.meshgrid(range(0, signal.shape[1]), t)
        profiletools.errorbar3d(
            a,
            T.ravel()[keep],
            CHAN.ravel()[keep],
            signal.ravel()[keep],
            zerr=uncertainty.ravel()[keep]
        )
        a.set_zlim(0, z_max)
        a.set_xlabel('$t$ [s]')
        a.set_ylabel('channel')
        a.set_zlabel('HiReX-SR signal [AU]')
        
        return f

class HirexPlotFrame(tk.Frame):
    """Frame to hold the plot with the HiReX-SR time-series data.
    """
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.f = Figure()
        self.suptitle = self.f.suptitle('')
        self.a = self.f.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='NESW')
        
        # Need to put the toolbar in its own frame, since it automatically calls
        # pack on itself, but I am using grid.
        self.toolbar_frame = tk.Frame(self)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar_frame.grid(row=1, column=0, sticky='EW')
        
        self.canvas.mpl_connect(
            'button_press_event',
            lambda event: self.canvas._tkcanvas.focus_set()
        )
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.a.set_xlabel('$t$ [s]')
        self.a.set_ylabel('HiReX-SR signal [AU]')
        # TODO: Get a more clever way to handle ylim!
        self.a.set_ylim(0, 1)
        
        self.l = []
        self.l_flagged = []
    
    def on_key_event(self, evt):
        """Tie keys to the toolbar.
        """
        key_press_handler(evt, self.canvas, self.toolbar)

class HirexWindow(tk.Tk):
    """GUI to flag bad HiReX-SR points.
    
    Parameters
    ----------
    data : :py:class:`RunData` instance
        The :py:class:`RunData` object holding the information to be processed.
    """
    def __init__(self, data, ar=False):
        print(
            "Type the indices of the bad points into the text box and press "
            "ENTER to flag them. Use the arrow keys to move between channels. "
            "When done, close the window to continue with the analysis."
        )
        tk.Tk.__init__(self)
        
        self.protocol("WM_DELETE_WINDOW", self.safe_exit)

        self.data = data
        self.ar = ar
        
        self.wm_title("HiReX-SR inspector")
        
        self.plot_frame = HirexPlotFrame(self)
        self.plot_frame.grid(row=0, column=0, sticky='NESW')
        
        # if self.ar:
        #     self.signal = data.ar_signal
        #     self.time = data.ar_time
        #     self.uncertainty = data.ar_uncertainty
        #     self.flagged = data.ar_flagged
        # else:
        self.signal = data.hirex_signal
        self.time = data.hirex_time
        self.uncertainty = data.hirex_uncertainty
        self.flagged = data.hirex_flagged
        
        self.idx_slider = tk.Scale(
            master=self,
            from_=0,
            to=self.signal.shape[1] - 1,
            command=self.update_slider,
            orient=tk.HORIZONTAL
        )
        self.idx_slider.grid(row=1, column=0, sticky='NESW')
        
        self.flagged_box = tk.Entry(self)
        self.flagged_box.grid(row=2, column=0, sticky='NESW')
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.bind("<Left>", self.on_arrow)
        self.bind("<Right>", self.on_arrow)
        self.bind("<Return>", self.process_flagged)
        # self.bind("<Enter>", self.process_flagged)
        self.bind("<KP_Enter>", self.process_flagged)
    
    # def destroy(self):
    #     self.process_flagged()
    #     tk.Tk.destroy(self)

    # def quit(self):
    #     self.process_flagged()
    #     tk.Tk.quit(self)
    
    def safe_exit(self):
        # Workaround for Tkinter hanging.
        self.process_flagged()
        self.destroy()
        self.quit()

    def on_arrow(self, evt):
        """Handle arrow keys to move slider.
        """
        if evt.keysym == 'Right':
            self.process_flagged()
            self.idx_slider.set(
                min(self.idx_slider.get() + 1, self.signal.shape[1] - 1)
            )
        elif evt.keysym == 'Left':
            self.process_flagged()
            self.idx_slider.set(
                max(self.idx_slider.get() - 1, 0)
            )
    
    def process_flagged(self, evt=None):
        """Process the flagged points which have been entered into the text box.
        """
        flagged = re.findall(
            LIST_REGEX,
            self.flagged_box.get()
        )
        flagged = scipy.asarray([int(i) for i in flagged], dtype=int)
        
        idx = self.idx_slider.get()
        self.flagged[:, idx] = False
        self.flagged[flagged, idx] = True
        
        remove_all(self.plot_frame.l_flagged)
        self.plot_frame.l_flagged = []
        
        self.plot_frame.l_flagged.append(
            self.plot_frame.a.plot(
                self.time[flagged],
                self.signal[flagged, idx],
                'rx',
                markersize=12
            )
        )
        
        self.plot_frame.canvas.draw()
    
    def update_slider(self, new_idx):
        """Update the slider to the new index.
        """
        # Remove the old lines:
        remove_all(self.plot_frame.l)
        self.plot_frame.l = []
        
        self.plot_frame.l.append(
            self.plot_frame.a.errorbar(
                self.time,
                self.signal[:, int(new_idx)],
                yerr=self.uncertainty[:, int(new_idx)],
                fmt='.',
                color='b'
            )
        )
        for i, x, y in zip(
                xrange(0, self.signal.shape[0]),
                self.time,
                self.signal[:, int(new_idx)]
            ):
            self.plot_frame.l.append(
                self.plot_frame.a.text(x, y, str(i))
            )
        
        # Insert the flagged points into the textbox:
        self.flagged_box.delete(0, tk.END)
        self.flagged_box.insert(
            0,
            ', '.join(map(str, scipy.where(self.flagged[:, int(new_idx)])[0]))
        )
        
        self.process_flagged()

def interp_max(x, y, err_y=None, s_guess=0.2, s_max=10.0, l_guess=0.005, fixed_l=False, debug_plots=False, method='GP'):
    """Compute the maximum value of the smoothed data.
    
    Estimates the uncertainty using Gaussian process regression and returns the
    mean and uncertainty.
    
    Parameters
    ----------
    x : array of float
        Abscissa of data to be interpolated.
    y : array of float
        Data to be interpolated.
    err_y : array of float, optional
        Uncertainty in `y`. If absent, the data are interpolated.
    s_guess : float, optional
        Initial guess for the signal variance. Default is 0.2.
    s_max : float, optional
        Maximum value for the signal variance. Default is 10.0
    l_guess : float, optional
        Initial guess for the covariance length scale. Default is 0.03.
    fixed_l : bool, optional
        Set to True to hold the covariance length scale fixed during the MAP
        estimate. This helps mitigate the effect of bad points. Default is True.
    debug_plots : bool, optional
        Set to True to plot the data, the smoothed curve (with uncertainty) and
        the location of the peak value.
    method : {'GP', 'spline'}, optional
        Method to use when interpolating. Default is 'GP' (Gaussian process
        regression). Can also use a cubic spline.
    """
    grid = scipy.linspace(max(0, x.min()), min(0.08, x.max()), 1000)
    if method == 'GP':
        hp = (
            gptools.UniformJointPrior([(0, s_max),]) *
            gptools.GammaJointPriorAlt([l_guess,], [0.1,])
        )
        k = gptools.SquaredExponentialKernel(
            # param_bounds=[(0, s_max), (0, 2.0)],
            hyperprior=hp,
            initial_params=[s_guess, l_guess],
            fixed_params=[False, fixed_l]
        )
        gp = gptools.GaussianProcess(k, X=x, y=y, err_y=err_y)
        gp.optimize_hyperparameters(verbose=True, random_starts=100)
        m_gp, s_gp = gp.predict(grid)
        i = m_gp.argmax()
    elif method == 'spline':
        m_gp = scipy.interpolate.UnivariateSpline(
            x, y, w=1.0 / err_y, s=2*len(x)
        )(grid)
        if scipy.isnan(m_gp).any():
            print(x)
            print(y)
            print(err_y)
        i = m_gp.argmax()
    else:
        raise ValueError("Undefined method %s" % (method,))
    
    if debug_plots:
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        a.errorbar(x, y, yerr=err_y, fmt='.', color='b')
        a.plot(grid, m_gp, color='g')
        if method == 'GP':
            a.fill_between(grid, m_gp - s_gp, m_gp + s_gp, color='g', alpha=0.5)
        a.axvline(grid[i])
    
    if method == 'GP':
        return (m_gp[i], s_gp[i])
    else:
        return m_gp[i]

#########################
class Injection(object):
    """Class to store information on a given injection.
    """
    def __init__(self, t_inj, t_start, t_stop):
        self.t_inj = t_inj
        self.t_start = t_start
        self.t_stop = t_stop




#########################
class Signal(object):
    def __init__(self, y, std_y, y_norm, std_y_norm, t, name, atomdat_idx, pos=None, sqrtpsinorm=None, weights=None, blocks=0,m=None,s=None ):
        """Class to store the data from a given diagnostic.
        
        In the parameter descriptions, `n` is the number of signals (both
        spatial and temporal) contained in the instance.
        
        Parameters
        ----------
        y : array, (`n_time`, `n`)
            The unnormalized, baseline-subtracted data as a function of time and
            space. If `pos` is not None, "space" refers to the chords. Wherever
            there is a bad point, it should be set to NaN.
        std_y : array, (`n_time`, `n`)
            The uncertainty in the unnormalized, baseline-subtracted data as a
            function of time and space.
        y_norm : array, (`n_time`, `n`)
            The normalized, baseline-subtracted data.
        std_y_norm : array, (`n_time`, `n`)
            The uncertainty in the normalized, baseline-subtracted data.
        t : array, (`n_time`,)
            The time vector of the data.
        name : str
            The name of the signal.
        atomdat_idx : int or array of int, (`n`,)
            The index or indices of the signals in the atomdat file. If a single
            value is given, it is used for all of the signals. If a 1d array is
            provided, these are the indices for each of the signals in `y`. If
            `atomdat_idx` (or one of its entries) is -1, it will be treated as
            an SXR measurement.
        pos : array, (4,) or (`n`, 4), optional
            The POS vector(s) for line-integrated data. If not present, the data
            are assumed to be local measurements at the locations in
            `sqrtpsinorm`. If a 1d array is provided, it is used for all of the
            chords in `y`. Otherwise, there must be one pos vector for each of
            the chords in `y`.
        sqrtpsinorm : array, (`n`,), optional
            The square root of poloidal flux grid the (local) measurements are
            given on. If line-integrated measurements with the standard STRAHL
            grid for their quadrature points are to be used this should be left
            as None.
        weights : array, (`n`, `n_quadrature`), optional
            The quadrature weights to use. This can be left as None for a local
            measurement or can be set later.
        blocks : int or array of int, (`n`), optional
            A set of flags indicating which channels in the :py:class:`Signal`
            should be treated together as a block when normalizing. If a single
            int is given, all of the channels will be taken together. Otherwise,
            any channels sharing the same block number will be taken together.
        m : float
            maximum signal recorded across any chords and any time for this diagnostic.
            This value is used for normalization of the signals. 
        s : float
            uncertainty in m (see above)
        """
        self.y = scipy.asarray(y, dtype=float)
        if self.y.ndim != 2:
            raise ValueError("y must have two dimensions!")
        self.std_y = scipy.asarray(std_y, dtype=float)
        if self.y.shape != self.std_y.shape:
            raise ValueError("The shapes of y and std_y must match!")
        self.y_norm = scipy.asarray(y_norm, dtype=float)
        if self.y.shape != self.y_norm.shape:
            raise ValueError("The shapes of y and y_norm must match!")
        self.std_y_norm = scipy.asarray(std_y_norm, dtype=float)
        if self.std_y_norm.shape != self.y.shape:
            raise ValueError("The shapes of y and std_y_norm must match!")
        self.t = scipy.asarray(t, dtype=float)
        if self.t.ndim != 1:
            raise ValueError("t must have one dimension!")
        if len(self.t) != self.y.shape[0]:
            raise ValueError("The length of t must equal the length of the leading dimension of y!")
        if isinstance(name, str):
            name = [name,] * self.y.shape[1]
        self.name = name
        try:
            iter(atomdat_idx)
        except TypeError:
            self.atomdat_idx = atomdat_idx * scipy.ones(self.y.shape[1], dtype=int)
        else:
            self.atomdat_idx = scipy.asarray(atomdat_idx, dtype=int)
            if self.atomdat_idx.ndim != 1:
                raise ValueError("atomdat_idx must have at most one dimension!")
            if len(self.atomdat_idx) != self.y.shape[1]:
                raise ValueError("1d atomdat_idx must have the same number of elements as the second dimension of y!")
        if pos is not None:
            pos = scipy.asarray(pos, dtype=float)
            if pos.ndim not in (1, 2):
                raise ValueError("pos must have one or two dimensions!")
            if pos.ndim == 1 and len(pos) != 4:
                raise ValueError("pos must have 4 elements!")
            if pos.ndim == 2 and (pos.shape[0] != self.y.shape[1] or pos.shape[1] != 4):
                raise ValueError("pos must have shape (n, 4)!")
        
        self.pos = pos
        self.sqrtpsinorm = sqrtpsinorm
        
        self.weights = weights
        
        try:
            iter(blocks)
        except TypeError:
            self.blocks = blocks * scipy.ones(self.y.shape[1], dtype=int)
        else:
            self.blocks = scipy.asarray(blocks, dtype=int)
            if self.blocks.ndim != 1:
                raise ValueError("blocks must have at most one dimension!")
            if len(self.blocks) != self.y.shape[1]:
                raise ValueError("1d blocks must have the same number of elements as the second dimension of y!")
        
        if isinstance(m,(float)):
            self.m=m
        elif m==None: 
            pass
        else:
            raise ValueError("maximum signal m must be a float!")
        if isinstance(s,(float)):
            self.s=s
        elif s==None:
            pass
        else: 
            raise ValueError("maximum signal m must be a float!")

    def sort_t(self):
        """Sort the time axis.
        """
        srt = self.t.argsort()
        self.t = self.t[srt]
        self.y = self.y[srt, :]
        self.std_y = self.std_y[srt, :]
        self.y_norm = self.y_norm[srt, :]
        self.std_y_norm = self.std_y_norm[srt, :]
    
    def plot_data(self, norm=False, f=None, share_y=False, y_label='$b$ [AU]',
                  max_ticks=None, rot_label=False, fast=False, ncol=6):
        """Make a big plot with all of the data.
        
        Parameters
        ----------
        norm : bool, optional
            If True, plot the normalized data. Default is False (plot
            unnormalized data).
        f : :py:class:`Figure`, optional
            The figure instance to make the subplots in. If not provided, a
            figure will be created.
        share_y : bool, optional
            If True, the y axes of all of the subplots will have the same scale.
            Default is False (each y axis is automatically scaled individually).
        y_label : str, optional
            The label to use for the y axes. Default is '$b$ [AU]'.
        max_ticks : int, optional
            The maximum number of ticks on the x and y axes. Default is no limit.
        rot_label : bool, optional
            If True, the x axis labels will be rotated 90 degrees. Default is
            False (do not rotate).
        fast : bool, optional
            If True, errorbars will not be drawn in order to make the plotting
            faster. Default is False
        ncol : int, optional
            The number of columns to use. Default is 6.
        """
        if norm:
            y = self.y_norm
            std_y = self.std_y_norm
        else:
            y = self.y
            std_y = self.std_y
        
        if f is None:
            f = plt.figure()
        
        ncol = int(min(ncol, self.y.shape[1]))
        nrow = int(scipy.ceil(1.0 * self.y.shape[1] / ncol))
        gs = mplgs.GridSpec(nrow, ncol)
        
        a = []
        i_col = 0
        i_row = 0
        
        for k in xrange(0, self.y.shape[1]):
            a.append(
                f.add_subplot(
                    gs[i_row, i_col],
                    sharex=a[0] if len(a) >= 1 else None,
                    sharey=a[0] if len(a) >= 1 and share_y else None
                )
            )
            if i_col > 0 and share_y:
                plt.setp(a[-1].get_yticklabels(), visible=False)
            else:
                a[-1].set_ylabel(y_label)
            if i_row < nrow - 2 or (i_row == nrow - 2 and i_col < self.y.shape[1] % (nrow - 1)):
                plt.setp(a[-1].get_xticklabels(), visible=False)
            else:
                a[-1].set_xlabel('$t$ [s]')
                if rot_label:
                    plt.setp(a[-1].xaxis.get_majorticklabels(), rotation=90)
            i_col += 1
            if i_col >= ncol:
                i_col = 0
                i_row += 1
            a[-1].set_title('%s, %d' % (self.name[k], k))
            good = ~scipy.isnan(self.y[:, k])
            if fast:
                a[-1].plot(self.t[good], y[good, k], '.')
            else:
                a[-1].errorbar(self.t[good], y[good, k], yerr=std_y[good, k], fmt='.')
            if max_ticks is not None:
                a[-1].xaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
                a[-1].yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
        
        if share_y:
            a[0].set_ylim(bottom=0.0)
            a[0].set_xlim(self.t.min(), self.t.max())
        
        f.canvas.draw()
        
        return (f, a)


def remove_all(v):
    """Yet another recursive remover, because matplotlib is stupid.
    """
    try:
        for vv in v:
            remove_all(vv)
    except TypeError:
        v.remove()