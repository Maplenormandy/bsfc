# -*- coding: utf-8 -*-
"""
Functions to visualize multidimensional data using a slider plot. 

@author: sciortino
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.gridspec as mplgs
import matplotlib.widgets as mplw
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import itertools
import pdb

mpl.rcParams['errorbar.capsize'] = 3

def slider_plot(x, y, z, z_unc, xlabel='', ylabel='', zlabel='', labels=None, plot_sum=False, **kwargs):
    """Make a plot to explore multidimensional data.
    
    x : array of float, (`M`,)
        The abscissa.
    y : array of float, (`N`,)
        The variable to slide over.
    z : array of float, (`P`, `M`, `N`)
        The variables to plot.
    z_unc : array of float, (`P`, `M`, `N`)
        Uncertainties in the variables to plot.
    xlabel : str, optional
        The label for the abscissa.
    ylabel : str, optional
        The label for the slider.
    zlabel : str, optional
        The label for the ordinate.
    labels : list of str with length `P`
        The labels for each curve in `z`.
    plot_sum : bool, optional
        If True, will also plot the sum over all `P` cases. Default is False.
    """
    if labels is None:
        labels = ['' for v in z]
    f = plt.figure()
    gs = mplgs.GridSpec(2, 1, height_ratios=[10, 1])
    a_plot = f.add_subplot(gs[0, :])
    a_slider = f.add_subplot(gs[1, :])
    
    a_plot.set_xlabel(xlabel)
    a_plot.set_ylabel(zlabel)
    
    color_vals = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    style_vals = ['-', '--', '-.', ':']
    ls_vals = []
    for s in style_vals:
        for c in color_vals:
            ls_vals.append(c + s)
    ls_cycle = itertools.cycle(ls_vals)
    
    err_list = []
    for v, v_unc, l_ in zip(z, z_unc, labels):
        x_error = np.zeros_like(x)
        h_err = a_plot.errorbar(x, v[:, 0],yerr=v_unc[:,0], xerr = x_error, fmt=ls_cycle.next(), label=l_, **kwargs)
        err_list.append(h_err)

    if plot_sum:
        l_sum, = a_plot.plot(x, z[:, :, 0].sum(axis=0), ls_cycle.next(), label='total', **kwargs)
    
    leg=a_plot.legend(loc='best')
    leg.draggable(True)
    title = f.suptitle('')
    
    def adjustErrbarxy(errobj, x, y, x_error, y_error):
        ln, (errx_top, errx_bot, erry_top, erry_bot), (barsx, barsy) = errobj

        ln.set_data(x,y)
        x_base = x
        y_base = y

        xerr_top = x_base + x_error
        xerr_bot = x_base - x_error
        yerr_top = y_base + y_error
        yerr_bot = y_base - y_error

        errx_top.set_xdata(xerr_top)
        errx_bot.set_xdata(xerr_bot)
        errx_top.set_ydata(y_base)
        errx_bot.set_ydata(y_base)

        erry_top.set_xdata(x_base)
        erry_bot.set_xdata(x_base)
        erry_top.set_ydata(yerr_top)
        erry_bot.set_ydata(yerr_bot)

        new_segments_x = [np.array([[xt, y], [xb,y]]) for xt, xb, y in zip(xerr_top, xerr_bot, y_base)]
        new_segments_y = [np.array([[x, yt], [x,yb]]) for x, yt, yb in zip(x_base, yerr_top, yerr_bot)]
        barsx.set_segments(new_segments_x)
        barsy.set_segments(new_segments_y)


    def update(dum):
        i = int(slider.val)
        
        for j,(v, v_unc) in enumerate(zip(z, z_unc)):
            x_error = np.zeros_like(x)
            adjustErrbarxy(err_list[j], x, v[:,i], x_error, v_unc[:,i])

        if plot_sum:
            l_sum.set_ydata(z[:, :, i].sum(axis=0))
                    
        a_plot.relim()
        a_plot.autoscale()
        
        if isinstance(y[i], int):
            title.set_text('%s %d' % (ylabel, y[i]) if ylabel else '%d' % (y[i],))
        else:
            title.set_text('%s = %.5f' % (ylabel, y[i]) if ylabel else '%.5f' % (y[i],))
        
        f.canvas.draw()
    
    def arrow_respond(slider, event):
        if event.key == 'right':
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key == 'left':
            slider.set_val(max(slider.val - 1, slider.valmin))
    
    slider = mplw.Slider(
        a_slider,
        ylabel,
        0,
        len(y) - 1,
        valinit=0,
        valfmt='%d'
    )
    slider.on_changed(update)
    update(0)
    f.canvas.mpl_connect(
        'key_press_event',
        lambda evt: arrow_respond(slider, evt)
    )

 