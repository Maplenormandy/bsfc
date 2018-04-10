# -*- coding: utf-8 -*-
"""
Functions to visualize multidimensional data using a slider plot. 

@author: sciortino
"""

import numpy as np
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

def slider_plot(x, y, z, xlabel='', ylabel='', zlabel='', labels=None, plot_sum=False, **kwargs):
    """Make a plot to explore multidimensional data.
    
    x : array of float, (`M`,)
        The abscissa.
    y : array of float, (`N`,)
        The variable to slide over.
    z : array of float, (`P`, `M`, `N`)
        The variables to plot.
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
    
    l = []
    # pdb.set_trace()
    for v, l_ in zip(z, labels):
        tmp, = a_plot.plot(x, v[:, 0], ls_cycle.next(), label=l_, **kwargs)
        l.append(tmp)
    
    if plot_sum:
        l_sum, = a_plot.plot(x, z[:, :, 0].sum(axis=0), ls_cycle.next(), label='total', **kwargs)
    
    leg=a_plot.legend(loc='best')
    leg.draggable(True)
    title = f.suptitle('')
    
    def update(dum):
        # ls_cycle = itertools.cycle(ls_vals)
        # remove_all(l)
        # while l:
        #     l.pop()
        
        i = int(slider.val)
        
        for v, l_ in zip(z, l):
            l_.set_ydata(v[:, i])
            # l.append(a_plot.plot(x, v[:, i], ls_cycle.next(), label=l_, **kwargs))
        
        if plot_sum:
            l_sum.set_ydata(z[:, :, i].sum(axis=0))
            # l.append(a_plot.plot(x, z[:, :, i].sum(axis=0), ls_cycle.next(), label='total', **kwargs))
        
        a_plot.relim()
        a_plot.autoscale()
        
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
