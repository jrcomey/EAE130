#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:18:19 2021

@author: jack
"""

# Preamble

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import copy
import mplcyberpunk

plt.style.use("default")
# plt.style.use("seaborn-bright")


params={#FONT SIZES
    'axes.labelsize':30,#Axis Labels
    'axes.titlesize':30,#Title
    'font.size':28,#Textbox
    'xtick.labelsize':22,#Axis tick labels
    'ytick.labelsize':22,#Axis tick labels
    'legend.fontsize':24,#Legend font size
    'font.family':'sans-serif',
    'font.fantasy':'xkcd',
    'font.sans-serif':'Helvetica',
    'font.monospace':'Courier',
    #AXIS PROPERTIES
    'axes.titlepad':2*6.0,#title spacing from axis
    'axes.grid':True,#grid on plot
    'figure.figsize':(8, 8),#square plots
    'savefig.bbox':'tight',#reduce whitespace in saved figures#LEGEND PROPERTIES
    'legend.framealpha':0.5,
    'legend.fancybox':True,
    'legend.frameon':True,
    'legend.numpoints':1,
    'legend.scatterpoints':1,
    'legend.borderpad':0.1,
    'legend.borderaxespad':0.1,
    'legend.handletextpad':0.2,
    'legend.handlelength':1.0,
    'legend.labelspacing':0,}
mpl.rcParams.update(params)

#%%###########################

# Plotting funcctions

def plothusly(ax, x, y, *, xtitle='', ytitle='',
              datalabel='', title='', linestyle='-',
              marker=''):
    """
    A little function to make graphing less of a pain.
    Creates a plot with titles and axis labels.
    Adds a new line to a blank figure and labels it.

    Parameters
    ----------
    ax : The graph object
    x : X axis data
    y : Y axis data
    xtitle : Optional x axis data title. The default is ''.
    ytitle : Optional y axis data title. The default is ''.
    datalabel : Optional label for data. The default is ''.
    title : Graph Title. The default is ''.

    Returns
    -------
    out : Resultant graph.

    """

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_title(title)
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle = linestyle,
                  marker = marker)
    ax.grid(True)
    ax.legend(loc='best')
    return out


def plothus(ax, x, y, *, datalabel='', linestyle = '-',
            marker = ''):
    """
    A little function to make graphing less of a pain

    Adds a new line to a blank figure and labels it
    """
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle = linestyle,
                  marker = marker)
    ax.legend(loc='best')

    return out

#%%###########################

fig, conops = plt.subplots(
                            figsize=(28, 32)
                           )

flightpath = np.array([[0, 0],
                       [5, 0],
                       [10, 595],
                       [15, 700],
                       [25, 700],
                       [30, 595],
                       [35, 595],
                       [45, 0],
                       [50, 0]])

t = flightpath[:, 0]
alt = flightpath[:, 1]
meters = mpl.ticker.EngFormatter("m")
conops.yaxis.set_major_formatter(meters)

plothusly(conops,
          t,
          alt,
           xtitle=r"Mission Time [minutes]",
           ytitle=r"Aircraft Altitude [m]",
           title=r"Concept of Operations",
          marker="")

conops.grid(False)

conops.annotate("Loading",
                (0, 10))

conops.annotate("Climb to minimum altitude",
                (9, 400))

conops.annotate("Climb to cruise altitude",
                (1, 650))

conops.annotate("Cruise",
                (20, 670))

conops.annotate("Descent to minimum",
                (28, 650))

conops.annotate("Loiter",
                (31, 550))

conops.annotate("Landing Descent",
                (40, 400))

conops.annotate("Landed",
                (46, 10))

conops.spines["right"].set_visible(False)
conops.spines["left"].set_visible(False)
conops.spines["top"].set_visible(False)
conops.spines["bottom"].set_visible(False)
conops.xaxis.set_visible(False)