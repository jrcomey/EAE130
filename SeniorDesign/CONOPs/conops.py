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

plt.style.use("default")
# plt.style.use("seaborn-bright")


params={#FONT SIZES
    'axes.labelsize':45,#Axis Labels
    'axes.titlesize':35,#Title
    'font.size':20,#Textbox
    'xtick.labelsize':30,#Axis tick labels
    'ytick.labelsize':30,#Axis tick labels
    'legend.fontsize':30,#Legend font size
    'font.family':'sans-serif',
    'font.fantasy':'xkcd',
    'font.sans-serif':'Helvetica',
    'font.monospace':'Courier',
    #AXIS PROPERTIES
    'axes.titlepad':2*6.0,#title spacing from axis
    'axes.grid':True,#grid on plot
    'figure.figsize':(16, 10),#square plots
    # 'savefig.bbox':'tight',#reduce whitespace in saved figures#LEGEND PROPERTIES
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

fig, conops = plt.subplots()

flightpath = np.array([[0, 0],
                       [5, 0],
                       [10, 595],
                       [15, 750],
                       [25, 750],
                       [30, 595],
                       [35, 595],
                       [45, 0],
                       [50, 0]])

flightpath2 = copy.deepcopy(flightpath[5:, :])
flightpath2[:, 0] += 20

flightpath2[0, 0] -= 20




t = flightpath[:, 0]
alt = flightpath[:, 1]
meters = mpl.ticker.EngFormatter("m")
conops.yaxis.set_major_formatter(meters)

plothusly(conops,
          t,
          alt,
           xtitle=r"Mission Time [minutes]",
           ytitle=r"Aircraft Altitude [m]",
           # title=r"Concept of Operations",
          marker="")

plt.plot(t, alt, color='k')

plt.plot(flightpath2[:, 0], flightpath2[:, 1], linestyle='--', color='black', alpha=0.8)

conops.grid(False)
plt.xlim([-7, 70])
conops.annotate("Loading",
                (-2, 12))

conops.annotate("Vertical climb to min. altitude",
                (8, 200))

conops.annotate("Cruise Altitude Climb",
                (-7, 650))

conops.annotate("Cruise",
                (17, 770))

conops.annotate("Descent to Minimum Altitude",
                (28, 680))

conops.annotate("Loiter",
                (30, 560))

conops.annotate("Landing Descent",
                (41, 300))

conops.annotate("Landed",
                (46, 10))

conops.annotate("Alternate Heliport Divert",
                (35, 610))

conops.annotate("Landing Desccent",
                (61, 300))

conops.annotate("Landed",
                (65, 10))

conops.spines["right"].set_visible(False)
conops.spines["left"].set_visible(False)
conops.spines["top"].set_visible(False)
conops.spines["bottom"].set_visible(False)
conops.xaxis.set_visible(False)