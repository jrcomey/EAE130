#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:04:19 2021

@author: jack
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

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
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle=linestyle,
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

plt.style.use("default")
plt.style.use("seaborn-bright")


params={#FONT SIZES
    'axes.labelsize':45,#Axis Labels
    'axes.titlesize':45,#Title
    'font.size':28,#Textbox
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
    'figure.figsize':(16, 8),#square plots
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

df = pd.read_csv("ATP-XW_V8-Drag-Data - Sheet1.csv")
fig, dragpolar = plt.subplots()
plothusly(dragpolar, df.AoA, df.CD, 
          marker='x', 
          xtitle=r"Angle of Attack $\alpha$", 
          ytitle = r"Drag Coefficienct $C_D$", 
          title=r"ATP-XW Blizzard Drag Polar",
          datalabel=r"$C_D$")

rad_format = mpl.ticker.EngFormatter(unit=r"$\degree$")
dragpolar.xaxis.set_major_formatter(rad_format)
