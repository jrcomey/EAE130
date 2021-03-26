#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 14:36:13 2021

@author: jack
"""
# Imports 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import time
plt.style.use("default")
plt.style.use("seaborn-bright")


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
    'figure.figsize':(12,12),#square plots
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

drag_polar = lambda cd0, cdi, CL: cd0 + cdi*CL**2

C_D_0 = 0.032453979015542
C_D_0_gear = 0.035471335337381
cdi_coeff = 0.002158206278271
S = 174
v = np.linspace(70, 300, 1000)
W = 3100
rho = 0.00237717
b = 36
e= 0.84
fps2mph = lambda v: (v*0.592484)
ftlbs2hp = lambda p: 0.0018*p

P_max = 235*550 * 0.85 # hp

P_req = C_D_0 * 0.5 * rho * v**3 * S + (W/b)**2 * (1/(0.5 * rho * v* np.pi * e)) 
P_gear=  0.5 * C_D_0_gear * rho * v**3 * S + (W/b)**2 * (1/ (0.5*rho*v*np.pi*e))

P_av = P_max * np.ones_like(P_req)

v_stall = 72.5  # fps
wall =np.linspace(0, 1000, 1000)
v_stall_vec = v_stall * np.ones_like(wall)

v_min = fps2mph(v[np.where(P_req == np.min(P_req))].item())
v_min_gear = fps2mph(v[np.where(P_gear == np.min(P_gear))].item())

P_min = np.min(P_gear)

max_excess_power = np.max(abs(P_req - P_av))
RC_max = max_excess_power / W

max_excess_power_gear = np.max(abs(P_gear - P_av))
RC_max_gear = max_excess_power_gear / W

print(f"Rate of climb: {RC_max * 60:.2f} ft/min")

fig, pplot = plt.subplots(figsize=(28,16))


plothusly(pplot, fps2mph(v), ftlbs2hp(P_req), xtitle=r"Airspeed $V_\infty$ (knots)", ytitle="Power ($BHP$)", title="C182 RG Power Curve", datalabel="Required Power")
plothus(pplot, fps2mph(v), ftlbs2hp(P_av), datalabel="")
plothus(pplot, fps2mph(v), ftlbs2hp(P_gear), datalabel="Required Power (Gear Down Configuration)", linestyle='--')
plothus(pplot, fps2mph(v_stall_vec), wall, datalabel=r"", linestyle="--")

v_max_gear = np.interp(235, ftlbs2hp(P_gear), fps2mph(v))
v_max = np.interp(235, ftlbs2hp(P_req), fps2mph(v))
v_max_gear_string = f"{v_max_gear:.2f} kts"
v_max_string = f"{v_max:.2f} kts"

plt.annotate(r"$V_{max_{gear}}$ = " + v_max_gear_string, xy=(v_max_gear, 235), xytext=(100,210), 
              arrowprops=dict(facecolor='red', shrink=0.1))

plt.annotate(r"$V_{max}$ = " + v_max_string, xy=(v_max, 235), xytext=(140,150), 
              arrowprops=dict(facecolor='blue', shrink=0.1))

plt.annotate("", xy=(v_min_gear, 233), xytext=(v_min_gear, 235-ftlbs2hp(max_excess_power_gear)), 
              arrowprops=dict(arrowstyle='<|-|>', facecolor="red"))

plt.annotate("", xy=(v_min, 233), xytext=(v_min, 235-ftlbs2hp(max_excess_power)), 
              arrowprops=dict(arrowstyle='<|-|>', facecolor="blue"))


plt.annotate(r"Max Excess Power$_{gear}$ = 176.91 $hp$", xy=(100,0), xytext=(62, 180))
plt.annotate(r"$RC_{max_{gear}}$ = 1902.35 $ft/min$", xy=(100,0), xytext=(62, 170))

plt.annotate(r"Max Excess Power = 178.14 $hp$", xy=(100,0), xytext=(65, 112))
plt.annotate(r"$RC_{max}$ = 1915.52 $ft/min$", xy=(100,0), xytext=(65, 102))

plt.annotate(r"Available", xy=(100,0), xytext=(25, 235))
plt.annotate(r"Required", xy=(100,0), xytext=(25, 55))

plt.annotate(r"$V_{stall}$", xy=(fps2mph(v_stall), 238), xytext=(50, 240), arrowprops=dict(facecolor='black', shrink=0.1))



plt.ylim([0, 250])
plt.xlim([20, 180])

#%%###########################

CL = np.linspace(0, 3, 1000)

fig, polarplot = plt.subplots(figsize=(24,16))

plothusly(polarplot, CL, drag_polar(C_D_0, cdi_coeff, CL), xtitle=r"$C_L$", ytitle=r"$C_D$", datalabel="Cruise Configuration", title=r"$e$ = 0.84")
plothus(polarplot, CL, drag_polar(C_D_0_gear, cdi_coeff, CL), datalabel="Extended Landing Gear")
plt.suptitle("C 182 RG Drag Polar")
cd0string = f"{C_D_0:.4f}"
cd0gearstring = f"{C_D_0_gear:.4f}"

plt.annotate(r"$C_{D_{0_{gear}}}$ = " + cd0gearstring, xy=(0, np.min(drag_polar(C_D_0_gear, cdi_coeff, CL))), xytext=(0, 0.04), 
              arrowprops=dict(facecolor='green', shrink=0.1))

plt.annotate(r"$C_{D_0}$ = " + cd0string, xy=(0, np.min(drag_polar(C_D_0, cdi_coeff, CL))), xytext=(1.5,0.035), 
              arrowprops=dict(facecolor='blue', shrink=0.1))

plt.annotate(r"Span Efficiency Factor $e$ = 0.84", xy=(0, np.min(drag_polar(C_D_0, cdi_coeff, CL))), xytext=(2, 0.035))

# plt.ylim([0.032, 0.03     8])