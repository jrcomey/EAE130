#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:49:29 2021

@author: jack
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import time
import pandas as pd
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
    'figure.figsize':(24,24),#square plots
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

#%%###########################

general_graph_title = "Roskam C.O.M. Method: "
img = plt.imread("Images/Aircraft/AircraftTopView.png")
img2 = plt.imread("Images/Aircraft/AircraftSideView.png")
cg_color = "red"
fig, massplot = plt.subplots()

nose_length = 2.935
tail_length = nose_length
fuselage_length = 3
fuselage_height = 2.5
skid_height = 1.1
rotor_duct_height = 0.35
aircraft_size = np.array([fuselage_length+2*nose_length, 
                          2*(3+0.4*0.3 + 2.935), 
                          fuselage_height+skid_height+rotor_duct_height])

alpha=0.6

top_x_conv = img.shape[1] / aircraft_size[0]
top_y_conv = img.shape[0] / aircraft_size[1]

# [x, y, z, m]
mass_coords = np.array([#[x, y, z, m],
                        [3.72, -2.935, 1.4, 5],  # BackR prop
                        [3.72, 2.935, 1.4, 5],  # Back L prop
                        [-0.72, 2.935, 1.4, 5],  # FrontL prop
                        [-0.72, -2.935, 1.4, 5], # FrontR prop
                        [3.72, -2.935, 1.4, 49],  # BackR motor
                        [3.72, 2.935, 1.4, 49],  # Back L motor
                        [-0.72, 2.935, 1.4, 49],  # FrontL motor
                        [-0.72, -2.935, 1.4, 49], # FrontR motor
                        [3.72, -2.935, 1.4, 5],  # BackR prop
                        [3.72, 2.935, 1.4, 5],  # Back L prop
                        [-0.72, 2.935, 1.4, 5],  # FrontL prop
                        [-0.72, -2.935, 1.4, 5], # FrontR prop
                        [3.72, -2.935, 1.4, 49],  # BackR motor
                        [3.72, 2.935, 1.4, 49],  # Back L motor
                        [-0.72, 2.935, 1.4, 49],  # FrontL motor
                        [-0.72, -2.935, 1.4, 49], # FrontR motor
                        [-0.5, 0, 0, 80],  # Pilot
                        [1, 0.5, 0, 80],  # Passenger
                        [1, -0.5, 0, 80], # Passenger
                        [2, -0.5, 0, 80],  # Passenger
                        [2, 0.5, 0, 80],  # Passenger
                        [1, 0.5, 0, 8],  # Passenger luggage
                        [1, -0.5, 0, 8], # Passenger luggage
                        [2, -0.5, 0, 8],  # Passenger luggage
                        [2, 0.5, 0, 8],  # Passenger luggage
                        [-0.5, 0, 0, 20],  # Pilot seat
                        [1, 0.5, 0, 20],  # Passenger seat
                        [1, -0.5, 0, 20], # Passenger seat
                        [2, -0.5, 0, 20],  # Passenger seat
                        [2, 0.5, 0, 20],  # Passenger seat
                        [1+0.75, 0, -0.9, 350], # Fuel Cells
                        [2, -0.5, -0.9, 140], # H2 tank 1
                        [2, 0.5, -0.9, 140], # H2 tank 2
                        [1.75, 0, -0.9, 70],  # Fuel Cell Aux systems
                        [-0.5, 0.5, -0.25, 20],  # Avionics
                        [4.5, 0, -0.25, 16],  # Weather system
                        [2, 0, 0, 750],  # COM of structure
                        [0, 0, 0, 0],],
                       dtype=float)

# [x, y, z, m]
mass_coords_empty = np.array([#[x, y, z, m],
                        [3.72, -2.935, 1.4, 5],  # BackR prop
                        [3.72, 2.935, 1.4, 5],  # Back L prop
                        [-0.72, 2.935, 1.4, 5],  # FrontL prop
                        [-0.72, -2.935, 1.4, 5], # FrontR prop
                        [3.72, -2.935, 1.4, 49],  # BackR motor
                        [3.72, 2.935, 1.4, 49],  # Back L motor
                        [-0.72, 2.935, 1.4, 49],  # FrontL motor
                        [-0.72, -2.935, 1.4, 49], # FrontR motor
                        [3.72, -2.935, 1.4, 5],  # BackR prop
                        [3.72, 2.935, 1.4, 5],  # Back L prop
                        [-0.72, 2.935, 1.4, 5],  # FrontL prop
                        [-0.72, -2.935, 1.4, 5], # FrontR prop
                        [3.72, -2.935, 1.4, 49],  # BackR motor
                        [3.72, 2.935, 1.4, 49],  # Back L motor
                        [-0.72, 2.935, 1.4, 49],  # FrontL motor
                        [-0.72, -2.935, 1.4, 49], # FrontR motor
                        # [-0.5, 0, 0, 80],  # Pilot
                        # [1, 0.5, 0, 80],  # Passenger
                        # [1, -0.5, 0, 80], # Passenger
                        # [2, -0.5, 0, 80],  # Passenger
                        # [2, 0.5, 0, 80],  # Passenger
                        # [1, 0.5, 0, 8],  # Passenger luggage
                        # [1, -0.5, 0, 8], # Passenger luggage
                        # [2, -0.5, 0, 8],  # Passenger luggage
                        # [2, 0.5, 0, 8],  # Passenger luggage
                        [-0.5, 0, 0, 20],  # Pilot seat
                        [1, 0.5, 0, 20],  # Passenger seat
                        [1, -0.5, 0, 20], # Passenger seat
                        [2, -0.5, 0, 20],  # Passenger seat
                        [2, 0.5, 0, 20],  # Passenger seat
                        [1+0.75, 0, -0.9, 350], # Fuel Cells
                        [2, -0.5, -0.9, 140], # H2 tank 1
                        [2, 0.5, -0.9, 140], # H2 tank 2
                        [1.75, 0, -0.9, 70],  # Fuel Cell Aux systems
                        [-0.5, 0.5, -0.25, 20],  # Avionics
                        [4.5, 0, -0.25, 16],  # Weather system
                        [2, 0, 0, 750],  # COM of structure
                        [0, 0, 0, 0],],
                       dtype=float)


mass_coords_top = copy.deepcopy(mass_coords)
mass_coords_top[:, 0] += nose_length
mass_coords_top[:, 1] += aircraft_size[1]/2

mass_top_empty = copy.deepcopy(mass_coords_empty)
mass_top_empty[:, 0] += nose_length
mass_top_empty[:, 1] += aircraft_size[1]/2

comx = 0
comy = 0
comx_empty = 0
comy_empty = 0


for i, _ in enumerate(mass_coords_top):
    massplot.scatter(mass_coords_top[i, 0] * top_x_conv, 
                     mass_coords_top[i, 1] * top_y_conv, 
                     s=mass_coords_top[i, 3]*30, 
                     color='k',
                     alpha=alpha)
    comx += mass_coords_top[i, 0] * mass_coords_top[i, 3] * top_x_conv
    comy += mass_coords_top[i, 1] * mass_coords_top[i, 3] * top_y_conv
    
for i, _ in enumerate(mass_top_empty):
    comx_empty += mass_top_empty[i, 0] * mass_top_empty[i, 3] * top_x_conv
    comy_empty += mass_top_empty[i, 1] * mass_top_empty[i, 3] * top_y_conv
    
comx /= np.sum(mass_coords_top[:, 3])
comy /= np.sum(mass_coords_top[:, 3])

comx_empty /= np.sum(mass_top_empty[:, 3])
comy_empty /= np.sum(mass_top_empty[:, 3])



plt.annotate(f"Empty CoM ({(comx_empty+nose_length)/top_x_conv:.2f}m, {comy_empty/top_y_conv:.2f}m)",
                  xy=(comx_empty, comy_empty),
                  xytext=(comx_empty+100, comy_empty-100),
                  arrowprops=dict(facecolor='green', shrink=0.1))

massplot.scatter(comx_empty, comy_empty, s=800, color='green')



plt.annotate(f"Center of Mass ({(comx+nose_length)/top_x_conv:.2f}m, {comy/top_y_conv:.2f}m)",
                  xy=(comx, comy),
                  xytext=(comx+100, comy+100),
                  arrowprops=dict(facecolor=cg_color, shrink=0.1))

massplot.scatter(comx, comy, s=800, color=cg_color)


massplot.imshow(img, alpha=0.3)
massplot.set_title(general_graph_title + "(Top View)")
massplot.set_xlabel("Distance From Aircaft Nose $x$ $_{[m]}$")
massplot.set_ylabel("Distance From Aircaft Centerline $y$ $_{[m]}$")

# Tick modification
xticks = massplot.get_xticks()
yticks = massplot.get_yticks()
for i, xtick in enumerate(xticks):
    xticks[i] = f"{xtick/top_x_conv:.2f}"
    
for i, ytick in enumerate(yticks):
    yticks[i] = f"{ytick/top_y_conv - aircraft_size[1]/2:.2f}"

massplot.set_xticklabels(xticks)
massplot.set_yticklabels(yticks)

# meters = mpl.ticker.EngFormatter("m")
plt.show()
#%%###########################

fig, massplot_side = plt.subplots()
massplot_side.imshow(img2, alpha=0.3)

z_offset = -1.25-skid_height
side_x_conv = img2.shape[1] / aircraft_size[0]
side_z_conv = img2.shape[0] / aircraft_size[2]

mass_coords_side = copy.deepcopy(mass_coords)
coord2img = lambda z: img2.shape[0] - z * side_z_conv
mass_coords_side[:, 0] += nose_length
mass_coords_side[:, 2] -= z_offset
mass_coords_side[:, 2] = coord2img(mass_coords_side[:, 2])

mass_side_empty = copy.deepcopy(mass_coords_empty)
mass_side_empty[:, 0] += nose_length
mass_side_empty[:, 2] -= z_offset
mass_side_empty[:, 2] = coord2img(mass_side_empty[:, 2])


comx, comz = 0, 0
comx_empty, comz_empty = 0, 0
for i, _ in enumerate(mass_coords_side):
    massplot_side.scatter(mass_coords_side[i, 0] * side_x_conv, 
                          mass_coords_side[i, 2], 
                          s=mass_coords_top[i, 3]*30,
                          color='k',
                          alpha = 0.4)
    comx += mass_coords_side[i, 0] * mass_coords_side[i, 3] * side_x_conv
    comz += mass_coords_side[i, 2] * mass_coords_side[i, 3]
    
for i, _ in enumerate(mass_side_empty):
    comx_empty += mass_side_empty[i, 0] * mass_side_empty[i, 3] * side_x_conv
    comz_empty += mass_side_empty[i, 2] * mass_side_empty[i, 3]

comx /= np.sum(mass_coords_side[:, 3])
comz /= np.sum(mass_coords_side[:, 3])

comx_empty /= np.sum(mass_side_empty[:, 3])
comz_empty /= np.sum(mass_side_empty[:, 3])

plt.annotate(f"Empty CoM ({(comx_empty+nose_length)/side_x_conv:.2f}m, {comz_empty/side_z_conv:.2f}m)",
             xy=(comx_empty, comz_empty),
             xytext=(comx_empty+100, comz_empty-100),
             arrowprops=dict(facecolor='green', shrink=0.1))

massplot_side.scatter(comx_empty, comz_empty, s=800, color='green')


plt.annotate(f"Center of Mass ({(comx+nose_length)/side_x_conv:.2f}m, {comz/side_z_conv:.2f}m)",
             xy=(comx, comz),
             xytext=(comx+100, comz+100),
             arrowprops=dict(facecolor=cg_color, shrink=0.1))

massplot_side.scatter(comx, comz, s=800, color=cg_color)
massplot_side.set_title(general_graph_title + "(Side View)")
massplot_side.set_xlabel("Distance From Aircaft Nose $x$ $_{[m]}$")
massplot_side.set_ylabel("Distance From Top of Fuselage $z$ $_{[m]}$")


# Tick modification
xticks = massplot_side.get_xticks()
zticks = massplot_side.get_yticks()
for i, xtick in enumerate(xticks):
    xticks[i] = f"{xtick/side_x_conv:.2f}"
    
for i, ztick in enumerate(zticks):
    zticks[i] = f"{ztick/side_z_conv:.2f}"

massplot_side.set_xticklabels(xticks)
massplot_side.set_yticklabels(zticks)
#%%###########################

fig, massplot_front = plt.subplots()
img3 = plt.imread("Images/Aircraft/AircraftFrontView.png")

front_y_conv = img3.shape[1] / aircraft_size[1]
front_z_conv = img3.shape[0] / aircraft_size[2]

massplot_front.imshow(img3, alpha=0.3)

mass_coords_front = copy.deepcopy(mass_coords)
coord2img = lambda z: img3.shape[0] - z * front_z_conv
mass_coords_front[:, 2] -= z_offset
mass_coords_front[:, 2] = coord2img(mass_coords_front[:, 2])
mass_coords_front[:, 1] += aircraft_size[1]/2

mass_coords_front[:, 1] *= front_y_conv
# mass_coords_front[:, 2] *= front_z_conv

mass_front_empty = copy.deepcopy(mass_coords_empty)
mass_front_empty[:, 2] -= z_offset
mass_front_empty[:, 2] = coord2img(mass_front_empty[:, 2])
mass_front_empty[:, 1] += aircraft_size[1]/2
mass_front_empty[:, 1] *= front_y_conv


massplot_front.set_title(general_graph_title + "(Front View)")
massplot_front.set_xlabel("Distance From Aircaft Centerline $y$ $_{[m]}$")
massplot_front.set_ylabel("Distance From Top of Fuselage $z$ $_{[m]}$")

comy, comz = 0, 0
comy_empty, comz_empty = 0, 0

for i, _ in enumerate(mass_coords_front):
    massplot_front.scatter(mass_coords_front[i, 1], 
                           mass_coords_front[i, 2], 
                           s=mass_coords_front[i, 3]*30, 
                           color='k',
                           alpha=alpha)
    comy += mass_coords_front[i, 1] * mass_coords_front[i, 3]
    comz += mass_coords_front[i, 2] * mass_coords_front[i, 3]
    
for i, _ in enumerate(mass_front_empty):
    comy_empty += mass_front_empty[i, 1] * mass_front_empty[i, 3]
    comz_empty += mass_front_empty[i, 2] * mass_front_empty[i, 3]

comy /= np.sum(mass_coords_front[:, 3])
comz /= np.sum(mass_coords_front[:, 3])

comy_empty /= np.sum(mass_front_empty[:, 3])
comz_empty /= np.sum(mass_front_empty[:, 3])

plt.annotate(f"Empty CoM ({comy_empty/front_y_conv:.2f}m, {comz_empty/front_z_conv:.2f}m)",
             xy=(comy_empty, comz_empty),
             xytext=(comy_empty+100, comz_empty-100),
             arrowprops=dict(facecolor='green', shrink=0.1))

massplot_front.scatter(comy_empty, comz_empty, s=800, color='green')

plt.annotate(f"Center of Mass ({comy/front_y_conv:.2f}m, {comz/front_z_conv:.2f}m)",
             xy=(comy, comz),
             xytext=(comy+100, comz+100),
             arrowprops=dict(facecolor=cg_color, shrink=0.1))

massplot_front.scatter(comy, comz, s=800, color=cg_color)

# Tick modification
yticks = massplot_front.get_xticks()
zticks = massplot_front.get_yticks()
for i, ytick in enumerate(yticks):
    yticks[i] = f"{ytick/front_y_conv - aircraft_size[1]/2:.2f}"
    
for i, ztick in enumerate(zticks):
    zticks[i] = f"{ztick/front_z_conv:.2f}"

massplot_front.set_xticklabels(yticks)
massplot_front.set_yticklabels(zticks)

print(f"Total mass: {np.sum(mass_coords_front[:, 3]):.2f} kg")
