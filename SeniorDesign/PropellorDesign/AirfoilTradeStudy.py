#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:16:47 2021

@author: jack
"""

"""
TO DO
    - Write properties for beta, alpha_0
    - Figure out how to guess v_0
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import time
import pandas as pd
import ambiance as amb
from propfuncs import *
import os

# plt.style.use("default")
# plt.style.use("seaborn-bright")
airfoil_files = [file_name for file_name in os.listdir("Data/Airfoils")]
airfoil_files.sort()

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
    'axes.prop_cycle': mpl.cycler('color', plt.cm.jet(np.linspace(0, 1, len(airfoil_files)))),
    'figure.figsize':(24,16),#square plots
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

ft_format = mpl.ticker.EngFormatter(unit="m")
rad_format = mpl.ticker.EngFormatter(unit=r"$\degree$")
pow_format = mpl.ticker.EngFormatter(unit="W")
thr_format = mpl.ticker.EngFormatter(unit="N")



# print(prop.head())
fig, liftplot = plt.subplots()
fig, dragpolar = plt.subplots()
plt.suptitle("Airfoil Test Data")
fig, cplot = plt.subplots()
fig, betaplot = plt.subplots()
# fig, tcplot = plt.subplots()
fig, clplot = plt.subplots()
# fig, cDplot = plt.subplots()
fig, etaplot = plt.subplots()
plt.ylim([0, 1])

# fig, rpmplot = plt.subplots()
# fig, powerplot = plt.subplots()
# plt.ylim([0, atmos.P_eng])
fig, coeffplot = plt.subplots()
fig, thrustplot = plt.subplots()

cplot.yaxis.set_major_formatter(ft_format)
betaplot.yaxis.set_major_formatter(rad_format)
# powerplot.yaxis.set_major_formatter(pow_format)
thrustplot.yaxis.set_major_formatter(thr_format)

#%%###########################
ambi = amb.Atmosphere(700)


RPM = 4000/3

mass = 2350 - 8 * (49 - 15)
g_0 = 9.81

# 50 m**2 swept

B = 3  # blades
D = 2.82 # m
c_l = 0.4
num_el = 86  # Number Elements
alpha_0 = np.deg2rad(0)
Q_des = 500*3  # Nm

DL = mass*g_0 / (D**2 * np.pi / 4)

V = np.sqrt( DL / 2*ambi.density) # m/s

# V = 45

print(f"Inlet velocity = {V}")

atmos = FlightConditions(V,
                         ambi.density,
                         ambi.dynamic_viscosity,
                         ambi.speed_of_sound,
                         RPM)
atmos.P_eng = 200E3 #260E3  # W



x_list = np.linspace(0.15, 1, num_el)

prop = pd.DataFrame()

prop["x"] = x_list
prop["c_l"] = c_l * np.ones_like(x_list)
# prop["t_over_c"] = 0.04 / (prop.x**1.2)
prop["D"] =  D * np.ones_like(x_list)
prop["R"] =  0.5 * prop.D
prop["r"] = prop.R * prop.x
prop["B"] = B * np.ones_like(x_list)
prop["alpha_0"] = alpha_0 * np.ones_like(x_list)

atmos.v_0 = 5.04

column_names = ["airfoil",
                    "C_T",
                    "C_Q",
                    "C_P",
                    "J",
                    "T",
                    "Q",
                    "P",
                    "eta_P",
                    "eta_P_max",
                    "AF",
                    "C_L",
                    "beta75"]

airfoil_table = pd.DataFrame(columns = column_names)
for airfoil_name in airfoil_files:
    # Reset Properties
    atmos.V = V
    df = pd.read_csv(f"Data/Airfoils/{airfoil_name}", skiprows=10)
    prop.alpha_0 = np.deg2rad(np.interp(0, df.Cl, df.Alpha))

    airfoil_name = airfoil_name.split('.')[0]
    plothusly(liftplot,
              df.Alpha,
              df.Cl,
              title="Airfoil Lift Curves",
              xtitle=r"Angle of Attack $\alpha$",
              ytitle=r"Sectional Lift Coefficient $C_l$",
              datalabel=airfoil_name)
    plothusly(dragpolar,
              df.Cl,
              df.Cd,
              title="Airfoil Drag Polar",
              xtitle=r"Sectional Lift Coefficient $C_l$",
              ytitle=r"Sectional Drag Coefficient $C_d$",
              datalabel=airfoil_name)
    prop.alpha_0 = np.deg2rad(np.interp(0, df.Cl, df.Alpha))
    prop["c_l"] = c_l * np.ones_like(x_list)
    prop = PropellorDesign(atmos, prop, df)
    
    
    dic = PropellorAnalysis(atmos, prop, df)
    dic["airfoil"] = airfoil_name
    
    # FM = prob2results["C_T"][0]**(3/2) / (np.sqrt(2)*prob2results["C_P"][0])
    
    # print(f"FM at max: {FM}")
    
    print("\n")
    print(airfoil_name)

    mass_test = pd.DataFrame(columns=column_names)
    
    
    V_list = np.linspace(0, 200, 201)
    
    # RPM_list = np.linspace(2, 400, 50)
    
    # n_list = RPM_list/60
    
    for i, atmos.V in enumerate(V_list):
        prop_line = PropellorAnalysis(atmos, prop, df)
        prop_line["airfoil"] = airfoil_name
        if prop_line["eta_P"] > 1 or prop_line["eta_P"] < 0:
            break
        else:
            mass_test = mass_test.append(prop_line, ignore_index=True)
    print("Analysis Complete!")
    dic["eta_P_max"] = np.max(mass_test.eta_P)
    airfoil_table = airfoil_table.append(dic, ignore_index=True)
    plothusly(cplot, prop.x, prop.c, 
            xtitle=r"Propeller Length $r$", 
            ytitle=r"Chord length $c$ $_{[m]}$", 
            title=r"ATP-XW Propeller: Chord Length $c$",
            datalabel=fr"{airfoil_name}")
    plothusly(betaplot, prop.x, np.rad2deg(prop.beta), 
            xtitle=r"Fraction of Propeller Blade Propeller Length $r$", 
            ytitle=r"Pitch Angle $\beta$ $_{[\degree]}$", 
            title=r"ATP-XW Propeller: Pitch Angle $\beta$",
            datalabel=airfoil_name
            )
    # plothusly(tcplot, prop.x, prop.t_over_c,  
    #         xtitle=r"Fraction of Propeller Blade Propeller Length $r$", 
    #         ytitle=r"Propeller Thickness Fraction $\frac{t}{c}$", 
    #         title=r"ATP-XW Propeller: Blade $\frac{t}{c}$",
    #         datalabel=airfoil_name)
    plothusly(clplot, prop.x, prop.c_l, 
            xtitle=r"Propeller Length $r$", 
            ytitle=r"$C_l$", 
            title="C_l Plot",
            datalabel=airfoil_name)
    # plothusly(cDplot, prop.x, prop.c/prop.D, 
    #           xtitle=r"Propeller Length $r$",
    #           ytitle=r"$\frac{c}{D}$",
    #           title=r"Propeller chord/diamter ratio",)
    
    
    
    plothusly(etaplot,
              mass_test.J,
              mass_test.eta_P, 
              xtitle=r"Advance Ratio $J$", 
              ytitle=r"Propellor Efficiency $\eta_P$", 
              title=r"$\eta_P$ Plot",
              datalabel=fr"{airfoil_name}")
    # plothusly(powerplot,
    #           mass_test.J,
    #           mass_test.P, 
    #           xtitle=r"Advance Ratio $J$", 
    #           ytitle=r"Propellor Power $P$ [BHP]", 
    #           title=r"$P$ Plot",
    #           # datalabel=airfoil_name
    #           )
    plothusly(coeffplot,
              mass_test.J,
              mass_test.C_T, 
              xtitle=r"Advance Ratio $J$", 
              ytitle=r"Coefficient Value", 
              title=r"$C_T$ Plot",
              datalabel=airfoil_name)
    # plothus(coeffplot,
    #         mass_test.J,
    #         mass_test.C_Q,
    #         datalabel=r"$C_Q$")
    # plothus(coeffplot,
    #         mass_test.J,
    #         mass_test.C_P,
    #         datalabel=r"$C_P$")
    
    plothusly(thrustplot,
              mass_test.J,
              mass_test["T"], 
              xtitle=r"Advance Ratio $J$", 
              ytitle=r"Propellor Thrust $T$ [N]", 
              title=r"$T$ Plot",
              datalabel=airfoil_name
              )
    
    # plothusly(rpmplot,
    #           RPM_list[:len(mass_test["T"])],
    #           mass_test["T"],
    #           xtitle="Engine RPM",
    #           ytitle="Propeller Thrust [N]",
    #           title="Propeller Thrust Curve")
    # break
airfoil_table["P"] /= 1E3
airfoil_table["T"] /= 1E3

#%%###########################

print_col_ind = ["airfoil", "T", "C_T", "Q", "P", "eta_P"]

print_col = airfoil_table[print_col_ind]

print_col["Score"] = 2*print_col["C_T"] + print_col["eta_P"]

print(print_col.to_latex(float_format="%.3f", index=False))
print(f"V_0 = {atmos.v_0}")

TWR = 8 * np.max(airfoil_table["T"]) * 1E3 / (mass * 9.81)

print(f"Optimum TWR of tested airfoils: {TWR}")