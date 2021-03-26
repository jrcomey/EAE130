#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:03:11 2021

@author: jack
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

def AddMotor(name, power, Q, RPM, mass):
    dic = {"Motor Name": name,
           "Continous Power": power,
           "Torque": Q,
           "RPM": RPM,
           "Mass": mass}
    
    return dic

# plt.style.use("default")
# plt.style.use("seaborn-bright")
airfoil_name = "BW FX 69-H-083.csv"

motor_list = [
              # AddMotor("H3X 1:1", 200E3, 95, 20E3, 15),
              # AddMotor("H3X 2:1", 200E3, 95*2, 20E3/2, 15),
              # AddMotor("H3X 4:1", 200E3, 95*4, 20E3/4, 15),
              # AddMotor("H3X 5:1", 200E3, 95*5, 20E3/5, 15),
              # AddMotor("H3X 8:1", 200E3, 95*8, 20E3/8, 15),
              # AddMotor("H3X 10:1", 200E3, 95*10, 20E3/10, 15),
              # AddMotor("SP200D", 204E3, 1500, 1300, 49),
              # AddMotor("SP260D", 260E3, 1000, 2000, 50),
              # AddMotor("SP260D-A", 260E3, 1000, 2000, 44),
              # AddMotor("Magni250", 280E3, 1407, 1900, 71),
              # AddMotor("Emrax 268", 107E3, 250, 4500, 20),
              AddMotor("Emrax 348 3:1", 210E3, 500*3, 4000/3, 41),
              ]

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
    'axes.prop_cycle': mpl.cycler('color', plt.cm.jet(np.linspace(0, 1, len(motor_list)))),
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
fig, liftplot = plt.subplots(1, 2, figsize = (24, 16))
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

gearbox = 3

RPM = 4E3 / gearbox

mass = 1958 # 
g_0 = 9.81

# 50 m**2 swept

B = 3  # blades
D = 2.82 # m
c_l = 0.4
num_el = 86  # Number Elements
alpha_0 = np.deg2rad(0)
Q_des = 500 * gearbox

DL = mass*g_0 / (D**2 * np.pi / 4)

V = np.sqrt( DL / 2*ambi.density) # m/s

# V = 45

# print(f"Inlet velocity = {V}")

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

column_names = ["motor",
                    "C_T",
                    "C_Q",
                    "C_P",
                    "J",
                    "T",
                    "Q",
                    "TWR"
                    "P",
                    "eta_P",
                    "eta_P_max",
                    "AF",
                    "C_L",
                    "beta75"]

airfoil_table = pd.DataFrame(columns = column_names)
df = pd.read_csv(f"Data/Airfoils/{airfoil_name}", skiprows=10)
for motor in motor_list:
    # Reset Properties
    DL = (mass + 8*motor["Mass"]) *g_0 / (D**2 * np.pi / 4)

    V = np.sqrt( DL / 2*ambi.density) # m/s
    print(f"Inlet velocity = {V}")
    atmos = FlightConditions(V,
                         ambi.density,
                         ambi.dynamic_viscosity,
                         ambi.speed_of_sound,
                         motor["RPM"])
    atmos.v_0 = 5.04
    atmos.P_eng = motor["Continous Power"]
    prop.alpha_0 = np.deg2rad(np.interp(0, df.Cl, df.Alpha))

    airfoil_name = airfoil_name.split('.')[0]
    plothusly(liftplot[0],
              df.Alpha,
              df.Cl,
              title="Airfoil Lift Curves",
              xtitle=r"Angle of Attack $\alpha$",
              ytitle=r"Sectional Lift Coefficient $C_l$",
              datalabel=airfoil_name)
    plothusly(liftplot[1],
              df.Cl,
              df.Cd,
              title="Airfoil Drag Polar",
              xtitle=r"Sectional Lift Coefficient $C_l$",
              ytitle=r"Sectional Drag Coefficient $C_d$",
              datalabel=airfoil_name)
    airfoil_name = motor["Motor Name"]
    prop.alpha_0 = np.deg2rad(np.interp(0, df.Cl, df.Alpha))
    prop["c_l"] = c_l * np.ones_like(x_list)
    prop = PropellorDesign(atmos, prop, df)
    
    
    dic = PropellorAnalysis(atmos, prop, df)
    dic["motor"] = airfoil_name
    dic["TWR"] = dic["T"] / motor["Mass"]
    
    
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
        prop_line["motor"] = airfoil_name
        if prop_line["eta_P"] > 1 or prop_line["eta_P"] < 0:
            break
        else:
            mass_test = mass_test.append(prop_line, ignore_index=True)
    print("Analysis Complete!")
    dic["eta_P_max"] = np.max(mass_test.eta_P)
    airfoil_table = airfoil_table.append(dic, ignore_index=True)
    plothusly(cplot, prop.r, prop.c, 
            xtitle=r"Fraction of Propeller Blade Propeller Length $r$", 
            ytitle=r"Chord length $c$ $_{[m]}$", 
            title=r"ATP-XW Propeller: Chord Length $c$",
            datalabel=fr"{airfoil_name}")
    plothusly(betaplot, prop.r, np.rad2deg(prop.beta), 
            xtitle=r"Propeller Length $r$", 
            ytitle=r"Pitch Angle $\beta$ $_{[\degree]}$", 
            title=r"ATP-XW Propeller: Pitch Angle $\beta$",
            datalabel=airfoil_name
            )
    # plothusly(tcplot, prop.x, prop.t_over_c,  
    #         xtitle=r"Fraction of Propeller Blade Propeller Length $r$", 
    #         ytitle=r"Propeller Thickness Fraction $\frac{t}{c}$", 
    #         title=r"ATP-XW Propeller: Blade $\frac{t}{c}$",
    #         datalabel=airfoil_name)
    plothusly(clplot, prop.r, prop.c_l, 
            xtitle=r"Propeller Length $r$", 
            ytitle=r"$C_l$", 
            title="ATP-XW Propeller: $C_l$",
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
              title=r"ATP-XW Propeller: $\eta_P$ Plot",
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
              title=r"ATP-XW Propeller: $C_T$ Plot",
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
              title=r"ATP-XW Propeller: $T$ Plot",
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

print_col_ind = ["motor", "T", "C_T", "Q", "P", "eta_P"]

print_col = airfoil_table[print_col_ind]

print_col["Score"] = 2*print_col["C_T"] + print_col["eta_P"]

print(print_col.to_latex(float_format="%.2f", index=False))
print(f"V_0 = {atmos.v_0}")

TWR = 8 * np.max(airfoil_table["T"]) * 1E3 / (mass * 9.81)

print(f"Optimum TWR of tested airfoils: {TWR}")