#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:03:59 2021

@author: jack
"""

"""

Example Code
"""

import UAVsym as usy
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from jrc import *
import montecarlofunctions as mcf

#%%###########################

# Design Variables (Touch these ones)

num_iterations = 5
max_random_pos = 0
max_att = np.pi
sim_time = 30
mute = False
alpha_val = 0.5

#%%###########################

# Aircraft Design, copy/paste
import UAVsym as usy
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import montecarlofunctions as mcf
from jrc import *

run_time = 10

# Defining motor thrust curve

max_thrust = 4.2E3   # Newtons

omega = np.linspace(0, 2500, 1000)  # Values between 0 and 1300 rad/s
thrust = max_thrust - max_thrust*np.exp(-omega/100)

# Creating base motor object

motor = usy.Motor(1E3)  # Set motor object with 100ms max PWM signal width
motor.SetTau(0.1)  # Set motor time constant in seconds
motor.SetThrustCurve(omega, thrust)  # Set motor thrust curve

# Defining UAV inertial properties
mass = 1958 + 8*(40)  # kg
# mass = 2200
Ixx = 1000  # kg-m^2
Iyy = 800 # kg-m^2
Izz = 800  # kg-m^2


num_motors = 8  # Number of UAV motors
clock_speed = 2.1E9  # Clock speed in Hz


mixer = np.array([[0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # X Forces
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Y Forces
                  [-1, -1, -1, -1, -1, -1, -1, -1],  # Z Forces
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty 
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty 
                  [2, -2, 2, -2, 2, -2, 2, -2],  # X Moments (Roll)
                  [2.5, 2.5, -2.5, -2.5, 2.5, 2.5, -2.5, -2.5],  # Y Moments (Pitch)
                  [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=float)  # Z Moments (Yaw)

drone = usy.UAV(mass, Ixx, Iyy, Izz, num_motors,
                motor, mixer, clock_speed)

drone.Setdt(0.001)  # Set time step size

# Ziegler-Nichols Tuning Rule

T = 100
L = 10

T_roll = 5
L_roll = 2

K_P_factor = 60
K_D_factor = 40

K_P_pos_factor = 4
K_I_factor = 0.48 #0.48
K_D_pos_factor = 1.3

K_P = K_P_factor * (1.2 * T_roll/L_roll * Ixx)  # P constant, angular
K_I = K_I_factor * 0.6 * T/L**2 # I constant, angular
K_D = K_D_factor * (0.6 * T_roll * Ixx)  # D constant, angular

K_P_pos = K_P_pos_factor * 1.2 * T/L * mass  # P constant, altitude
K_D_pos = K_D_pos_factor * (0.6 * T * mass)  # D constant, altitude

K_P_pos_xy = 0.1*K_P  # XY translational P constant
K_D_pos_xy = 0.3*K_D   # XY translational D constant

drone.SetPIDPD(K_P,
               K_I,
               K_D,
               K_P_pos,
               K_D_pos,
               K_P_pos_xy,
               K_D_pos_xy)

#%%###########################

# Counters

# Iteration counter
iter_count = 0
total_real_time = 0


#%%##########################

# Plotting stuff

colors = sns.color_palette(palette="bright", n_colors=5)

metres = mpl.ticker.EngFormatter("m")
newtons = mpl.ticker.EngFormatter("N")
seconds = mpl.ticker.EngFormatter("s")
radians = mpl.ticker.EngFormatter("rad")
degrees = mpl.ticker.EngFormatter(r"$\degree$")

fig, posplot = plt.subplots()
fig, attplot = plt.subplots()

#%%###########################

# Monte Carlo Loop:

for i in range(num_iterations):
    
    # The simulation
    # drone = usy.UAV(mass, Ixx, Iyy, Izz, num_motors,
    #                 motor, mixer, clock_speed)
    # drone.Setdt(0.001)
    # drone.SetPIDPD(K_P,
    #                K_I,
    #                K_D,
    #                K_P_pos,
    #                K_D_pos,
    #                K_P_pos_xy,
    #                K_D_pos_xy)
    tic = time.time()
    drone.Reset()
    drone = mcf.RandomizeDronePosition(drone, max_random_pos, mag_att=max_att)
    drone.RunSim(sim_time)
    toc = time.time()
    tictoc = toc-tic
    df = drone.ExportData()
    
    # The analysis
    
    df["Z Position"] *= -1
    
    # Plotting
    
    
    if iter_count == 0:
        # Xpos
        posplot.plot(df["Time"],
                     df["X Position"],
                     color=colors[0],
                     label="X Position",
                     alpha=alpha_val)
        # Y Pos
        posplot.plot(df["Time"],
                     df["Y Position"],
                     color=colors[2],
                     label="Y Position",
                     alpha=alpha_val)
        # Z Plot
        posplot.plot(df["Time"],
                     df["Z Position"],
                     color=colors[3],
                     label="Z Position",
                     alpha=alpha_val)
        # Pitch
        attplot.plot(df["Time"],
                     np.rad2deg(df["Pitch"]),
                     color=colors[0],
                     label="Pitch",
                     alpha=alpha_val)
        # Yaw
        attplot.plot(df["Time"],
                     np.rad2deg(df["Yaw"]),
                     color=colors[2],
                     label="Yaw", 
                     alpha=alpha_val)
        # Roll
        attplot.plot(df["Time"],
                     np.rad2deg(df["Roll"]),
                     color=colors[3],
                     label="Roll",
                     alpha=alpha_val)
        
    else:
        posplot.plot(df["Time"],
                     df["X Position"],
                     color=colors[0],
                     alpha=alpha_val)
        posplot.plot(df["Time"],
                     df["Y Position"],
                     color=colors[2],
                     alpha=alpha_val)
        posplot.plot(df["Time"],
                     df["Z Position"],
                     color=colors[3],
                     alpha=alpha_val)
        # Pitch
        attplot.plot(df["Time"],
                     np.rad2deg(df["Pitch"]),
                     color=colors[0],
                     alpha=alpha_val)
        # Yaw
        attplot.plot(df["Time"],
                     np.rad2deg(df["Yaw"]),
                     color=colors[2],
                     alpha=alpha_val)
        # Roll
        attplot.plot(df["Time"],
                     np.rad2deg(df["Roll"]),
                     color=colors[3],
                     alpha=alpha_val)

    # Print Code
    if mute == False:
        print(f"Simulation {i+1} of {num_iterations} completed.")
        print(f"Time Elapsed: {tictoc:.2f} seconds")
    else:
        pass
    iter_count += 1
    total_real_time += tictoc

#%%###########################

# Graph Post Processing

posplot.set_title("NED UAM Position")
posplot.set_xlabel(r"Time $_{[s]}$")
posplot.set_ylabel(r"Position $_{[m]}$")
posplot.legend()

attplot.set_title("NED UAM Attitude")
attplot.set_xlabel(r"Time $_{[s]}$")
attplot.set_ylabel(r"Attitude $_{[\degree]}$")
attplot.legend()

posplot.xaxis.set_major_formatter(seconds)
attplot.xaxis.set_major_formatter(seconds)

posplot.yaxis.set_major_formatter(metres)
attplot.yaxis.set_major_formatter(degrees)

print("\n"); print("\n")
print("FINAL REPORT"); print("\n")
print(f"Total Time Elapsed: {total_real_time:.2f} seconds")
print(f"Average Simulation Time: {total_real_time/num_iterations:.2f} seconds")
# Set initial state to showcase control

# drone.Reset()

# drone.state_vector[0] = 0
# drone.state_vector[1] = 2
# drone.state_vector[2] = 0
# drone.state_vector[5] = -20
# drone.state_vector[6] = 0
# drone.state_vector[7] = 0.5
# drone.state_vector[8] = 0

# drone.final_state[2] = -100
# # Main loop:
# tic = time.time()
# drone.RunSim(1)
# toc = time.time()
# tictoc = toc-tic
# print(f"Simulation ran for {tictoc:.2f} seconds")
# # Exports data to pandas dataframe
# df = drone.ExportData()

#%%###########################

# # For plotting results only! Not contained in sample code.


# # Position plot
# fig, zplot = plt.subplots()
# plothusly(zplot, df["Time"], df["Z Position"], 
#           xtitle="Time in seconds",
#           ytitle="Position in metres", 
#           datalabel="Z Position", 
#           title="NED Drone Position")
# plothus(zplot, df["Time"], df["Y Position"], datalabel="Y Position")
# plothus(zplot, df["Time"], df["X Position"], datalabel="X Position")

# zplot.xaxis.set_major_formatter(seconds)
# zplot.yaxis.set_major_formatter(meters)

# # Attitude plots
# fig, angleplot = plt.subplots()
# plothusly(angleplot, df["Time"], df["Pitch"], 
#           xtitle="Time in seconds",
#           ytitle="Angle from neutral position in radians", 
#           datalabel="Pitch", 
#           title="NED Aircraft Attitude")
# plothus(angleplot, df["Time"], df["Yaw"], datalabel="Yaw")
# plothus(angleplot, df["Time"], df["Roll"], datalabel="Roll")
# angleplot.xaxis.set_major_formatter(seconds)
# angleplot.yaxis.set_major_formatter(radians)


# # Motor Signals
# fig, signalplot = plt.subplots()
# plothusly(signalplot, 0, 0, xtitle=r"Time [s]", ytitle=r"Motor Signal", datalabel='', title="Motor Signal Plot")

# for i, motor in enumerate(drone.motors):
#     plothus(signalplot, df["Time"], df[f"Motor {i} Signal"], datalabel=f"Motor {i} Signal")
    
# signalplot.xaxis.set_major_formatter(seconds)
# # signalplot.yaxis.set_major_formatter()

    
# fig, thrustplot = plt.subplots()
# plothusly(thrustplot, 0, 0, xtitle=r"Time [s]", ytitle=r"Motor Thrust", datalabel='', title="Motor Thrust Plot")

# for i, motor in enumerate(drone.motors):
#     plothus(thrustplot, df["Time"], df[f"Motor {i} Force"], datalabel=f"Motor {i} Thrust")
    
# thrustplot.xaxis.set_major_formatter(seconds)
# thrustplot.yaxis.set_major_formatter(newtons)