#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:03:59 2021

@author: jack
"""

"""

Example Code
"""

import UAVsym_new as usy
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from jrc import *
from montecarlofunctions import *
import multiprocessing
import pandas as pd
import copy
from scipy.fft import fft, fftfreq
import os
#%%###########################

# Design Variables (Touch these ones)

num_iterations = 10
max_random_pos = 0
max_att = np.deg2rad(20)
sim_time = 60
mute = False
alpha_val = 0.5
n = 5
gamma_list = [0]
misc_plots = True
aircraft_names = ["Blizzard Attitude Response"]
thrust_max = 3.8E3

#%%###########################

def RunMC(drone, max_random_pos, max_att, sim_time, mute, alpha_val, procnum):
    """
    Runs a single randomized Simulation

    Parameters
    ----------
    drone : UAV object
    max_random_pos : Maximum random translational position
    max_att : Maximum random rotational position
    sim_time : Simulated time
    mute : Print reports on/off
    alpha_val : Transparancy of lines on plots
    procnum : What number iteration it is

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    tic = time.time()
    drone.Reset()
    drone = RandomizeDronePosition(drone, max_random_pos, mag_att=max_att)
    drone.RunSim(sim_time)
    toc = time.time()
    tictoc = toc-tic
    if mute == False:
        print(f"Run {procnum} completed with {tictoc:.2f} s")
    df = drone.ExportData()
    return df
    # return iter_count, tictoc, posplot, attplot

def FFTAnalysis(df_dict, pos_freq_plot, freq_plot):
    
    global xpos_freq
    global ypos_freq
    global zpos_freq
    global pitch_freq
    global roll_freq
    global yaw_freq

    xpos_freq = 0
    ypos_freq = 0
    zpos_freq = 0
    pitch_freq = 0
    roll_freq = 0
    yaw_freq = 0
    for i, df in enumerate(df_dict.values()):
        xpos_freq += fft(df["X Position"].to_numpy())
        
                    # ypos
        ypos_freq += fft(df["Y Position"].to_numpy())
        
                    # zpos
        zpos_freq += fft(df["Z Position"].to_numpy())
        
                    # Pitch
        pitch_freq += fft(df["Pitch"].to_numpy())
        
                    # Yaw
        yaw_freq += fft(df["Yaw"].to_numpy())
        
                    # Roll
        roll_freq += fft(df["Roll"].to_numpy())

        N = len(df["Time"])
        xf = fftfreq(N, drone.dt)[:N//2]
    
    xpos_freq = np.abs(xpos_freq[0:N//2])
    ypos_freq = np.abs(ypos_freq[0:N//2])
    zpos_freq = np.abs(zpos_freq[0:N//2])
    pitch_freq = np.abs(pitch_freq[0:N//2])
    yaw_freq = np.abs(yaw_freq[0:N//2])
    roll_freq = np.abs(roll_freq[0:N//2])
    
    xpos_freq /= i+1
    ypos_freq /= i+1
    zpos_freq /= i+1
    pitch_freq /= i+1
    roll_freq /= i+1
    yaw_freq /= i+1
    
    xpos_freq /= np.max(np.abs(xpos_freq))
    ypos_freq /= np.max(np.abs(ypos_freq))
    zpos_freq /= np.max(np.abs(zpos_freq))
    pitch_freq /= np.max(np.abs(pitch_freq))
    roll_freq /= np.max(np.abs(roll_freq))
    yaw_freq /= np.max(np.abs(yaw_freq))
    
    alpha_val = 1.0
    pos_freq_plot.plot(xf,
                   xpos_freq,
                   color=blue,
                   label="X Position",
                   alpha=alpha_val)
    

    
    pos_freq_plot.plot(xf,
                   ypos_freq,
                   color=green,
                   label="Y Position",
                   alpha=alpha_val)
    

    
    pos_freq_plot.plot(xf,
                   zpos_freq,
                   color=red,
                   label="Z Position",
                   alpha=alpha_val)
    

    
    freq_plot.plot(xf,
                   pitch_freq,
                   color=blue,
                   label="Pitch",
                   alpha=alpha_val)
    
    
    freq_plot.plot(xf,
                   yaw_freq,
                   color=red,
                   label="Yaw",
                   alpha=alpha_val)
            
    freq_plot.plot(xf,
                   roll_freq,
                   color=green,
                   label="Roll",
                   alpha=alpha_val)
    
    new_row = {"pitch_peak": xf[np.where(pitch_freq == np.max(pitch_freq))[0]].item(),
               "yaw_peak": xf[np.where(yaw_freq == np.max(yaw_freq))[0]].item(),
               "roll_peak": xf[np.where(roll_freq == np.max(roll_freq))[0]].item()}
                
    return pos_freq_plot, freq_plot, new_row
    
def AnalyzeResults(df_dict, posplot, attplot):
    
    for i, df in enumerate(df_dict.values()):
            
    # The analysis
    
        df["Z Position"] *= -1
        
        # Plotting
        
        
        if i == 0:
            # Xpos
            posplot.plot(df["Time"],
                          df["X Position"],
                          color=blue,
                          label="X Position",
                          alpha=alpha_val)
            # Y Pos
            posplot.plot(df["Time"],
                          df["Y Position"],
                          color=green,
                          label="Y Position",
                          alpha=alpha_val)
            # Z Plot
            posplot.plot(df["Time"],
                          df["Z Position"],
                          color=red,
                          label="Z Position",
                          alpha=alpha_val)
            # Pitch
            attplot.plot(df["Time"],
                          np.rad2deg(df["Pitch"]),
                          color=blue,
                          label="Pitch",
                          alpha=alpha_val)
            # Yaw
            attplot.plot(df["Time"],
                          np.rad2deg(df["Yaw"]),
                          color=red,
                          label="Yaw", 
                          alpha=alpha_val)
            # Roll
            attplot.plot(df["Time"],
                          np.rad2deg(df["Roll"]),
                          color=green,
                          label="Roll",
                          alpha=alpha_val)
            
        else:
            posplot.plot(df["Time"],
                          df["X Position"],
                          color=blue,
                          alpha=alpha_val)
            posplot.plot(df["Time"],
                          df["Y Position"],
                          color=green,
                          alpha=alpha_val)
            posplot.plot(df["Time"],
                          df["Z Position"],
                          color=red,
                          alpha=alpha_val)
            # Pitch
            attplot.plot(df["Time"],
                          np.rad2deg(df["Pitch"]),
                          color=blue,
                          alpha=alpha_val)
            # Yaw
            attplot.plot(df["Time"],
                          np.rad2deg(df["Yaw"]),
                          color=red,
                          alpha=alpha_val)
            # Roll
            attplot.plot(df["Time"],
                          np.rad2deg(df["Roll"]),
                          color=green,
                          alpha=alpha_val)
    
    # # Print Code
    # if mute == False:
    #     print(f"Simulation {iter_count+1} of {num_iterations} completed.")
    #     print(f"Time Elapsed: {tictoc:.2f} seconds")
    # else:
    #     pass
    # iter_count += 1
    # total_real_time += tictoc
    
    return posplot, attplot

#%%###########################

def RunWithGamma(gamma, *, misc_plots=False):
    drone = DefineAircraft(gamma)
    
    # Plotting stuff
    global colors, red, green, blue
    colors = sns.color_palette(palette="bright", n_colors=5)
    blue = colors[0]
    red = colors[1]
    green = colors[2]
    
    
    metres = mpl.ticker.EngFormatter("m")
    newtons = mpl.ticker.EngFormatter("N")
    seconds = mpl.ticker.EngFormatter("s")
    radians = mpl.ticker.EngFormatter("rad")
    degrees = mpl.ticker.EngFormatter(r"$\degree$")
        
    def worker(procnum, return_dict):
        """worker function"""
        df = RunMC(drone, max_random_pos, max_att, sim_time, mute, alpha_val, procnum)
        return_dict[procnum] = df
    
    tic = time.time()
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(num_iterations):
        p = multiprocessing.Process(target=worker, args=(i, return_dict))
        jobs.append(p)
        p.start()
    
    for proc in jobs:
        proc.join()
        # print(return_dict.values())
    toc = time.time()
    total_real_time = toc - tic    
    
    if misc_plots==True:
        fig, posplot = plt.subplots()
        fig, attplot = plt.subplots()
        posplot, attplot = AnalyzeResults(return_dict, posplot, attplot)
        
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
    print("MC SIM REPORT"); print("\n")
    print(f"Total Time Elapsed: {total_real_time:.2f} seconds")
    print(f"Average Simulation Time: {total_real_time/num_iterations:.2f} seconds")

    fig, freq_plot_arr = plt.subplots(2, 1, figsize=(16, 13))
    pos_freq_plot = freq_plot_arr[0]
    freq_plot = freq_plot_arr[1]
    pos_freq_plot, freq_plot, new_row = FFTAnalysis(return_dict, pos_freq_plot, freq_plot)
    new_row['gamma'] = gamma
    pos_freq_plot.set_xscale("log")
    freq_plot.set_xscale("log")
    freq_plot.legend(loc='best')
    pos_freq_plot.set_title("Position Fourier Transform")
    freq_plot.set_title("Attitude Fourier Transform")
    freq_plot.set_ylabel("Magnitude")
    pos_freq_plot.set_ylabel("Magnitude")
    freq_plot.set_xlabel("Frequency [Hz]")
    pos_freq_plot.legend(loc="best")
    Hz = mpl.ticker.EngFormatter(r"Hz")
    pos_freq_plot.xaxis.set_major_formatter(Hz)
    freq_plot.xaxis.set_major_formatter(Hz)
    plt.suptitle(fr"{aircraft_name}: $\gamma$ = {gamma}$\degree$")
    plt.show()
    
    return fig, freq_plot_arr, new_row

def InsertVector(vec, psi, theta, phi):
    vec_local = copy.deepcopy(vec)
    vec_local = vec_local.reshape((3,1))
    vec_rotated = usy.Local2Inertial(vec_local, phi, theta, psi)
    vec_rotated = vec_rotated.reshape(vec.shape)
    return vec_rotated

def RotateMotors(angle_sets, mixer):
    i = 0
    for psi, theta, phi in zip(angle_sets[:, 0], angle_sets[:, 1], angle_sets[:, 2]):
        mixer[3:6, i] = InsertVector(mixer[3:6, i], psi, theta, phi)
        local = np.cross(mixer[9:12, i], mixer[3:6, i])
        mixer[9, i] = local[1]
        mixer[10, i] = -local[0]
        i += 1
    return mixer

def DefineAircraft(gamma):    

    mixer = np.array([[0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                      [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                      [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                      [0, 0, 0, 0, 0, 0, 0, 0],  # X Forces
                      [0, 0, 0, 0, 0, 0, 0, 0],  # Y Forces
                      [-1, -1, -1, -1, -1, -1, -1, -1],  # Z Forces
                      [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                      [0, 0, 0, 0, 0, 0, 0, 0],  # Empty 
                      [0, 0, 0, 0, 0, 0, 0, 0],  # Empty 
                      [3, -3, 3, -3, 3, -3, 3, -3],  # X Moments (Roll)
                      [2.5, 2.5, -2.5, -2.5, 2.5, 2.5, -2.5, -2.5],  # Y Moments (Pitch)
                      [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=float)  # Z Moments (Yaw)
    
    run_time = 60
    
    # Defining motor thrust curve
    
    max_thrust = 3.8E3   # Newtons
    
    omega = np.linspace(0, 1300, 1000)  # Values between 0 and 1300 rad/s
    b = 0.8
    a = (max_thrust - b*np.max(omega)) / (np.max(omega)**2)
    thrust = a*(omega)**2 + b*omega
    
    # Creating base motor object
    
    motor = usy.Motor(1E3)  # Set motor object with 1000ms max PWM signal width
    motor.SetTau(0.4)  # Set motor time constant in seconds
    motor.SetThrustCurve(omega, thrust)  # Set motor thrust curve
    
    # Defining UAV inertial properties
    mass = 2200
    I = np.array([[600, 0, 0],
                  [0, 800, 0],
                  [0, 0, 800]])
    Ixx = I[0,0]
    Iyy = I[1,1]
    Izz = I[2,2]
    
    num_motors = 8  # Number of UAV motors
    clock_speed = 2.1E9  # Clock speed in Hz
    
    
    # drone = usy.UAV(mass, Ixx, Iyy, Izz, num_motors,
    #                 motor, mixer, clock_speed)
    global drone
    drone = usy.UAV(mass, I, num_motors, motor, mixer, clock_speed)
    
    drone.Setdt(0.001)  # Set time step size
    
    # Ziegler-Nichols Tuning Rule
    
    T = 100
    L = 10
    
    T_roll = 5
    L_roll = 2
    
    K_P_factor = 0*60
    K_D_factor = 0*40
    
    K_P_pos_factor = 4
    K_I_factor = 0.3#0.48
    K_D_pos_factor = 1.5 #1.3
    
    K_P = K_P_factor * (1.2 * T_roll/L_roll * Ixx)  # P constant, angular
    K_I = K_I_factor * 0.6 * T/L**2 # I constant, angular
    K_D = K_D_factor * (0.6 * T_roll * Ixx)  # D constant, angular
    
    # K_P_pos = K_P_pos_factor * 1.2 * T/L * mass  # P constant, altitude
    # K_D_pos = K_D_pos_factor * (0.6 * T * mass)  # D constant, altitude
    
    K_P_pos_xy = 0*0.1*K_P  # XY translational P constant
    K_D_pos_xy = 0*0.3*K_D   # XY translational D constant
    
    K_P_pos = 105.6E3
    K_D_pos = 198E3
    
    K_P = 0.18E3
    K_I = 0.18 # 0
    K_D = 1E3 # 1E3
    
    drone.SetPIDPD(K_P,
                   K_I,
                   K_D,
                   K_P_pos,
                   K_D_pos,
                   K_P_pos_xy,
                   K_D_pos_xy)

    return drone
#%%##########################
column_names = ["gamma", "pitch_peak", "yaw_peak", "roll_peak"]
runs_df = pd.DataFrame(columns=column_names)
plot_list = []
for aircraft_name in aircraft_names:
    for gamma_ticker, gamma in enumerate(gamma_list):
        fig, freq_plot_arr, new_row = RunWithGamma(gamma, misc_plots=misc_plots)
        runs_df = runs_df.append(new_row, ignore_index=True)
        plot_list.append(freq_plot_arr)
        directory = f"Results/{aircraft_name}"
        if os.path.isdir(directory) == False:
            os.mkdir(directory)
        fig.savefig(directory+f"/{aircraft_name}_time_{sim_time}_iters_{num_iterations}_gamma_{int(gamma)}")
        
#%%###########################

# Plot Gamma Angles

deg = mpl.ticker.EngFormatter(r"$\degree$")
Hz = mpl.ticker.EngFormatter(r"Hz")


fig, gammaplot = plt.subplots()
gammaplot.plot(runs_df.gamma, runs_df.pitch_peak, color=blue, label="Pitch Peak Frequency")
gammaplot.plot(runs_df.gamma, runs_df.roll_peak, color=red, label="Roll Peak Frequency")
gammaplot.plot(runs_df.gamma, runs_df.yaw_peak, color=green, label="Yaw Peak Frequency")
gammaplot.set_title("Attitude Frequency Response Plot")
gammaplot.set_xlabel(r"$\gamma$ [$\degree$]")
gammaplot.set_ylabel(r"Peak Frequency [Hz]")
gammaplot.xaxis.set_major_formatter(deg)
gammaplot.yaxis.set_major_formatter(Hz)
plt.legend(loc='best')

fig.savefig(directory +f"{aircraft_name}_gamma_dist")


"""
Looking for:
    Low magnitude, high frequency position response
    High magnitude, low frequency attitude response
"""