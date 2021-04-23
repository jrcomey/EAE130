#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:12:00 2021

@author: jack
"""

import numpy as np
import random
import time
from jrc import *

def RandomizeDronePosition(drone, magnitude, mag_att = 1, mag_att_vel = 0):
    
    for i, _ in enumerate(drone.state_vector):
        if i >= 5:
            if i <= 8:
                drone.state_vector[i] = random.uniform(-mag_att, mag_att)
            else:
                drone.state_vector[i] = random.uniform(-mag_att_vel, mag_att_vel)
        else:
            drone.state_vector[i] = random.uniform(-magnitude, magnitude)
    return drone

def RunMC(drone, max_random_pos, max_att, sim_time, mute, alpha_val, first=False):
    tic = time.time()
    drone.Reset()
    drone = RandomizeDronePosition(drone, max_random_pos, mag_att=max_att)
    drone.RunSim(sim_time)
    toc = time.time()
    tictoc = toc-tic
    df = drone.ExportData()
    
    # The analysis
    
    df["Z Position"] *= -1
    
    # Plotting
    
    
    if first == False:
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