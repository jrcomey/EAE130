#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:12:00 2021

@author: jack
"""

import numpy as np
import random

def RandomizeDronePosition(drone, magnitude, mag_att = 1, mag_att_vel = 0):
    
    for i, _ in enumerate(drone.state_vector):
        if i >= 5:
            if i < 8:
                drone.state_vector[i] = random.uniform(-mag_att, mag_att)
            else:
                drone.state_vector[i] = random.uniform(-mag_att_vel, mag_att_vel)
        else:
            drone.state_vector[i] = random.uniform(-magnitude, magnitude)
    return drone