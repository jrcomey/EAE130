#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:43:11 2021

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import UAVsym as usy
from jrc import *

P = np.array([[0],
              [0],
              [-1]])

psi = np.deg2rad(-45)
theta = np.deg2rad(0)
phi = np.deg2rad(10)

P_new = usy.Local2Inertial(P, phi, theta, psi)

print(P)
print(P_new)

zero_vec = np.zeros_like(P)
P_arr = np.concatenate((zero_vec, P_new), axis=1)

print(P_arr)

fig, topplot = plt.subplots()

plothusly(topplot, P_arr[1, :], P_arr[0, :],
          xtitle="Y",
          ytitle="X",
          title="Top Down Vector Comparison",
          )

plt.xlim([-1, 1])
plt.ylim([-1, 1])
topplot.invert_xaxis()
