#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:13:44 2021

@author: jack
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from jrc import *

data = pd.read_csv("Data/V6_DUCTED_SKID_neg20_90_33.csv", index_col=False)
data2 = pd.read_csv("Data/V6_DUCTED_SKIDS_neg20_90_33.csv", index_col=False)

fig, dragpolar = plt.subplots()

plothusly(dragpolar,
          data["AOA"],
          data["CDi"],
          datalabel=r"$C_{D_i}$",
          marker='x',
          title="ATP-XW Blizzard Drag Polar",
          xtitle=r"Angle of Attack $\alpha$ [$\degree$]",
          ytitle=r"Drag Coefficient $C_D$")

plothus(dragpolar,
        data["AOA"],
        data["CDo"],
        datalabel=r"$C_{D_o}$",
        marker='d')

plothus(dragpolar,
        data["AOA"],
        data["CDo"] + data["CDi"],
        datalabel=r"$C_{D}$",
        marker='*')

plothus(dragpolar,
          data2["AOA"],
          data2["CDi"],
          datalabel=r"$C_{D_i}$ new",
          marker='x')

plothus(dragpolar,
        data2["AOA"],
        data2["CDo"],
        datalabel=r"$C_{D_o}$ new",
        marker='d',)

plothus(dragpolar,
        data2["AOA"],
        data2["CDo"] + data2["CDi"],
        datalabel=r"$C_{D}$ new",
        marker='*',)