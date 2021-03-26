"""

Example Code
"""

import UAVsym as usy
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import montecarlofunctions as mcf
from jrc import *

run_time = 15

# Defining motor thrust curve

max_thrust = 3.8E3   # Newtons

omega = np.linspace(0, 2500, 1000)  # Values between 0 and 1300 rad/s
thrust = max_thrust - max_thrust*np.exp(-omega/100)

# Creating base motor object

motor = usy.Motor(1E3)  # Set motor object with 100ms max PWM signal width
motor.SetTau(0.1)  # Set motor time constant in seconds
motor.SetThrustCurve(omega, thrust)  # Set motor thrust curve

# Defining UAV inertial properties
mass = 2200 - 8 * (49 - 40) # kg
# mass = 2200
Ixx = 1000  # kg-m^2
Iyy = 800 # kg-m^2
Izz = 800  # kg-m^2


num_motors = 7  # Number of UAV motors
clock_speed = 2.1E9  # Clock speed in Hz


mixer = np.array([[0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0],  # X Forces
                  [0, 0, 0, 0, 0, 0, 0],  # Y Forces
                  [-1, -1, -1, -1, -1, -1, -1],  # Z Forces
                  [0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0],  # Empty 
                  [0, 0, 0, 0, 0, 0, 0],  # Empty 
                  [2, -2, 2, -2, 2, -2, 2],  # X Moments (Roll)
                  [2.5, 2.5, -2.5, -2.5, 2.5, 2.5, -2.5],  # Y Moments (Pitch)
                  [-1, 1, 1, -1, 1, -1, -1]], dtype=float)  # Z Moments (Yaw)

drone = usy.UAV(mass, Ixx, Iyy, Izz, num_motors,
                motor, mixer, clock_speed)

drone.Setdt(0.001)  # Set time step size

# Ziegler-Nichols Tuning Rule

T = 100
L = 10

T_roll = 5
L_roll = 2

K_P_factor = 30
K_D_factor = 120

K_P_pos_factor = 4
K_I_factor = 0.48 #0.48
K_D_pos_factor = 1.3

K_P = K_P_factor * (1.2 * T_roll/L_roll * Ixx)  # P constant, angular
K_I = K_I_factor * 0.6 * T/L**2 # I constant, angular
K_D = K_D_factor * (0.6 * T_roll * Ixx)  # D constant, angular

K_P_pos = K_P_pos_factor * 1.2 * T/L * mass  # P constant, altitude
K_D_pos = K_D_pos_factor * (0.6 * T * mass)  # D constant, altitude

K_P_pos_xy = 0*K_P  # XY translational P constant
K_D_pos_xy = 0*K_D   # XY translational D constant

drone.SetPIDPD(K_P,
               K_I,
               K_D,
               K_P_pos,
               K_D_pos,
               K_P_pos_xy,
               K_D_pos_xy)

#%%###########################
# Set initial state to showcase control
drone.Reset()

drone.state_vector[0] = 10  # x
drone.state_vector[1] = -15  # y
drone.state_vector[2] = 15  # z
drone.state_vector[3] = 0  # x'
drone.state_vector[4] = 0  #y'
drone.state_vector[5] = -2  # z'
drone.state_vector[6] = -0.1*np.pi  # roll
# drone.state_vector[7] = 0.5*np.pi   # pitch
# drone.state_vector[8] = 0.25*np.pi   # yaw
# drone.state_vector[9] = -0.5   # roll dot
# drone.state_vector[10] = 2   # pitch dot
# drone.state_vector[11] = 0.5   # yaw dot


drone.final_state[2] = 0

# drone = mcf.RandomizeDronePosition(drone, 5, -0.2*np.pi)
# Main loop:
tic = time.time()
drone.RunSim(run_time)
toc = time.time()
tictoc = toc-tic
print(f"Simulation ran for {tictoc:.2f} seconds")
# Exports data to pandas dataframe
df = drone.ExportData()
#%%###########################


#%%###########################

# For plotting results only! Not contained in sample code.
df["Z Position"] *= -1

meters = mpl.ticker.EngFormatter("m")
newtons = mpl.ticker.EngFormatter("N")
seconds = mpl.ticker.EngFormatter("s")
radians = mpl.ticker.EngFormatter("rad")


# Position plot
fig, zplot = plt.subplots()
plothusly(zplot, df["Time"], df["Z Position"], 
          xtitle="Time in seconds",
          ytitle="Position in metres", 
          datalabel="Z Position", 
          title="NED Drone Position")
plothus(zplot, df["Time"], df["Y Position"], datalabel="Y Position")
plothus(zplot, df["Time"], df["X Position"], datalabel="X Position")

zplot.xaxis.set_major_formatter(seconds)
zplot.yaxis.set_major_formatter(meters)

#%%###########################
# Attitude plots
fig, angleplot = plt.subplots()
plothusly(angleplot, df["Time"], df["Pitch"], 
          xtitle="Time in seconds",
          ytitle="Angle from neutral position in radians", 
          datalabel="Pitch", 
          title="NED Aircraft Attitude")
plothus(angleplot, df["Time"], df["Yaw"], datalabel="Yaw")
plothus(angleplot, df["Time"], df["Roll"], datalabel="Roll")
angleplot.xaxis.set_major_formatter(seconds)
angleplot.yaxis.set_major_formatter(radians)
#%%###########################
# Motor Signals
fig, signalplot = plt.subplots()
plothusly(signalplot, 0, 0, xtitle=r"Time [s]", ytitle=r"Motor Signal", datalabel='', title="Motor Signal Plot")

for i, motor in enumerate(drone.motors):
    plothus(signalplot, df["Time"], df[f"Motor {i} Signal"], datalabel=f"Motor {i} Signal")
    
signalplot.xaxis.set_major_formatter(seconds)
# signalplot.yaxis.set_major_formatter()

#%%###########################
fig, thrustplot = plt.subplots()
plothusly(thrustplot, 0, 0, xtitle=r"Time [s]", ytitle=r"Motor Thrust", datalabel='', title="Motor Thrust Plot")
for i, motor in enumerate(drone.motors):
    plothus(thrustplot, df["Time"], df[f"Motor {i} Force"], datalabel=f"Motor {i} Thrust")
thrustplot.xaxis.set_major_formatter(seconds)
thrustplot.yaxis.set_major_formatter(newtons)