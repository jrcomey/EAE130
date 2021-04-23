"""

Example Code
"""

import UAVsym_new as usy
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import montecarlofunctions as mcf
from jrc import *
import copy

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
        i += 1
    return mixer

I = np.array([[500, 0, 0],
              [0, 500, 0],
              [0, 0, 500]])


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


gamma = 0

angle_sets = np.array([[-45, 0, gamma],
                       [45, 0, -gamma],
                       [45, 0, gamma],
                       [-45, 0, -gamma],
                       [-45, 0, gamma],
                       [45, 0, -gamma],
                       [45, 0, gamma],
                       [-45, 0, -gamma]])

angle_sets = np.deg2rad(angle_sets)

# angle_sets *= 0

# mixer = RotateMotors(angle_sets, mixer)

run_time = 20

# Defining motor thrust curve

max_thrust = 10E3   # Newtons

omega = np.linspace(0, 2500, 1000)  # Values between 0 and 1300 rad/s
thrust = max_thrust - max_thrust*np.exp(-omega/100)

# Creating base motor object

motor = usy.Motor(1E3)  # Set motor object with 100ms max PWM signal width
motor.SetTau(0.001)  # Set motor time constant in seconds
motor.SetThrustCurve(omega, thrust)  # Set motor thrust curve

# Defining UAV inertial properties
mass = 1958 + 8*(40)# mass = 2200
Ixx = 100  # kg-m^2
Iyy = 100 # kg-m^2
Izz = 100  # kg-m^2


num_motors = 8  # Number of UAV motors
clock_speed = 2.1E9  # Clock speed in Hz


# drone = usy.UAV(mass, Ixx, Iyy, Izz, num_motors,
#                 motor, mixer, clock_speed)

drone = usy.UAV(mass, I, num_motors, motor, mixer, clock_speed)

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

LQR = np.loadtxt("Matlab/K.csv", dtype=float, delimiter=',')
LQR *= 1E10
# for i in range(12):
#     for j in range(12):
#         if LQR[i, j] < 1E-3:
#             LQR[i,j] = 0
#         else:
#             pass
# print(LQR)
# drone.AlterControlMat(LQR)
# drone.K_I = 0
#%%###########################
# Set initial state to showcase control
drone.Reset()

drone.state_vector[0] = 0  # x
drone.state_vector[1] = 0  # y
drone.state_vector[2] = 0  # z
drone.state_vector[3] = 0  # x'
drone.state_vector[4] = 0  #y'
drone.state_vector[5] = -2  # z'
drone.state_vector[6] = 3  # roll
drone.state_vector[7] = -3   # pitch
drone.state_vector[8] = 3 # yaw
# drone.state_vector[9] = 0   # roll dot
# drone.state_vector[10] = 0   # pitch dot
# drone.state_vector[11] = 0   # yaw dot


# drone.final_state[2] = -5

# drone = mcf.RandomizeDronePosition(drone, 0, -np.pi)
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
# plothus(signalplot, df["Time"], df[f"Motor 0 Signal"], datalabel=f"Motor 0 Signal")
    
signalplot.xaxis.set_major_formatter(seconds)

#%%###########################
# fig, thrustplot = plt.subplots()
# plothusly(thrustplot, 0, 0, xtitle=r"Time [s]", ytitle=r"Motor Thrust", datalabel='', title="Motor Thrust Plot")
# for i, motor in enumerate(drone.motors):
#     plothus(thrustplot, df["Time"], df[f"Motor {i} Force"], datalabel=f"Motor {i} Thrust")
# thrustplot.xaxis.set_major_formatter(seconds)
# thrustplot.yaxis.set_major_formatter(newtons)

#%%###########################

# From stack overflow

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

# print(bmatrix(drone.control_mat) + '\n')

