#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:35:29 2020

@author: Jack Rhys Comey
"""

# Imports

import numpy as np
import pandas as pd
import copy
#%###########################

# Objects


class Motor():

    def __init__(self, signal_width):
        """
        Initializes motor object with empty properties and sets maximum signal
        width

        Parameters
        ----------
        signal_width : Maximum PWM signal value

        Returns
        -------
        None.

        """
        self.signal_width = signal_width
        
        # Empty properties
        self.tau = 0
        self.constants = np.zeros((4,1))
        self.force = 0
        self.omega = 0
        self.omega_max = 1
        self.omega_signal_convert = 0.1

    def SetTau(self, tau):
        """
        Sets motor time constant

        Parameters
        ----------
        tau : First order time constant in seconds

        Returns
        -------
        None.

        """
        self.tau = tau

    def SetThrustCurve(self, omega, thrust):
        """
        Defines thrust curve given thrust data using 6 degree polynomial fit.
        Defines max angular velocity and the signal conversion to omega
        constant.

        Parameters
        ----------
        omega : Angular velocity of motor in rad/s
        thrust : Thrust of motor in Newtons

        Returns
        -------
        None.

        """
        self.constants = np.polyfit(omega, thrust, 6)
        self.omega_max = np.max(omega)
        self.omega_signal_convert = self.omega_max/self.signal_width

    def InToOut(self, signal, dt):
        """
        Finds first order response of the motor given dt and input signal.

        Parameters
        ----------
        signal : Input signal
        dt : Change in time in seconds

        Returns
        -------
        None.

        """
        # Find current motor speed given signal and dt
        omega_set = self.omega_signal_convert * signal
        self.omega = (omega_set
                      + (self.omega
                         - omega_set)
                      * np.exp(-1*dt/self.tau)
                      )
        self.PropertyCheck()
        self.ForceFind(self.omega)

    def ForceFind(self, omega):
        """
        Finds the force of the motor given the current angular velocity using
        6th degree polynomial fit.

        Parameters
        ----------
        omega : Angular velocity of the motor

        Returns
        -------
        None.

        """
        self.force = (self.constants[0]*omega**6
                      + self.constants[1]*omega**5
                      + self.constants[2]*omega**4
                      + self.constants[3]*omega**3
                      + self.constants[4]*omega**2
                      + self.constants[5]*omega**1
                      + self.constants[6]*omega**0
                      )

    def PropertyCheck(self):
        """
        Checks properties of the motor to ensure that they haven't exceeded
        bounds

        Returns
        -------
        None.

        """
        if self.omega > self.omega_max:
            self.omega = self.omega_max
        elif self.omega < 0:
            self.omega = 0
        else:
            pass


class UAV():
    """
    Generalized multirotor UAV object.
    """
    
    def __init__(self, mass, Ixx, Iyy, Izz, num_motors, motor_obj, mixer, clock_speed):
        """
        Initializes and defines basic properties of an n-motor UAV

        Parameters
        ----------
        mass : Total mass of UAV in kg
        num_motors : Total number of motors on the UAV
        motor_obj : Motor type used by the UAV
        mixer : A 12x(num_motors) array defining the forces and moments
            exerted on each axis
        clock_speed : Clock speed of the flight computer

        Returns
        -------
        None.

        """
        # Motor setup
        
        self.MotorInit(num_motors, motor_obj)

        # UAV physical properties
        self.mass = mass  # kg
        self.Ixx = Ixx      # kg-m**2
        self.Iyy = Iyy      # kg-m**2
        self.Izz = Izz      # kg-m**2

        # Environmental Properties
        self.dt = 0 # s, default
        self.time = np.array([0], ndmin=2, dtype=float)  # s, ndim fixes a bug in RecordData()

        # Configuration

        self.mixer = mixer  # Sets mixer as object property
        
        # Divides mixer by mass and I values to reflect inertial properties
        for i in range(6):
            self.mixer[i] = self.mixer[i]/self.mass
        self.mixer[6] = self.mixer[6] / self.Ixx
        self.mixer[9] = self.mixer[9] / self.Ixx
        self.mixer[7] = self.mixer[7] / self.Iyy
        self.mixer[10] = self.mixer[10] / self.Iyy
        self.mixer[8] = self.mixer[8] / self.Izz
        self.mixer[11] = self.mixer[11] / self.Izz

        # State Space Vector

        self.state_vector = np.array([[0],   # x
                                      [0],   # y
                                      [0],   # z
                                      [0],   # x'
                                      [0],   # y'
                                      [0],   # z'
                                      [0],   # phi
                                      [0],   # theta
                                      [0],   # psi
                                      [0],   # phi'
                                      [0],   # theta'
                                      [0]], dtype=float)  # psi'

        # State Space Matricies

        self.A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # x
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   # y
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # z
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # x'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # y'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # z'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # phi
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   # theta
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # psi
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # phi'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # theta'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)  # psi'

        self.C = np.array([[1, 0, 0, self.dt, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                           [0, 1, 0, 0, self.dt, 0, 0, 0, 0, 0, 0, 0],  # y
                           [0, 0, 1, 0, 0, self.dt, 0, 0, 0, 0, 0, 0],  # z
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        # x'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        # y'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        # z'
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, self.dt, 0, 0],  # phi
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, self.dt, 0],  # theta
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, self.dt],  # psi
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        # phi'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        # theta'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)       # psi'
        
        self.forward_matrix = np.array([[0, 0, 0, self.dt**2/2, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                                        [0, 0, 0, 0, self.dt**2/2, 0, 0, 0, 0, 0, 0, 0],  # y
                                        [0, 0, 0, 0, 0, self.dt**2/2, 0, 0, 0, 0, 0, 0],  # z
                                        [1, 0, 0, self.dt, 0, 0, 0, 0, 0, 0, 0, 0],  # x'
                                        [0, 1, 0, 0, self.dt, 0, 0, 0, 0, 0, 0, 0],  # y'
                                        [0, 0, 1, 0, 0, self.dt, 0, 0, 0, 0, 0, 0],  # z'
                                        [0, 0, 0, 0, 0, 0, self.dt**2/2, 0, 0, 0, 0, 0],  # phi
                                        [0, 0, 0, 0, 0, 0, 0, self.dt**2/2, 0, 0, 0, 0],  # theta
                                        [0, 0, 0, 0, 0, 0, 0, 0, self.dt**2/2, 0, 0, 0],  # psi
                                        [0, 0, 0, 0, 0, 0, 1, 0, 0, self.dt, 0, 0],  # phi'
                                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, self.dt, 0],  # theta'
                                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, self.dt]],dtype=float)  # psi'

        # Sets gravitational acceleration

        self.acc_grav = np.array([[0],
                                  [0],
                                  [0],
                                  [0],
                                  [0],
                                  [9.807],  # ms**-2
                                  [0],
                                  [0],
                                  [0],
                                  [0],
                                  [0],
                                  [0]])

        # Sets feedforward matrix for motors

        self.D = np.zeros((12, num_motors))  # Initialize as zeros

        # Sets u matrix (Forces in N, Initialized at 0)

        self.input = np.zeros((num_motors, 1))

        # Final state vector, initialized stationary at 0,0,0

        self.final_state = np.array([[0],   # x
                                     [0],   # y
                                     [0],   # z
                                     [0],   # x'
                                     [0],   # y'
                                     [0],   # z'
                                     [0],   # phi
                                     [0],   # theta
                                     [0],   # psi
                                     [0],   # phi'
                                     [0],   # theta'
                                     [0]], dtype=float)  # psi'

        # State vector to hold integral values

        self.int_vec = np.array([[0],   # x
                                 [0],   # y
                                 [0],   # z
                                 [0],   # x'
                                 [0],   # y'
                                 [0],   # z'
                                 [0],   # phi
                                 [0],   # theta
                                 [0],   # psi
                                 [0],   # phi'
                                 [0],   # theta'
                                 [0]], dtype=float)  # psi'
        # Data storage

        self.storage_array = np.zeros((1, (len(self.state_vector)
                                           + len(self.signal)
                                           + len(self.motors)
                                           + len(self.time))), dtype=float)
        self.InitialCheck()

    def InitialCheck(self):
        pass

    def MotorInit(self, num_motors, motor_obj):
        """
        Duplicates motor objects into a list.
        Necessary for time-response modelling.

        Parameters
        ----------
        num_motors : Number of UAV motors.
        motor_obj : Pre-made motor object to be duplicated.

        Returns
        -------
        None.

        """
        self.motors = np.empty((num_motors, 1), dtype=object)
        for i in range(num_motors):
            self.motors[i] = copy.deepcopy(motor_obj)
        
        # Normally int, set to float for senior design
        self.signal = np.zeros((num_motors, 1), dtype=int)

    def Setdt(self, dt):
        self.dt = dt

        self.A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # x
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   # y
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # z
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # x'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # y'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # z'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # phi
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   # theta
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # psi
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # phi'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # theta'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # psi'
                          dtype=float)

        self.C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # y
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # z
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],        # x'
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],        # y'
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],        # z'
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # phi
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # theta
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # psi
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],        # phi'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],        # theta'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])       # psi'
        
        self.forward_matrix = np.array([[self.dt, 0, 0, self.dt**2/2, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                                        [0, self.dt, 0, 0, self.dt**2/2, 0, 0, 0, 0, 0, 0, 0],  # y
                                        [0, 0, self.dt, 0, 0, self.dt**2/2, 0, 0, 0, 0, 0, 0],  # z
                                        [0, 0, 0, self.dt, 0, 0, 0, 0, 0, 0, 0, 0],  # x'
                                        [0, 0, 0, 0, self.dt, 0, 0, 0, 0, 0, 0, 0],  # y'
                                        [0, 0, 0, 0, 0, self.dt, 0, 0, 0, 0, 0, 0],  # z'
                                        [0, 0, 0, 0, 0, 0, self.dt, 0, 0, self.dt**2/2, 0, 0],  # phi
                                        [0, 0, 0, 0, 0, 0, 0, self.dt, 0, 0, self.dt**2/2, 0],  # theta
                                        [0, 0, 0, 0, 0, 0, 0, 0, self.dt, 0, 0, self.dt**2/2],  # psi
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, self.dt, 0, 0],  # phi'
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.dt, 0],  # theta'
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.dt]])  # psi'
    
    def RunSim(self, finish_time):
        while self.time < finish_time:
            self.RunSimTimeStep()  # Calls control loop and physics simulation
            self.RecordData()  # Records current state at timestamp

    def RunSimNoRecord(self, finish_time):
        while self.time < finish_time:
            self.RunSimTimeStep()  # Calls control loop and physics simulation

    def RunSimTimeStep(self):
        """
        Runs simulation at new time step for interval of t=dt

        Returns
        -------
        None.

        """
        self.MotorControl()
        self.Update()

    def MotorControl(self):
        """
        Determines motor signals in response to PID control loop.

        Returns
        -------
        None.

        """
        
        # Determines error vector, and updates error integral vector
        err_vec = self.final_state - self.state_vector
        self.int_vec = err_vec*self.dt + self.int_vec
        
        # Correction of integral vector
        # self.int_vec[6] = AngleCorrect(self.int_vec[6], -0.5*np.pi, np.pi/2)
        # self.int_vec[7] = AngleCorrect(self.int_vec[8], -0.5*np.pi, np.pi/2)
        # self.int_vec[8] = AngleCorrect(self.int_vec[8], -0.5*np.pi, np.pi/2)

        # Sum of error vector and error integral
        err_vec = err_vec + self.int_vec*self.K_I

        # Set error rate integrals to zero
        self.int_vec[3], self.int_vec[4], self.int_vec[5], self.int_vec[9], self.int_vec[10], self.int_vec[11] = 0,0,0,0,0,0

        # Dot product R**-1, err vec
        err_vec = np.dot(TransformationMatrix(self.state_vector[6].item(),
                                       self.state_vector[7].item(),
                                       self.state_vector[8].item()).transpose(),
                         err_vec)
        # Err vec -> Motor forces
        err_vec = np.dot(self.control_mat,
                          err_vec)
        
        # print(self.int_vec)
        # Anglechecks the error vector. Stops the integral from flipping over too much
        # err_vec[9] = AngleCorrect(err_vec[9], -0.5*np.pi, np.pi/2)
        # err_vec[10] = AngleCorrect(err_vec[10], -0.5*np.pi, np.pi/2)
        # err_vec[11] = AngleCorrect(err_vec[11], -0.5*np.pi, np.pi/2)
        # Motor forces -> PWM signal
        self.signal = np.dot(self.mixer.transpose(), err_vec)
        
        # Check signal validity & convert to integer
        self.SignalCheck()

    def Update(self):
        """
        Updates motor forces, and determines new state vector.
        
        Returns
        -------
        None.

        """
        self.UpdateMotors()
        self.StateSpaceKinematics()

    def UpdateMotors(self):
        """
        Determines current motor output, given input signal and motor time 
        constant.

        Returns
        -------
        None.

        """
        for i in range(len(self.motors)):
            self.motors[i].item().InToOut(self.signal[i], self.dt)
            self.input[i] = self.motors[i].item().force

    def StateSpaceKinematics(self):
        """
        Performs dynamics modelling for time step dt.

        Returns
        -------
        None.

        """
        # X'(t) calculation
        xdot = (
                np.dot(self.A, self.state_vector)             # State vector A
                + np.dot(TransformationMatrix(self.state_vector[6].item(),  # Local forces
                                              self.state_vector[7].item(),  # -> Earth axis
                                              self.state_vector[8].item()),
                         np.dot(self.mixer,
                                self.input))                  # Local forces
                + self.acc_grav                               # Gravity
                )

        self.state_vector = (
                             np.dot(self.C, self.state_vector)    # State C
                             + np.dot(self.D, self.input)         # Feedforward
                             + np.dot(self.forward_matrix, xdot)  # Adds acc
                             )
        # Can't use += because self.time is an array, so it can be concat later
        self.time = self.time + self.dt
        self.AngleCheck()
    
    def RecordData(self):
        """
        Records data at current time step into storage matrix.

        Returns
        -------
        None.

        """
        motor_forces = np.zeros((1, len(self.motors)))
        for i in range(len(self.motors)):
            motor_forces[0][i] = self.motors[i].item().force
        new_row = np.concatenate((self.time.transpose(),
                                  self.state_vector.transpose(),
                                  self.signal.transpose(),
                                  motor_forces), axis=1)
        self.storage_array = np.concatenate((self.storage_array,
                                             new_row), axis=0)

    def ExportData(self):
        """
        Exports data from storage matrix into pandas dataframe.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        columns =  ["Time",
                    "X Position",
                    "Y Position",
                    "Z Position",
                    "X Velocity",
                    "Y Velocity",
                    "Z Velocity",
                    "Roll",
                    "Pitch",
                    "Yaw",
                    "Roll Rate",
                    "Pitch Rate",
                    "Yaw Rate"]
        for i in range(len(self.motors)):
            tempname = f'Motor {i} Signal'
            columns.append(tempname)
        for i in range(len(self.motors)):
            tempname = f'Motor {i} Force'
            columns.append(tempname)
        self.storage_array = np.delete(self.storage_array, obj=0, axis=0)
        return pd.DataFrame(columns=columns, data=self.storage_array)
        self.storage_array = np.zeros((1, (len(self.state_vector)
                                           + len(self.signal)
                                           + len(self.motors)
                                           + len(self.time))), dtype=float)
        self.time = np.array([0], ndmin=2, dtype=float)

    def SignalCheck(self):
        """
        Limits PWM motor signals to pre-defined bounds and converts to integer

        Returns
        -------
        None.

        """
        for i in range(len(self.signal)):
            if self.signal[i] > self.motors[i].item().signal_width:
                self.signal[i] = self.motors[i].item().signal_width
            elif self.signal[i] < 0:
                self.signal[i] = 0
            else:
                self.signal[i] = round(self.signal[i].item())
        
        # print(self.signal)
    def AngleCheck(self):
        """
        Checks Euler angles to see if they're out of bounds, and corrects them'

        Returns
        -------
        None.

        """
        while (self.state_vector[6] > np.pi
               or self.state_vector[6] < -1*np.pi
               or self.state_vector[7] > np.pi
               or self.state_vector[7] < -1*np.pi
               or self.state_vector[8] > np.pi
               or self.state_vector[8] < -1*np.pi):

            if self.state_vector[6] > np.pi:
                self.state_vector[6] -= 2*np.pi
            elif self.state_vector[6] < -1*np.pi:
                self.state_vector[6] += 2*np.pi
            else:
                pass
        
            if self.state_vector[7] > np.pi:
                self.state_vector[7] -= 2*np.pi
            elif self.state_vector[7] < -1*np.pi:
                self.state_vector[7] += 2*np.pi
            else:
                pass
        
            if self.state_vector[8] > np.pi:
                self.state_vector[8] -= 2*np.pi
            elif self.state_vector[8] < -1*np.pi:
                self.state_vector[8] += 2*np.pi
            else:
                pass

    def SetPIDPD(self, P, I, D, P_pos, D_pos, P_pos_xy, D_pos_xy, meter_per_rad=100):
        """
        Sets PID constants and PD constants for hover function

        Parameters
        ----------
        P : Proportional constant for angular control
        I : Integral constant for angular control
        D : Derivative constant for angular control
        P_pos : Proportioal constant for altitude control
        D_pos : Derivative constant for altitude control

        Returns
        -------
        None.

        """
        # Set PID values as object values
        self.K_P = P
        self.K_I = I
        self.K_D = D
        self.K_P_pos = P_pos
        self.K_D_pos = D_pos
        self.K_P_pos_xy = P_pos_xy
        self.K_D_pos_xy = D_pos_xy
        
        # Define the control matrix
        self.control_mat = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [self.K_P_pos, 0, 0, self.K_D_pos, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, self.K_P_pos, 0, 0, self.K_D_pos, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, self.K_P_pos, 0, 0, self.K_D_pos, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, self.K_P_pos_xy, 0, 0, self.K_D_pos_xy, 0, self.K_P, 0, 0, self.K_D, 0, 0],
                                     [-1*self.K_P_pos_xy, 0, 0, -1*self.K_D_pos_xy, 0, 0, 0, self.K_P, 0, 0, self.K_D, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, self.K_P, 0, 0, self.K_D]])


    def AlterControlMat(self, control_mat_new):
        """
        Checks new PIDPD contol matrix is able to replace existing one, then
        replaces it.
        
        Function is in case you want to directly alter the control matrix.

        Parameters
        ----------
        control_mat_new : New control matrix. Must be same shape as existing
            version.

        Returns
        -------
        None.

        """
        if self.control_mat.shape == control_mat_new.shape:
            self.control_mat = control_mat_new
        else:
            pass

    def Reset(self):
        """
        Resets all state values to zero

        Returns
        -------
        None.

        """
        
        self.ClearStorage()
        self.state_vector = np.zeros((12,1))
        self.int_vec = np.zeros_like(self.int_vec)
        self.input = np.zeros((len(self.input), 1))
        self.time = np.zeros_like(self.time)
        self.final_state = np.zeros_like(self.final_state)
        
        for motor in self.motors:
            motor[0].omega = 0
        self.signal = np.zeros_like(self.signal)
        self.input = np.zeros_like(self.input)
        
    
    def ClearStorage(self):
        self.storage_array = np.zeros((1, (len(self.state_vector)
                                           + len(self.signal)
                                           + len(self.motors)
                                           + len(self.time))), dtype=float)
#%###########################

# Functions


def TransformationMatrix(phi, theta, psi):
    """
    Transforms 6 DOF state vector from body axis to earth axis

    Parameters
    ----------
    phi : Roll angle in radians
    theta : Pitch angle in radians
    psi : Yaw angle in radians

    Returns
    -------
    array : 12x12 transformation matrix
    """
    return np.array([[np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi), np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi), 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [np.sin(psi)*np.cos(theta), np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi), 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [-1*np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi), 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi), np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi), 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, np.sin(psi)*np.cos(theta), np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi), 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, -1*np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi), 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi), np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi), 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, np.sin(psi)*np.cos(theta), np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi), 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, -1*np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi), 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi), np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi)],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, np.sin(psi)*np.cos(theta), np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi)],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -1*np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)]])

def AngleCorrect(angle, min_ang, max_ang):
    if angle < min_ang:
        angle = min_ang
    elif angle > max_ang:
        angle = max_ang
    else:
        pass
    return angle

def Autotune(drone, rise_time, os_percent_input, exp_time_allowed, *, tol=0.05, max_tests=100):
    
    # Resets position if previously simulated
    drone.Reset()
    
    # Set height
    drone.final_state[2] = -10
    
    # Flag for loop
    done = False
    
    # Ticker counter
    ticker = 0
    # While the flag is unset
    while done == False:
        # K_D loop
        
        # Set target
        drone.final_state[2] = -10
        # Run Simulation
        drone.RunSim(exp_time_allowed)
        
        # Export data
        df = drone.ExportData()
        
        # Print header
        print("\n" + f"K_P Loop: Run {ticker+1} Completed")
        
        # Normalize Z position to up
        df["Z Position"] *= -1
        
        # Find rise time
        for i, t in enumerate(df["Time"]):
            if df["Z Position"][i] > 0.9*abs(drone.final_state[2]):
                rise_time_exp = t
                break
            else:
                pass
        
        # Find peak
        flag = False
        for i, t in enumerate(df["Time"]):    
            
            if i > 0:
                if df["Z Position"][i] > df["Z Position"][i-1]:
                    flag = True
            
            if flag == True:
                if df["Z Position"][i] < df["Z Position"][i-1]:
                    peak = df["Z Position"][i]
                    break
        
        # Find overshoot, report data
        overshoot = (peak/abs(drone.final_state[2]) * 100).item()
        print(f"Rise_time = {rise_time_exp:.4f} s")
        print(f"Peak = {peak:.2f} m")
        print(f"OS% = {overshoot:.2f}%")
        
        # Iterate K_P value to match rise time
        if abs(rise_time - rise_time_exp)/rise_time < tol:
            break
        elif abs(rise_time - rise_time_exp)/rise_time > tol:
            drone.K_P_pos += (rise_time_exp - rise_time)
            print(f"New K_P_pos: {drone.K_P_pos:.4f}")
        else:
            pass
        
        # Set new control matrix
        drone.SetPIDPD(drone.K_P,
                       0,
                       drone.K_D,
                       drone.K_P_pos,
                       0,
                       0,
                       0)
        # Reset, check to see how long this has been going on for
        drone.Reset()
        ticker += 1
        if ticker > max_tests:
            print("Broken!")
            break
    
    done = False
    ticker = 0
    while done == False:
        
        # K_D loop
        
        # Set target
        drone.final_state[2] = -10
        # Run Simulation
        drone.RunSim(exp_time_allowed)
        
        # Export data
        df = drone.ExportData()
        
        # Print header
        print("\n" + f"K_D Loop: Run {ticker+1} Completed")
        
        # Normalize Z position to up
        df["Z Position"] *= -1
        
        # Find rise time
        for i, t in enumerate(df["Time"]):
            if df["Z Position"][i] > 0.9*abs(drone.final_state[2]):
                rise_time_exp = t
                break
            else:
                pass
        
        # Find peak
        flag = False
        for i, t in enumerate(df["Time"]):    
            
            if i > 0:
                if df["Z Position"][i] > df["Z Position"][i-1]:
                    flag = True
            
            if flag == True:
                if df["Z Position"][i] < df["Z Position"][i-1]:
                    peak = df["Z Position"][i]
                    break
        
        # Find overshoot, report data
        overshoot = (peak/abs(drone.final_state[2]) * 100).item()
        print(f"Rise_time = {rise_time_exp:.4f} s")
        print(f"Peak = {peak:.2f} m")
        print(f"OS% = {overshoot:.2f}%")
        
        # Find new K_D values
        if abs(os_percent_input - overshoot) < tol:
            break
        elif abs(os_percent_input - overshoot) > tol:
            drone.K_D_pos += (overshoot - os_percent_input)/1E2
            print(f"New K_D_pos: {drone.K_D_pos:.4f}")
        
        # Set new control matrix
        drone.SetPIDPD(drone.K_P,
                       0,
                       drone.K_D,
                       drone.K_P_pos,
                       drone.K_D_pos,
                       drone.K_P_pos_xy,
                       drone.K_D_pos_xy)
        # Reset, check to see how long this has been going on for
        drone.Reset()
        ticker += 1
        # if ticker > max_tests:
        #     print("Broken!")
        #     break
    
    print("\n\n" + "Done Constant Information")
    print(f"K_P_pos: {drone.K_P_pos}")
    print(f"K_D_pos: {drone.K_D_pos}")
    
    
    return drone