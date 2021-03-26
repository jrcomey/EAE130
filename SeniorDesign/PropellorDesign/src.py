#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:31:55 2021

@author: jack
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import time
import pandas as pd


def plothusly(ax, x, y, *, xtitle='', ytitle='',
              datalabel='', title='', linestyle='-',
              marker=''):
    """
    A little function to make graphing less of a pain.
    Creates a plot with titles and axis labels.
    Adds a new line to a blank figure and labels it.

    Parameters
    ----------
    ax : The graph object
    x : X axis data
    y : Y axis data
    xtitle : Optional x axis data title. The default is ''.
    ytitle : Optional y axis data title. The default is ''.
    datalabel : Optional label for data. The default is ''.
    title : Graph Title. The default is ''.

    Returns
    -------
    out : Resultant graph.

    """

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_title(title)
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle = linestyle,
                  marker = marker)
    ax.grid(True)
    ax.legend(loc='best')
    return out


def plothus(ax, x, y, *, datalabel='', linestyle = '-',
            marker = ''):
    """
    A little function to make graphing less of a pain

    Adds a new line to a blank figure and labels it
    """
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle = linestyle,
                  marker = marker)
    ax.legend(loc='best')

    return out

class Propeller:
    
    def __init__(self, B, D, c_x, beta_x, alpha_0_x, P_eng, x):
        self.B = B
        self.D = D
        self.R = D/2
        self.c_x = c_x
        self.P_eng = P_eng
    
    
        # FIX ME 
        self.beta = np.zeros_like(x)
        self.alpha_0 = np.zeros_like(x)

    def GetSigma(self, x):
        return self.B * self.c_x(x) / (np.pi * self.R)
    
    # def beta(self, i):
    #     return self.beta[i]
    
    # def alpha_0(self, i):
    #     return self.alpha_0[i]


class FlightConditions:
    
    def __init__(self, V, rho, mu, a, RPM):
        self.V = V
        self.rho = rho
        self.mu = mu
        self.a = a
        self.RPM = RPM
        self.n = RPM2n(RPM)


#%%###########################

def SectionalLift(M):
    m_0 = np.zeros_like(M)
    for n, m in enumerate(M):
        if m >= 0.9:
            m_0[n] = (2 * np.pi)
            m_0[n] /= (1 - 0.9**2)**0.5
        else:
            m_0[n] = (2 * np.pi) 
            m_0[n] /= (1 - m**2)**0.5 
    return m_0

def GetSigma(atmos, prop):
    return prop.B * prop.c / (np.pi * prop.R)

def LocalM(atmos, prop):
    M = prop.V_R/atmos.a
    return M

def LocalV(atmos, omega, r):
    V = (atmos.V**2 + (omega*r)**2)**0.5
    return V

def KTAS2FPS(v):
    return 1.68781 * v

def RPM2n(RPM):
    RPS = RPM / 60
    return RPS

def GetV_R(prop, phi, x):
    r = x * prop.R
    V_R = 2 * np.pi * atmos.n * r
    V_R /= np.cos(phi)
    return V_R

def GetJ(atmos, prop):
    J = atmos.V / (atmos.n * prop.D)
    return J

def GetPhi(J, x):
    phi = np.arctan2(J, np.pi*x)
    return phi

def GetTheta(atmos, prop):
    A_0 = -1 * (prop.beta - prop.phi - prop.alpha_0) * prop.sigma * prop.m_0 / (8 * prop.x)
    
    A_1 = (atmos.V / prop.V_R)
    A_1 += ((prop.beta - prop.phi - prop.alpha_0) * np.tan(prop.phi) + 1) * prop.sigma * prop.m_0 / (8 * prop.x)
    
    A_2 = np.cos(prop.phi) - prop.sigma * np.tan(prop.phi) * prop.m_0 / (8 * prop.x) 
    
    theta = -A_1 + (A_1**2 - 4*A_2*A_0)**0.5
    theta /= 2*A_2
    return theta

def GetLambda_T(c_l, c_d, phi, theta):
    lambda_t = c_l * np.cos(phi+theta) - c_d * np.sin(phi+theta)
    lambda_t /= (np.cos(phi))**2
    return lambda_t

def GetLambda_Q(c_l, c_d, phi, theta):
    lambda_q = c_l * np.sin(phi+theta) + c_d * np.cos(phi+theta)
    lambda_q /= (np.cos(phi))**2
    return lambda_q

def GetAlpha(beta, phi, theta):
    return beta - phi - theta

def GetAlphaDesign(atmos, prop):
    return (prop.c_l / prop.m_0) + prop.alpha_0

def Getc_l(atmos, prop):
    c_l = prop.m_0 * (prop.alpha - prop.alpha_0)
    return c_l

def Getc_d(c_l):
    c_d_min = 0.0095
    k = 0.0040
    c_l_min = 0.2
    
    return c_d_min + k*(c_l - c_l_min)**2

def GetThetaBetz(atmos, prop):
    a = atmos.V + atmos.v_0
    b = 2 * np.pi * atmos.n *prop.r
    theta = np.arctan2(a, b) - prop.phi
    return theta

def Getc_lDesign(phi, theta, sigma, x):
    c_l_des = 8*x*np.cos(phi) * np.tan(theta+phi)
    c_l_des /= sigma
    return c_l_des

def GetsigmaDesign(atmos, prop):
    sigma = 8 * prop.x * prop.theta * np.cos(prop.phi) * np.tan(prop.theta + prop.phi)
    sigma /= prop.c_l
    return sigma


def PropellorDesign2(atmos, prop, x_list):    
    atmos.v_0 = 40
    for iterval in range(200):
        
        for i in range(len(x_list)-1):
            x = x_list[i]
            sigma = prop.GetSigma(x)
            J = GetJ(atmos, prop)
            phi = GetPhi(J, x)
            V_R = GetV_R(prop, phi, x)
            M = LocalM(atmos, prop, x)
            m_0 = SectionalLift(M)
            theta = GetThetaBetz(atmos, prop, phi, x)
            c_l = Getc_lDesign(phi, theta, sigma, x)
            alpha = GetAlphaDesign(prop, c_l, m_0, i)
            prop.beta[i] = alpha + phi + theta 
            c_d = Getc_d(c_l)
            lambda_t = GetLambda_T(c_l, c_d, phi, theta)
            lambda_q = GetLambda_Q(c_l, c_d, phi, theta)
            
        x_list, C_T_list, C_Q_list, eta_P, T, Q, P = PropellorAnalysis(atmos, prop, x_list)
                
        if abs(prop.P_eng - P) <= 1:
            print("break")
            break
        elif prop.P_eng > P:
            atmos.v_0 -= 0.1
        elif prop.P_eng < P:
            atmos.v_0 += 0.1
    
    return PropellorAnalysis(atmos, prop, x_list)

def PropellorAnalysis(atmos, prop):
    prop.J = GetJ(atmos, prop)
    prop.omega = 2 * np.pi * atmos.n
    prop.phi = GetPhi(prop.J, prop.x)
    prop.V_R = LocalV(atmos, prop.omega, prop.r)
    prop.M = LocalM(atmos, prop)
    prop.m_0 = SectionalLift(prop.M)
    prop.sigma = GetSigma(atmos, prop)
    prop.theta = GetTheta(atmos, prop)
    prop.alpha = GetAlpha(prop.beta, prop.phi, prop.theta)
    prop.c_l = Getc_l(atmos, prop)
    prop.c_d = Getc_d(prop.c_l)
    prop.lambda_t = GetLambda_T(prop.c_l, prop.c_d, prop.phi, prop.theta)
    prop.lambda_q = GetLambda_Q(prop.c_l, prop.c_d, prop.phi, prop.theta)
    prop["dCTdx"] = prop.sigma * (np.pi**3 * prop.x**2) * prop.lambda_t / 8
    prop["dCQdx"] = prop.sigma * (np.pi**3 * prop.x**3) * prop.lambda_q / 16
    
    C_T = np.trapz(prop.dCTdx, prop.x)
    C_Q = np.trapz(prop.dCQdx, prop.x)
    C_P = 2 * np.pi * C_Q

    
    T = C_T * atmos.rho * atmos.n**2 * prop.D[0]**4
    Q = C_Q * atmos.rho * atmos.n**2 * prop.D[0]**5
    P = C_P * atmos.rho * atmos.n**3 * prop.D[0]**5 / 550
    
    eta_P = prop.J[0] * C_T / C_P
    
    J = prop.J[0]

    AF = (10E5 / 16) * np.trapz(prop.c * prop.x**3 / prop.D, prop.x)
    C_L = 4 * np.trapz(prop.c_l*prop.x**3, prop.x)
    prop["p"]= 2 * np.pi * np.tan(prop.beta)
    p75 = np.interp(0.75, prop.x, prop.p)
    beta75 = np.arctan2(4*p75, 3*np.pi*prop.D[0])
    
    new_row = {"C_T": C_T,
               "C_Q": C_Q,
               "C_P": C_P,
               "J": J,
               "T": T,
               "Q": Q,
               "P": P,
               "eta_P": eta_P,
               "AF": AF,
               "C_L": C_L,
               "beta75": beta75}
    
    return new_row
    

def PropellorDesign(atmos, prop):
    repeat = True
    
    while repeat==True:
        prop["J"] = GetJ(atmos, prop)
        prop["omega"] = 2 * np.pi * atmos.n
        prop["phi"] = GetPhi(prop.J, prop.x)
        prop["V_R"] = LocalV(atmos, prop.omega, prop.r)
        prop["M"] = LocalM(atmos, prop)
        prop["m_0"] = SectionalLift(prop.M)
        prop["theta"] = GetThetaBetz(atmos, prop)
        prop["sigma"] = GetsigmaDesign(atmos, prop)
        prop["c"] = prop.sigma * np.pi * prop.R / prop.B
        prop["alpha"] = GetAlphaDesign(atmos, prop)
        prop["beta"] = prop.alpha + prop.phi + prop.theta
        prop["c_d"] = Getc_d(prop.c_l)
        prop["lambda_t"] = GetLambda_T(prop.c_l, prop.c_d, prop.phi, prop.theta)
        prop["lambda_q"] = GetLambda_Q(prop.c_l, prop.c_d, prop.phi, prop.theta)
        prop["dCTdx"] = prop.sigma * (np.pi**3 * prop.x**2) * prop.lambda_t / 8
        prop["dCQdx"] = prop.sigma * (np.pi**3 * prop.x**3) * prop.lambda_q / 16
        
        # prop, C_T, C_Q, C_P, eta_P = PropellorAnalysis(atmos, prop)
        
        C_T = np.trapz(prop.dCTdx, prop.x)
        C_Q = np.trapz(prop.dCQdx, prop.x)
        C_P = 2 * np.pi * C_Q
        
        T = C_T * atmos.rho * atmos.n**2 * prop.D[0]**4
        Q = C_Q * atmos.rho * atmos.n**2 * prop.D[0]**5
        P = C_P * atmos.rho * atmos.n**3 * prop.D[0]**5 / 550
        
        eta_P = prop.J[0] * C_T / C_P
        # print(eta_P)

            
        if abs(atmos.P_eng - P) <= 0.1:
            print("break")
            repeat = False
        elif atmos.P_eng >= P:
            atmos.v_0 += 0.001
        elif atmos.P_eng < P:
            atmos.v_0 -= 0.001
        else:
            pass
    
        # print(f"P = {P:.2f} hp")
        

    # print(f"eta_p = {eta_P:.2f}")
    # print(f"C_T = {C_T:.2f}")
    # print(f"C_Q = {C_Q:.2f}")
    # print(f"C_P = {C_P:.2f}")

    
    return prop


def RK4(Fdot, y, t, delta_t):
    k1 = delta_t * Fdot(y, t)
    k2 = delta_t * Fdot(y + k1/2, t + 0.5*delta_t)
    k3 = delta_t * Fdot(y + k2/2, t + 0.5*delta_t)
    k4 = delta_t * Fdot(y + k3, t+delta_t)
    
    y_nplusone = y + (k1 + 2*k2 + 2*k3 + k4)/6
    
    return y_nplusone

def GetThrust(atmos, prop, C_T_list, x_list):
    C_T = np.trapz(C_T_list, x_list)
    T = C_T * atmos.rho * atmos.n**2 * prop.D**4
    return T
    
def GetTorque(atmos, prop, C_Q_list, x_list):
    C_Q = np.trapz(C_Q_list, x_list)
    Q = C_Q * atmos.rho * atmos.n**2 * prop.D**5
    return Q

def GetPower(atmos, prop, C_P):
    return C_P * atmos.rho * atmos.n**3 * prop.D**5