import torch
import torch.nn.functional as Ffunctional
import copy
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pylab import grid
import time

#@title CSTR code from tutorial 6

eps  = np.finfo(float).eps

###############
#  CSTR model #
###############

# Taken from http://apmonitor.com/do/index.php/Main/NonlinearControl

def cstr(x,t,u):

    # ==  Inputs == #
    Tc  = u   # Temperature of cooling jacket (K)

    # == States == #
    Ca = x[0] # Concentration of A in CSTR (mol/m^3)
    T  = x[1] # Temperature in CSTR (K)

    # == Process parameters == #
    Tf     = 350    # Feed temperature (K)
    q      = 100    # Volumetric Flowrate (m^3/sec)
    Caf    = 1      # Feed Concentration (mol/m^3)
    V      = 100    # Volume of CSTR (m^3)
    rho    = 1000   # Density of A-B Mixture (kg/m^3)
    Cp     = 0.239  # Heat capacity of A-B Mixture (J/kg-K)
    mdelH  = 5e4    # Heat of reaction for A->B (J/mol)
    EoverR = 8750   # E -Activation energy (J/mol), R -Constant = 8.31451 J/mol-K
    k0     = 7.2e10 # Pre-exponential factor (1/sec)
    UA     = 5e4    # U -Heat Transfer Coefficient (W/m^2-K) A -Area - (m^2)
    
    # == Equations == #
    rA     = k0*np.exp(-EoverR/T)*Ca # reaction rate
    dCadt  = q/V*(Caf - Ca) - rA     # Calculate concentration derivative
    dTdt   = q/V*(Tf - T) + mdelH/(rho*Cp)*rA + UA/V/rho/Cp*(Tc-T)   # Calculate temperature derivative

    # == Return xdot == #
    xdot    = np.zeros(2)
    xdot[0] = dCadt
    xdot[1] = dTdt
    return xdot

# Dictonary for concentration and temperature of the reactor data
data_res = {} 

# Initial conditions for the states
x0             = np.zeros(2)
x0[0]          = 0.87725294608097
x0[1]          = 324.475443431599
data_res['x0'] = x0

# Time interval (min)
n             = 101 # number of intervals
tp            = 25 # process time (min)
t             = np.linspace(0,tp,n)
data_res['t'] = t
data_res['n'] = n

# Store results for plotting
Ca = np.zeros(len(t));      Ca[0]  = x0[0]
T  = np.zeros(len(t));      T[0]   = x0[1]    
Tc = np.zeros(len(t)-1);   

data_res['Ca_dat'] = copy.deepcopy(Ca)
data_res['T_dat']  = copy.deepcopy(T) 
data_res['Tc_dat'] = copy.deepcopy(Tc)

# noise level
noise             = 0.1
data_res['noise'] = noise

# control upper and lower bounds
data_res['Tc_ub']  = 305
data_res['Tc_lb']  = 295
Tc_ub              = data_res['Tc_ub']
Tc_lb              = data_res['Tc_lb']

# desired setpoints
n_1                = int(n/2)
n_2                = n - n_1
Ca_des             = [0.8 for i in range(n_1)] + [0.9 for i in range(n_2)]
T_des              = [330 for i in range(n_1)] + [320 for i in range(n_2)]
data_res['Ca_des'] = Ca_des
data_res['T_des']  = T_des

# for key, value in data_res.items():
#     print(f"{key}:")
#     print(value)

#@title Ploting routines

####################################
# plot control actions performance #
####################################

def plot_simulation(Ca_dat, T_dat, Tc_dat, data_simulation):    
    
    Ca_des = data_simulation['Ca_des']
    T_des = data_simulation['T_des']
    
    plt.figure(figsize=(8, 5))

    plt.subplot(3,1,1)
    plt.plot(t, np.median(Ca_dat,axis=1), 'r-', lw=3)
    plt.gca().fill_between(t, np.min(Ca_dat,axis=1), np.max(Ca_dat,axis=1), 
                           color='r', alpha=0.2)
    plt.step(t, Ca_des, '--', lw=1.5, color='black')
    plt.ylabel('Ca (mol/m^3)')
    plt.xlabel('Time (min)')
    plt.legend(['Concentration of A in CSTR'],loc='best')
    plt.xlim(min(t), max(t))

    plt.subplot(3,1,2)
    plt.plot(t, np.median(T_dat,axis=1), 'c-', lw=3)
    plt.gca().fill_between(t, np.min(T_dat,axis=1), np.max(T_dat,axis=1), 
                           color='c', alpha=0.2)
    plt.step(t, T_des, '--', lw=1.5, color='black')
    plt.ylabel('T (K)')
    plt.xlabel('Time (min)')
    plt.legend(['Reactor Temperature'],loc='best')
    plt.xlim(min(t), max(t))

    plt.subplot(3,1,3)
    plt.step(t[1:], np.median(Tc_dat,axis=1), 'b--', lw=3)
    plt.ylabel('Cooling T (K)')
    plt.xlabel('Time (min)')
    plt.legend(['Jacket Temperature'],loc='best')
    plt.xlim(min(t), max(t))

    plt.tight_layout()
    plt.show()

# ##################
# # Training plots #
# ##################

# def plot_training(data_simulation, repetitions):
#     t        = data_simulation['t'] 
#     Ca_train = np.array(data_simulation['Ca_train'])
#     T_train = np.array(data_simulation['T_train'])
#     Tc_train = np.array(data_simulation['Tc_train'])
#     Ca_des   = data_simulation['Ca_des']
#     T_des    = data_simulation['T_des']

#     c_    = [(repetitions - float(i))/repetitions for i in range(repetitions)]

#     plt.figure(figsize=(8, 5))

#     plt.subplot(3,1,1)
#     for run_i in range(repetitions):
#         plt.plot(t, Ca_train[run_i,:], 'r-', lw=1, alpha=c_[run_i])
#     plt.step(t, Ca_des, '--', lw=1.5, color='black')
#     plt.ylabel('Ca (mol/m^3)')
#     plt.xlabel('Time (min)')
#     plt.legend(['Concentration of A in CSTR'],loc='best')
#     plt.title('Training plots')
#     plt.ylim([.75, .95])
#     plt.xlim(min(t), max(t))
#     grid(True)

#     plt.subplot(3,1,2)
#     for run_i in range(repetitions):
#         plt.plot(t, T_train[run_i,:], 'c-', lw=1, alpha=c_[run_i])
#     plt.step(t, T_des, '--', lw=1.5, color='black')
#     plt.ylabel('T (K)')
#     plt.xlabel('Time (min)')
#     plt.legend(['Reactor Temperature'],loc='best')
#     plt.ylim([335, 317])
#     plt.xlim(min(t), max(t))
#     grid(True)

#     plt.subplot(3,1,3)
#     for run_i in range(repetitions):
#         plt.step(t[1:], Tc_train[run_i,:], 'b--', lw=1, alpha=c_[run_i])
#     plt.ylabel('Cooling T (K)')
#     plt.xlabel('Time (min)')
#     plt.legend(['Jacket Temperature'],loc='best')
#     plt.xlim(min(t), max(t))
#     grid(True)
    
#     plt.tight_layout()

#     plt.show()

# #####################
# # Convergence plots #
# #####################

# def plot_convergence(Xdata, best_Y, Objfunc=None):
#     '''
#     Plots to evaluate the convergence of standard Bayesian optimization algorithms
#     '''
#     ## if f values are not given
#     f_best  = 1e8
#     if best_Y==None: 
#         best_Y = []
#         for i_point in range(Xdata.shape[0]):
#             f_point = Objfunc(Xdata[i_point,:], collect_training_data=False)
#             if f_point < f_best:
#                 f_best = f_point 
#             best_Y.append(f_best)
#         best_Y = np.array(best_Y)

#     n = Xdata.shape[0]
#     aux = (Xdata[1:n,:]-Xdata[0:n-1,:])**2
#     distances = np.sqrt(aux.sum(axis=1))

#     ## Distances between consecutive x's
#     plt.figure(figsize=(9,3))
#     plt.subplot(1, 2, 1)
#     plt.plot(list(range(n-1)), distances, '-ro')
#     plt.xlabel('Iteration')
#     plt.ylabel('d(x[n], x[n-1])')
#     plt.title('Distance between consecutive x\'s')
#     plt.xlim(0, n)
#     grid(True)

#     # Best objective value found over iterations
#     plt.subplot(1, 2, 2)
#     plt.plot(list(range(n)), best_Y,'-o')
#     plt.title('Value of the best selected sample')
#     plt.xlabel('Iteration')
#     plt.ylabel('Best y')
#     grid(True)
#     plt.xlim(0, n)
#     plt.tight_layout()
#     plt.show()
