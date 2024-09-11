# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:57:28 2024

@author: pbosn
"""
import numpy as np
from oneDOFVibrationODESolver import general_solver, plot_results, plot_results2, test_three_steps, list_errors  # Import from the core module  # Import from the core module

"""
-----------------------------------------------------------------------------------------------------------------------------------------
                                            USER INPUT

Solves the system: m*u'' + f(u') + s(u) = F(t) using multiple methods.

- m: mass
- damping_func: user-defined function for f(u') (damping term)
- restoring_func: user-defined function for s(u) (restoring term)
- external_force: user-defined function for F(t) (external force)
- initial_conditions: tuple (u(0), v(0)) -> initial displacement and velocity
- dt: time step
- T: total time
- methods: list of methods to use for solving ('CD', 'FE', 'BE', 'CN', 'RK2', 'RK4', 'EC')


-----------------------------------------------------------------------------------------------------------------------------------------

"""
"""
-----------------------------------------------------------------------------------------------------------------------------------------
                                                               1DOF Problems
-----------------------------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------------------------
                                                A Sliding Mass Attached to a Spring
-----------------------------------------------------------------------------------------------------------------------------------------
"""

# User-defined damping function (frictional damping): f(u')
def damping_function(v):
    mu = 0.1  # Friction coefficient
    g = 9.81  # Gravitational acceleration (m/s^2)
    return mu * g * np.sign(v)  # Damping force due to friction

# User-defined restoring force (spring restoring force): s(u)
def restoring_function(u):
    k = 50  # Spring constant (N/m)
    return k * u  # Restoring force from the spring

# User-defined external force (constant force): F(t)
def external_force(t):
    F_0 = 10  # Constant external force (N)
    return F_0  # Constant external force applied

# Initial conditions: u(0) = 1 (displacement), v(0) = 0 (velocity)
initial_conditions = (1.0, 0.0)

# System parameters
m = 1.0  # Mass (kg)
dt = 0.01  # Time step size (s)
T = 10.0  # Total simulation time (s)

"""
-----------------------------------------------------------------------------------------------------------------------------------------
                                                        Choose Numerical Methods
-----------------------------------------------------------------------------------------------------------------------------------------
"""

# # Define the methods that solver will use
methods = ['CD', 'FE', 'BE', 'CN', 'RK2', 'RK4', 'EC']  # Numerical methods ('CD', 'FE', 'BE', 'CN', 'RK2', 'RK4', 'EC')


"""
-----------------------------------------------------------------------------------------------------------------------------------------
                                                ACTIVATE/DEACTIVATE FUNCTIONS
If exact solution is exits and is provided, define it inside functions that you are calling: exact_solution = u_exact, 
else exact_solution = None
-----------------------------------------------------------------------------------------------------------------------------------------

"""

# # Solve the system using the general solver
results = general_solver(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=None)

# # Plot the results (with exact solution)
plot_results(results, exact_solution=None, plot_velocity=True, filename="Ex3_plot_results.png") 
plot_results2(results, exact_solution=None, plot_velocity=None, filename="Ex3_plot_results2.png")

# Test the first three steps against the exact solution (if defined)
test_three_steps(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=None)


# Perform error analysis and print errors
list_errors(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=None)
