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
"""
"""
-----------------------------------------------------------------------------------------------------------------------------------------
                                                Oscillating Mass Attached to a Spring with Damper
-----------------------------------------------------------------------------------------------------------------------------------------
"""

# User-defined Exact solution (for an undamped case)
def u_exact(t):

    return np.exp(-0.25 * t) * (0.125988 * np.sin(1.98431 * t) + np.cos(1.98431 * t))

# User-defined Damping function : f(u')
def damping_function(v):
    c = 0.5  # Damping coefficient
    return c * v

# User-defined restoring force : s(u)
def restoring_function(u):
    k = 4  # Spring constant (stiffness)
    return k * u

def external_force(t):
    A = 0  # No external force in this case, set A to a positive number if needed
    phi = 1.0  # Frequency of external force
    return A * np.sin(phi * t)

# Initial conditions: u(0) = I, v(0) = V
initial_conditions = (1.0, 0.0)  # Initial displacement and velocity

# Solver parameters
m = 1.0  # Mass
dt = 0.01  # Time step size
T = 10.0  # Total time to integrate

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
results = general_solver(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=u_exact)

# # Plot the results (with exact solution)
plot_results(results, exact_solution=u_exact, plot_velocity=True, filename="Ex2_plot_results.png") 
plot_results2(results, exact_solution=u_exact, plot_velocity=None, filename="Ex2_plot_results2.png")

# Test the first three steps against the exact solution (if defined)
test_three_steps(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=u_exact)


# Perform error analysis and print errors
list_errors(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=u_exact)

