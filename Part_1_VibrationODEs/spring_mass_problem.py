
import numpy as np
from oneDOFVibrationODESolver import general_solver, plot_results, plot_results2, test_three_steps, list_errors  # Import from the core module

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
                                                Oscillating Mass Attached to a Spring
-----------------------------------------------------------------------------------------------------------------------------------------
"""

# User-defined Exact solution (if available)
def u_exact(t):
    I = 1.0  # Initial displacement
    w = 2 * np.pi  # Angular frequency (depends on spring constant and mass)
    return I * np.cos(w * t)

# User-defined Damping function : f(u')
def damping_function(v):
    return 0.0  # No damping in this idealized case, but can be added if needed

# User-defined restoring force : s(u) (Hooke's Law for a spring)
def restoring_function(u):
    k = 2 * np.pi  # Spring constant
    return k**2 * u  # Force exerted by the spring (Hooke's Law: F = -kx)

# User-defined External force : F(t)
def external_force(t):
    return 0.0  # No external force applied in this scenario

# User-defined Initial conditions: u(0) = I, v(0) = V
initial_conditions = (1.0, 0.0)  # Displacement I = 1.0, velocity V = 0.0

# User-defined Solver Parameters
m = 1.0  # Mass attached to the spring
dt = 0.01  # Time step size
T = 1.0  # Total time to integrate

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
plot_results(results, exact_solution=u_exact, plot_velocity=True, filename="Ex1_plot_results.png") 
plot_results2(results, exact_solution=u_exact, plot_velocity=None, filename="Ex1_plot_results2.png")

# Test the first three steps against the exact solution (if defined)
test_three_steps(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=u_exact)


# Perform error analysis and print errors
list_errors(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=u_exact)

