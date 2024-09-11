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
                                                y'' - 8 y' + 16 y = e^(4 x), y(0) = 0, y'(0) = 1
-----------------------------------------------------------------------------------------------------------------------------------------
"""

# Damping function: f(u')
def damping_function(v):
    return -8 * v

# Restoring function: s(u)
def restoring_function(u):
    return 16 * u

# External force: F(t)
def external_force(t):
    return np.exp(4 * t)

def u_exact(t):
    return 0.5 * np.exp(4 * t) * (t**2 + 2 * t)

# Initial conditions: u(0), v(0)
initial_conditions = (0, 1)  # y(0) = 0, y'(0) = 1

# User-defined Solver Parameters
m = 1.0  # Mass attached to the spring
dt = 0.01  # Time step size
T = 1  # Total time to integrate

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

# # Plot the results 
plot_results(results, exact_solution=u_exact, plot_velocity=True, filename="Ex5_plot_results.png") 
plot_results2(results, exact_solution=u_exact, plot_velocity=None, filename="Ex5_plot_results2.png")

# Test the first three steps against the exact solution (if defined)
test_three_steps(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=u_exact)


# Perform error analysis 
list_errors(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=u_exact)