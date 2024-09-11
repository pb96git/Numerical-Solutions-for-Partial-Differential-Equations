# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:57:28 2024

@author: pbosn
"""
import numpy as np
from oneDOFVibrationODESolver import general_solver, plot_results, plot_results2, test_three_steps, list_errors  # Import from the core module  # Import from the core module
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
                                                              Simple Pendulum
-----------------------------------------------------------------------------------------------------------------------------------------
"""

# Define damping, restoring, and external forces for the pendulum

def damping_function(v):
    C_D = 1.0  # Drag coefficient
    rho = 1.225  # Density of air in kg/m^3
    A = 0.1  # Cross-sectional area of the pendulum bob in m^2
    L = 1.0  # Length of the pendulum
    return 0.5 * C_D * rho * A * L * np.abs(v) * v

def restoring_function(theta):
    g = 9.81  # Acceleration due to gravity
    L = 1.0  # Length of the pendulum
    return (g / L) * np.sin(theta)

def external_force(t):
    return 0.0  # No external force

# Initial conditions and system parameters
initial_conditions = (np.pi / 2, 0.0)  # Initial angle of 45 degrees and zero velocity
m = 1.0  # Mass in kg
dt = 0.1  # Time step in seconds
T = 10  # Total time in seconds

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
plot_results(results, exact_solution=None, plot_velocity=True, filename="Ex4_plot_results.png") 
plot_results2(results, exact_solution=None, plot_velocity=None, filename="Ex4_plot_results2.png")

# Test the first three steps against the exact solution (if defined)
test_three_steps(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=None)


# Perform error analysis and print errors
list_errors(m, damping_function, restoring_function, external_force, initial_conditions, dt, T, methods, exact_solution=None)



"""
-----------------------------------------------------------------------------------------------------------------------------------------
                                               ANIMATION SETUP
-----------------------------------------------------------------------------------------------------------------------------------------
"""
# Extract results for the first method
u, v, t = results[methods[5]] # Method 5(6) is RK4

#Length of pendulum, defined as in restoring function
L = 1


# Convert angular displacement (theta = u) to x, y coordinates of the pendulum bob
x = L * np.sin(u)
y = -L * np.cos(u)

# Set up the figure and axis for the animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-L * 1.2, L * 1.2)
ax.set_ylim(-L * 1.2, L * 0.2)
ax.set_aspect('equal', 'box')
ax.set_xlabel("Distance x")
ax.set_ylabel("Distance y")

# Add grid and background
# ax.grid(True)

# Plot elements: the pendulum rod, bob, and the path
rod, = ax.plot([], [], lw=2, color='black')  # The pendulum rod
bob, = ax.plot([], [], 'o', markersize=15, color='red')  # The pendulum bob
path, = ax.plot([], [], '--', color='blue', alpha=0.5)  # The path traced by the bob

# Add support line at the top (from x = -0.2 to x = 0.2 at y = 0)
support_line, = ax.plot([-0.2, 0.2], [0, 0], lw=3, color='gray')  # The support


# Add labels for time and angle (θ)
angle_text = ax.text(-L * 1.1, 0.1, '', fontsize=12, color='black')  # Adjusted to stay within the y-axis bounds
time_text = ax.text(L * 0.5, 0.1, '', fontsize=12, color='black')  # Text to display time

# Function to initialize the animation
def init():
    rod.set_data([], [])
    bob.set_data([], [])
    path.set_data([], [])
    angle_text.set_text('')
    time_text.set_text('')
    return rod, bob, path, angle_text, time_text

# Function to update the animation at each frame
def update(frame):
    # Update rod coordinates
    rod.set_data([0, x[frame]], [0, y[frame]])
    # Update bob coordinates
    bob.set_data(x[frame], y[frame])
    # Update the path (trace of the pendulum bob)
    path.set_data(x[:frame+1], y[:frame+1])
    # Update angle and time display
    angle_text.set_text(f'θ = {np.degrees(u[frame]):.1f}°')
    time_text.set_text(f'Time = {t[frame]:.2f} s')
    return rod, bob, path, angle_text, time_text

# Create the animation with real-time duration (T seconds in real time)
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=dt*1000, repeat=False)

# Save the animation as a GIF
gif_filename = 'real_time_pendulum_animation.gif'
writergif = PillowWriter(fps=30)
ani.save(gif_filename, writer=writergif)

print(f"Animation saved as {gif_filename}")

# Show the animation (if running interactively)
plt.title("Simple Pendulum Motion Animation")
plt.show()