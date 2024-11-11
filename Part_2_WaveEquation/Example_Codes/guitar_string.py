import numpy as np
import matplotlib.pyplot as plt
import os
from wave_eq_solver_dev import (
    solve_wave_equation,
    save_wave_image,
    generate_gif_from_images,
    generate_html_animation,
)

# ===============================
# Problem Setup and Parameters
# ===============================

# Define the physical parameters of the guitar string
L = 0.75           # Length of the domain representing the guitar string
x0 = 0.8 * L       # Position where the initial displacement changes slope
a = 0.005          # Amplitude of the initial displacement, representing the maximum displacement
freq = 440         # Frequency of the wave in Hertz (standard frequency for the A note)
wavelength = 2 * L # Wavelength of the wave, assuming it spans twice the domain length
c = freq * wavelength  # Wave speed, calculated as frequency times wavelength
omega = 2 * np.pi * freq  # Angular frequency of the wave (used for time-dependent calculations)
num_periods = 1    # Number of wave periods to simulate

# Total simulation time, ensuring we capture one complete wave period
T = 2 * np.pi / omega * num_periods  # Total simulation time for one period

# Courant number, a factor that influences stability in the numerical method
C = 0.5            # Courant number, should be <= 1 for stability in this scheme

# Spatial resolution parameters
Nx = 50            # Number of spatial points along the domain, affects resolution of the string
dx = L / Nx        # Spatial step size, derived from the length and resolution of the domain

# Time step size, calculated based on Courant number, spatial step, and wave speed
dt = C * dx / c    # Time step size

# ===============================
# Initial and Boundary Conditions
# ===============================

# Define initial displacement, velocity, and source term
def initial_displacement(x):
    """
    Initial condition for displacement. The displacement forms a triangular shape,
    representing how a guitar string might be initially plucked.
    """
    return a * x / x0 if x < x0 else a / (L - x0) * (L - x)

def initial_velocity(x):
    """
    Initial condition for velocity. Assumes the string is released from rest,
    so the initial velocity is zero at all points.
    """
    return np.zeros_like(x)

def source_term(x, t):
    """
    Source term f(x, t), assumed zero here as there is no external force acting
    on the string after it is plucked.
    """
    return np.zeros_like(x)

# ===============================
# Data Storage and Animation
# ===============================

# Directory to save generated frames for the animation
save_dir = 'C:\\Users\\pbosn\\OneDrive - USN\\PhD WorkFlow\\WorkFlow\\Courses\\Required Courses\\Numerical Solutions for PDE\\2_Wave_Equations\\Develop\\FullyWorkingSolverAndTest\\guitar_string_simulation'

# Ensure that the directory for saving images exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Array to store solution data for HTML animation
results = []

# Capture results at each time step for HTML animation
def capture_results(u, x, t, n):
    """
    Function to capture each time step's solution for later animation.
    Appends the current solution and time to the results array.
    """
    results.append((u.copy(), t[n]))

# Save each frame as an image for GIF creation
def save_guitar_string_image(u, x, t, n, C, save_dir=save_dir, ymin=-0.005, ymax=0.005):
    """
    Wrapper around the save_wave_image function to save each frame of the string
    motion with appropriate y-axis limits.
    """
    save_wave_image(u, x, t, n, C, save_dir=save_dir, ymin=ymin, ymax=ymax)

# Combined user action that calls both capture_results and save_wave_image
def combined_user_action(u, x, t, n):
    """
    This function is called at each time step of the solver. It captures
    results for HTML animation and saves images for the GIF.
    """
    capture_results(u, x, t, n)  # Capture data for HTML animation
    save_guitar_string_image(u, x, t, n, C)  # Save image for GIF

# ===============================
# Solver Execution Based on User Choice
# ===============================

# Prompt user to choose between scalar or vectorized solver versions
solver_choice = input("Choose solver (scalar/vectorized/vectorized2): ").strip()

# Validate user choice and run the solver if a valid choice is made
if solver_choice in ["scalar", "vectorized", "vectorized2"]:
    solve_wave_equation(
        I=initial_displacement,    # Initial displacement function
        V=initial_velocity,        # Initial velocity function
        f=source_term,             # Source term function
        c=c,                       # Wave speed
        L=L,                       # Length of the domain
        dt=dt,                     # Time step size
        C=C,                       # Courant number
        T=T,                       # Total simulation time
        user_action=combined_user_action,  # Combined action for results capture and image saving
        version=solver_choice,     # Solver version: 'scalar', 'vectorized', or 'vectorized2'
        save_dir=save_dir,         # Directory to save images
    )
else:
    print("Invalid choice. Please choose 'scalar', 'vectorized', or 'vectorized2'.")

# ===============================
# Animation Generation
# ===============================

# Generate a GIF from the saved images to visualize the wave propagation over time
generate_gif_from_images(image_folder=save_dir, gif_name='wave_animation.gif', duration=0.1)

# Generate HTML animation from captured results, providing an interactive view
x = np.linspace(0, L, int(L / dx) + 1)  # Create spatial grid
generate_html_animation(x, results, save_dir, ymin=-0.005, ymax=0.005, fps=10)
