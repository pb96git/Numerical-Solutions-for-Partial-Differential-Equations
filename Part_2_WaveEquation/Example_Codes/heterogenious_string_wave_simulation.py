import numpy as np
import matplotlib.pyplot as plt
import os
from wave_eq_solver_dev import solve_wave_equation_variable_velocity, save_wave_image, generate_gif_from_images, generate_html_animation

# ===============================
# Problem Setup and Parameters
# ===============================

# Length of the domain and the number of spatial points
L = 1.0                      # Total length of the domain
Nx = 100                      # Number of spatial points for resolution

# Courant number, which is essential for stability in wave equations.
# Here, we set C = 1 for the maximum stability within homogeneous regions.
C = 1

# Total simulation time, which will determine how long we observe the wave motion.
T = 1

# Periodicity of pulses at the left boundary; we emit a pulse every `pulse_period` seconds.
pulse_period = 2.0

# Duration of each pulse; each pulse lasts `pulse_duration` seconds.
pulse_duration = 0.16

# Directory to save generated frames for the animation
save_dir = 'C:\\Users\\pbosn\\OneDrive - USN\\PhD WorkFlow\\WorkFlow\\Courses\\Required Courses\\Numerical Solutions for PDE\\2_Wave_Equations\\Develop\\FullyWorkingSolverAndTest\\heterogenious_moving_wave_simulation'

# Ensure that the directory for saving images exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ===============================
# Material Properties and Wave Speeds
# ===============================

# Density and wave speed for each medium
rho1 = 1.0                    # Density in medium 1
rho2 = 4.0                    # Density in medium 2, 8 times denser than medium 1
c1 = 1                        # Wave speed in medium 1
c2 = 0.5                      # Wave speed in medium 2, slower in denser medium

# # Density and wave speed for each medium
# rho1 = 4.0                    # Density in medium 1, 4 times denser than medium 2
# rho2 = 1.0                    # Density in medium 2
# c1 = 0.5                        # Wave speed in medium 1, slower in denser medium
# c2 = 1                        # Wave speed in medium 2

# Derived values
dx = L / Nx                   # Spatial step size based on the number of points
dt = C * dx / max(c1, c2)     # Time step size derived from the Courant number for stability
Z1 = rho1 * c1                # Impedance in medium 1
Z2 = rho2 * c2                # Impedance in medium 2

# Reflection and Transmission coefficients based on impedance mismatch.
# These will be used to calculate the amplitude of reflected and transmitted waves at the boundary between media.
reflection_coeff = (Z1 - Z2) / (Z1 + Z2)
transmission_coeff = (2 * Z1) / (Z1 + Z2)

# ===============================
# Define Wave Speed Across the Domain
# ===============================

def variable_wave_speed(x):
    """
    Function to define q(x) = c(x)^2 across the domain.
    The wave speed changes at x = L/3, representing an impedance discontinuity.
    """
    midpoint = L / 3
    return np.where(x < midpoint, c1**2, c2**2)

# ===============================
# Initial Conditions and Boundary Pulses
# ===============================

# Initial displacement and velocity (both set to zero for a stationary initial state)
def initial_displacement(x):
    return np.zeros_like(x)

def initial_velocity(x):
    return np.zeros_like(x)

# Define a periodic pulse at the left boundary that triggers every `pulse_period`
def U_0(t):
    """
    Boundary condition at x=0, generating a sinusoidal pulse at regular intervals.
    Only active for `pulse_duration` within each `pulse_period`.
    """
    pulse_time = t % pulse_period
    return 0.25 * np.sin(6 * np.pi * pulse_time) if pulse_time <= pulse_duration else 0

# Source term (no external source, so it is set to zero)
def source_term(x, t):
    return np.zeros_like(x)

# ===============================
# Solution Storage and Animation Functions
# ===============================

# Array to store solution data for HTML animation
results = []

# Capture results at each time step for HTML animation
def capture_results(u, x, t, n):
    results.append((u.copy(), t[n]))

# Combined user action to capture results and save images for GIF creation
def combined_user_action(u, x, t, n):
    capture_results(u, x, t, n)  # Capture data for HTML animation
    save_wave_image(u, x, t, n, C, save_dir=save_dir, ymin=-0.6, ymax=0.7)  # Save image for GIF

# ===============================
# Setting Up the Spatial Grid and Wave Speed
# ===============================

# Spatial grid across the domain and calculating q(x) based on wave speed
x = np.linspace(0, L, Nx + 1)
q = variable_wave_speed(x)

# ===============================
# Solver Wrapper for Variable Velocity
# ===============================

def solve_wave_equation_with_variable_velocity(I, V, f, q, L, dt, C, T, U_0, user_action=None, version='scalar', boundary='Dirichlet'):
    """
    Wrapper to solve the wave equation with variable wave velocity `q(x)` and a moving boundary condition `U_0`.
    
    Parameters:
    - I: Initial displacement function
    - V: Initial velocity function
    - f: Source term function
    - q: Spatially varying wave speed squared (q(x) = c(x)^2)
    - L: Length of the domain
    - dt: Time step size
    - C: Courant number
    - T: Total simulation time
    - U_0: Left boundary pulse function
    - user_action: Action to take at each time step (e.g., save images)
    """
    Nt = int(round(T / dt))   # Number of time steps based on T and dt
    t = np.linspace(0, Nt * dt, Nt + 1)  # Time grid
    dx = L / Nx               # Reconfirming spatial step
    x = np.linspace(0, L, Nx + 1)  # Spatial grid
    
    # Wrapped action to apply boundary condition at x = 0 (left boundary)
    def wrapped_user_action(u, x, t, n):
        u[0] = U_0(t[n])  # Apply the boundary condition at each time step
        if user_action:
            user_action(u, x, t, n)
    
    # Run the variable velocity solver with the wrapped boundary condition
    solve_wave_equation_variable_velocity(
        I=I, V=V, f=f, q=q, L=L, dt=dt, C=C, T=T,
        user_action=wrapped_user_action,
        version=version,
        boundary=boundary
    )

# ===============================
# Execute the Solver and Generate Animations
# ===============================

# Run the solver with variable velocity and the periodic boundary pulse
solve_wave_equation_with_variable_velocity(
    I=initial_displacement,
    V=initial_velocity,
    f=source_term,
    q=q,
    L=L,
    dt=dt,
    C=C,
    T=T,
    U_0=U_0,
    user_action=combined_user_action,
    version='scalar',
    boundary='Dirichlet'
)

# Generate a GIF from the saved images for visualizing the wave propagation
generate_gif_from_images(image_folder=save_dir, gif_name='variable_velocity_pulse.gif', duration=0.1)

# Generate HTML animation from captured results
generate_html_animation(x, results, save_dir, filename="wave_with_single_pulse.html", ymin=-0.6, ymax=0.7)
