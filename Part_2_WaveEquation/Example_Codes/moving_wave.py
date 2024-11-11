import numpy as np
import matplotlib.pyplot as plt
import os
from wave_eq_solver_dev import solve_wave_equation, generate_gif_from_images, generate_html_animation

# ===================================
# Main Simulation Parameters
# ===================================

L = 1.0                       # Length of the domain
c = 1.0                       # Wave speed
Nx = 50                       # Number of spatial points for resolution
C = 1                         # Courant number, controls stability of the numerical method
T = 2.0                       # Total simulation time
dt = C * (L / Nx) / c         # Time step size, calculated based on wave speed and Courant number
save_dir = 'C:\\Users\\pbosn\\OneDrive - USN\\PhD WorkFlow\\WorkFlow\\Courses\\Required Courses\\Numerical Solutions for PDE\\2_Wave_Equations\\Develop\\FullyWorkingSolverAndTest\\moving_wave'  # Directory to save images

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ===================================
# Boundary Condition
# ===================================

def U_0(t):
    """
    Defines the sinusoidal oscillation at the left boundary of the domain.
    Oscillates at specific intervals and remains zero otherwise.
    """
    return 0.25 * np.sin(6 * np.pi * t) if ((t < 1./6) or (0.5 + 3./12 <= t <= 0.5 + 4./12) or (1.5 <= t <= 1.5 + 1./3)) else 0


# # Define a periodic pulse at the left boundary that triggers every `pulse_period`
# def U_0(t):
#     """
#     Boundary condition at x=0, generating a sinusoidal pulse at regular intervals.
#     Only active for `pulse_duration` within each `pulse_period`.
#     """
#     # Periodicity of pulses at the left boundary; we emit a pulse every `pulse_period` seconds.
#     pulse_period = 2.0
#     # Duration of each pulse; each pulse lasts `pulse_duration` seconds.
#     pulse_duration = 0.15
#     pulse_time = t % pulse_period
#     return 0.25 * np.sin(6 * np.pi * pulse_time) if pulse_time <= pulse_duration else 0

# ===================================
# Initial Conditions and Source Term
# ===================================

def initial_displacement(x):
    """
    Sets the initial displacement to zero across the entire domain.
    """
    return np.zeros_like(x)

def initial_velocity(x):
    """
    Sets the initial velocity to zero across the entire domain.
    """
    return np.zeros_like(x)

def source_term(x, t):
    """
    Defines the source term f(x, t), which is zero for this simulation.
    """
    return np.zeros_like(x)

# ===================================
# Animation Data Storage
# ===================================

# Array to store solution data for HTML animation
results = []

# Function to capture each time step's solution for HTML animation
def capture_results(u, x, t, n):
    """
    Stores the wave state (displacement values u) and current time step
    for later use in creating the HTML animation.
    """
    results.append((u.copy(), t[n]))

# Function to save each time step's image
def save_wave_image(u, x, t, n, save_dir='wave_images', ymin=-0.6, ymax=0.7):
    """
    Saves the current wave state as an image for each time step, enabling
    creation of a visual record of the wave propagation.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(8, 4))
    plt.plot(x, u, label=f"Numerical Solution (t = {t[n]:.3f})", color="blue")
    plt.ylim(ymin, ymax)
    plt.xlim(0, L)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f"Wave Propagation at t = {t[n]:.3f}, Courant number = {C}")
    plt.legend()
    plt.grid(True)

    filename = os.path.join(save_dir, f'wave_step_{n:04d}.png')
    plt.savefig(filename)
    plt.close()

# Combined user action to capture both results and images
def combined_user_action(u, x, t, n):
    """
    Called at each time step of the solver to store data for HTML animation
    and save the wave state as an image for GIF creation.
    """
    capture_results(u, x, t, n)  # Store data for HTML animation
    save_wave_image(u, x, t, n, save_dir=save_dir, ymin=-0.6, ymax=0.7)  # Save image for GIF

# ===================================
# Solver Execution with Boundary Condition
# ===================================

def solve_wave_equation_with_moving_left_boundary(I, V, f, c, L, dt, C, T, U_0, user_action=None, version='scalar', boundary='Dirichlet'):
    """
    Wrapper function for solving the wave equation with a moving left boundary
    condition defined by U_0.
    """
    Nt = int(round(T / dt))
    t = np.linspace(0, Nt * dt, Nt + 1)
    dx = L / Nx
    x = np.linspace(0, L, Nx + 1)
    
    def wrapped_user_action(u, x, t, n):
        u[0] = U_0(t[n])  # Apply the moving left boundary condition at each step
        if user_action:
            user_action(u, x, t, n)
    
    # Call the solver with the modified boundary condition
    solve_wave_equation(
        I=I, V=V, f=f, c=c, L=L, dt=dt, C=C, T=T,
        user_action=wrapped_user_action,
        version=version,
        boundary=boundary
    )

# ===================================
# Execute Solver and Generate Outputs
# ===================================

# Run the solver with the specified initial conditions, boundary, and user actions
solve_wave_equation_with_moving_left_boundary(
    I=initial_displacement,            # Initial displacement function
    V=initial_velocity,                # Initial velocity function (zero)
    f=source_term,                     # Source term function (zero)
    c=c,                               # Wave speed
    L=L,                               # Length of the domain
    dt=dt,                             # Time step size
    C=C,                               # Courant number
    T=T,                               # Total simulation time
    U_0=U_0,                           # Moving boundary condition at the left end
    user_action=combined_user_action,  # Combined user action for capturing results
    version='scalar',                  # Solver version: 'scalar', 'vectorized', or 'vectorized2'
    boundary='leftFree'                # Boundary condition (left free for oscillation)
)

# ===================================
# Animation Creation
# ===================================

# Generate a GIF from the saved images to visualize the wave propagation over time
generate_gif_from_images(image_folder=save_dir, gif_name='wave_with_moving_boundary.gif', duration=0.1)

# Generate HTML animation from captured results
x = np.linspace(0, L, Nx + 1)  # Spatial grid for plotting
generate_html_animation(x, results, save_dir, filename="wave_with_moving_boundary.html", ymin=-0.6, ymax=0.7)
