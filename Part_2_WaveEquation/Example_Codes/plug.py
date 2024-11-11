import numpy as np
import matplotlib.pyplot as plt
import os
from wave_eq_solver_dev import solve_wave_equation, generate_gif_from_images, generate_html_animation

# ===================================
# Main Simulation Parameters
# ===================================

L = 1.0              # Length of the domain
c = 1                # Wave speed
Nx = 50              # Number of spatial points for resolution
C = 1                # Courant number (used for stability)
T = 2                # Total simulation time
loc = 0.5            # Location of the initial plug profile in the domain

# Derived parameters
dx = L / Nx          # Spatial step size based on domain length and spatial points
dt = C * dx / c      # Time step size based on Courant condition for stability
save_dir = 'C:\\Users\\pbosn\\OneDrive - USN\\PhD WorkFlow\\WorkFlow\\Courses\\Required Courses\\Numerical Solutions for PDE\\2_Wave_Equations\\Develop\\FullyWorkingSolverAndTest\\plug_wave_images'  # Directory to save images

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ===================================
# Initial Conditions and Source Term
# ===================================

def initial_displacement(x):
    """
    Defines the initial plug profile as a rectangular pulse centered at 'loc'.
    The displacement is 1.0 within a certain range around 'loc' and 0.0 elsewhere.
    """
    return np.where(np.abs(x - loc) <= 0.1, 1.0, 0.0)

def initial_velocity(x):
    """
    Sets the initial velocity across the domain to zero, meaning the wave starts from rest.
    """
    return np.zeros_like(x)

def source_term(x, t):
    """
    No external source term in this example, so it returns zero across the domain.
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
    Stores the current wave state (displacement values u) and time 
    for later use in HTML animation.
    """
    results.append((u.copy(), t[n]))

# Function to save each time step's image
def save_wave_image(u, x, t, n, C, save_dir='wave_images', ymin=-1, ymax=1.5):
    """
    Saves the current wave state as an image for each time step, 
    creating a visual record of the wave propagation.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(8, 4))
    plt.plot(x, u, label=f"Numerical Solution (t = {t[n]:.3f})", color="blue")
    plt.ylim(ymin, ymax)
    plt.xlim(0, L)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f"Plug Profile Wave at t = {t[n]:.3f}, Courant number = {C}")
    plt.legend()
    plt.grid(True)

    filename = os.path.join(save_dir, f'plug_step_{n:04d}.png')
    plt.savefig(filename)
    plt.close()

# Combined user action to capture both results and images
def combined_user_action(u, x, t, n):
    """
    Called at each time step of the solver. It captures results for the HTML animation
    and saves the wave state as an image for GIF creation.
    """
    capture_results(u, x, t, n)  # Store data for HTML animation
    save_wave_image(u, x, t, n, C, save_dir=save_dir, ymin=-1, ymax=1.5)  # Save image

# ===================================
# Solver Execution
# ===================================

# Run the wave equation solver with the initial conditions, source term, and Neumann boundary conditions
solve_wave_equation(
    I=initial_displacement,          # Initial displacement function
    V=initial_velocity,              # Initial velocity function (zero)
    f=source_term,                   # Source term function (zero)
    c=c,                             # Wave speed
    L=L,                             # Length of the domain
    dt=dt,                           # Time step size
    C=C,                             # Courant number
    T=T,                             # Total simulation time
    user_action=combined_user_action,  # User action for capturing and saving results
    version='scalar',                # Choose solver version: 'scalar', 'vectorized', or 'vectorized2'
    boundary='Neumann'               # Boundary condition type (Neumann for free ends)
)

# ===================================
# Animation Creation
# ===================================

# Generate a GIF from the saved images to visualize the wave propagation over time
generate_gif_from_images(image_folder=save_dir, gif_name='plug_wave_animation.gif', duration=0.1)

# Generate HTML animation from captured results
x = np.linspace(0, L, Nx + 1)  # Spatial grid for plotting
generate_html_animation(x, results, save_dir, filename="plug_wave_animation.html", ymin=-1, ymax=1.5, fps=10)
