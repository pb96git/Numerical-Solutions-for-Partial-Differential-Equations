import numpy as np
import matplotlib.pyplot as plt
import os
from wave_eq_solver_dev import solve_wave_equation, generate_gif_from_images, generate_html_animation

# ===============================
# Simulation Parameters
# ===============================

L = 10.0             # Length of the domain
c = 10               # Wave speed
Nx = 50              # Number of spatial points for resolution in the domain
C = 1                # Courant number (controls stability and time step size)
T = 3                # Total simulation time
sigma = 0.5          # Standard deviation for the Gaussian initial condition
loc = 5              # Location of the Gaussian peak

# ===============================
# Derived Numerical Parameters
# ===============================

# Spatial and temporal discretization
dx = L / Nx          # Spatial step size, derived from the domain length and number of points
dt = C * dx / c      # Time step size, calculated using the Courant number for stability

# Directory to save images for GIF generation and HTML animation
save_dir = 'C:\\Users\\pbosn\\OneDrive - USN\\PhD WorkFlow\\WorkFlow\\Courses\\Required Courses\\Numerical Solutions for PDE\\2_Wave_Equations\\Develop\\FullyWorkingSolverAndTest\\gaussian_wave_images'

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ===============================
# Initial Conditions and Source Term
# ===============================

# Gaussian initial displacement function
def initial_displacement(x):
    """
    Creates an initial Gaussian-shaped displacement centered at 'loc' with
    a spread defined by 'sigma'. This simulates an initial pulse.
    """
    return (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-0.5 * ((x - loc) / sigma) ** 2)

# Initial velocity function (set to zero for a stationary initial pulse)
def initial_velocity(x):
    """
    Sets the initial velocity across the domain to zero. This simulates
    a pulse that starts from rest.
    """
    return np.zeros_like(x)

# Source term function (set to zero since no external forcing is applied)
def source_term(x, t):
    """
    Defines the source term of the wave equation as zero, implying there is no
    external force acting on the wave.
    """
    return np.zeros_like(x)

# ===============================
# Animation Data Storage
# ===============================

# Array to store solution data for HTML animation
results = []

# Function to capture each time step's solution for HTML animation
def capture_results(u, x, t, n):
    """
    Captures the solution at each time step for later animation. This function
    appends the current wave state (u) and the time step to the 'results' array.
    """
    results.append((u.copy(), t[n]))

# Save wave image for GIF generation
def save_wave_image(u, x, t, n, C, save_dir='wave_images', ymin=-0.6, ymax=0.7):
    """
    Saves the current state of the wave at each time step as an image.
    These images are later used to create a GIF showing the wave propagation.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(8, 4))
    plt.plot(x, u, label=f"Numerical Solution (t = {t[n]:.3f})", color="blue")
    plt.ylim(ymin, ymax)
    plt.xlim(0, L)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f"Gaussian Wave Propagation at t = {t[n]:.3f}, Courant number = {C}")
    plt.legend()
    plt.grid(True)
    filename = os.path.join(save_dir, f'gaussian_step_{n:04d}.png')
    plt.savefig(filename)
    plt.close()

# Combined user action that calls both capture_results and save_wave_image
def combined_user_action(u, x, t, n):
    """
    This function is called at each time step of the solver.
    It captures results for HTML animation and saves images for GIF creation.
    """
    capture_results(u, x, t, n)  # Store data for HTML animation
    save_wave_image(u, x, t, n, C, save_dir=save_dir, ymin=-0.6, ymax=0.7)  # Save image for GIF

# ===============================
# Solver Execution
# ===============================

# Run the wave equation solver with the initial conditions, source term, and boundary conditions
solve_wave_equation(
    I=initial_displacement,        # Initial displacement function (Gaussian)
    V=initial_velocity,            # Initial velocity function (zero)
    f=source_term,                 # Source term function (zero)
    c=c,                           # Wave speed
    L=L,                           # Length of the domain
    dt=dt,                         # Time step size
    C=C,                           # Courant number
    T=T,                           # Total simulation time
    user_action=combined_user_action,  # Function to capture data and save images at each step
    version='scalar',              # Choose solver version: 'scalar', 'vectorized', or 'vectorized2'
    boundary='leftFree'            # Boundary condition type: 'Dirichlet', 'Neumann', 'leftFree', or 'rightFree'
)

# ===============================
# Animation Creation
# ===============================

# Generate a GIF from the saved images to visualize the wave propagation over time
generate_gif_from_images(image_folder=save_dir, gif_name='gaussian_wave_animation.gif', duration=0.1)

# Generate HTML animation from captured results
x = np.linspace(0, L, Nx + 1)  # Spatial grid for plotting
generate_html_animation(x, results, save_dir, filename="gaussian_wave_animation.html", ymin=-0.6, ymax=0.7)
