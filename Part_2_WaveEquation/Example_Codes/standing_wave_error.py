import numpy as np
import matplotlib.pyplot as plt
import os
from wave_eq_solver_dev import solve_wave_equation, generate_gif_from_images, generate_html_animation

# ===============================
# Parameters for the Wave Equation
# ===============================

L = 0.75               # Length of the domain
A = 0.005              # Amplitude of the standing wave
m = 1                  # Mode number of the standing wave
c = 2                  # Wave speed
C = 0.5                # Courant number for stability
Nx = 25                # Number of spatial points
dx = L / Nx            # Spatial step size
dt = C * dx / c        # Time step size based on the Courant number
T = 2                  # Total simulation time

save_dir = 'C:\\Users\\pbosn\\OneDrive - USN\\PhD WorkFlow\\WorkFlow\\Courses\\Required Courses\\Numerical Solutions for PDE\\2_Wave_Equations\\Develop\\FullyWorkingSolverAndTest\\ErrorNumBC'

# ===============================
# Exact Solution for Comparison
# ===============================

def exact_solution(x, t):
    """
    Computes the exact solution of the wave equation for a standing wave
    at spatial points x and time t.
    """
    return A * np.sin(np.pi * m * x / L) * np.cos(np.pi * m * c * t / L)

# ===============================
# Initial and Boundary Conditions
# ===============================

def initial_displacement(x):
    """
    Defines the initial displacement of the wave, matching the exact solution.
    """
    return A * np.sin(np.pi * m * x / L)

def initial_velocity(x):
    """
    Defines the initial velocity of the wave (set to zero for a standing wave).
    """
    return np.zeros_like(x)

def source_term(x, t):
    """
    Source term for the wave equation (set to zero for this example).
    """
    return np.zeros_like(x)

# ===============================
# Error Calculation Function
# ===============================

def calculate_error_terminal_out(u_num, x, t, n):
    """
    Calculates the error between the numerical and exact solutions at a given
    time step. Prints the L2 and maximum error norms.
    """
    u_exact = exact_solution(x, t[n])   # Exact solution at time step n
    error_L2 = np.sqrt(np.mean((u_num - u_exact) ** 2))   # L2 norm error
    error_max = np.max(np.abs(u_num - u_exact))           # Maximum error

    print(f"Time step {n}, L2 Error: {error_L2:.5e}, Max Error: {error_max:.5e}")
    
def calculate_error(u_num, u_exact):
    """Calculates L2 and Max errors."""
    error_L2 = np.sqrt(np.mean((u_num - u_exact) ** 2))  # L2 norm error
    error_max = np.max(np.abs(u_num - u_exact))          # Max error
    return error_L2, error_max
# ===============================
# Animation Data Storage
# ===============================

results = []  # Array to store solution data for HTML animation

def capture_results(u, x, t, n):
    """
    Captures the solution at each time step for later animation.
    Appends the current wave state (u) and the time step to the 'results' array.
    """
    results.append((u.copy(), t[n]))

# ===============================
# Visualization Functions
# ===============================

def save_wave_image_with_exact(u_num, x, t, n, C, save_dir='wave_images', ymin=-0.01, ymax=0.01):
    """
    Saves an image at each time step showing both numerical and exact solutions.
    Calculates error norms using `calculate_error` and displays them in the plot title.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    u_exact = exact_solution(x, t[n])  # Exact solution at this time step

    # Calculate error norms using the `calculate_error` function
    error_L2, error_max = calculate_error(u_num, u_exact)

    # Plot numerical and exact solutions
    plt.figure(figsize=(8, 4))
    plt.plot(x, u_num, label="Numerical Solution", linestyle="-", color="blue")
    plt.plot(x, u_exact, label="Exact Solution", linestyle="--", color="red")
    plt.ylim(ymin, ymax)
    plt.xlim(0, L)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    
    # Add Courant number, time, and error norms to the plot title
    plt.title(f"Wave Propagation at t = {t[n]:.3f}, C = {C}, L2 Error = {error_L2:.5e}, Max Error = {error_max:.5e}")
    
    plt.legend()
    plt.grid(True)
    filename = os.path.join(save_dir, f'wave_step_{n:04d}.png')
    plt.savefig(filename)
    plt.close()


# ===============================
# Combined User Action
# ===============================

def combined_user_action(u, x, t, n):
    """
    Called at each time step of the solver to capture data for HTML animation
    and save images for GIF creation.
    """
    capture_results(u, x, t, n)  # Capture data for HTML animation
    save_wave_image_with_exact(u, x, t, n, C, save_dir=save_dir, ymin=-0.01, ymax=0.01)  # Save image for GIF

# ===============================
# Run the Solver
# ===============================

solve_wave_equation(
    I=initial_displacement,
    V=initial_velocity,
    f=source_term,
    c=c,
    L=L,
    dt=dt,
    C=C,
    T=T,
    user_action=combined_user_action,
    version="scalar",  # Choose solver version: 'scalar', 'vectorized', or 'vectorized2'
    boundary='Dirichlet'  # Use Dirichlet or Neumann boundaries
)

# ===============================
# Generate Animation Outputs
# ===============================

# Generate a GIF from the saved images
generate_gif_from_images(image_folder=save_dir, gif_name='wave_comparison_animation.gif', duration=0.1)

