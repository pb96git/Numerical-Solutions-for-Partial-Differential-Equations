import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, HTMLWriter
import os
import imageio
import inspect


"""
------------------------------------------------------------------------------------------------------
                                        Solver
------------------------------------------------------------------------------------------------------
"""
def solve_wave_equation(I, V, f, c, L, dt, C, T, user_action=None, version='scalar', save_dir=None, boundary='Dirichlet'):
    """
    Solve u_tt = c^2 * u_xx + f on (0, L) x (0, T] with boundary conditions.

    Parameters:
    - I: Initial displacement function
    - V: Initial velocity function
    - f: Source term function
    - c: Wave speed
    - L: Domain length
    - dt: Time step size
    - C: Courant number
    - T: Total simulation time
    - user_action: Function called at each time step
    - version: Solver version ('scalar', 'vectorized', or 'vectorized2')
    - save_dir: Directory to save images or output data
    - boundary: Type of boundary condition ('Dirichlet' or 'Neumann' or 'leftFree' or 'rightFree')
    """
    
    if save_dir is None:
        save_dir = os.getcwd()

    Nt = int(round(T / dt))  # Number of time steps
    t = np.linspace(0, Nt * dt, Nt + 1)  # Time mesh points
    dx = dt * c / float(C)  # Spatial step size based on Courant number
    Nx = int(round(L / dx))  # Number of spatial points
    x = np.linspace(0, L, Nx + 1)  # Spatial mesh points
    C2 = C ** 2  # Courant number squared

    # Ensure that f is a function returning an array
    if f is None or f == 0:
        f = lambda x, t: np.zeros_like(x)
    if V is None or V == 0:
        V = lambda x: np.zeros_like(x)

    # Initialize solution arrays
    u = np.zeros(Nx + 1)
    u_1 = np.zeros(Nx + 1)
    u_2 = np.zeros(Nx + 1)

    # Start measuring CPU time
    t0 = time.time()

    # Load initial condition into u_1
    for i in range(Nx + 1):
        u_1[i] = I(x[i])

    if user_action is not None:
        user_action_args = inspect.signature(user_action).parameters
        if len(user_action_args) == 4:
            user_action(u_1, x, t, 0)  # Call without save_dir
        else:
            user_action(u_1, x, t, 0, save_dir)  # Call with save_dir

    # Special formula for the first time step
    n = 0
    for i in range(1, Nx):
        u[i] = u_1[i] + dt * V(x[i]) + \
               0.5 * C2 * (u_1[i - 1] - 2 * u_1[i] + u_1[i + 1]) + \
               0.5 * dt ** 2 * f(x[i], t[n])

    # Apply boundary conditions for the first time step
    if boundary == 'Dirichlet':
        u[0] = 0
        u[Nx] = 0
    elif boundary == 'Neumann':
        u[0] = u[1]  # Zero-gradient at left boundary
        u[Nx] = u[Nx - 1]  # Zero-gradient at right boundary
    elif boundary == 'leftFree':
            u[0] = u[1]  # Zero-gradient at left boundary
            u[Nx] = 0  # Zero-gradient at right boundary
    elif boundary == 'rightFree':
            u[0] = 0  # Zero-gradient at left boundary
            u[Nx] = u[Nx - 1]  # Zero-gradient at right boundary

    if user_action is not None:
        if len(user_action_args) == 4:
            user_action(u, x, t, 1)
        else:
            user_action(u, x, t, 1, save_dir)

    u_2[:] = u_1
    u_1[:] = u

    # Time-stepping loop
    for n in range(1, Nt):
        if version == 'scalar':
            # Scalar loop implementation
            for i in range(1, Nx):
                u[i] = -u_2[i] + 2 * u_1[i] + \
                       C2 * (u_1[i - 1] - 2 * u_1[i] + u_1[i + 1]) + \
                       dt ** 2 * f(x[i], t[n])

        elif version == 'vectorized':
            # Vectorized implementation using slice style (1:-1)
            f_a = f(x, t[n])  # Precompute in array
            u[1:-1] = -u_2[1:-1] + 2 * u_1[1:-1] + \
                      C2 * (u_1[0:-2] - 2 * u_1[1:-1] + u_1[2:]) + \
                      dt ** 2 * f_a[1:-1]

        elif version == 'vectorized2':
            # Vectorized implementation using (1:Nx) slice style
            f_a = f(x, t[n])  # Precompute in array
            u[1:Nx] = -u_2[1:Nx] + 2 * u_1[1:Nx] + \
                      C2 * (u_1[0:Nx - 1] - 2 * u_1[1:Nx] + u_1[2:Nx + 1]) + \
                      dt ** 2 * f_a[1:Nx]

        # Apply boundary conditions at each time step
        if boundary == 'Dirichlet':
            u[0] = 0
            u[Nx] = 0
        elif boundary == 'Neumann':
            u[0] = u[1]  # Zero-gradient at left boundary
            u[Nx] = u[Nx - 1]  # Zero-gradient at right boundary
        elif boundary == 'leftFree':
            u[0] = u[1]  # Zero-gradient at left boundary
            u[Nx] = 0  # Zero-gradient at right boundary
        elif boundary == 'rightFree':
            u[0] = 0  # Zero-gradient at left boundary
            u[Nx] = u[Nx - 1]  # Zero-gradient at right boundary

        if user_action is not None:
            if len(user_action_args) == 4:
                user_action(u, x, t, n + 1)
            else:
                user_action(u, x, t, n + 1, save_dir)

        # Update previous solutions
        u_2[:], u_1[:] = u_1, u  

    cpu_time = time.time() - t0  # Measure the CPU time
    print(f"CPU time (s) of {version} solver: {cpu_time:.5f}")
    return u, x, t, cpu_time



def solve_wave_equation_variable_velocity(I, V, f, q, L, dt, C, T, user_action=None, version='scalar', save_dir=None, boundary='Dirichlet'):
    """
    Solve u_tt = (q(x) * u_x)_x + f with variable wave velocity q(x) = c(x)^2.
    
    Parameters:
    - I: Initial displacement function.
    - V: Initial velocity function.
    - f: Source term function.
    - q: Spatially varying wave speed squared (q(x) = c(x)^2).
    - L: Domain length.
    - dt: Time step size.
    - C: Courant number.
    - T: Total simulation time.
    - user_action: Optional function for post-processing at each time step.
    - version: Solver version ('scalar' or other optimized options).
    - save_dir: Directory to save output files.
    - boundary: Boundary condition type ('Dirichlet' or 'Neumann').
    
    Returns:
    - u, x, t arrays for the final solution and computational grid.
    """
    
    if save_dir is None:
        save_dir = os.getcwd()

    Nt = int(round(T / dt))  # Number of time steps
    t = np.linspace(0, Nt * dt, Nt + 1)  # Time mesh points
    dx = L / (len(q) - 1)  # Assuming q is defined on the spatial grid
    x = np.linspace(0, L, len(q))  # Spatial mesh points

    # Initialize solution arrays
    u = np.zeros(len(q))
    u_1 = np.zeros(len(q))
    u_2 = np.zeros(len(q))

    # Initialize with I(x)
    for i in range(len(q)):
        u_1[i] = I(x[i])

    if user_action:
        user_action(u_1, x, t, 0)

    # First time step using initial velocity V(x)
    for i in range(1, len(q) - 1):
        avg_q_right = (q[i] + q[i + 1]) / 2
        avg_q_left = (q[i] + q[i - 1]) / 2
        u[i] = (u_1[i] + dt * V(x[i]) +
                0.5 * dt**2 * (avg_q_right * (u_1[i + 1] - u_1[i]) / dx**2 - avg_q_left * (u_1[i] - u_1[i - 1]) / dx**2) +
                0.5 * dt**2 * f(x[i], t[0]))

    # Main time-stepping loop
    for n in range(1, Nt):
        for i in range(1, len(q) - 1):
            # Averaged q values to reduce artificial amplification
            avg_q_right = (q[i] + q[i + 1]) / 2
            avg_q_left = (q[i] + q[i - 1]) / 2
            u[i] = (2 * u_1[i] - u_2[i] +
                    dt**2 * (avg_q_right * (u_1[i + 1] - u_1[i]) - avg_q_left * (u_1[i] - u_1[i - 1])) / dx**2 +
                    dt**2 * f(x[i], t[n]))

        # Apply boundary conditions
        if boundary == 'Dirichlet':
            u[0] = u[-1] = 0
        elif boundary == 'Neumann':
            u[0] = u[1]
            u[-1] = u[-2]

        if user_action:
            user_action(u, x, t, n)

        # Update previous solutions
        u_2[:], u_1[:] = u_1, u  

    return u, x, t






"""
------------------------------------------------------------------------------------------------------
                                            Visualization
------------------------------------------------------------------------------------------------------
"""

def save_wave_image(u, x, t, n, C, save_dir='wave_images', ymin=None, ymax=None):
    """Save the wave at each time step as an image, with Courant number in the title."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(8, 4))
    plt.plot(x, u, label=f"t = {t[n]:.5f}")
    
    # Set y-axis limits
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
    else:
        plt.ylim(min(u), max(u))  # Dynamic ylim if no limits are provided
    
    plt.xlim(0, max(x))
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    
    # Include Courant number in the plot title
    plt.title(f"Wave propagation at time t = {t[n]:.5f}, Courant number = {C}")
    
    plt.legend()
    plt.grid(True)
    filename = os.path.join(save_dir, f'wave_step_{n:04d}.png')
    
    # Debug message to confirm function call
    print(f"Saving image: {filename}")
    
    plt.savefig(filename)
    plt.close()

def generate_gif_from_images(image_folder='wave_images', gif_name='wave_animation.gif', duration=0.1):
    """Generate a GIF from the images stored in a folder."""
    images = []
    image_files = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

    if not image_files:
        print("Warning: No images found in the specified folder to create a GIF. Ensure that images are saved correctly during the wave equation simulation.")
        return

    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        images.append(imageio.imread(image_path))

    gif_path = os.path.join(image_folder, gif_name)
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF saved as {gif_path}")

def generate_html_animation(x, results, save_dir, filename="movie.html", ymin=-0.005, ymax=0.005, fps=10):
    """
    Generate an HTML animation from stored results for the guitar string wave.
    
    Parameters:
    - x: Spatial grid points array.
    - results: List of tuples (u, t) capturing wave amplitude and time at each step.
    - save_dir: Directory to save the HTML animation file.
    - ymin, ymax: y-axis limits for the animation plot.
    - fps: Frames per second for the animation.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, x[-1])  # Set x-axis based on spatial grid
    ax.set_ylim(ymin, ymax)  # Set y-axis based on provided limits
    line, = ax.plot([], [], lw=2, color="blue")

    def init():
        line.set_data([], [])
        return line,

    def animate(n):
        u, current_time = results[n]
        line.set_data(x, u)
        ax.set_title(f"Wave Solution at t = {current_time:.3f}")
        return line,

    ani = FuncAnimation(fig, animate, init_func=init, frames=len(results), blit=True)
    html_path = os.path.join(save_dir, filename)
    ani.save(html_path, writer=HTMLWriter())
    print(f"HTML animation saved as {html_path}")
    
"""
------------------------------------------------------------------------------------------------------
                                    Test verification
------------------------------------------------------------------------------------------------------
"""
def test_quadratic():
    """
    Check that the scalar and vectorized versions work for
    a quadratic solution u(x,t) = x(L-x)(1+t/2) that is exactly reproduced.
    
    The test compares the computed numerical solution with the exact solution
    and asserts that the difference is within a small tolerance.
    """

    # Parameters for the test
    L = 2.5
    c = 1.5
    C = 0.75
    Nx = 6  # Very coarse mesh for this exact test
    dt = C * (L / Nx) / c
    T = 18

    # Exact solution for u(x, t) = x(L - x)(1 + 0.5*t)
    u_exact = lambda x, t: x * (L - x) * (1 + 0.5 * t)
    I = lambda x: u_exact(x, 0)  # Initial displacement
    V = lambda x: 0.5 * u_exact(x, 0)  # Initial velocity
    f = lambda x, t: np.zeros_like(x) + 2 * c ** 2 * (1 + 0.5 * t)  # Source term

    # Assert function to check solution accuracy
    def assert_no_error(u, x, t, n):
        u_e = u_exact(x, t[n])  # Exact solution at time step n
        tol = 1E-13  # Tolerance for error comparison
        diff = np.abs(u - u_e).max()  # Max difference between computed and exact
        print(diff)
        
        assert diff < tol, f"Difference {diff} exceeds tolerance at step {n}"

    # Test all solver versions
    for version in ['scalar', 'vectorized', 'vectorized2']:
        print(f"Testing {version} solver...")

        solve_wave_equation(I, V, f, c, L, dt, C, T, user_action=assert_no_error, version=version)

    print("All tests passed successfully!")

def test_convrate_sincos():
    # Define parameters
    n = m = 2
    L = 1.0
    
    # Define the exact solution as a lambda function
    u_exact = lambda x, t: np.cos(m * np.pi / L * t) * np.sin(m * np.pi / L * x)
    
    # Call convergence_rates function with the exact solution and appropriate parameters
    rates = convergence_rates(
        u_exact=u_exact,
        I=lambda x: u_exact(x, 0),   # Initial displacement
        V=lambda x: 0,               # Initial velocity
        f=0,                         # Source term, no external force
        c=1,                         # Wave speed
        L=L,                         # Length of the domain
        dt0=0.1,                     # Initial time step size
        num_meshes=6,                # Number of grid refinements
        C=0.9,                       # Courant number
        T=1                          # Total simulation time
    )
    
    # Output the computed convergence rates, rounded for readability
    print('Rates for sin(x) * cos(t) solution:', [round(r_, 2) for r_ in rates])
    
    # Assert that the last computed rate is close to 2, indicating second-order accuracy
    assert abs(rates[-1] - 2) < 0.002, f"Expected rate ~2, but got {rates[-1]}"
    
"""
------------------------------------------------------------------------------------------------------
                                    Convergence of solution
------------------------------------------------------------------------------------------------------
"""

def calculate_convergence_rate_no_exact_solution(L, c, C, T, solver_choice, initial_displacement, initial_velocity, source_term):
    """
    Estimate the convergence rate without knowing the exact solution.
    This is done by comparing solutions on progressively finer grids.

    Parameters:
    - L: Length of the domain
    - c: Wave speed
    - C: Courant number
    - T: Total simulation time
    - solver_choice: Which solver to use ('scalar', 'vectorized', 'vectorized2')
    - initial_displacement: Function defining the initial displacement
    - initial_velocity: Function defining the initial velocity
    - source_term: Function defining the source term
    """

    def compute_error(u_coarse, u_fine, x_coarse, x_fine):
        """
        Compute the difference (error) between the solutions on two grids.
        Assumes u_fine has a finer grid resolution than u_coarse.
        """
        # Interpolate coarse solution to match fine grid
        u_coarse_interp = np.interp(x_fine, x_coarse, u_coarse)
        # Compute L2 norm of the difference
        return np.sqrt(np.sum((u_fine - u_coarse_interp) ** 2) / len(u_fine))

    # Grid resolutions (Nx1 < Nx2 < Nx3)
    Nx_values = [10, 20, 40]  # Coarse, medium, and fine grids
    dt_values = [C * (L / Nx) / c for Nx in Nx_values]  # Corresponding time steps

    # Solve the wave equation on each grid resolution
    solutions = []
    grids = []
    for Nx, dt in zip(Nx_values, dt_values):
        x = np.linspace(0, L, Nx + 1)
        u_num, _, _, _ = solve_wave_equation(initial_displacement, initial_velocity, source_term, c, L, dt, C, T, user_action=None, version=solver_choice)
        solutions.append(u_num)
        grids.append(x)

    # Coarse, medium, and fine grids
    u_coarse, u_medium, u_fine = solutions
    x_coarse, x_medium, x_fine = grids

    # Compute errors between successive grid resolutions
    error_12 = compute_error(u_coarse, u_medium, x_coarse, x_medium)  # Error between coarse and medium
    error_23 = compute_error(u_medium, u_fine, x_medium, x_fine)      # Error between medium and fine

    # Calculate the convergence rate
    h1, h2 = L / Nx_values[0], L / Nx_values[1]
    p = np.log(error_12 / error_23) / np.log(h1 / h2)

    print(f"Convergence rate (without exact solution): {p:.2f}")
    
def convergence_rates(u_exact, I, V, f, c, L, dt0, num_meshes, C, T, solver_choice='scalar'):
    """
    Half the time step and estimate convergence rates for num_meshes simulations.
    """
    global error
    error = 0  # error computed in the user action function

    def compute_error(u, x, t, n):
        global error
        if n == 0:
            error = 0
        else:
            # Calculate max error for the current time step
            current_error = np.abs(u - u_exact(x, t[n])).max()
            error = max(error, current_error)

    # Initialize lists to store error and step size values
    E = []
    h = []

    # Starting time step and corresponding spatial step
    dt = dt0
    dx = dt * c / C  # Based on Courant number and wave speed

    for i in range(num_meshes):
        # Run the solver with the current resolution
        solve_wave_equation(I, V, f, c, L, dt, C, T, user_action=compute_error, version=solver_choice)
        
        # Store the computed error and step size
        E.append(error)
        h.append(dt)

        # Debugging output to trace errors and steps
        print(f"Mesh {i + 1}: dt = {dt:.5e}, error = {error:.5e}")

        # Halve the time step for the next iteration
        dt /= 2
        dx = dt * c / C

    # Calculate convergence rates
    rates = []
    for i in range(1, num_meshes):
        if E[i] > 0 and E[i - 1] > 0:
            rate = np.log(E[i] / E[i - 1]) / np.log(h[i] / h[i - 1])
        else:
            rate = float('inf')  # Avoid division by zero

        rates.append(rate)
        
        print(f"Convergence rates {i + 1}: r = {rate:.5e}, error = {E[i]:.5e}")

    return rates



# Run the test function
# test_quadratic()
# test_convrate_sincos()
