from Diffusion_Equation_Solver_dev import theta_diffusion_2D, diffusion_PDE_solver, theta_diffusion_solver, save_diffusion_images_basic, save_diffusion_images, visualize_2D_diffusion_on_2D_and_3D_plots, generate_gif_from_images, generate_html_animation
import numpy as np
from math import exp, sin, pi
import time
import os
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, HTMLWriter
from numpy import linspace, zeros, linalg
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve

# ------------------------------------------------------------
#               Numerical experiments: solver 1 (1D)
# ------------------------------------------------------------

def plug(method, version):
    """
    Solve the diffusion equation with a plug profile initial condition.
    """
    # ------------------------------
    # Parameters for the simulation
    # ------------------------------
    L = 1.0               # Length of the spatial domain
    a = 1.0               # Diffusion coefficient
    T = 0.1               # Total simulation time
    F = 0.5               # Fourier number (a*dt/dx^2)
    Nx = 50               # Number of spatial grid points
    dx = L / Nx           # Spatial step size
    dt = F * dx**2 / a    # Time step size

    # ------------------------------
    # Define the initial condition
    # ------------------------------
    def I(x):
        """
        Plug profile: Value is 1 for |x - L/2| <= 0.1, else 0.
        """
        return 1 if abs(x - L / 2.0) <= 0.1 else 0

    def f(x, t):
        """
        Defines the source term of the wave equation as zero, implying there is no
        external force acting on the wave.
        """
        return np.zeros_like(x)
    # ------------------------------
    # Directories and settings
    # ------------------------------
    save_dir = r'C:\Users\pbosn\OneDrive - USN\PhD WorkFlow\WorkFlow\Courses\Required Courses\Numerical Solutions for PDE\3_Diffusion_Equations\DevelopingCode\plug_output'
    results = []  # Store solutions at each time step
    title_comment = "The diffusion equation (Plug Profile)"


    # ------------------------------
    # Define user action function
    # ------------------------------
    def combined_user_action(u, x, t, n):
        """
        User action: Collect results and save plots at the final step.
        """
        results.append((u.copy(), t[n]))
        if n == len(t) - 1:  # After the final time step
            save_diffusion_images(
                results,
                x,
                save_dir=save_dir,
                title_comment=title_comment,
                method=method,
                version=version,
                F=F,
                # Define y label size!
                ymin=-0.2,
                ymax=1.2
            )

    # ------------------------------
    # Run the solver
    # ------------------------------
    print(f"Running: {title_comment} using {method} ({version})...")
    u, x, t, cpu_time = diffusion_PDE_solver(
        I=I,
        f=f,
        a=a,
        L=L,
        dt=dt,
        F=F,
        T=T,
        method=method,
        version=version,
        user_action=combined_user_action
    )
    print("Finished simulation!")

    # ------------------------------
    # Post-processing: Generate animations
    # ------------------------------
    print("Generating animations!")
    generate_gif_from_images(save_dir, gif_name="plug_animation.gif")
    generate_html_animation(x, results, save_dir, filename="plug_movie.html")
    print(f"CPU time for plug: {cpu_time:.5f} seconds")

def gaussian(method, version):
    """
    Solve the diffusion equation with a Gaussian initial condition.
    """
    # ------------------------------
    # Parameters for the simulation
    # ------------------------------
    L = 1.0               # Length of the spatial domain
    a = 1.0               # Diffusion coefficient
    T = 0.1               # Total simulation time
    F = 0.5               # Fourier number (a*dt/dx^2)
    Nx = 50               # Number of spatial grid points
    sigma = 0.05          # Width of the Gaussian profile
    dx = L / Nx           # Spatial step size
    dt = F * dx**2 / a    # Time step size

    # ------------------------------
    # Define the initial condition
    # ------------------------------
    def I(x):
        """
        Gaussian profile centered at L/2 with width sigma.
        """
        return exp(-0.5 * ((x - L / 2.0) ** 2) / sigma**2)

    def f(x, t):
        """
        Defines the source term of the wave equation as zero, implying there is no
        external force acting on the wave.
        """
        return np.zeros_like(x)
    
    # ------------------------------
    # Directories and settings
    # ------------------------------
    save_dir = r'C:\Users\pbosn\OneDrive - USN\PhD WorkFlow\WorkFlow\Courses\Required Courses\Numerical Solutions for PDE\3_Diffusion_Equations\DevelopingCode\gaussian_output'
    results = []  # Store solutions at each time step
    title_comment = "The diffusion equation (Gaussian Profile)"
    

    # ------------------------------
    # Define user action function
    # ------------------------------
    def combined_user_action(u, x, t, n):
        """
        User action: Collect results and save plots at the final step.
        """
        results.append((u.copy(), t[n]))
        if n == len(t) - 1:  # After the final time step
            save_diffusion_images(
                results,
                x,
                save_dir=save_dir,
                title_comment=title_comment,
                method=method,
                version=version,
                F=F,
                # Define y label size!
                ymin=-1.0,
                ymax=1.0
            )

    # ------------------------------
    # Run the solver
    # ------------------------------
    print(f"Running: {title_comment} using {method} ({version})...")
    u, x, t, cpu_time = diffusion_PDE_solver(
        I=I,
        f=f,
        a=a,
        L=L,
        dt=dt,
        F=F,
        T=T,
        method=method,
        version=version,
        user_action=combined_user_action
    )
    print("Finished simulation!")

    # ------------------------------
    # Post-processing: Generate animations
    # ------------------------------
    print("Generating animations!")
    generate_gif_from_images(save_dir, gif_name="gaussian_animation.gif")
    generate_html_animation(x, results, save_dir, filename="gaussian_movie.html")
    print(f"CPU time for Gaussian: {cpu_time:.5f} seconds")

def expsin(method, version):
    """
    Solve the diffusion equation with an exponential sine wave initial condition.
    """
    # ------------------------------
    # Parameters for the simulation
    # ------------------------------
    L = 10.0              # Length of the spatial domain
    a = 1.0               # Diffusion coefficient
    T = 1.2               # Total simulation time
    F = 0.5               # Fourier number (a*dt/dx^2)
    m = 3                 # Mode number of the sine wave
    Nx = 80               # Number of spatial grid points
    dx = L / Nx           # Spatial step size
    dt = F * dx**2 / a    # Time step size

    # ------------------------------
    # Define the exact solution and initial condition
    # ------------------------------
    def exact(x, t):
        """
        Exact solution for exponential decay of a sine wave.
        """
        return exp(-m**2 * np.pi**2 * a / L**2 * t) * sin(m * pi / L * x)

    def I(x):
        """
        Initial condition: Sine wave at t = 0.
        """
        return exact(x, 0)

    def f(x, t):
        """
        Defines the source term of the wave equation as zero, implying there is no
        external force acting on the wave.
        """
        return np.zeros_like(x)
    
    # ------------------------------
    # Directories and settings
    # ------------------------------
    save_dir = r'C:\Users\pbosn\OneDrive - USN\PhD WorkFlow\WorkFlow\Courses\Required Courses\Numerical Solutions for PDE\3_Diffusion_Equations\DevelopingCode\test_expsin_output'
    results = []  # Store solutions at each time step
    title_comment = "The diffusion equation (Exponential Sine Wave)"
    

    # ------------------------------
    # Define user action function
    # ------------------------------
    def combined_user_action(u, x, t, n):
        """
        User action: Collect results and save plots at the final step.
        """
        results.append((u.copy(), t[n]))
        if n == len(t) - 1:  # After the final time step
            save_diffusion_images(
                results,
                x,
                save_dir=save_dir,
                title_comment=title_comment,
                method=method,
                version=version,
                F=F,
                # Define y label size!
                ymin=-1.0,
                ymax=1.0
            )

    # ------------------------------
    # Run the solver
    # ------------------------------
    print(f"Running: {title_comment} using {method} ({version})...")
    u, x, t, cpu_time = diffusion_PDE_solver(
        I=I,
        f=f,
        a=a,
        L=L,
        dt=dt,
        F=F,
        T=T,
        method=method,
        version=version,
        user_action=combined_user_action
    )
    print("Finished simulation!")

    # ------------------------------
    # Post-processing: Generate animations
    # ------------------------------
    print("Generating animations!")
    generate_gif_from_images(save_dir, gif_name="expsin_animation.gif")
    generate_html_animation(x, results, save_dir, filename="expsin_movie.html")
    print(f"CPU time for exponential sine: {cpu_time:.5f} seconds")

# ------------------------------------------------------------
#               Numerical experiments: solver 2 (1D)
# ------------------------------------------------------------

def plug_theta(theta):
    """
    Solve the diffusion equation with a plug profile initial condition.
    """
    # ------------------------------
    # Parameters for the simulation
    # ------------------------------
    L = 1.0               # Length of the spatial domain
    a = 1.0               # Diffusion coefficient
    T = 0.1               # Total simulation time
    F = 3               # Fourier number (a*dt/dx^2)
    Nx = 50               # Number of spatial grid points
    dx = L / Nx           # Spatial step size
    dt = F * dx**2 / a    # Time step size

    # ------------------------------
    # Define the initial condition
    # ------------------------------
    def I(x):
        """
        Plug profile: Value is 1 for |x - L/2| <= 0.1, else 0.
        """
        return 1 if abs(x - L / 2.0) <= 0.1 else 0

    def f(x, t):
        """
        Defines the source term of the wave equation as zero, implying there is no
        external force acting on the wave.
        """
        return np.zeros_like(x)
    # ------------------------------
    # Directories and settings
    # ------------------------------
    save_dir = f'C:/Users/pbosn/OneDrive - USN/PhD WorkFlow/WorkFlow/Courses/Required Courses/Numerical Solutions for PDE/3_Diffusion_Equations/DevelopingCode/plug_theta_{theta}_output'
    results = []  # Store solutions at each time step
    title_comment = f"The Unified Theta {theta} (Plug Profile)"


    # ------------------------------
    # Define user action function
    # ------------------------------
    def combined_user_action(u, x, t, n):
        """
        User action: Collect results and save plots at the final step.
        """
        results.append((u.copy(), t[n]))
        if n == len(t) - 1:  # After the final time step
            save_diffusion_images_basic(
                results,
                x,
                save_dir=save_dir,
                # Define y label size!
                ymin=-0.2,
                ymax=1.2
            )

    # ------------------------------
    # Run the solver
    # ------------------------------
    print(f"Running: {title_comment} using The Unifying Theta Rule. Theta:{theta}...")
    u, x, t, cpu_time = theta_diffusion_solver(
        I=I,
        f=f,
        a=a,
        L=L,
        dt=dt,
        F=F,
        T=T,
        theta=theta,
        user_action=combined_user_action
    )
    print("Finished simulation!")
    print(f"CPU simulation time for plug theta={theta}: {cpu_time:.5f} seconds")
    # ------------------------------
    # Post-processing: Generate animations
    # ------------------------------
    print("Generating animations!")
    generate_gif_from_images(save_dir, gif_name="plug_animation.gif")
    generate_html_animation(x, results, save_dir, filename="plug_movie.html")
    print("Done!")
    
def plug_theta_all():
    """
    Solve the diffusion equation with a plug profile initial condition for θ = 0, 0.5, 1.
    Generates and saves images with all three θ values plotted at each time step.
    """
    # ------------------------------
    # Parameters for the simulation
    # ------------------------------
    L = 1.0               # Length of the spatial domain
    a = 1.0               # Diffusion coefficient
    T = 0.1               # Total simulation time
    F = 0.5               # Fourier number (a*dt/dx^2)
    Nx = 50               # Number of spatial grid points
    dx = L / Nx           # Spatial step size
    dt = F * dx**2 / a    # Time step size
    thetas = [0, 0.5, 1]  # Forward Euler, Crank-Nicolson, Backward Euler
    labels = ['Forward Euler (θ=0)', 'Crank-Nicolson (θ=0.5)', 'Backward Euler (θ=1)']
    colors = ['blue', 'green', 'red']

    # ------------------------------
    # Define the initial condition
    # ------------------------------
    def I(x):
        """
        Plug profile: Value is 1 for |x - L/2| <= 0.1, else 0.
        """
        return 1 if abs(x - L / 2.0) <= 0.1 else 0

    def f(x, t):
        """
        Source term (zero for this example).
        """
        return np.zeros_like(x)

    def theta_step_matrices(u, f, a, L, dt, F, theta, x, t_next):
        """
        Generate the matrix A and right-hand side b for the θ-rule.
        Parameters:
        - u: Solution at the current time step.
        - f: Source term function.
        - a: Diffusion coefficient.
        - L: Length of spatial domain.
        - dt: Time step size.
        - F: Fourier number (a*dt/dx^2).
        - theta: Weighting factor for the θ-rule.
        - x: Spatial points.
        - t_next: Next time step.
        Returns:
        - A: Sparse matrix for the system.
        - b: Right-hand side vector.
        """
        Nx = len(u) - 1  # Number of spatial intervals
        diagonal = np.ones(Nx + 1) * (1 + 2 * theta * F)
        lower = np.ones(Nx) * (-theta * F)
        upper = np.ones(Nx) * (-theta * F)

        # Adjust boundary conditions
        diagonal[0] = diagonal[-1] = 1  # Dirichlet boundary conditions
        lower[-1] = 0
        upper[0] = 0

        # Ensure arrays match the expected size
        lower = np.append(lower, 0)  # Match matrix dimensions
        upper = np.append(0, upper)  # Match matrix dimensions

        # Create sparse matrix
        A = spdiags([lower, diagonal, upper], [-1, 0, 1], Nx + 1, Nx + 1).tocsc()

        # Compute the right-hand side
        b = u + (1 - theta) * F * (np.roll(u, -1) - 2 * u + np.roll(u, 1))
        b[0] = b[-1] = 0  # Dirichlet boundary conditions
        b += dt * ((1 - theta) * f(x, t_next) + theta * f(x, t_next))

        return A, b

    # ------------------------------
    # Directories and settings
    # ------------------------------
    import os
    save_dir = f'C:/Users/pbosn/OneDrive - USN/PhD WorkFlow/WorkFlow/Courses/Required Courses/Numerical Solutions for PDE/3_Diffusion_Equations/DevelopingCode/plug_theta_ALL_output'
    os.makedirs(save_dir, exist_ok=True)  # Create output folder if it doesn't exist
    
    # ------------------------------
    # Initialize storage for solutions
    # ------------------------------
    results = {theta: [] for theta in thetas}  # Store results for each θ
    solutions = {theta: np.zeros(Nx + 1) for theta in thetas}  # Current solutions for each θ

    # ------------------------------
    # Time-stepping loop
    # ------------------------------
    Nt = int(round(T / dt))  # Number of time steps
    x = np.linspace(0, L, Nx + 1)  # Spatial points
    t = np.linspace(0, T, Nt + 1)  # Time points

    # Set initial conditions for all θ
    for theta in thetas:
        solutions[theta][:] = [I(xi) for xi in x]  # Apply initial condition

    # Time-stepping loop
    for n in range(Nt):
        # Calculate next step for all θ values
        for theta in thetas:
            A, b = theta_step_matrices(solutions[theta], f, a, L, dt, F, theta, x, t[n + 1])
            solutions[theta][:] = spsolve(A, b)  # Solve the linear system
            results[theta].append((solutions[theta].copy(), t[n + 1]))

        # Plot all θ values for this step
        plt.figure(figsize=(10, 6))
        for i, theta in enumerate(thetas):
            plt.plot(x, solutions[theta], label=f"{labels[i]} (t={t[n + 1]:.3f})", color=colors[i])
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.title(f"Solution of the Diffusion Equation at t={t[n + 1]:.3f}")
        plt.legend()
        plt.grid()
        plt.xlim(x[0], x[-1])
        # Define y label size!
        ymin=-0.2
        ymax=1.2
        plt.ylim(ymin, ymax)

        # Save plot
        file_path = os.path.join(save_dir, f"theta_comparison_step_{n + 1:04d}.png")
        plt.savefig(file_path)
        plt.close()

    print("Finished simulation!")
      
    # ------------------------------
    # Post-processing: Generate animations
    # ------------------------------
    print("Generating animations!")
    generate_gif_from_images(save_dir, gif_name="plug_animation.gif")
    print("Done!")

    return results

# ------------------------------------------------------------
#               Numerical experiments: solver 3 (3D)
# ------------------------------------------------------------

def twoD_diffusion_solver_num_experiment(mode="both", z_min=0, z_max=1):
    """
    Test the theta diffusion 2D solver and save 2D and/or 3D visualizations for every time step.
    """
    import numpy as np

    # Problem setup
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 50, 50
    T = 0.1
    alpha = 1.0
    theta = 0.5
    dt = 0.5 * min((Lx / Nx)**2, (Ly / Ny)**2) / alpha  # Ensuring stability

    save_dir = f'C:/Users/pbosn/OneDrive - USN/PhD WorkFlow/WorkFlow/Courses/Required Courses/Numerical Solutions for PDE/3_Diffusion_Equations/DevelopingCode/2d_solver_test_{mode}'

    # Define initial condition
    def I(x, y):
        """Initial condition: a peak at the center."""
        return np.exp(-100 * ((x - Lx/2)**2 + (y - Ly/2)**2))

    # Define source term
    def f(x, y, t):
        """Source term: zero everywhere."""
        return np.zeros_like(x)

    # User action to visualize at each time step
    def user_action(u, x, xv, y, yv, t, n):
        visualize_2D_diffusion_on_2D_and_3D_plots(
            u=u,
            x=x,
            y=y,
            time_step=n,
            save_dir=save_dir,
            title=f"2D and 3D Diffusion (theta={theta})",
            mode=mode,
            z_min=z_min,
            z_max=z_max
        )

    # Run the solver
    t, cpu_time = theta_diffusion_2D(
        I=I,
        f=f,
        alpha=alpha,
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        T=T,
        dt=dt,
        theta=theta,
        user_action=user_action
    )

    print(f"Simulation completed. Images saved in: {save_dir}")
    print(f"CPU time: {cpu_time:.5f} seconds")
    # print('Generating GIF animation!')
    # generate_gif_from_images(save_dir, gif_name="2d_diffusion_eq_animation.gif")
    # print('Done!')

if __name__ == "__main__":

    # ------------------------------------------------------------
    #                           1.
    # ------------------------------------------------------------
    #                   Basic 1D diffusion solver 
    # Define: method: 'FE' or 'BE' and version: 'scalar' or 'vectorized' 
    # ------------------------------------------------------------
    #                   Run each simulation (uncomment!)
    # ------------------------------------------------------------

    # plug(method='BE', version='vectorized')
    # gaussian(method='FE', version='vectorized')
    # expsin(method='FE', version='vectorized')

    # ------------------------------------------------------------
    #                           2.
    # ------------------------------------------------------------
    #        1D diffusion solver with the Unified Theta Rule 
    #        Define: # theta: 0 --> FE, 1 --> BE, 0.5 --> CN
    # ------------------------------------------------------------
    #                   Run each simulation (uncomment!)
    # ------------------------------------------------------------
    
    # plug_theta(theta=0.5) # theta: 0 --> FE, 1 --> BE, 0.5 --> CN
    # Generate the plot for θ = 0, 0.5, 1 at once
    # plug_theta_all()

    # ------------------------------------------------------------
    #                           3.
    # ------------------------------------------------------------
    #   2D diffusion equation solver with The Unified Theta Rule 
    #       Define visualization mode: "2D", "3D", or "both"
    # ------------------------------------------------------------
    mode = "both"  # User-selected visualization mode: "2D", "3D", or "both"
    z_min = 0      # Minimum value for z-axis
    z_max = 1.1      # Maximum value for z-axis
    save_dir = f'C:/Users/pbosn/OneDrive - USN/PhD WorkFlow/WorkFlow/Courses/Required Courses/Numerical Solutions for PDE/3_Diffusion_Equations/DevelopingCode/2d_solver_test_{mode}'
    # ------------------------------------------------------------
    #                   Run each simulation (uncomment!)
    # ------------------------------------------------------------
    # twoD_diffusion_solver_num_experiment(mode=mode, z_min=z_min, z_max=z_max)
    # generate_gif_from_images(save_dir, gif_name="2d_diffusion_eq_animation.gif", duration=0.2)
    




   
