import numpy as np
import time
import os
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, HTMLWriter
from mpl_toolkits.mplot3d import Axes3D
from numpy import linspace, zeros, linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve



"""
------------------------------------------------------------------------------------------------------
                                              Solvers
------------------------------------------------------------------------------------------------------
"""

def diffusion_PDE_solver(I, f, a, L, dt, F, T, method='FE', version='scalar', user_action=None):
    """
    Unified solver for the diffusion equation with a source term.

    Solves:
        u_t = a * u_xx + f(x, t)
    Parameters:
    - I: Initial condition function.
    - f: Source term function f(x, t).
    - a: Diffusion coefficient.
    - L: Length of spatial domain.
    - F: Fourier number (a*dt/dx**2).
    - T: Total simulation time.
    - method: Solver method ('FE' for Forward Euler, 'BE' for Backward Euler).
    - version: Solver version ('scalar' or 'vectorized').
    - user_action: Optional function for post-processing/testing/ect. at each time step.
    """

    # Start measuring CPU time
    t0 = time.time()

    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
    dx = np.sqrt(a*dt/F)
    Nx = int(round(L/dx)) # Number of spatial grid points.
    x = np.linspace(0, L, Nx+1)       # Mesh points in space
    
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u   = np.zeros(Nx+1)
    u_n = np.zeros(Nx+1)

    # Set initial condition
    for i in range(Nx + 1):
        u_n[i] = I(x[i])

    if user_action is not None:
        user_action(u_n, x, t, 0)

    if method == 'FE':  # Forward Euler
        for n in range(Nt):
            if version == 'scalar':
                """
                Simplest expression of the computational algorithm
                using the Forward Euler method and explicit Python loops.
                For this method F <= 0.5 for stability.
                """
                for i in range(1, Nx):
                    u[i] = u_n[i] + F * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + dt * f(x[i], t[n])
            
            elif version == 'vectorized':
                u[1:Nx] = u_n[1:Nx] + F * (u_n[:-2] - 2 * u_n[1:-1] + u_n[2:]) + dt * f(x[1:Nx], t[n])
            else:
                raise ValueError("Invalid version: use 'scalar' or 'vectorized'.")

            u[0] = u[Nx] = 0  # Dirichlet BCs

            if user_action is not None:
                user_action(u, x, t, n + 1)

            u_n, u = u, u_n

    elif method == 'BE':  # Backward Euler
       
        if version == 'scalar':
            """
            Simplest expression of the computational algorithm
            for the Backward Euler method, using explicit Python loops
            and a dense matrix format for the coefficient matrix.
            """
            # Construct dense matrix A
            A = np.zeros((Nx + 1, Nx + 1))
            b = np.zeros(Nx + 1)
            
            for i in range(1, Nx):
                A[i, i - 1] = -F
                A[i, i + 1] = -F
                A[i, i] = 1 + 2 * F
            A[0, 0] = A[Nx, Nx] = 1  # Dirichlet boundary conditions

            for n in range(Nt):
                # Compute right-hand side b
                for i in range(1, Nx):
                    b[i] = u_n[i] + dt * f(x[i], t[n + 1])
                b[0] = b[Nx] = 0  # Dirichlet boundary conditions

                # Solve the system
                u[:] = np.linalg.solve(A, b)

                if user_action is not None:
                    user_action(u, x, t, n + 1)

                # Update for the next step
                u_n, u = u, u_n
       
        elif version == 'vectorized':
            """
            Vectorized implementation of solver_BE_simple using also
            a sparse (tridiagonal) matrix for efficiency.
            """
            # Sparse matrix setup (unchanged)
            diagonal = np.ones(Nx + 1) * (1 + 2 * F)
            lower = np.ones(Nx) * -F
            upper = np.ones(Nx) * -F

            diagonal[0] = diagonal[-1] = 1
            lower[-1] = 0
            upper[0] = 0

            A = spdiags(
                [np.append(lower, 0), diagonal, np.append(0, upper)],
                [-1, 0, 1],
                Nx + 1,
                Nx + 1,
            ).tocsc()

            for n in range(Nt):
                b = u_n + dt * f(x, t[n + 1])
                b[0] = b[-1] = 0.0  # Dirichlet boundary conditions

                u[:] = spsolve(A, b)

                if user_action is not None:
                    user_action(u, x, t, n + 1)

                u_n, u = u, u_n
        else:
            raise ValueError("Invalid version: use 'scalar' or 'vectorized'.")


    else:
        raise ValueError("Invalid method: use 'FE' or 'BE'.")

    t1 = time.time()
    return u_n, x, t, t1 - t0

def theta_diffusion_solver(I, f, a, L, dt, F, T, theta, user_action=None):
    """
    θ-rule for the 1D diffusion equation.

    Parameters:
    - I: Initial condition function.
    - f: Source term function f(x, t).
    - a: Diffusion coefficient.
    - L: Length of the spatial domain.
    - dt: Number of time grid points.
    - F: Fourier number (a * dt / dx^2).
    - T: Total simulation time.
    - theta: Weighting factor (0 <= theta <= 1).
    - user_action: Optional function for processing each time step.
    """
    

    # Start measuring CPU time
    t0 = time.time()

    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
    dx = np.sqrt(a*dt/F)
    Nx = int(round(L/dx)) # Number of spatial grid points.
    x = np.linspace(0, L, Nx+1)       # Mesh points in space
    
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Initialize solution arrays
    u = np.zeros(Nx + 1)
    u_n = np.zeros(Nx + 1)

    # Set initial condition
    for i in range(Nx + 1):
        u_n[i] = I(x[i])

    if user_action is not None:
        user_action(u_n, x, t, 0)

    # Construct sparse matrix A
    diagonal = np.ones(Nx + 1) * (1 + 2 * theta * F)
    lower = np.ones(Nx) * (-theta * F)
    upper = np.ones(Nx) * (-theta * F)

    # Adjust boundary conditions
    diagonal[0] = diagonal[-1] = 1
    lower[-1] = 0
    upper[0] = 0

    A = diags([lower, diagonal, upper], offsets=[-1, 0, 1], format="csr")

    # Time-stepping loop
    for n in range(Nt):
        # Compute the right-hand side
        b = u_n.copy()
        b[1:Nx] += (1 - theta) * F * (u_n[0:Nx-1] - 2 * u_n[1:Nx] + u_n[2:Nx+1])
        b += dt * ((1 - theta) * f(x, t[n]) + theta * f(x, t[n + 1]))
        b[0] = b[-1] = 0  # Dirichlet boundary conditions

        # Solve the linear system
        u[:] = spsolve(A, b)

        if user_action is not None:
            user_action(u, x, t, n + 1)

        u_n[:] = u

    t1 = time.time()

    return u, x, t, t1 - t0

def theta_diffusion_2D(
    I, f, alpha, Lx, Ly, Nx, Ny, dt, T, theta=0.5,
    U_0x=0, U_0y=0, U_Lx=0, U_Ly=0, user_action=None
):
    """
    Full solver for the 2D diffusion equation using the theta-rule.
    Dense matrix and Gaussian solve.

    Parameters:
    - I: Initial condition function I(x, y).
    - f: Source term function f(x, y, t).
    - alpha: Diffusion coefficient.
    - Lx, Ly: Dimensions of the spatial domain.
    - Nx, Ny: Number of grid points in the x and y directions.
    - dt: Time step size.
    - T: Total simulation time.
    - theta: Weighting parameter (0 <= theta <= 1).
    - U_0x, U_0y, U_Lx, U_Ly: Boundary conditions (can be functions or constants).
    - user_action: Optional function for visualization or output.

    Returns:
    - t: Time array.
    - cpu_time: Total CPU time for the simulation.
    """
    import time
    t0 = time.time()  # Start measuring CPU time

    x = np.linspace(0, Lx, Nx + 1)  # Mesh points in x direction
    y = np.linspace(0, Ly, Ny + 1)  # Mesh points in y direction
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    Nt = int(round(T / dt))  # Number of time steps
    t = np.linspace(0, Nt * dt, Nt + 1)  # Time array

    # Fourier numbers
    Fx = alpha * dt / dx**2
    Fy = alpha * dt / dy**2

    # Allow f to be None or zero
    if f is None or f == 0:
        f = lambda x, y, t: 0

    u = np.zeros((Nx + 1, Ny + 1))  # Solution at the new time level
    u_n = np.zeros((Nx + 1, Ny + 1))  # Solution at the previous time level

    Ix = range(0, Nx + 1)
    Iy = range(0, Ny + 1)
    It = range(0, Nt + 1)

    if not callable(U_0x):
        U_0x_val = float(U_0x)  # Convert scalar to float once
        U_0x = lambda t: U_0x_val
    if not callable(U_0y):
        U_0y_val = float(U_0y)  # Convert scalar to float once
        U_0y = lambda t: U_0y_val
    if not callable(U_Lx):
        U_Lx_val = float(U_Lx)  # Convert scalar to float once
        U_Lx = lambda t: U_Lx_val
    if not callable(U_Ly):
        U_Ly_val = float(U_Ly)  # Convert scalar to float once
        U_Ly = lambda t: U_Ly_val

    # Load initial condition into u_n
    for i in Ix:
        for j in Iy:
            u_n[i, j] = I(x[i], y[j])

    # Prepare for user_action
    xv = x[:, np.newaxis]
    yv = y[np.newaxis, :]

    if user_action is not None:
        user_action(u_n, x, xv, y, yv, t, 0)

    # Data structures for the linear system
    N = (Nx + 1) * (Ny + 1)  # Total number of unknowns
    A = np.zeros((N, N))  # Coefficient matrix
    b = np.zeros(N)  # Right-hand side vector

    # Fill the matrix A
    m = lambda i, j: j * (Nx + 1) + i  # Linear index for (i, j)

    # Boundary conditions and internal points
    for j in Iy:
        for i in Ix:
            p = m(i, j)  # Linear index for the current point
            if j == 0 or j == Ny or i == 0 or i == Nx:  # Boundary points
                A[p, p] = 1
            else:  # Internal points
                A[p, m(i, j - 1)] = -theta * Fy  # y-direction lower
                A[p, m(i - 1, j)] = -theta * Fx  # x-direction lower
                A[p, p] = 1 + 2 * theta * (Fx + Fy)  # Center
                A[p, m(i + 1, j)] = -theta * Fx  # x-direction upper
                A[p, m(i, j + 1)] = -theta * Fy  # y-direction upper

    # Time-stepping loop
    for n in It[:-1]:
        # Compute the right-hand side
        for j in Iy:
            for i in Ix:
                p = m(i, j)
                if j == 0:
                    b[p] = U_0y(t[n + 1])  # Bottom boundary
                elif j == Ny:
                    b[p] = U_Ly(t[n + 1])  # Top boundary
                elif i == 0:
                    b[p] = U_0x(t[n + 1])  # Left boundary
                elif i == Nx:
                    b[p] = U_Lx(t[n + 1])  # Right boundary
                else:
                    b[p] = (
                        u_n[i, j]
                        + (1 - theta)
                        * (
                            Fx * (u_n[i + 1, j] - 2 * u_n[i, j] + u_n[i - 1, j])
                            + Fy * (u_n[i, j + 1] - 2 * u_n[i, j] + u_n[i, j - 1])
                        )
                        + theta * dt * f(i * dx, j * dy, (n + 1) * dt)
                        + (1 - theta) * dt * f(i * dx, j * dy, n * dt)
                    )

        # Solve the linear system
        c = scipy.linalg.solve(A, b)

        # Fill u with the solution vector c
        for i in Ix:
            for j in Iy:
                u[i, j] = c[m(i, j)]

        if user_action is not None:
            user_action(u, x, xv, y, yv, t, n + 1)

        # Update u_n for the next time step
        u_n, u = u, u_n

    t1 = time.time()

    return t, t1 - t0




def build_theta_matrix_sparse(Nx, Ny, Fx, Fy, theta):
    """
    Build the sparse matrix for the theta-rule in 2D.
    """
    N = (Nx + 1) * (Ny + 1)
    main_diag = (1 + 2 * theta * (Fx + Fy)) * np.ones(N)
    side_diag_x = -theta * Fx * np.ones(N - 1)
    side_diag_y = -theta * Fy * np.ones(N - (Nx + 1))

    # Handle boundaries in the x-direction
    side_diag_x[Nx::Nx + 1] = 0

    diagonals = [main_diag, side_diag_x, side_diag_x, side_diag_y, side_diag_y]
    offsets = [0, 1, -1, Nx + 1, -(Nx + 1)]

    A = scipy.sparse.diags(diagonals, offsets, shape=(N, N), format="csr")
    return A


def build_rhs_2D(u_n, f, x, y, t_n, t_np1, Fx, Fy, theta, dt):
    """
    Build the right-hand side vector for the theta-rule in 2D.
    """
    Nx, Ny = u_n.shape[0] - 1, u_n.shape[1] - 1
    b = np.zeros_like(u_n)

    # Compute the RHS
    for i in range(1, Nx):
        for j in range(1, Ny):
            b[i, j] = (
                u_n[i, j]
                + (1 - theta) * Fx * (u_n[i - 1, j] - 2 * u_n[i, j] + u_n[i + 1, j])
                + (1 - theta) * Fy * (u_n[i, j - 1] - 2 * u_n[i, j] + u_n[i, j + 1])
                + dt * ((1 - theta) * f(x[i], y[j], t_n) + theta * f(x[i], y[j], t_np1))
            )

    return b

"""
------------------------------------------------------------------------------------------------------
                                            Visualization
------------------------------------------------------------------------------------------------------
"""

def save_diffusion_images_basic(results, x, save_dir='diffusion_images', ymin=None, ymax=None):
    """Save diffusion solution at each time step as images."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for n, (u, t) in enumerate(results):
        plt.figure(figsize=(8, 4))
        plt.plot(x, u, label=f"t = {t:.5f}")
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.title(f"Diffusion at t = {t:.5f}")
        plt.legend()
        plt.grid()
        plt.xlim(x[0], x[-1])
        plt.ylim(ymin if ymin is not None else min(u), ymax if ymax is not None else max(u))
        plt.savefig(os.path.join(save_dir, f'diffusion_step_{n:04d}.png'))
        plt.close()

def save_diffusion_images(results, x, save_dir='diffusion_images', ymin=None, ymax=None, 
                          title_comment='', method='FE', version='scalar', F=0.5):
    """
    Save diffusion solution at each time step as images with enhanced titles.

    Parameters:
    - results: List of tuples (u, t), where u is the solution at time t.
    - x: Spatial grid points.
    - save_dir: Directory to save the images.
    - ymin, ymax: Limits for the y-axis in the plot.
    - title_comment1: Additional text to include in the title.
    - method: Solver method ('FE' or 'BE').
    - version: Solver version ('scalar' or 'vectorized').
    - F: Fourier number.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for n, (u, t) in enumerate(results):
        plt.figure(figsize=(8, 4))
        plt.plot(x, u, label=f"t = {t:.5f}")
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.title(
            f"{title_comment}, t = {t:.5f}, solver: {method} ({version}), F = {F:.1f}"
        )
        plt.legend()
        plt.grid()
        plt.xlim(x[0], x[-1])
        plt.ylim(ymin if ymin is not None else min(u), ymax if ymax is not None else max(u))
        plt.savefig(os.path.join(save_dir, f'diffusion_step_{n:04d}.png'))
        plt.close()

def generate_gif_from_images(image_folder='diffusion_images', gif_name='diffusion_animation.gif', duration=0.1):
    """Generate a GIF from saved images."""
    images = []
    files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    for file in files:
        images.append(imageio.imread(os.path.join(image_folder, file)))
    imageio.mimsave(os.path.join(image_folder, gif_name), images, duration=duration)
    # print(f"GIF saved as {os.path.join(image_folder, gif_name)}")

def generate_html_animation(x, results, save_dir='diffusion_images', filename='diffusion_movie.html'):
    """Generate HTML animation for diffusion results."""
    fig, ax = plt.subplots()
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(min(min(u) for u, _ in results), max(max(u) for u, _ in results))
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(n):
        u, t = results[n]
        line.set_data(x, u)
        ax.set_title(f"Diffusion at t = {t:.3f}")
        return line,

    ani = FuncAnimation(fig, animate, init_func=init, frames=len(results), blit=True)
    html_path = os.path.join(save_dir, filename)
    ani.save(html_path, writer=HTMLWriter())
    # print(f"HTML animation saved as {html_path}")

def visualize_2D_diffusion_on_2D_and_3D_plots(u, x, y, time_step, save_dir, title="2D and 3D Diffusion", mode="both", z_min=0, z_max=1):
    """
    Visualize the 2D diffusion solution at a specific timestep with 2D and 3D plots.

    Parameters:
    - u: Solution matrix at the current timestep.
    - x: x-axis spatial grid.
    - y: y-axis spatial grid.
    - time_step: Current timestep index.
    - save_dir: Directory to save the plot images.
    - title: Title of the plots.
    - mode: '2D', '3D', or 'both' (default: 'both').
    - z_min: Minimum value for the z-axis (fixed scale).
    - z_max: Maximum value for the z-axis (fixed scale).
    """
        # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Initialize the figure
    if mode == "both":
        fig = plt.figure(figsize=(16, 8))
        ax2d = fig.add_subplot(1, 2, 1)  # 2D plot
        ax3d = fig.add_subplot(1, 2, 2, projection='3d')  # 3D plot
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection='3d' if mode == "3D" else None)

    if mode in ["2D", "both"]:
        ax = ax2d if mode == "both" else ax
        c = ax.contourf(X, Y, u, levels=np.linspace(z_min, z_max, 100), cmap='viridis', vmin=z_min, vmax=z_max)
        plt.colorbar(c, ax=ax, label='u(x, y)')
        ax.set_title(f"{title} - 2D Plot\nTime Step: {time_step}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')

    if mode in ["3D", "both"]:
        ax = ax3d if mode == "both" else ax
        surf = ax.plot_surface(X, Y, u, cmap='viridis', edgecolor='none', vmin=z_min, vmax=z_max)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u(x, y)')
        ax.set_title(f"{title} - 3D Plot\nTime Step: {time_step}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_zlim(z_min, z_max)  # Fixed z-axis range
    
        # Adjust the viewing angle to make the plot appear flatter
        ax.view_init(elev=15, azim=120)  # Set elevation and azimuth angles


    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"diffusion_step_{time_step:04d}.png"), bbox_inches='tight', transparent=False)
    plt.close()

"""
------------------------------------------------------------------------------------------------------
                                            Testing Solvers
------------------------------------------------------------------------------------------------------
"""

def test_solvers_1():
    """
    Test unified_diffusion_solver against an exact solution to verify correctness.
    """
    # ------------------------------
    # Define the problem parameters
    # ------------------------------
    def u_exact(x, t):
        """
        Exact solution: satisfies boundary conditions at x=0 and x=L.
        """
        return x * (L - x) * 5 * t

    def I(x):
        """
        Initial condition: u_exact at t=0.
        """
        return u_exact(x, 0)

    def f(x, t):
        """
        Source term for the diffusion equation.
        """
        return 5 * x * (L - x) + 10 * a * t

    a = 3.5  # Diffusion coefficient
    L = 1.5  # Length of the spatial domain
    Nx = 4   # Number of spatial grid points
    F = 0.5  # Fourier number
    dx = L / Nx
    dt = F * dx**2 / a  # Time step size

    # ------------------------------
    # Define the user_action for comparison
    # ------------------------------
    def compare(u, x, t, n):
        """
        Compare exact and computed solutions at each time step.
        """
        u_e = u_exact(x, t[n])  # Exact solution
        diff = abs(u_e - u).max()  # Maximum absolute difference
        tol = 1E-14  # Tolerance for numerical error
        assert diff < tol, f"Max diff at step {n}: {diff:.2e}"

    # ------------------------------
    # List of solvers to test
    # ------------------------------
    import functools
    s = functools.partial  # Simplifies function calls with preset arguments

    solvers = [
        # Forward Euler (scalar)
        s(
            diffusion_PDE_solver, 
            I=I, f=f, a=a, L=L, dt=dt, F=F, T=0.2,
            method='FE', version='scalar', user_action=compare
        ),
        # Forward Euler (vectorized)
        s(
            diffusion_PDE_solver, 
            I=I, f=f, a=a, L=L, dt=dt, F=F, T=2,
            method='FE', version='vectorized', user_action=compare
        ),
        # Backward Euler (scalar)
        s(
            diffusion_PDE_solver, 
            I=I, f=f, a=a, L=L, dt=dt, F=F, T=2,
            method='BE', version='scalar', user_action=compare
        ),
        # Backward Euler (vectorized)
        s(
            diffusion_PDE_solver, 
            I=I, f=f, a=a, L=L, dt=dt, F=F, T=2,
            method='BE', version='vectorized', user_action=compare
        ),
    ]

    # ------------------------------
    # Test the solvers
    # ------------------------------
    for solver in solvers:
        print(f"Testing {solver.keywords['method']} ({solver.keywords['version']})...")
        u, x, t, cpu_time = solver()
        u_e = u_exact(x, t[-1])
        diff = abs(u_e - u).max()
        print (f"Max diff:{diff:.20e}")
        tol = 1E-14
        assert diff < tol, f"Final max diff for {solver.keywords['method']} ({solver.keywords['version']}): {diff:g}"
        print(f"{solver.keywords['method']} ({solver.keywords['version']}) passed. CPU time: {cpu_time:.10f} seconds\n")

    print("All solvers passed the tests!")

def quadratic_solvers_3(theta, Nx, Ny):
    """Exact discrete solution of the scheme."""
    def u_exact(x, y, t):
        """Exact solution."""
        return 5 * t * x * (Lx - x) * y * (Ly - y)

    def I(x, y):
        """Initial condition."""
        return u_exact(x, y, 0)

    def f(x, y, t):
        """Source term."""
        return 5 * x * (Lx - x) * y * (Ly - y) + 10 * a * t * (y * (Ly - y) + x * (Lx - x))

    # Domain parameters
    Lx = 0.75
    Ly = 1.5
    a = 3.5
    dt = 0.5
    T = 2  # Total simulation time

    def assert_no_error(u, x, xv, y, yv, t, n):
        """Assert zero error at all mesh points."""
        xv, yv = np.meshgrid(x, y, indexing='ij')
        u_e = u_exact(xv, yv, t[n])
        diff = abs(u - u_e).max()
        tol = 1E-12
        msg = f"diff={diff:.6e}, step {n}, time={t[n]}"
        print(msg)
        assert diff < tol, msg

    # Test the solver with `theta_diffusion_2D`
    print(f"\nTesting theta={theta}, Nx={Nx}, Ny={Ny}")
    t, cpu = theta_diffusion_2D(
        I=I,
        f=f,
        alpha=a,
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        T=T,
        dt=dt,
        theta=theta,
        U_0x=0,    # Explicit scalar boundary condition
        U_0y=0,    # Explicit scalar boundary condition
        U_Lx=0,    # Explicit scalar boundary condition
        U_Ly=0,    # Explicit scalar boundary condition
        user_action=assert_no_error,
    )

    print(f"Test completed for theta={theta}, Nx={Nx}, Ny={Ny}.")
    return t, cpu

def test_quadratic_solvers_3():
    """Test quadratic solution for various meshes and theta values."""
    for theta in [1, 0.5, 0]:  # Backward Euler, Crank-Nicolson, Forward Euler
        for Nx in range(2, 6, 2):
            for Ny in range(2, 6, 2):
                print(f"\n*** Testing for {Nx}x{Ny} mesh with theta={theta}")
                quadratic_solvers_3(theta, Nx, Ny)

# -----------------------------------------------------------
#               --------------------------
# ----------------------------------------------------------- 

def validate_boundary_conditions():
    # Problem setup
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 10, 10
    alpha = 1.0
    dt = 0.01
    T = 0.1
    theta = 0.5  # Crank-Nicolson

    # Initial condition: Zero everywhere
    def I(x, y):
        return 0

    # Source term: None
    f = None

    # Boundary conditions
    U_0x = lambda t: np.sin(t)  # Left boundary
    U_0y = lambda t: np.cos(t)  # Bottom boundary
    U_Lx = lambda t: t          # Right boundary
    U_Ly = lambda t: t**2       # Top boundary

    def user_action(u, x, xv, y, yv, t, n):
        if n == 0 or n == len(t) - 1:  # Check first and last time step
            print(f"Time step {n}:")
            print(f"Left boundary (x=0): {u[0, :]}")
            print(f"Right boundary (x=Lx): {u[-1, :]}")
            print(f"Bottom boundary (y=0): {u[:, 0]}")
            print(f"Top boundary (y=Ly): {u[:, -1]}")

    # Run the solver
    t, cpu_time = theta_diffusion_2D(
        I=I,
        f=f,
        alpha=alpha,
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        dt=dt,
        T=T,
        theta=theta,
        U_0x=U_0x,
        U_0y=U_0y,
        U_Lx=U_Lx,
        U_Ly=U_Ly,
        user_action=user_action,
    )

    print("Boundary condition validation completed.")


if __name__ == "__main__":
# -----------------------------------------------------------
# You can run tests with py.test or activate functions bellow
# -----------------------------------------------------------

# -----------------------------------------------------------
#              Test solver 1 (1D diffusion)
# -----------------------------------------------------------
    # try:
    #     test_solvers_1()
    #     print("All solvers passed the tests!")
    # except AssertionError as e:
    #     print(e)
# -----------------------------------------------------------
#               Test solver 3 (2D diffusion)
# -----------------------------------------------------------   
    # try:
    #     test_quadratic_solvers_3()
    #     print("All solvers passed the tests!")
    # except AssertionError as e:
    #     print(e)
# -----------------------------------------------------------
#               --------------------------
# ----------------------------------------------------------- 
    # validate_boundary_conditions()