""" 
Riemann Solver for the Time-Dependent 
    One-Dimensional Euler Equations using 
    Gudonov Upwind and Center Schemes. Exact solver
    for Riemann problem is imported prom previous code

Purpose: To solve the Riemann problem exactly and numerical
         for the time-dependent one-dimensional 
         Euler equations for an ideal gas.

Theory is found in Ref. 1, Chapter 4, 5 and 6 and 7 (centered scheme).

Reference:
1. Toro, E. F., "Riemann Solvers and Numerical 
                Methods for Fluid Dynamics" 
                Springer-Verlag, 1997
                Second Edition, 1999

This program logic is part of:

NUMERICA
A Library of Source Codes for Teaching, 
Research and Applications, 
by E. F. Toro
Published by NUMERITEK LTD, 1999
Website: www.numeritek.com
 """

import numpy as np
import math
import matplotlib.pyplot as plt
from main_EXACT_dev import main as exact_riemann_solver
from main_EXACT_dev import solve_Riemann_problem, read_input
import os

class Globals:
    def __init__(self):
        # Mesh and domain parameters
        self.domlen = None
        self.diaph1 = None
        self.diaph2 = None
        self.cells = None

        # Physical parameters
        self.gamma = None
        self.g8 = None  # Derived later
        self.timeou = None
        self.pscale = None
        self.cflcoe = None

        # Boundary conditions
        self.ibclef = None
        self.ibcrig = None

        # Initial conditions
        self.dlinits = None
        self.ulinit = None
        self.plinit = None
        self.dminit = None
        self.uminit = None
        self.pminit = None
        self.drinit = None
        self.urinit = None
        self.prinit = None

        # Simulation control
        self.nfrequ = None
        self.ntmaxi = None
        self.flux = None

        # Time control
        self.time = 0.0
        self.timtol = 1.0e-6
        self.dt = None
        self.dx = None

         # Arrays (including boundary indices: -1 to cells + 2)
        size = 3003  # Adjust based on maximum CELLS + boundary padding
        self.d = [0.0] * size
        self.u = [0.0] * size
        self.p = [0.0] * size
        self.cs = [[0.0] * size for _ in range(3)]
        self.fi = [[0.0] * size for _ in range(3)]

    def compute_g8(self):
        """ Compute derived gamma constant. """
        self.g8 = self.gamma - 1.0

globals = Globals()

def clean_line(line):
    """ Remove comments and strip extra spaces from a line. """
    return line.split('!')[0].split('#')[0].strip()

def reader(globals):
    """
    Purpose: to read initial parameters of the problem and update the Globals instance.
    """
    def read_input(filename):
        """ Reads and parses input data from the file. """
        with open(filename, "r") as file:
            lines = file.readlines()
        # Clean and filter lines
        return [clean_line(line) for line in lines if clean_line(line)]

    # Read and clean input lines
    clean_lines = read_input('E1FOCE.ini')
    
    # Assign variables to Globals instance
    globals.domlen = float(clean_lines[0])
    globals.diaph1 = float(clean_lines[1])
    globals.cells = int(clean_lines[2])
    globals.gamma = float(clean_lines[3])
    globals.timeou = float(clean_lines[4])
    globals.dlinits = float(clean_lines[5])
    globals.ulinit = float(clean_lines[6])
    globals.plinit = float(clean_lines[7])
    globals.dminit = float(clean_lines[8])
    globals.uminit = float(clean_lines[9])
    globals.pminit = float(clean_lines[10])
    globals.drinit = float(clean_lines[11])
    globals.urinit = float(clean_lines[12])
    globals.prinit = float(clean_lines[13])
    globals.diaph2 = float(clean_lines[14])
    globals.cflcoe = float(clean_lines[15])
    globals.ibclef = int(clean_lines[16])
    globals.ibcrig = int(clean_lines[17])
    globals.nfrequ = int(clean_lines[18])
    globals.ntmaxi = int(clean_lines[19])
    globals.pscale = float(clean_lines[20])
    globals.flux = int(clean_lines[21])

    # Compute derived values
    globals.compute_g8()

    # Input data echoed to screen
    print("\nInput data echoed to screen\n")
    print(f"DOMLEN = {globals.domlen}")
    print(f"DIAPH1 = {globals.diaph1}")
    print(f"CELLS  = {globals.cells}")
    print(f"GAMMA  = {globals.gamma}")
    print(f"TIMEOU = {globals.timeou}")
    print(f"DLINIT = {globals.dlinits}")
    print(f"ULINIT = {globals.ulinit}")
    print(f"PLINIT = {globals.plinit}")
    print(f"DMINIT = {globals.dminit}")
    print(f"UMINIT = {globals.uminit}")
    print(f"PMINIT = {globals.pminit}")
    print(f"DRINIT = {globals.drinit}")
    print(f"URINIT = {globals.urinit}")
    print(f"PRINIT = {globals.prinit}")
    print(f"DIAPH2 = {globals.diaph2}")
    print(f"CFLCOE = {globals.cflcoe}")
    print(f"IBCLEF = {globals.ibclef}")
    print(f"IBCRIG = {globals.ibcrig}")
    print(f"NFREQU = {globals.nfrequ}")
    print(f"NTMAXI = {globals.ntmaxi}")
    print(f"PSCALE = {globals.pscale}")
    print(f"FLUX   = {globals.flux}")

def initia(cells, globals):
    """
    Purpose: to set initial conditions.
    """
    # Compute gamma-related constant
    globals.g8 = globals.gamma - 1.0

    # Calculate mesh size DX
    globals.dx = globals.domlen / float(cells)

    # Initialize arrays
    globals.d = [0.0] * (cells + 3)
    globals.u = [0.0] * (cells + 3)
    globals.p = [0.0] * (cells + 3)
    globals.cs = [[0.0] * (cells + 3) for _ in range(3)]

    # Set initial data in the domain
    for i in range(1, cells + 1):
        xpos = (float(i) - 0.5) * globals.dx

        if xpos <= globals.diaph1:
            # Set initial values in left section of the domain
            globals.d[i] = globals.dlinits
            globals.u[i] = globals.ulinit
            globals.p[i] = globals.plinit

        elif globals.diaph1 < xpos <= globals.diaph2:
            # Set initial values in middle section of the domain
            globals.d[i] = globals.dminit
            globals.u[i] = globals.uminit
            globals.p[i] = globals.pminit

        else:
            # Set initial values in right section of the domain
            globals.d[i] = globals.drinit
            globals.u[i] = globals.urinit
            globals.p[i] = globals.prinit

        # Compute conserved variables
        globals.cs[0][i] = globals.d[i]
        globals.cs[1][i] = globals.d[i] * globals.u[i]
        globals.cs[2][i] = 0.5 * globals.cs[1][i] * globals.u[i] + globals.p[i] / globals.g8

def bcondi(cells, globals):
    """
    Purpose: to set boundary conditions.
    """
    # Left boundary condition
    if globals.ibclef == 0:
        # Transmissive boundary conditions on the left
        globals.d[0] = globals.d[1]
        globals.u[0] = globals.u[1]
        globals.p[0] = globals.p[1]
    else:
        # Reflective boundary conditions on the left
        globals.d[0] = globals.d[1]
        globals.u[0] = -globals.u[1]
        globals.p[0] = globals.p[1]

    # Right boundary condition
    if globals.ibcrig == 0:
        # Transmissive boundary conditions on the right
        globals.d[cells + 1] = globals.d[cells]
        globals.u[cells + 1] = globals.u[cells]
        globals.p[cells + 1] = globals.p[cells]
    else:
        # Reflective boundary conditions on the right
        globals.d[cells + 1] = globals.d[cells]
        globals.u[cells + 1] = -globals.u[cells]
        globals.p[cells + 1] = globals.p[cells]

def cflcon(cflcoe, cells, n, time, timeou, globals):
    """
    Apply the CFL condition to find a stable time step size DT.
    """
    smax = 0.0

    # Find maximum velocity and speed of sound
    for i in range(0, cells + 2):
        if globals.d[i] > 0.0 and globals.p[i] > 0.0:  # Ensure valid density and pressure
            c = math.sqrt(globals.gamma * globals.p[i] / globals.d[i])
            sbextd = abs(globals.u[i]) + c
            if sbextd > smax:
                smax = sbextd
        else:
            raise ValueError(f"Unphysical state at cell {i}: density={globals.d[i]}, pressure={globals.p[i]}")

    # Compute time step DT
    globals.dt = cflcoe * globals.dx / smax

    # Reduce DT size for early times
    if n <= 5:
        globals.dt *= 0.2

    # Ensure time does not exceed output time
    if (time + globals.dt) > timeou:
        globals.dt = timeou - time

    return time + globals.dt

def output(cells, pscale, globals):
    """
    Purpose: to output the solution at a specified time.
    Writes position, density, velocity, pressure (scaled), and energy.
    """
    # Open the output file
    with open("e1foce.out", "w") as file:
        for i in range(1, cells + 1):
            xpos = (float(i) - 0.5) * globals.dx  # Calculate position
            energi = globals.p[i] / globals.d[i] / globals.g8 / pscale  # Compute energy
            file.write(f"{xpos:14.6f}  {globals.d[i]:14.6f}  {globals.u[i]:14.6f}  "
                       f"{globals.p[i] / pscale:14.6f}  {energi:14.6f}\n")

    print("Solution successfully written to 'e1foce.out'.")

def update(cells, globals):
    """
    Update the solution using the conservative formula and compute physical variables.
    """
    dtodx = globals.dt / globals.dx

    for i in range(1, cells + 1):
        for k in range(3):
            globals.cs[k][i] = globals.cs[k][i] + dtodx * (globals.fi[k][i - 1] - globals.fi[k][i])

    # Compute primitive variables (density, velocity, pressure)
    for i in range(1, cells + 1):
        globals.d[i] = globals.cs[0][i]
        globals.u[i] = globals.cs[1][i] / globals.d[i]

        # Pressure calculation with safety check
        pressure = globals.g8 * (globals.cs[2][i] - 0.5 * globals.cs[1][i] * globals.u[i])
        if pressure < 0.0:
            print(f"Warning: Negative pressure corrected at cell {i}.")
            pressure = 1e-8  # Set to small positive value

        globals.p[i] = pressure

def flueval(cs, flux, globals):
    """
    Purpose: to compute flux vector components FLUX(K) given the
    components CS(K) of the vector of conserved variables.
    """
    d = cs[0]
    u = cs[1] / d
    p = globals.g8 * (cs[2] - 0.5 * d * u * u)
    e = cs[2]

    flux[0] = d * u
    flux[1] = d * u * u + p
    flux[2] = u * (e + p)

def godcen(cells, globals):
    """
    Purpose: to compute an intercell flux FI(K, I) according
    to the Godunov centered scheme (non-monotone).
    """
    # Compute conserved variables and fluxes on data
    for i in range(0, cells + 2):
        globals.cs[0][i] = globals.d[i]
        globals.cs[1][i] = globals.d[i] * globals.u[i]
        globals.cs[2][i] = 0.5 * globals.d[i] * globals.u[i] * globals.u[i] + globals.p[i] / globals.g8

        globals.fi[0][i] = globals.cs[1][i]
        globals.fi[1][i] = globals.cs[1][i] * globals.u[i] + globals.p[i]
        globals.fi[2][i] = globals.u[i] * (globals.cs[2][i] + globals.p[i])

    # At interface (I, I + 1), compute intermediate state GODC(K)
    godc = [0.0] * 3
    godf = [0.0] * 3

    for i in range(0, cells + 1):
        for k in range(3):
            godc[k] = 0.5 * (globals.cs[k][i] + globals.cs[k][i + 1]) + \
                      (globals.dt / globals.dx) * (globals.fi[k][i] - globals.fi[k][i + 1])

        # Compute the flux GODF(K) at the state GODC(K)
        flueval(godc, godf, globals)

        for k in range(3):
            globals.fi[k][i] = godf[k]

def godunov_upwind_flux(cells, globals):
    """
    Compute numerical fluxes using the exact Riemann solver.
    """

    rho_star_list, u_star_list, p_star_list = [], [], []  # Lists to store star region values

    for i in range(0, cells + 1):
        # Left and right states at the interface
        rho_L, u_L, p_L = globals.d[i], globals.u[i], globals.p[i]
        rho_R, u_R, p_R = globals.d[i + 1], globals.u[i + 1], globals.p[i + 1]

        if rho_L <= 0 or p_L <= 0 or rho_R <= 0 or p_R <= 0:
            raise ValueError(f"Invalid states: rho_L={rho_L}, p_L={p_L}, rho_R={rho_R}, p_R={p_R}")


        # Call the exact Riemann solver
        rho_star, u_star, p_star = exact_riemann_solver(rho_L, u_L, p_L, rho_R, u_R, p_R, globals.gamma)

        # Compute fluxes using the star region solution
        flux = [
            rho_star * u_star,                               # Mass flux
            rho_star * u_star * u_star + p_star,            # Momentum flux
            u_star * (0.5 * rho_star * u_star ** 2 + p_star / globals.g8 + p_star)  # Energy flux
        ]

        # Store flux at the interface
        globals.fi[0][i] = flux[0]
        globals.fi[1][i] = flux[1]
        globals.fi[2][i] = flux[2]

        # Store star region values
        rho_star_list.append(rho_star)
        u_star_list.append(u_star)
        p_star_list.append(p_star)

    return  rho_star_list, u_star_list, p_star_list

def plot_Riemann_problem(file_name, positions, density_exact, velocity_exact, pressure_exact, internal_energy_exact, 
                         save=False, figname="TestX_solution.png"
                         ):

    # Initialize lists to store data
    xpos, density_upwind, velocity_upwind, pressure_upwind, energy_upwind = [], [], [], [], []

    # Read data from the output file
    with open(file_name, "r") as file:
        for line in file:
            values = [float(v) for v in line.split()]
            xpos.append(values[0])
            density_upwind.append(values[1])
            velocity_upwind.append(values[2])
            pressure_upwind.append(values[3])
            energy_upwind.append(values[4])

    # Subsample data to plot every 2nd point
    xpos = xpos[::1]
    density_upwind = density_upwind[::1]
    velocity_upwind = velocity_upwind[::1]
    pressure_upwind = pressure_upwind[::1]
    energy_upwind = energy_upwind[::1]

    # Create the Results folder if it doesn't exist
    results_folder = os.path.join(os.getcwd(), "Results")
    if save:
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
    
    # Full path to save the file
    save_path = os.path.join(results_folder, figname)

    # Plot the results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2 rows, 2 columns

    # Density profile (Top-left)
    axes[0, 0].plot(positions, density_exact, label='exact')
    axes[0, 0].plot(xpos, density_upwind, '*', markersize = 2 ,label='upwind', color='black')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Density')
    axes[0, 0].grid()
    axes[0, 0].legend()

    # Velocity profile (Top-right)
    axes[0, 1].plot(positions, velocity_exact, label='Velocity_exact', color='orange')
    axes[0, 1].plot(xpos, velocity_upwind, '*', markersize = 2 ,label='upwind', color='black')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].set_title('Velocity')
    axes[0, 1].grid()
    axes[0, 1].legend()

    # Pressure profile (Bottom-left)
    axes[1, 0].plot(positions, pressure_exact, label='exact', color='red')
    axes[1, 0].plot(xpos, pressure_upwind, '*', markersize = 2 ,label='upwind', color='black')
    axes[1, 0].set_ylabel('Pressure')
    axes[1, 0].set_title('Pressure')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].grid()
    axes[1, 0].legend()

    # Internal energy profile (Bottom-right)
    axes[1, 1].plot(positions, internal_energy_exact, label='exact', color='purple')
    axes[1, 1].plot(xpos, energy_upwind, '*', markersize = 2 ,label='upwind', color='black')
    axes[1, 1].set_ylabel('Internal Energy')
    axes[1, 1].set_title('Internal Energy')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].grid()
    axes[1, 1].legend()

    # Save and/or show the plot
    if save:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to '{save_path}'")
    plt.show()

def main():
    """
    Driver program to solve the one-dimensional Euler equations
    using first-order schemes.
    """
    # Initialize time and tolerance
    globals.time = 0.0
    globals.timtol = 1.0e-6

    # Read parameters from input file
    reader(globals)

    # Set up initial conditions
    initia(globals.cells, globals)

    # Time marching procedure
    print('---------------------------------------------')
    print('   Time step N        TIME           TIMEOU')
    print('---------------------------------------------')

    for n in range(1, globals.ntmaxi + 1):
        # Set boundary conditions
        bcondi(globals.cells, globals)

        # Apply CFL condition
        globals.time = cflcon(globals.cflcoe, globals.cells, n, globals.time, globals.timeou, globals)

        # Compute numerical fluxes
        if globals.flux == 1:
            godunov_upwind_flux(globals.cells, globals)
        elif globals.flux == 2:
            godcen(globals.cells, globals)
      

        # Update solution
        update(globals.cells, globals)

        # Print progress at specified intervals
        if n % globals.nfrequ == 0:
            print(f"{n:12}      {globals.time:12.7f}    {globals.timeou:12.7f}")

        # Check if output time is reached
        timdif = abs(globals.time - globals.timeou)
        if timdif <= globals.timtol:
            output(globals.cells, globals.pscale, globals)
            print('---------------------------------------------')
            print(f"   Number of time steps = {n}")
            print('---------------------------------------------')
            break

    #--------------------------------------------------------------#
    #                   START: User Defined Parameters                 
    #--------------------------------------------------------------#
    # Input file containing parameters
    input_file = 'Test5_input.txt'  # Specify the input file name for exact Riemann solver
    filename = 'e1foce.out' # Specify the output file name for numerical Riemann solver  
    filename_png = 'Test5.png' # Name of image
    
    #--------------------------------------------------------------#
    #                   END: User Defined Parameters                 
    #--------------------------------------------------------------#

    #--------------------------------------------------------------#
    #                        Post-process
    #--------------------------------------------------------------#

    # Read parameters from input file
    domlen, diaph, cells, timeout, mpa = read_input(input_file)

    positions, density, velocity, pressure, internal_energy = solve_Riemann_problem(domlen, timeout, cells)
    
    plot_Riemann_problem(filename, positions, density, velocity, pressure, internal_energy,
                     save=True, figname=filename_png)      

if __name__ == "__main__":
    
    globals = Globals()  
    
    main()

    



