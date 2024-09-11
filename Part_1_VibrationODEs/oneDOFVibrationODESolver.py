import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

"""
-----------------------------------------------------------------------------------------------------------------------------------------

            Solver calibrated to solve the 1 DOF system defined by m*u'' + f(u') + s(u) = F(t) using numerical methods.

-----------------------------------------------------------------------------------------------------------------------------------------
"""
def general_solver(m, damping_func, restoring_func, external_force, initial_conditions, dt, T, methods, exact_solution=None):
    """
            Dynamic system: m*u'' + f(u') + s(u) = F(t) 
    
    - m: mass
    - damping_func: user-defined function for damping (f(v))
    - restoring_func: user-defined function for restoring force (s(u))
    - external_force: user-defined function for external force (F(t))
    - initial_conditions: tuple (u(0), v(0)) -> initial displacement and velocity
    - dt: time step
    - T: total time
    - methods: list of methods to use for solving ('CD', 'FE', 'BE', 'CN', 'RK2', 'RK4', 'EC')
    """
    Nt = int(T / dt)
    t = np.linspace(0, T, Nt + 1)  # Time array
    results = {}  # Dictionary to store results for each method

    for method in methods:
        u = np.zeros(Nt + 1)  # Displacement array
        v = np.zeros(Nt + 1)  # Velocity array
        u[0], v[0] = initial_conditions  # Initial conditions: u(0), v(0)


        if method == 'CD':  # Central Difference Method
            # Initialization of the first step using the known initial conditions
            u[1] = u[0] + v[0] * dt + 0.5 * dt**2 * (external_force(t[0]) - damping_func(v[0]) - restoring_func(u[0])) / m
            
            # Main loop for updating displacement and velocity using the central difference scheme
            for n in range(1, Nt):
                F_n = external_force(t[n])  # External force at time step n
                f_v = damping_func((u[n] - u[n-1]) / dt)  # Damping force, approximated using the difference in displacement
                s_u = restoring_func(u[n])  # Restoring force
        
                # Update displacement using the central difference formula
                u[n+1] = 2 * u[n] - u[n-1] + (dt**2 / m) * (F_n - f_v - s_u)
        
                # Update velocity using the central difference approximation
                v[n] = (u[n+1] - u[n-1]) / (2 * dt)

        elif method == 'FE':  # Forward Euler Method
            for n in range(Nt):
                a_n = (external_force(t[n]) - damping_func(v[n]) - restoring_func(u[n])) / m
                v[n+1] = v[n] + dt * a_n
                u[n+1] = u[n] + dt * v[n]
                
        elif method == 'BE':  # Backward Euler Method
            for n in range(Nt):
                # We need to solve the implicit system for u[n+1] and v[n+1].
                # Using fsolve to solve this implicit equation.
                
                def implicit_eq(x):
                    # x[0] = u_next (u[n+1]), x[1] = v_next (v[n+1])
                    u_next = x[0]
                    v_next = x[1]
                    f_u = restoring_func(u_next)
                    f_v = damping_func(v_next)
                    F_next = external_force(t[n+1])

                    # Return the system of equations for u_next and v_next
                    return [
                        u_next - u[n] - dt * v_next,  # u_next implicit equation
                        v_next - v[n] - dt * (F_next - f_v - f_u) / m  # v_next implicit equation
                    ]

                # Solve for u[n+1] and v[n+1] using fsolve
                sol = fsolve(implicit_eq, [u[n], v[n]])  # Initial guess for the solution
                u[n+1] = sol[0]
                v[n+1] = sol[1]

        elif method == 'CN':  # Crank-Nicolson Method

            for n in range(Nt):
                # Compute acceleration at current step
                a_n = (external_force(t[n]) - damping_func(v[n]) - restoring_func(u[n])) / m
        
                # Predict next displacement using the current velocity
                u_predict = u[n] + dt * v[n]
        
                # Predict next velocity using current acceleration
                v_predict = v[n] + dt * a_n
        
                # Compute acceleration at the next time step using the predicted displacement and velocity
                a_predict = (external_force(t[n+1]) - damping_func(v_predict) - restoring_func(u_predict)) / m
        
                # Crank-Nicolson average update for displacement and velocity
                u[n+1] = u[n] + dt * 0.5 * (v[n] + v_predict)
                v[n+1] = v[n] + dt * 0.5 * (a_n + a_predict)

        elif method == 'RK2':  # Runge-Kutta 2 (Heun's Method)
            for n in range(Nt):
                k1_u = v[n]
                k1_v = (external_force(t[n]) - damping_func(v[n]) - restoring_func(u[n])) / m
                u_half = u[n] + 0.5 * dt * k1_u
                v_half = v[n] + 0.5 * dt * k1_v
                k2_u = v_half
                k2_v = (external_force(t[n] + 0.5 * dt) - damping_func(v_half) - restoring_func(u_half)) / m
                u[n+1] = u[n] + dt * k2_u
                v[n+1] = v[n] + dt * k2_v

        elif method == 'RK4':  # Runge-Kutta 4 Method
            for n in range(Nt):
                k1_u = v[n]
                k1_v = (external_force(t[n]) - damping_func(v[n]) - restoring_func(u[n])) / m
                u_half = u[n] + 0.5 * dt * k1_u
                v_half = v[n] + 0.5 * dt * k1_v
                k2_u = v_half
                k2_v = (external_force(t[n] + 0.5 * dt) - damping_func(v_half) - restoring_func(u_half)) / m
                u_half = u[n] + 0.5 * dt * k2_u
                v_half = v[n] + 0.5 * dt * k2_v
                k3_u = v_half
                k3_v = (external_force(t[n] + 0.5 * dt) - damping_func(v_half) - restoring_func(u_half)) / m
                u_full = u[n] + dt * k3_u
                v_full = v[n] + dt * k3_v
                k4_u = v_full
                k4_v = (external_force(t[n] + dt) - damping_func(v_full) - restoring_func(u_full)) / m
                u[n+1] = u[n] + (dt / 6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
                v[n+1] = v[n] + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        elif method == 'EC':  # Euler-Cromer Method
           for n in range(Nt):
                if n == 0:
                    # Special case for the first time step, similar to what you provided
                    a_n = (external_force(t[n]) - damping_func(v[n]) - restoring_func(u[n])) / m
                    v[1] = v[0] + 0.5 * dt * a_n
                else:
                    # General case for all other time steps
                    a_n = (external_force(t[n]) - damping_func(v[n]) - restoring_func(u[n])) / m
                    v[n+1] = v[n] + dt * a_n

                # Update displacement using the newly updated velocity
                u[n+1] = u[n] + dt * v[n+1]

        results[method] = (u, v, t)  # Store both displacement and velocity

    return results

def plot_results(results, exact_solution=None, plot_velocity=True, filename='a'):
    plt.figure(figsize=(10, 12))

    # Subplot for Displacement
    ax1 = plt.subplot(2, 1, 1)
    for method, (u, v, t) in results.items():
        if np.isnan(u).any() or np.isinf(u).any():
            print(f"Warning: Displacement contains NaN or Inf for method {method}. Skipping plot for this method.")
            continue
        plt.plot(t, u, '-', label=f"{method} Displacement")

    # If exact solution for displacement is provided, plot it with a fine time resolution
    if exact_solution is not None:
        t_fine = np.linspace(0, max(t), 1000)  # Fine time grid for exact solution
        u_exact_vals = exact_solution(t_fine)
        plt.plot(t_fine, u_exact_vals, 'k--', label='Exact Displacement', linewidth=2)

    plt.xlabel("Time (t)")
    plt.ylabel("Displacement (u)")
    plt.title("Solution for Displacement using Numerical Methods")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)  # Move legend below x-axis

    # Add grid
    plt.grid(True)

    # Ensure axis limits are set correctly to avoid errors
    if not np.isnan(u).any() and not np.isinf(u).any():
        ax1.set_xlim()#([0, max(t)])
        ax1.set_ylim()#([min(u) * 1.1, max(u) * 1.1])

    # Subplot for Velocity (if enabled)
    if plot_velocity:
        ax2 = plt.subplot(2, 1, 2)
        for method, (u, v, t) in results.items():
            if np.isnan(v).any() or np.isinf(v).any():
                print(f"Warning: Velocity contains NaN or Inf for method {method}. Skipping plot for this method.")
                continue
            plt.plot(t, v, '--', label=f"{method} Velocity")

        plt.xlabel("Time (t)")
        plt.ylabel("Velocity (v)")
        plt.title("Solution for Velocity using Numerical Methods")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)  # Move legend below x-axis

        # Add grid
        plt.grid(True)

        # Ensure axis limits are set correctly to avoid errors
        if not np.isnan(v).any() and not np.isinf(v).any():
            ax2.set_xlim()#([0, max(t)])
            ax2.set_ylim()#([min(v) * 1.1, max(v) * 1.1])

    # Prevent out-of-bound issues with tight_layout
    try:
        plt.tight_layout()
    except ValueError as e:
        print(f"Warning: Could not apply tight layout due to axis bounds: {e}")

    
    # Save the plot to file
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()
    
    plt.close()

def plot_results2(results, exact_solution=None, plot_velocity=True, filename='a2'):
    """
    Visualization of displacement and velocity for each method, 
    with each method on its own subplot, and comparison to the exact solution.
    """
    num_methods = len(results)  # Number of methods

    # Determine if we need 1 or 2 columns of subplots
    if plot_velocity:
        fig, axs = plt.subplots(num_methods, 2, figsize=(12, num_methods * 4))  # 2 columns: Displacement and Velocity
    else:
        fig, axs = plt.subplots(num_methods, 1, figsize=(15, num_methods * 4))  # 1 column: Displacement only

    # If there is only one method, axs is not a list, so we must handle that case
    if num_methods == 1:
        axs = [axs]

    # Loop over each method and plot its results
    for i, (method, (u, v, t)) in enumerate(results.items()):
        if np.isnan(u).any() or np.isinf(u).any():
            print(f"Warning: Displacement contains NaN or Inf for method {method}. Skipping plot for this method.")
            continue

        # Displacement subplot
        axs[i].plot(t, u, 'r--', label=f'{method} Displacement')
        
        if exact_solution is not None:
            t_fine = np.linspace(0, max(t), 1000)
            u_exact_vals = exact_solution(t_fine)
            axs[i].plot(t_fine, u_exact_vals, 'b-', label='Exact Displacement')

        axs[i].legend(loc='lower left')
        axs[i].set_xlabel('Time (t)')
        axs[i].set_ylabel('Displacement (u)')
        axs[i].set_title(f'{method}: Displacement')
        axs[i].grid(True)

        # Check axis limits for displacement
        axs[i].set_xlim()#([0, max(t)])
        axs[i].set_ylim()#([min(u) * 1.1, max(u) * 1.1])

        # Velocity subplot (if enabled)
        if plot_velocity:
            if np.isnan(v).any() or np.isinf(v).any():
                print(f"Warning: Velocity contains NaN or Inf for method {method}. Skipping plot for this method.")
                continue

            axs[i, 1].plot(t, v, 'g--', label=f'{method} Velocity')
            axs[i, 1].legend(loc='lower left')
            axs[i, 1].set_xlabel('Time (t)')
            axs[i, 1].set_ylabel('Velocity (v)')
            axs[i, 1].set_title(f'{method}: Velocity')
            axs[i, 1].grid(True)

            # Check axis limits for velocity
            axs[i, 1].set_xlim([0, max(t)])
            axs[i, 1].set_ylim([min(v) * 1.1, max(v) * 1.1])

    plt.tight_layout()


    # Save the plot to file
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()
    plt.close()

"""
-----------------------------------------------------------------------------------------------------------------------------------------

Test Error Analysis
The test is skipped if the exact_solution is not provided.

-----------------------------------------------------------------------------------------------------------------------------------------

"""

def test_three_steps(m, damping_func, restoring_func, external_force, initial_conditions, dt, T, methods, exact_solution=None):
    """
    Test procedure for comparing the first three steps of each numerical method to the exact solution.
    The test is skipped if the exact_solution is not provided.
    """
    if exact_solution is None:
        print("Skipping test_three_steps: No exact solution provided.")
        return

    # Expected result using exact solution for the first 3 steps
    t_exact = np.array([0, dt, 2*dt])
    u_by_hand = exact_solution(t_exact)

    # Tolerance for numerical comparison
    tol = 1E-3  # Set a tolerance as floating-point calculations can have small errors

    print("3 STEP ERROR TEST")

    # Run the solver for all methods
    results = general_solver(m, damping_func, restoring_func, external_force, initial_conditions, dt, T, methods)
    
    # Test each method and print if it PASSED or FAILED
    for method, (u, v, t) in results.items():
        try:
            # Compare the first three steps of the solver result to the expected result
            diff = np.abs(u_by_hand - u[:3]).max()
            
            # Check if the difference is larger than tolerance
            if diff < tol:
                print("Method {}: PASSED. Max difference is {}, expected {}, but got {}".format(method, diff, u_by_hand, u[:3]))
            else:
                print("Method {}: FAILED. Max difference is {}, expected {}, but got {}".format(method, diff, u_by_hand, u[:3]))
        
        except Exception as e:
            print("Method {}: FAILED ( Difference: {},  Tolerance: {}, Error: {})".format(method, diff, tol, e))




            
# Error calculation function for each method
def calculate_errors(m, damping_func, restoring_func, external_force, initial_conditions, dt, T, method, exact_solution=None):
    """
    Calculates local and global errors for a given method, compared to the exact solution.
    Skips error calculation if exact_solution is not provided.
    """
    if exact_solution is None:
        print("Skipping error calculation: No exact solution provided.")
        return None, None
    
    # Solve the system for this method using the general solver
    results = general_solver(m, damping_func, restoring_func, external_force, initial_conditions, dt, T, [method])
    u_numerical, v_numerical, t = results[method]

    # Get the exact solution at all time steps
    u_exact_vals = exact_solution(t)

    # Calculate local error (pointwise difference between numerical and exact solution)
    local_error = np.abs(u_numerical - u_exact_vals)

    # Calculate global error (sum of local errors)
    global_error = np.sum(local_error)

    return local_error, global_error

# Function to calculate and display the errors for all methods
def list_errors(m, damping_func, restoring_func, external_force, initial_conditions, dt, T, methods, exact_solution=None):
    """
    List local and global errors for multiple methods.
    Skips error calculation if exact_solution is not provided.
    """
    print("ERROR CALCULATION:")
    if exact_solution is None:
        print("Skipping error listing: No exact solution provided.")
        return
    
    # Iterate over methods and compute errors
    for method in methods:
        local_error, global_error = calculate_errors(m, damping_func, restoring_func, external_force, initial_conditions, dt, T, method, exact_solution)
        
        if local_error is not None:
            print(f"Method {method}, dt {dt}:")
            print(f"  Global Error = {global_error:.6f}")
            print(f"  Local Errors for first few time steps: {local_error[:5]}")  # Print the first few local errors

 

