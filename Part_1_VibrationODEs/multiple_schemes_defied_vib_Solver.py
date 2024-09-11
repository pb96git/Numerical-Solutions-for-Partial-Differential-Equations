
import numpy as np
import matplotlib.pyplot as plt
from math import pi


"""
  Solve u'' + w**2*u = 0, for t in (0, T], u(0) = I and u'(0)=0, by a different fin. diff. method 
  with time step dt
""" 
def u_exact(t, I, w):
    return I*np.cos(w*t)
   
'''CENTRAL DIFFERENCE + 2nd to 1st order schemes '''

# Central Difference, and Advanced Schemes Solver
def solver(I, w, dt, T, method):
    dt = float(dt)
    Nt = int(round(T/dt))
    u = np.zeros(Nt + 1)
    t = np.linspace(0, Nt*dt, Nt+1)

    u[0] = I  # Initial condition u(0) = I


    if method == 'CD':  # Central Difference Method
        u[1] = I - 0.5 * w**2 * u[0] * dt**2 # u'(0) = 0
        for n in range(1, Nt):
            u[n+1] = 2 * u[n] - u[n-1] - w**2 * u[n] * dt**2


    elif method == 'FE':  # Forward Euler Method
        v = np.zeros(Nt + 1)
        v[0] = 0  # Initial velocity condition u'(0) = 0
        for n in range(Nt):
            u[n+1] = u[n] + dt * v[n]
            v[n+1] = v[n] - dt * w**2 * u[n]

    elif method == 'BE':  # Backward Euler Method
        v = np.zeros(Nt + 1)
        v[0] = 0  # Initial velocity condition u'(0) = 0
        for n in range(Nt):
            v[n+1] = (v[n] - dt * w**2 * u[n]) / (1 + dt**2 * w**2)
            u[n+1] = u[n] + dt * v[n+1]

    elif method == 'CN':  # Crank-Nicolson Method
        v = np.zeros(Nt + 1)
        v[0] = 0  # Initial velocity condition u'(0) = 0
        for n in range(Nt):
            u_star = u[n] + 0.5 * dt * v[n]
            v_star = v[n] - 0.5 * dt * w**2 * u[n]
            u[n+1] = u[n] + dt * v_star
            v[n+1] = v[n] - dt * w**2 * u_star

    elif method == 'RK2':  # Runge-Kutta 2 (Heun's Method)
        v = np.zeros(Nt + 1)
        v[0] = 0  # Initial velocity condition u'(0) = 0
    
        for n in range(Nt):
            # Step size
            h = dt
            
            # Compute k1 for u and v
            k1_u = v[n]  # v represents u'
            k1_v = -w**2 * u[n]  # The equation for v'
    
            # Compute tentative values for u and v (Heun's method)
            u_tilde = u[n] + h * k1_u
            v_tilde = v[n] + h * k1_v
    
            # Compute k2 for u and v at the tentative points
            k2_u = v_tilde
            k2_v = -w**2 * u_tilde
    
            # Update u and v using the average of k1 and k2
            u[n+1] = u[n] + (h / 2) * (k1_u + k2_u)
            v[n+1] = v[n] + (h / 2) * (k1_v + k2_v)

    elif method == 'RK4':  # Runge-Kutta 4 Method
        v = np.zeros(Nt + 1)
        v[0] = 0  # Initial velocity condition u'(0) = 0
        for n in range(Nt):
            k1_u = v[n]
            k1_v = -w**2 * u[n]
            u_half = u[n] + 0.5 * dt * k1_u
            v_half = v[n] + 0.5 * dt * k1_v
            k2_u = v_half
            k2_v = -w**2 * u_half
            u_half = u[n] + 0.5 * dt * k2_u
            v_half = v[n] + 0.5 * dt * k2_v
            k3_u = v_half
            k3_v = -w**2 * u_half
            u_full = u[n] + dt * k3_u
            v_full = v[n] + dt * k3_v
            k4_u = v_full
            k4_v = -w**2 * u_full
            u[n+1] = u[n] + (dt/6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
            v[n+1] = v[n] + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    elif method == 'EC':  # Euler-Cromer Method
        v = np.zeros(Nt + 1)
        v[0] = 0  # Initial velocity condition u'(0) = 0
        for n in range(Nt):
            if n == 0:
                v[1] = v[0] - 0.5*dt*w**2*u[n]
            else:
                v[n+1] = v[n] - dt * w**2 * u[n]
            u[n+1] = u[n] + dt * v[n+1]


    return u, t


# Visualization function
def visualize(u_cd, u_fd, u_bd, u_fe, u_be, u_cn, u_rk2, u_rk4, u_ec, t, I, w, dt, ax):
    t_fine = np.linspace(0, t[-1], 1001)
    u_e = u_exact(t_fine, I, w)
    
    ax.plot(t, u_cd, 'r--', label='Central Difference (CD)')
    ax.plot(t, u_fe, 'c--', label='Forward Euler')
    ax.plot(t, u_be, 'y--', label='Backward Euler')
    ax.plot(t, u_cn, 'k--', label='Crank-Nicolson')
    ax.plot(t, u_rk2, 'b--', label='Runge-Kutta 2')
    ax.plot(t, u_rk4, 'y--', label='Runge-Kutta 4')
    ax.plot(t, u_ec, 'm--', label='Euler-Cromer')
    ax.plot(t_fine, u_e, 'b-', label='Exact Solution')
    
    ax.legend(loc='lower left')
    ax.set_xlabel('t')
    ax.set_ylabel('u')
    ax.set_title('dt=%g' % dt)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True)
    



# Function to visualize the solutions of all methods for different time steps
def main(I, w, dt_values, T, methods):
    
    # Create subplots
    fig, axs = plt.subplots(len(dt_values), 1, figsize=(10, 10))

    # Iterate over dt values and solve using all methods
    for i, dt in enumerate(dt_values):
        u_cd, t = solver(I, w, dt, T, method='CD')
        u_fd, _ = solver(I, w, dt, T, method='FD')
        u_bd, _ = solver(I, w, dt, T, method='BD')
        u_fe, _ = solver(I, w, dt, T, method='FE')
        u_be, _ = solver(I, w, dt, T, method='BE')
        u_cn, _ = solver(I, w, dt, T, method='CN')
        u_rk2, _ = solver(I, w, dt, T, method='RK2')
        u_rk4, _ = solver(I, w, dt, T, method='RK4')
        u_ec, _ = solver(I, w, dt, T, method='EC')
        
        # Plot the results
        visualize(u_cd, u_fd, u_bd, u_fe, u_be, u_cn, u_rk2, u_rk4, u_ec, t, I, w, dt, axs[i])

    plt.tight_layout()
    plt.savefig('Diff_Num_Methods.png')
    plt.show()
    
# Visualization function: Each method on its own subplot with the exact solution
def visualize_individual_methods(u_methods, t, I, w, dt, methods):
    t_fine = np.linspace(0, t[-1], 1001)
    u_e = u_exact(t_fine, I, w)
    
    num_methods = len(methods)
    
    # Create subplots, one for each method
    fig, axs = plt.subplots(num_methods, 1, figsize=(10, 10))

    for i, method in enumerate(methods):
        axs[i].plot(t, u_methods[method], 'r--', label=f'{method}')
        axs[i].plot(t_fine, u_e, 'b-', label='Exact Solution')
        
        axs[i].legend(loc='lower left')
        axs[i].set_xlabel('t')
        axs[i].set_ylabel('u')
        axs[i].set_title(f'{method} with dt={dt}')
        axs[i].set_xlim(t[0], t[-1])
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig(f'Each_Method_Comparison_dt_{dt}.png')
    plt.show()

# Function to solve and store the results for each method
def main2(I, w, dt_values, T, methods):
    for dt in dt_values:
        u_methods = {}
        
        # Solve for each method
        for method in methods:
            u, t = solver(I, w, dt, T, method=method)
            u_methods[method] = u
        
        # Plot the results, each method on its own subplot
        visualize_individual_methods(u_methods, t, I, w, dt, methods)

''' Test procedure '''

def test_three_steps(I, w, dt, T, methods):

    
    # Expected result using exact solution (u_exact) for the first 3 steps
    t_exact = np.array([0, dt, 2*dt])
    u_by_hand = u_exact(t_exact, I, w)

    # Tolerance for numerical comparison
    tol = 1E-3  # We set a tolerance as floating-point calculations can have small errors

    print("3 STEP ERROR TEST")
    # Test each method and print if it PASSED or FAILED
    for method in methods:
        try:
            u, t = solver(I, w, dt, T, method=method)

            # Compare the first three steps of the solver result to the expected result
            diff = np.abs(u_by_hand - u[:3]).max()
            
            assert diff < tol

            # Check if the difference is larger than tolerance
            if diff < tol:
                print ("Method {}: PASSED. Max difference is {}, expected {}, but got {}".format(method, diff, u_by_hand, u[:3]))
            else:
                print ("Method {}: FAILED Max difference is {}, expected {}, but got {}".format(method, diff, u_by_hand, u[:3]))
        except Exception as e:
            print ("Method {}: FAILED ({}Difference: {},  Tolerance:{})".format(method, e, diff, tol))


def convergence_rates(I, w, dt, T, m, methods):
    """
    Return m-1 empirical estimates of the convergence rate
    based on m simulations, where the time step is halved
    for each simulation.
    solver_function(I, w, dt, T, method) solves each problem, where T
    is based on simulation for num_periods periods.
    """
    dt_values = []
    E_values = {method: [] for method in methods}

    # Iterate over the methods and halving time steps
    for i in range(m):
        dt_values.append(dt)

        for method in methods:
            u_numerical, t = solver(I, w, dt, T, method=method)
            u_exact_vals = u_exact(t, I, w)
            
            # Calculate global error using the L2 norm
            global_error = np.sqrt(dt * np.sum((u_exact_vals - u_numerical) ** 2))
            E_values[method].append(global_error)

        dt /= 2  # Halve the time step

    # Convergence rate calculation
    rates = {}
    for method in methods:
        rates[method] = [np.log(E_values[method][i-1] / E_values[method][i]) /
                         np.log(dt_values[i-1] / dt_values[i]) for i in range(1, m)]
    return rates, E_values, dt_values


def test_convergence_rates(I, w, dt, T, m, methods):
    """
    Test convergence rates for the given methods. Continues testing other methods even if one fails.
    """
    # Get convergence rates and errors for each method
    rates, E_values, dt_values = convergence_rates(I, w, dt, T, m, methods)

    # Set a tolerance level for the convergence rate (to one decimal place)
    tol = 0.1

    print("ERROR CONVERGENCE TEST")
    
    # Check if each method achieves a second-order convergence (rate ~ 2.0)
    for method in methods:
        try:
            r = rates[method][-1]  # Get the last convergence rate
            assert abs(r - 2.0) < tol, f"Convergence rate {r} deviates from expected 2.0"
            print(f"Method {method} PASSED: Convergence rate {r} is within tolerance {tol}")
        except AssertionError as e:
            # Print method-specific failure message
            print(f"Method {method} FAILED: {e}")
        except Exception as e:
            # Catch any other unexpected exceptions
            print(f"An unexpected error occurred for {method}: {e}")


''' Error evaluation '''

# Error calculation function for each method
def calculate_errors(I, w, dt, T, method):
    u_numerical, t = solver(I, w, dt, T, method=method)  # Get the numerical solution
    u_exact_vals = u_exact(t, I, w)  # Get the exact solution at all time steps

    # Calculate local error (pointwise difference between numerical and exact solution)
    local_error = np.abs(u_numerical - u_exact_vals)

    
    # Calculate global error (suma of local error)
    global_error = np.sum(local_error)


    return local_error, global_error

# Function to calculate and display the errors for all methods
def errors(I, w, dt, T, methods):
    print("ERROR CALCULATION:")
    # Iterate over methods and compute errors
    for method in methods:
        local_error, global_error = calculate_errors(I, w, dt, T, method)
        
        
        print("Method {}, dt: {}:".format(method, dt))
        print("  Global Error = {:.6f}".format(global_error))
        print("  Local Errors for each time step: {} ...".format(local_error[1:4]))  # Print the first few local errors




if __name__ == "__main__":
   
    # Define Parameters
    I = 1
    w = 2 * pi
    num_periods = 5
    P = 2 * pi / w
    T = P * num_periods
    m = 5

    # Define the dt values
    dt_values = [0.01, 0.1]
    
    # Define the methods to test
    methods = ['CD', 'FE', 'BE', 'CN', 'RK2', 'RK4', 'EC']  #['CD', 'FE', 'BE', 'CN', 'RK2', 'RK4', 'EC'] 

    # Call the main function
    main(I, w, dt_values, T, methods)
    main2(I, w, dt_values, T, methods)
    
    # Test steps and errors
    test_three_steps(I, w, 0.01, T, methods)
    test_convergence_rates(I, w, 0.01, T, m, methods)
    errors(I, w, 0.01, T, methods)  # Use a specific dt for error testing
    