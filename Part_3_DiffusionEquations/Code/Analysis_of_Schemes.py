import numpy as np
import matplotlib.pyplot as plt
import os
import math

def plot_amplification_factors(F, p_range, save_dir):
    """
    Compute and plot amplification factors for different methods, saving results.

    Parameters:
    - F: Fourier number (float or list of floats).
    - p_range: Array of p values (numpy array).
    - save_dir: Directory to save the plots.
    """
    # Define the exact amplification factor
    def A_exact(F, p):
        return np.exp(-4 * F * p**2)

    # Define the Forward Euler amplification factor
    def A_FE(F, p):
        return 1 - 4 * F * np.sin(p)**2

    # Define the Backward Euler amplification factor
    def A_BE(F, p):
        return 1 / (1 + 4 * F * np.sin(p)**2)

    # Define the Crank-Nicolson amplification factor
    def A_CN(F, p):
        return (1 - 2 * F * np.sin(p)**2) / (1 + 2 * F * np.sin(p)**2)

    def plot_single(F_val, p_range, save_path):
        """
        Plot amplification factors for a single Fourier number.
        """
        # Compute amplification factors
        A_exact_vals = A_exact(F_val, p_range)
        A_FE_vals = A_FE(F_val, p_range)
        A_BE_vals = A_BE(F_val, p_range)
        A_CN_vals = A_CN(F_val, p_range)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(p_range, A_exact_vals, label="Exact Amplification", linestyle="-", linewidth=2)
        plt.plot(p_range, A_FE_vals, label="Forward Euler", linestyle="-.", linewidth=2)
        plt.plot(p_range, A_BE_vals, label="Backward Euler", linestyle="-.", linewidth=2)
        plt.plot(p_range, A_CN_vals, label="Crank-Nicolson", linestyle="-.", linewidth=2)

        # Add labels, legend, and title
        plt.xlabel("p = kΔx / 2", fontsize=12)
        plt.ylabel("Amplification Factor A(p)", fontsize=12)
        plt.title(f"Amplification Factors for F = {F_val}", fontsize=14)
        plt.legend(fontsize=12)
        plt.xlim(0, 1.5)

        # Adjust y-limits based on F
        if 0.1 <= F_val <= 0.25:
            plt.ylim(0, 1.1)
        elif F_val < 0.1:
            plt.ylim(0.9, 1.1)
        else:
            plt.ylim(-1.1, 1.1)

        plt.grid()
        plt.savefig(save_path, dpi=300, transparent=True)  # Save without background
        plt.close()  # Close the plot to free memory

    def plot_multiple(F_vals, p_range, save_dir):
        """
        Plot amplification factors for multiple Fourier numbers with subplots in two columns.
        """
        num_subplots = len(F_vals)
        num_columns = 2
        num_rows = math.ceil(num_subplots / num_columns)

        fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 6 * num_rows), sharex=True)

        axs = axs.flatten()  # Flatten for easy iteration

        for i, F_val in enumerate(F_vals):
            # Compute amplification factors
            A_exact_vals = A_exact(F_val, p_range)
            A_FE_vals = A_FE(F_val, p_range)
            A_BE_vals = A_BE(F_val, p_range)
            A_CN_vals = A_CN(F_val, p_range)

            # Plot
            axs[i].plot(p_range, A_exact_vals, label="Exact Amplification", linestyle="-", linewidth=2)
            axs[i].plot(p_range, A_FE_vals, label="Forward Euler", linestyle="-.", linewidth=2)
            axs[i].plot(p_range, A_BE_vals, label="Backward Euler", linestyle="-.", linewidth=2)
            axs[i].plot(p_range, A_CN_vals, label="Crank-Nicolson", linestyle="-.", linewidth=2)

            # Add labels, legend, and title
            axs[i].set_title(f"F = {F_val}", fontsize=12)
            axs[i].legend(fontsize=8)
            axs[i].grid()
            axs[i].set_xlim(0, max(p_range))

            # Adjust y-limits based on F
            if 0.1 <= F_val <= 0.25:
                axs[i].set_ylim(0, 1.1)
            elif F_val < 0.1:
                axs[i].set_ylim(0.9, 1.0)
            else:
                axs[i].set_ylim(-1.1, 1.1)

        # Handle empty subplots
        for j in range(num_subplots, len(axs)):
            fig.delaxes(axs[j])

        fig.supylabel("Amplification Factor A(p)", fontsize=14)
        fig.supxlabel("p = kΔx / 2", fontsize=14)
        fig.suptitle("Amplification Factors for Multiple Fourier Numbers", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plot_path = os.path.join(save_dir, "amplification_factors_multiple_F.png")
        plt.savefig(plot_path, dpi=300, transparent=True)  # Save without background
        plt.close()  # Close the plot to free memory

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Check if F is a single value or a list
    if isinstance(F, (int, float)):
        save_path = os.path.join(save_dir, f"amplification_factors_F_{F}.png")
        plot_single(F, p_range, save_path)
    elif isinstance(F, (list, np.ndarray)):
        plot_multiple(F, p_range, save_dir)
    else:
        raise ValueError("F must be a single number or a list of numbers.")



if __name__ == "__main__":

    F_single = 100  # Single Fourier number
    F_multiple = [0.01, 0.1, 0.5, 0.8, 2.0,  20.0]  # Multiple Fourier numbers
    p_range = np.linspace(0, np.pi, 100)  # Range of p values
    save_dir = r'C:\Users\pbosn\OneDrive - USN\PhD WorkFlow\WorkFlow\Courses\Required Courses\Numerical Solutions for PDE\3_Diffusion_Equations\DevelopingCode\Analysis_of_Schemes'

    # Call with a single F
    plot_amplification_factors(F_single, p_range, save_dir)

    # Call with multiple F values
    # plot_amplification_factors(F_multiple, p_range, save_dir)
