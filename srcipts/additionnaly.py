import numpy as np
import matplotlib.pyplot as plt


# --- Configuration and Constants ---
# Domain and simulation parameters from the homework description
L_DOMAIN = 2 * np.pi    # Domain size [m]
NU = 1.10555e-5         # Kinematic viscosity [m^2/s]
N_PENSILS = 48          # Total number of pencils (3 directions * 4x4 grid)
N_POINTS = 2**15        # Number of grid points per pencil
N_FILES = 16            # Number of files per direction

# Kolmogorov csts
C1 = 1.5
C2 = 2.1

# DEFAULT DATA PATH
DATA_PATH = './Data/'
RESULTS_PATH = '../Results/'

def fourth_order_derivative(f, dx):
    """Computes the derivative using a fourth-order central difference scheme."""
    # f_minus_2, f_minus_1, f_plus_1, f_plus_2
    f_m2 = np.roll(f, 2)
    f_m1 = np.roll(f, 1)
    f_p1 = np.roll(f, -1)
    f_p2 = np.roll(f, -2)
    df_dx = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * dx)
    return df_dx


def plot_one_pencil(pencils, i, j):
    """Plots the velocity components of a single pencil."""
    
    dx = np.linspace(0, L_DOMAIN, N_POINTS)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(dx, pencils['x'][i, :, j])
    axes[0].set_title('Pencil Data - u')
    axes[0].set_ylabel('Velocity')
    axes[0].grid()

    axes[1].plot(dx, pencils['y'][i, :, j])
    axes[1].set_title('Pencil Data - v')
    axes[1].set_ylabel('Velocity')
    axes[1].grid()

    axes[2].plot(dx, pencils['z'][i, :, j])
    axes[2].set_title('Pencil Data - w')
    axes[2].set_xlabel('Position [m]')
    axes[2].set_ylabel('Velocity')
    axes[2].grid()

    plt.tight_layout()
    plt.show()
