import numpy as np
import matplotlib.pyplot as plt
import os


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


def plot_sf_loglog(r_values, D11_avg, D22_avg, eta, results_path):
    r_over_eta = r_values / eta
    plt.figure(figsize=(10, 6))
    plt.loglog(r_over_eta, D11_avg, label='$D_{11}(r)$ (Longitudinal)')
    plt.loglog(r_over_eta, D22_avg, label='$D_{22}(r)$ (Transverse)')
    r_theory = np.array([1e2, 1e3])
    plt.loglog(r_theory, 5 * r_theory**(2/3), 'k--', label='Théorie $r^{2/3}$')
    plt.title('Fonctions de structure')
    plt.xlabel('$r / \\eta$')
    plt.ylabel('$D_{ii}(r)$')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'sf_loglog.png'))
    plt.close()

def plot_sf_comp(r_values, D11_avg, D22_avg, eps, eta, results_path):
    r_over_eta = r_values / eta
    compensated_D11 = D11_avg / (eps * r_values)**(2/3)
    compensated_D22 = D22_avg / (eps * r_values)**(2/3)
    plt.figure(figsize=(10, 6))
    plt.semilogx(r_over_eta, compensated_D11, label='Comp $D_{11}$')
    plt.semilogx(r_over_eta, compensated_D22, label='Comp $D_{22}$')
    plt.axhline(2.1, color='k', linestyle='--', label='$C_2 = 2.1$')
    plt.title('Fonctions de structure compensées')
    plt.xlabel('$r / \\eta$')
    plt.ylabel('$(\\epsilon r)^{-2/3} D_{ii}(r)$')
    plt.ylim(0, 4)
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'sf_comp.png'))
    plt.close()