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


# --- Fonctions de traçage pour les spectres d'énergie ---

def plot_spectra_loglog(k_values, E11_avg, E22_avg, globals_dict, results_path):
    """Trace les spectres d'énergie normalisés en échelle log-log."""
    eta = globals_dict['eta']
    eps = globals_dict['eps']
    
    # Normalisation par les échelles de Kolmogorov
    k_eta = k_values * eta
    # Le facteur de normalisation (ε ν^5)^(1/4) est aussi u_η² η
    norm_factor = (eps * NU**5)**0.25
    E11_norm = E11_avg / norm_factor
    E22_norm = E22_avg / norm_factor

    plt.figure(figsize=(10, 6))
    plt.loglog(k_eta, E11_norm, label='$E_{11}(k_1)$ (Longitudinal)')
    plt.loglog(k_eta, E22_norm, label='$E_{22}(k_1)$ (Transverse)')
    
    # Ajout de la pente théorique en -5/3
    k_theory = np.array([1e-2, 1e-1])
    plt.loglog(k_theory, 0.5 * k_theory**(-5/3), 'k--', label='Théorie $k^{-5/3}$')
    
    plt.title("Spectres d'énergie 1D normalisés")
    plt.xlabel('$k_1 \\eta$')
    plt.ylabel('$E_{ii}(k_1) / (\\nu^5 \\epsilon)^{1/4}$')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.ylim(1e-5, 1e2)
    plt.savefig(os.path.join(results_path, 'slide_energy_spectra.png'))
    plt.close()
    print("Image sauvegardée : slide_energy_spectra.png")



def plot_spectra_comp(k_values, E11_avg, E22_avg, globals_dict, results_path):
    """Trace les spectres d'énergie compensés."""
    eta = globals_dict['eta']
    eps = globals_dict['eps']
    
    # Normalisation
    k_eta = k_values * eta
    norm_factor = (eps * NU**5)**0.25
    E11_norm = E11_avg / norm_factor
    E22_norm = E22_avg / norm_factor

    # Compensation
    compensated_E11 = E11_norm * k_eta**(5/3)
    compensated_E22 = E22_norm * k_eta**(5/3)

    plt.figure(figsize=(10, 6))
    plt.semilogx(k_eta, compensated_E11, label='Compensé $E_{11}$')
    plt.semilogx(k_eta, compensated_E22, label='Compensé $E_{22}$')
    
    # Constantes théoriques C1 et C1' = 4/3 * C1
    C1_val = 1.5 # (valeur souvent utilisée pour CK dans E(k)) * 18/55
    C1_prime_val = (4/3) * C1_val
    plt.axhline(C1_val, color='k', linestyle='--', label=f'$C_1 \\approx {C1_val:.2f}$')
    plt.axhline(C1_prime_val, color='k', linestyle='-.', label=f"$C'_1 \\approx {C1_prime_val:.2f}$")

    plt.title("Spectres d'énergie 1D compensés")
    plt.xlabel('$k_1 \\eta$')
    plt.ylabel('$(k_1\\eta)^{5/3} E_{ii} / (\\nu^5\\epsilon)^{1/4}$')
    plt.ylim(0, 3)
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'slide_compensated_spectra.png'))
    plt.close()
    print("Image sauvegardée : slide_compensated_spectra.png")

# --- Fonction de traçage pour l'autocorrélation ---

def plot_autocorrelation(r_values, f_r, g_r, eta, results_path):
    """Trace les fonctions d'autocorrélation f(r) et g(r)."""
    r_over_eta = r_values / eta
    
    plt.figure(figsize=(10, 6))
    plt.plot(r_over_eta, f_r, label='$f(r)$ (Longitudinal)')
    plt.plot(r_over_eta, g_r, label='$g(r)$ (Transverse)')
    
    plt.title("Fonctions d'autocorrélation")
    plt.xlabel('$r / \\eta$')
    plt.ylabel('Corrélation')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    # On se concentre sur les petites valeurs de r pour voir la parabole
    plt.xlim(0, r_over_eta[len(r_over_eta)//10]) 
    plt.ylim(min(np.min(f_r), np.min(g_r)), 1.05)
    plt.savefig(os.path.join(results_path, 'slide_autocorrelation.png'))
    plt.close()
    print("Image sauvegardée : slide_autocorrelation.png")