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
C1 = 1.6 * (18/55)  # C1 constant for 1D energy spectrum
C2 = 2.1            # C2 constant for 2nd order structure function

# DEFAULT DATA PATH
DATA_PATH = './Data/'
RESULTS_PATH = '../Results/'


# --- Schémas de différences finies ---

def fourth_order_derivative(f, dx):
    """Computes the derivative using a fourth-order central difference scheme."""
    # f_minus_2, f_minus_1, f_plus_1, f_plus_2
    f_m2 = np.roll(f, 2)
    f_m1 = np.roll(f, 1)
    f_p1 = np.roll(f, -1)
    f_p2 = np.roll(f, -2)
    df_dx = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * dx)
    return df_dx


# --- Fonctions de traçage pour les fonctions de structure en échelle log-log ---

def plot_sf_loglog(r_values, D11_avg, D22_avg, eta, results_path):
    # Limit to r/eta <= 5e3 and avoid zero/NaN values for log plotting
    r_over_eta = r_values / eta
    mask = np.isfinite(r_over_eta) & np.isfinite(D11_avg) & np.isfinite(D22_avg) & (r_over_eta > 0)
    mask &= (r_over_eta <= 5e3)

    if not np.any(mask):
        # fallback to all valid points if mask empty
        mask = np.isfinite(r_over_eta) & np.isfinite(D11_avg) & np.isfinite(D22_avg) & (r_over_eta > 0)

    r_plot = r_over_eta[mask]
    D11_plot = D11_avg[mask]
    D22_plot = D22_avg[mask]

    # avoid exact zeros (log scale)
    tiny = 1e-30
    D11_plot = np.maximum(D11_plot, tiny)
    D22_plot = np.maximum(D22_plot, tiny)

    plt.figure(figsize=(10, 6))
    plt.loglog(r_plot, D11_plot, label='$D_{11}(r)$ (Longitudinal)')
    plt.loglog(r_plot, D22_plot, label='$D_{22}(r)$ (Transverse)')

    # theory line spanning plotted range, scaled to median of D11 in the plotted interval
    if len(r_plot) > 0 and np.all(r_plot > 0):
        r_theory = np.logspace(np.log10(r_plot.min()), np.log10(r_plot.max()), 2)
        scale = np.median(D11_plot / (r_plot**(2/3)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        plt.loglog(r_theory, scale * r_theory**(2/3), 'k--', label='Theory $r^{2/3}$')

    plt.title('Second-Order Structure Functions')
    plt.xlabel('$r / \\eta$')
    plt.ylabel('$D_{ii}(r)$')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'sf_loglog.png'))
    plt.close()


# --- Fonctions de traçage pour les fonctions de structure compensées ---

def plot_sf_comp(r_values, D11_avg, D22_avg, eps, eta, results_path):

    # prepare r/eta and mask invalid entries and limit to r/eta <= 5e3
    r_over_eta = r_values / eta
    mask = np.isfinite(r_over_eta) & np.isfinite(D11_avg) & np.isfinite(D22_avg) & (r_over_eta > 0)
    mask &= (r_over_eta <= 5e3)
    if not np.any(mask):
        mask = np.isfinite(r_over_eta) & np.isfinite(D11_avg) & np.isfinite(D22_avg) & (r_over_eta > 0)

    r_plot = r_over_eta[mask]
    r_phys = r_values[mask]
    D11_plot = D11_avg[mask]
    D22_plot = D22_avg[mask]

    # avoid zero division in (eps * r)^(2/3)
    tiny = 1e-30
    denom = np.maximum(eps * r_phys, tiny)
    compensated_D11 = D11_plot / denom**(2/3)
    compensated_D22 = D22_plot / denom**(2/3)

    # choose a plausible "pseudo-plateau" window in r/eta (adjust if needed)
    plateau_mask = (r_plot >= 100) & (r_plot <= 1000)
    if not np.any(plateau_mask):
        # fallback to a broader window if the chosen one is empty
        plateau_mask = (r_plot >= 1) & (r_plot <= 1e3)

    # median values in the plateau for comparison
    measured_C2 = np.median(compensated_D11[plateau_mask]) if np.any(plateau_mask) else np.nan
    measured_C2p = np.median(compensated_D22[plateau_mask]) if np.any(plateau_mask) else np.nan

    plt.figure(figsize=(10, 6))
    plt.loglog(r_plot, compensated_D11, label='Compensated $D_{11}$')
    plt.loglog(r_plot, compensated_D22, label='Compensated $D_{22}$')

    # theoretical lines
    plt.axhline(C2, color='blue', linestyle='--', label=f'$C_2 = 2.1$')
    plt.axhline((4/3) * C2, color='orange', linestyle='--', label=f"C'_2 ≈ 2.8")

    # measured plateau lines
    if np.isfinite(measured_C2):
        plt.axhline(measured_C2, color='blue', linestyle=':', label=f'Measured $C_2^{{meas}}={measured_C2:.2f}$')
    if np.isfinite(measured_C2p):
        plt.axhline(measured_C2p, color='orange', linestyle=':', label=f"Measured $C_2^{{\\prime,meas}}={measured_C2p:.2f}$")

    # annotate the plateau window on the x-axis (fixed fill_betweenx usage)
    ymax = max(4, np.nanmax([measured_C2, measured_C2p, C2, (4/3)*C2]) + 0.5)
    if np.any(plateau_mask):
        x1 = np.min(r_plot[plateau_mask])
        x2 = np.max(r_plot[plateau_mask])
    else:
        x1, x2 = 1, 10
    plt.fill_betweenx([0, ymax], x1, x2, color='grey', alpha=0.1, label='Pseudo-plateau window')

    plt.title('Compensated Structure Functions')
    plt.xlabel('$r / \\eta$')
    plt.ylabel('$(\\epsilon r)^{-2/3} D_{ii}(r)$')
    plt.ylim(0, ymax)
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'sf_comp.png'))
    plt.close()

    # print measured plateau values for quick numeric comparison
    print(f"Measured C2 (longitudinal, plateau)      = {measured_C2:.4f}")
    print(f"Measured C2' (transverse, plateau)       = {measured_C2p:.4f}")



# --- Fonctions de traçage pour les spectres d'énergie ---

def plot_spectra_loglog(k_values, E11_avg, E22_avg, globals_dict, results_path):
    """Trace les spectres d'énergie normalisés en échelle log-log."""
    eta = globals_dict['eta']
    eps = globals_dict['eps']

    # normalisation et filtrage
    k_eta = k_values * eta
    mask = np.isfinite(k_eta) & np.isfinite(E11_avg) & np.isfinite(E22_avg) & (k_eta > 0)
    if not np.any(mask):
        mask = np.isfinite(k_eta) & (k_eta > 0)
    k_plot = k_eta[mask]
    E11_plot = E11_avg[mask]
    E22_plot = E22_avg[mask]

    norm_factor = (eps * NU**5)**0.25
    E11_norm = E11_plot / norm_factor
    E22_norm = E22_plot / norm_factor

    # ensure output directory exists
    os.makedirs(results_path, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.loglog(k_plot, E11_norm, label='E₁₁(k₁) / (ν⁵ ε)^(1/4)')
    plt.loglog(k_plot, E22_norm, label='E₂₂(k₁) / (ν⁵ ε)^(1/4)')

    # théorie : droites en -5/3 couvrant la plage tracée, amplitudes C1 et C1'
    if len(k_plot) > 0:
        k_theory = np.logspace(np.log10(k_plot.min()), np.log10(k_plot.max()), 100)
        C1_val = C1  # utilise la constante définie en haut du fichier
        C1p_val = (4.0 / 3.0) * C1_val
               
        plt.loglog(k_theory, C1_val * k_theory**(-5/3), 'k--', label=f'C₁ (k₁η)^(-5/3), C₁={C1_val:.2f}')
        plt.loglog(k_theory, C1p_val * k_theory**(-5/3), 'k-.', label=f'C₁\' (k₁η)^(-5/3), C₁\'={C1p_val:.2f}')

    plt.title("Normalized 1D Energy Spectra")
    plt.xlabel('$k_1 \\eta$')
    plt.ylabel('$E_{ii}(k_1) / (\\nu^5 \\epsilon)^{1/4}$')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.ylim(bottom=1e-8)  # éviter log plot crash si valeurs très petites
    plt.savefig(os.path.join(results_path, 'slide_energy_spectra.png'))
    plt.close()
    print("Image sauvegardée : slide_energy_spectra.png")



def plot_spectra_comp(k_values, E11_avg, E22_avg, globals_dict, results_path):
    """Trace les spectres d'énergie compensés."""
    eta = globals_dict['eta']
    eps = globals_dict['eps']
    
    # Normalisation et filtrage
    k_eta = k_values * eta
    mask = np.isfinite(k_eta) & np.isfinite(E11_avg) & np.isfinite(E22_avg) & (k_eta > 0)
    if not np.any(mask):
        mask = np.isfinite(k_eta) & (k_eta > 0)
    k_plot = k_eta[mask]
    E11_plot = E11_avg[mask]
    E22_plot = E22_avg[mask]

    # normalisation Kolmogorov
    norm_factor = (eps * NU**5)**0.25
    E11_norm = E11_plot / norm_factor
    E22_norm = E22_plot / norm_factor

    # Compensation (kη)^{5/3} · E / (εν^5)^{1/4}
    compensated_E11 = E11_norm * k_plot**(5/3)
    compensated_E22 = E22_norm * k_plot**(5/3)

    # choix d'une fenêtre pseudo-plateau en kη (ajuster si nécessaire)
    plateau_mask = (k_plot >= 1e-3) & (k_plot <= 1e-2)

    if not np.any(plateau_mask):
        plateau_mask = (k_plot >= np.percentile(k_plot, 5)) & (k_plot <= np.percentile(k_plot, 95))

    measured_C1 = np.median(compensated_E11[plateau_mask]) if np.any(plateau_mask) else np.nan
    measured_C1p = np.median(compensated_E22[plateau_mask]) if np.any(plateau_mask) else np.nan

    # constantes théoriques
    C1_val = C1
    C1p_val = (4.0 / 3.0) * C1_val

    # ensure output directory exists
    os.makedirs(results_path, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.loglog(k_plot, compensated_E11, label='Compensated $E_{11}$')
    plt.loglog(k_plot, compensated_E22, label='Compensated $E_{22}$')

    # lignes théoriques & mesurées
    plt.axhline(C1_val, color='k', linestyle='--', label=f'$C_1={C1_val:.2f}$')
    plt.axhline(C1p_val, color='k', linestyle='-.', label=f"$C'_1={C1p_val:.2f}$")
    if np.isfinite(measured_C1):
        plt.axhline(measured_C1, color='blue', linestyle=':', label=f'Measured $C_1^{{meas}}={measured_C1:.2f}$')
    if np.isfinite(measured_C1p):
        plt.axhline(measured_C1p, color='orange', linestyle=':', label=f"Measured $C_1^{{\\prime,meas}}={measured_C1p:.2f}$")

    # annoter la fenêtre pseudo-plateau
    if np.any(plateau_mask):
        x1 = np.min(k_plot[plateau_mask])
        x2 = np.max(k_plot[plateau_mask])
    else:
        x1, x2 = (k_plot.min() if len(k_plot)>0 else 1e-2), (k_plot.max() if len(k_plot)>0 else 1e1)
    # déterminer ymax de façon robuste (ne pas dépasser inutilement)
    combined_max = np.nan
    try:
        combined_max = np.nanmax(np.concatenate([compensated_E11, compensated_E22]))
    except Exception:
        combined_max = np.nanmax(compensated_E11) if compensated_E11.size>0 else np.nan
    ymax = max(1.6, combined_max + 0.05 if np.isfinite(combined_max) else 1.6)

    plt.fill_betweenx([0, ymax], x1, x2, color='grey', alpha=0.12, label='Pseudo-plateau window')

    plt.title("Compensated 1D Energy Spectra")
    plt.xlabel('$k_1 \\eta$')
    plt.ylabel('$(k_1\\eta)^{5/3} E_{ii} / (\\nu^5\\epsilon)^{1/4}$')
    plt.ylim(0, ymax)
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'slide_compensated_spectra.png'))
    plt.close()
    print(f"Measured C1  = {measured_C1:.4f}")
    print(f"Measured C1' = {measured_C1p:.4f}")
    print("Image sauvegardée : slide_compensated_spectra.png")



# --- Fonction de traçage pour l'autocorrélation ---

def plot_autocorrelation(r_values, f_r, g_r, eta, results_path):
    """Trace les fonctions d'autocorrélation f(r) et g(r) pour r/η ≤ 5e2."""
    # prepare r/eta and mask invalid entries and limit to r/eta <= 5e2
    r_over_eta = r_values / eta
    mask = np.isfinite(r_over_eta) & np.isfinite(f_r) & np.isfinite(g_r) & (r_over_eta >= 0)
    mask &= (r_over_eta <= 5e2)

    if not np.any(mask):
        # fallback to all finite values if strict window empty
        mask = np.isfinite(r_over_eta) & np.isfinite(f_r) & np.isfinite(g_r)

    r_plot = r_over_eta[mask]
    f_plot = f_r[mask]
    g_plot = g_r[mask]

    # ensure output directory exists
    os.makedirs(results_path, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(r_plot, f_plot, label='$f(r)$ (Longitudinal)')
    plt.plot(r_plot, g_plot, label='$g(r)$ (Transverse)')

    plt.title("Autocorrelation Functions")
    plt.xlabel('$r / \\eta$')
    plt.ylabel('Correlation')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()

    # focus on r/eta up to 5e2 (or maximum available)
    xmax = min(5e2, r_plot.max()) if len(r_plot) > 0 else 5e2
    plt.xlim(0, xmax)

    # set sensible y-limits: from min data (capped at -1) to slightly above 1
    ymin = min(np.nanmin(f_plot), np.nanmin(g_plot))
    ymin = max(ymin, -1.0)
    plt.ylim(ymin, 1.05)

    plt.savefig(os.path.join(results_path, 'slide_autocorrelation.png'))
    plt.close()
    print("Image sauvegardée : slide_autocorrelation.png")