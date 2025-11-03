
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

from additionnaly import fourth_order_derivative, plot_sf_loglog, plot_sf_comp, plot_spectra_loglog, plot_spectra_comp, plot_autocorrelation

# --- 1. Prépa et constantes ---

L_DOMAIN = 2 * np.pi                                # Taille du domaine [m]
NU = 1.10555e-5                                     # Viscosité cinématique [m^2/s]
N_POINTS = 2**15                                    # Nombre de points par "pencil"
N_FILES_PER_DIR = 16                                # Grille 4x4 = 16 pencils par direction
DIRECTIONS = ['x', 'y', 'z']                        # Directions des pencils
TOTAL_PENCILS = len(DIRECTIONS) * N_FILES_PER_DIR   # 3 * 16 = 48 pencils

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data'))
RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Results'))



# --- 2. Chargement des données ---
def load_pencils(path=None):
    """Charge tous les fichiers de données de velocity pencils."""
    if path is None:
        path = DATA_PATH

    pencils = {}
    indexs = [f"{i}_{j}" for i in range(4) for j in range(4)]  # '0_0' à '3_3'

    for d in DIRECTIONS:
        folder = os.path.join(path, f'pencils_{d}')
        pencils[d] = np.zeros((N_FILES_PER_DIR, N_POINTS, 3))

        for i, idx in enumerate(indexs):
            file_path = os.path.join(folder, f'{d}_{idx}.txt')
            print(f'Loading file: {file_path}')
            data = np.loadtxt(file_path)
            pencils[d][i, :, :] = data

    return pencils



# --- 3. Quantités globales ---
# k, epsilon
# on calcule pour chaque pencil, puis faire la moyenne à la fin

def calculate_global_quantities(pencils):
    """Calcule toutes les quantités globales en moyennant sur les 48 pencils."""
    print("--- Tâche 1 : Calcul des quantités globales ---")
    
    dx = L_DOMAIN / N_POINTS # pas spatial pour un pencil
    
    # listes pour stocker k et epsilon de chaque pencil
    k_from_each_pencil = []
    epsilon_from_each_pencil = []

    for direction in DIRECTIONS:
        for i in range(N_FILES_PER_DIR):
            u, v, w = pencils[direction][i, :, 0], pencils[direction][i, :, 1], pencils[direction][i, :, 2] # compo de vitesse
            
            # --- Calcul de TKE pour ce pencil ---
            # On enlève la moyenne pour avoir les fluctuations u', v', w'
            u_fluc = u - np.mean(u)
            v_fluc = v - np.mean(v)
            w_fluc = w - np.mean(w)
            
            # k = 0.5 * (<u'u'> + <v'v'> + <w'w'>)
            k_pencil = 0.5 * (np.mean(u_fluc**2) + np.mean(v_fluc**2) + np.mean(w_fluc**2))
            k_from_each_pencil.append(k_pencil)
            
            # --- Calcul de la dissipation pour ce pencil ---
            # Formule ε = 15ν <(∂u/∂x)²> ok pour HIT
            if direction == 'x': longitudinal_fluc = u_fluc
            if direction == 'y': longitudinal_fluc = v_fluc
            if direction == 'z': longitudinal_fluc = w_fluc
            
            # Dérivée avec schéma d'ordre 4 
            du_dx = fourth_order_derivative(longitudinal_fluc, dx)
        
            epsilon_pencil = 15 * NU * np.mean(du_dx**2)
            epsilon_from_each_pencil.append(epsilon_pencil)


    # --- Moyenne sur tous les pencils ---
    k_mean = np.mean(k_from_each_pencil)
    epsilon_mean = np.mean(epsilon_from_each_pencil)


    # --- Quantités dérivées ---
    u_rms = np.sqrt(2.0 * k_mean / 3.0)                         # Vitesse RMS
    L_integral = k_mean**1.5 / epsilon_mean                     # Échelle intégrale
    eta = (NU**3 / epsilon_mean)**0.25                          # Échelle de Kolmogorov
    lambda_taylor = np.sqrt(10 * NU * k_mean / epsilon_mean)    # Micro-échelle de Taylor
    Re_lambda = u_rms * lambda_taylor / NU                      # Nombre de Reynolds de Taylor
    Re = k_mean**2 / (NU * epsilon_mean)                        # Nombre de Reynolds global


    # --- Résumé un peu plus propre ---
    print("\n--- Résultats des quantités globales moyennées ---")
    print(f"  Énergie cinétique turbulente (k) = {k_mean:.4f} m²/s²")
    print(f"  Taux de dissipation (ε)         = {epsilon_mean:.4f} m²/s³")
    print("-" * 45)
    print(f"  Échelle de Kolmogorov (η)      = {eta * 1000:.4f} mm")
    print(f"  Micro-échelle de Taylor (λ)     = {lambda_taylor * 1000:.4f} mm")
    print(f"  Échelle intégrale (L)           = {L_integral:.4f} m")
    print("-" * 45)
    print(f"  Nombre de Reynolds de Taylor (Re_λ) = {Re_lambda:.2f}")
    print(f"  Nombre de Reynolds global (Re)       = {Re:.2f}")
    print("-" * 45)
    
    # Return les résultats sous forme d'un dictionnaire
    return {'k': k_mean, 'eps': epsilon_mean, 'eta': eta, 'lambda': lambda_taylor}





# --- 4. Fonctions de structure ---
# Calcul de D11 et D22 (pour chaque pencil puis on fait lamoyenne)

def calculate_structure_functions(pencils, globals):
    """Calcule et trace les fonctions de structure du 2ème ordre."""
    print("\n--- Tâche 2 : Calcul des fonctions de structure ---")
    
    eps = globals['eps']
    eta = globals['eta']
    dx = L_DOMAIN / N_POINTS

    max_shift = N_POINTS // 2
    r_values = np.arange(1, max_shift) * dx
    
    D11_accumulator = np.zeros_like(r_values)
    D22_accumulator = np.zeros_like(r_values)
    
    for direction in DIRECTIONS:
        if direction == 'x': long_idx, trans_idx1, trans_idx2 = 0, 1, 2
        if direction == 'y': long_idx, trans_idx1, trans_idx2 = 1, 0, 2
        if direction == 'z': long_idx, trans_idx1, trans_idx2 = 2, 1, 0

        for i in range(N_FILES_PER_DIR):
            u_fluc = pencils[direction][i, :, long_idx] - np.mean(pencils[direction][i, :, long_idx])
            v_fluc = pencils[direction][i, :, trans_idx1] - np.mean(pencils[direction][i, :, trans_idx1])
            w_fluc = pencils[direction][i, :, trans_idx2] - np.mean(pencils[direction][i, :, trans_idx2])
            
            for shift_idx, shift in enumerate(range(1, max_shift)):
                # D11 : longitudinal (composante parallèle)
                diff_u = u_fluc - np.roll(u_fluc, -shift)
                D11_accumulator[shift_idx] += np.mean(diff_u**2)
                
                # D22 : transverse (composantes perpendiculaires)
                diff_v = v_fluc - np.roll(v_fluc, -shift)
                diff_w = w_fluc - np.roll(w_fluc, -shift)
                D22_accumulator[shift_idx] += np.mean(diff_v**2) + np.mean(diff_w**2)

    # Moyenne finale
    D11_avg = D11_accumulator / TOTAL_PENCILS
    D22_avg = D22_accumulator / (TOTAL_PENCILS * 2)

    return r_values, D11_avg, D22_avg




# --- 5. Spectres d'énergie 1D ---

def calculate_energy_spectra(pencils, globals):
    """Calcule les spectres d'énergie 1D E11 et E22."""
    print("\n--- Tâche 3 : Calcul des spectres d'énergie ---")
    
    dx = L_DOMAIN / N_POINTS
    
    # Préparation des nombres d'onde (k)
    k = fftfreq(N_POINTS, d=dx) * 2 * np.pi
    k_pos = k[1:N_POINTS // 2]  # on garde que les k positifs

    # Accumulateurs pour les spectres
    E11_accumulator = np.zeros_like(k_pos)
    E22_accumulator = np.zeros_like(k_pos)

    for direction in DIRECTIONS:
        if direction == 'x': long_idx, trans_idx1, trans_idx2 = 0, 1, 2
        if direction == 'y': long_idx, trans_idx1, trans_idx2 = 1, 0, 2
        if direction == 'z': long_idx, trans_idx1, trans_idx2 = 2, 1, 0

        for i in range(N_FILES_PER_DIR):
            # On prend les fluctuations
            u_fluc = pencils[direction][i, :, long_idx] - np.mean(pencils[direction][i, :, long_idx])
            v_fluc = pencils[direction][i, :, trans_idx1] - np.mean(pencils[direction][i, :, trans_idx1])
            w_fluc = pencils[direction][i, :, trans_idx2] - np.mean(pencils[direction][i, :, trans_idx2])

            # Transformée de fourier (FFT)
            u_hat = fft(u_fluc)[1:N_POINTS // 2]
            v_hat = fft(v_fluc)[1:N_POINTS // 2]
            w_hat = fft(w_fluc)[1:N_POINTS // 2]
            
            # Calcul du spectre de puissance (PSD)
            # La normalisation L_DOMAIN / (pi * N_POINTS^2) c pour que l'intégrale du spectre donne la variance
            E11_pencil = (L_DOMAIN / (np.pi * N_POINTS**2)) * np.abs(u_hat)**2
            E22_pencil = (L_DOMAIN / (np.pi * N_POINTS**2)) * (np.abs(v_hat)**2 + np.abs(w_hat)**2)
            
            E11_accumulator += E11_pencil
            E22_accumulator += E22_pencil
            
    # Moyenne finale
    E11_avg = E11_accumulator / TOTAL_PENCILS
    E22_avg = E22_accumulator / (TOTAL_PENCILS * 2) # pcq 2 composantes transverses par pencil

    return k_pos, E11_avg, E22_avg


# --- 6. Fonctions d'autocorrélation (BONUS) ---

def calculate_autocorrelation_functions(pencils, globals):
    """Calcule les fonctions d'autocorrélation f(r) et g(r) et les échelles de Taylor."""
    print("\n--- Tâche BONUS : Calcul des fonctions d'autocorrélation ---")
    
    dx = L_DOMAIN / N_POINTS
    max_shift = N_POINTS // 2
    r_values = np.arange(1, max_shift) * dx
    
    # Accumulateurs
    R11_accumulator = np.zeros_like(r_values)
    R22_accumulator = np.zeros_like(r_values)
    variance_u_accumulator = 0
    variance_v_w_accumulator = 0

    for direction in DIRECTIONS:
        if direction == 'x': long_idx, trans_idx1, trans_idx2 = 0, 1, 2
        if direction == 'y': long_idx, trans_idx1, trans_idx2 = 1, 0, 2
        if direction == 'z': long_idx, trans_idx1, trans_idx2 = 2, 1, 0

        for i in range(N_FILES_PER_DIR):
            u_fluc = pencils[direction][i, :, long_idx] - np.mean(pencils[direction][i, :, long_idx])
            v_fluc = pencils[direction][i, :, trans_idx1] - np.mean(pencils[direction][i, :, trans_idx1])
            w_fluc = pencils[direction][i, :, trans_idx2] - np.mean(pencils[direction][i, :, trans_idx2])
            
            # On accumule la variance pour la normalisation finale
            variance_u_accumulator += np.mean(u_fluc**2)
            variance_v_w_accumulator += np.mean(v_fluc**2) + np.mean(w_fluc**2)
            
            # Calcul de R(r) = <u'(x) * u'(x+r)>
            for shift_idx, shift in enumerate(range(1, max_shift)):
                R11_accumulator[shift_idx] += np.mean(u_fluc * np.roll(u_fluc, -shift))
                R22_accumulator[shift_idx] += np.mean(v_fluc * np.roll(v_fluc, -shift)) + \
                                              np.mean(w_fluc * np.roll(w_fluc, -shift))

    # Moyenne et normalisation
    variance_u_avg = variance_u_accumulator / TOTAL_PENCILS
    variance_v_w_avg = variance_v_w_accumulator / (TOTAL_PENCILS * 2)
    
    f_r = R11_accumulator / (TOTAL_PENCILS * variance_u_avg)
    g_r = R22_accumulator / (TOTAL_PENCILS * 2 * variance_v_w_avg)
    
    # Calcul des échelles de Taylor avec la parabole osculatrice
    # f(r) ≈ 1 - r² / λ_f²  => λ_f² ≈ -r² / (f(r) - 1)
    # On utilise le deuxième point (le premier étant r=dx) pour une meilleure stabilité
    r2_sq = r_values[1]**2
    lambda_f_sq = -r2_sq / (f_r[1] - 1)
    lambda_g_sq = -r2_sq / (g_r[1] - 1)
    lambda_f = np.sqrt(lambda_f_sq)
    lambda_g = np.sqrt(lambda_g_sq)
    
    print("\n--- Micro-échelles de Taylor (via autocorrélation) ---")
    print(f"  Échelle longitudinale (λ_f) = {lambda_f * 1000:.4f} mm")
    print(f"  Échelle transverse (λ_g)    = {lambda_g * 1000:.4f} mm")
    # Vérification théorique : λ_f² devrait être ≈ 2 * λ_g²
    print(f"  Vérification : λ_f² / λ_g² = {lambda_f_sq / lambda_g_sq:.2f} (théorie = 2.0)")

    return r_values, f_r, g_r





    
# --- 7. Exécution principale ---

if __name__ == "__main__":
    pencils = load_pencils(DATA_PATH)
    globals_dict = calculate_global_quantities(pencils)
    
    # Tâche 2
    r_vals, D11, D22 = calculate_structure_functions(pencils, globals_dict)
    plot_sf_loglog(r_vals, D11, D22, globals_dict['eta'], RESULTS_PATH)                         # Log-Log
    plot_sf_comp(r_vals, D11, D22, globals_dict['eps'], globals_dict['eta'], RESULTS_PATH)      # Compensé 
    
    # Tâche 3
    k_vals, E11, E22 = calculate_energy_spectra(pencils, globals_dict)
    plot_spectra_loglog(k_vals, E11, E22, globals_dict, RESULTS_PATH)
    plot_spectra_comp(k_vals, E11, E22, globals_dict, RESULTS_PATH)
    
    # Tâche BONUS
    r_vals_corr, f_r, g_r = calculate_autocorrelation_functions(pencils, globals_dict)
    plot_autocorrelation(r_vals_corr, f_r, g_r, globals_dict['eta'], RESULTS_PATH)

    print("\nAnalyse complète terminée")
    # plt.show()