
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

from additionnaly import fourth_order_derivative, plot_sf_loglog, plot_sf_comp


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


    
# --- 5. Exécution principale ---

if __name__ == "__main__":
    pencils = load_pencils(DATA_PATH)
    globals_dict = calculate_global_quantities(pencils)
    r_vals, D11, D22 = calculate_structure_functions(pencils, globals_dict)

    # les plots
    plot_sf_loglog(r_vals, D11, D22, globals_dict['eta'], RESULTS_PATH)                         # Log-Log
    plot_sf_comp(r_vals, D11, D22, globals_dict['eps'], globals_dict['eta'], RESULTS_PATH)      # Compensé 

    print("Analyse terminée.")