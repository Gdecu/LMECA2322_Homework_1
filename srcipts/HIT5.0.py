# -*- coding: utf-8 -*-

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
            # Les champs fournis sont déjà les fluctuations : pas de soustraction de la moyenne
            k_pencil = 0.5 * (np.mean(u**2) + np.mean(v**2) + np.mean(w**2))
            k_from_each_pencil.append(k_pencil)
            
            # --- Calcul de la dissipation pour ce pencil ---
            # Formule ε = 15ν <(∂u/∂x)²> ok pour HIT
            if direction == 'x': longitudinal_fluc = u
            if direction == 'y': longitudinal_fluc = v
            if direction == 'z': longitudinal_fluc = w
            
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

def calculate_structure_functions(pencils, globals):
    """Calcule les fonctions de structure du 2ème ordre.

    Intuition :
    ----------------
    On cherche à quantifier comment les vitesses fluctuent entre deux points séparés par une distance r.
    - D11(r) : différence de la composante de vitesse **parallèle** à la direction du point de référence.
    - D22(r) : différences des composantes de vitesse **perpendiculaires** à cette direction.
    
    Pour chaque "pencil" (ligne de points dans le domaine), on calcule :
        (u(x) - u(x+r))^2
    puis on moyenne sur tous les points de la ligne. 
    Ensuite, on moyenne sur tous les pencils pour obtenir une statistique globale. 
    Cela permet de voir :
        - à petite échelle : les micro-fluctuations (dissipation)
        - à grande échelle : comment l'énergie est distribuée dans le domaine
    La pente de D11(r) et D22(r) dans l’inertial range suit la loi universelle de Kolmogorov r^(2/3)
    """

    print("\n--- Tâche 2 : Calcul des fonctions de structure ---")
    
    eps = globals['eps'] 
    eta = globals['eta']     
    dx = L_DOMAIN / N_POINTS # pas spatial pour un pencil

    max_shift = N_POINTS // 2                   # on calcule jusqu'à la moitié du domaine (pcq périodique)
    r_values = np.arange(1, max_shift) * dx     # vecteur des distances r = n * dx (commence à dx)
    
    D11_accumulator = np.zeros_like(r_values)
    D22_accumulator = np.zeros_like(r_values)
    
    for direction in DIRECTIONS:
        if direction == 'x': long_idx, trans_idx1, trans_idx2 = 0, 1, 2
        if direction == 'y': long_idx, trans_idx1, trans_idx2 = 1, 0, 2
        if direction == 'z': long_idx, trans_idx1, trans_idx2 = 2, 1, 0

        for i in range(N_FILES_PER_DIR):
            # Les champs sont déjà des fluctuations
            u_fluc = pencils[direction][i, :, long_idx]
            v_fluc = pencils[direction][i, :, trans_idx1]
            w_fluc = pencils[direction][i, :, trans_idx2]
            
            for shift_idx, shift in enumerate(range(1, max_shift)):
                # D11 : longitudinal (composante parallèle)
                diff_u = u_fluc - np.roll(u_fluc, -shift)
                D11_accumulator[shift_idx] += np.mean(diff_u**2)
                
                # D22 : transverse (composantes perpendiculaires)
                diff_v = v_fluc - np.roll(v_fluc, -shift)
                diff_w = w_fluc - np.roll(w_fluc, -shift)
                D22_accumulator[shift_idx] += np.mean(diff_v**2) + np.mean(diff_w**2)

    # Moyenne finale sur les 48 pencils
    D11_avg = D11_accumulator / TOTAL_PENCILS
    D22_avg = D22_accumulator / (TOTAL_PENCILS * 2)

    return r_values, D11_avg, D22_avg




# --- 5. Spectres d'énergie 1D ---

def calculate_energy_spectra(pencils, globals):
    """Calcule les spectres d'énergie 1D E11 et E22.

    Intuition :
    ----------------
    On veut voir comment l'énergie cinétique est distribuée selon les différentes
    échelles (ou longueurs d'onde) dans la turbulence.
    - E11(k) : énergie de la composante **longitudinale** à l'échelle associée à k
    - E22(k) : énergie des composantes **transverses**
    
    Pour chaque pencil :
        1. On prend les fluctuations de vitesse (u', v', w')
        2. On fait la FFT pour passer du domaine spatial au domaine fréquentiel
        3. On calcule la densité spectrale de puissance |u_hat|^2 etc.
    Enfin, on moyenne sur tous les pencils pour obtenir un spectre global.
    
    Cela permet de repérer :
        - à grande échelle (petit k) : où l'énergie est injectée
        - à petite échelle (grand k) : où l'énergie est dissipée par la viscosité
    """
    
    print("\n--- Tâche 3 : Calcul des spectres d'énergie ---")
    
    dx = L_DOMAIN / N_POINTS
    
    # Préparation des nombres d'onde (k)
    k = fftfreq(N_POINTS, d=dx) * 2 * np.pi     # vecteur des nombres d'onde en [rad/m]
    k_pos = k[1:N_POINTS // 2]                  # on garde que les k positifs (exclut k=0)

    # Accumulateurs pour les spectres
    E11_accumulator = np.zeros_like(k_pos)
    E22_accumulator = np.zeros_like(k_pos)

    for direction in DIRECTIONS:
        if direction == 'x': long_idx, trans_idx1, trans_idx2 = 0, 1, 2
        if direction == 'y': long_idx, trans_idx1, trans_idx2 = 1, 0, 2
        if direction == 'z': long_idx, trans_idx1, trans_idx2 = 2, 1, 0

        for i in range(N_FILES_PER_DIR):
            # Les champs fournis sont déjà les fluctuations : pas de soustraction de la moyenne
            u_fluc = pencils[direction][i, :, long_idx]
            v_fluc = pencils[direction][i, :, trans_idx1]
            w_fluc = pencils[direction][i, :, trans_idx2]

            # Transformée de fourier (FFT) 
            u_hat = fft(u_fluc)[1:N_POINTS // 2]
            v_hat = fft(v_fluc)[1:N_POINTS // 2]
            w_hat = fft(w_fluc)[1:N_POINTS // 2]
            
            # Calcul du spectre de puissance (PSD)
            E11_pencil = (L_DOMAIN / (np.pi * N_POINTS**2)) * np.abs(u_hat)**2
            E22_pencil = (L_DOMAIN / (np.pi * N_POINTS**2)) * (np.abs(v_hat)**2 + np.abs(w_hat)**2)
            
            E11_accumulator += E11_pencil
            E22_accumulator += E22_pencil
            
    # Moyenne finale sur les 48 pencils
    E11_avg = E11_accumulator / TOTAL_PENCILS
    E22_avg = E22_accumulator / (TOTAL_PENCILS * 2)


    return k_pos, E11_avg, E22_avg


# --- 6. Fonctions d'autocorrélation (BONUS) ---

def calculate_autocorrelation_functions(pencils, globals):
    """Calcule les fonctions d'autocorrélation f(r) et g(r) et estime les micro‑échelles de Taylor.

    Intuition :
    ----------------
    Les fonctions d'autocorrélation mesurent à quel point la vitesse en un point
    est corrélée avec la vitesse à distance r. 
    - f(r) : corrélation **longitudinale** <u(x) u(x+r)> / <u'u'>
    - g(r) : corrélation **transverse** <v(x) v(x+r)> / <v'v'>

    Pourquoi c'est utile :
        - Pour r=0, f=g=1 (parfaitement corrélé)
        - Quand r augmente, f et g décroissent vers 0 (pas de corrélation)
        - La pente initiale de f(r) et g(r) à r~0 donne la micro-échelle de Taylor λ
          → caractérise la taille typique des petites structures turbulentes.

    Comment c'est fait :
        1. Pour chaque pencil, on prend les fluctuations u', v', w'
        2. On calcule R11(r) et R22(r) pour différentes distances r
        3. On normalise pour obtenir f(r) et g(r)
        4. On estime λ_f et λ_g par fit quadratique sur les petits r
    """

    print("\n--- Tâche BONUS : Calcul des fonctions d'autocorrélation ---")
    
    dx = L_DOMAIN / N_POINTS                    # pas spatial pour un pencil
    max_shift = N_POINTS // 2                   # on calcule jusqu'à la moitié du domaine (pcq périodique)
    r_values = np.arange(1, max_shift) * dx     # vecteur des distances r = n * dx (commence à dx)
    
    # Accumulateurs
    R11_accumulator = np.zeros_like(r_values)
    R22_accumulator = np.zeros_like(r_values)
    variance_u_accumulator = 0                  # pour normalisation 
    variance_v_w_accumulator = 0

    for direction in DIRECTIONS:
        if direction == 'x': long_idx, trans_idx1, trans_idx2 = 0, 1, 2
        if direction == 'y': long_idx, trans_idx1, trans_idx2 = 1, 0, 2
        if direction == 'z': long_idx, trans_idx1, trans_idx2 = 2, 1, 0

        for i in range(N_FILES_PER_DIR):
            # Les champs sont déjà des fluctuations : pas de soustraction de la moyenne
            u_fluc = pencils[direction][i, :, long_idx]
            v_fluc = pencils[direction][i, :, trans_idx1]
            w_fluc = pencils[direction][i, :, trans_idx2]
            
            # On accumule la variance pour la normalisation finale
            variance_u_accumulator += np.mean(u_fluc**2)
            variance_v_w_accumulator += np.mean(v_fluc**2) + np.mean(w_fluc**2)
            
            # Calcul de R(r) = <u'(x) * u'(x+r)> 
            for shift_idx, shift in enumerate(range(1, max_shift)):
                R11_accumulator[shift_idx] += np.mean(u_fluc * np.roll(u_fluc, -shift))
                R22_accumulator[shift_idx] += np.mean(v_fluc * np.roll(v_fluc, -shift)) + np.mean(w_fluc * np.roll(w_fluc, -shift))

    # Moyenne et normalisation
    variance_u_avg = variance_u_accumulator / TOTAL_PENCILS
    variance_v_w_avg = variance_v_w_accumulator / (TOTAL_PENCILS * 2)
    
    f_r = R11_accumulator / (TOTAL_PENCILS * variance_u_avg)
    g_r = R22_accumulator / (TOTAL_PENCILS * 2 * variance_v_w_avg)


    # --- Estimation des micro‑échelles de Taylor par fit quadratique (linéaire en r^2) ---
    # On suppose pour petits r : f(r) ≈ 1 - (1/λ^2) r^2  => 1 - f ≈ a * r^2  avec a = 1/λ^2
    n_fit = 3                                  # nombre de points à utiliser pour le fit (petits r)
    n_fit = min(n_fit, len(r_values))
    x = r_values[:n_fit]**2                    # variable indépendante r^2
    y_f = 1.0 - f_r[:n_fit] 
    y_g = 1.0 - g_r[:n_fit]

    try:
        # fit linéaire y = a * x + b  (on s'intéresse surtout à la pente a)
        a_f, b_f = np.polyfit(x, y_f, 1)
        a_g, b_g = np.polyfit(x, y_g, 1)

        if a_f <= 0 or a_g <= 0:
            raise ValueError("pente non positive, fallback")

        lambda_f = np.sqrt(1.0 / a_f)
        lambda_g = np.sqrt(1.0 / a_g)

    except Exception:
        # fallback : estimation sur le deuxième point (comme précédemment)
        r2_sq = r_values[1]**2
        lambda_f = np.sqrt(-r2_sq / (f_r[1] - 1))
        lambda_g = np.sqrt(-r2_sq / (g_r[1] - 1))
    
    print("\n--- Micro-échelles de Taylor (via autocorrélation) ---")
    print(f"  Échelle longitudinale (λ_f) = {lambda_f * 1000:.4f} mm")
    print(f"  Échelle transverse (λ_g)    = {lambda_g * 1000:.4f} mm")
    # Vérification théorique : λ_f² devrait être ≈ 2 * λ_g²
    print(f"  Vérification : λ_f² / λ_g² = {lambda_f**2 / lambda_g**2:.2f} (théorie = 2.0)")

    return r_values, f_r, g_r


    
# --- 7. Exécution principale ---

if __name__ == "__main__":
    pencils = {'x': None, 'y': None, 'z': None}
    #pencils = load_pencils(DATA_PATH)
    #np.save(DATA_PATH + '/saved_data/pencils_x.npy', pencils['x'])
    #np.save(DATA_PATH + '/saved_data/pencils_y.npy', pencils['y'])
    #np.save(DATA_PATH + '/saved_data/pencils_z.npy', pencils['z'])
    #print("--------------------------------")
    #print("Pencils data loaded and saved.")
    #print("--------------------------------")
    
    pencils['x'] = np.load(DATA_PATH + '/saved_data/pencils_x.npy')
    pencils['y'] = np.load(DATA_PATH + '/saved_data/pencils_y.npy')
    pencils['z'] = np.load(DATA_PATH + '/saved_data/pencils_z.npy')
    print("--------------------------------")
    print("Saved pencils data loaded")
    print("--------------------------------")

    globals_dict = calculate_global_quantities(pencils)
    
    # Tâche 2
    r_vals, D11, D22 = None, None, None
    r_vals, D11, D22 = calculate_structure_functions(pencils, globals_dict)
    np.save(DATA_PATH + '/saved_data/r_values_sf.npy', r_vals)
    np.save(DATA_PATH + '/saved_data/D11_sf.npy', D11)
    np.save(DATA_PATH + '/saved_data/D22_sf.npy', D22)
    np.load(DATA_PATH + '/saved_data/r_values_sf.npy')
    np.load(DATA_PATH + '/saved_data/D11_sf.npy')
    np.load(DATA_PATH + '/saved_data/D22_sf.npy')
    plot_sf_loglog(r_vals, D11, D22, globals_dict['eta'], RESULTS_PATH)                         # Log-Log
    plot_sf_comp(r_vals, D11, D22, globals_dict['eps'], globals_dict['eta'], RESULTS_PATH)      # Compensé 
    
    #raise SystemExit

    # Tâche 3
    k_vals, E11, E22 = None, None, None
    k_vals, E11, E22 = calculate_energy_spectra(pencils, globals_dict)
    np.save(DATA_PATH + '/saved_data/k_values_spectra.npy', k_vals)
    np.save(DATA_PATH + '/saved_data/E11_spectra.npy', E11)
    np.save(DATA_PATH + '/saved_data/E22_spectra.npy', E22)
    np.load(DATA_PATH + '/saved_data/k_values_spectra.npy')
    np.load(DATA_PATH + '/saved_data/E11_spectra.npy')
    np.load(DATA_PATH + '/saved_data/E22_spectra.npy')
    plot_spectra_loglog(k_vals, E11, E22, globals_dict, RESULTS_PATH)
    plot_spectra_comp(k_vals, E11, E22, globals_dict, RESULTS_PATH)
    
    # Tâche BONUS
    r_vals_corr, f_r, g_r = None, None, None
    r_vals_corr, f_r, g_r = calculate_autocorrelation_functions(pencils, globals_dict)
    np.save(DATA_PATH + '/saved_data/r_values_corr.npy', r_vals_corr)
    np.save(DATA_PATH + '/saved_data/f_r.npy', f_r)
    np.save(DATA_PATH + '/saved_data/g_r.npy', g_r)
    np.load(DATA_PATH + '/saved_data/r_values_corr.npy')
    np.load(DATA_PATH + '/saved_data/f_r.npy')
    np.load(DATA_PATH + '/saved_data/g_r.npy')
    plot_autocorrelation(r_vals_corr, f_r, g_r, globals_dict['eta'], RESULTS_PATH)

    print("\nAnalyse complète terminée")
    # plt.show()