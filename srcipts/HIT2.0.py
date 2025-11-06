import numpy as np
from additionnaly import *
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# --- Configuration and Constants ---
# Domain and simulation parameters from the homework description
L_DOMAIN = 2 * np.pi    # Domain size [m]
NU = 1.10555e-5         # Kinematic viscosity [m^2/s]
N_PENSILS = 48          # Total number of pencils (3 DIRECTIONS * 4x4 grid)
N_POINTS = 2**15        # Number of grid points per pencil
N_FILES = 16            # Number of files per direction
DIRECTIONS = ['x', 'y', 'z']

# Kolmogorov csts
C1 = 1.5
C2 = 2.1

# DEFAULTS PATHS
DATA_PATH = '../Data/'
RESULTS_PATH = '../Results/'


def load_pencils(path=None):
    """Generates data for the simulation."""

    if path == None:
        path = '../Data/'

    pencils = {}
    indexs = [ '0_0', '0_1', '0_2', '0_3', '1_0', '1_1', '1_2', '1_3', 
               '2_0', '2_1', '2_2', '2_3', '3_0', '3_1', '3_2', '3_3' ]
    
    for d in DIRECTIONS:
        folder = path + 'pencils_' + d + '/'
        files = [""]*(N_FILES)
        for i in range(N_FILES):
            files[i] = folder + d + '_' + indexs[i] + '.txt'

        #print(files)
        
        pencils[d] = np.zeros((N_FILES, N_POINTS, 3))
        
        for i, f in enumerate(files):
            print(f'Loading file: {f}')
            data = np.loadtxt(f)
            pencils[d][i,:,:] = data
            
    return pencils

def get_global_quantities(pencils, dx):
    k_list = np.zeros(N_PENSILS)
    du_dx_list = np.zeros(N_PENSILS)
    u_bis_list = np.zeros(N_PENSILS*N_POINTS)
    v_bis_list = np.zeros(N_PENSILS*N_POINTS)
    w_bis_list = np.zeros(N_PENSILS*N_POINTS)

    for j, d in enumerate(DIRECTIONS):
        for i in range(N_FILES):
            index = j * N_FILES + i     # Psk on prend les 3 * 16 pencils = 48 pencils
            #print(index, type(index))
            
            u = pencils[d][i, :, 0]
            v = pencils[d][i, :, 1]
            w = pencils[d][i, :, 2]

            u_mean = np.mean(u*u)
            v_mean = np.mean(v*v)
            w_mean = np.mean(w*w)

            print(f'Pencil index {index}: <u*u> = {u_mean}, <v*v> = {v_mean}, <w*w> = {w_mean}')

            # Comme u_mean != 0, u = u'mean + u'(=u_fluc) 
            u_fluc = u - u_mean
            v_fluc = v - v_mean
            w_fluc = w - w_mean

            u_bis_list[index*N_POINTS:(index+1)*N_POINTS] = u_fluc
            v_bis_list[index*N_POINTS:(index+1)*N_POINTS] = v_fluc
            w_bis_list[index*N_POINTS:(index+1)*N_POINTS] = w_fluc

            # Calcul de <u'u'> = moyenne de u' carré
            u_caré_moy = np.mean(u_fluc**2)
            v_caré_moy = np.mean(v_fluc**2)
            w_caré_moy = np.mean(w_fluc**2)
            

            # turbulent kinetic energy : k = 1/2 ( <u'u'> + <v'v'> + <w'w'> ) != 3/2 <u'u'>
            #k_list[index] = (3 / 2) * u_caré_moy  --> je pense qu'on peut pas utiliser cette formule
            k_list[index] = 1/2 * (u_caré_moy + v_caré_moy + w_caré_moy)

            # on approxime la dérivée par la méthode des différences finies d'ordre 4
            du_dx = fourth_order_derivative(u_fluc, dx)
            du_dx_mean = np.mean(du_dx**2)
            du_dx_list[index] = du_dx_mean

    uu_mean = np.mean(u_bis_list**2)
    vv_mean = np.mean(v_bis_list**2)
    ww_mean = np.mean(w_bis_list**2)
    print(f'uu_mean = {uu_mean}')
    print(f'vv_mean = {vv_mean}')
    print(f'ww_mean = {ww_mean}')



    ###################################################################################################
    # ?????????????????????
    # ON FAIT LA MOYENNE DES 48 PENCILS ?
    # ou
    # ON FAIT LA MOYENNE DES 16 PENCILS PAR DIRECTION ? pour ensuite faire la moyenne des 3 directions
    ###################################################################################################
    
    k_mean = np.mean(k_list)    # ici on fait la moyenne des 48 pencils

    du_dx_mean = np.mean(du_dx_list)    # ici aussi on fait la moyenne des 48 pencils
    #print(f'<(du/dx)^2>_mean = {du_dx_mean}')
    eps_mean = 15 * NU * du_dx_mean

    L_mean = (k_mean)**(3/2) / eps_mean

    Re = ((L_mean * (k_mean)**(1/2)) / eps_mean)

    eta = (L_mean) / (Re)**(3/4)

    lambda_ = (10 * NU * k_mean / eps_mean)**(1/2)

    Re_lambda = ( (20/3) * Re)**(1/2)

    print()
    print("---- Global Quantities ----")
    print(f'k_mean    = {k_mean}')
    print(f'eps_mean  = {eps_mean}')
    print(f'L_mean    = {L_mean}')
    print(f'Re        = {Re}')
    print(f'eta       = {eta}')
    print(f'lambda    = {lambda_}')
    print(f'Re_lambda = {Re_lambda}')
    print()
    return k_mean, eps_mean, L_mean, Re, eta, lambda_, Re_lambda

def compute_structure_functions(pencils, max_r):
    # TODO
    pass

def compute_energy_spectra(pencils, dx):
    # TODO
    pass

def compute_autocorrelation(pencils, max_lag):
    # TODO
    pass
        


def main():
    
    #
    # -- Load Pencils Data --
    #
    pencils = {'x': None, 'y': None, 'z': None}
    
    # -- Uncomment to load and save pencils data --
    #pencils = load_pencils(DATA_PATH)
    np.save(DATA_PATH + '/saved_data/pencils_x.npy', pencils['x'])
    np.save(DATA_PATH + '/saved_data/pencils_y.npy', pencils['y'])
    np.save(DATA_PATH + '/saved_data/pencils_z.npy', pencils['z'])
    print("Pencils data loaded and saved.")
    
    pencils['x'] = np.load(DATA_PATH + '/saved_data/pencils_x.npy')
    pencils['y'] = np.load(DATA_PATH + '/saved_data/pencils_y.npy')
    pencils['z'] = np.load(DATA_PATH + '/saved_data/pencils_z.npy')
    
    #print(pencils)
    print("Saved pencils data loaded")
    print("Shapes of loaded data:")
    print(pencils['x'].shape, pencils['y'].shape, pencils['z'].shape)

    #
    # -- Example usage of the loaded data --
    #
    """print("u_mean(x[0,0]) = ", np.mean(pencils['x'][0,:,0]))

    plot_one_pencil(pencils, 0, 0)
    dx = L_DOMAIN / N_POINTS
    print(f'dx = {dx}')"""

    #
    # -- Compute global quantities --
    #
    dx = L_DOMAIN / N_POINTS
    k, eps, L, Re, eta, lambda_, Re_lambda = get_global_quantities(pencils, dx)

    #
    # -- Structure functions --
    # 
    # TODO

    #
    # -- One-dimensional energy spectra --
    # 
    # TODO

    #
    # -- BONUS: Autocorrelation functions --
    #
    # TODO

if __name__ == "__main__":
    main()