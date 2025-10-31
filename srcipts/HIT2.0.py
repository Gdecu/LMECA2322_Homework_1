import numpy as np
from additionnaly import *
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
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

# DEFAULTS PATHS
DATA_PATH = '../Data/'
RESULTS_PATH = '../Results/'


def load_pencils(path=None):
    """Generates data for the simulation."""

    if path == None:
        path = '../Data/'

    directions = ['x', 'y', 'z']
    pencils = {}
    indexs = [ '0_0', '0_1', '0_2', '0_3', '1_0', '1_1', '1_2', '1_3', 
               '2_0', '2_1', '2_2', '2_3', '3_0', '3_1', '3_2', '3_3' ]
    
    for d in directions:
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
    k_list, eps_list, lambda_list, u_rms_list = [""]*N_FILES, [""]*N_FILES, [""]*N_FILES, [""]*N_FILES

    for i in range(N_FILES):
        u = pencils['x'][:, :,i]
        v = pencils['y'][:, :,i]
        w = pencils['z'][:, :,i]

        u_mean = np.mean(u)
        v_mean = np.mean(v)
        w_mean = np.mean(w)

        # Comme u_mean != 0, u = u'mean + u'(=u_fluc) 
        u_fluc = u - u_mean
        v_fluc = v - v_mean
        w_fluc = w - w_mean

        u_caré_moy = np.mean(u_fluc**2)

        k = (3 / 2) * u_caré_moy        # turbulent kinetic energy : k = 1/2 ( <u'u'> + <v'v'> + <w'w'> ) = 3/2 <u'u'>
        print(f'k = {k}')

        du_dx = fourth_order_derivative(u_fluc, dx)
        print(du_dx)

        


def main():
    
    #
    # -- Load Pencils Data --
    #
    pencils = {'x': None, 'y': None, 'z': None}
    
    # -- Uncomment to load and save pencils data --
    #pencils = load_pencils(DATA_PATH)
    #np.save(DATA_PATH + '/saved_data/pencils_x.npy', pencils['x'])
    #np.save(DATA_PATH + '/saved_data/pencils_y.npy', pencils['y'])
    #np.save(DATA_PATH + '/saved_data/pencils_z.npy', pencils['z'])
    #print("Pencils data loaded and saved.")
    
    pencils['x'] = np.load(DATA_PATH + '/saved_data/pencils_x.npy')
    pencils['y'] = np.load(DATA_PATH + '/saved_data/pencils_y.npy')
    pencils['z'] = np.load(DATA_PATH + '/saved_data/pencils_z.npy')
    
    #print(pencils)
    print("Saved pencils data loaded:")
    print(pencils['x'].shape, pencils['y'].shape, pencils['z'].shape)

    #
    # -- Example usage of the loaded data --
    #
    print("u_mean(x[0,0]) = ", np.mean(pencils['x'][0,:,0]))

    plot_one_pencil(pencils, 0, 0)
    dx = L_DOMAIN / N_POINTS
    print(f'dx = {dx}')

    #
    # -- Compute global quantities --
    #
    get_global_quantities(pencils, dx)


if __name__ == "__main__":
    main()