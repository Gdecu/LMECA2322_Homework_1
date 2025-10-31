import numpy as np
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

# DEFALT DATA PATH
DATA_PATH = './Data/'

def load_pencils(path=None):
    """Generates data for the simulation."""

    if path == None:
        path = './Data/'

    directions = ['x', 'y', 'z']
    pencils = {}
    indexs = [ '0_0', '0_1', '0_2', '0_3', '1_0', '1_1', '1_2', '1_3', 
               '2_0', '2_1', '2_2', '2_3', '3_0', '3_1', '3_2', '3_3' ]
    
    for d in directions:
        folder = path + 'pencils_' + d + '/'
        files = [""]*(N_FILES)
        for i in range(N_FILES):
            files[i] = folder + d + '_' + indexs[i] + '.txt'

        print(files)
        pencils[d] = [""]*N_FILES
        
        for f in files:
            print(f'Loading file: {f}')
            data = np.loadtxt(f)
            pencils[d] = data

    return pencils

pencils = load_pencils(DATA_PATH)
print(pencils)