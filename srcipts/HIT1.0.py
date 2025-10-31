import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- Configuration and Constants ---
# Domain and simulation parameters from the homework description
L_DOMAIN = 2 * np.pi    # Domain size [m]
NU = 1.10555e-5         # Kinematic viscosity [m^2/s]
N_PENCILS = 48          # Total number of pencils (3 directions * 4x4 grid)
N_POINTS = 2**15        # Number of grid points per pencil

# Kolmogorov csts
C1 = 1.5
C2 = 2.1

# --- Data Loading et Preparation ---

def load_synthetic_data(n_points, n_pencils):
    """Generates synthetic velocity data with turbulence-like properties."""
    z = np.linspace(0, L_DOMAIN, n_points, endpoint=False)

    dz = z[1] - z[0]
    all_pencils_data = [0]*n_pencils

    # synthetic velocity fields u, v, w for each pencil
    for _ in range(n_pencils):
        # Create a signal with a k^(-5/3) spectrum
        k = fftfreq(n_points, d=dz) * 2 * np.pi
        k[0] = 1e-6 # Avoid division by zero
        
        # Generate random phases
        phase_u = np.exp(2j * np.pi * np.random.rand(n_points))
        phase_v = np.exp(2j * np.pi * np.random.rand(n_points))
        phase_w = np.exp(2j * np.pi * np.random.rand(n_points))
        
        # Create Fourier coefficients with a -5/3 power law
        uk_hat = np.sqrt(np.abs(k)**(-5/3)) * phase_u
        vk_hat = np.sqrt(np.abs(k)**(-5/3)) * phase_v
        wk_hat = np.sqrt(np.abs(k)**(-5/3)) * phase_w
        
        # Inverse FFT to get velocity signals
        u = np.real(np.fft.ifft(uk_hat))
        v = np.real(np.fft.ifft(vk_hat))
        w = np.real(np.fft.ifft(wk_hat))
        
        # Normalize to have zero mean and unit variance
        u = (u - np.mean(u)) / np.std(u)
        v = (v - np.mean(v)) / np.std(v)
        w = (w - np.mean(w)) / np.std(w)

        all_pencils_data[_] = {'z': z, 'u': u, 'v': v, 'w': w}

    return all_pencils_data, dz


# --- Analysis Functions ---

def fourth_order_derivative(f, dx):
    """Computes the derivative using a fourth-order central difference scheme."""
    # f_minus_2, f_minus_1, f_plus_1, f_plus_2
    f_m2 = np.roll(f, 2)
    f_m1 = np.roll(f, 1)
    f_p1 = np.roll(f, -1)
    f_p2 = np.roll(f, -2)
    df_dx = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * dx)
    return df_dx

def compute_global_quantities(pencils_data, dz):
    """Computes and averages all global quantities."""
    k_list, eps_list, lambda_list = [], [], []

    for data in pencils_data:
        u, v, w = data['u'], data['v'], data['w']
        
        # Velocity fluctuations (mean is already ~0 for synthetic data)
        u_prime = u - np.mean(u)
        v_prime = v - np.mean(v)
        w_prime = w - np.mean(w)
        
        # Turbulent Kinetic Energy
        k = 0.5 * (np.mean(u_prime**2) + np.mean(v_prime**2) + np.mean(w_prime**2))
        k_list.append(k)
        
        # Dissipation Rate
        du_dz = fourth_order_derivative(u_prime, dz)
        eps = 15 * NU * np.mean(du_dz**2)
        eps_list.append(eps)
        
        # Taylor Micro-scale
        lambda_sq = np.mean(u_prime**2) / np.mean(du_dz**2)
        lambda_list.append(np.sqrt(lambda_sq))

    # Average results over all pencils
    k_avg = np.mean(k_list)
    eps_avg = np.mean(eps_list)
    lambda_avg = np.mean(lambda_list)
    
    # Other quantities from averaged values
    eta = (NU**3 / eps_avg)**0.25
    u_rms = np.sqrt(2./3. * k_avg) # From k = 3/2 * u_rms^2 for isotropy
    
    # Integral Scale (computed from the energy spectrum)
    _, E11_avg, _ = compute_energy_spectra(pencils_data, dz)
    L_int = (np.pi / np.mean(u_prime**2)) * E11_avg[0] # Using E11 at k=0
    
    Re_lambda = u_rms * lambda_avg / NU
    Re_L = u_rms * L_int / NU
    
    return {
        "k": k_avg, "epsilon": eps_avg, "eta": eta, "lambda": lambda_avg,
        "L_integral": L_int, "Re_lambda": Re_lambda, "Re_L": Re_L
    }


def compute_structure_functions(pencils_data, dz, eta):
    """Computes longitudinal and transverse structure functions."""
    max_r = int(5e3 * eta / dz)
    D11 = np.zeros(max_r)
    D22 = np.zeros(max_r)
    n_points = len(pencils_data[0]['z'])
    
    for data in pencils_data:
        u, v, w = data['u'], data['v'], data['w']
        for i in range(1, max_r):
            # Longitudinal (along z)
            du = u - np.roll(u, -i)
            D11[i] += np.mean(du**2)
            # Transverse (using both v and w)
            dv = v - np.roll(v, -i)
            dw = w - np.roll(w, -i)
            D22[i] += 0.5 * (np.mean(dv**2) + np.mean(dw**2))
            
    D11 /= len(pencils_data)
    D22 /= len(pencils_data)
    r = np.arange(max_r) * dz
    
    return r, D11, D22

def compute_energy_spectra(pencils_data, dz):
    """Computes and averages the 1D energy spectra."""
    n_points = len(pencils_data[0]['z'])
    E11_sum = np.zeros(n_points // 2)
    E22_sum = np.zeros(n_points // 2)
    
    k = 2 * np.pi * fftfreq(n_points, d=dz)[:n_points//2]
    
    for data in pencils_data:
        u_hat = fft(data['u'])
        v_hat = fft(data['v'])
        w_hat = fft(data['w'])
        
        # Energy spectrum E(k) = 2 * (1/L) * |u_hat(k)|^2
        E11_current = 2 * (1 / L_DOMAIN) * np.abs(u_hat[:n_points//2])**2
        E22_current = (1 / L_DOMAIN) * (np.abs(v_hat[:n_points//2])**2 + np.abs(w_hat[:n_points//2])**2)
        
        E11_sum += E11_current
        E22_sum += E22_current
        
    E11_avg = E11_sum / len(pencils_data)
    E22_avg = E22_sum / len(pencils_data)
    
    return k, E11_avg, E22_avg

def compute_autocorrelation(pencils_data, dz, eta):
    """Computes autocorrelation functions f(r) and g(r)."""
    max_r = int(5e2 * eta / dz)
    f_r = np.zeros(max_r)
    g_r = np.zeros(max_r)
    
    for data in pencils_data:
        u_prime = data['u'] - np.mean(data['u'])
        v_prime = data['v'] - np.mean(data['v'])
        u_var = np.mean(u_prime**2)
        v_var = np.mean(v_prime**2)
        
        for i in range(max_r):
            f_r[i] += np.mean(u_prime * np.roll(u_prime, -i)) / u_var
            g_r[i] += np.mean(v_prime * np.roll(v_prime, -i)) / v_var
            
    f_r /= len(pencils_data)
    g_r /= len(pencils_data)
    r = np.arange(max_r) * dz
    
    return r, f_r, g_r




# --- Main ---

# 1. Data
pencils, dz = load_synthetic_data(N_POINTS, N_PENCILS)
print(dz)
# 2. Compute foundational quantities first
k_list, eps_list, lambda_list, u_rms_list = [""]*N_PENCILS, [""]*N_PENCILS, [""]*N_PENCILS, [""]*N_PENCILS
for i, data in enumerate(pencils):
    u, v, w = data['u'], data['v'], data['w']
    u_prime = u - np.mean(u)
    v_prime = v - np.mean(v)
    w_prime = w - np.mean(w)
    k = 0.5 * (np.mean(u_prime**2) + np.mean(v_prime**2) + np.mean(w_prime**2))
    du_dz = fourth_order_derivative(u_prime, dz)
    eps = 15 * NU * np.mean(du_dz**2)
    lambda_sq = np.mean(u_prime**2) / np.mean(du_dz**2)

    k_list[i] = k
    eps_list[i] = eps
    lambda_list[i] = np.sqrt(lambda_sq)
    u_rms_list[i] = np.sqrt(np.mean(u_prime**2))

k_avg = np.mean(k_list)
eps_avg = np.mean(eps_list)
lambda_avg = np.mean(lambda_list)
u_rms_avg = np.mean(u_rms_list)
eta_avg = (NU**3 / eps_avg)**0.25

# 3. BONUS: Compute Autocorrelation to find L_integral
r_ac, f_r, g_r = compute_autocorrelation(pencils, dz, eta_avg)


# Find the first zero-crossing of the autocorrelation function
try:
    zero_crossing_index = np.where(f_r <= 0)[0][0]
    # Integrate f(r) from r=0 up to the zero-crossing
    L_integral_avg = np.trapz(f_r[:zero_crossing_index], r_ac[:zero_crossing_index])
except IndexError:
    # If it never crosses zero, integrate over the whole domain (less ideal)
    L_integral_avg = np.trapz(f_r, r_ac)

# 4. Compute Final Global Quantities
Re_lambda_avg = u_rms_avg * lambda_avg / NU
Re_L_avg = u_rms_avg * L_integral_avg / NU

global_q = {
    "k": k_avg, "epsilon": eps_avg, "eta": eta_avg, "lambda": lambda_avg,
    "L_integral": L_integral_avg, "Re_lambda": Re_lambda_avg, "Re_L": Re_L_avg
}

print("--- Global Quantities ---")
for key, val in global_q.items():
    print(f"{key:<12}: {val:.4e}")

# 5. Compute Structure Functions and Energy Spectra for plotting
r_sf, D11, D22 = compute_structure_functions(pencils, dz, eta_avg)
r_over_eta = r_sf / eta_avg
k_spec, E11, E22 = compute_energy_spectra(pencils, dz)
kn = k_spec * eta_avg
r_ac_over_eta = r_ac / eta_avg



# --- Plots ---

plt.style.use('seaborn-v0_8-talk')

# Plot 1: Structure Functions
plt.figure(figsize=(10, 7))
plt.loglog(r_over_eta[1:], D11[1:], label=r'$D_{11}(r)$ (Longitudinal)')
plt.loglog(r_over_eta[1:], D22[1:], label=r'$D_{22}(r)$ (Transverse)')
plt.title('Longitudinal and Transverse Structure Functions')
plt.xlabel(r'$r/\eta$')
plt.ylabel(r'$D_{ii}(r)$')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('../Results/slide_structure_functions.png')
plt.close()

# Plot 2: Compensated Structure Functions
r_plot = r_over_eta[1:]
comp_D11 = D11[1:] / ((eps_avg * r_sf[1:])**(2/3))
comp_D22 = D22[1:] / ((eps_avg * r_sf[1:])**(2/3))

plt.figure(figsize=(10, 7))
plt.semilogx(r_plot, comp_D11, label=r'$(εr)^{-2/3} D_{11}(r)$')
plt.semilogx(r_plot, comp_D22, label=r'$(εr)^{-2/3} D_{22}(r)$')
plt.axhline(y=C2, color='k', linestyle='--', label=f'C2 = {C2}')
plt.title('Compensated Structure Functions')
plt.xlabel(r'$r/\eta$')
plt.ylabel(r'$(εr)^{-2/3} D_{ii}(r)$')
plt.ylim(0, 4)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('../Results/slide_compensated_sf.png')
plt.close()

# Plot 3: Energy Spectra
E_norm = (eps * NU**5)**0.25
plt.figure(figsize=(10, 7))
plt.loglog(kn[1:], E11[1:]/E_norm, label=r'$E_{11}(k)$')
plt.loglog(kn[1:], E22[1:]/E_norm, label=r'$E_{22}(k)$')
# Theoretical slope
k_inertial = np.logspace(np.log10(kn[10]), np.log10(kn[1000]), 10)
plt.loglog(k_inertial, 1.5 * k_inertial**(-5/3), 'k--', label=r'$k^{-5/3}$ slope')
plt.title('Dimensionless One-Dimensional Energy Spectra')
plt.xlabel(r'$k\eta$')
plt.ylabel(r'$E_{ii}(k) / (\epsilon \nu^5)^{1/4}$')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('../Results/slide_energy_spectra.png')
plt.close()

# Plot 4: Compensated Energy Spectra
comp_E11 = (kn**(5/3)) * E11 / E_norm
comp_E22 = (kn**(5/3)) * E22 / E_norm
plt.figure(figsize=(10, 7))
plt.semilogx(kn[1:], comp_E11[1:], label=r'$(k\eta)^{5/3} E_{11}$')
plt.semilogx(kn[1:], comp_E22[1:], label=r'$(k\eta)^{5/3} E_{22}$')
plt.axhline(y=C1 * (18/55), color='k', linestyle='--', label=r'$C_1 \approx 1.5$') # C1' = 18/55 C1
plt.title('Compensated Energy Spectra')
plt.xlabel(r'$k\eta$')
plt.ylabel(r'$(k\eta)^{5/3} E_{ii} / (\epsilon \nu^5)^{1/4}$')
plt.ylim(0, 2)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('../Results/slide_compensated_spectra.png')
plt.close()

# Plot 5: Autocorrelation Functions
plt.figure(figsize=(10, 7))
plt.plot(r_ac_over_eta, f_r, label=r'$f(r)$ (Longitudinal)')
plt.plot(r_ac_over_eta, g_r, label=r'$g(r)$ (Transverse)')
plt.title('BONUS: Autocorrelation Functions')
plt.xlabel(r'$r/\eta$')
plt.ylabel(r'Autocorrelation')
plt.grid(True)
plt.legend()
plt.xlim(0, 500)
plt.savefig('../Results/slide_autocorrelation.png')
plt.close()

print("\nAnalysis complete. All plots saved to disk.")