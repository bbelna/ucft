import numpy as np

def proton_decay_lifetime(m_p, M_XY, M_T, g5, y_u, y_d, A_R, beta_H):
    """
    Compute the proton lifetime (in years) from the effective dimension-6 operator.
    
    We define the effective operator coefficient as:
      1/M_eff^2 = g5^2/M_XY^2 + y_u*y_d/M_T^2,
    and assume the decay rate:
      Gamma ~ (m_p^5 / M_eff^4) * (alpha_GUT)^2 * A_R^2 * (beta_H)^2,
    where alpha_GUT = g5^2/(4π).
    
    Returns: proton lifetime in years.
    """
    inv_M_eff_sq = (g5**2) / (M_XY**2) + (y_u * y_d) / (M_T**2)
    M_eff = 1.0 / np.sqrt(inv_M_eff_sq)
    alpha_GUT = g5**2 / (4.0 * np.pi)
    
    Gamma = (m_p**5) * (alpha_GUT**2) * (A_R**2) * (beta_H**2) / (M_eff**4)
    tau_natural = 1.0 / Gamma  # lifetime in GeV^-1
    
    sec_per_GeV_inv = 6.582e-25  # 1 GeV^-1 in seconds
    tau_seconds = tau_natural * sec_per_GeV_inv
    tau_years = tau_seconds / 3.156e7
    return tau_years

# Nominal parameters
m_p = 1.0  # Proton mass in GeV

# Refined parameter ranges:
# Increase M_XY to [5e15, 1e16] GeV and M_T to [1e14, 1e16] GeV.
M_XY_values = np.linspace(5e15, 1e16, 5)  # Gauge boson mass range
M_T_values  = np.linspace(1e14, 1e16, 5)   # Color-triplet mass range
g5_values   = np.linspace(0.65, 0.75, 3)    # Unified gauge coupling
y_u_values  = np.linspace(0.50, 0.60, 3)    # Up-type Yukawa
y_d_values  = np.linspace(0.40, 0.50, 3)    # Down-type Yukawa

# Fixed parameters for operator running and hadronic matrix element:
A_R = 2.5        # Typical renormalization factor
beta_H = 0.01    # Typical hadronic matrix element in GeV^3

# Global parameter scan:
results = []
for M_XY in M_XY_values:
    for M_T in M_T_values:
        for g5 in g5_values:
            for y_u in y_u_values:
                for y_d in y_d_values:
                    tau_p = proton_decay_lifetime(m_p, M_XY, M_T, g5, y_u, y_d, A_R, beta_H)
                    results.append((M_XY, M_T, g5, y_u, y_d, tau_p))

# Convert results to a structured numpy array for easier analysis.
results_arr = np.array(results, dtype=[('M_XY', float), ('M_T', float), 
                                         ('g5', float), ('y_u', float), 
                                         ('y_d', float), ('tau', float)])

# Display a few sample results:
print("Sample proton lifetime predictions (in years):")
for r in results_arr[:5]:
    print("M_XY: {:.2e} GeV, M_T: {:.2e} GeV, g5: {:.3f}, y_u: {:.3f}, y_d: {:.3f} => tau_p: {:.3e} years"
          .format(r['M_XY'], r['M_T'], r['g5'], r['y_u'], r['y_d'], r['tau']))

print("\nParameter sets with tau_p > 1e34 years:")
found = False
for r in results_arr:
    if r['tau'] > 1e34:
        found = True
        print("M_XY: {:.2e} GeV, M_T: {:.2e} GeV, g5: {:.3f}, y_u: {:.3f}, y_d: {:.3f} => tau_p: {:.3e} years"
              .format(r['M_XY'], r['M_T'], r['g5'], r['y_u'], r['y_d'], r['tau']))
if not found:
    print("No parameter sets found with tau_p > 1e34 years in the current scan.")
