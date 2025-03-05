#!/usr/bin/env python3
"""
Complete Numerical Analysis Script for UCFT SU(5) RG Running with Thresholds

This script integrates the RG equations for all couplings of our unified clock field theory (UCFT)
based on SU(5), including threshold corrections at:
  - M_GUT: Full SU(5)
  - M_XY: Decoupling of the heavy X,Y gauge bosons
  - M_Phi: Decoupling of the adjoint clock field Φ(24)
  - M_Sigma: Decoupling of the extra Higgs Σ(50)
  - M_5C: Decoupling of the color-triplet components in H(5) and 𝟝̅(5̅)
  - M_T: Matching onto the Standard Model (SM)
  - m_Z: Electroweak scale

The one-loop beta–coefficients (b) used in each stage are based on our derivations:
    b_full       = 9.83
    b_XY         = 0.66
    b_noPhi      = 2.33
    b_noPhi_noSig= 4.83
    b_doublet    = 5.03

Two-loop coefficients (B) are kept as placeholder values.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

###############################################################################
# 1) Configuration: Mass Spectrum (in GeV)
###############################################################################
M_GUT  = 1.0e16   # GUT scale
M_XY   = 5.0e15   # Mass of X,Y gauge bosons
M_Phi  = 2.0e15   # Mass of clock field Φ (24)
M_Sig  = 1.0e15   # Mass of extra Higgs Σ (50)
M_5C   = 3.0e14   # Mass of color-triplet component in H(5)
M_T    = 1.0e9    # Final threshold for matching to SM
m_Z    = 91.1876  # Electroweak scale

###############################################################################
# 2) Group-Theoretic Data & Loop Factors
###############################################################################
pi = np.pi
loop1 = 1.0/(16.0 * pi**2)
loop2 = 1.0/((16.0 * pi**2)**2)

# One-loop beta coefficients from our analysis:
b_full        = 9.83      # Stage 1: Full SU(5)
b_XY          = 0.66      # Stage 2: Minus X,Y gauge bosons
b_noPhi       = 2.33      # Stage 3: Minus Φ (24)
b_noPhi_noSig = 4.83      # Stage 4: Minus Σ (50)
b_doublet     = 5.03      # Stage 5: Only light doublets remain

# Two-loop coefficients (placeholders; adjust if necessary)
B_full        = 258.5
B_XY          = 230.0
B_noPhi       = 180.0
B_noPhi_noSig = 150.0
B_doublet     = 140.0

###############################################################################
# 3) Beta Functions for Each Stage (Including All Couplings)
###############################################################################
# The couplings we run are: g5 (gauge), y_u, y_d (Yukawas), lam_H (Higgs quartic),
# lam_Phi (Φ quartic) and lam_Sig (Σ quartic).

def beta_full_su5(t, y):
    """
    Full SU(5) beta functions (Stage 1).
    b_full = 9.83, B_full = 258.5.
    """
    g5, y_u, y_d, lam_H, lam_Phi, lam_Sig = y
    dg5_dt    = - loop1 * b_full * g5**3    + loop2 * B_full * g5**5
    dyu_dt    =  (y_u * (6.0*y_u**2 + y_d**2 - 24.0*g5**2)) * loop1
    dyd_dt    =  (y_d * (4.0*y_d**2 + y_u**2 - 20.4*g5**2)) * loop1
    dlamH_dt  =  (12.0*lam_H**2 - 9.0*g5**2*lam_H + 12.0*g5**4 +
                  4.0*y_u**2*lam_H - 4.0*y_u**4) * loop1
    dlamPhi_dt=  (10.0*lam_Phi**2 - 8.0*g5**2*lam_Phi + 5.0*g5**4) * loop1
    dlamSig_dt=  (8.0*lam_Sig**2 - 6.0*g5**2*lam_Sig + 4.0*g5**4) * loop1
    return [dg5_dt, dyu_dt, dyd_dt, dlamH_dt, dlamPhi_dt, dlamSig_dt]

def beta_su5_XY(t, y):
    """
    Beta functions after removing X,Y.
    b_XY = 0.66, B_XY = 230.0.
    """
    g5, y_u, y_d, lam_H, lam_Phi, lam_Sig = y
    dg5_dt    = - loop1 * b_XY * g5**3    + loop2 * B_XY * g5**5
    dyu_dt    =  (y_u * (6.0*y_u**2 + y_d**2 - 23.0*g5**2)) * loop1
    dyd_dt    =  (y_d * (4.0*y_d**2 + y_u**2 - 19.8*g5**2)) * loop1
    dlamH_dt  =  (12.0*lam_H**2 - 8.2*g5**2*lam_H + 11.0*g5**4 +
                  4.0*y_u**2*lam_H - 4.0*y_u**4) * loop1
    dlamPhi_dt=  (10.0*lam_Phi**2 - 7.5*g5**2*lam_Phi + 4.5*g5**4) * loop1
    dlamSig_dt=  (8.0*lam_Sig**2 - 5.5*g5**2*lam_Sig + 3.5*g5**4) * loop1
    return [dg5_dt, dyu_dt, dyd_dt, dlamH_dt, dlamPhi_dt, dlamSig_dt]

def beta_su5_noPhi(t, y):
    """
    Beta functions after removing Φ (Stage 3).
    b_noPhi = 2.33, B_noPhi = 180.0.
    Here, lam_Phi is frozen (set to 0).
    """
    g5, y_u, y_d, lam_H, lam_Phi, lam_Sig = y
    dg5_dt    = - loop1 * b_noPhi * g5**3    + loop2 * B_noPhi * g5**5
    dyu_dt    =  (y_u * (6.0*y_u**2 + y_d**2 - 22.0*g5**2)) * loop1
    dyd_dt    =  (y_d * (4.0*y_d**2 + y_u**2 - 18.5*g5**2)) * loop1
    dlamH_dt  =  (12.0*lam_H**2 - 7.8*g5**2*lam_H + 9.5*g5**4 +
                  4.0*y_u**2*lam_H - 4.0*y_u**4) * loop1
    dlamPhi_dt=  0.0
    dlamSig_dt=  (8.0*lam_Sig**2 - 5.0*g5**2*lam_Sig + 3.0*g5**4) * loop1
    return [dg5_dt, dyu_dt, dyd_dt, dlamH_dt, dlamPhi_dt, dlamSig_dt]

def beta_su5_noPhi_noSig(t, y):
    """
    Beta functions after removing Σ (Stage 4).
    b_noPhi_noSig = 4.83, B_noPhi_noSig = 150.0.
    Here, lam_Sig is frozen to 0.
    """
    g5, y_u, y_d, lam_H, lam_Phi, lam_Sig = y
    dg5_dt    = - loop1 * b_noPhi_noSig * g5**3    + loop2 * B_noPhi_noSig * g5**5
    dyu_dt    =  (y_u * (6.0*y_u**2 + y_d**2 - 21.5*g5**2)) * loop1
    dyd_dt    =  (y_d * (4.0*y_d**2 + y_u**2 - 17.0*g5**2)) * loop1
    dlamH_dt  =  (12.0*lam_H**2 - 7.0*g5**2*lam_H + 8.0*g5**4 +
                  4.0*y_u**2*lam_H - 4.0*y_u**4) * loop1
    dlamPhi_dt=  0.0
    dlamSig_dt=  0.0
    return [dg5_dt, dyu_dt, dyd_dt, dlamH_dt, dlamPhi_dt, dlamSig_dt]

def beta_su5_noPhi_noSig_no5C(t, y):
    """
    Beta functions after removing the color-triplet components (Stage 5).
    b_doublet = 5.03, B_doublet = 140.0.
    """
    g5, y_u, y_d, lam_H, lam_Phi, lam_Sig = y
    dg5_dt    = - loop1 * b_doublet * g5**3    + loop2 * B_doublet * g5**5
    dyu_dt    = (y_u * (5.5*y_u**2 + 1.0*y_d**2 - 20.0*g5**2)) * loop1
    dyd_dt    = (y_d * (3.8*y_d**2 + 1.0*y_u**2 - 16.0*g5**2)) * loop1
    dlamH_dt  = (12.0*lam_H**2 - 6.5*g5**2*lam_H + 7.0*g5**4 +
                 4.0*y_u**2*lam_H - 4.0*y_u**4) * loop1
    dlamPhi_dt= 0.0
    dlamSig_dt= 0.0
    return [dg5_dt, dyu_dt, dyd_dt, dlamH_dt, dlamPhi_dt, dlamSig_dt]

###############################################################################
# 4) Threshold Correction Functions
###############################################################################
def gauge_threshold_correction(g_above, n, M_above, M_below):
    """
    Compute the threshold correction for the gauge coupling.
    For n heavy states integrated out between M_above and M_below:
       1/alpha_below = 1/alpha_above - (n/(6π)) ln(M_above/M_below)
    """
    alpha_above = g_above**2 / (4.0 * pi)
    alpha_inv_above = 1.0 / alpha_above
    ln_ratio = np.log(M_above / M_below)
    delta_alpha_inv = -(n / (6.0 * pi)) * ln_ratio
    alpha_inv_below = alpha_inv_above + delta_alpha_inv
    alpha_below = 1.0 / alpha_inv_below
    return np.sqrt(4.0 * pi * alpha_below)

def yukawa_threshold_correction(y_above, c, M_above, M_below):
    """
    Compute the threshold correction for a Yukawa coupling.
    Using:
      y_below = y_above * exp(- (c/(6π)) ln(M_above/M_below))
    where c is a constant from group factors.
    """
    ln_ratio = np.log(M_above / M_below)
    delta_y = -(c / (6.0 * pi)) * ln_ratio
    return y_above * np.exp(delta_y)

###############################################################################
# 5) Standard Model Beta Functions (One-Loop & Two-Loop)
###############################################################################
def beta_sm_2loop(t, y):
    """
    Two-loop beta functions for SM couplings:
      y = [g1, g2, g3, y_t, lam]
    g1 is GUT normalized (g1' = sqrt(3/5)*g1).
    """
    g1, g2, g3, y_t, lam = y
    # One-loop coefficients:
    b1 = 41.0/10.0
    b2 = -19.0/6.0
    b3 = -7.0
    # Two-loop coefficients (placeholders)
    B1 = (199.0/50.0)*g1**2 + (27.0/10.0)*g2**2 + (44.0/5.0)*g3**2 - (17.0/10.0)*y_t**2
    B2 = (9.0/10.0)*g1**2 + (35.0/6.0)*g2**2 + 12.0*g3**2 - (3.0/2.0)*y_t**2
    B3 = (11.0/10.0)*g1**2 + (9.0/2.0)*g2**2 - 26.0*g3**2 - 2.0*y_t**2

    dg1_dt = (g1**3/(16*pi**2))*b1 + (g1**3/((16*pi**2)**2))*B1
    dg2_dt = (g2**3/(16*pi**2))*b2 + (g2**3/((16*pi**2)**2))*B2
    dg3_dt = (g3**3/(16*pi**2))*b3 + (g3**3/((16*pi**2)**2))*B3

    # Two-loop beta for top Yukawa (approximate):
    beta_y1 = (9.0/2.0)*y_t**2 - ((17.0/20.0)*g1**2 + (9.0/4.0)*g2**2 + 8.0*g3**2)
    beta_y2 = (-12.0*y_t**4 +
               y_t**2*((131.0/16.0)*g1**2 + (225.0/16.0)*g2**2 + 36.0*g3**2) -
               (1187.0/216.0)*g1**4 - (3.0/4.0)*g1**2*g2**2 +
               (19.0/9.0)*g1**2*g3**2 - (23.0/4.0)*g2**4 - 108.0*g2**2*g3**2 +
               108.0*g3**4)
    dy_t_dt = (y_t/(16*pi**2))*beta_y1 + (y_t/((16*pi**2)**2))*beta_y2

    # One-loop beta for Higgs quartic:
    dlam_dt = (1.0/(16*pi**2)) * (
                  24.0*lam**2 - 6.0*y_t**4 +
                  (3.0/8.0)*(2.0*g2**4 + (g1**2+g2**2)**2) -
                  3.0*lam*(3.0*g2**2 + g1**2 - 4.0*y_t**2)
              )
    return [dg1_dt, dg2_dt, dg3_dt, dy_t_dt, dlam_dt]

###############################################################################
# 6) MASTER FUNCTION: Multi-Threshold Run with Two-Loop Corrections
###############################################################################
def run_ucft_su5_multithreshold():
    """
    Perform multi-threshold RG running:
      1) Full SU(5): M_GUT -> M_XY
      2) SU(5)-X,Y: M_XY -> M_Phi
      3) SU(5)-X,Y-Φ: M_Phi -> M_Sig
      4) SU(5)-X,Y-Φ-Σ: M_Sig -> M_5C
      5) SU(5)-X,Y-Φ-Σ-5C: M_5C -> M_T
      6) Match onto SM at M_T, then run SM: M_T -> m_Z
    """
    # Log scales for integration
    t_GUT = np.log(M_GUT)
    t_XY  = np.log(M_XY)
    t_Phi = np.log(M_Phi)
    t_Sig = np.log(M_Sig)
    t_5C  = np.log(M_5C)
    t_T   = np.log(M_T)
    t_Z   = np.log(m_Z)

    # Initial conditions at M_GUT (example values)
    # For instance, 1/alpha_U = 25 => g5_init = sqrt(4pi/25)
    g5_init     = np.sqrt(4.0 * pi / 25.0)  # ~0.709
    y_u_init    = 0.50
    y_d_init    = 0.40
    lamH_init   = 0.10
    lamPhi_init = 0.05
    lamSig_init = 0.20
    y0_full = [g5_init, y_u_init, y_d_init, lamH_init, lamPhi_init, lamSig_init]

    # 1) Full SU(5): from M_GUT to M_XY
    sol1 = solve_ivp(beta_full_su5, (t_GUT, t_XY), y0_full, method='Radau', rtol=1e-7)
    g5_XY, yu_XY, yd_XY, lh_XY, lPhi_XY, lSig_XY = sol1.y[:, -1]

    # Threshold at M_XY: Remove X,Y gauge bosons (n=12)
    g5_XY_thr = gauge_threshold_correction(g5_XY, n=12, M_above=M_XY, M_below=M_Phi)
    y0_after_XY = [g5_XY_thr, yu_XY, yd_XY, lh_XY, lPhi_XY, lSig_XY]

    # 2) Run from M_XY to M_Phi using beta_su5_XY
    sol2 = solve_ivp(beta_su5_XY, (t_XY, t_Phi), y0_after_XY, method='Radau', rtol=1e-7)
    g5_Phi, yu_Phi, yd_Phi, lh_Phi, lPhi_Phi, lSig_Phi = sol2.y[:, -1]

    # Threshold at M_Phi: Remove Φ (n=5) => freeze lamPhi to 0
    g5_Phi_thr = gauge_threshold_correction(g5_Phi, n=5, M_above=M_Phi, M_below=M_Sig)
    y0_after_Phi = [g5_Phi_thr, yu_Phi, yd_Phi, lh_Phi, 0.0, lSig_Phi]

    # 3) Run from M_Phi to M_Sig with beta_su5_noPhi
    sol3 = solve_ivp(beta_su5_noPhi, (t_Phi, t_Sig), y0_after_Phi, method='Radau', rtol=1e-7)
    g5_Sig, yu_Sig, yd_Sig, lh_Sig, lPhi_Sig, lSig_Sig = sol3.y[:, -1]

    # Threshold at M_Sig: Remove Σ (n=7.5) => freeze lamSig to 0
    g5_Sig_thr = gauge_threshold_correction(g5_Sig, n=7.5, M_above=M_Sig, M_below=M_5C)
    y0_after_Sig = [g5_Sig_thr, yu_Sig, yd_Sig, lh_Sig, lPhi_Sig, 0.0]

    # 4) Run from M_Sig to M_5C with beta_su5_noPhi_noSig
    sol4 = solve_ivp(beta_su5_noPhi_noSig, (t_Sig, t_5C), y0_after_Sig, method='Radau', rtol=1e-7)
    g5_5C, yu_5C, yd_5C, lh_5C, lPhi_5C, lSig_5C = sol4.y[:, -1]

    # Threshold at M_5C: Remove color-triplet (n ~0.3, adjusted for the index splitting)
    g5_5C_thr = gauge_threshold_correction(g5_5C, n=0.3, M_above=M_5C, M_below=M_T)
    y0_after_5C = [g5_5C_thr, yu_5C, yd_5C, lh_5C, lPhi_5C, lSig_5C]

    # 5) Run from M_5C to M_T with beta_su5_noPhi_noSig_no5C
    sol5 = solve_ivp(beta_su5_noPhi_noSig_no5C, (t_5C, t_T), y0_after_5C, method='Radau', rtol=1e-7)
    g5_T, yu_T, yd_T, lh_T, lPhi_T, lSig_T = sol5.y[:, -1]

    # 6) Match onto the Standard Model (SM) at M_T.
    # Using matching conditions (here we use plausible example values):
    g1_init = 0.65  # U(1)_Y coupling (GUT-normalized)
    g2_init = 0.70  # SU(2)_L coupling
    g3_init = 0.75  # SU(3)_C coupling
    y_t_init = yu_T  # Use the unified up-type Yukawa as proxy
    lam_init = lh_T  # Higgs quartic coupling
    y0_sm = [g1_init, g2_init, g3_init, y_t_init, lam_init]

    # 7) Run the SM RG equations (using two-loop beta functions) from M_T to m_Z.
    solSM = solve_ivp(beta_sm_2loop, (t_T, t_Z), y0_sm, method='Radau', rtol=1e-7)
    g1_Z, g2_Z, g3_Z, y_t_Z, lam_Z = solSM.y[:, -1]

    print("===================================================================")
    print("Final SM couplings at m_Z = %.3f GeV:" % m_Z)
    print("   g1(m_Z) = %.4f, g2(m_Z) = %.4f, g3(m_Z) = %.4f" % (g1_Z, g2_Z, g3_Z))
    print("   y_t(m_Z) = %.4f, lambda_H(m_Z) = %.4f" % (y_t_Z, lam_Z))
    print("===================================================================")

    # 8) Compute electromagnetic coupling from SM couplings.
    alpha1_Z = g1_Z**2 / (4.0 * pi)
    alpha2_Z = g2_Z**2 / (4.0 * pi)
    inv_alpha_EM = (5.0/3.0)/alpha1_Z + 1.0/alpha2_Z  # Matching relation
    alpha_EM = 1.0/inv_alpha_EM
    e = np.sqrt(4.0 * pi * alpha_EM)
    print("Electromagnetic coupling at m_Z:")
    print("   alpha_EM(m_Z) = %.5f" % alpha_EM)
    print("   e(m_Z) = %.4f" % e)

    # Optional: Plot the running of the unified gauge coupling (g5) from M_GUT to m_Z.
    t_vals = np.linspace(t_GUT, np.log(m_Z), 1000)
    sol_plot = solve_ivp(beta_full_su5, (t_GUT, np.log(m_Z)), y0_full, t_eval=t_vals, method='Radau', rtol=1e-7)
    plt.figure(figsize=(8,6))
    plt.semilogx(np.exp(sol_plot.t), sol_plot.y[0], label='g5 (Full SU(5))')
    plt.xlabel(r'Energy scale $\mu$ [GeV]', fontsize=14)
    plt.ylabel(r'Unified gauge coupling $g_5$', fontsize=14)
    plt.title(r'Running of $g_5$ (Full SU(5)) from $M_{GUT}$ to $m_Z$', fontsize=16)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_ucft_su5_multithreshold()
