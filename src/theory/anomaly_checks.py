#!/usr/bin/env python3
"""
anomaly_checks.py

Production-grade Python script to compute anomaly contributions for matter 
fields under SO(10) x U(1) subsets of E6.

Usage:
  python anomaly_checks.py --n27 3 --spinor16 2 --vector10 1

Command-line Args:
  --n27 : Number of E6 27-plets (each containing 16_1 + 10_-2 + 1_4)
  --spinor16 : Number of additional 16_{Q} spinors (user must specify Q)
  --vector10 : Number of additional 10_{Q} vectors (user must specify Q)
  ... etc.

Author: Your Name
Date: 2025
"""

import sys
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s: %(message)s')

# Dynkin indices for key SO(10) representations
SO10_DYNKIN_INDEX = {
    '16': 2,   # Spinor
    '10': 2,   # Vector
    '45': 8,   # Adjoint
    '1' : 0    # Singlet
}

# Dimensions for key SO(10) representations
SO10_DIMENSIONS = {
    '16': 16,
    '10': 10,
    '45': 45,
    '1' : 1
}

def so10_cubic_anomaly(rep_list):
    """
    Computes the (SO(10))^3 anomaly contribution.

    Parameters:
    -----------
    rep_list : list of tuples (rep, count)
       rep = '16', '10', '45', or '1'
       count = how many copies of that rep

    Returns:
    --------
    total_anomaly : int
        Summation of I_2(R) * dimension_of_rep, or more precisely we sum 
        I_2(R) for chiral fermions. 
        For chiral anomalies, we typically use I_2(R) but dimension check can also be relevant. 
    """
    total_anomaly = 0
    for (rep, count) in rep_list:
        if rep not in SO10_DYNKIN_INDEX:
            raise ValueError(f"Representation {rep} not recognized for SO(10).")
        # For chiral fermions, the anomaly coefficient is proportional to I_2(R).
        # The dimension doesn't multiply in a naive way unless you count each 
        # component as a separate chiral fermion if it is chiral. 
        # Here we assume each "rep" is a distinct chiral multiplet.
        total_anomaly += count * SO10_DYNKIN_INDEX[rep]
    return total_anomaly

def u1_cubic_anomaly(u1_charges):
    """
    Computes (U(1))^3 anomaly by summing Q^3 over all chiral fermions.

    Parameters:
    -----------
    u1_charges : list of floats or ints
        List of charges for each chiral fermion state.

    Returns:
    --------
    total_u1_cubic : int or float
        Sum of Q^3 over all states.
    """
    total_u1_cubic = sum([q**3 for q in u1_charges])
    return total_u1_cubic

def mixed_so10_sq_u1_anomaly(rep_list_u1):
    """
    Computes the (SO(10))^2 x U(1) anomaly. We sum I_2(R) * Q for each rep.

    Parameters:
    -----------
    rep_list_u1 : list of tuples (rep, charge, count)
        rep = '16', '10', '1'
        charge = float, the U(1) charge
        count = how many copies of that rep

    Returns:
    --------
    total_mixed : int or float
    """
    total_mixed = 0
    for (rep, q, count) in rep_list_u1:
        if rep not in SO10_DYNKIN_INDEX:
            raise ValueError(f"Representation {rep} not recognized for SO(10).")
        total_mixed += count * SO10_DYNKIN_INDEX[rep] * q
    return total_mixed

def grav_u1_anomaly(u1_charges):
    """
    Computes the [Gravity]^2 x U(1) anomaly. We sum Q over all chiral fermions.

    Parameters:
    -----------
    u1_charges : list of floats or ints

    Returns:
    --------
    total_grav_u1 : int or float
        If zero, the gravitational x U(1) anomaly cancels.
    """
    return sum(u1_charges)

def build_matter_content(n27=1, extra_reps=None):
    """
    Build a matter content dictionary for an example model. 
    By default, we have n27 E6 27-plets, each containing:
      - 16_{+1}, 10_{-2}, 1_{+4}

    extra_reps is a list of tuples (rep, Q, count).

    Returns lists suitable for anomaly calculations.
    """
    if extra_reps is None:
        extra_reps = []

    # So(10)^3 anomaly list is just (rep, count).
    so10_list = []
    # (SO(10))^2 x U(1) list is (rep, charge, count).
    so10_u1_list = []
    # We also track a list of charges for Q^3 or gravitational anomaly.
    u1_charge_list = []

    # 1) The E6 27-plet
    #   => 16_{+1} x 16 copies, 10_{-2} x 10 copies, 1_{+4} x 1 copy
    # For anomaly code, we treat each 'rep' as a single chiral multiplet 
    # with a certain multiplicity.
    # Actually each 16_{+1} is dimension 16 but is 1 chiral rep in that sense. 
    # We'll assume 'count' refers to "how many 27-plets".
    
    # spinor 16_{+1}
    so10_list.append(('16',  n27))
    so10_u1_list.append(('16', +1, n27))
    # add +1 charge repeated 16 times (for each chiral component) 
    # if you want a more granular approach. 
    # But for a simpler approach, treat it as 1 chiral multiplet of '16' with charge +1. 
    # For cubic anomalies we do dimension * charge^3 if we want the total. 
    # We'll do the simpler approach of just "1 rep" of 16_{+1}.
    u1_charge_list.extend([+1]*n27*16)  # 16 chiral states per 27

    # vector 10_{-2}
    so10_list.append(('10', n27))
    so10_u1_list.append(('10', -2, n27))
    u1_charge_list.extend([-2]*n27*10)  

    # singlet 1_{+4}
    so10_list.append(('1',  n27))
    so10_u1_list.append(('1',  +4, n27))
    u1_charge_list.extend([+4]*n27*1)

    # 2) Add any extra reps
    for (rep, q, count, dim) in extra_reps:
        # rep is e.g. '16', q is float, count is int, dim is dimension 
        if rep not in SO10_DYNKIN_INDEX:
            raise ValueError(f"Unrecognized rep {rep} in extra_reps.")
        so10_list.append((rep, count))
        so10_u1_list.append((rep, q, count))
        # add dimension*count chiral states each with charge q
        u1_charge_list.extend([q]*(dim*count))

    return so10_list, so10_u1_list, u1_charge_list

def main():
    parser = argparse.ArgumentParser(
        description="Compute anomaly contributions for matter fields under SO(10)xU(1)."
    )
    parser.add_argument("--n27", type=int, default=1,
                        help="Number of E6 27-plets.")
    # Example of how to pass extra reps: we skip a full command-line approach for simplicity.
    args = parser.parse_args()

    # Possibly, user can add extra representations here if needed:
    # e.g. extra_reps = [('16', +1, 2, 16), ('10', -2, 3, 10)] 
    # => 2 spinor16_{+1} reps, 3 vector10_{-2} reps, etc.
    extra_reps = []

    # Build matter content
    so10_list, so10_u1_list, u1_charge_list = build_matter_content(
        n27=args.n27,
        extra_reps=extra_reps
    )

    # (SO(10))^3 anomaly:
    anom_so10_cubic = so10_cubic_anomaly(so10_list)

    # (U(1))^3 anomaly:
    anom_u1_cubic = u1_cubic_anomaly(u1_charge_list)

    # (SO(10))^2 x U(1) anomaly:
    anom_mix = mixed_so10_sq_u1_anomaly(so10_u1_list)

    # [Gravity]^2 x U(1) anomaly:
    anom_grav_u1 = grav_u1_anomaly(u1_charge_list)

    logging.info("========== Anomaly Results ==========")
    logging.info(f"(SO(10))^3 anomaly    = {anom_so10_cubic}")
    logging.info(f"(U(1))^3 anomaly      = {anom_u1_cubic}")
    logging.info(f"(SO(10))^2 x U(1)    = {anom_mix}")
    logging.info(f"[Gravity]^2 x U(1)   = {anom_grav_u1}")
    logging.info("=====================================")

    # Interpret:
    # For anomaly freedom, we want all these sums = 0 (or to arrange so they vanish net).
    # This code just calculates them; the user can add further logic to 
    # check if each sum is zero or not.

if __name__ == "__main__":
    main()
