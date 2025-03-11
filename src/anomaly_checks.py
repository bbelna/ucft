#!/usr/bin/env python3
"""
anomaly_checks.py

Production-grade Python script that checks anomalies for a model 
based on SO(10)xU(1) and includes options to:
1. Add an extra custom multiplet that partially or fully cancels anomalies.
2. Model a simple Green-Schwarz offset.
3. Rescale U(1) charges by a user-specified factor.

Author: Your Name
Date: 2025
"""

import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Basic Dynkin indices for SO(10) we might need
SO10_DYNKIN_INDEX = {
    '16': 2,   # spinor
    '10': 2,   # vector
    '45': 8,   # adjoint
    '1' : 0
}
SO10_DIM = {
    '16': 16,
    '10': 10,
    '45': 45,
    '1' : 1
}

def so10_cubic_anomaly(rep_list):
    """
    (SO(10))^3 anomaly from chiral fermions in each representation.
    rep_list: list of (rep, count)
    """
    total = 0
    for (rep, count) in rep_list:
        if rep not in SO10_DYNKIN_INDEX:
            raise ValueError(f"Unknown rep: {rep}")
        total += SO10_DYNKIN_INDEX[rep]*count
    return total

def u1_cubic_anomaly(charges):
    """
    (U(1))^3 anomaly = sum(Q^3).
    """
    return sum(q**3 for q in charges)

def mixed_so10_sq_u1_anomaly(rep_list_u1):
    """
    (SO(10))^2 x U(1) anomaly = sum( I_2(R)*Q ) over chiral reps.
    rep_list_u1: list of (rep, charge, count)
    """
    total = 0
    for (rep, q, count) in rep_list_u1:
        if rep not in SO10_DYNKIN_INDEX:
            raise ValueError(f"Unknown rep: {rep}")
        total += SO10_DYNKIN_INDEX[rep]*q*count
    return total

def grav_u1_anomaly(charges):
    """
    [Gravity]^2 x U(1) anomaly = sum(Q).
    """
    return sum(charges)

def build_27_matter(n27=1, u1scale=1.0):
    """
    Creates lists for a single (or multiple) 27-plet(s) with scaled U(1) charges.
    Returns (so10_list, so10_u1_list, charges_list).
    Each 27 has 16_{+1}, 10_{-2}, 1_{+4} but each multiplied by factor u1scale.
    """
    so10_list = []
    so10_u1_list = []
    charges = []

    # 16_{+1*u1scale}
    so10_list.append(('16', n27)) 
    so10_u1_list.append(('16', +1*u1scale, n27))
    charges.extend([+1*u1scale]*(16*n27))

    # 10_{-2*u1scale}
    so10_list.append(('10', n27))
    so10_u1_list.append(('10', -2*u1scale, n27))
    charges.extend([-2*u1scale]*(10*n27))

    # 1_{+4*u1scale}
    so10_list.append(('1', n27))
    so10_u1_list.append(('1', +4*u1scale, n27))
    charges.extend([+4*u1scale]*(1*n27))

    return so10_list, so10_u1_list, charges

def main():
    parser = argparse.ArgumentParser(description="Anomaly checks with advanced fixes.")
    parser.add_argument("--n27", type=int, default=1, help="Number of E6 27-plets.")
    parser.add_argument("--u1scale", type=float, default=1.0,
                        help="Normalization factor for U(1) generator.")
    parser.add_argument("--x_so10", type=int, default=0,
                        help="Custom addition to (SO(10))^3 anomaly from extra multiplets.")
    parser.add_argument("--x_mix", type=float, default=0.0,
                        help="Custom addition to (SO(10))^2 x U(1) anomaly from extra multiplets.")
    parser.add_argument("--gs_so10", type=int, default=0,
                        help="Green–Schwarz offset for (SO(10))^3 anomaly.")
    parser.add_argument("--gs_mix", type=float, default=0.0,
                        help="Green–Schwarz offset for (SO(10))^2 x U(1) anomaly.")
    args = parser.parse_args()

    # Build the default 27-plet content
    so10_list, so10_u1_list, charges_list = build_27_matter(n27=args.n27, u1scale=args.u1scale)

    # Calculate anomalies for the 27-plet(s):
    so10_cubic = so10_cubic_anomaly(so10_list)
    u1_cubic   = u1_cubic_anomaly(charges_list)
    so10_mix   = mixed_so10_sq_u1_anomaly(so10_u1_list)
    grav_mix   = grav_u1_anomaly(charges_list)

    # Add custom offsets (representing extra multiplets):
    so10_cubic += args.x_so10
    so10_mix   += args.x_mix

    # Add Green–Schwarz offsets:
    so10_cubic += args.gs_so10
    so10_mix   += args.gs_mix

    logging.info("========== Anomaly Results with Fixes ==========")
    logging.info(f"(SO(10))^3 anomaly         = {so10_cubic}")
    logging.info(f"(U(1))^3 anomaly           = {u1_cubic}")
    logging.info(f"(SO(10))^2 x U(1) anomaly = {so10_mix}")
    logging.info(f"[Gravity]^2 x U(1) anomaly= {grav_mix}")
    logging.info("-----------------------------------------------")
    logging.info("Interpreting results:")
    logging.info(" - We want all anomalies to be zero for a consistent theory.")
    logging.info(" - Adjust `--n27`, `--u1scale`, `--x_so10`, `--x_mix`, `--gs_so10`, `--gs_mix` as needed.")
    logging.info("-----------------------------------------------")

if __name__ == "__main__":
    main()
